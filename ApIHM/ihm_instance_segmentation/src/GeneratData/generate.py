import os

import cv2
import numpy as np
import PIL
import torch
import torchvision
from loadingpy import pybar
from transformers import SamModel, SamProcessor

from .cocoloader import get_coco_dataset
from .cocosplit import COCOSplit
from .student_extract import SpitPrivateData


class SetGenerator:
    def __init__(
        self,
        coco_path: str,
        annotations_path: str,
        target_dir: str = os.path.join(os.getcwd(), "SegSets"),
    ) -> None:
        print("--- prepare datasets for the practical sessions ---".center(60))
        self.splitter = COCOSplit(
            coco_path=coco_path,
            annotations_path=annotations_path,
            target_dir=target_dir,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_visualitzation(
        self,
        images: torch.Tensor,
        masks: np.ndarray,
        bbox: np.ndarray,
        cpt: int,
        folder: str,
    ) -> None:
        color = np.array([0, 0, 255], dtype="uint8")
        masked_img = np.where(masks[..., None], color, images)
        out = cv2.addWeighted(images, 0.5, masked_img, 0.5, 0)
        color = np.array([255, 0, 0], dtype="uint8")
        cv2.rectangle(
            out,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 0, 255),
            2,
        )
        cv2.imwrite(os.path.join(folder, f"vis_{str(cpt).zfill(6)}.png"), out)

    def register_a_set(
        self, processor, dataset, model, device, set_, path_to_images
    ) -> None:
        with torch.no_grad():
            for cpt, (
                raw_image,
                input_boxes,
            ) in enumerate(pybar(dataset, base_str=f"sam on {set_}")):
                raw_image = raw_image[0]
                input_boxes = input_boxes[0]
                inputs = processor(raw_image, return_tensors="pt", do_rescale=False).to(
                    device
                )
                image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
                inputs = processor(
                    raw_image,
                    input_boxes=[[[input_boxes]]],
                    return_tensors="pt",
                    do_rescale=False,
                ).to(device)
                inputs.pop("pixel_values", None)
                inputs.update({"image_embeddings": image_embeddings})

                outputs = model(**inputs)
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )

                original_mask = masks[0][0][-1]
                original_image = raw_image
                cropped_img = torchvision.transforms.functional.crop(
                    img=original_image,
                    top=int(input_boxes[1]),
                    left=int(input_boxes[0]),
                    height=int(input_boxes[2] - input_boxes[0]),
                    width=int(input_boxes[3] - input_boxes[1]),
                )
                cropped_mask = torchvision.transforms.functional.crop(
                    img=original_mask,
                    top=int(input_boxes[1]),
                    left=int(input_boxes[0]),
                    height=int(input_boxes[2] - input_boxes[0]),
                    width=int(input_boxes[3] - input_boxes[1]),
                )
                torchvision.transforms.functional.to_pil_image(cropped_img[0]).save(
                    os.path.join(path_to_images, f"image_{str(cpt).zfill(6)}.png")
                )
                np.save(
                    os.path.join(path_to_images, f"label_{str(cpt).zfill(6)}.npy"),
                    cropped_mask.numpy(),
                )
                self.save_visualitzation(
                    images=np.array(
                        torchvision.transforms.functional.to_pil_image(cropped_img)
                    ),
                    masks=cropped_mask.numpy(),
                    bbox=input_boxes,
                    cpt=cpt,
                    folder=path_to_images,
                )

    def setup(self) -> None:
        self.splitter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        if len(os.listdir(os.path.join(os.getcwd(), "SegSets", "train"))) != 4000:
            dataset = get_coco_dataset(
                path_to_images=self.splitter.train_set_folder,
                path_to_bboxes=os.path.join(os.getcwd(), "train_bboxes.npy"),
            )
            self.register_a_set(
                processor=processor,
                model=model,
                dataset=dataset,
                device=device,
                set_="train",
                path_to_images=self.splitter.train_set_folder,
            )
        if len(os.listdir(os.path.join(os.getcwd(), "SegSets", "val"))) != 800:
            dataset = get_coco_dataset(
                path_to_images=self.splitter.val_set_folder,
                path_to_bboxes=os.path.join(os.getcwd(), "val_bboxes.npy"),
            )
            self.register_a_set(
                processor=processor,
                model=model,
                dataset=dataset,
                device=device,
                set_="val",
                path_to_images=self.splitter.val_set_folder,
            )
        if len(os.listdir(os.path.join(os.getcwd(), "SegSets", "test"))) != 1600:
            dataset = get_coco_dataset(
                path_to_images=self.splitter.test_set_folder,
                path_to_bboxes=os.path.join(os.getcwd(), "test_bboxes.npy"),
            )
            self.register_a_set(
                processor=processor,
                model=model,
                dataset=dataset,
                device=device,
                set_="test",
                path_to_images=self.splitter.test_set_folder,
            )
        SpitPrivateData()(
            train_folder=self.splitter.train_set_folder,
            val_folder=self.splitter.val_set_folder,
            test_folder=self.splitter.test_set_folder,
        )
