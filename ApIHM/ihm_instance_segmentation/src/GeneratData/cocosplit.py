import json
import os
import shutil
from typing import Any, Dict, List

import numpy as np
from loadingpy import pybar
from pycocotools import coco


class COCOSplit:
    def __init__(
        self,
        coco_path: str,
        annotations_path: str,
        target_dir: str = os.path.join(os.getcwd(), "SegSets"),
    ) -> None:
        self.coco_path = coco_path
        self.target_dir = target_dir
        self.annotations_path = annotations_path
        self.create_folders()

    def create_folders(self) -> None:
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
        self.train_set_folder = os.path.join(self.target_dir, "train")
        self.val_set_folder = os.path.join(self.target_dir, "val")
        self.test_set_folder = os.path.join(self.target_dir, "test")
        if not os.path.exists(self.train_set_folder):
            os.mkdir(self.train_set_folder)
        if not os.path.exists(self.val_set_folder):
            os.mkdir(self.val_set_folder)
        if not os.path.exists(self.test_set_folder):
            os.mkdir(self.test_set_folder)

    def images_id_to_file_name(
        self, image_data: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        output = {}
        for data in image_data:
            output[data["id"]] = data["file_name"]
        return output

    def extract_people_examples(self) -> List[str]:
        with open(self.annotations_path, "r") as f:
            data = json.load(f)
        people_id = 1
        img_ids_to_files = self.images_id_to_file_name(data["images"])
        extracted_file_paths = []
        bboxes = []
        binary_masks = []
        for data in pybar(data["annotations"], base_str="extract labels"):
            if data["category_id"] == people_id:
                img_pth = img_ids_to_files[data["image_id"]]
                if img_pth not in extracted_file_paths:
                    extracted_file_paths.append(img_pth)
                    bboxes.append(data["bbox"])
                    binary_masks.append(self.mask_extract(data))
        print(f"found {len(extracted_file_paths)} examples in total (expected 64115)")
        return extracted_file_paths, bboxes, binary_masks

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if (
            len(os.listdir(self.train_set_folder)) == 0
            or len(os.listdir(self.val_set_folder)) == 0
            or len(os.listdir(self.test_set_folder)) == 0
        ):
            self.mask_extract = coco.COCO(
                annotation_file=self.annotations_path
            ).annToMask
            all_examples, all_boxes, binary_masks = self.extract_people_examples()
        if len(os.listdir(self.train_set_folder)) == 0:
            for cpt, data_path in enumerate(
                pybar(all_examples[:1000], base_str="split - train")
            ):
                shutil.copyfile(
                    os.path.join(self.coco_path, data_path),
                    os.path.join(self.train_set_folder, str(cpt).zfill(6) + ".jpg"),
                )
                np.save(
                    os.path.join(
                        self.train_set_folder, "mask_" + str(cpt).zfill(6) + ".jpg"
                    ),
                    binary_masks[cpt],
                )
            np.save(
                os.path.join(os.getcwd(), "train_bboxes.npy"),
                np.array(all_boxes[:1000]),
            )
        if len(os.listdir(self.val_set_folder)) == 0:
            for cpt, data_path in enumerate(
                pybar(all_examples[1000:1200], base_str="split - val")
            ):
                shutil.copyfile(
                    os.path.join(self.coco_path, data_path),
                    os.path.join(self.val_set_folder, str(cpt).zfill(6) + ".jpg"),
                )
                np.save(
                    os.path.join(
                        self.val_set_folder, "mask_" + str(cpt).zfill(6) + ".jpg"
                    ),
                    binary_masks[cpt],
                )
            np.save(
                os.path.join(os.getcwd(), "val_bboxes.npy"),
                np.array(all_boxes[1000:1200]),
            )
        if len(os.listdir(self.test_set_folder)) == 0:
            for cpt, data_path in enumerate(
                pybar(all_examples[1200:1600], base_str="split - test")
            ):
                shutil.copyfile(
                    os.path.join(self.coco_path, data_path),
                    os.path.join(self.test_set_folder, str(cpt).zfill(6) + ".jpg"),
                )
                np.save(
                    os.path.join(
                        self.test_set_folder, "mask_" + str(cpt).zfill(6) + ".jpg"
                    ),
                    binary_masks[cpt],
                )
            np.save(
                os.path.join(os.getcwd(), "test_bboxes.npy"),
                np.array(all_boxes[1200:1600]),
            )
