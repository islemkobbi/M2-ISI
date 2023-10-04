import os
from typing import Any, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image


class CustomCocoDetection:
    def __init__(
        self,
        path_to_images: str,
        path_to_bboxes: str,
    ) -> None:
        self.root = path_to_images
        self.labels = np.load(path_to_bboxes)
        self.paths = [
            elem for elem in sorted(os.listdir(path_to_images)) if "mask" not in elem
        ]
        self.masks_paths = [
            elem for elem in sorted(os.listdir(path_to_images)) if "mask" in elem
        ]
        self.index = 0
        self.pil_to_tensor = torchvision.transforms.ToTensor()

    def _load_image(self, index: int) -> np.ndarray:
        image = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        return image

    def __next__(self) -> Tuple[Any, Any]:
        original_image = self._load_image(self.index)
        label = self.labels[self.index]
        label = [label[0], label[1], label[0] + label[2], label[1] + label[3]]
        self.index += 1
        return (original_image, label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        original_image = self._load_image(idx)
        label = self.labels[idx]
        label = [label[0], label[1], label[0] + label[2], label[1] + label[3]]
        return self.pil_to_tensor(original_image), torch.tensor(label)

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self) -> int:
        return len(self.paths)


def get_coco_dataset(
    path_to_images: str, path_to_bboxes: str
) -> torch.utils.data.DataLoader:
    """
    creates a torch data loader for 1 element at a time coco loading
    """
    # return CustomCocoDetection(
    #     path_to_images=path_to_images, path_to_bboxes=path_to_bboxes
    # )
    dataset = CustomCocoDetection(
        path_to_images=path_to_images, path_to_bboxes=path_to_bboxes
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=len(dataset) != 0,
        pin_memory=True,
        num_workers=4,
    )
