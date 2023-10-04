import os

import numpy as np
import torch
import torchvision
from PIL import Image


class SegmentationDataset:
    def __init__(self, path_to_data: str) -> None:
        self.image_paths = [
            os.path.join(path_to_data, elem)
            for elem in sorted(os.listdir(path_to_data))
            if ".png" in elem
        ]
        self.label_paths = [
            os.path.join(path_to_data, elem)
            for elem in sorted(os.listdir(path_to_data))
            if ".npy" in elem
        ]
        self.resize_img = torchvision.transforms.Resize(size=(224, 224), antialias=True)
        self.resize_lbl = torchvision.transforms.Resize(size=(28, 28), antialias=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_np = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        img_np = np.transpose(img_np, axes=[2, 0, 1])
        original_size = torch.tensor(img_np.shape)
        img = self.resize_img(torch.tensor(img_np, dtype=torch.float))
        if len(self.label_paths) > 0:
            lbl_np = np.load(self.label_paths[idx])
            lbl_np = np.expand_dims(lbl_np, axis=0)
            lbl = self.resize_lbl(torch.tensor(lbl_np, dtype=torch.float))
            return img, lbl
        return img, original_size


def create_dataloader(
    path_to_data: str, batch_size: int = 16
) -> torch.utils.data.DataLoader:
    torch.backends.cudnn.benchmark = True
    dataset = SegmentationDataset(path_to_data=path_to_data)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=len(dataset.label_paths) != 0,
        pin_memory=True,
        num_workers=4,
    )
