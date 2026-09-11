"""Dataset for training the vessel segmenter on the BV_segmentation dataset.

Expects an images folder and a masks folder with matching (sorted) filenames,
as in the prior work's BV_segmentation_dataset.zip.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


def _list_images(folder: str, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)
    )


class RetinaSegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, img_size=(256, 256)):
        self.image_paths = _list_images(image_dir)
        self.mask_paths = _list_images(mask_dir)
        assert len(self.image_paths) == len(self.mask_paths), (
            f"image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}"
        )
        self.img_size = tuple(img_size)
        self.image_tf = transforms.Compose(
            [transforms.Resize(self.img_size), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        img = self.image_tf(img)
        mask = mask.resize(self.img_size, resample=Image.NEAREST)
        mask = ToTensor()(mask)
        mask = (mask > 0).float()
        return img, mask
