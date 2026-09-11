"""Fundus dataset with optional aligned vessel maps. Identity is never a tensor input."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.classify.image_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.classify.next_architecture.experiments import (
    is_early_fusion_4ch,
    is_vessel_only,
    uses_vessel,
)
from src.classify.next_architecture.losses import PLUS_INDEX, plus_target_from_labels
from src.classify.next_architecture.vessel_maps import derived_channels, load_prob_map, resize_prob_map

IDENTITY_COLUMNS = (
    "source",
    "group_id",
    "patient_id",
    "exam_id",
    "identity_level",
    "image_path",
)


def letterbox_rgb(img: Image.Image, size: int, fill=(0, 0, 0)) -> Image.Image:
    """Deterministic pad-to-square then resize. Preserves FOV; no random crop."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def letterbox_map(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2_resize(arr, nh, nw)
    canvas = np.zeros((size, size), dtype=np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def cv2_resize(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    return resize_prob_map(arr, height, width)


class SyncedTransform:
    """Geometric transforms identical for RGB and vessel; color jitter RGB-only."""

    def __init__(self, img_size: int, train: bool, cfg_aug: dict):
        self.img_size = int(img_size)
        self.train = train
        self.letterbox = bool(cfg_aug.get("letterbox", False))
        self.flip = bool(cfg_aug.get("random_horizontal_flip", True)) and train
        self.rot = float(cfg_aug.get("random_rotation_deg", 10)) if train else 0.0
        jitter = cfg_aug.get("color_jitter", [0.1, 0.1, 0.1])
        self.jitter = transforms.ColorJitter(*jitter) if train else None
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def _resize_rgb(self, img: Image.Image) -> Image.Image:
        if self.letterbox:
            return letterbox_rgb(img, self.img_size)
        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def _resize_map(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"vessel map must be HxW, got {arr.shape}")
        if self.letterbox:
            return letterbox_map(arr, self.img_size)
        return resize_prob_map(arr, self.img_size, self.img_size)

    def __call__(
        self, img: Image.Image, vessel: np.ndarray | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        img = self._resize_rgb(img)
        vmap = None if vessel is None else self._resize_map(np.asarray(vessel, dtype=np.float32))
        if self.train:
            if self.flip and torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                if vmap is not None:
                    vmap = np.ascontiguousarray(np.fliplr(vmap))
            if self.rot:
                angle = float((torch.rand(1).item() * 2 - 1) * self.rot)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                if vmap is not None:
                    v_img = Image.fromarray((np.clip(vmap, 0, 1) * 255).astype(np.uint8))
                    v_img = TF.rotate(v_img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                    vmap = np.array(v_img, dtype=np.float32) / 255.0
            if self.jitter is not None:
                img = self.jitter(img)
        tensor = self.normalize(TF.to_tensor(img))
        if vmap is None:
            return tensor, None
        return tensor, torch.from_numpy(np.clip(vmap, 0.0, 1.0)).float()


class NextArchitectureDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        img_size: int,
        train: bool,
        aug_cfg: dict,
        *,
        experiment: str = "E0",
        vessel_prob_dir: Path | None = None,
        topology_threshold: float = 0.20,
        plus_idx: int = PLUS_INDEX,
        smoke_limit: int | None = None,
        synthetic_vessel: bool = False,
    ):
        self.df = frame.reset_index(drop=True).copy()
        if smoke_limit is not None:
            self.df = self.df.iloc[: int(smoke_limit)].reset_index(drop=True)
        self.experiment = experiment.upper()
        self.tf = SyncedTransform(img_size, train, aug_cfg)
        self.vessel_prob_dir = Path(vessel_prob_dir) if vessel_prob_dir else None
        self.topology_threshold = topology_threshold
        self.plus_idx = plus_idx
        self.needs_vessel = uses_vessel(self.experiment)
        self.synthetic_vessel = synthetic_vessel

    def __len__(self) -> int:
        return len(self.df)

    def _load_probability(self, image_path: str, orig_size: tuple[int, int]) -> np.ndarray:
        if self.synthetic_vessel:
            h, w = orig_size[1], orig_size[0]
            yy, xx = np.mgrid[0:h, 0:w]
            return (0.5 + 0.5 * np.sin(xx / 12.0) * np.cos(yy / 12.0)).astype(np.float32)
        if "prob_path" in self.df.columns:
            matches = self.df.loc[self.df["image_path"] == image_path, "prob_path"]
            if len(matches):
                candidate = Path(str(matches.iloc[0]))
                if candidate.exists():
                    return load_prob_map(candidate)
        if self.vessel_prob_dir is None:
            raise FileNotFoundError("vessel_prob_dir is required for E4/E5")
        stem = Path(image_path).stem
        found = sorted(self.vessel_prob_dir.glob(f"{stem}_*.npy"))
        if found:
            return load_prob_map(found[0])
        raise FileNotFoundError(f"No probability map for {image_path}")

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.needs_vessel:
            prob = self._load_probability(str(row["image_path"]), img.size)
            rgb_t, prob_t = self.tf(img, prob)
            if is_early_fusion_4ch(self.experiment):
                image = torch.cat([rgb_t, prob_t.unsqueeze(0)], dim=0)
                return self._pack(image, label, row, None)
            derived = derived_channels(
                prob_t.numpy(), topology_threshold=self.topology_threshold
            )
            derived_t = torch.from_numpy(derived)
            if is_vessel_only(self.experiment):
                return self._pack(derived_t, label, row, None)
            return self._pack(rgb_t, label, row, derived_t)
        rgb_t, _ = self.tf(img, None)
        return self._pack(rgb_t, label, row, None)

    def _pack(self, image: torch.Tensor, label: int, row, vessel: torch.Tensor | None) -> dict:
        label_t = torch.tensor(label, dtype=torch.long)
        plus = plus_target_from_labels(label_t, self.plus_idx)
        sample = {
            "image": image,
            "label": label_t,
            "plus_target": plus,
            "image_path": str(row["image_path"]),
        }
        if vessel is not None:
            sample["vessel"] = vessel
        for col in IDENTITY_COLUMNS:
            if col in row.index:
                sample[f"meta_{col}"] = row[col]
        return sample

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].astype(int).values


def collate_next(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    plus = torch.stack([b["plus_target"] for b in batch], dim=0)
    out = {
        "image": images,
        "label": labels,
        "plus_target": plus,
        "image_path": [b["image_path"] for b in batch],
        "sources": [str(b.get("meta_source", "")) for b in batch],
        "group_id": [str(b.get("meta_group_id", "")) for b in batch],
    }
    if "vessel" in batch[0]:
        out["vessel"] = torch.stack([b["vessel"] for b in batch], dim=0)
    return out
