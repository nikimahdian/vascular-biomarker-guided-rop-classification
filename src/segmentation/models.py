"""Vessel-segmentation model factory + losses (segmentation_models_pytorch).

Mirrors the architectures used in the prior work so existing checkpoints load:
DeepLabV3Plus / DeepLabV3 / UnetPlusPlus / Unet / MAnet, resnet/efficientnet encoders.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

_ARCHS = {
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "DeepLabV3": smp.DeepLabV3,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "Unet": smp.Unet,
    "MAnet": smp.MAnet,
}


def build_model(
    arch: str = "DeepLabV3Plus",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Options: {list(_ARCHS)}")
    return _ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class DiceBCELoss(nn.Module):
    """Dice + BCE. Model outputs logits (no sigmoid); sigmoid applied here."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = 1 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(probs, targets, reduction="mean")
        return bce + dice


@torch.no_grad()
def dice_coef(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    probs = (torch.sigmoid(logits) > 0.5).float().view(-1)
    targets = targets.view(-1)
    inter = (probs * targets).sum()
    return float((2 * inter + smooth) / (probs.sum() + targets.sum() + smooth))
