"""RGB, 4-channel, and residual-gated fusion networks. No tree heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

from src.classify.next_architecture.experiments import (
    is_early_fusion_4ch,
    is_gated_fusion,
    is_vessel_only,
    uses_multiclass_ce,
)


FORBIDDEN_FORWARD_KEYS = {
    "source",
    "device_id",
    "filename",
    "path",
    "folder",
    "patient_id",
    "exam_id",
    "group_id",
}


class IdentityLeakageError(ValueError):
    pass


def _batchnorm_to_groupnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = 32 if child.num_features >= 32 else max(1, child.num_features // 2 or 1)
            while child.num_features % groups != 0 and groups > 1:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _batchnorm_to_groupnorm(child)


def _assert_no_identity_kwargs(kwargs: dict) -> None:
    bad = FORBIDDEN_FORWARD_KEYS & set(kwargs)
    if bad:
        raise IdentityLeakageError(f"Identity metadata is not a legal forward input: {sorted(bad)}")


class RGBEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        in_chans: int = 3,
        dropout: float = 0.5,
        proj_dim: int | None = 512,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
        )
        feat = int(self.backbone.num_features)
        self.dropout = nn.Dropout(dropout)
        if proj_dim is None or proj_dim == feat:
            self.proj = nn.Identity()
            self.out_dim = feat
        else:
            self.proj = nn.Linear(feat, proj_dim)
            self.out_dim = proj_dim
        if in_chans == 4:
            _init_extra_input_channel(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(self.backbone(x)))


def _init_extra_input_channel(model: nn.Module) -> None:
    """If timm copied RGB into 4-ch, average RGB into the extra channel."""
    conv = getattr(model, "conv_stem", None)
    if conv is None:
        return
    if not isinstance(conv, nn.Conv2d) or conv.in_channels != 4:
        return
    with torch.no_grad():
        conv.weight[:, 3] = conv.weight[:, :3].mean(dim=1)


class NextArchitectureNet(nn.Module):
    """Unified network. Experiment ID selects heads and inputs."""

    def __init__(
        self,
        experiment: str,
        backbone: str = "efficientnet_b5",
        pretrained: bool = True,
        dropout: float = 0.5,
        rgb_dim: int = 512,
        vessel_dim: int = 128,
        vessel_encoder: str = "resnet18",
        gate_bias_init: float = -2.5,
        modality_dropout: float = 0.2,
        in_channels: int = 3,
    ):
        super().__init__()
        self.experiment = experiment.upper()
        self.modality_dropout = float(modality_dropout)
        self.uses_vessel = is_gated_fusion(self.experiment) or is_vessel_only(self.experiment)
        self.uses_gate = is_gated_fusion(self.experiment)
        native_dim = self.experiment in {"E0", "E1"}
        if is_vessel_only(self.experiment):
            in_ch = 4
            backbone = "resnet18"
            pretrained = False
            proj = rgb_dim
        elif is_early_fusion_4ch(self.experiment):
            in_ch = 4
            proj = None if native_dim else rgb_dim
        else:
            in_ch = 3
            proj = None if native_dim else rgb_dim
        self.rgb = RGBEncoder(
            backbone,
            pretrained=pretrained,
            in_chans=in_ch,
            dropout=dropout,
            proj_dim=proj,
        )
        if is_vessel_only(self.experiment):
            _batchnorm_to_groupnorm(self.rgb.backbone)
        self.multiclass_head = nn.Linear(self.rgb.out_dim, 3)
        self.plus_head = nn.Linear(self.rgb.out_dim, 1)
        self.ordinal_head = nn.Linear(self.rgb.out_dim, 2)
        self.vessel_encoder = None
        self.vessel_proj = None
        self.gate = None
        self.vessel_to_rgb = None
        if self.uses_gate:
            self.vessel_encoder = timm.create_model(
                vessel_encoder,
                pretrained=pretrained and not is_vessel_only(self.experiment),
                num_classes=0,
                global_pool="avg",
                in_chans=4,
            )
            v_feat = int(self.vessel_encoder.num_features)
            self.vessel_proj = nn.Linear(v_feat, vessel_dim)
            self.vessel_to_rgb = nn.Linear(vessel_dim, self.rgb.out_dim)
            self.gate = nn.Sequential(
                nn.Linear(self.rgb.out_dim + vessel_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))

    def forward(
        self,
        images: torch.Tensor,
        vessel: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _assert_no_identity_kwargs(kwargs)
        h_rgb = self.rgb(images)
        gate = images.new_zeros((images.size(0), 1))
        h_v = None
        fused = h_rgb
        if self.uses_gate:
            if vessel is None:
                raise ValueError("E5 requires vessel tensor of shape (B, 4, H, W).")
            h_v = self.vessel_proj(self.vessel_encoder(vessel))
            gate_in = torch.cat([h_rgb, h_v], dim=1)
            gate = torch.sigmoid(self.gate(gate_in))
            if self.training and self.modality_dropout > 0:
                drop = (torch.rand(gate.size(0), 1, device=gate.device) > self.modality_dropout).float()
                gate = gate * drop
            fused = h_rgb + gate * self.vessel_to_rgb(h_v)
        features = fused
        plus_logit = self.plus_head(features).squeeze(-1)
        multiclass_logits = self.multiclass_head(features)
        ordinal_logits = self.ordinal_head(features)
        if uses_multiclass_ce(self.experiment):
            plus_logit = multiclass_logits[:, 2]
        out = {
            "features": features,
            "h_rgb": h_rgb,
            "plus_logit": plus_logit,
            "multiclass_logits": multiclass_logits,
            "ordinal_logits": ordinal_logits,
            "gate": gate.squeeze(-1),
        }
        if h_v is not None:
            out["h_v"] = h_v
            out["vessel_residual"] = gate * self.vessel_to_rgb(h_v)
        return out


def initial_gate_residual_small(model: NextArchitectureNet, atol_ratio: float = 0.25) -> bool:
    """With random RGB/vessel, mean |residual| should be << mean |h_rgb| at init."""
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    vessel = torch.rand(2, 4, 32, 32)
    with torch.no_grad():
        out = model(images, vessel=vessel)
    residual = out["vessel_residual"].abs().mean()
    rgb = out["h_rgb"].abs().mean()
    return bool((residual / rgb.clamp(min=1e-8)) < atol_ratio)
