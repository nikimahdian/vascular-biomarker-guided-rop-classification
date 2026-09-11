"""Static audit of historical E4. Does not rewrite E4 results. Not a reason to --force-e5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.classify.next_architecture.experiments import ORDINAL_AUX, uses_ordinal_aux


E4_DEFINITION = {
    "experiment": "E4",
    "historical_status": "completed_seed42_only",
    "do_not_overwrite": True,
    "do_not_force_e5": True,
    "objective": {
        "plus_bce": True,
        "coral_ordinal": True,
        "pairwise_ranking": True,
        "matches_e2_binary_only": False,
        "matches_e3_auxiliaries": True,
        "evidence": "ORDINAL_AUX includes E4; trainer applies coral+ranking whenever uses_ordinal_aux(E4)",
        "ordinal_aux_set": sorted(ORDINAL_AUX),
        "uses_ordinal_aux_e4": uses_ordinal_aux("E4"),
        "uses_ordinal_aux_e2": uses_ordinal_aux("E2"),
        "uses_ordinal_aux_e4b": uses_ordinal_aux("E4B"),
    },
    "input": {
        "kind": "early_fusion_4ch",
        "channels": "ImageNet-normalized RGB (3) concatenated with vessel probability (1)",
        "vessel_channel_range": "[0, 1] after clip; not ImageNet-normalized",
        "rgb_normalized": True,
        "scale_mismatch": True,
        "scale_mismatch_note": (
            "RGB after IMAGENET_MEAN/STD is roughly [-2, 2]; vessel stays [0, 1]. "
            "Fourth stem channel therefore sees a different scale than pretrained RGB."
        ),
        "not_residual_gated": True,
        "not_four_derived_channels": True,
    },
    "stem_init": {
        "timm_in_chans": 4,
        "extra_channel_init": "mean of pretrained RGB conv_stem weights",
        "function": "_init_extra_input_channel",
    },
    "augmentation": {
        "synced_geometry": True,
        "color_jitter_rgb_only": True,
        "letterbox_default": False,
        "resize": "bilinear distorting Resize to square (same family as Branch B / E2)",
        "vessel_interpolation": "cv2 INTER_LINEAR / PIL bilinear — can blur thin vessels",
        "random_crop": False,
    },
    "alignment": {
        "probability_loaded_at_native_resolution_then_resized_with_rgb": True,
        "padding_letterbox": False,
    },
    "optimizer_schedule": {
        "shared_trainer_with_e2_e3": True,
        "same_lr_early_stop_defaults": True,
        "seed_42_only": True,
        "comparator_on_val": "E3 not E2",
        "n_seeds": 1,
        "paired_ci_includes_zero": True,
    },
    "followup": {
        "e4b_required": True,
        "e4b_reason": "E4 is not the binary-only 4-channel control; E4B isolates that factor",
        "historical_e5_gate_unchanged": True,
        "e5x_uses_new_gate": True,
    },
}


def write_e4_definition_audit(dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(E4_DEFINITION, indent=2, default=str) + "\n", encoding="utf-8")
    return E4_DEFINITION
