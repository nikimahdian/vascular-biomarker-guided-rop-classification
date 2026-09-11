"""Optional E6 auxiliary targets. Disabled until anatomy-aware provenance exists."""
from __future__ import annotations

from pathlib import Path


FORBIDDEN_PROXY_FEATURES = {
    "vessel_pixels",
    "area",
    "overall_length",
    "n_startpoints",
    "n_endpoints",
    "n_intersections",
}


def load_auxiliary_biomarker_targets(path: Path | None) -> None:
    if path is None:
        raise FileNotFoundError(
            "Anatomy-aware biomarker auxiliary targets are not configured (E6 disabled)."
        )
    raise FileNotFoundError(
        f"No valid anatomy-aware biomarker provenance at {path}. "
        "Refusing raw 23-feature PVBM table as Plus-logit input."
    )
