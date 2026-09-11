"""Experiment IDs and architecture families. E6/E7 stay blocked. Historical E5 stays blocked."""

from __future__ import annotations

# Historical ladder IDs. Do not change E4/E5 semantics.
HISTORICAL = frozenset({"E0", "E1", "E2", "E3", "E4", "E5", "E8"})
FOLLOWUP = frozenset({"E4B", "V1", "E5X"})
ALLOWED_EXPERIMENTS = HISTORICAL | FOLLOWUP
MULTICLASS_CE = frozenset({"E0", "E1"})
BINARY_PLUS = frozenset({"E2", "E3", "E4", "E5", "E8", "E4B", "V1", "E5X"})
# Historical: E4 inherited E3 auxiliaries. E4B/E5X do not.
ORDINAL_AUX = frozenset({"E3", "E4", "E5"})
EARLY_FUSION_4CH = frozenset({"E4", "E4B"})
VESSEL_ONLY = frozenset({"V1"})
USES_VESSEL = frozenset({"E4", "E4B", "E5", "E5X", "V1"})
USES_GATE = frozenset({"E5", "E5X"})
SOURCE_BALANCED = frozenset({"E8"})
BLOCKED_EXPERIMENTS = {
    "E6": "anatomy-aware biomarker provenance absent",
    "E7": "MIL blocked: no eye_id/visit_id",
}


def normalize_experiment(name: str) -> str:
    exp = str(name).upper()
    if exp in BLOCKED_EXPERIMENTS:
        raise SystemExit(f"{exp} blocked: {BLOCKED_EXPERIMENTS[exp]}")
    if exp not in ALLOWED_EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {exp}")
    return exp


def uses_multiclass_ce(experiment: str) -> bool:
    return experiment.upper() in MULTICLASS_CE


def uses_ordinal_aux(experiment: str) -> bool:
    return experiment.upper() in ORDINAL_AUX


def uses_vessel(experiment: str) -> bool:
    return experiment.upper() in USES_VESSEL


def is_source_balanced(experiment: str) -> bool:
    return experiment.upper() in SOURCE_BALANCED


def is_early_fusion_4ch(experiment: str) -> bool:
    return experiment.upper() in EARLY_FUSION_4CH


def is_vessel_only(experiment: str) -> bool:
    return experiment.upper() in VESSEL_ONLY


def is_gated_fusion(experiment: str) -> bool:
    return experiment.upper() in USES_GATE


def gate_bias_init(experiment: str, cfg_default: float = -2.5) -> float:
    if experiment.upper() == "E5X":
        return -3.0
    return float(cfg_default)
