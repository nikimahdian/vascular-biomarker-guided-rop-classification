"""MIL readiness validator. MIL is blocked until eye/visit metadata exists."""
from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_MIL_FIELDS = (
    "patient_id",
    "eye_id",
    "visit_id",
    "laterality",
    "view",
    "timestamp",
    "fov",
)
FORBIDDEN_SUBSTITUTE = "group_id"


class MILBlockedError(RuntimeError):
    """Raised when MIL is requested without complete clinical grouping fields."""


def mil_readiness(frame: pd.DataFrame) -> dict[str, Any]:
    """Inspect metadata. Never infers eye/visit from group_id."""
    missing = [name for name in REQUIRED_MIL_FIELDS if name not in frame.columns]
    present_incomplete = []
    for name in REQUIRED_MIL_FIELDS:
        if name in frame.columns and frame[name].isna().any():
            present_incomplete.append(name)
    source_exam = {}
    if "source" in frame.columns and "exam_id" in frame.columns:
        for source, group in frame.groupby("source"):
            source_exam[str(source)] = {
                "n": int(len(group)),
                "exam_id_non_null": int(group["exam_id"].notna().sum()),
            }
    ready = not missing and not present_incomplete
    return {
        "ready": ready,
        "missing_fields": missing,
        "incomplete_fields": present_incomplete,
        "group_id_is_not_eye_visit": True,
        "source_exam_coverage": source_exam,
        "message": (
            "MIL ready."
            if ready
            else (
                "MIL blocked: missing or incomplete eye/visit metadata "
                f"(missing={missing}, incomplete={present_incomplete}). "
                f"Do not substitute {FORBIDDEN_SUBSTITUTE}."
            )
        ),
    }


def assert_mil_blocked(frame: pd.DataFrame) -> None:
    report = mil_readiness(frame)
    if report["ready"]:
        raise MILBlockedError(
            "MIL metadata appears complete; review grouping definitions before enabling MIL."
        )
    raise MILBlockedError(report["message"])


def refuse_group_id_mil() -> None:
    raise MILBlockedError(
        "Refusing MIL over group_id. Images sharing a patient/exam are not "
        "guaranteed to be the same eye or visit."
    )
