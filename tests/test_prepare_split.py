from pathlib import Path

import pandas as pd

from src.data.prepare_split import (
    exclude_ambiguous_exact_duplicates,
    infer_identity,
    infer_label,
)


def test_plus_identity_uses_official_patient_token() -> None:
    image = Path("001_F_GA41_BW2905_PA44_DG2_PF0_D1_S01_1.jpg")
    assert infer_identity(Path("data/raw/plus"), image) == (
        "plus::001",
        "plus::001",
        "plus::001::S01::PA44",
        "verified_patient_official_dataset",
    )


def test_farabi_identity_groups_frames_under_uuid() -> None:
    root = Path("data/raw/farabi")
    first = infer_identity(root, Path("abc-def.1.jpg"))
    second = infer_identity(root, Path("abc-def.29.jpg"))
    assert first == (
        "farabi::abc-def",
        None,
        "farabi::abc-def",
        "verified_exam_patient_unknown",
    )
    assert second == first


def test_label_inference_uses_source_relative_path() -> None:
    assert (
        infer_label(
            Path("Plus_dataset_main/Normal_comp/x.jpg"),
            ["pre plus"],
            ["plus"],
            ["normal_comp"],
        )
        == 0
    )
    assert (
        infer_label(
            Path("Plus/Pre Plus/x.jpg"),
            ["pre plus"],
            ["plus"],
            ["no plus"],
        )
        == 1
    )


def test_cross_patient_duplicate_excludes_whole_implicated_identities(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / name for name in ["a1.jpg", "a2.jpg", "b1.jpg", "c1.jpg"]]
    paths[0].write_bytes(b"duplicate")
    paths[1].write_bytes(b"unique-a")
    paths[2].write_bytes(b"duplicate")
    paths[3].write_bytes(b"unique-c")
    frame = pd.DataFrame(
        {
            "image_path": [str(path) for path in paths],
            "group_id": ["a", "a", "b", "c"],
            "patient_id": ["a", "a", "b", "c"],
            "label": [0, 0, 0, 0],
            "source": ["test"] * 4,
            "identity_level": ["test"] * 4,
        }
    )

    retained, excluded = exclude_ambiguous_exact_duplicates(frame)

    assert set(retained["group_id"]) == {"c"}
    assert set(excluded["group_id"]) == {"a", "b"}
    assert excluded["triggering_duplicate"].sum() == 2
