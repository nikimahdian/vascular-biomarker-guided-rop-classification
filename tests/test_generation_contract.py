"""Segmentation generation identity and overwrite protection.

Two regressions this suite exists to prevent, both observed in this project:

1. a mask file silently overwritten because its name is a deterministic function of the
   image path;
2. masks produced from an unpinned or mismatched state, so that two different mask sets
   share one name and no later analysis can separate them.

The contract tests are hermetic: they build a throwaway root in tmp_path. One test reads the
real SEG_CURRENT_V1 contract from the repository and checks it is well formed, which needs no
data or weights.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.segmentation.generation import (
    GenerationError,
    MaskOverwriteError,
    assert_store_writable,
    assert_writable,
    content_identity,
    contract_path,
    load_contract,
    output_store,
    sha256_file,
    verify_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {
    "generation_id": "SEG_TEST_V1",
    "architecture": "MAnet",
    "encoder": "resnet34",
    "checkpoint_path": "weights/fake_checkpoint",
    "input_size": [256, 256],
    "resize_policy": "resize",
    "aspect_ratio_policy": "stretch",
    "normalization": "none",
    "activation": "sigmoid",
    "threshold": 0.2,
    "connected_component_policy": "8-connectivity",
    "minimum_component_area": 50,
    "morphological_operations": "close 3x3",
    "binary_output_values": [0, 255],
    "output_dtype": "uint8",
    "output_resize_policy": "nearest",
    "inference_script": "src/segmentation/infer_masks.py",
    "created_at": "2026-01-01T00:00:00+0000",
    "canonical_population_fingerprint": "0" * 64,
    "output_store": "data/masks/SEG_TEST_V1",
    "store_frozen": False,
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def fake_root(tmp_path):
    """A throwaway project root with a checkpoint, a script, a config and a contract."""
    root = tmp_path
    _write(root / "configs" / "config.yaml", "seed: 42\n")
    _write(root / "weights" / "fake_checkpoint", "checkpoint-bytes")
    _write(root / "src" / "segmentation" / "infer_masks.py", "# inference script\n")

    contract = dict(REQUIRED_FIELDS)
    contract["checkpoint_sha256"] = sha256_file(root / "weights" / "fake_checkpoint")
    contract["checkpoint_size_bytes"] = (root / "weights" / "fake_checkpoint").stat().st_size
    contract["inference_script_sha256"] = sha256_file(
        root / "src" / "segmentation" / "infer_masks.py")
    contract["config_sha256"] = sha256_file(root / "configs" / "config.yaml")

    import yaml

    _write(root / "configs" / "segmentation_generation_test_v1.yaml",
           yaml.safe_dump(contract, sort_keys=True))
    return root, contract


def _cfg(root: Path) -> dict:
    return {"_root": root, "paths": {"masks_dir": root / "data" / "masks"}}


# --------------------------------------------------------------- contract resolution
def test_contract_path_drops_the_seg_prefix():
    cfg = {"_root": Path("/tmp/whatever")}
    assert contract_path(cfg, "SEG_CURRENT_V1").name == \
        "segmentation_generation_current_v1.yaml"
    assert contract_path(cfg, "SEG_CURRENT_V1_REPRO").name == \
        "segmentation_generation_current_v1_repro.yaml"


def test_the_repository_contract_is_well_formed():
    cfg = {"_root": REPO_ROOT}
    contract = load_contract(cfg, "SEG_CURRENT_V1")
    assert contract["generation_id"] == "SEG_CURRENT_V1"
    assert contract["store_frozen"] is True
    assert contract["threshold"] == 0.20
    assert contract["input_size"] == [256, 256]
    assert contract["minimum_component_area"] == 50
    assert contract["binary_output_values"] == [0, 255]
    assert len(contract["checkpoint_sha256"]) == 64
    assert len(contract["inference_script_sha256"]) == 64
    # the real generation must never claim to be the historical one, and must not carry the
    # repro-only marker
    assert "not_a_scientific_generation" not in contract
    assert contract["generation_id"] == "SEG_CURRENT_V1"


def test_missing_contract_fails_loudly(tmp_path):
    with pytest.raises(GenerationError, match="no generation contract"):
        load_contract({"_root": tmp_path}, "SEG_NOPE_V1")


def test_generation_id_mismatch_fails_loudly(fake_root):
    """A contract sitting at the right path but declaring a different id must be rejected."""
    root, contract = fake_root
    import yaml

    _write(root / "configs" / "segmentation_generation_other_v1.yaml",
           yaml.safe_dump(contract, sort_keys=True))  # content still says SEG_TEST_V1
    with pytest.raises(GenerationError, match="declares generation_id"):
        load_contract(_cfg(root), "SEG_OTHER_V1")


def test_missing_required_field_fails_loudly(fake_root):
    root, contract = fake_root
    import yaml

    del contract["threshold"]
    _write(root / "configs" / "segmentation_generation_test_v1.yaml",
           yaml.safe_dump(contract, sort_keys=True))
    with pytest.raises(GenerationError, match="missing required fields"):
        load_contract(_cfg(root), "SEG_TEST_V1")


# --------------------------------------------------------------- byte-hash verification
def test_contract_verifies_when_state_matches(fake_root):
    root, _ = fake_root
    observed = verify_contract(_cfg(root), load_contract(_cfg(root), "SEG_TEST_V1"))
    assert observed["checkpoint_sha256"] == sha256_file(root / "weights" / "fake_checkpoint")
    assert observed["config_sha256"] == sha256_file(root / "configs" / "config.yaml")


def test_changed_checkpoint_invalidates_the_contract(fake_root):
    root, _ = fake_root
    contract = load_contract(_cfg(root), "SEG_TEST_V1")
    _write(root / "weights" / "fake_checkpoint", "different-checkpoint-bytes")
    with pytest.raises(GenerationError, match="does not match the current state"):
        verify_contract(_cfg(root), contract)


def test_changed_code_invalidates_the_contract(fake_root):
    root, _ = fake_root
    contract = load_contract(_cfg(root), "SEG_TEST_V1")
    _write(root / "src" / "segmentation" / "infer_masks.py", "# changed\n")
    with pytest.raises(GenerationError, match="inference_script_sha256"):
        verify_contract(_cfg(root), contract)


def test_changed_config_invalidates_the_contract(fake_root):
    root, _ = fake_root
    contract = load_contract(_cfg(root), "SEG_TEST_V1")
    _write(root / "configs" / "config.yaml", "seed: 7\n")
    with pytest.raises(GenerationError, match="config_sha256"):
        verify_contract(_cfg(root), contract)


def test_verify_reports_every_mismatch_at_once(fake_root):
    root, _ = fake_root
    contract = load_contract(_cfg(root), "SEG_TEST_V1")
    _write(root / "weights" / "fake_checkpoint", "x")
    _write(root / "configs" / "config.yaml", "seed: 1\n")
    with pytest.raises(GenerationError) as exc:
        verify_contract(_cfg(root), contract)
    assert "checkpoint_sha256" in str(exc.value)
    assert "config_sha256" in str(exc.value)


# --------------------------------------------------------------- output store policy
def test_frozen_store_refuses_any_write(fake_root):
    root, contract = fake_root
    contract = dict(contract, store_frozen=True, output_store="data/masks")
    with pytest.raises(GenerationError, match="FROZEN"):
        assert_store_writable(contract, allow_overwrite=False)
    with pytest.raises(GenerationError, match="FROZEN"):
        assert_store_writable(contract, allow_overwrite=True)


def test_non_frozen_store_rejects_the_destructive_flag(fake_root):
    _, contract = fake_root
    with pytest.raises(GenerationError, match="does not apply to a per-generation store"):
        assert_store_writable(contract, allow_overwrite=True)


def test_non_frozen_store_is_writable(fake_root):
    _, contract = fake_root
    assert_store_writable(contract, allow_overwrite=False)


def test_output_store_resolves_under_root(fake_root):
    root, contract = fake_root
    assert output_store(_cfg(root), contract) == root / "data" / "masks" / "SEG_TEST_V1"


# --------------------------------------------------------------- overwrite protection
def test_existing_output_is_an_error_by_default(tmp_path):
    target = tmp_path / "mask.png"
    target.write_bytes(b"previous run's bytes")
    with pytest.raises(MaskOverwriteError, match="refusing to overwrite"):
        assert_writable(target, allow_overwrite=False, generation_id="SEG_TEST_V1")


def test_existing_output_is_allowed_only_with_the_explicit_flag(tmp_path):
    target = tmp_path / "mask.png"
    target.write_bytes(b"previous run's bytes")
    assert_writable(target, allow_overwrite=True, generation_id="SEG_TEST_V1")


def test_absent_output_is_always_writable(tmp_path):
    assert_writable(tmp_path / "not-yet.png", allow_overwrite=False,
                    generation_id="SEG_TEST_V1")


def test_overwrite_error_names_the_generation_and_the_file(tmp_path):
    target = tmp_path / "mask.png"
    target.write_bytes(b"x")
    with pytest.raises(MaskOverwriteError) as exc:
        assert_writable(target, allow_overwrite=False, generation_id="SEG_CURRENT_V1")
    assert "SEG_CURRENT_V1" in str(exc.value)
    assert target.name in str(exc.value)


# --------------------------------------------------------------- content identity
def test_identity_is_content_based_not_filename_based(tmp_path):
    image = tmp_path / "whatever_name.jpg"
    image.write_bytes(b"image-bytes")
    mask = tmp_path / "a_completely_different_name.png"
    mask.write_bytes(b"mask-bytes")

    ident = content_identity(str(image), str(mask))
    assert ident["image_sha256"] == hashlib.sha256(b"image-bytes").hexdigest()
    assert ident["mask_sha256"] == hashlib.sha256(b"mask-bytes").hexdigest()
    assert ident["mask_bytes"] == len(b"mask-bytes")

    renamed = tmp_path / "renamed.png"
    renamed.write_bytes(b"mask-bytes")
    assert content_identity(str(image), str(renamed))["mask_sha256"] == ident["mask_sha256"]


def test_identity_changes_when_mask_bytes_change(tmp_path):
    image = tmp_path / "i.jpg"
    image.write_bytes(b"image-bytes")
    mask = tmp_path / "m.png"
    mask.write_bytes(b"mask-bytes")
    before = content_identity(str(image), str(mask))
    mask.write_bytes(b"mask-bytes ")
    assert content_identity(str(image), str(mask))["mask_sha256"] != before["mask_sha256"]


def test_two_images_with_identical_bytes_share_an_image_hash(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    m = tmp_path / "m.png"
    m.write_bytes(b"mask")
    assert (content_identity(str(a), str(m))["image_sha256"]
            == content_identity(str(b), str(m))["image_sha256"])
