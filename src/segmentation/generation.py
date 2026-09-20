"""Immutable segmentation generation identity.

A segmentation generation is one frozen combination of checkpoint bytes, inference code,
config, preprocessing, threshold and postprocessing. Every mask is traceable to that
combination plus its own input and output bytes.

Two failure modes this module exists to prevent, both observed in this project:

1. a mask file being silently overwritten, because its name is a deterministic function of
   the image path (`infer_masks.py::unique_mask_name`);
2. a generation being reconstructed from an ambiguous state, so that two different mask sets
   share one name and no later analysis can tell them apart.

Nothing here changes any scientific definition. It only refuses to proceed when identity is
not pinned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

CONTRACT_DIRNAME = "configs"
CONTRACT_SUFFIX = ".yaml"


class GenerationError(RuntimeError):
    """The generation contract is missing, malformed, or does not match the current state."""


class MaskOverwriteError(RuntimeError):
    """A mask output already exists and no explicit destructive override was given."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_path(cfg: dict, generation_id: str) -> Path:
    """configs/segmentation_generation_<id>.yaml, with a leading SEG_ dropped from the id.

    SEG_CURRENT_V1 -> configs/segmentation_generation_current_v1.yaml
    """
    root = Path(cfg["_root"])
    slug = str(generation_id).lower()
    if slug.startswith("seg_"):
        slug = slug[4:]
    return root / CONTRACT_DIRNAME / f"segmentation_generation_{slug}{CONTRACT_SUFFIX}"


def load_contract(cfg: dict, generation_id: str) -> dict:
    path = contract_path(cfg, generation_id)
    if not path.exists():
        raise GenerationError(
            f"no generation contract for {generation_id!r} at {path}. "
            "Every mask-producing run must be described by a contract; create one and give "
            "the new generation its own id."
        )
    with path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise GenerationError(f"{path} is not a mapping")
    if contract.get("generation_id") != generation_id:
        raise GenerationError(
            f"{path} declares generation_id={contract.get('generation_id')!r}, "
            f"not {generation_id!r}"
        )
    required = [
        "generation_id", "architecture", "encoder", "checkpoint_path", "checkpoint_sha256",
        "checkpoint_size_bytes", "input_size", "resize_policy", "aspect_ratio_policy",
        "normalization", "activation", "threshold", "connected_component_policy",
        "minimum_component_area", "morphological_operations", "binary_output_values",
        "output_dtype", "output_resize_policy", "inference_script",
        "inference_script_sha256", "config_sha256", "created_at",
        "canonical_population_fingerprint", "output_store", "store_frozen",
    ]
    missing = [k for k in required if k not in contract]
    if missing:
        raise GenerationError(f"{path} is missing required fields: {missing}")
    return contract


def verify_contract(cfg: dict, contract: dict) -> dict:
    """Recompute every byte hash in the contract and compare. Raises on any mismatch."""
    root = Path(cfg["_root"])
    observed: dict[str, str] = {}

    checkpoint = Path(contract["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.exists():
        raise GenerationError(f"checkpoint missing: {checkpoint}")
    observed["checkpoint_sha256"] = sha256_file(checkpoint)
    observed["checkpoint_size_bytes"] = checkpoint.stat().st_size

    script = Path(contract["inference_script"])
    if not script.is_absolute():
        script = root / script
    if not script.exists():
        raise GenerationError(f"inference script missing: {script}")
    observed["inference_script_sha256"] = sha256_file(script)

    cfg_path = root / "configs" / "config.yaml"
    observed["config_sha256"] = sha256_file(cfg_path)

    mismatches = []
    for key in ("checkpoint_sha256", "checkpoint_size_bytes",
                "inference_script_sha256", "config_sha256"):
        if contract.get(key) != observed[key]:
            mismatches.append(f"{key}: contract={contract.get(key)!r} observed={observed[key]!r}")
    if mismatches:
        raise GenerationError(
            "generation contract does not match the current state:\n  "
            + "\n  ".join(mismatches)
            + "\nIf the checkpoint, code or config changed, the output is a NEW generation and "
              "needs a NEW generation_id. Do not edit this contract to make it match."
        )
    return observed


def output_store(cfg: dict, contract: dict) -> Path:
    """Where this generation's masks live. A frozen store may never be written to."""
    store = Path(contract["output_store"])
    if not store.is_absolute():
        store = Path(cfg["_root"]) / store
    if contract.get("store_frozen") and store != Path(cfg["paths"]["masks_dir"]):
        raise GenerationError(
            f"{contract['generation_id']} declares store_frozen with output_store={store}; "
            "a frozen store must be the canonical masks directory"
        )
    return store


def assert_store_writable(contract: dict, *, allow_overwrite: bool) -> None:
    if contract.get("store_frozen"):
        raise GenerationError(
            f"{contract['generation_id']} is FROZEN (output_store={contract['output_store']}). "
            "Its masks are the reference bytes for the canonical cohort and must not be "
            "recomputed in place. To produce masks again, define a new generation with its own "
            "id and its own output directory."
        )
    if allow_overwrite:
        raise GenerationError(
            "--allow-overwrite does not apply to a per-generation store: write the new "
            "generation into its own directory instead of overwriting an existing one."
        )


def assert_writable(target: Path, *, allow_overwrite: bool, generation_id: str) -> None:
    """Refuse to clobber an existing output unless a destructive override was explicit."""
    if target.exists() and not allow_overwrite:
        raise MaskOverwriteError(
            f"refusing to overwrite existing mask {target} (generation {generation_id}). "
            "Default behaviour is ERROR, not overwrite. Pass --allow-overwrite only if you "
            "intend to destroy the previous bytes; otherwise use a new generation id."
        )


def content_identity(image_path: str, mask_path: str) -> dict:
    """Scientific identity of one mask: input bytes + generation + output bytes."""
    return {
        "image_sha256": sha256_file(image_path),
        "mask_sha256": sha256_file(mask_path),
        "mask_bytes": Path(mask_path).stat().st_size,
    }


def write_generation_manifest(path: Path, contract: dict, records: list[dict]) -> None:
    """Persist the generation manifest with its contract attached, atomically."""
    payload = {
        "generation_id": contract["generation_id"],
        "contract_sha256": sha256_file(contract_path(
            {"_root": Path(path).parents[2]}, contract["generation_id"])),
        "n": len(records),
        "records": records,
    }
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
