"""Load next-architecture config and verify the canonical split fingerprint."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.utils.common import ROOT, sha256_file

DEFAULT_CONFIG = ROOT / "configs" / "next_architecture.yaml"
CANONICAL_SPLIT_SHA = "0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8"
FORBIDDEN_RESULT_PREFIXES = (
    "branch_a_",
    "branch_b_",
    "branch_c_",
)


def config_sha256(path: Path) -> str:
    return sha256_file(path)


def load_next_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_root"] = ROOT
    cfg["_config_path"] = cfg_path.resolve()
    cfg["_config_sha256"] = config_sha256(cfg_path)
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = (ROOT / rel).resolve()
    return cfg


def resolved_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(cfg)
    payload.pop("_root", None)
    for key, value in list(payload.get("paths", {}).items()):
        payload["paths"][key] = str(value)
    payload["_config_path"] = str(cfg["_config_path"])
    return payload


def dump_resolved_config(cfg: dict[str, Any]) -> str:
    return json.dumps(resolved_config_dict(cfg), sort_keys=True, default=str)


def resolved_config_sha(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(dump_resolved_config(cfg).encode("utf-8")).hexdigest()


def verify_canonical_split(cfg: dict[str, Any], *, allow_noncanonical: bool = False) -> str:
    split_path = cfg["paths"]["splits_dir"] / "all.csv"
    observed = sha256_file(split_path)
    expected = cfg.get("expected_split_sha256", CANONICAL_SPLIT_SHA)
    if observed != expected:
        if allow_noncanonical:
            return observed
        raise SystemExit(
            f"Canonical split hash mismatch.\n"
            f"  expected: {expected}\n"
            f"  observed: {observed}\n"
            f"  file: {split_path}\n"
            "Refusing to run. Pass --allow-noncanonical only for a separately named experiment."
        )
    return observed


def assert_output_namespace(path: Path) -> None:
    """Refuse writes that would clobber canonical Branch A/B/C files."""
    name = path.name
    if any(name.startswith(prefix) for prefix in FORBIDDEN_RESULT_PREFIXES):
        raise SystemExit(f"Refusing to write canonical artifact path: {path}")
    if "next_architecture" not in path.as_posix() and "smoke" not in path.as_posix():
        # Allow manifests/vessel dirs which are versioned new paths.
        if "vessel_prob" in path.as_posix() or "vessel_derived" in path.as_posix():
            return
        if "manifests" in path.as_posix():
            return
        if path.suffix in {".yaml", ".yml"}:
            return
        raise SystemExit(
            f"Next-architecture outputs must live under a versioned namespace, got {path}"
        )


def short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]
