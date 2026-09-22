"""Create the SEG_CURRENT_V1_HVDROPDB_EXTERNAL generation contract + index, then it is run via
`python -m src.segmentation.infer_masks --generation SEG_CURRENT_V1_HVDROPDB_EXTERNAL --index ...`.

The contract is a faithful copy of the frozen SEG_CURRENT_V1 contract with a NEW generation_id, a
NEW (writable) output store, and the population fields pointed at the 100 HVDROPDB references.
Model, preprocessing, threshold, postprocessing, inference-script SHA, config SHA and checkpoint
SHA are byte-identical - no threshold tuning, no preprocessing change, no checkpoint substitution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("/Users/moniaz/niki")
GEN = "SEG_CURRENT_V1_HVDROPDB_EXTERNAL"
SRC_CONTRACT = ROOT / "configs/segmentation_generation_current_v1.yaml"
NEW_CONTRACT = ROOT / "configs/segmentation_generation_current_v1_hvdrodb_external.yaml"
STORE = "data/masks_external/hvdrodb_seg_current_v1"
INDEX = ROOT / "_private_audit/task5c_j_external_index.csv"
BV = ROOT / "data/raw/hvdro/segmentation/HVDROPDB_RetCam_Neo_Segmentation/HVDROPDB-BV"

CAMERAS = {"RetCam": "RetCam_Vessels", "Neo": "Neo_Vessels"}


def main() -> None:
    rows = []
    for cam, stem in CAMERAS.items():
        idir, mdir = BV / f"{stem}_images", BV / f"{stem}_masks"
        for ip in sorted(idir.iterdir()):
            if ip.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                continue
            hits = list(mdir.glob(ip.stem + ".*"))
            rows.append({"image_path": str(ip), "expert_mask": str(hits[0]) if hits else "",
                         "source": f"hvdrodb_{cam.lower()}", "split": "external_benchmark",
                         "label": -1, "group_id": ip.stem, "patient_id": ip.stem,
                         "exam_id": ip.stem, "identity_level": "external_reference"})
    idx = pd.DataFrame(rows)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(INDEX, index=False)
    print(f"index rows {len(idx)}  RetCam {int((idx.source == 'hvdrodb_retcam').sum())} "
          f"Neo {int((idx.source == 'hvdrodb_neo').sum())}  with expert mask "
          f"{int((idx.expert_mask != '').sum())}")

    c = yaml.safe_load(SRC_CONTRACT.read_text(encoding="utf-8"))
    c["generation_id"] = GEN
    c["created_at"] = "2026-09-22T13:00:00+0330"
    c["supersedes"] = None
    c["store_frozen"] = False
    c["output_store"] = STORE
    c["canonical_population_fingerprint"] = "EXTERNAL_BENCHMARK_NOT_CANONICAL"
    c["canonical_population_n"] = int(len(idx))
    c["canonical_population_file"] = str(INDEX.relative_to(ROOT))
    c["canonical_mask_mapping_file"] = "created_by_this_run"
    c["canonical_mask_mapping_sha256"] = "created_by_this_run"
    c["source_counts"] = idx.source.value_counts().to_dict()
    c["split_counts"] = idx.split.value_counts().to_dict()
    c["note"] = ("EXTERNAL DEVELOPMENT / MEASUREMENT BENCHMARK inference. Same frozen model, "
                 "preprocessing, threshold and postprocessing as SEG_CURRENT_V1, run on the 100 "
                 "HVDROPDB expert-reference images. These masks are NOT canonical-cohort masks and "
                 "must not overwrite data/masks/.")
    c["input_inventory"] = "_private_audit/task5c_j_external_inputs.csv"
    c["mask_inventory"] = "_private_audit/task5c_j_external_masks.csv"
    c["inventory_summary"] = "_private_audit/task5c_j_external_inventory.json"
    NEW_CONTRACT.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"contract written {NEW_CONTRACT}")
    for k in ("architecture", "encoder", "checkpoint_path", "checkpoint_sha256", "input_size",
              "threshold", "minimum_component_area", "inference_script_sha256", "config_sha256"):
        print(f"  {k:28s} {c[k]}")


if __name__ == "__main__":
    main()

