#!/usr/bin/env python
"""Task 5A (K): public generation summary. No patient identifiers, no absolute paths."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = "/Users/moniaz/niki"
PRIV = f"{ROOT}/_private_audit"
ART = f"{ROOT}/artifacts"
os.makedirs(ART, exist_ok=True)


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


inv = json.load(open(f"{PRIV}/seg_current_v1_inventory.json"))
sub = json.load(open(f"{PRIV}/seg_current_v1_repro_subset.json"))
sub_r = json.load(open(f"{PRIV}/seg_current_v1_repro_result.json"))
full_r = json.load(open(f"{PRIV}/seg_current_v1_full_repro_result.json"))
pair = json.load(open(f"{PRIV}/seg_current_v1_pairing_proof.json"))

summary = {
    "generation_id": "SEG_CURRENT_V1",
    "classification": "CURRENT_MASK_GENERATION",
    "not_historical_equivalent": True,
    "historical_counterpart": {
        "name": "Aug-27 generation",
        "status": "IRRECOVERABLE",
        "note": "produced data/features/biomarker_features.csv on 2026-08-27; not reproducible "
                "by any available checkpoint (see docs/HISTORICAL_SEGMENTATION_RECOVERY.md)",
    },
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "contract_file": "configs/segmentation_generation_current_v1.yaml",
    "contract_sha256": sha(f"{ROOT}/configs/segmentation_generation_current_v1.yaml"),

    "population_n": inv["input_rows"],
    "canonical_population_fingerprint":
        "0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8",
    "source_counts": inv["input_source_counts"],
    "split_counts": inv["input_split_counts"],

    "checkpoint_path": "weights/best_weight_DeepLabV3+_resize_27",
    "checkpoint_sha256": "c373f53813ee60b89651a04a98bc5f1d6bc60a5a50f45c4e10f42475650d5374",
    "checkpoint_size_bytes": 127358298,
    "checkpoint_note": "MAnet weights under a DeepLabV3+-shaped filename; byte-identical to "
                       "the Google Drive artifact best_weight_MAnet_res34_resize_31",
    "architecture": "MAnet",
    "encoder": "resnet34",
    "config_sha256": "77a6c9230fd8b6f7230e1085d3415f46ece862d3e160ace983a1be83b97f0a60",
    "inference_code_sha256": "b76f820025353841908111674a9ca3460776f7add496d6c2a27ff4bbf09f1a1b",
    "inference_code_file": "src/segmentation/infer_masks.py",

    "preprocessing": {
        "input_size": [256, 256],
        "resize_policy": "torchvision Resize((256,256)), bilinear, before ToTensor",
        "aspect_ratio_policy": "not preserved; source geometries stretched to square",
        "normalization": "none beyond ToTensor scaling to [0,1]",
    },
    "decision_rule": {
        "activation": "sigmoid once",
        "threshold": 0.20,
        "binarization_operator": "strict greater-than",
        "connected_component_policy": "8-connectivity, components below min area removed",
        "minimum_component_area": 50,
        "morphological_operations": "MORPH_CLOSE 3x3 after component removal",
    },
    "output": {
        "binary_output_values": [0, 255],
        "dtype": "uint8 grayscale PNG",
        "output_resize_policy": "cv2.resize INTER_NEAREST to source size",
        "store": "data/masks",
        "store_frozen": True,
    },

    "mask_count": inv["mask_rows"],
    "mask_unique_sha256": inv["mask_unique_sha256"],
    "input_unique_image_sha256": inv["input_unique_image_sha256"],
    "aggregate_dimensions": inv["input_dimensions"],
    "vessel_pixel_total": inv["vessel_pixel_total"],
    "identity_ok": inv["identity_ok"],
    "identity_tuple": "(image_sha256, generation_id, mask_sha256)",

    "store_note": "the frozen store holds 8960 PNGs; 8870 are the SEG_CURRENT_V1 canonical "
                  "masks and 90 belong to images outside the canonical cohort. Those 90 are "
                  "left untouched and are not part of this generation's inventory.",

    "reproducibility": {
        "subset_n": sub_r["SUBSET_N"],
        "subset_bitwise_identical_n": sub_r["BITWISE_IDENTICAL_N"],
        "subset_covers_all_sources": sub["covers_all_sources"],
        "subset_covers_all_splits": sub["covers_all_splits"],
        "subset_covers_all_geometries": sub["covers_all_geometries"],
        "subset_strata_covered": sub["strata_covered"],
        "subset_strata_total": sub["strata_total"],
        "full_reproduction_run": True,
        "full_reproduction_n": full_r["SUBSET_N"],
        "full_bitwise_identical_n": full_r["BITWISE_IDENTICAL_N"],
        "deterministic": full_r["DETERMINISTIC"],
        "gate": full_r["GATE"],
        "method": "re-ran the frozen recipe under the SEG_CURRENT_V1_REPRO contract into a "
                  "scratch directory; the frozen store was never written to",
    },

    "pairing": {
        "strict_image_mask_pairing": pair["STRICT_IMAGE_MASK_PAIRING"],
        "images": pair["images"],
        "resolved_exactly_one": pair["resolved_exactly_one"],
        "resolution_errors": len(pair["errors"]),
        "mismatches_vs_pinned": len(pair["mismatches"]),
        "resolver": "src/segmentation/mask_pairing.py::resolve_mask (exact stem parse, "
                    "never first-match, never prefix-match)",
    },

    "overwrite_protection": {
        "default_behaviour": "ERROR on existing output",
        "frozen_store_refuses_write": True,
        "allow_overwrite_flag_required_to_clobber": True,
        "new_state_requires_new_generation_id": True,
    },
}
json.dump(summary, open(f"{ART}/seg_current_v1_summary.json", "w"), indent=2, default=str)
print(f"wrote {ART}/seg_current_v1_summary.json")
print()
for k in ("generation_id", "population_n", "mask_count", "checkpoint_sha256",
          "config_sha256", "inference_code_sha256", "checkpoint_size_bytes"):
    print(f"  {k}: {summary[k]}")
print(f"  reproducibility: {summary['reproducibility']['subset_bitwise_identical_n']}/"
      f"{summary['reproducibility']['subset_n']} subset, "
      f"{summary['reproducibility']['full_bitwise_identical_n']}/"
      f"{summary['reproducibility']['full_reproduction_n']} full")
print(f"  pairing: {summary['pairing']['strict_image_mask_pairing']}")
