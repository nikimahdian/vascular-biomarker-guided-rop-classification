#!/bin/sh
# Quarantine the August-25 legacy image-level-split artifacts so they cannot be mistaken
# for canonical results. Moves files; deletes nothing.
set -e
cd /Users/moniaz/niki/results || exit 1
mkdir -p legacy_invalid

for f in comparison.csv rigorous_eval.json rigorous_summary.csv stats_tests.json \
         operating_points.csv calibration_summary.json source_macro_auc.csv \
         hybrid_ablation.csv fusion_results.json branch_b_results.json.old \
         calibration_reliability.png comparison_bar.png roc_comparison.png; do
  if [ -f "$f" ]; then
    mv "$f" legacy_invalid/
    echo "moved $f"
  else
    echo "absent $f"
  fi
done

cat > legacy_invalid/README.md <<'EOF'
# Legacy artifacts — INVALID, do not cite

These files were produced on 2026-08-25 on an **image-level split**
(`data/splits_legacy_image_level_20260826`, 1330 test rows / 216 Plus) — before the
grouped, leakage-controlled canonical split was built.

They report far higher numbers than the canonical results and, critically, a **different
ranking**:

| | legacy (these files) | canonical (locked test) |
|---|---|---|
| Branch A | 0.905 | 0.799926 |
| Branch B | 0.981 | 0.928008 |
| Branch C (fusion) | **0.988** | 0.912315 |

The legacy run made fusion look like the winner. The canonical run does not.
`stats_tests.json` here also has `n_test_rows: 1330`, not the canonical 1331.

Canonical sources of truth:
* `results/branch_a_results.json`, `results/branch_b_results.json`, `results/branch_c_results.json`
* `results/branch_c_late_fusion_probe.json`
* `artifacts/phase5_locked.csv` (public repo)
* split fingerprint `0d4c3b3a60761ca1bda88924dbc0cbf6f1be604a6e10dd5e981e40b73f05f9c8`

A copy of this legacy set also exists at `results/current_provisional_20260826/`.
Nothing here should be quoted in the thesis or the paper.
EOF

echo "---"
echo "legacy_invalid/ now contains:"
ls -la legacy_invalid
