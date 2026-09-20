#!/bin/zsh
# Task 4B: remaining search sections. READ-ONLY, bounded so it cannot time out.
set -u
N=/Users/moniaz/niki
H=/Users/moniaz

echo "===== A. data/ INVENTORY ====="
ls -la "$N/data/"
echo "--- data/masks non-png ---"
ls -la "$N/data/masks/" | grep -v '\.png$'
echo "--- data/masks png count ---"
ls "$N/data/masks/" | grep -c '\.png$'
echo "--- other data subdirs file counts ---"
for d in "$N"/data/*/; do printf '%-40s %s\n' "$(basename "$d")" "$(find "$d" -maxdepth 1 -type f | wc -l)"; done

echo
echo "===== B. LEGACY / INVALID RESULT DIRS ====="
for d in results/legacy_invalid data/splits_legacy_image_level_20260826 data/raw/farfum_rop/archives; do
  echo "-- $d --"
  ls -la "$N/$d" 2>/dev/null | head -12
done

echo
echo "===== C. ANY DIR OUTSIDE data/masks HOLDING >100 PNGs (depth<=6) ====="
find "$N" -maxdepth 6 -type d -not -path '*/.venv/*' -not -path "$N/data/masks*" 2>/dev/null | while read -r d; do
  n=$(ls "$d"/*.png 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 100 ]; then echo "$n  $d"; fi
done | sort -rn | head -15

echo
echo "===== D. PNG COUNT ANYWHERE ELSE UNDER HOME (depth<=5) ====="
find "$H" -maxdepth 5 -type d -not -path '*/.venv/*' -not -path '*/Library/*' -not -path "$N/*" 2>/dev/null | while read -r d; do
  n=$(ls "$d"/*.png 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -gt 50 ]; then echo "$n  $d"; fi
done | sort -rn | head -15

echo
echo "===== E. EXPERT PACKS AND PROB MAPS ====="
for p in results/biomarker_diagnostics/expert_audit_pack_v1/auto_masks \
         results/biomarker_diagnostics/expert_audit_pack_v1/expert_masks \
         results/biomarker_diagnostics/expert_mask_eval_v1 data/vessel_prob_v1 data/manifests; do
  if [ -e "$N/$p" ]; then
    echo "-- $p -- files=$(find "$N/$p" -type f | wc -l)"
    ls -la "$N/$p" | head -6
  else
    echo "-- $p -- ABSENT"
  fi
done

echo
echo "===== F. TRASH / SNAPSHOTS / TIME MACHINE ====="
ls -la "$H/.Trash" 2>/dev/null | head -15
echo "--- tmutil listlocalsnapshots / ---"
tmutil listlocalsnapshots / 2>&1 | head -15
echo "--- tmutil listbackups ---"
tmutil listbackups 2>&1 | head -15
echo "--- diskutil apfs list (snapshot lines) ---"
diskutil apfs list 2>&1 | grep -i snapshot | head -10

echo
echo "===== G. LOGS MENTIONING MASK INFERENCE ====="
grep -rl -i -E 'infer_masks|best_weight_DeepLabV3|segmentation weight' "$N/logs" "$N/results" 2>/dev/null | head -20
echo "--- logs dir listing ---"
ls -la "$N/logs" | head -40

echo
echo "===== H. SHELL HISTORY ====="
for f in "$H/.zsh_history" "$H/.bash_history"; do
  if [ -f "$f" ]; then
    echo "-- $f lines=$(wc -l < "$f") --"
    grep -n -i -E 'infer_masks|DeepLabV3|resume|mask' "$f" 2>/dev/null | head -30
  else
    echo "-- $f ABSENT --"
  fi
done

echo
echo "===== I. MASK PNG MTIME HISTOGRAM ====="
find "$N/data/masks" -maxdepth 1 -name '*.png' -exec stat -f '%Sm' -t '%Y-%m-%d %H' {} + 2>/dev/null | sort | uniq -c | sort -k2 | head -20

echo
echo "===== J. SEGMENTATION-RELATED FILES IN results/ ====="
find "$N/results" -maxdepth 2 -type f -iname '*seg*' -o -maxdepth 2 -type f -iname '*mask*' 2>/dev/null | head -20

echo
echo "===== DONE ====="
