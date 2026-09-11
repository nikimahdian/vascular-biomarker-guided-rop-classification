#!/usr/bin/env bash
# GPU follow-up after E8: E4B then V1. Canonical test locked. No --force-e5.
# Skip-complete if results.json already exists for that seed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
LOG="$ROOT/results/next_architecture/logs"
mkdir -p "$LOG"

run_train() {
  local exp="$1" seed="$2"
  echo "[followup] train $exp seed=$seed"
  "$PYTHON" -m src.classify.next_architecture train \
    --experiment "$exp" --resolution 384 --seed "$seed" \
    --config "$ROOT/configs/next_architecture.yaml"
}

for seed in 42 43 44; do
  run_train E4B "$seed"
done
for seed in 42 43 44; do
  run_train V1 "$seed"
done
echo "[followup] V1 LOSO seed 42"
"$PYTHON" -m src.classify.next_architecture loso --run \
  --experiment V1 --resolution 384 --seed 42 \
  --config "$ROOT/configs/next_architecture.yaml"

echo "[followup] pack-report after V1"
"$PYTHON" -m src.classify.next_architecture pack-report \
  --config "$ROOT/configs/next_architecture.yaml"
echo "[followup] done. Do not --force-e5. Write E5X gate from V1/E4B JSONs next."
