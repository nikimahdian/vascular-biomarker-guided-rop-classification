#!/usr/bin/env bash
# Harmonized benchmark training: legacy B5 3-class, then ROPDeepX, all three outer folds.
set -u
cd /root/niki_rop_task6_isolated || exit 9
PY=./.venv/bin/python
for a in legacy ropdeepx; do
  for f in 0 1 2; do
    echo "=== START $a fold$f $(date -u +%H:%M:%S)"
    $PY _bench/p01_train.py "$a" "$f" || echo "FAILED $a $f"
    echo "=== END $a fold$f $(date -u +%H:%M:%S)"
  done
done
echo ALL_TRAINING_DONE
