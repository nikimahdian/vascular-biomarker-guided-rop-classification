#!/usr/bin/env bash
# Preserve the OneCycleLR variant, then re-run the harmonized ROPDeepX comparator with the
# plateau-decayed constant schedule (fair early-stopping interaction).
set -u
cd /root/niki_rop_task6_isolated || exit 9
PY=./.venv/bin/python
RES=results/fast_benchmark
for f in 0 1 2; do
  if [ -f "$RES/03_ropdeepx_oof_fold${f}.csv" ]; then
    cp -f "$RES/03_ropdeepx_oof_fold${f}.csv" "$RES/03_ropdeepx_onecycle_oof_fold${f}.csv"
  fi
  if [ -f "_bench/ckpt/ropdeepx_fold${f}_selection.json" ]; then
    cp -f "_bench/ckpt/ropdeepx_fold${f}_selection.json" \
          "$RES/03_ropdeepx_onecycle_selection_fold${f}.json"
  fi
done
echo "backed up onecycle variant:"; ls -1 "$RES" | grep onecycle
for f in 0 1 2; do
  echo "=== START ropdeepx-plateau fold$f $(date -u +%H:%M:%S)"
  $PY _bench/p01_train.py ropdeepx "$f" || echo "FAILED ropdeepx $f"
  echo "=== END ropdeepx-plateau fold$f $(date -u +%H:%M:%S)"
done
echo ROPDEEPX_PLATEAU_DONE
