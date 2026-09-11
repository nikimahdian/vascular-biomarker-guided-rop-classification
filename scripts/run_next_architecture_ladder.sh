#!/usr/bin/env bash
# Sequential next-architecture ladder with resume. Canonical Branch B/C files are never written.
#
# Studio:
#   bash scripts/run_next_architecture_ladder.sh
#   bash scripts/run_next_architecture_ladder.sh --profile e0
#   bash scripts/run_next_architecture_ladder.sh --stop-after e0
#   bash scripts/run_next_architecture_ladder.sh --dry-run
#
# If seed 43 (or any train) is already running, this waits, then continues remaining jobs.
# Re-run the same command after a crash: completed results.json are skipped, train.pt resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/next_architecture/logs results/next_architecture/ladder

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ "$(uname -s)" == "Darwin" ]]; then
  exec caffeinate -dimsu "$PYTHON" "$ROOT/scripts/run_next_architecture_ladder.py" "$@"
fi
exec "$PYTHON" "$ROOT/scripts/run_next_architecture_ladder.py" "$@"
