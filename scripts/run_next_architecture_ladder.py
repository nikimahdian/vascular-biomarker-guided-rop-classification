#!/usr/bin/env python3
"""Entry point: python scripts/run_next_architecture_ladder.py

Prefer the shell wrapper on studio (lock + caffeinate):
    bash scripts/run_next_architecture_ladder.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify.next_architecture.ladder import main


if __name__ == "__main__":
    raise SystemExit(main())
