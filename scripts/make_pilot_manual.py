"""Section 7: the pilot derivative of the frozen annotation manual.

The ONLY body change is the workload count (120 -> 30). Annotation definitions, vessel and disc
rules, uncertainty rules and the grading sheet are byte-identical to the frozen master. Everything
else appears in a clearly marked header, which is not part of the frozen body.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EV = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_repo_clean\expert_validation")
MASTER = EV / "ANNOTATION_MANUAL.md"
PILOT = EV / "ANNOTATION_MANUAL_PILOT.md"

master = MASTER.read_text(encoding="utf-8")

HEADER = """<!--
Operational derivative of expert_validation/ANNOTATION_MANUAL.md (the frozen master).

The ONLY change to the body is the workload count: the master is written for the full 120-image
study, this derivative is written for the current 30-image pilot. Every annotation definition,
every vessel and disc rule, every uncertainty rule and the grading sheet are identical to the
frozen master. The master remains authoritative; if the two ever disagree, the master wins.
-->

> **Pilot edition — 30 images.** This is an operational derivative of the frozen annotation manual.
> The only difference is the size of the workload: this pilot contains **30** fundus photographs
> rather than the study's 120. Nothing about how to annotate has changed — the seven steps per
> image, the minimum-visible-width rule, the not-gradable disc rule, the uncertainty rules and the
> grading sheet are exactly as in the frozen master.
>
> The illustrations in `examples/` are **schematic drawings** made for this manual, not
> photographs from any dataset.

---

"""

body = master.replace("You are annotating **120 retinopathy-of-prematurity fundus photographs**",
                      "You are annotating **30 retinopathy-of-prematurity fundus photographs**", 1)
assert body != master, "the workload substitution did not apply"

PILOT.write_text(HEADER + body, encoding="utf-8")

# prove the body diff is exactly one line
a = master.splitlines()
b = body.splitlines()
diff = list(difflib.unified_diff(a, b, lineterm="", n=0))
changed = [d for d in diff if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))]
print("=== body diff between frozen master and pilot derivative ===")
for d in diff:
    print(f"  {d}")
print(f"\nchanged body lines: {len(changed)} (must be 2: one removed, one added)")
assert len(changed) == 2

print(f"\n[done] {PILOT.name}: {len(body.splitlines())} body lines + {len(HEADER.splitlines())} "
      f"header lines")
print(f"master unchanged: {MASTER.read_text(encoding='utf-8') == master}")
