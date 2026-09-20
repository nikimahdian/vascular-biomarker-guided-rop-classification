# Pilot release status

```
PILOT TECHNICALLY READY - GOVERNANCE CLEARANCE PENDING
```

Recorded 2026-09-20. This file is a release pin. It defines the exact artefact that is waiting and
the exact conditions under which it may be sent. Nothing here is a scientific result.

## The artefact awaiting clearance

| field | value |
|---|---|
| package | `expert_pilot30_v1.0.zip` |
| SHA-256 | `0338c81ce129e05d100f711127154b6583c03deabaa10cba3bd3b3c45ebf0091` |
| size | 36,444,980 bytes |
| images | 30, the frozen pilot cohort |
| validator | `PILOT_PACKAGE_VALIDATION_PASS` — 35 checks, 0 failures |
| pixel identity | 30 of 30 identical to the previous safe build, 0 changed |

The archive itself is **not** in this repository and must not be added to it. The checksum above is
the pin; if the archive is rebuilt for any reason, the pin becomes invalid and the package must be
re-validated before it can be sent.

## Why it is not being sent yet

Two items, both outside the repository:

1. **Host-side purge.** The pre-rewrite commits are removed from the branch, but the hosting
   provider still serves the old objects by commit hash. Request prepared in
   `docs/GITHUB_SUPPORT_PURGE_REQUEST.md`; it requires the repository owner's GitHub account.
2. **Data-governance decision.** The exposure has not yet been recorded or assessed by the
   institution. Notice prepared in `docs/DATA_GOVERNANCE_NOTIFICATION.md`.

## Frozen until clearance — do not modify

- the 30 pilot study IDs and the 90 final study IDs, and their cohort membership
- the frozen blinding-key content (its checksum is unchanged)
- the pilot package and its images
- the annotation protocol and the annotation manual master
- the primary endpoints and the statistical analysis plan
- the model weights, predictions, splits and thresholds
- the leave-one-source-out results and the A/B/C comparison

The security finding does not invalidate any scientific result. It is a data-protection matter and
was not caused by, and does not change, the study design.

## Sequence once clearance is confirmed

```
Governance clearance
        |
        v
Send the 30-image pilot package to the annotators
        |
        v
Independent annotation by two graders
        |
        v
expert_validation/analysis/validate_annotations.py      <- the gate, run before any number
        |
        v
Pilot quality control and inter-grader agreement
        |
        v
At most one protocol refinement, if the pilot exposes an ambiguity
        |
        v
Protocol v1.0 frozen
        |
        v
Final cohort annotation (90 images)
        |
        v
Target-domain expert validation
        |
        v
Thesis freeze
```

The next scientific action after clearance is **collecting human annotation**, not writing new
code.
