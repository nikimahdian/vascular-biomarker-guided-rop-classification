# Security remediation

**Status: Pilot technically ready, governance clearance pending.**
The clinician pilot package is built, validated and pixel-verified. It should not be sent until the
two outstanding items in section *RESIDUAL EXPOSURE* are closed.

**Date of remediation:** 2026-09-20
**Scope:** removal of sensitive research metadata from the public repository and from its reachable
git history.

**Follow-ups, outside this repository:**

| item | owner | document |
|---|---|---|
| Garbage-collect the old server-side objects | repository owner (GitHub account required) | `docs/GITHUB_SUPPORT_PURGE_REQUEST.md` |
| Record and assess the exposure | institutional data governance | `docs/DATA_GOVERNANCE_NOTIFICATION.md` |

## What category of file was removed

Three categories of file had been committed to this public repository and have now been removed
from every commit, not only from the current revision:

1. **A private study key.** A CSV mapping blinded study identifiers to their source images, with
   acquisition source, diagnostic label, split assignment and absolute local file paths.
2. **Dataset manifest tables.** Several CSV files listing every image in the study together with
   absolute local paths, acquisition source, diagnostic label, split assignment, and the grouping
   and identity columns used to keep the same patient or examination out of more than one split.
3. **Retinal image figures.** Three documentation figures that contained identifiable fundus
   photographs rather than aggregate plots.

## Why these should not have been public

Each of them links an individual fundus photograph to its diagnostic label, its acquisition source,
and either a patient or examination identifier, or a local file path that contains such an
identifier. Published together with the repository they amount to a re-identification resource for
a clinical dataset of premature infants, and they disclose the study's blinding structure. None of
this is necessary to reproduce the method, and none of it belongs in a public repository.

The figures carried the same problem in visual form: a full-colour fundus photograph, and two
heat-map overlays rendered on top of identifiable fundus photographs.

## What was done

- The files were removed from every commit in the history, not merely from the current revision.
- The rewritten history was pushed, and the old commits are no longer reachable from any branch.
- The removed files are retained privately by the study, under access control, outside this
  repository.
- `.gitignore` now blocks these paths so they cannot be re-added by accident.
- The full object database was re-scanned after the rewrite: no credential, no patient or
  examination identifier, no clinical-image data and no dataset manifest remains in any object
  that a clone receives.

## RESIDUAL EXPOSURE — NOT YET RESOLVED

Rewriting the history removes the old commits from the branch, but it does not remove the old
objects from the hosting provider. **Every pre-rewrite commit is still retrievable from GitHub by
its commit hash**, and this was verified after the rewrite:

- the dataset manifest table and the private study key were both successfully fetched from the
  remote by their old commit hashes, and
- the fundus-overlay figure was likewise retrievable.

Because those commit hashes were public, the data must be treated as disclosed for the period it
was online. Two actions remain, and neither can be completed from inside this repository:

1. **Ask GitHub Support to garbage-collect the unreachable objects** and purge cached views of the
   repository. GitHub documents this as the required follow-up to a history rewrite: the rewritten
   history takes effect for normal access, but the old objects persist until Support runs a prune.
   A ready-to-send request is in `docs/GITHUB_SUPPORT_PURGE_REQUEST.md`.
2. **Treat the affected dataset as disclosed** and follow the study's data-governance route. The
   exposure is of a research manifest linking images to labels and of a small number of figures
   containing identifiable fundus photographs; it is not a credential leak.

Until action 1 completes, this remediation is **partial**: the branch and any fresh clone are
clean, but the old objects are still served on request.

## Confirmed not present

A fresh clone of the remote contains no credential, no patient or examination identifier, no
clinical image and no dataset manifest. Aggregate result tables and aggregate plots remain, and
they contain no patient data. Source code references the author's local home directory in script
paths; that reveals a username and nothing clinical.

## Statement on the scientific content

**The scientific sample was not changed.** The 120 selected study identifiers, their cohort
membership, their group assignments, the random seed, the source, class and geometry allocation,
the primary endpoints, the statistical analysis plan and the annotation protocol are all exactly as
they were.

**The frozen blinding-key content was not changed.** The key was moved out of the repository
without being edited; its content hash is unchanged and is recorded in the expert-validation
manifest. The hash is computed over the key's content and is the same before and after this
remediation.

No model, weight, prediction, split or annotation rule was modified.
