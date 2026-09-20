# Security remediation

**Date of remediation:** 2026-09-20
**Scope:** removal of sensitive research metadata from the public repository and from its reachable
git history.

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
  examination identifier, no clinical-image data and no dataset manifest remains in any object,
  reachable or unreachable.

Aggregate result tables and aggregate figures remain in the repository. They contain no patient
data. Source code continues to reference a local absolute path for the author's own machine; that
reveals a username and nothing clinical, and it is left in place because the scripts are written to
run there.

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
