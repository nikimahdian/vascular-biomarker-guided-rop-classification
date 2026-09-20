# Data governance notification — unauthorised public exposure of research metadata

**Status:** for notification and assessment. Drafted 2026-09-20.
**Owner of the repository and dataset:** the study author.
**Recipients:** the institutional data protection / research governance officer, and the supervising
academic responsible for the dataset.

This document deliberately contains no patient identifiers, no file paths, no dataset contents and
no credentials. It describes categories of data and the actions taken.

---

## 1. What happened

A research code repository was published publicly on GitHub. During preparation and subsequent work
on the project, several files that should have been kept private were committed to it and were
therefore publicly downloadable.

The repository was created and populated between 2026-09-11 and 2026-09-20. The exposure therefore
lasted from the initial public release until the remediation described in section 4, with the
longest-exposed files online for the whole period.

## 2. Categories of data exposed

Four categories, in order of sensitivity:

1. **A private study key** linking blinded study identifiers to source images, and for each image:
   the acquisition source, the diagnostic label, the train / validation / test assignment, and a
   local file path.
2. **Dataset manifest tables**, listing every image in the study together with a local file path,
   the acquisition source, the diagnostic label, the split assignment, and the identifier columns
   used to group images belonging to the same patient or examination.
3. **Three documentation figures** that contained identifiable retinal fundus photographs rather
   than aggregate plots.
4. **A blinding manifest** that revealed the split of the image set into two study cohorts.

The local file paths in categories 1 and 2 contained per-image identifiers that are derived from,
or identical to, the identifier used to group images by patient or examination. In one source the
grouping identifier is not resolvable to a named individual; in another the grouping identifier is
an official patient identifier from the source dataset.

## 3. Who could have accessed it, and what the risk is

The repository was publicly readable, so the material could have been downloaded by anyone, indexed
by code search and archival services, and mirrored. No credentials, keys, passwords or
authentication material were exposed.

The material does **not** contain names, dates of birth, addresses or contact details. Its
identifying power comes from the combination of a clinical image reference with a diagnostic label
and a patient or examination identifier. Taken together with access to the imaging archive, that
combination would allow re-identification of the infants concerned and would disclose their
diagnosis. Published alongside a public repository, it is a re-identification resource for a
clinical dataset of premature infants.

The images are retinal fundus photographs of premature infants. Those in category 3 are directly
visible clinical images; they carry no burned-in identifiers, but the anatomy is visible.

## 4. What has been done

- The files were removed from the repository **and from every commit in its history**, not merely
  from the current revision.
- The rewritten history was published; a fresh clone of the repository now contains none of the
  material, and the object database was rescanned to confirm that no identifier, clinical image or
  manifest remains in any object a clone receives.
- The removed files are retained privately by the study under access control, outside the public
  repository.
- Repository ignore rules were added so these paths cannot be re-added by accident.
- A public remediation record was committed to the repository, stating the category of file
  removed, why it should not have been public, the date, and that the scientific sample and the
  frozen key content were not changed.
- A technique for scanning the full history for this class of data was added to the repository, so
  the check is repeatable.

## 5. What remains outstanding — and why it cannot be closed from here

**The hosting provider still serves the pre-rewrite commits.** Rewriting history removes the old
commits from the branch but does not remove the objects from GitHub. It was verified after the
rewrite that every pre-rewrite commit could still be retrieved from GitHub by its commit hash, and
that the manifest, the private key and a retinal image figure were each recoverable that way.

Until GitHub Support runs a garbage collection on the repository, the material remains obtainable
by anyone who knows those commit hashes, which were public.

A request to GitHub Support is prepared at `docs/GITHUB_SUPPORT_PURGE_REQUEST.md`, listing the
affected commit hashes. **Sending it requires the repository owner's GitHub account**, so it cannot
be completed from within the project.

## 6. What is requested of governance

1. **Record this as a personal-data exposure** in the study's incident log, with the date range
   above, and advise whether it meets your threshold for a reportable incident under the applicable
   framework.
2. **Advise on notification.** The images are those of minors. Please confirm whether the source
   dataset's data-use agreement requires notification of the dataset provider, and whether any
   further notification is required.
3. **Confirm the retention position** for the privately held copies of the removed files: they are
   still needed to reproduce the study's blinding, so they have not been destroyed.
4. **Confirm whether the repository should become private.** It currently remains public, and the
   project would like it to stay public so the method can be reproduced. If governance prefers, it
   can be made private instead.
5. **Request confirmation from GitHub Support** once actioned, and keep the confirmation on file.

## 7. What is not affected

The security finding does not invalidate any scientific result, and nothing in the study design was
changed in response to it.

- The scientific sample is unchanged: the same study identifiers, cohort membership, group
  assignments, random seed, and source, class and geometry allocation.
- The frozen key content was not edited; it was moved out of the repository and its content hash is
  unchanged.
- No model, weight, prediction, split, endpoint or annotation rule was modified.
- The prepared clinician pilot package was rebuilt from the same frozen cohort and verified
  pixel-identical to the previous build; its checksum is recorded and it contains no patient
  identifier, source, label or model output.

## 8. Sequence once clearance is confirmed

```
GitHub Support confirms the purge
        ↓
Send the pilot package to the expert annotators
        ↓
Independent annotation by two graders
        ↓
Schema and leakage gate on the returned masks
        ↓
Pilot quality control and inter-grader agreement
        ↓
At most one protocol revision, if the pilot shows an ambiguity
        ↓
Protocol v1.0 frozen
        ↓
Final cohort annotation
```

Until the purge is confirmed, the pilot package should not be sent, on the principle that no new
work should depend on data whose exposure is still open.
