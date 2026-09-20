# Request to GitHub Support — purge unreachable objects

Send this from the account that owns the repository, through
<https://support.github.com/contact> (choose **Repository → Sensitive data / remove data**).

---

**Subject:** Request to purge unreachable objects after a history rewrite — sensitive clinical
research data

**Repository:** `nikimahdian/vascular-biomarker-guided-rop-classification` (public)

**Request**

I rewrote the history of this repository to remove sensitive research metadata that had been
committed. The rewritten history is pushed and the branch no longer contains those files, but the
pre-rewrite commits are still retrievable from GitHub by their commit hashes, and I have confirmed
this by fetching them after the rewrite.

Please garbage-collect the unreachable objects for this repository and purge any cached views, so
that the pre-rewrite commits can no longer be fetched.

**What was removed**

- A private study key mapping blinded study identifiers to source images, with acquisition source,
  diagnostic label, split assignment and local file paths.
- Dataset manifest tables listing every study image together with source, label, split and the
  patient or examination grouping identifiers.
- Three documentation figures containing identifiable fundus photographs.

These link individual clinical images of premature infants to diagnostic labels and to patient or
examination identifiers.

**Pre-rewrite commit hashes still reachable on the server**

```
36b908b29873ac7ec62bc1db5d2502b206a5a20e
91309217e53c65a045316f8f6b5827b4077474b4
a69bdab97f38afe95e38646e67f85f6b4f5ebd6b
fb0794666fa8c36143bb6b614000dfefff99363e
764c14c28ff6a5875955c0533bf94c43fd1a4785
db99c38e93572af928e6a96e547b98f533d5c9de
89bb6a845e987fce5f22fd1c0407e1a69bcf1134
838ed8c44f8f72f529338fac8408566650704d0a
e9e59c30ae9e5829ccd4da72f3ab1ef1b788a2ef
d0c9c484c9da3449652951218aa7bf4aa93f4f8d
dc090e8772e15c5d9d1322e1ed0a8dacfe8b3ffa
```

**Current clean tip:** `6750f62`

The files are already gone from the branch and from any fresh clone. What remains is the
server-side copy of the old objects.
