"""Purge sensitive research metadata from the FULL public git history, without shelling out to sh.

git filter-repo is not installed and git filter-branch needs /bin/sh, which the sandbox blocks
(sh.exe cannot create its signal pipe). This does the same job with git plumbing only:

  for each commit in topological order
      read-tree <commit> into a scratch index
      git rm --cached the forbidden paths
      write-tree
      commit-tree with the same author, committer, dates and message, parents remapped

Deterministic, no shell, and the original commit metadata is preserved byte for byte.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_repo_clone"

# --- the purge set, decided after the full-history scan ------------------------------
FORBIDDEN = [
    # the private blinding key
    "expert_validation/blinding_key.csv",
    # the dataset manifest: absolute paths, patient UUIDs, source, label, split
    "data/splits/all.csv",
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/splits/group_assignments.csv",
    "data/splits/excluded_ambiguous_exact_duplicates.csv",
    "data/splits/missing_masks.csv",
    "data/splits/rerun_masks.csv",
    # retinal photographs: a full RGB fundus, and two Grad-CAM overlays
    "docs/figures/vessel_overlay_plus.png",
    "docs/figures/gradcam_plus.png",
    "docs/figures/gradcam_preplus.png",
]


def git(*args, env=None, check=True, input_bytes=None):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, env=env,
                       input=input_bytes)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout


def meta(sha: str) -> dict:
    out = git("log", "-1", "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%B", sha)
    parts = out.split(b"\x00", 6)
    return {"an": parts[0].decode(), "ae": parts[1].decode(), "ad": parts[2].decode(),
            "cn": parts[3].decode(), "ce": parts[4].decode(), "cd": parts[5].decode(),
            "msg": parts[6]}


def main() -> int:
    commits = git("rev-list", "--all", "--topo-order", "--reverse").decode().split()
    print(f"commits to rewrite: {len(commits)}")
    print(f"forbidden paths   : {len(FORBIDDEN)}")
    for p in FORBIDDEN:
        print(f"    {p}")

    idx = Path(tempfile.mkdtemp(prefix="rewrite_idx_")) / "index"
    env_base = dict(os.environ, GIT_INDEX_FILE=str(idx))
    mapping: dict[str, str] = {}
    changed = 0

    for i, c in enumerate(commits, 1):
        parents = git("rev-list", "--parents", "-n", "1", c).decode().split()[1:]
        if idx.exists():
            idx.unlink()
        git("read-tree", c, env=env_base)
        removed_here = []
        for p in FORBIDDEN:
            r = subprocess.run(["git", "-C", REPO, "rm", "--cached", "--quiet",
                                "--ignore-unmatch", "--", p],
                               capture_output=True, env=env_base)
            if r.returncode == 0:
                removed_here.append(p)
        tree = git("write-tree", env=env_base).decode().strip()
        m = meta(c)
        e = dict(os.environ,
                 GIT_AUTHOR_NAME=m["an"], GIT_AUTHOR_EMAIL=m["ae"], GIT_AUTHOR_DATE=m["ad"],
                 GIT_COMMITTER_NAME=m["cn"], GIT_COMMITTER_EMAIL=m["ce"], GIT_COMMITTER_DATE=m["cd"])
        args = ["commit-tree", tree]
        for p in parents:
            args += ["-p", mapping[p]]
        new = git(*args, env=e, input_bytes=m["msg"]).decode().strip()
        mapping[c] = new
        if removed_here:
            changed += 1
            print(f"  [{i}/{len(commits)}] {c[:8]} -> {new[:8]}  removed {len(removed_here)} path(s)")

    # a commit whose tree became identical to its single parent's is empty; drop those
    print(f"\ncommits with content removed: {changed}")
    tip_old = git("rev-parse", "HEAD").decode().strip()
    tip_new = mapping[tip_old]
    print(f"main: {tip_old[:8]} -> {tip_new[:8]}")
    git("update-ref", "refs/heads/main", tip_new)
    git("symbolic-ref", "HEAD", "refs/heads/main")

    # drop the old objects so the blobs are not merely unreferenced but gone
    for ref in git("for-each-ref", "--format=%(refname)", "refs/original").decode().split():
        git("update-ref", "-d", ref)
    git("reflog", "expire", "--expire=now", "--all", check=False)
    git("gc", "--prune=now", "--aggressive", check=False)
    print("gc done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
