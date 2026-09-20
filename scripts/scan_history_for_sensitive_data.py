"""Section 4: scan the COMPLETE reachable git history for sensitive research metadata.

Walks every blob reachable from any ref and reports hits by category, with the commits and paths
each blob belongs to. Read-only.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_repo_clone"

CATEGORIES = {
    "credential/token": re.compile(
        rb"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}"
        rb"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
        rb"|\b(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+", re.I),
    "absolute local path": re.compile(rb"/Users/[A-Za-z0-9._-]+/|C:\\\\Users\\\\", re.I),
    "patient/exam UUID": re.compile(
        rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
    "leaky column name": re.compile(
        rb"image_path_original|auto_mask_path_original|blinding_key|patient_id|exam_id", re.I),
    "class-folder path": re.compile(rb"Plus/No Plus|Plus/Pre Plus|Plus/Plus", re.I),
    "source name": re.compile(rb"farabi|farfum_rop|Plus_dataset_main|Normal_comp", re.I),
    "embedding image bytes": re.compile(rb"\x89PNG\r\n\x1a\n|\xff\xd8\xff\xe0|\xff\xd8\xff\xe1"),
}

BINARY_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".zip", ".pt", ".pth",
              ".npz", ".joblib", ".pkl", ".gz", ".pdf")


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True).stdout


def main() -> int:
    print("=== refs ===")
    print(git("for-each-ref", "--format=%(refname) %(objectname)").decode().strip())
    print(f"\ncommits reachable: {git('rev-list', '--all', '--count').decode().strip()}")
    print(f"objects reachable: {len(git('rev-list', '--objects', '--all').decode().splitlines())}")

    # path -> blobs, and blob -> paths, for context
    blob_paths: dict[str, set[str]] = defaultdict(set)
    for line in git("rev-list", "--objects", "--all").decode("utf-8", "replace").splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            blob_paths[parts[0]].add(parts[1])

    # commits touching each path
    path_commits: dict[str, set[str]] = defaultdict(set)
    log = git("log", "--all", "--pretty=format:@@%h", "--name-only").decode("utf-8", "replace")
    cur = "?"
    for line in log.splitlines():
        if line.startswith("@@"):
            cur = line[2:]
        elif line.strip():
            path_commits[line.strip()].add(cur)

    print("\n=== every path that has ever existed ===")
    for p in sorted(path_commits):
        print(f"  {p}")

    print("\n=== binary artefacts ever committed ===")
    bins = [p for p in sorted(path_commits) if p.lower().endswith(BINARY_EXT)]
    for p in bins:
        print(f"  {p}   commits={sorted(path_commits[p])}")
    if not bins:
        print("  (none)")

    print("\n=== scanning every blob for sensitive content ===")
    shas = list(blob_paths.keys())
    raw = subprocess.run(["git", "-C", REPO, "cat-file", "--batch"],
                         input="\n".join(shas).encode(), capture_output=True)
    out = raw.stdout
    hits: dict[str, list[dict]] = defaultdict(list)
    n_blob = 0
    pos = 0
    while pos < len(out):
        nl = out.find(b"\n", pos)
        if nl < 0:
            break
        header = out[pos:nl].decode("ascii", "replace").split()
        if len(header) != 3:
            break
        sha, otype, size = header[0], header[1], int(header[2])
        body = out[nl + 1: nl + 1 + size]
        pos = nl + 1 + size + 1
        if otype != "blob":
            continue
        n_blob += 1
        for cat, pat in CATEGORIES.items():
            m = pat.search(body)
            if not m:
                continue
            paths = sorted(blob_paths.get(sha, {"<no-path>"}))
            if cat == "source name":
                # a mention inside code or documentation is not by itself a leak; only flag when
                # it appears together with a path-like or patient-like context
                ctx = body[max(0, m.start() - 120): m.start() + 120]
                if not re.search(rb"/|data/raw|image_path|\.jpg|\.png", ctx, re.I):
                    continue
            sample = body[max(0, m.start() - 40): m.start() + 60]
            hits[cat].append({
                "sha": sha[:10], "bytes": size, "paths": paths,
                "commits": sorted({c for p in paths for c in path_commits.get(p, set())}),
                "sample": sample.decode("utf-8", "replace").replace("\n", " ")[:110],
            })
    print(f"blobs scanned: {n_blob}")

    print("\n=== hits by category ===")
    if not hits:
        print("  none")
    for cat, items in hits.items():
        print(f"\n--- {cat}: {len(items)} blob(s) ---")
        for it in items:
            print(f"    {it['sha']}  {it['bytes']:>7}B  paths={it['paths']}")
            print(f"        commits={it['commits']}")
            print(f"        sample={it['sample']!r}")

    print("\n=== which commits contain a file named blinding_key.csv ===")
    print(git("log", "--all", "--oneline", "--follow", "--", "expert_validation/blinding_key.csv")
          .decode().strip() or "  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
