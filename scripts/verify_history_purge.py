"""Post-purge verification: scan EVERY object in the object database, reachable or not."""
from __future__ import annotations

import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_repo_clone"

PATTERNS = {
    "credential": re.compile(rb"ghp_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}"
                             rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "absolute path": re.compile(rb"/Users/[A-Za-z0-9._-]+/|C:\\\\Users\\\\"),
    "patient UUID": re.compile(rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                               rb"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
    "class-folder path": re.compile(rb"Plus/No Plus|Plus/Pre Plus|Plus/Plus"),
    "image bytes": re.compile(rb"\x89PNG\r\n\x1a\n|\xff\xd8\xff"),
}
# a UUID inside the code is only a leak when it sits in a data-like context
DATA_CTX = re.compile(rb"image_path|group_id|patient_id|exam_id|\.jpg|\.png|data/raw")


def main() -> int:
    listing = subprocess.run(["git", "-C", REPO, "cat-file", "--batch-all-objects",
                              "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                             capture_output=True).stdout.decode().splitlines()
    blobs = [l.split() for l in listing if l.endswith("blob") or " blob " in l]
    blobs = [(p[0], int(p[2])) for p in blobs]
    print(f"blobs in object database (reachable AND unreachable): {len(blobs)}")

    raw = subprocess.run(["git", "-C", REPO, "cat-file", "--batch"],
                         input="\n".join(s for s, _ in blobs).encode(),
                         capture_output=True).stdout
    pos, hits, n = 0, {k: [] for k in PATTERNS}, 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        if nl < 0:
            break
        header = raw[pos:nl].decode("ascii", "replace").split()
        if len(header) != 3:
            break
        sha, otype, size = header[0], header[1], int(header[2])
        body = raw[nl + 1: nl + 1 + size]
        pos = nl + 1 + size + 1
        if otype != "blob":
            continue
        n += 1
        for cat, pat in PATTERNS.items():
            for m in pat.finditer(body):
                if cat == "patient UUID":
                    ctx = body[max(0, m.start() - 200): m.start() + 200]
                    if not DATA_CTX.search(ctx):
                        continue
                hits[cat].append((sha[:10], size, body[max(0, m.start() - 30):m.start() + 40]))
                break

    print(f"blobs scanned: {n}\n")
    ok = True
    for cat, items in hits.items():
        print(f"--- {cat}: {len(items)} ---")
        for sha, size, sample in items[:10]:
            print(f"    {sha}  {size:>8}B  {sample!r}")
        if cat in ("credential", "patient UUID", "class-folder path", "image bytes") and items:
            ok = False
    print()
    print("CLEAN — no credential, patient UUID, class-folder path or embedded image in the "
          "object database" if ok else "NOT CLEAN — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
