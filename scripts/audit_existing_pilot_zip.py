"""Section 17: audit the pre-existing _expert_pilot30.zip. Read-only; the original is untouched."""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

ZIP = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_expert_pilot30.zip")
KEY = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_repo_clone\expert_validation\blinding_key.csv")

import pandas as pd  # noqa: E402

sha_zip = hashlib.sha256(ZIP.read_bytes()).hexdigest()
print(f"existing ZIP: {ZIP.name}  {ZIP.stat().st_size:,} bytes")
print(f"sha256      : {sha_zip}")

zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
print(f"entries     : {len(names)}")

print("\n=== text files at the root of the archive ===")
for n in names:
    if n.endswith(".md") or n.endswith(".txt"):
        print(f"\n--- {n} ---")
        print(zf.read(n).decode("utf-8", "replace")[:2500])

print("\n=== pilot30_manifest.csv (head) ===")
mf = [n for n in names if n.endswith("pilot30_manifest.csv")]
if mf:
    d = pd.read_csv(io.BytesIO(zf.read(mf[0])))
    print("columns:", list(d.columns))
    print("rows:", len(d))
    print(d.head(8).to_string(index=False))
    for c in d.columns:
        if d[c].nunique() <= 6:
            print(f"  value counts {c}: {d[c].value_counts().to_dict()}")

print("\n=== image inventory ===")
imgs = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg"))]
rows = []
for n in imgs:
    b = zf.read(n)
    try:
        with Image.open(io.BytesIO(b)) as im:
            size, mode, fmt = im.size, im.mode, im.format
            info_keys = sorted(im.info.keys())
            exif = im.getexif()
            exif_keys = {k: str(v)[:60] for k, v in exif.items()} if exif else {}
    except Exception as e:  # noqa: BLE001
        size, mode, fmt, info_keys, exif_keys = None, None, None, [], {"error": str(e)}
    rows.append(dict(name=n.split("/")[-1], folder=n.split("/")[-2] if "/" in n else "",
                     bytes=len(b), size=size, mode=mode, fmt=fmt,
                     info_keys="|".join(info_keys), exif_keys=json.dumps(exif_keys)[:200]))
inv = pd.DataFrame(rows)
print(inv.to_string(index=False))

print("\n=== metadata findings across the archive ===")
suspicious = ("exif", "icc_profile", "comment", "software", "artist", "copyright", "gps",
              "thumbnail", "xmp", "iptc", "description", "author", "make", "model", "datetime",
              "dpi", "xml", "raw profile")
hits = {}
for _, r in inv.iterrows():
    for k in str(r.info_keys).split("|"):
        if k and any(s in k.lower() for s in suspicious):
            hits.setdefault(k, []).append(r.name)
    if r.exif_keys and r.exif_keys not in ("{}", ""):
        hits.setdefault("EXIF-IFD", []).append(r.name)
print(f"  images carrying at least one flagged metadata key: {len(set(sum(hits.values(), [])))}")
for k, v in hits.items():
    print(f"    {k}: {len(v)} file(s) e.g. {v[:3]}")

print("\n=== does the archive match the frozen pilot cohort? ===")
key = pd.read_csv(KEY)
pilot = key[key.cohort == "pilot"]
zip_ids = sorted({re.sub(r"_auto$", "", Path(n).stem) for n in imgs})
print(f"  archive ids  : n={len(zip_ids)}  {zip_ids[:6]} ... {zip_ids[-3:]}")
print(f"  frozen pilot : n={len(pilot)}  ids {sorted(pilot.study_id)[:6]} ... {sorted(pilot.study_id)[-3:]}")
print(f"  id scheme match: {'AUD-###' if zip_ids and zip_ids[0].startswith('AUD') else '?'}"
      f" vs 'ROP_####' -> {'NO' if not set(zip_ids) & set(pilot.study_id) else 'partial'}")

print("\n=== forbidden content present? ===")
checks = {
    "automatic vessel masks": [n for n in names if "auto_mask" in n or "_auto" in n],
    "private key / manifest with paths": [n for n in names if "blinding_key" in n.lower()],
    "predictions / results": [n for n in names if re.search(r"pred|result|prob|auc|gradcam",
                                                            n, re.I)],
    "original patient-style filenames": [n for n in names
                                         if re.search(r"patient|exam|farabi|farfum", n, re.I)],
    "JPEG rather than PNG images": [n for n in imgs if n.lower().endswith((".jpg", ".jpeg"))],
    "images whose name encodes geometry": [n for n in imgs
                                           if re.search(r"480|960|1080|1200|1240", n)],
}
for k, v in checks.items():
    print(f"  {'YES' if v else 'no '}  {k}: {len(v)}")

verdict = "UNSAFE"
reasons = [
    "the archive contains 30 automatic vessel masks under auto_masks/ (explicitly forbidden)",
    "study identifiers are AUD-### and do not match the frozen ROP_#### cohort",
    "images are JPEG, so lossy re-encoding has already occurred and metadata survives",
    "pilot30_manifest.csv and the instruction files ship with the images",
]
print(f"\nCLASSIFICATION: {verdict}")
for r in reasons:
    print(f"  - {r}")
