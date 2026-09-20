#!/usr/bin/env python
"""Snapshot or compare pixel digests of the packaged images.

  python package_pixels.py snapshot <dir> <json>
  python package_pixels.py compare  <dir> <json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def digests(d: Path) -> dict[str, str]:
    out = {}
    for p in sorted(d.glob("*.png")):
        with Image.open(p) as im:
            im.load()
            a = np.array(im.convert("RGB"))
        out[p.stem] = hashlib.sha256(a.tobytes()).hexdigest()
    return out


def main() -> int:
    mode, folder, json_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    cur = digests(folder)
    if mode == "snapshot":
        json.dump(cur, open(json_path, "w"), indent=1, sort_keys=True)
        print(f"snapshot: {len(cur)} images -> {json_path}")
        return 0
    prev = json.load(open(json_path))
    same = [k for k in cur if prev.get(k) == cur[k]]
    diff = [k for k in cur if k in prev and prev[k] != cur[k]]
    missing = [k for k in prev if k not in cur]
    extra = [k for k in cur if k not in prev]
    print(f"compared {len(cur)} current vs {len(prev)} previous")
    print(f"  identical      : {len(same)}")
    print(f"  CHANGED pixels : {len(diff)} {diff[:5]}")
    print(f"  missing now    : {len(missing)} {missing[:5]}")
    print(f"  new            : {len(extra)} {extra[:5]}")
    ok = not diff and not missing
    print("PIXEL_IDENTITY_PRESERVED" if ok else "PIXEL_IDENTITY_CHANGED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
