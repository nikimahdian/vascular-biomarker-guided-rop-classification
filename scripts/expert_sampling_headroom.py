"""Can the proposed expert sampling design actually be executed on this dataset?

The design requires at most one image per patient/exam group. That rule interacts badly with the
geometry x label imbalance, so the available head-room has to be counted before any allocation is
promised. This script counts it.
"""
import numpy as np
import pandas as pd

B = r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\_work"
d = pd.read_csv(f"{B}\\v2b.csv")[["image_path", "label", "source", "split", "group_id"]]
disc = pd.read_csv(f"{B}\\disc_predictions_all.csv")[["image_path", "w", "h", "peak_prob"]]
d = d.merge(disc, on="image_path", how="left")
d["min_side"] = np.minimum(d.w, d.h).astype(int)

print("total images:", len(d))
print("\n=== groups available per (geometry, label), ALL splits ===")
t = d.groupby(["min_side", "label"]).agg(images=("image_path", "size"),
                                         groups=("group_id", "nunique")).reset_index()
piv = t.pivot(index="min_side", columns="label", values=["images", "groups"])
print(piv.fillna(0).astype(int).to_string())

print("\n=== the same, restricted to val+test (images the segmentation model never trained on) ===")
pool = d[d.split.isin(["val", "test"])]
print("pool size:", len(pool))
t2 = pool.groupby(["min_side", "label"]).agg(images=("image_path", "size"),
                                             groups=("group_id", "nunique")).reset_index()
piv2 = t2.pivot(index="min_side", columns="label", values=["images", "groups"])
print(piv2.fillna(0).astype(int).to_string())

print("\n=== SOURCE x label (all splits) ===")
print(pd.crosstab([d.source, d.min_side], d.label).to_string())

print("\n=== HEAD-ROOM under 'at most one image per group' ===")
for nm, frame in (("all splits", d), ("val+test only", pool)):
    g = frame.groupby("label")["group_id"].nunique()
    print(f"  {nm:14s} max Plus images = {int(g.get(2, 0)):3d}   "
          f"max Pre-Plus = {int(g.get(1, 0)):3d}   max Normal = {int(g.get(0, 0)):3d}")

print("\n=== the binding constraint: Plus groups per geometry ===")
for nm, frame in (("all splits", d), ("val+test only", pool)):
    p = frame[frame.label == 2].groupby("min_side")["group_id"].nunique()
    print(f"  {nm:14s} " + "  ".join(f"{int(k)}->{int(v)}" for k, v in p.items()))

print("\n=== consequence ===")
p = pool[pool.label == 2].groupby("min_side")["group_id"].nunique()
print(f"  Restricting to val+test (the honest pool for a measurement study) allows at most")
print(f"  {int(p.sum())} Plus images in total, because 480 has zero Plus groups there and")
print(f"  1240 has only {int(p.get(1240, 0))}. A 90-image set with 30 Plus is reachable, but it")
print(f"  will be dominated by geometries 960 and 1200, and geometry 480 will carry no Plus at all.")
