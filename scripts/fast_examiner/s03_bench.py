"""Step 3 (v2) - speed benchmark on M2 Ultra (MPS), num_workers=0."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

ROOT = Path("/Users/moniaz/niki")
OUT = ROOT / "results/fast_examiner"
man = pd.read_csv(OUT / "02_common_farFUM_3fold_manifest.csv")
dev = torch.device("mps")
print("device", dev, "images", len(man), flush=True)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ev = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                         transforms.Normalize(mean, std)])

paths = list(man.image_path[:40])
t0 = time.time()
for p in paths:
    ev(Image.open(p).convert("RGB"))
dt = (time.time() - t0) / len(paths)
print(f"PIL decode+resize384+normalize: {dt * 1000:.0f} ms/image -> "
      f"{dt * 760 / 60:.1f} min for 760 images/epoch", flush=True)

t0 = time.time()
cache = np.stack([np.asarray(Image.open(p).convert("RGB").resize((448, 448), Image.BILINEAR))
                  for p in paths])
print(f"one-off cache build (448px): {(time.time() - t0) / len(paths) * 1000:.0f} ms/image, "
      f"array {cache.shape} {cache.nbytes / 1e6:.0f} MB for 40", flush=True)
tr = transforms.Compose([transforms.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(1.0, 1.0)),
                         transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
                         transforms.ColorJitter(0.10, 0.10, 0.10, 0.0),
                         transforms.ToTensor(), transforms.Normalize(mean, std)])
t0 = time.time()
for i in range(len(paths)):
    tr(Image.fromarray(cache[i]))
dt = (time.time() - t0) / len(paths)
print(f"cached-tensor augment+normalize: {dt * 1000:.0f} ms/image -> "
      f"{dt * 760 / 60:.1f} min per epoch", flush=True)


class Wrap(nn.Module):
    def __init__(self, m, h):
        super().__init__()
        self.m, self.h = m, h

    def forward(self, x):
        return self.h(self.m(x))


for arch in ("efficientnet_b5", "efficientnet_b4"):
    model = timm.create_model(arch, pretrained=True, num_classes=0, global_pool="avg").to(dev)
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.num_features, 3)).to(dev)
    net = Wrap(model, head).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss()
    x = torch.randn(16, 3, 384, 384, device=dev)
    y = torch.zeros(16, dtype=torch.long, device=dev)
    for _ in range(2):
        opt.zero_grad(); lossf(net(x), y).backward(); opt.step()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad(); lossf(net(x), y).backward(); opt.step()
    tr_dt = (time.time() - t0) / 5
    net.eval()
    with torch.no_grad():
        net(x)
        t0 = time.time()
        for _ in range(5):
            net(x)
    inf_dt = (time.time() - t0) / 5
    print(f"{arch}@384 batch16: train {tr_dt:.2f} s/step  infer {inf_dt:.2f} s/step  "
          f"-> epoch(760 train imgs) {tr_dt * 48 / 60:.1f} min", flush=True)
    del model, head, net
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
