"""Train (or fine-tune) the vessel segmenter on the BV_segmentation dataset.

Only needed if you do NOT have the prior pretrained checkpoint. If a checkpoint
already exists at segmentation.weight_path, you can skip straight to infer_masks.

Usage:
    python -m src.segmentation.train --images data/raw/bv_seg/<images> --masks data/raw/bv_seg/<masks>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.segmentation.dataset import RetinaSegDataset
from src.segmentation.models import DiceBCELoss, build_model, dice_coef
from src.utils.common import ensure_dirs, get_device, load_config, set_seed


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    sc, st = cfg["segmentation"], cfg["seg_train"]
    device = get_device()

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of training fundus images")
    ap.add_argument("--masks", required=True, help="folder of matching vessel masks")
    ap.add_argument("--out", default=str(cfg["paths"]["weights_dir"] / "vessel_seg.pth"))
    args = ap.parse_args()

    ds = RetinaSegDataset(args.images, args.masks, img_size=sc["img_size"])
    n_val = max(1, int(len(ds) * st["val_frac"]))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(cfg["seed"])
    )
    train_dl = DataLoader(train_ds, batch_size=st["batch_size"], shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=st["batch_size"], shuffle=False, num_workers=2)

    model = build_model(
        sc["arch"], sc["encoder"], sc["encoder_weights"], sc["in_channels"], classes=1
    ).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=st["lr"])

    best_dice = -1.0
    for epoch in range(1, st["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, desc=f"train {epoch}/{st['epochs']}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                dices.append(dice_coef(model(x), y))
        val_dice = sum(dices) / len(dices)
        print(f"epoch {epoch}: train_loss={running / len(train_ds):.4f} val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  saved best -> {args.out} (dice={best_dice:.4f})")

    print("[done] best val dice:", best_dice)


if __name__ == "__main__":
    main()
