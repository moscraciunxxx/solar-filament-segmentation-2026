#!/usr/bin/env python3
"""Train CondInst on medoid outlines of contest train JPEGs.

Fold 0 is the report split and is not used to pick the checkpoint.
Does not read test images. Does not submit. Does not resume the paused YOLO run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, rasterize_polygons
from filament.condinst import STRIDE, CondInst, condinst_loss
from filament.paths import ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR

IMGSZ = 1024
EPOCHS = 8
FAMILY = "condinst_r18_1024_bodypos"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _own_pid_running() -> None:
    import subprocess

    proc = subprocess.run(
        ["pgrep", "-af", "train_yolo.py|train_unet|ultralytics|train_condinst.py"],
        capture_output=True,
        text=True,
    )
    me = os.getpid()
    for line in (proc.stdout or "").splitlines():
        parts = line.split(maxsplit=1)
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == me or "pgrep" in line:
            continue
        if "train_condinst.py" in line or "train_yolo.py" in line or "train_unet" in line or "ultralytics" in line:
            raise SystemExit(f"refusing to train while pid {pid} owns the machine: {line[:180]}")


class MedoidSet(Dataset):
    def __init__(self, files: list[str], views, by_file) -> None:
        self.files = files
        self.views = views
        self.by_file = by_file
        self.medoid_id = {
            name: medoid_view_id(views, by_file[name]) for name in files
        }

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        name = self.files[index]
        path = TRAIN_IMG_DIR / name
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        small = cv2.resize(gray, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
        rgb = np.stack([small, small, small], axis=-1).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        image = torch.from_numpy(rgb.transpose(2, 0, 1))
        view = self.views[self.medoid_id[name]]
        instances = []
        scale = IMGSZ / float(IMAGE_SIZE)
        for filament in view.filaments:
            full = rasterize_polygons([filament.polygon])
            mask = cv2.resize(full, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
            if int(mask.sum()) < 4:
                continue
            x, y, w, h = filament.bbox
            box = torch.tensor(
                [x * scale, y * scale, (x + w) * scale, (y + h) * scale],
                dtype=torch.float32,
            )
            instances.append({"mask": torch.from_numpy(mask.astype(np.float32)), "box": box})
        return image, instances


@torch.no_grad()
def _train_peak(model, files, views, by_file, device) -> float:
    """Max decode score on six train images. Fold 0 is not in `files`."""
    model.eval()
    span = max(len(files) - 1, 1)
    picks = sorted({files[int(round(k * span / 5))] for k in range(6)})
    peak = 0.0
    subset = MedoidSet(picks, views, by_file)
    for index in range(len(subset)):
        image, _instances = subset[index]
        outputs = model(image.unsqueeze(0).to(device))
        score = outputs["logit"][0, 0].sigmoid() * outputs["centerness"][0, 0].sigmoid()
        peak = max(peak, float(score.max()))
    model.train()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=0, help="Debug cap on train images. 0 means all.")
    args = parser.parse_args()
    _own_pid_running()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    device = torch.device("mps")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(name for name, fold in fold_of.items() if fold != 0)
    if args.limit:
        files = files[: args.limit]
    if len([name for name, fold in fold_of.items() if fold == 0]) != 142:
        raise SystemExit("fold 0 is not the locked 142-image split")
    loader = DataLoader(
        MedoidSet(files, views, by_file),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch[0],
    )
    model = CondInst(backbone="resnet18", pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    dest = WEIGHTS_DIR / FAMILY
    dest.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = []
        masks = []
        for step, (image, instances) in enumerate(loader, start=1):
            image = image.to(device)
            moved = [
                {"mask": inst["mask"].to(device), "box": inst["box"].to(device)}
                for inst in instances
            ]
            opt.zero_grad(set_to_none=True)
            outputs = model(image.unsqueeze(0))
            unbatched = {key: value[0] for key, value in outputs.items()}
            losses = condinst_loss(unbatched, moved, stride=STRIDE)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals.append(float(losses["total"].detach()))
            masks.append(float(losses["mask"].detach()))
            if step == 1 or step % 50 == 0:
                print(
                    f"epoch {epoch + 1} step {step}/{len(loader)} "
                    f"loss {totals[-1]:.4f} mask {masks[-1]:.4f} pos {int(losses['n_pos'])}",
                    flush=True,
                )
        mean_loss = float(np.mean(totals)) if totals else 0.0
        peak = _train_peak(model, files, views, by_file, device)
        row = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "mask": float(np.mean(masks)) if masks else 0.0,
            "train_peak": peak,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "epoch": epoch + 1, "loss": mean_loss, "imgsz": IMGSZ}
        torch.save(state, dest / "last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(state, dest / "best.pt")
    meta = {
        "family": FAMILY,
        "imgsz": IMGSZ,
        "epochs": args.epochs,
        "n_train": len(files),
        "seed": SEED,
        "fold0_held_out": True,
        "labels": "medoid polygons, not erode-gt and not a merge",
        "hypothesis": "class and centerness targets cover the whole filament body, not the centroid cluster",
        "fold0_scored": False,
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "history": history,
        "best_loss": best_loss,
        "best_train_peak": max((row["train_peak"] for row in history), default=0.0),
    }
    (dest / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (ARTIFACTS / f"{FAMILY}_train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"best_loss": best_loss, "weights": str(dest / "best.pt")}), flush=True)


if __name__ == "__main__":
    main()
