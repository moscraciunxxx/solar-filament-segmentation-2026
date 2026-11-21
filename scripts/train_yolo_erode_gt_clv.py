#!/usr/bin/env python3
"""Parent yolo11s erode-gt recipe at 2048, with flat-field images only.

Hyperparameters come from the parent args.yaml. data and name are the only
overrides. Does not write into yolo11s_seg_erode_gt_2048. Never submits.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import FAST_CACHE, WEIGHTS_DIR

PARENT_ARGS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "args.yaml"
DATA = FAST_CACHE / "yolo_seg_erode_gt_clv" / "data.yaml"
NAME = "yolo11s_seg_erode_gt_2048_clv"
KEEP = (
    "epochs",
    "imgsz",
    "batch",
    "patience",
    "device",
    "workers",
    "cache",
    "seed",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "close_mosaic",
    "amp",
    "fraction",
    "multi_scale",
    "overlap_mask",
    "mask_ratio",
    "dropout",
    "val",
    "iou",
    "plots",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "dfl",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
    "nbs",
)


def _busy() -> str | None:
    mine = {os.getpid(), os.getppid()}
    try:
        text = subprocess.check_output(
            ["pgrep", "-af", "train_yolo|train_unet|train_condinst|train_soft_clv"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    for line in text.splitlines():
        if "pgrep" in line:
            continue
        pid = int(line.split(maxsplit=1)[0])
        if pid not in mine:
            return line.strip()
    return None


def main() -> None:
    if _busy():
        raise SystemExit(f"MPS busy: {_busy()}")
    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}; run scripts/prepare_yolo_erode_gt_clv.py")
    parent = yaml.safe_load(PARENT_ARGS.read_text())
    if parent.get("epochs") != 40 or parent.get("imgsz") != 2048 or parent.get("seed") != 20260910:
        raise SystemExit("parent args.yaml is not the 40-epoch 2048 seed-20260910 recipe")
    dest = WEIGHTS_DIR / NAME
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    kwargs = {key: parent[key] for key in KEEP}
    kwargs["data"] = str(DATA)
    kwargs["project"] = str(WEIGHTS_DIR)
    kwargs["name"] = NAME
    kwargs["exist_ok"] = False
    changed = []
    for key in KEEP:
        if kwargs[key] != parent[key]:
            changed.append(key)
    if changed or kwargs["data"] == parent["data"] or kwargs["name"] == parent["name"]:
        raise SystemExit(f"recipe drift: {changed}")
    import torch
    from ultralytics import YOLO

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required")
    kwargs["device"] = "mps"
    note = {
        "parent_args": str(PARENT_ARGS),
        "only_changes": ["data", "name"],
        "data": str(DATA),
        "name": NAME,
        "epochs": kwargs["epochs"],
        "imgsz": kwargs["imgsz"],
        "batch": kwargs["batch"],
        "patience": kwargs["patience"],
        "seed": kwargs["seed"],
        "mosaic": kwargs["mosaic"],
        "close_mosaic": kwargs["close_mosaic"],
        "copy_paste": kwargs["copy_paste"],
        "mixup": kwargs["mixup"],
        "optimizer": kwargs["optimizer"],
        "lr0": kwargs["lr0"],
        "kaggle_submit_called": False,
    }
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (WEIGHTS_DIR / f"{NAME}_train_meta.json").write_text(json.dumps(note, indent=2) + "\n")
    print(json.dumps(note), flush=True)
    model = YOLO(parent["model"])
    model.train(**kwargs)


if __name__ == "__main__":
    main()
