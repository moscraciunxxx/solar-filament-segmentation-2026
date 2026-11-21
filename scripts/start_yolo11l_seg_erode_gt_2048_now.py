#!/usr/bin/env python3
"""Train YOLO11l-seg at native 2048 on eroded labels. Default is dry.

Dest is weights/yolo11l_seg_erode_gt_2048 only. The 1280 large run and the
2048 small eroded parent are left in place. Batch is 1 because the small
model already used about 40 GB at batch 4. Conf 0.35 is locked for the
later score. Never submits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, WEIGHTS_DIR, YOLO_ERODE_GT_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
RECEIPT = ARTIFACTS / "start_yolo11l_seg_erode_gt_2048_now.json"
DEST = WEIGHTS_DIR / "yolo11l_seg_erode_gt_2048"
MODEL = ROOT / "yolo11l-seg.pt"
IMGSZ = 2048
CLOSED = {
    "yolo11l_seg_erode_gt",
    "yolo11s_seg_erode_gt_2048",
    "yolo11s_seg_full_2048",
    "yolo11m_seg_erode_gt",
    "yolo11s_seg_erode_gt",
}


def _long_side() -> int:
    images = sorted((YOLO_ERODE_GT_DIR / "images" / "train").glob("*.jpeg"))
    if not images:
        raise SystemExit(f"no train images in {YOLO_ERODE_GT_DIR}")
    from PIL import Image

    width, height = Image.open(images[0]).size
    return max(width, height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if IMGSZ != IMAGE_SIZE:
        raise SystemExit(f"imgsz must be {IMAGE_SIZE}")
    owner = live_mps_owner()
    data = YOLO_ERODE_GT_DIR / "data.yaml"
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_flag": args.start,
        "started": False,
        "family": DEST.name,
        "labels": "thin_erode_gt",
        "data": str(data),
        "dest": str(DEST),
        "model": str(MODEL),
        "imgsz": IMGSZ,
        "batch": 1,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": 0.35,
        "why": (
            "Full-medoid yolo11s at 2048 scored PQ 0.43375, below the eroded "
            "small parent at 0.43872, and mean IoU stayed 0.677. The large "
            "eroded model exists only at 1280, where it scored 0.43084."
        ),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "test_labels_used": False,
        "mps_owner": owner,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.name in CLOSED:
        out["refused"] = f"dest {DEST.name} is closed"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST / 'weights' / 'best.pt'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if not MODEL.is_file():
        out["refused"] = f"missing {MODEL}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if owner:
        out["refused"] = f"{owner} owns MPS"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        raise SystemExit(1)
    if not data.exists():
        out["refused"] = f"missing {data}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    long_side = _long_side()
    out["sample_image_long_side"] = long_side
    if long_side < IMGSZ:
        out["refused"] = f"train images are {long_side}px, refusing to upscale to {IMGSZ}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11l_seg_erode_gt_2048_family.out"
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            DEST.name,
            "--data",
            str(data),
            "--model",
            str(MODEL),
            "--imgsz",
            str(IMGSZ),
            "--batch",
            "1",
            "--epochs",
            "40",
            "--patience",
            "12",
        ],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=run_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    out.update(
        {
            "started": True,
            "dry": False,
            "train_pid": run.pid,
            "train_log": str(run_log.relative_to(ROOT)),
        }
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
