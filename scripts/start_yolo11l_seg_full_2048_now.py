#!/usr/bin/env python3
"""Train YOLO11l-seg at native 2048 on full medoid labels. Default is dry.

Dest is weights/yolo11l_seg_full_2048 only. The small full-medoid run and
every eroded dest stay in place. Batch is 1. Conf 0.35 is locked for the
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

from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, WEIGHTS_DIR, YOLO_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
RECEIPT = ARTIFACTS / "start_yolo11l_seg_full_2048_now.json"
DEST = WEIGHTS_DIR / "yolo11l_seg_full_2048"
MODEL = ROOT / "yolo11l-seg.pt"
DATA = YOLO_DIR / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner()
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started": False,
        "family": DEST.name,
        "labels": "full_medoid_yolo_seg",
        "data": str(DATA),
        "dest": str(DEST),
        "model": str(MODEL),
        "imgsz": IMAGE_SIZE,
        "batch": 1,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": 0.35,
        "why": (
            "Eroded medoid masks score only PQ 0.644 on fold 0, forecast "
            "0.575. The full medoid oracle scores 0.794. YOLO11s full at "
            "2048 scored 0.434. The large model has not been trained on "
            "full outlines at 2048."
        ),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "test_labels_used": False,
        "mps_owner": owner,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if not MODEL.is_file() or not DATA.is_file():
        out["refused"] = f"missing model or data: {MODEL} {DATA}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if owner:
        out["refused"] = f"{owner} owns MPS"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        raise SystemExit(1)
    images = sorted((YOLO_DIR / "images" / "train").glob("*.jpeg"))
    if not images:
        raise SystemExit(f"no train images in {YOLO_DIR}")
    from PIL import Image

    long_side = max(Image.open(images[0]).size)
    out["sample_image_long_side"] = long_side
    if long_side < IMAGE_SIZE:
        raise SystemExit(f"train images are {long_side}px")
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11l_seg_full_2048_family.out"
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            DEST.name,
            "--data",
            str(DATA),
            "--model",
            str(MODEL),
            "--imgsz",
            str(IMAGE_SIZE),
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
    out.update({"started": True, "train_pid": run.pid, "train_log": str(run_log.relative_to(ROOT))})
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
