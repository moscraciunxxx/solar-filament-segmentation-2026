#!/usr/bin/env python3
"""Train YOLO11m-seg at 2048 on the parent erode-gt split. Default is dry.

Same yaml, augmentation, optimizer, and seed as yolo11s_seg_erode_gt_2048.
Batch is 2: the small parent already used about 40 GB at batch 4 on a 48 GB
machine, and YOLO11m is twice the width. Conf 0.35 stays locked for the
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
RECEIPT = ARTIFACTS / "start_yolo11m_seg_erode_gt_2048_now.json"
DEST = WEIGHTS_DIR / "yolo11m_seg_erode_gt_2048"
MODEL = ROOT / "yolo11m-seg.pt"
IMGSZ = 2048
BATCH = 2


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
        "batch": BATCH,
        "epochs": 40,
        "patience": 12,
        "seed": 20260910,
        "locked_conf_a_priori": 0.35,
        "batch_why": (
            "Parent yolo11s at imgsz 2048 used batch 4 and about 40 GB. "
            "YOLO11m doubles width, so batch 2 matches that footprint. "
            "Batch 4 is refused on this 48 GB machine."
        ),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "test_labels_used": False,
        "mps_owner": owner,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
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
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    run_log = ARTIFACTS / "yolo11m_seg_erode_gt_2048.out"
    run = subprocess.Popen(
        [
            str(PYTHON),
            "-u",
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
            str(BATCH),
            "--epochs",
            "40",
            "--patience",
            "12",
            "--seed",
            "20260910",
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
