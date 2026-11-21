#!/usr/bin/env python3
"""Fine-tune the eroded 2048 small model on one extra copy of faint images.

Dest is weights/yolo11s_seg_erode_gt_2048_faint only. The parent weights
stay in place. Learning rate is 0.0001. Confidence 0.35 is locked for the
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

from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, WEIGHTS_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
RECEIPT = ARTIFACTS / "start_yolo11s_erode_gt_2048_faint_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_faint"
PARENT = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
DATA = FAST_CACHE / "yolo_seg_erode_gt_faint" / "data.yaml"
IMGSZ = 2048


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if IMGSZ != IMAGE_SIZE:
        raise SystemExit(f"imgsz must be {IMAGE_SIZE}")
    owner = live_mps_owner()
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_flag": args.start,
        "started": False,
        "family": DEST.name,
        "labels": "thin_erode_gt_plus_one_faint_copy",
        "data": str(DATA),
        "dest": str(DEST),
        "model": str(PARENT),
        "imgsz": IMGSZ,
        "batch": 2,
        "epochs": 15,
        "patience": 8,
        "lr0": 0.0001,
        "locked_conf_a_priori": 0.35,
        "why": (
            "YOLO11l eroded at 2048 scored PQ 0.41958 and full-medoid s at "
            "2048 scored 0.43375. Both left mean overlap near 0.68. The "
            "missed filaments are faint, so the winning small model is "
            "fine-tuned with one extra copy of those training images."
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
    if not PARENT.is_file():
        out["refused"] = f"missing parent weights {PARENT}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if not DATA.is_file():
        out["refused"] = f"missing {DATA}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if owner:
        out["refused"] = f"{owner} owns MPS"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        raise SystemExit(1)
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_family.out"
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            DEST.name,
            "--data",
            str(DATA),
            "--model",
            str(PARENT),
            "--imgsz",
            str(IMGSZ),
            "--batch",
            "2",
            "--epochs",
            "15",
            "--patience",
            "8",
            "--lr0",
            "0.0001",
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
