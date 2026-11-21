#!/usr/bin/env python3
"""Start Method 1 random-rater YOLO11s@1024. Default is dry. Never submits.

Dest is weights/yolo11s_seg_rater only. Refuses closed YOLO dests and a busy MPS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR, YOLO_RATER_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
WATCH = ROOT / "scripts" / "watch_rater_then_score.py"
RECEIPT = ARTIFACTS / "start_rater_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_rater"
CLOSED = {
    "yolo11s_seg",
    "yolo11m_seg_1280",
    "yolo11s_seg_all",
    "yolo11s_seg_phys",
    "yolo11s_seg_shift",
    "yolo11s_seg_tiles",
    "yolo11s_seg_tiles_cm",
    "yolo11s_seg_faint",
    "yolo11s_seg_residual",
    "unet_resnet34",
    "unet_semantic",
    "unet_patch_dhebe",
    "unet_3ch_disk",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "start_flag": args.start,
        "auto_start": False,
        "family": "yolo11s_seg_rater",
        "data": str(YOLO_RATER_DIR / "data.yaml"),
        "dest": str(DEST),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "mps_owner": owner,
        "slot2_stays": 56148889,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.name in CLOSED:
        out["refused"] = f"dest {DEST.name} is a closed overwrite"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST / 'weights' / 'best.pt'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if owner:
        out["refused"] = f"Refusing rater start: {owner} owns MPS."
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(0 if not args.start else 1)
    if not (YOLO_RATER_DIR / "data.yaml").exists():
        out["refused"] = f"missing {YOLO_RATER_DIR / 'data.yaml'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
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
    run_log = ARTIFACTS / "rater_family.out"
    watch_log = ARTIFACTS / "rater_watch.out"
    watch = subprocess.Popen(
        [str(PYTHON), str(WATCH)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=watch_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            "yolo11s_seg_rater",
            "--data",
            str(YOLO_RATER_DIR / "data.yaml"),
            "--model",
            "yolo11s-seg.pt",
            "--imgsz",
            "1024",
            "--batch",
            "8",
            "--epochs",
            "36",
            "--patience",
            "12",
            "--copy-paste",
            "0.15",
            "--mixup",
            "0.08",
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
            "train_pid": run.pid,
            "watch_pid": watch.pid,
            "train_log": str(run_log.relative_to(ROOT)),
            "watch_log": str(watch_log.relative_to(ROOT)),
        }
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
