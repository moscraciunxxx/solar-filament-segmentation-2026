#!/usr/bin/env python3
"""Continue the faint fine-tune for 15 more epochs. Default is dry.

Dest is weights/yolo11s_seg_erode_gt_2048_faint_e30 only. The scored
15-epoch checkpoint stays in place. Same data, batch 2, learning rate
0.0001. Confidence 0.35 is locked for the later score. Never submits.
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
RECEIPT = ARTIFACTS / "start_yolo11s_erode_gt_2048_faint_e30_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_faint_e30"
PARENT = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_faint" / "weights" / "best.pt"
DATA = FAST_CACHE / "yolo_seg_erode_gt_faint" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner()
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started": False,
        "family": DEST.name,
        "model": str(PARENT),
        "data": str(DATA),
        "imgsz": IMAGE_SIZE,
        "batch": 2,
        "epochs": 15,
        "patience": 8,
        "lr0": 0.0001,
        "locked_conf_a_priori": 0.35,
        "why": "The 15-epoch faint fine-tune was still best at epoch 14, and its vote scored fold PQ 0.45423. Continue the same run without overwriting that checkpoint.",
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "test_labels_used": False,
        "mps_owner": owner,
    }
    if (DEST / "weights" / "best.pt").exists():
        raise SystemExit(f"already trained: {DEST}")
    if not PARENT.is_file() or not DATA.is_file():
        raise SystemExit(f"missing parent or data: {PARENT} {DATA}")
    if owner:
        raise SystemExit(f"{owner} owns MPS")
    if not args.start:
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11s_seg_erode_gt_2048_faint_e30_family.out"
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
            str(IMAGE_SIZE),
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
    out.update({"started": True, "train_pid": run.pid, "train_log": str(run_log.relative_to(ROOT))})
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
