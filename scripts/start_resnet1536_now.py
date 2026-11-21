#!/usr/bin/env python3
"""Start ResNet@1536 train + dest-safe scorer. Default is dry.

Refuses while residual YOLO (or any other job) owns MPS.
Never calls train_unet_resnet.py. Never submits.
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

from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR
from gate_unet_family import RESNET_GATE, live_mps_owner, refuse_resnet

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_unet_resnet1536.py"
WATCH = ROOT / "scripts" / "watch_resnet1536_then_score.py"
RECEIPT = ARTIFACTS / "start_resnet1536_now.json"
DEST = WEIGHTS_DIR / "unet_resnet34_1536"
FORBIDDEN = WEIGHTS_DIR / "unet_resnet34"
RESIDUAL_OFFICIAL = ARTIFACTS / "yolo11s_seg_residual_fold0_official.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "start_flag": args.start,
        "auto_start": False,
        "family": "unet_resnet34_1536",
        "dest": str(DEST),
        "forbidden": str(FORBIDDEN),
        "do_not_submit": True,
        "mps_owner": owner,
        "residual_official_exists": RESIDUAL_OFFICIAL.exists(),
        "never": "scripts/train_unet_resnet.py --imgsz 1536",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.resolve() == FORBIDDEN.resolve():
        out["refused"] = "dest resolved to unet_resnet34"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if owner:
        out["refused"] = f"Refusing 1536 start: {owner} owns MPS."
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(0 if not args.start else 1)
    if not RESIDUAL_OFFICIAL.exists():
        out["refused"] = (
            "Refusing 1536 start: wait for artifacts/yolo11s_seg_residual_fold0_official.json "
            "so residual score is not sharing MPS."
        )
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(0 if not args.start else 1)
    reason = refuse_resnet(RESNET_GATE)
    if reason:
        out["refused"] = reason
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(reason)
        raise SystemExit(0 if not args.start else 1)
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    train_log = ARTIFACTS / "resnet1536_train.out"
    watch_log = ARTIFACTS / "resnet1536_watch.out"
    watch = subprocess.Popen(
        [str(PYTHON), str(WATCH)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=watch_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    train = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--imgsz",
            "1536",
            "--batch",
            "1",
            "--epochs",
            "40",
            "--device",
            "auto",
        ],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=train_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    out.update({"started": True, "train_pid": train.pid, "watch_pid": watch.pid})
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
