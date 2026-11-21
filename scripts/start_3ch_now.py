#!/usr/bin/env python3
"""Start 3ch U-Net train + score watcher. Default is dry.

Requires refuse_3ch() is None (patch merge exists, not PROMOTE_LOOK, no MPS owner).
Never submits. Weights unet_3ch_disk. Not yolo_seg_phys.
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

from filament.paths import ARTIFACTS, FAST_CACHE
from gate_unet_family import refuse_3ch

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_unet_3ch.py"
WATCH = ROOT / "scripts" / "watch_unet_3ch_then_score.py"
RANK_WAIT = ROOT / "scripts" / "wait_upload_then_confirm_rank.py"
RECEIPT = ARTIFACTS / "start_3ch_now.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    reason = refuse_3ch()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "start_flag": args.start,
        "auto_start": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "not_yolo_phys": True,
        "weights": "unet_3ch_disk",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if reason:
        out["refused"] = reason
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(reason)
        raise SystemExit(0 if not args.start else 1)
    if not args.start:
        out["dry"] = True
        out["would_start"] = [str(TRAIN), str(WATCH), str(RANK_WAIT)]
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    train_log = ARTIFACTS / "ch3_train.out"
    watch_log = ARTIFACTS / "ch3_watch.out"
    rank_log = ARTIFACTS / "ch3_rank_wait.out"
    train_fp = train_log.open("a")
    watch_fp = watch_log.open("a")
    rank_fp = rank_log.open("a")
    watch = subprocess.Popen(
        [str(PYTHON), str(WATCH)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=watch_fp,
        stderr=subprocess.STDOUT,
    )
    train = subprocess.Popen(
        [str(PYTHON), str(TRAIN)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=train_fp,
        stderr=subprocess.STDOUT,
    )
    rank_wait = subprocess.Popen(
        [str(PYTHON), str(RANK_WAIT)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=rank_fp,
        stderr=subprocess.STDOUT,
    )
    out["train_log"] = str(train_log)
    out["watch_log"] = str(watch_log)
    out["started"] = True
    out["watch_pid"] = watch.pid
    out["train_pid"] = train.pid
    out["rank_wait_pid"] = rank_wait.pid
    out["rank_wait_why"] = (
        "Fresh waiter loads on-disk UPLOADS including upload_if_3ch_promote.json. "
        "Live 48415/54463 predate that receipt; do not kill them."
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
