#!/usr/bin/env python3
"""Start native-2048 patch train + score watcher. Default is dry.

Requires: ResNet merge gate in START_PATCH_AFTER_RESNET, not PROMOTE_LOOK,
and refuse_patch() is None. Never submits. Weights unet_patch_dhebe.
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
from gate_unet_family import (
    PROMOTE,
    RESNET_GATE,
    START_PATCH_AFTER_RESNET,
    live_mps_owner,
    read_gate,
    refuse_patch,
    resolve_start_gate,
)

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_unet_patch.py"
WATCH = ROOT / "scripts" / "watch_patch_then_score.py"
RANK_WAIT = ROOT / "scripts" / "wait_upload_then_confirm_rank.py"
RECEIPT = ARTIFACTS / "start_patch_now.json"


def refuse() -> str | None:
    if not RESNET_GATE.exists():
        return f"Refusing patch start: missing {RESNET_GATE}."
    decision, _ = read_gate(RESNET_GATE)
    if decision == PROMOTE:
        return "Refusing patch start: ResNet PROMOTE_LOOK — one merged look, no second family."
    if decision not in START_PATCH_AFTER_RESNET:
        return f"Refusing patch start: ResNet gate is {decision!r}."
    reason = refuse_patch(resolve_start_gate())
    if reason:
        return reason
    owner = live_mps_owner()
    if owner:
        return f"Refusing patch start: {owner} owns MPS."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        action="store_true",
        help="Actually spawn train_unet_patch.py + watch_patch_then_score.py. Default is dry.",
    )
    args = parser.parse_args()
    reason = refuse()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "start_flag": args.start,
        "auto_start": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "weights": "unet_patch_dhebe",
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
    train_log = ARTIFACTS / "patch_train.out"
    watch_log = ARTIFACTS / "patch_watch.out"
    rank_log = ARTIFACTS / "patch_rank_wait.out"
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
        "Fresh waiter loads on-disk UPLOADS (v1+ResNet+patch+3ch). "
        "Live 48415/54463 predate patch/3ch receipts; do not kill them."
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
