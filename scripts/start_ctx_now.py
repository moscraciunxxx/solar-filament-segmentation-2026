#!/usr/bin/env python3
"""Start ResNet extras + Slot1-context fold-0 merge + CPU watcher. Default is dry.

Does not overwrite logistic or HGB dests. Never submits.
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
from gate_unet_family import (
    EXTRA_FILTER_CTX_GATE,
    FILTER_CTX_WEIGHTS_NAME,
    FILTER_HGB_WEIGHTS_NAME,
    FILTER_WEIGHTS_NAME,
    RESNET_GATE,
    live_mps_owner,
    refuse_extra_filter,
)

PYTHON = Path("~/.venvs/filament/bin/python")
EVAL = ROOT / "scripts" / "eval_unet_resnet_extra_filter_ctx_merge.py"
WATCH = ROOT / "scripts" / "watch_resnet_ctx_then_score.py"
RECEIPT = ARTIFACTS / "start_ctx_now.json"
CTX_DEST = WEIGHTS_DIR / FILTER_CTX_WEIGHTS_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner()
    reason = refuse_extra_filter(RESNET_GATE, filter_dest=CTX_DEST)
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "start_flag": args.start,
        "auto_start": False,
        "family": "unet_resnet_extra_filter_ctx_slot1_merge",
        "filter_dest": str(CTX_DEST),
        "gate": str(EXTRA_FILTER_CTX_GATE),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "overwrites_logistic_or_hgb": False,
        "mps_owner": owner,
        "refuse": reason,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if CTX_DEST.name in {FILTER_WEIGHTS_NAME, FILTER_HGB_WEIGHTS_NAME}:
        reason = "Refusing ctx dest: would overwrite logistic or HGB."
    if owner:
        out["refused"] = f"Refusing ctx start: {owner} owns MPS."
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(0 if not args.start else 1)
    if reason:
        out["refused"] = reason
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(reason)
        raise SystemExit(0 if not args.start else 1)
    if not args.start:
        out["dry"] = True
        out["would_start"] = [str(EVAL) + " --run", str(WATCH)]
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    eval_log = ARTIFACTS / "resnet_ctx_extra_filter_eval.out"
    watch_log = ARTIFACTS / "resnet_ctx_extra_filter_watch.out"
    eval_fp = eval_log.open("a")
    watch_fp = watch_log.open("a")
    watch = subprocess.Popen(
        [str(PYTHON), str(WATCH)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=watch_fp,
        stderr=subprocess.STDOUT,
    )
    ev = subprocess.Popen(
        [str(PYTHON), str(EVAL), "--run"],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=eval_fp,
        stderr=subprocess.STDOUT,
    )
    out.update(
        {
            "started": True,
            "eval_pid": ev.pid,
            "watch_pid": watch.pid,
            "eval_log": str(eval_log.relative_to(ROOT)),
            "watch_log": str(watch_log.relative_to(ROOT)),
        }
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
