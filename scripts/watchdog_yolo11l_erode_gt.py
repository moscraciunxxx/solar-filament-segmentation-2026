#!/usr/bin/env python3
"""CPU watchdog: restart yolo11l_seg_erode_gt only after a silent mid-train death.

Never submits. Never scores. Never starts a second family.
Stops once best.pt or official PQ exists (score watcher owns that path).
Max 3 restarts so a hard crash cannot loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, WEIGHTS_DIR

PYTHON = Path("~/.venvs/filament/bin/python")
START = ROOT / "scripts" / "start_yolo11l_erode_gt_now.py"
DEST = WEIGHTS_DIR / "yolo11l_seg_erode_gt"
OFFICIAL = ARTIFACTS / "yolo11l_seg_erode_gt_fold0_official.json"
RECEIPT = ARTIFACTS / "watchdog_yolo11l_erode_gt.json"
MAX_RESTARTS = 3
SLEEP = 60


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def train_alive() -> bool:
    pid = os.getpid()
    return any(p != pid for p in _pgrep("train_yolo.py --name yolo11l_seg_erode_gt"))


def _stamp(payload: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    payload["written"] = datetime.now(timezone.utc).isoformat()
    payload["do_not_submit"] = True
    payload["kaggle_submit_called"] = False
    payload["slot2_stays"] = 56148889
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    restarts = 0
    print("watchdog yolo11l_seg_erode_gt", flush=True)
    while True:
        best = DEST / "weights" / "best.pt"
        if OFFICIAL.exists() or best.exists():
            _stamp({"status": "done_or_scored", "official": OFFICIAL.exists(), "best_pt": best.exists(), "restarts": restarts})
            print("best.pt or official exists; watchdog exit", flush=True)
            return
        if train_alive():
            _stamp({"status": "train_alive", "restarts": restarts})
            time.sleep(SLEEP)
            continue
        if restarts >= MAX_RESTARTS:
            _stamp({"status": "max_restarts", "restarts": restarts})
            raise SystemExit("watchdog: max restarts without best.pt")
        print(f"train dead, no best.pt; restart {restarts + 1}/{MAX_RESTARTS}", flush=True)
        proc = subprocess.run(
            [str(PYTHON), str(START), "--start"],
            cwd=str(ROOT),
            check=False,
        )
        restarts += 1
        _stamp({"status": "restarted", "restarts": restarts, "start_returncode": proc.returncode})
        if proc.returncode != 0:
            print(f"start returncode {proc.returncode}", flush=True)
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
