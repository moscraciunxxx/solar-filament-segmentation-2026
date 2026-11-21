#!/usr/bin/env python3
"""CPU keeper: restart the D score-watch only. Never trains. Never submits.

If train is alive and watch dies, respawn watch with start_new_session.
If train is dead, best.pt exists, official missing, and watch is dead, respawn
watch so official PQ still lands. Never starts train_yolo or start_* --start.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR

PYTHON = Path("~/.venvs/filament/bin/python")
WATCH = ROOT / "scripts" / "watch_yolo11l_erode_gt_then_score.py"
OFFICIAL = ARTIFACTS / "yolo11l_seg_erode_gt_fold0_official.json"
BEST = WEIGHTS_DIR / "yolo11l_seg_erode_gt" / "weights" / "best.pt"
LOG = ARTIFACTS / "yolo11l_erode_gt_watch.out"
RECEIPT = ARTIFACTS / "keep_yolo11l_erode_gt_watch.json"
SLEEP = 60
MAX_RESTARTS = 5


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def _alive(pattern: str) -> bool:
    pid = os.getpid()
    return any(p != pid for p in _pgrep(pattern))


def train_alive() -> bool:
    return _alive("train_yolo.py --name yolo11l_seg_erode_gt")


def watch_alive() -> bool:
    return _alive("watch_yolo11l_erode_gt_then_score.py")


def _stamp(payload: dict) -> None:
    payload.update(
        {
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "do_not_submit": True,
            "kaggle_submit_called": False,
            "never_starts_train": True,
            "slot2_stays": 56148889,
        }
    )
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")


def spawn_watch() -> int:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(
            f"---- keep-watch spawn {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} ----\n"
        )
        fh.flush()
        proc = subprocess.Popen(
            [str(PYTHON), str(WATCH)],
            cwd=str(ROOT),
            env=env,
            start_new_session=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    return proc.pid


def main() -> None:
    restarts = 0
    print("keep watch yolo11l_seg_erode_gt", flush=True)
    while True:
        if OFFICIAL.exists():
            _stamp({"status": "official_exists", "restarts": restarts})
            print("official exists; keeper exit", flush=True)
            return
        tr, wa = train_alive(), watch_alive()
        need = (not wa) and (tr or BEST.exists())
        if need:
            if restarts >= MAX_RESTARTS:
                _stamp({"status": "max_restarts", "restarts": restarts, "train_alive": tr})
                raise SystemExit("keep-watch: max restarts")
            pid = spawn_watch()
            restarts += 1
            _stamp(
                {
                    "status": "restarted_watch",
                    "restarts": restarts,
                    "watch_pid": pid,
                    "train_alive": tr,
                    "best_pt": BEST.exists(),
                }
            )
            print(f"spawned watch {pid} ({restarts}/{MAX_RESTARTS})", flush=True)
        else:
            _stamp(
                {
                    "status": "watching",
                    "restarts": restarts,
                    "train_alive": tr,
                    "watch_alive": wa,
                    "best_pt": BEST.exists(),
                }
            )
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
