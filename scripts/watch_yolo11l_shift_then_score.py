#!/usr/bin/env python3
"""When yolo11l_seg_shift leaves MPS, score official fold-0 at locked conf 0.35.

Never submits. Year-shift Slot 2 stays. Parent bar is locked erode 0.4365.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR

PYTHON = Path("~/.venvs/filament/bin/python")
WEIGHTS = WEIGHTS_DIR / "yolo11l_seg_shift" / "weights" / "best.pt"
GATE_PATH = ARTIFACTS / "yolo11l_seg_shift_fold0_official.json"
ERODE_BAR = 0.4365
LOCKED_CONF = 0.35


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def pipeline_alive() -> bool:
    pid = os.getpid()
    return any(
        p != pid
        for p in (
            _pgrep("train_yolo.py --name yolo11l_seg_shift")
            + _pgrep("start_yolo11l_shift_now.py --start")
        )
    )


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching yolo11l_seg_shift", flush=True)
    waited = 0
    seen_train = pipeline_alive()
    while not seen_train and waited < 180:
        time.sleep(5)
        waited += 5
        seen_train = pipeline_alive()
    while pipeline_alive():
        time.sleep(30)
        seen_train = True
    if not seen_train:
        raise SystemExit("train never appeared; refuse to score a stale best.pt")
    if not WEIGHTS.exists():
        raise SystemExit(f"train ended but missing {WEIGHTS}")
    print("train gone; scoring official PQ at locked conf 0.35", flush=True)
    score = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "score_family.py"),
            "--weights",
            str(WEIGHTS),
            "--family",
            "yolo11l_seg_shift",
            "--imgsz",
            "1280",
            "--confs",
            "0.35",
        ],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if score.returncode != 0:
        raise SystemExit(f"score_family failed: {score.returncode}")
    payload = json.loads(GATE_PATH.read_text())
    locked = payload.get("locked_row") or payload.get("best") or {}
    pq = float(locked.get("pq", payload.get("pq", 0.0)))
    parent = {
        "family": "yolo11l_seg_shift",
        "locked_conf_a_priori": LOCKED_CONF,
        "pq": pq,
        "erode_bar": ERODE_BAR,
        "beats_erode": pq > ERODE_BAR,
        "score_family_gate": payload.get("gate"),
        "parent_upload": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "slot2_stays": 56148889,
        "note": (
            "score_family may print PROMOTE vs 0.38. Parent requires pq > 0.4365 "
            "AND hidden-exam (standalone, not extras-OR) before any look."
        ),
    }
    (ARTIFACTS / "yolo11l_shift_parent_bar.json").write_text(json.dumps(parent, indent=2) + "\n")
    print(json.dumps(parent, indent=2), flush=True)
    print("do_not_submit true; year-shift 56148889 stays", flush=True)


if __name__ == "__main__":
    main()
