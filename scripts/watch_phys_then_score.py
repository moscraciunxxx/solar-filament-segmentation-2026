#!/usr/bin/env python3
"""When yolo11s_seg_phys leaves MPS, run official PQ with --phys-stack.

Never submits. Never auto-starts rater/union/tiles. Never reads test labels.
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
WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_phys" / "weights" / "best.pt"
GATE_PATH = ARTIFACTS / "yolo11s_seg_phys_fold0_official.json"


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def train_alive() -> bool:
    return any(pid != os.getpid() for pid in _pgrep("train_yolo.py.*yolo11s_seg_phys"))


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching yolo11s_seg_phys", flush=True)
    while train_alive():
        time.sleep(30)
    if not WEIGHTS.exists():
        raise SystemExit(f"train ended but missing {WEIGHTS}")
    print("train gone; scoring official PQ --phys-stack", flush=True)
    score = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "score_family.py"),
            "--weights",
            str(WEIGHTS),
            "--family",
            "yolo11s_seg_phys",
            "--imgsz",
            "1024",
            "--confs",
            "0.25,0.35,0.40",
            "--phys-stack",
        ],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if score.returncode != 0:
        raise SystemExit(f"score_family failed: {score.returncode}")
    gate = json.loads(GATE_PATH.read_text())["gate"]
    print(json.dumps(gate, indent=2), flush=True)
    print("phys scored; overwatch decides next. no auto-submit, no auto-rater.", flush=True)


if __name__ == "__main__":
    main()
