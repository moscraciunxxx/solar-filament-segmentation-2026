#!/usr/bin/env python3
"""When Method 2 residual YOLO leaves MPS, score official PQ with --residual-stack.

Never submits. Never restarts phys. Never reads test labels.
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
WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_residual" / "weights" / "best.pt"
GATE_PATH = ARTIFACTS / "yolo11s_seg_residual_fold0_official.json"


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
            _pgrep("run_residual_family.py")
            + _pgrep("train_yolo.py --name yolo11s_seg_residual")
            + _pgrep("prepare_yolo.py --residual-stack")
        )
    )


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching yolo11s_seg_residual", flush=True)
    waited = 0
    while not pipeline_alive() and not WEIGHTS.exists() and waited < 180:
        time.sleep(5)
        waited += 5
    while pipeline_alive():
        time.sleep(30)
    if not WEIGHTS.exists():
        raise SystemExit(f"train ended but missing {WEIGHTS}")
    print("train gone; scoring official PQ --residual-stack", flush=True)
    score = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "score_family.py"),
            "--weights",
            str(WEIGHTS),
            "--family",
            "yolo11s_seg_residual",
            "--imgsz",
            "1024",
            "--confs",
            "0.25,0.35,0.40",
            "--residual-stack",
        ],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if score.returncode != 0:
        raise SystemExit(f"score_family failed: {score.returncode}")
    gate = json.loads(GATE_PATH.read_text())["gate"]
    print(json.dumps(gate, indent=2), flush=True)
    print("residual scored; no auto-submit, no phys restart.", flush=True)


if __name__ == "__main__":
    main()
