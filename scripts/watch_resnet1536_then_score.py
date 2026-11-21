#!/usr/bin/env python3
"""When ResNet@1536 leaves MPS, score dest-safe standalone then Slot-1 merge.

Never writes unet_resnet34_* 1024 official JSONs. Never submits.
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
WEIGHTS = WEIGHTS_DIR / "unet_resnet34_1536" / "best.pt"
CURVE = ARTIFACTS / "unet_resnet34_1536_train_curve.json"
CURVE_FAST = WEIGHTS_DIR / "unet_resnet34_1536" / "train_curve.json"
GATE_PATH = ARTIFACTS / "unet_resnet34_1536_slot1_merge_fold0_official.json"
MIN_EPOCHS_TO_SCORE = 12


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def train_alive() -> bool:
    return any(pid != os.getpid() for pid in _pgrep("train_unet_resnet1536.py"))


def epochs_done() -> int:
    for path in (CURVE, CURVE_FAST):
        if path.exists():
            hist = json.loads(path.read_text()).get("history") or []
            return int(len(hist))
    return 0


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching train_unet_resnet1536.py", flush=True)
    waited = 0
    while not train_alive() and not WEIGHTS.exists() and waited < 180:
        time.sleep(5)
        waited += 5
    if not train_alive() and not WEIGHTS.exists():
        raise SystemExit("train_unet_resnet1536.py never appeared and best.pt is missing")
    while train_alive():
        time.sleep(30)
    if not WEIGHTS.exists():
        raise SystemExit(f"train ended but missing {WEIGHTS}")
    n_ep = epochs_done()
    if n_ep < MIN_EPOCHS_TO_SCORE:
        raise SystemExit(
            f"STALL: train_unet_resnet1536.py gone after {n_ep} epochs "
            f"(<{MIN_EPOCHS_TO_SCORE}). Do not official-score a short checkpoint."
        )
    print(f"train gone after {n_ep} epochs; scoring 1536 standalone", flush=True)
    standalone = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "eval_unet_resnet1536.py"),
            "--weights",
            str(WEIGHTS),
            "--imgsz",
            "1536",
        ],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if standalone.returncode != 0:
        raise SystemExit(f"eval_unet_resnet1536 failed: {standalone.returncode}")
    print("scoring Slot-1 + ResNet@1536 merge", flush=True)
    merge = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "eval_unet_resnet1536_merge.py"),
            "--weights",
            str(WEIGHTS),
            "--imgsz",
            "1536",
            "--min-area",
            "642",
            "--min-areas",
            "642,1034",
            "--thrs",
            "0.45,0.50,0.55,0.60",
        ],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if merge.returncode != 0:
        raise SystemExit(f"eval_unet_resnet1536_merge failed: {merge.returncode}")
    if GATE_PATH.name == "unet_resnet34_slot1_merge_fold0_official.json":
        raise SystemExit("Refusing: 1536 gate path collapsed to 1024 official.")
    gate = json.loads(GATE_PATH.read_text())["gate"]
    print(json.dumps(gate, indent=2), flush=True)
    print("resnet1536 scored; no auto-submit.", flush=True)


if __name__ == "__main__":
    main()
