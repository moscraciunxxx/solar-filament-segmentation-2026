#!/usr/bin/env python3
"""When 3ch training leaves MPS, score standalone PQ then Slot-1 merge. Never submits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR
from gate_unet_family import refuse_3ch

PYTHON = Path("~/.venvs/filament/bin/python")
WEIGHTS = WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"
CURVE = ARTIFACTS / "unet_3ch_disk_train_curve.json"
CURVE_FAST = WEIGHTS_DIR / "unet_3ch_disk" / "train_curve.json"
GATE_PATH = ARTIFACTS / "unet_3ch_disk_slot1_merge_fold0_official.json"
MIN_EPOCHS_TO_SCORE = 12


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def train_alive() -> bool:
    return any(pid != os.getpid() for pid in _pgrep("train_unet_3ch.py"))


def epochs_done() -> int:
    for path in (CURVE, CURVE_FAST):
        if path.exists():
            hist = json.loads(path.read_text()).get("history") or []
            return int(len(hist))
    return 0


def main() -> None:
    reason = refuse_3ch(ignore_owners=("train_unet_3ch.py",))
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(1)
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching train_unet_3ch.py; will score standalone then Slot-1 merge", flush=True)
    waited = 0
    while not train_alive() and not WEIGHTS.exists() and waited < 180:
        time.sleep(5)
        waited += 5
    if not train_alive() and not WEIGHTS.exists():
        raise SystemExit("train_unet_3ch.py never appeared and best.pt is missing")
    while train_alive():
        time.sleep(30)
    if not WEIGHTS.exists():
        raise SystemExit(f"train ended but missing {WEIGHTS}")
    n_ep = epochs_done()
    if n_ep < MIN_EPOCHS_TO_SCORE:
        raise SystemExit(f"STALL: train_unet_3ch.py gone after {n_ep} epochs (<{MIN_EPOCHS_TO_SCORE}).")
    standalone = subprocess.run([str(PYTHON), str(ROOT / "scripts" / "eval_unet_3ch.py"), "--weights", str(WEIGHTS)], cwd=str(FAST_CACHE), env=env, check=False)
    if standalone.returncode != 0:
        raise SystemExit(f"eval_unet_3ch failed: {standalone.returncode}")
    merge = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts" / "eval_unet_3ch_merge.py"), "--weights", str(WEIGHTS), "--min-areas", "642,1034", "--thrs", "0.45,0.50,0.55,0.60"],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if merge.returncode != 0:
        raise SystemExit(f"eval_unet_3ch_merge failed: {merge.returncode}")
    print(json.dumps(json.loads(GATE_PATH.read_text())["gate"], indent=2), flush=True)
    print("3ch scored; overwatch decides next. no auto-submit.", flush=True)


if __name__ == "__main__":
    main()
