#!/usr/bin/env python3
"""When yolo11s_seg_erode_gt_1408_ms leaves MPS, score official fold-0 at locked conf 0.35.

Infer imgsz stays 1408. Never submits. Parent bar is 0.4365.
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
WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_1408_ms" / "weights" / "best.pt"
GATE_PATH = ARTIFACTS / "yolo11s_seg_erode_gt_1408_ms_fold0_official.json"
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
            _pgrep("train_yolo.py --name yolo11s_seg_erode_gt_1408_ms")
            + _pgrep("start_yolo11s_erode_gt_1408_ms_now.py --start")
        )
    )


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching yolo11s_seg_erode_gt_1408_ms", flush=True)
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
            "yolo11s_seg_erode_gt_1408_ms",
            "--imgsz",
            "1408",
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
    locked = payload.get("locked_row") or {}
    pq = float(locked.get("pq", 0.0))
    parent = {
        "family": "yolo11s_seg_erode_gt_1408_ms",
        "locked_conf_a_priori": LOCKED_CONF,
        "pq": pq,
        "erode_bar": ERODE_BAR,
        "beats_erode": pq > ERODE_BAR,
        "ignore_score_family_PROMOTE_vs_0.38": True,
        "parent_upload": False,
        "do_not_submit": True,
        "slot2_stays": 56148889,
        "note": "Parent look only if pq>0.4365 AND hidden-exam pass. This script never submits.",
    }
    (ARTIFACTS / "yolo11s_erode_gt_1408_ms_parent_bar.json").write_text(json.dumps(parent, indent=2) + "\n")
    print(json.dumps(parent, indent=2), flush=True)


if __name__ == "__main__":
    main()
