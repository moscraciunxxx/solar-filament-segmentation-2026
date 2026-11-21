#!/usr/bin/env python3
"""When yolo11l_seg_1280 leaves MPS, score official fold-0 at locked conf 0.35.

Never submits. Year-shift stays Slot 2. Parent bar is locked erode 0.4365,
not the old s-medoid 0.38 gate printed by score_family.
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

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_CACHE, WEIGHTS_DIR

PYTHON = Path("~/.venvs/filament/bin/python")
WEIGHTS = WEIGHTS_DIR / "yolo11l_seg_1280" / "weights" / "best.pt"
GATE_PATH = ARTIFACTS / "yolo11l_seg_1280_fold0_official.json"
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
            _pgrep("train_yolo.py --name yolo11l_seg_1280")
            + _pgrep("start_yolo11l_now.py --start")
        )
    )


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    print("watching yolo11l_seg_1280", flush=True)
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
            "yolo11l_seg_1280",
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
        "locked_conf_a_priori": LOCKED_CONF,
        "pq": pq,
        "erode_bar": ERODE_BAR,
        "beats_erode": pq > ERODE_BAR,
        "score_family_gate": payload.get("gate"),
        "parent_upload": pq > ERODE_BAR,
        "do_not_submit": True,
        "slot2_stays": 56148889,
        "note": "score_family may print PROMOTE at 0.38 vs s-medoid. Parent still requires pq > 0.4365 and hidden-exam before any look.",
    }
    (ARTIFACTS / "yolo11l_parent_bar.json").write_text(json.dumps(parent, indent=2) + "\n")
    print(json.dumps(parent, indent=2), flush=True)
    print("do_not_submit true; year-shift 56148889 stays", flush=True)
    pred_csv = FAST_ARTIFACTS / "yolo11l_seg_1280_fold0_conf0.35_pred.csv"
    if not pred_csv.exists():
        raise SystemExit(f"score_family wrote no pred CSV: {pred_csv}")
    print("CPU locked erode 3x1 on yolo11l pred (MPS free before erode-GT train)", flush=True)
    erode = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "eval_locked_erode_on_csv.py"),
            "--pred",
            str(pred_csv),
            "--family",
            "yolo11l_seg_1280",
        ],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    parent["erode_post_returncode"] = erode.returncode
    (ARTIFACTS / "yolo11l_parent_bar.json").write_text(json.dumps(parent, indent=2) + "\n")
    if erode.returncode != 0:
        raise SystemExit(f"yolo11l erode-post returncode {erode.returncode}")
    print("handoff: start_erode_gt_now.py --start (dest yolo11m_seg_erode_gt only)", flush=True)
    handoff = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts" / "start_erode_gt_now.py"), "--start"],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    (ARTIFACTS / "yolo11l_to_erode_gt_handoff.json").write_text(
        json.dumps(
            {
                "yolo11l_pq": pq,
                "beats_erode": pq > ERODE_BAR,
                "handoff_returncode": handoff.returncode,
                "do_not_submit": True,
                "slot2_stays": 56148889,
            },
            indent=2,
        )
        + "\n"
    )
    if handoff.returncode != 0:
        print(f"erode-GT handoff returncode {handoff.returncode} (not a submit)", flush=True)


if __name__ == "__main__":
    main()
