#!/usr/bin/env python3
"""When yolo11s_seg_all leaves MPS, run official PQ. Start phys only on a documented fail.

Never submits. Never reads MAGFiLO test labels. Never rmtree yolo_seg.
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
WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_all" / "weights" / "best.pt"
LOCK = FAST_CACHE / "watch_all_annotator.lock"
GATE_PATH = ARTIFACTS / "yolo11s_seg_all_fold0_official.json"
PHYS_CMD = [
    str(PYTHON),
    str(ROOT / "scripts" / "train_yolo.py"),
    "--name",
    "yolo11s_seg_phys",
    "--data",
    str(FAST_CACHE / "yolo_seg_phys" / "data.yaml"),
    "--model",
    "yolo11s-seg.pt",
    "--imgsz",
    "1024",
    "--batch",
    "8",
    "--epochs",
    "36",
    "--patience",
    "12",
    "--copy-paste",
    "0.15",
    "--mixup",
    "0.08",
]


def _pgrep(pattern: str) -> list[int]:
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    return [int(x) for x in proc.stdout.split() if x.strip().isdigit()]


def train_alive() -> bool:
    pids = _pgrep("train_yolo.py.*yolo11s_seg_all")
    return any(pid != os.getpid() for pid in pids)


def phys_alive() -> bool:
    return bool(_pgrep("train_yolo.py.*yolo11s_seg_phys"))


def main() -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        print(f"lock exists {LOCK}; another watcher may be running", flush=True)
    LOCK.write_text(str(os.getpid()))
    print("watching yolo11s_seg_all", flush=True)
    while train_alive():
        time.sleep(30)
    if not WEIGHTS.exists():
        raise SystemExit(f"train ended but missing {WEIGHTS}")
    print("train gone; scoring official PQ", flush=True)
    score = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "score_family.py"),
            "--weights",
            str(WEIGHTS),
            "--family",
            "yolo11s_seg_all",
            "--imgsz",
            "1024",
            "--confs",
            "0.25,0.35,0.40",
        ],
        cwd=str(FAST_CACHE),
        env=env,
        check=False,
    )
    if score.returncode != 0:
        raise SystemExit(f"score_family failed: {score.returncode}")
    gate = json.loads(GATE_PATH.read_text())["gate"]
    print(json.dumps(gate, indent=2), flush=True)
    if gate["decision"] in {"KILL_START_PHYS", "FAIL_START_PHYS"}:
        if phys_alive() or train_alive():
            print("MPS busy; not starting phys", flush=True)
            return
        if not (FAST_CACHE / "yolo_seg_phys" / "data.yaml").exists():
            raise SystemExit("yolo_seg_phys missing")
        print("starting yolo11s_seg_phys on documented fail-path", flush=True)
        log = FAST_CACHE / "yolo11s_seg_phys_watch.log"
        with log.open("ab") as fh:
            subprocess.Popen(
                PHYS_CMD,
                cwd=str(FAST_CACHE),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    else:
        print("not starting phys:", gate["decision"], flush=True)


if __name__ == "__main__":
    main()
