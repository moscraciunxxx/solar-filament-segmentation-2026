#!/usr/bin/env python3
"""Score fold 0. Parent is the default. --ens3 decodes the 3-seed champion."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filament.paths import WEIGHTS_DIR

WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"


def main() -> None:
    if "--ens3" in sys.argv:
        script = Path(__file__).resolve().parent / "decode_ens3.py"
        raise SystemExit(subprocess.call([sys.executable, str(script), "fold0"], cwd=str(PROJECT_ROOT)))
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "score_family.py"),
        "--weights",
        str(WEIGHTS),
        "--family",
        "yolo11s_seg_erode_gt_2048",
        "--imgsz",
        "2048",
        "--confs",
        "0.35",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
