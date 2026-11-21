#!/usr/bin/env python3
"""Build a test CSV. Parent is the default. --ens3 decodes the 3-seed champion.

Does not call Kaggle and does not overwrite the frozen champion CSV.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filament.paths import FAST_SUBMISSION_DIR, WEIGHTS_DIR

WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"
OUT = FAST_SUBMISSION_DIR / "submission_yolo11s_erode_gt_2048_conf035_solution.csv"


def main() -> None:
    if "--ens3" in sys.argv:
        script = Path(__file__).resolve().parent / "decode_ens3.py"
        raise SystemExit(subprocess.call([sys.executable, str(script), "test"], cwd=str(PROJECT_ROOT)))
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "make_submission.py"),
        "--mode",
        "yolo",
        "--weights",
        str(WEIGHTS),
        "--conf",
        "0.35",
        "--imgsz",
        "2048",
        "--out",
        str(OUT),
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
