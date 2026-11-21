#!/usr/bin/env python3
"""Parent recipe. Exits when the parent weights already exist. Does not train."""

from __future__ import annotations

import sys

from config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filament.paths import WEIGHTS_DIR

WEIGHTS = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"


def main() -> None:
    if WEIGHTS.exists():
        print(f"parent weights present: {WEIGHTS}")
        print("not starting a second train")
        return
    raise SystemExit(
        "best.pt is missing. Start scripts/train_yolo.py only when MPS is idle. "
        "Labels: python scripts/prepare_yolo.py --erode-gt"
    )


if __name__ == "__main__":
    main()
