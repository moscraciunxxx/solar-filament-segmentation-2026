#!/usr/bin/env python3
"""Test CSV for the fold-0 winner: 2048 parent plus pixel-disjoint year-shift.

Uses filament.disjoint_add. Does not read test labels. Does not submit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.disjoint_add import disjoint_add
from filament.metric import pred_to_df
from filament.paths import FAST_SUBMISSION_DIR

PARENT = FAST_SUBMISSION_DIR / "submission_yolo11s_erode_gt_2048_conf035.csv"
SHIFT = FAST_SUBMISSION_DIR / "submission_yolo11l_shift_erode3x1_conf035_sz1280.csv"
OUT = FAST_SUBMISSION_DIR / "submission_disjoint_shift_on_2048.csv"


def _by_stem(frame: pd.DataFrame) -> dict[str, list[str]]:
    stems = frame["filament_id"].str.split("_", n=1).str[0]
    grouped: dict[str, list[str]] = {}
    for stem, rle in zip(stems, frame["segmentation_rle"], strict=True):
        grouped.setdefault(stem, []).append(rle)
    return grouped


def main() -> None:
    parent_by = _by_stem(pd.read_csv(PARENT))
    shift_by = _by_stem(pd.read_csv(SHIFT))
    frames = []
    added = 0
    for stem in sorted(set(parent_by) | set(shift_by)):
        parent_rles = parent_by.get(stem, [])
        kept, n_added = disjoint_add(parent_rles, shift_by.get(stem, []))
        if kept[: len(parent_rles)] != parent_rles:
            raise SystemExit(f"parent masks were not kept intact for {stem}")
        added += n_added
        if kept:
            frames.append(pred_to_df(stem, kept))
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT, index=False)
    print(json.dumps({
        "path": str(OUT),
        "n_rows": int(len(pred)),
        "shift_instances_added": added,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }, indent=2))


if __name__ == "__main__":
    main()
