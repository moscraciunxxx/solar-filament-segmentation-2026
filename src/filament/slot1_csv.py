"""Locked Slot-1 test CSV (ref 56148686). Load only. Never infers. Never submits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import FAST_SUBMISSION_DIR

SLOT1_TEST_CSV = FAST_SUBMISSION_DIR / "submission_yolo11m_1280_conf035.csv"
# These two test JPEGs have zero Slot-1 rows — YOLO proposed nothing. Empty list is correct.
EMPTY_SLOT1_TEST_STEMS = ("20200310113230Ch", "20211214174010Mh")


def load_slot1_test_rles(
    images: list[Path],
    csv: Path = SLOT1_TEST_CSV,
) -> dict[str, list[str]]:
    """Map test stem -> Slot-1 RLEs. Missing stems are empty (valid zero-det)."""
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    if "filament_id" not in df.columns or "segmentation_rle" not in df.columns:
        raise ValueError(f"Slot-1 CSV missing columns: {csv}")
    out: dict[str, list[str]] = {}
    for path in images:
        stem = path.stem
        sub = df[df["filament_id"].astype(str).str.startswith(stem + "_")]
        out[stem] = sub["segmentation_rle"].astype(str).tolist()
    return out
