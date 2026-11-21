"""Drop a mask larger than the train-fold filament area cap.

The cap is the 99th percentile of ground-truth areas on folds other than 0.
Fold 0 does not set it. Nothing is added and no mask is reshaped.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_rle, encode_rle

PERCENTILE = 99


def train_area_cap(views: dict, by_file: dict, fold_of: dict, percentile: int = PERCENTILE) -> float:
    areas = [
        filament.area
        for file_name, fold in fold_of.items()
        if fold != 0
        for image_id in by_file[file_name]
        for filament in views[image_id].filaments
    ]
    if not areas:
        raise ValueError("no train-fold filaments")
    return float(np.percentile(np.asarray(areas, dtype=np.float64), percentile))


def drop_over_cap(masks: list[np.ndarray], cap: float) -> list[np.ndarray]:
    return [mask for mask in masks if float(np.count_nonzero(mask)) <= cap]


def drop_over_cap_rles(rles: list[str], cap: float) -> tuple[list[str], int]:
    kept: list[str] = []
    dropped = 0
    for rle in rles:
        mask = decode_rle(rle)
        if float(np.count_nonzero(mask)) <= cap:
            kept.append(encode_rle(mask))
        else:
            dropped += 1
    return kept, dropped
