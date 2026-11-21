"""Drop the smallest masks once an image exceeds its year's train count.

The cap for a year is the rounded median filament count on folds other than
0. Fold 0 does not set it. Mask outlines are not edited.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .rle import decode_rle, encode_rle


def _image_count(views: dict, image_ids: list[str]) -> float:
    counts = [len(views[image_id].filaments) for image_id in image_ids]
    return float(np.median(counts)) if counts else 0.0


def year_count_caps(views: dict, by_file: dict, fold_of: dict) -> tuple[dict[int, int], int]:
    by_year: dict[int, list[float]] = defaultdict(list)
    for file_name, fold in fold_of.items():
        if fold == 0:
            continue
        by_year[int(file_name[:4])].append(_image_count(views, by_file[file_name]))
    flat = [count for counts in by_year.values() for count in counts]
    if not flat:
        raise ValueError("no train-fold images")
    default = max(1, int(np.round(float(np.median(flat)))))
    caps = {
        year: max(1, int(np.round(float(np.median(counts)))))
        for year, counts in by_year.items()
    }
    return caps, default


def drop_smallest(masks: list[np.ndarray], cap: int) -> list[np.ndarray]:
    if cap < 1 or len(masks) <= cap:
        return list(masks)
    order = sorted(masks, key=lambda mask: int(np.count_nonzero(mask)), reverse=True)
    return order[:cap]


def drop_smallest_rles(rles: list[str], cap: int) -> tuple[list[str], int]:
    if cap < 1 or len(rles) <= cap:
        return list(rles), 0
    decoded = [decode_rle(rle) for rle in rles]
    kept = drop_smallest(decoded, cap)
    return [encode_rle(mask) for mask in kept], len(rles) - len(kept)
