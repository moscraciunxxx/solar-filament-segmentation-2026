"""Drop a mask whose centroid sits outside the train-fold radius band.

The band is the 1st to 99th percentile of filament centroids on folds other
than 0. Fold 0 does not set it. The mask outline is not edited.
"""

from __future__ import annotations

import numpy as np

from .rle import decode_rle, encode_rle

LOW_PCT = 1
HIGH_PCT = 99
CENTER_Y = 1023.5
CENTER_X = 1023.5


def centroid_radius(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0
    return float(np.hypot(float(ys.mean()) - CENTER_Y, float(xs.mean()) - CENTER_X))


def train_radius_band(
    views: dict,
    by_file: dict,
    fold_of: dict,
    low_pct: int = LOW_PCT,
    high_pct: int = HIGH_PCT,
) -> tuple[float, float]:
    radii = []
    for file_name, fold in fold_of.items():
        if fold == 0:
            continue
        for image_id in by_file[file_name]:
            for filament in views[image_id].filaments:
                poly = filament.polygon
                if len(poly) == 0:
                    continue
                cy = float(poly[:, 1].mean())
                cx = float(poly[:, 0].mean())
                radii.append(float(np.hypot(cy - CENTER_Y, cx - CENTER_X)))
    if not radii:
        raise ValueError("no train-fold filaments")
    lo, hi = np.percentile(np.asarray(radii, dtype=np.float64), [low_pct, high_pct])
    return float(lo), float(hi)


def drop_outside_band(masks: list[np.ndarray], lo: float, hi: float) -> list[np.ndarray]:
    return [mask for mask in masks if lo <= centroid_radius(mask) <= hi]


def drop_outside_band_rles(rles: list[str], lo: float, hi: float) -> tuple[list[str], int]:
    kept: list[str] = []
    dropped = 0
    for rle in rles:
        mask = decode_rle(rle)
        if lo <= centroid_radius(mask) <= hi:
            kept.append(encode_rle(mask))
        else:
            dropped += 1
    return kept, dropped
