"""Optional v2 instance split: cut semantic blobs along train-JSON spine ridges.

Not wired into eval_unet.py v1. Use only after RESEARCH_HOLD_SPINE_V2.
"""

from __future__ import annotations

import cv2
import numpy as np

from .postprocess import connected_instances, resolve_overlaps


def split_binary_on_ridge(
    binary: np.ndarray,
    ridge: np.ndarray,
    min_area: int = 96,
    max_area: int = 80_000,
) -> list[np.ndarray]:
    """Remove 1px ridge from the mask, then CC. Falls back to unsplit CC if empty."""
    cut = ((binary > 0) & (ridge == 0)).astype(np.uint8)
    instances = connected_instances(cut, min_area=min_area, max_area=max_area)
    if not instances:
        instances = connected_instances(binary, min_area=min_area, max_area=max_area)
    return resolve_overlaps(instances)
