"""Decide which training images get one extra copy.

An image is faint when the median contrast of its measurable medoid
filaments sits below the training-fold median. Fold 0 is never an input
to this choice.
"""

from __future__ import annotations

import numpy as np


def image_is_faint(contrasts: list[float | None], median: float) -> bool:
    measurable = [value for value in contrasts if value is not None]
    if not measurable:
        return False
    return float(np.median(measurable)) < median
