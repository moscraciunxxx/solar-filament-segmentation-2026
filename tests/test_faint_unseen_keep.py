"""An unseen mask drops only when it is fainter than the locked floor."""

from __future__ import annotations

import unittest

import numpy as np

from filament.outside_band import drop_faint_unseen_masks

FLOOR = 8.381084949352285


def _image() -> np.ndarray:
    return np.full((96, 96), 180, dtype=np.uint8)


def _box(image: np.ndarray, y0: int, y1: int, x0: int, x1: int, gray: int) -> np.ndarray:
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    image[y0:y1, x0:x1] = gray
    return mask


class FaintUnseenKeepTest(unittest.TestCase):
    def test_an_unseen_faint_mask_is_dropped(self) -> None:
        image = _image()
        faint = _box(image, 40, 56, 40, 56, 176)
        far = np.zeros_like(faint)
        far[4:12, 4:12] = 1
        kept = drop_faint_unseen_masks([faint], [far], [far], [far], image, FLOOR)
        self.assertEqual(kept, [])

    def test_an_unseen_dark_mask_stays(self) -> None:
        image = _image()
        dark = _box(image, 40, 56, 40, 56, 40)
        far = np.zeros_like(dark)
        far[4:12, 4:12] = 1
        kept = drop_faint_unseen_masks([dark], [far], [far], [far], image, FLOOR)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], dark))

    def test_a_supported_faint_mask_stays(self) -> None:
        image = _image()
        faint = _box(image, 40, 56, 40, 56, 176)
        touch = faint.copy()
        kept = drop_faint_unseen_masks([faint], [touch], [touch], [touch], image, FLOOR)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], faint))


if __name__ == "__main__":
    unittest.main()
