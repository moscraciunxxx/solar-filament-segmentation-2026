"""A faint added mask is dropped. A faint base mask stays."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreed_add import drop_faint_adds

FLOOR = 8.381084949352285


def _image() -> np.ndarray:
    return np.full((96, 96), 180, dtype=np.uint8)


def _box(image: np.ndarray, y0: int, y1: int, x0: int, x1: int, gray: int) -> np.ndarray:
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    image[y0:y1, x0:x1] = gray
    return mask


class FaintAddsTest(unittest.TestCase):
    def test_a_faint_added_mask_is_dropped(self) -> None:
        image = _image()
        base = _box(image, 4, 20, 4, 20, 40)
        faint = _box(image, 40, 56, 40, 56, 176)
        kept, n_kept = drop_faint_adds([base], [faint], image, FLOOR)
        self.assertEqual(n_kept, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_a_dark_added_mask_stays(self) -> None:
        image = _image()
        base = _box(image, 4, 20, 4, 20, 40)
        dark = _box(image, 40, 56, 40, 56, 40)
        kept, n_kept = drop_faint_adds([base], [dark], image, FLOOR)
        self.assertEqual(n_kept, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[1], dark))

    def test_a_faint_base_mask_stays(self) -> None:
        image = _image()
        base = _box(image, 40, 56, 40, 56, 176)
        kept, n_kept = drop_faint_adds([base], [], image, FLOOR)
        self.assertEqual(n_kept, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))


if __name__ == "__main__":
    unittest.main()
