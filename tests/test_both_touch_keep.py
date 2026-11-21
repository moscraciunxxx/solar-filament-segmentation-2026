"""An unseen mask stays only when two extra models both share a pixel."""

from __future__ import annotations

import unittest

import numpy as np

from filament.outside_band import drop_unseen_unless_both_touch


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class BothTouchKeepTest(unittest.TestCase):
    def test_one_extra_model_is_not_enough_to_keep_an_unseen_mask(self) -> None:
        base = _box(2, 10, 2, 10)
        far = _box(20, 28, 20, 28)
        touch = _box(8, 14, 8, 14)
        kept = drop_unseen_unless_both_touch([base], [far], [far], [far], [touch], [far])
        self.assertEqual(kept, [])

    def test_both_extra_models_keep_the_unseen_mask(self) -> None:
        base = _box(2, 10, 2, 10)
        far = _box(20, 28, 20, 28)
        touch = _box(8, 14, 8, 14)
        kept = drop_unseen_unless_both_touch([base], [far], [far], [far], [touch], [touch])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_a_witnessed_mask_stays_when_both_extra_models_miss(self) -> None:
        base = _box(2, 12, 2, 12)
        touch = _box(8, 16, 8, 16)
        far = _box(20, 28, 20, 28)
        kept = drop_unseen_unless_both_touch([base], [touch], [far], [far], [far], [far])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_neither_extra_model_is_appended(self) -> None:
        base = _box(2, 10, 2, 10)
        touch = _box(8, 14, 8, 14)
        extra = _box(18, 28, 18, 28)
        kept = drop_unseen_unless_both_touch([base], [touch], [extra], [extra], [extra], [extra])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))


if __name__ == "__main__":
    unittest.main()
