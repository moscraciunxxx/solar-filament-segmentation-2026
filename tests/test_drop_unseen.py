"""A mask nobody else sees is dropped. Nothing is added."""

from __future__ import annotations

import unittest

import numpy as np

from filament.outside_band import drop_unseen_masks


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class DropUnseenTest(unittest.TestCase):
    def test_a_mask_missed_by_all_three_is_dropped(self) -> None:
        base = _box(2, 8, 2, 8)
        far = _box(20, 28, 20, 28)
        self.assertEqual(drop_unseen_masks([base], [far], [far], [far]), [])

    def test_a_touch_from_any_model_keeps_the_mask(self) -> None:
        base = _box(2, 12, 2, 12)
        touch = _box(8, 16, 8, 16)
        far = _box(20, 28, 20, 28)
        kept = drop_unseen_masks([base], [touch], [far], [far])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_an_all_annotator_extra_is_not_added(self) -> None:
        base = _box(2, 10, 2, 10)
        touch = _box(8, 14, 8, 14)
        extra = _box(18, 28, 18, 28)
        kept = drop_unseen_masks([base], [touch, extra], [extra], [extra])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))


if __name__ == "__main__":
    unittest.main()
