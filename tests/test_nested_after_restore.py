"""A second strict-interior pass drops only a mask fully inside a sibling."""

from __future__ import annotations

import unittest

import numpy as np

from filament.interior_drop import drop_nested_after_restore


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class NestedAfterRestoreTest(unittest.TestCase):
    def test_a_restored_mask_strictly_inside_another_is_dropped(self) -> None:
        outer = _box(2, 20, 2, 20)
        inner = _box(6, 12, 6, 12)
        kept = drop_nested_after_restore([outer, inner])
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], outer))

    def test_a_partial_overlap_stays(self) -> None:
        left = _box(2, 16, 2, 16)
        right = _box(8, 22, 8, 22)
        kept = drop_nested_after_restore([left, right])
        self.assertEqual(len(kept), 2)

    def test_equal_masks_both_stay(self) -> None:
        one = _box(4, 14, 4, 14)
        other = one.copy()
        kept = drop_nested_after_restore([one, other])
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
