"""Only pixel-disjoint masks inside one instance are joined."""

from __future__ import annotations

import unittest

import numpy as np

from filament.partner_merge import CONTAIN, merge_disjoint_inside_partner


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class DisjointMergeTest(unittest.TestCase):
    def test_two_disjoint_masks_inside_one_partner_merge(self) -> None:
        partner = _box(8, 56, 8, 56)
        left = _box(16, 36, 12, 28)
        right = _box(16, 36, 36, 52)
        far = _box(0, 6, 0, 6)
        merged, n_merged = merge_disjoint_inside_partner([left, far, right], [partner])
        self.assertEqual(CONTAIN, 0.5)
        self.assertEqual(n_merged, 1)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.array_equal(merged[0], np.maximum(left, right)))
        self.assertTrue(np.array_equal(merged[1], far))

    def test_overlapping_masks_inside_one_partner_stay_apart(self) -> None:
        partner = _box(8, 56, 8, 56)
        left = _box(16, 36, 12, 28)
        overlap = _box(16, 36, 20, 40)
        merged, n_merged = merge_disjoint_inside_partner([left, overlap], [partner])
        self.assertEqual(n_merged, 0)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.array_equal(merged[0], left))
        self.assertTrue(np.array_equal(merged[1], overlap))

    def test_singletons_merge_while_an_overlapping_pair_stays(self) -> None:
        partner = _box(0, 60, 0, 60)
        left = _box(4, 20, 4, 20)
        overlap = _box(4, 20, 12, 28)
        first = _box(40, 52, 4, 16)
        second = _box(40, 52, 40, 52)
        merged, n_merged = merge_disjoint_inside_partner(
            [left, overlap, first, second],
            [partner],
        )
        self.assertEqual(n_merged, 1)
        self.assertEqual(len(merged), 3)
        self.assertTrue(np.array_equal(merged[0], left))
        self.assertTrue(np.array_equal(merged[1], overlap))
        self.assertTrue(np.array_equal(merged[2], np.maximum(first, second)))

    def test_a_mask_less_than_half_inside_is_not_joined(self) -> None:
        partner = _box(10, 40, 0, 40)
        inside = _box(12, 32, 4, 24)
        partial = _box(12, 32, 32, 52)
        merged, n_merged = merge_disjoint_inside_partner([inside, partial], [partner])
        self.assertEqual(n_merged, 0)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.array_equal(merged[0], inside))
        self.assertTrue(np.array_equal(merged[1], partial))


if __name__ == "__main__":
    unittest.main()
