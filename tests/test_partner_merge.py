"""Two masks inside one full-medoid instance become their union."""

from __future__ import annotations

import unittest

import numpy as np

from filament.partner_merge import CONTAIN, merge_inside_partner


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class PartnerMergeTest(unittest.TestCase):
    def test_two_masks_inside_one_partner_become_their_union(self) -> None:
        partner = _box(8, 56, 8, 56)
        left = _box(16, 36, 12, 28)
        right = _box(16, 36, 36, 52)
        far = _box(0, 6, 0, 6)
        merged, n_merged = merge_inside_partner([left, far, right], [partner])
        self.assertEqual(CONTAIN, 0.5)
        self.assertEqual(n_merged, 1)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.array_equal(merged[0], np.maximum(left, right)))
        self.assertTrue(np.array_equal(merged[1], far))
        self.assertGreater(int(partner.sum()), int(merged[0].sum()))

    def test_one_mask_inside_a_partner_stays(self) -> None:
        partner = _box(8, 56, 8, 56)
        left = _box(16, 36, 12, 28)
        merged, n_merged = merge_inside_partner([left], [partner])
        self.assertEqual(n_merged, 0)
        self.assertEqual(len(merged), 1)
        self.assertTrue(np.array_equal(merged[0], left))

    def test_separate_partners_do_not_merge(self) -> None:
        left = _box(16, 36, 12, 28)
        right = _box(16, 36, 36, 52)
        merged, n_merged = merge_inside_partner([left, right], [left, right])
        self.assertEqual(n_merged, 0)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.array_equal(merged[0], left))
        self.assertTrue(np.array_equal(merged[1], right))

    def test_a_mask_less_than_half_inside_stays_out(self) -> None:
        partner = _box(10, 40, 0, 40)
        inside = _box(12, 32, 4, 24)
        partial = _box(12, 32, 32, 52)
        merged, n_merged = merge_inside_partner([inside, partial], [partner])
        self.assertEqual(n_merged, 0)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.array_equal(merged[0], inside))
        self.assertTrue(np.array_equal(merged[1], partial))

    def test_three_masks_inside_one_partner_become_one(self) -> None:
        partner = _box(0, 60, 0, 60)
        first = _box(4, 16, 4, 16)
        second = _box(4, 16, 24, 36)
        third = _box(4, 16, 44, 56)
        merged, n_merged = merge_inside_partner([first, second, third], [partner])
        self.assertEqual(n_merged, 1)
        self.assertEqual(len(merged), 1)
        self.assertTrue(np.array_equal(merged[0], np.maximum(np.maximum(first, second), third)))


if __name__ == "__main__":
    unittest.main()
