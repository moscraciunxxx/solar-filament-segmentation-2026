"""A mask takes the full-medoid outline only when the two already agree."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreed_outline import IOU_CUT, swap_agreed_outline


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class AgreedOutlineTest(unittest.TestCase):
    def test_an_overlapping_partner_replaces_the_base(self) -> None:
        base = _box(10, 40, 10, 40)
        partner = _box(10, 40, 16, 46)
        swapped, n_swapped = swap_agreed_outline([base], [partner])
        self.assertEqual(n_swapped, 1)
        self.assertTrue(np.array_equal(swapped[0], partner))
        self.assertEqual(IOU_CUT, 0.5)

    def test_iou_at_one_half_stays_the_base(self) -> None:
        base = _box(0, 30, 0, 30)
        partner = _box(0, 30, 10, 40)
        swapped, n_swapped = swap_agreed_outline([base], [partner])
        self.assertEqual(n_swapped, 0)
        self.assertTrue(np.array_equal(swapped[0], base))

    def test_a_partner_with_no_overlap_is_not_appended(self) -> None:
        base = _box(0, 20, 0, 20)
        far = _box(40, 60, 40, 60)
        swapped, n_swapped = swap_agreed_outline([base], [far])
        self.assertEqual(n_swapped, 0)
        self.assertEqual(len(swapped), 1)
        self.assertTrue(np.array_equal(swapped[0], base))

    def test_one_partner_goes_to_the_higher_overlap(self) -> None:
        high = _box(10, 40, 12, 42)
        low = _box(10, 40, 8, 38)
        partner = _box(10, 40, 16, 46)
        swapped, n_swapped = swap_agreed_outline([low, high], [partner])
        self.assertEqual(n_swapped, 1)
        self.assertTrue(np.array_equal(swapped[0], low))
        self.assertTrue(np.array_equal(swapped[1], partner))


if __name__ == "__main__":
    unittest.main()
