"""A component is added only when every donor marks it and the base misses it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreed_add import add_unanimous_masks
from filament.small_drop import MIN_AREA


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class UnanimousAddTest(unittest.TestCase):
    def test_two_of_three_donors_add_nothing(self) -> None:
        base = _box(0, 16, 0, 16)
        far = _box(40, 56, 40, 56)
        kept, added = add_unanimous_masks([base], [[far], [far], []])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_all_three_donors_add_a_component_the_base_misses(self) -> None:
        base = _box(0, 16, 0, 16)
        far = _box(40, 56, 40, 56)
        self.assertGreaterEqual(int(far.sum()), MIN_AREA)
        kept, added = add_unanimous_masks([base], [[far], [far], [far]])
        self.assertEqual(added, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], base))
        self.assertTrue(np.array_equal(kept[1], far))

    def test_agreement_that_touches_the_base_is_not_added(self) -> None:
        base = _box(0, 20, 0, 20)
        touch = _box(10, 30, 10, 30)
        kept, added = add_unanimous_masks([base], [[touch], [touch], [touch]])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_a_speck_under_96_pixels_is_not_added(self) -> None:
        base = _box(0, 16, 0, 16)
        speck = _box(40, 44, 40, 44)
        self.assertLess(int(speck.sum()), MIN_AREA)
        kept, added = add_unanimous_masks([base], [[speck], [speck], [speck]])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
