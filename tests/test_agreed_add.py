"""Agreed donor pixels are added only where the base file is empty."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreed_add import add_agreed_masks
from filament.vote import MIN_VOTES
from filament.small_drop import MIN_AREA


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class AgreedAddTest(unittest.TestCase):
    def test_two_donors_add_a_component_the_base_misses(self) -> None:
        base = _box(0, 16, 0, 16)
        far = _box(40, 56, 40, 56)
        kept, added = add_agreed_masks([base], [[far], [far], []])
        self.assertEqual(added, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], base))
        self.assertTrue(np.array_equal(kept[1], far))
        self.assertEqual(MIN_VOTES, 2)
        self.assertGreaterEqual(int(far.sum()), MIN_AREA)

    def test_one_donor_adds_nothing(self) -> None:
        base = _box(0, 16, 0, 16)
        far = _box(40, 56, 40, 56)
        kept, added = add_agreed_masks([base], [[far], [], []])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_an_agreed_component_that_touches_the_base_is_not_added(self) -> None:
        base = _box(0, 20, 0, 20)
        touch = _box(10, 30, 10, 30)
        kept, added = add_agreed_masks([base], [[touch], [touch], [touch]])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_a_speck_under_96_pixels_is_not_added(self) -> None:
        base = _box(0, 16, 0, 16)
        speck = _box(40, 44, 40, 44)
        self.assertLess(int(speck.sum()), MIN_AREA)
        kept, added = add_agreed_masks([base], [[speck], [speck], [speck]])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
