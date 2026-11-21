"""A unanimous core grows only into the two-donor component that contains it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreed_add import extend_core_to_pair_fringe
from filament.vote import MIN_VOTES


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class CoreFringeTest(unittest.TestCase):
    def test_a_core_grows_into_the_pair_component_that_contains_it(self) -> None:
        wide = _box(20, 44, 8, 56)
        center = _box(20, 44, 24, 40)
        grown = extend_core_to_pair_fringe(center, [wide, wide, center], [])
        self.assertTrue(np.array_equal(grown, wide))
        self.assertEqual(MIN_VOTES, 2)

    def test_a_separate_pair_blob_is_not_taken(self) -> None:
        wide = _box(20, 44, 8, 56)
        center = _box(20, 44, 24, 40)
        far = _box(0, 16, 0, 16)
        pair = np.maximum(wide, far)
        grown = extend_core_to_pair_fringe(center, [pair, pair, center], [])
        self.assertTrue(np.array_equal(grown, wide))
        self.assertEqual(int(np.logical_and(grown, far).sum()), 0)

    def test_pixels_in_a_base_mask_are_not_taken(self) -> None:
        wide = _box(20, 44, 8, 56)
        center = _box(20, 44, 24, 40)
        blocked = _box(20, 44, 8, 24)
        grown = extend_core_to_pair_fringe(center, [wide, wide, center], [blocked])
        expected = wide.copy()
        expected[20:44, 8:24] = 0
        self.assertTrue(np.array_equal(grown, expected))
        self.assertEqual(int(np.logical_and(grown, blocked).sum()), 0)

    def test_a_core_with_no_larger_fringe_stays(self) -> None:
        core = _box(20, 44, 24, 40)
        grown = extend_core_to_pair_fringe(core, [core, core, core], [])
        self.assertTrue(np.array_equal(grown, core))


if __name__ == "__main__":
    unittest.main()
