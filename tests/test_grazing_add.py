"""A donor is added only when it grazes the parent set."""

from __future__ import annotations

import unittest

import numpy as np

from filament.grazing_add import HI, LO, add_grazing_donor


class GrazingAddTest(unittest.TestCase):
    def test_a_disjoint_donor_is_not_added(self) -> None:
        base = np.zeros((16, 16), dtype=np.uint8)
        base[0:8, 0:8] = 1
        donor = np.zeros((16, 16), dtype=np.uint8)
        donor[10:14, 10:14] = 1
        kept, added = add_grazing_donor([base], [donor])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)

    def test_a_light_overlap_is_appended(self) -> None:
        base = np.zeros((20, 20), dtype=np.uint8)
        base[0:10, 0:10] = 1
        donor = np.zeros((20, 20), dtype=np.uint8)
        donor[8:13, 0:10] = 1
        kept, added = add_grazing_donor([base], [donor])
        self.assertEqual(added, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], base))
        inter = int(np.logical_and(base, donor).sum())
        union = int(np.logical_or(base, donor).sum())
        overlap = inter / union
        self.assertGreater(overlap, LO)
        self.assertLess(overlap, HI)

    def test_a_near_copy_is_not_added(self) -> None:
        base = np.zeros((12, 12), dtype=np.uint8)
        base[2:10, 2:10] = 1
        donor = base.copy()
        donor[10, 2:10] = 1
        kept, added = add_grazing_donor([base], [donor])
        self.assertEqual(added, 0)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
