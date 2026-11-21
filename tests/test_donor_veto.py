"""A mask stays unless the donor shares no pixel with it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.donor_veto import drop_missed_by_donor


class DonorVetoTest(unittest.TestCase):
    def test_a_mask_the_donor_misses_is_dropped(self) -> None:
        base = np.zeros((8, 8), dtype=np.uint8)
        base[1:3, 1:3] = 1
        donor = np.zeros((8, 8), dtype=np.uint8)
        donor[5:7, 5:7] = 1
        kept, dropped = drop_missed_by_donor([base], [donor])
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [])

    def test_one_shared_pixel_keeps_the_mask(self) -> None:
        base = np.zeros((8, 8), dtype=np.uint8)
        base[1:4, 1:4] = 1
        donor = np.zeros((8, 8), dtype=np.uint8)
        donor[3:6, 3:6] = 1
        kept, dropped = drop_missed_by_donor([base], [donor])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_nothing_is_added_when_the_donor_is_empty(self) -> None:
        base = np.zeros((4, 4), dtype=np.uint8)
        base[0, 0] = 1
        kept, dropped = drop_missed_by_donor([base], [])
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
