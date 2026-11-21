"""A small mask is dropped only when no partner covers half of it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.bare_drop import SMALL_BARE, drop_small_uncovered


class BareDropTest(unittest.TestCase):
    def test_a_small_mask_outside_every_partner_is_dropped(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0:4, 0:4] = 1
        partner = np.zeros((32, 32), dtype=np.uint8)
        partner[20:28, 20:28] = 1
        self.assertLess(int(mask.sum()), SMALL_BARE)
        kept, dropped = drop_small_uncovered([mask], [partner])
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [])

    def test_a_small_mask_half_inside_a_partner_stays(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0:4, 0:4] = 1
        partner = mask.copy()
        kept, dropped = drop_small_uncovered([mask], [partner])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], mask))

    def test_a_large_uncovered_mask_stays(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[0:20, 0:20] = 1
        self.assertGreaterEqual(int(mask.sum()), SMALL_BARE)
        kept, dropped = drop_small_uncovered([mask], [])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
