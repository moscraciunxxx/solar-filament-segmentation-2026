"""An 8-pixel tip is kept only where the partner already marks it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.tips import TIP_PX, extend_tips_inside


class TipInsideTest(unittest.TestCase):
    def test_a_partner_that_misses_the_mask_changes_nothing(self) -> None:
        mask = np.zeros((48, 180), dtype=np.uint8)
        mask[20:26, 40:100] = 1
        far = np.zeros_like(mask)
        far[20:26, 140:170] = 1
        kept = extend_tips_inside(mask, far)
        self.assertTrue(np.array_equal(kept, mask))

    def test_only_partner_pixels_are_added_at_the_tips(self) -> None:
        mask = np.zeros((48, 180), dtype=np.uint8)
        mask[20:26, 40:100] = 1
        partner = np.zeros_like(mask)
        partner[20:26, 40:100 + TIP_PX] = 1
        kept = extend_tips_inside(mask, partner, TIP_PX)
        added = (kept > 0) & (mask == 0)
        self.assertGreater(int(added.sum()), 0)
        self.assertTrue(bool(np.all(partner[added])))
        self.assertEqual(int(kept[22, 40 - TIP_PX]), 0)
        self.assertEqual(int(kept[22, 99 + TIP_PX]), 1)

    def test_a_mask_already_equal_to_its_partner_stays(self) -> None:
        mask = np.zeros((48, 180), dtype=np.uint8)
        mask[20:26, 40:100] = 1
        kept = extend_tips_inside(mask, mask.copy(), TIP_PX)
        self.assertTrue(np.array_equal(kept, mask))


if __name__ == "__main__":
    unittest.main()
