"""Pixels outside the covering partner are removed. Nothing is added."""

from __future__ import annotations

import unittest

import numpy as np

from filament.partner_clip import clip_outside_partner


class PartnerClipTest(unittest.TestCase):
    def test_a_spill_outside_a_covering_partner_is_removed(self) -> None:
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:9, 1:9] = 1
        partner = np.zeros((12, 12), dtype=np.uint8)
        partner[1:9, 1:7] = 1
        kept, changed = clip_outside_partner([mask], [partner])
        self.assertEqual(changed, 1)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], np.logical_and(mask, partner).astype(np.uint8)))
        self.assertLess(int(kept[0].sum()), int(mask.sum()))

    def test_a_mask_less_than_half_inside_is_unchanged(self) -> None:
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:11, 1:11] = 1
        partner = np.zeros((12, 12), dtype=np.uint8)
        partner[1:4, 1:4] = 1
        kept, changed = clip_outside_partner([mask], [partner])
        self.assertEqual(changed, 0)
        self.assertTrue(np.array_equal(kept[0], mask))

    def test_no_partner_pixel_is_added(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        partner = np.ones((8, 8), dtype=np.uint8)
        kept, changed = clip_outside_partner([mask], [partner])
        self.assertEqual(changed, 0)
        self.assertFalse(np.any((kept[0] > 0) & (mask == 0)))


if __name__ == "__main__":
    unittest.main()
