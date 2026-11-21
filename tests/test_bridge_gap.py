"""A merged pair gains only the partner pixels both pieces can reach."""

from __future__ import annotations

import unittest

import numpy as np

from filament.bridge_gap import BRIDGE_PX, bridge_inside_partner, bridge_masks


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class BridgeGapTest(unittest.TestCase):
    def test_a_short_gap_inside_the_partner_is_filled(self) -> None:
        left = _box(20, 36, 8, 20)
        right = _box(20, 36, 26, 38)
        union = np.maximum(left, right)
        partner = _box(10, 50, 0, 50)
        bridged = bridge_inside_partner(union, partner)
        self.assertEqual(BRIDGE_PX, 8)
        self.assertEqual(int(bridged[28, 22]), 1)
        self.assertEqual(int(bridged[28, 4]), 0)
        self.assertGreater(int(bridged.sum()), int(union.sum()))
        self.assertLess(int(bridged.sum()), int(partner.sum()))

    def test_a_wide_gap_stays_open(self) -> None:
        left = _box(20, 36, 4, 12)
        right = _box(20, 36, 42, 50)
        union = np.maximum(left, right)
        partner = _box(10, 50, 0, 60)
        bridged = bridge_inside_partner(union, partner)
        self.assertTrue(np.array_equal(bridged, union))

    def test_one_piece_stays_unchanged(self) -> None:
        only = _box(20, 36, 8, 40)
        partner = _box(10, 50, 0, 50)
        bridged = bridge_inside_partner(only, partner)
        self.assertTrue(np.array_equal(bridged, only))

    def test_only_the_mask_inside_the_partner_is_bridged(self) -> None:
        left = _box(20, 36, 8, 20)
        right = _box(20, 36, 26, 38)
        union = np.maximum(left, right)
        alone = _box(4, 12, 4, 16)
        partner = _box(10, 50, 0, 50)
        bridged, n_bridged = bridge_masks([union, alone], [partner])
        self.assertEqual(n_bridged, 1)
        self.assertEqual(len(bridged), 2)
        self.assertEqual(int(bridged[0][28, 22]), 1)
        self.assertTrue(np.array_equal(bridged[1], alone))

    def test_a_partner_that_misses_the_gap_adds_nothing(self) -> None:
        left = _box(20, 36, 8, 20)
        right = _box(20, 36, 26, 38)
        union = np.maximum(left, right)
        partner = union.copy()
        bridged = bridge_inside_partner(union, partner)
        self.assertTrue(np.array_equal(bridged, union))


if __name__ == "__main__":
    unittest.main()
