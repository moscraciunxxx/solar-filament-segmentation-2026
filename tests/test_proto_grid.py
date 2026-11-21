"""The prototype round-trip matches Ultralytics resize, including thin-mask loss."""

from __future__ import annotations

import unittest

import numpy as np

from filament.proto_grid import roundtrip_mask


class ProtoGridTest(unittest.TestCase):
    def test_ratio_one_keeps_a_binary_mask(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        self.assertTrue(np.array_equal(roundtrip_mask(mask, 1), mask))

    def test_a_one_pixel_line_vanishes_at_ratio_four(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, 32] = 1
        self.assertEqual(int(roundtrip_mask(mask, 4).sum()), 0)

    def test_a_wide_bar_survives_ratio_four(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, 24:40] = 1
        kept = roundtrip_mask(mask, 4)
        self.assertEqual(int(kept[32, 32]), 1)
        self.assertEqual(int(kept[0, 0]), 0)

    def test_ratio_below_one_is_rejected(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        with self.assertRaises(ValueError):
            roundtrip_mask(mask, 0)


if __name__ == "__main__":
    unittest.main()
