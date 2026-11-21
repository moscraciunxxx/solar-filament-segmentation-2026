"""A thin local dip is proposed. A wide sunspot center and a parent filament are not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.local_dip import WINDOW, propose_dip_rles
from filament.paths import IMAGE_SIZE
from filament.rle import decode_rle, encode_rle


class LocalDipTest(unittest.TestCase):
    def test_thin_bar_kept_wide_spot_and_parent_dropped(self) -> None:
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 140, dtype=np.uint8)
        image[500:503, 400:560] = 90
        yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
        image[(yy - 1200) ** 2 + (xx - 1200) ** 2 <= 30**2] = 70
        parent = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        parent[800:840, 800:900] = 1
        rles = propose_dip_rles(image, [encode_rle(parent)])
        masks = [decode_rle(rle) for rle in rles]
        self.assertEqual(WINDOW, 31)
        self.assertTrue(any(int(mask[501, 480]) == 1 for mask in masks))
        self.assertTrue(all(int(mask[1200, 1200]) == 0 for mask in masks))
        self.assertTrue(all(int(np.count_nonzero(mask & parent)) == 0 for mask in masks))


if __name__ == "__main__":
    unittest.main()
