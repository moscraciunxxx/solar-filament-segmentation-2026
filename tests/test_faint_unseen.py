"""A faint mask stays when yolo11n overlaps it, and goes when yolo11n does not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.faint_unseen import WITNESS_IOU, drop_faint_unseen
from filament.rle import encode_rle


def _paint(image: np.ndarray, x: int, y: int, value: int) -> str:
    image[y : y + 12, x : x + 12] = value
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y : y + 12, x : x + 12] = 1
    return encode_rle(mask)


class FaintUnseenTest(unittest.TestCase):
    def test_witness_cut_stays_one_half(self) -> None:
        self.assertEqual(WITNESS_IOU, 0.5)

    def test_faint_mask_is_dropped_only_without_a_witness(self) -> None:
        image = np.full((128, 128), 100, dtype=np.uint8)
        faint = _paint(image, 16, 16, 96)
        witnessed = _paint(image, 80, 16, 96)
        dark = _paint(image, 16, 80, 60)
        kept, dropped = drop_faint_unseen([faint, witnessed, dark], image, [witnessed], floor=10)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [witnessed, dark])


if __name__ == "__main__":
    unittest.main()
