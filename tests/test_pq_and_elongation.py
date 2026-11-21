"""Official PQ identity and the pre-registered elongation cut."""

from __future__ import annotations

import unittest

import numpy as np

from filament.elongation import keep_elongated, mask_eccentricity
from filament.metric import pred_to_df, score_pq
from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle


def _rect(y0: int, x0: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 1
    return mask


class PqIdentityTest(unittest.TestCase):
    def test_formula_matches_score_pq(self) -> None:
        gt = pred_to_df("ann-img", [encode_rle(_rect(10, 10, 40, 12))])
        gt["filament_id"] = ["ann-img_1"]
        pred = pred_to_df("img", [encode_rle(_rect(10, 10, 40, 12)), encode_rle(_rect(400, 400, 20, 20))])
        first = score_pq(gt, pred)
        second = score_pq(gt, pred)
        denom = first.tp + 0.5 * first.fp + 0.5 * first.fn
        reconstructed = (first.mean_tp_iou * first.tp) / denom
        self.assertEqual((first.pq, first.tp, first.fp, first.fn), (second.pq, second.tp, second.fp, second.fn))
        self.assertAlmostEqual(first.pq, reconstructed, places=9)
        self.assertEqual(first.tp, 1)
        self.assertEqual(first.fp, 1)
        self.assertEqual(first.fn, 0)


class ElongationTest(unittest.TestCase):
    def test_line_kept_and_disk_dropped(self) -> None:
        line = _rect(100, 100, 8, 80)
        disk = _rect(800, 800, 40, 40)
        self.assertGreater(mask_eccentricity(line), 0.90)
        self.assertLess(mask_eccentricity(disk), 0.90)
        self.assertTrue(keep_elongated(line))
        self.assertFalse(keep_elongated(disk))
        tiny = _rect(2, 2, 3, 3)
        self.assertTrue(keep_elongated(tiny))


if __name__ == "__main__":
    unittest.main()
