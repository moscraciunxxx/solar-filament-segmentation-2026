"""Flat field, soft target, and ConvNeXt logit shape. No contest test images."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_soft_clv import cc_masks, choose_min_area, rles_by_min_area
from filament.flatfield import center_to_limb_flat
from filament.unet_convnext import ConvNeXtUNet, soft_bce_dice


class SoftClvTest(unittest.TestCase):
    def test_flat_field_zeros_the_corner_of_a_disk(self) -> None:
        image = np.zeros((256, 256), dtype=np.uint8)
        yy, xx = np.mgrid[0:256, 0:256]
        radius = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
        image[radius < 100] = 160
        image[radius < 40] = 220
        flat = center_to_limb_flat(image)
        self.assertEqual(flat.dtype, np.float32)
        self.assertGreater(float(flat[128, 128]), 0.0)
        self.assertEqual(float(flat[0, 0]), 0.0)
        self.assertTrue(np.isfinite(flat).all())

    def test_logits_match_the_input(self) -> None:
        model = ConvNeXtUNet(pretrained=False)
        images = torch.zeros(1, 3, 64, 64)
        self.assertEqual(tuple(model(images).shape), (1, 1, 64, 64))

    def test_threshold_cc_returns_one_blob(self) -> None:
        prob = np.zeros((64, 64), dtype=np.float32)
        prob[10:40, 12:50] = 0.9
        masks = cc_masks(prob, min_area=64)
        self.assertEqual(len(masks), 1)
        self.assertEqual(int(masks[0].sum()), 30 * 38)

    def test_min_area_grid_keeps_a_component_only_when_it_clears_the_cut(self) -> None:
        prob = np.zeros((80, 80), dtype=np.float32)
        prob[0:10, 0:30] = 0.9  # 300 px
        prob[20:40, 0:35] = 0.9  # 700 px
        prob[50:55, 0:10] = 0.9  # 50 px
        buckets = rles_by_min_area(prob, areas=(64, 250, 642, 1000))
        self.assertEqual([len(buckets[area]) for area in (64, 250, 642, 1000)], [2, 2, 1, 0])

    def test_min_area_tie_prefers_the_predeclared_cut(self) -> None:
        chosen = choose_min_area(
            [
                {"area": 64, "pq": 0.40},
                {"area": 642, "pq": 0.40},
                {"area": 1000, "pq": 0.20},
            ]
        )
        self.assertEqual(chosen["area"], 642)

    def test_soft_dice_accepts_a_fractional_target(self) -> None:
        logits = torch.zeros(1, 1, 8, 8)
        target = torch.full((1, 1, 8, 8), 0.4)
        loss = soft_bce_dice(logits, target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
