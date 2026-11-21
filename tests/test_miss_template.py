"""A copied filament patch is found; the holdout cut prefers the quieter score."""

from __future__ import annotations

import unittest

import numpy as np

from filament.miss_template import (
    PATCH_H,
    PATCH_W,
    bank_threshold,
    oriented_patch,
    propose_template_masks,
)


class MissTemplateTest(unittest.TestCase):
    def test_threshold_uses_the_higher_holdout_cut(self) -> None:
        easy = bank_threshold(np.full(10, 0.8), np.full(20, 0.2))
        overlap = bank_threshold(np.full(10, 0.3), np.full(20, 0.6))
        self.assertAlmostEqual(easy, 0.8)
        self.assertAlmostEqual(overlap, 0.6)

    def test_patch_is_horizontal_64_by_32(self) -> None:
        image = np.full((80, 80), 100, dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:70, 36:44] = 1
        image[mask > 0] = 40
        patch, patch_mask = oriented_patch(image, mask)
        self.assertEqual(patch.shape, (PATCH_H, PATCH_W))
        self.assertGreater(int(patch_mask.sum()), 10)
        self.assertLess(int(patch[patch_mask > 0].mean()), 80)

    def test_exact_patch_is_detected_and_blocked_by_parent(self) -> None:
        patch = np.full((16, 40), 80, dtype=np.float32)
        patch[6:10, :] = 40
        mask = np.zeros((16, 40), dtype=np.uint8)
        mask[6:10, :] = 1
        image = np.full((128, 128), 80, dtype=np.float32)
        image[50:66, 30:70] = patch
        found = propose_template_masks(image, [(patch, mask)], threshold=0.8)
        self.assertEqual(len(found), 1)
        self.assertGreater(int(found[0][50:66, 30:70].sum()), 20)
        occupied = np.zeros_like(image, dtype=np.uint8)
        occupied[50:66, 30:70] = 1
        self.assertEqual(propose_template_masks(image, [(patch, mask)], 0.8, occupied), [])


if __name__ == "__main__":
    unittest.main()
