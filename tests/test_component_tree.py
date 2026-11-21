"""Component-tree antichain. No images and no test labels."""

from __future__ import annotations

import unittest

import numpy as np

from filament.component_tree import area_prior, cap_count, fit_log_area_prior, select_antichain, tree_components


class ComponentTreeTest(unittest.TestCase):
    def test_antichain_prefers_the_higher_score(self) -> None:
        # Parent 1.0 versus two children at 0.8. The children win.
        chosen = select_antichain([1.0, 0.8, 0.8], [[1, 2], [], []], [0])
        self.assertEqual(chosen, [1, 2])
        chosen = select_antichain([5.0, 1.0, 1.0], [[1, 2], [], []], [0])
        self.assertEqual(chosen, [0])

    def test_prior_peaks_at_the_training_median(self) -> None:
        median, sigma = fit_log_area_prior([100.0, 400.0, 1600.0])
        self.assertAlmostEqual(median, 400.0)
        self.assertGreater(area_prior(400, median, sigma), area_prior(50, median, sigma))

    def test_tree_splits_a_dim_bridge(self) -> None:
        prob = np.zeros((40, 40), dtype=np.float32)
        prob[5:15, 5:15] = 0.9
        prob[5:15, 25:35] = 0.9
        prob[8:12, 15:25] = 0.2
        masks = tree_components(prob, median=100.0, sigma=0.5, levels=np.linspace(0.15, 0.8, 8), min_area=20)
        self.assertEqual(len(masks), 2)
        self.assertEqual(int(sum(int(mask.sum()) for mask in masks)), 200)

    def test_elongation_gate_drops_the_square(self) -> None:
        prob = np.zeros((40, 80), dtype=np.float32)
        prob[5:15, 5:15] = 0.9
        prob[18:24, 5:70] = 0.9
        masks = tree_components(
            prob,
            median=100.0,
            sigma=0.5,
            levels=np.array([0.5]),
            min_area=20,
            min_mean=0.5,
            min_elongation=3.0,
        )
        self.assertEqual(len(masks), 1)
        self.assertGreater(int(masks[0].sum()), 100)

    def test_cap_keeps_the_highest_score(self) -> None:
        masks = [np.zeros((2, 2), dtype=np.uint8) for _ in range(3)]
        kept = cap_count(masks, [0.2, 0.9, 0.4], cap=1)
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], masks[1])


if __name__ == "__main__":
    unittest.main()
