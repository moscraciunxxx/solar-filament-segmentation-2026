"""A strictly smaller interior mask replaces the base, and a mask that sticks out does not."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from filament.inner_swap import inner_swap_masks


def _gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_inner_swap.py"
    spec = importlib.util.spec_from_file_location("eval_stream_inner_swap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InnerSwapTest(unittest.TestCase):
    def test_empty_base_stays_empty(self) -> None:
        inner = np.zeros((16, 16), dtype=np.uint8)
        inner[4:10, 4:10] = 1
        self.assertEqual(inner_swap_masks([], [inner]), [])

    def test_a_strict_subset_replaces_the_base(self) -> None:
        base = np.zeros((32, 32), dtype=np.uint8)
        base[8:24, 8:24] = 1
        inner = np.zeros((32, 32), dtype=np.uint8)
        inner[10:20, 10:20] = 1
        swapped = inner_swap_masks([base], [inner])[0]
        self.assertTrue(np.array_equal(swapped, inner))

    def test_an_inner_mask_that_sticks_out_is_ignored(self) -> None:
        base = np.zeros((32, 32), dtype=np.uint8)
        base[8:16, 8:16] = 1
        stick = np.zeros((32, 32), dtype=np.uint8)
        stick[8:16, 8:20] = 1
        swapped = inner_swap_masks([base], [stick])[0]
        self.assertTrue(np.array_equal(swapped, base))

    def test_one_inner_mask_replaces_only_the_smaller_base(self) -> None:
        small = np.zeros((40, 40), dtype=np.uint8)
        large = np.zeros((40, 40), dtype=np.uint8)
        small[2:12, 2:12] = 1
        large[2:30, 20:38] = 1
        left = np.zeros((40, 40), dtype=np.uint8)
        left[4:8, 4:8] = 1
        swapped = inner_swap_masks([large, small], [left])
        self.assertTrue(np.array_equal(swapped[1], left))
        self.assertTrue(np.array_equal(swapped[0], large))

    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _gate()
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.45902))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
