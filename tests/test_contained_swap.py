"""A strictly larger containing mask replaces one base and nothing else."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from filament.contained_swap import contained_swap_masks


def _gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_contained_swap.py"
    spec = importlib.util.spec_from_file_location("eval_stream_contained_swap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ContainedSwapTest(unittest.TestCase):
    def test_empty_base_stays_empty(self) -> None:
        partner = np.zeros((16, 16), dtype=np.uint8)
        partner[2:12, 2:12] = 1
        self.assertEqual(contained_swap_masks([], [partner]), [])

    def test_a_strict_superset_replaces_the_base(self) -> None:
        base = np.zeros((32, 32), dtype=np.uint8)
        base[10:20, 10:20] = 1
        partner = np.zeros((32, 32), dtype=np.uint8)
        partner[8:22, 8:22] = 1
        swapped = contained_swap_masks([base], [partner])[0]
        self.assertTrue(np.array_equal(swapped, partner))

    def test_a_partner_that_misses_the_base_is_ignored(self) -> None:
        base = np.zeros((32, 32), dtype=np.uint8)
        base[4:12, 4:12] = 1
        far = np.zeros((32, 32), dtype=np.uint8)
        far[18:30, 18:30] = 1
        swapped = contained_swap_masks([base], [far])[0]
        self.assertTrue(np.array_equal(swapped, base))

    def test_one_partner_replaces_only_the_larger_base(self) -> None:
        small = np.zeros((32, 40), dtype=np.uint8)
        large = np.zeros((32, 40), dtype=np.uint8)
        small[4:10, 4:10] = 1
        large[4:16, 20:36] = 1
        partner = np.ones((32, 40), dtype=np.uint8)
        swapped = contained_swap_masks([small, large], [partner])
        self.assertTrue(np.array_equal(swapped[1], partner))
        self.assertTrue(np.array_equal(swapped[0], small))

    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _gate()
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.45728))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
