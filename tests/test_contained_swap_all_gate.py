"""The second containment swap promotes only above a 0.6 forecast."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_contained_swap_all.py"
    spec = importlib.util.spec_from_file_location("eval_stream_contained_swap_all", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ContainedSwapAllGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.ALL.name, "yolo11s_seg_all_fold0_conf0.40_pred.csv")
        self.assertEqual(gate.EXPECTED_VOTE_SWAP_PQ, 0.4590209919721129)
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.45902))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
