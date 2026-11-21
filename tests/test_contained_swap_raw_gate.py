"""The faint/continued/raw-shift containment swap promotes only above a 0.6 forecast."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_contained_swap_raw.py"
    spec = importlib.util.spec_from_file_location("eval_stream_contained_swap_raw", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ContainedSwapRawGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.MIN_VOTES, 2)
        self.assertEqual(gate.MIN_AREA, 96)
        self.assertEqual(gate.RAW.name, "yolo11l_seg_shift_fold0_conf0.35_pred.csv")
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.45902))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
