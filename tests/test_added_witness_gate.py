"""The thick-model filter on added masks promotes only above a 0.6 forecast."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_added_witness.py"
    spec = importlib.util.spec_from_file_location("eval_stream_added_witness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AddedWitnessGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.THICK.name, "yolo11s_seg_1408_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.EXPECTED_UNANIMOUS_PQ, 0.4639777995958413)
        self.assertEqual(gate.MIN_AREA, 96)
        self.assertFalse(gate.should_promote(0.46398))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
