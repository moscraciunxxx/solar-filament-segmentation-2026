"""The unanimous leftover add promotes only when the public forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_unanimous_add.py"
    spec = importlib.util.spec_from_file_location("eval_stream_unanimous_add", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UnanimousAddGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.LARGE.name, "yolo11l_seg_1280_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.RCNN.name, "maskrcnn_r50_1024_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.FULL.name, "yolo11s_seg_full_2048_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.EXPECTED_TRIPLE_PQ, 0.46289891835415004)
        self.assertEqual(gate.MIN_AREA, 96)
        self.assertFalse(gate.should_promote(0.46290))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
