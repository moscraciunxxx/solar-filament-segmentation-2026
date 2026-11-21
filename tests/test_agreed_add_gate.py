"""The agreed leftover add promotes only when the public forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_agreed_add.py"
    spec = importlib.util.spec_from_file_location("eval_stream_agreed_add", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AgreedAddGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.LARGE.name, "yolo11l_seg_1280_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.THICK.name, "yolo11s_seg_1408_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.RCNN.name, "maskrcnn_r50_1024_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.EXPECTED_UNSEEN_PQ, 0.4620537818986917)
        self.assertEqual(gate.MIN_VOTES, 2)
        self.assertEqual(gate.MIN_AREA, 96)
        self.assertFalse(gate.should_promote(0.46205))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))


if __name__ == "__main__":
    unittest.main()
