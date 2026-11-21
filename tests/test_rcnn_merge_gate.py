"""The Mask R-CNN container merge promotes only when the forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_rcnn_merge.py"
    spec = importlib.util.spec_from_file_location("eval_stream_rcnn_merge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RcnnMergeGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.RCNN.name, "maskrcnn_r50_1024_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.EXPECTED_LARGE_PQ, 0.46860702795003834)
        self.assertEqual(gate.CONTAIN, 0.5)
        self.assertFalse(gate.should_promote(0.46861))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))


if __name__ == "__main__":
    unittest.main()
