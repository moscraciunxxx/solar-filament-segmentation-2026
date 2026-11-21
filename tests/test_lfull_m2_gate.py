"""The stride-2 full-medoid score promotes only when the forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_lfull_m2.py"
    spec = importlib.util.spec_from_file_location("eval_stream_lfull_m2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LfullM2GateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.FAMILY, "yolo11l_seg_full_2048_m2")
        self.assertEqual(gate.CONF, 0.35)
        self.assertEqual(gate.IMGSZ, 2048)
        self.assertEqual(gate.WEIGHTS.name, "best.pt")
        self.assertFalse(gate.should_promote(0.42518))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))

    def test_written_score_obeys_the_shipped_gate(self) -> None:
        import json

        gate = _load()
        payload = json.loads(gate.OUT.read_text())
        row = payload["lfull_m2"]
        pq = row["pq"]
        self.assertEqual(payload["promote"], gate.should_promote(pq))
        self.assertIs(payload["test_labels_used"], False)
        self.assertIs(payload["kaggle_submit_called"], False)
        self.assertEqual(set(row) >= {"pq", "tp", "fp", "fn"}, True)
        self.assertNotIn("mask_map", payload)
        self.assertNotIn("mAP", payload)
        self.assertAlmostEqual(payload["formula_pq"], pq, places=9)


if __name__ == "__main__":
    unittest.main()
