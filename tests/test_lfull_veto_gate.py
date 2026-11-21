"""The full-medoid veto promotes only when the public forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_lfull_veto.py"
    spec = importlib.util.spec_from_file_location("eval_stream_lfull_veto", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LfullVetoGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.LFULL.name, "yolo11l_seg_full_2048_fold0_conf0.35_pred.csv")
        self.assertEqual(gate.EXPECTED_BRIDGE_PQ, 0.46863305638589764)
        self.assertFalse(gate.should_promote(0.46863305638589764))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertIn("drop_missed_by_donor", Path(gate.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
