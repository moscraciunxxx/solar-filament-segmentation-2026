"""The large full-medoid score promotes only when the public forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_lfull_2048.py"
    spec = importlib.util.spec_from_file_location("eval_stream_lfull_2048", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LFull2048GateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.64440))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
