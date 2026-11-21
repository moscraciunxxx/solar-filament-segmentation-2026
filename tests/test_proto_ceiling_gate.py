"""The prototype ceiling is a label round-trip and is never an upload."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_proto_ceiling.py"
    spec = importlib.util.spec_from_file_location("eval_stream_proto_ceiling", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProtoCeilingGateTest(unittest.TestCase):
    def test_a_high_score_still_does_not_upload(self) -> None:
        gate = _load()
        self.assertEqual(gate.RATIOS, (4, 2))
        self.assertFalse(gate.may_upload(0.90))
        self.assertFalse(gate.may_upload(0.670))
        self.assertIn("ceiling", gate.HYPOTHESIS)


if __name__ == "__main__":
    unittest.main()
