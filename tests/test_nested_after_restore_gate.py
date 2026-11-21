"""The second interior pass promotes only when the public forecast is above 0.6."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_nested_after_restore.py"
    spec = importlib.util.spec_from_file_location("eval_stream_nested_after_restore", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NestedAfterRestoreGateTest(unittest.TestCase):
    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _load()
        self.assertEqual(gate.EXPECTED_TRIPLE_PQ, 0.46289891835415004)
        self.assertFalse(gate.should_promote(0.46290))
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
