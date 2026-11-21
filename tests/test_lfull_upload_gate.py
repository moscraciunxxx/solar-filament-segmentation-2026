"""A full-medoid upload needs a forecast above 0.6 and two matching public reads."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "lfull_upload_gate.py"
    spec = importlib.util.spec_from_file_location("lfull_upload_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LFullUploadGateTest(unittest.TestCase):
    def test_only_a_promoted_forecast_above_0_6_may_upload(self) -> None:
        gate = _load()
        self.assertEqual(gate.upload_decision(None), "await_score")
        self.assertEqual(
            gate.upload_decision({"public_forecast": 0.388, "promote": False}),
            "start_m2",
        )
        self.assertEqual(
            gate.upload_decision({"public_forecast": 0.6, "promote": False}),
            "start_m2",
        )
        self.assertEqual(
            gate.upload_decision({"public_forecast": 0.67, "promote": True}),
            "upload",
        )

    def test_two_reads_must_agree_and_clear_0_6(self) -> None:
        gate = _load()
        row = {"ref": 56456485, "status": "COMPLETE", "publicScore": 0.61}
        other = {"ref": 56456485, "status": "COMPLETE", "publicScore": 0.37}
        pending = {"ref": 56456485, "status": "PENDING", "publicScore": 0.61}
        self.assertEqual(gate.confirmed_public_score([row], [row], 56456485), 0.61)
        self.assertIsNone(gate.confirmed_public_score([row], [other], 56456485))
        self.assertIsNone(gate.confirmed_public_score([pending], [pending], 56456485))
        low = {"ref": 1, "status": "COMPLETE", "publicScore": 0.6}
        self.assertIsNone(gate.confirmed_public_score([low], [low], 1))


if __name__ == "__main__":
    unittest.main()
