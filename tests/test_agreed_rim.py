"""A touching partner can fill the locked erosion rim and nothing farther."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import importlib.util
from pathlib import Path

from filament.agreed_rim import agreed_rim_masks
from filament.yolo_export import ERODE_GT_ITERS, ERODE_GT_K


def _gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_agreed_rim.py"
    spec = importlib.util.spec_from_file_location("eval_stream_agreed_rim", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module



class AgreedRimTest(unittest.TestCase):
    def test_empty_base_stays_empty(self) -> None:
        partner = np.zeros((32, 32), dtype=np.uint8)
        partner[4:20, 4:20] = 1
        self.assertEqual(agreed_rim_masks([], [partner]), [])

    def test_touching_partner_fills_the_rim_and_stops_there(self) -> None:
        original = np.zeros((64, 64), dtype=np.uint8)
        original[16:48, 12:52] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_GT_K, ERODE_GT_K))
        eroded = cv2.erode(original, kernel, iterations=ERODE_GT_ITERS)
        restored = agreed_rim_masks([eroded], [original])[0]
        self.assertGreater(int(restored.sum()), int(eroded.sum()))
        self.assertTrue(np.all(restored[eroded > 0] == 1))
        self.assertTrue(np.all(restored[original == 0] == 0))

    def test_a_partner_with_no_shared_pixel_is_ignored(self) -> None:
        base = np.zeros((32, 64), dtype=np.uint8)
        base[8:24, 4:20] = 1
        far = np.zeros((32, 64), dtype=np.uint8)
        far[8:24, 40:60] = 1
        restored = agreed_rim_masks([base], [far])[0]
        self.assertTrue(np.array_equal(restored, base))

    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _gate()
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.45728))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
