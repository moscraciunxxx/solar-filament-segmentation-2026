"""Instance-wise undo of the locked 3x3 ellipse erosion."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import importlib.util
from pathlib import Path

from filament.restore_erosion import restore_eroded_masks
from filament.yolo_export import ERODE_GT_ITERS, ERODE_GT_K


def _gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_stream_restore_stack.py"
    spec = importlib.util.spec_from_file_location("eval_stream_restore_stack", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RestoreErosionTest(unittest.TestCase):
    def test_empty_input_stays_empty(self) -> None:
        self.assertEqual(restore_eroded_masks([]), [])

    def test_rim_comes_back_on_a_shape_the_locked_kernel_opened(self) -> None:
        original = np.zeros((64, 64), dtype=np.uint8)
        original[16:48, 12:52] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_GT_K, ERODE_GT_K))
        eroded = cv2.erode(original, kernel, iterations=ERODE_GT_ITERS)
        restored = restore_eroded_masks([eroded])[0]
        self.assertGreater(int(restored.sum()), int(eroded.sum()))
        self.assertTrue(np.all(restored[eroded > 0] == 1))
        inter = int(np.logical_and(restored > 0, original > 0).sum())
        union = int(np.logical_or(restored > 0, original > 0).sum())
        self.assertGreater(inter / union, 0.98)

    def test_cores_stay_and_rims_do_not_overlap(self) -> None:
        left = np.zeros((32, 64), dtype=np.uint8)
        right = np.zeros((32, 64), dtype=np.uint8)
        left[8:24, 8:30] = 1
        right[8:24, 31:54] = 1
        restored = restore_eroded_masks([left, right])
        self.assertTrue(np.all(restored[0][left > 0] == 1))
        self.assertTrue(np.all(restored[1][right > 0] == 1))
        self.assertEqual(int(np.logical_and(restored[0] > 0, restored[1] > 0).sum()), 0)
        self.assertGreater(int(restored[0].sum()), int(left.sum()))

    def test_promote_is_strictly_above_the_public_bar(self) -> None:
        gate = _gate()
        self.assertFalse(gate.should_promote(0.668))
        self.assertTrue(gate.should_promote(0.670))
        self.assertFalse(gate.should_promote(0.64440))
        self.assertEqual(
            gate.should_promote(0.670),
            (0.670 - gate.PUBLIC_GAP) > gate.PUBLIC_BAR,
        )


if __name__ == "__main__":
    unittest.main()
