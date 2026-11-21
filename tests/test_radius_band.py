"""The radius band ignores fold 0, and an out-of-band mask is dropped."""

from __future__ import annotations

import unittest

import numpy as np

from filament.coco_io import AnnotatorView, Filament
from filament.radius_band import HIGH_PCT, LOW_PCT, drop_outside_band, train_radius_band


def _view(image_id: str, file_name: str, y: float, x: float) -> AnnotatorView:
    view = AnnotatorView(image_id=image_id, file_name=file_name, annotator=image_id.split("-")[0])
    poly = np.array([[x, y], [x + 2, y], [x, y + 2]], dtype=np.float32)
    view.filaments = [Filament(polygon=poly, bbox=(x, y, 2, 2), area=3, category_id=1)]
    return view


class RadiusBandTest(unittest.TestCase):
    def test_fold_zero_does_not_widen_the_band(self) -> None:
        views = {
            "a-1": _view("a-1", "a.jpg", 1024, 1124),
            "b-1": _view("b-1", "b.jpg", 10, 10),
        }
        by_file = {"a.jpg": ["a-1"], "b.jpg": ["b-1"]}
        fold_of = {"a.jpg": 1, "b.jpg": 0}
        lo, hi = train_radius_band(views, by_file, fold_of)
        self.assertEqual(LOW_PCT, 1)
        self.assertEqual(HIGH_PCT, 99)
        self.assertLess(hi, 500)

    def test_corner_mask_dropped_center_mask_kept(self) -> None:
        center = np.zeros((2048, 2048), dtype=np.uint8)
        center[1020:1028, 1020:1028] = 1
        corner = np.zeros((2048, 2048), dtype=np.uint8)
        corner[0:8, 0:8] = 1
        kept = drop_outside_band([center, corner], lo=0, hi=200)
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept[0][1024, 1024]), 1)


if __name__ == "__main__":
    unittest.main()
