"""The area cap ignores fold 0, and a mask above the cap is dropped."""

from __future__ import annotations

import unittest

import numpy as np

from filament.area_cap import PERCENTILE, drop_over_cap, train_area_cap
from filament.coco_io import AnnotatorView, Filament


def _view(image_id: str, file_name: str, area: float) -> AnnotatorView:
    view = AnnotatorView(image_id=image_id, file_name=file_name, annotator=image_id.split("-")[0])
    view.filaments = [
        Filament(polygon=np.zeros((3, 2), dtype=np.float32), bbox=(0, 0, 1, 1), area=area, category_id=1)
    ]
    return view


class AreaCapTest(unittest.TestCase):
    def test_fold_zero_area_does_not_set_the_cap(self) -> None:
        views = {
            "a-1": _view("a-1", "a.jpg", 10),
            "a-2": _view("a-2", "a.jpg", 30),
            "b-1": _view("b-1", "b.jpg", 1_000_000),
        }
        by_file = {"a.jpg": ["a-1", "a-2"], "b.jpg": ["b-1"]}
        fold_of = {"a.jpg": 1, "b.jpg": 0}
        cap = train_area_cap(views, by_file, fold_of)
        self.assertEqual(PERCENTILE, 99)
        self.assertLess(cap, 1_000)

    def test_mask_over_cap_is_dropped(self) -> None:
        small = np.zeros((8, 8), dtype=np.uint8)
        small[1:3, 1:4] = 1
        large = np.ones((8, 8), dtype=np.uint8)
        kept = drop_over_cap([small, large], cap=10)
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept[0].sum()), int(small.sum()))


if __name__ == "__main__":
    unittest.main()
