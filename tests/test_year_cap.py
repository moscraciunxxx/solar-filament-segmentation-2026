"""Fold 0 does not set the yearly count cap, and extras drop smallest-first."""

from __future__ import annotations

import unittest

import numpy as np

from filament.coco_io import AnnotatorView, Filament
from filament.year_cap import drop_smallest, year_count_caps


def _view(image_id: str, file_name: str, n_filaments: int) -> AnnotatorView:
    view = AnnotatorView(image_id=image_id, file_name=file_name, annotator=image_id.split("-")[0])
    view.filaments = [
        Filament(
            polygon=np.zeros((3, 2), dtype=np.float32),
            bbox=(0, 0, 1, 1),
            area=1,
            category_id=1,
        )
        for _ in range(n_filaments)
    ]
    return view


class YearCapTest(unittest.TestCase):
    def test_fold_zero_count_does_not_raise_the_cap(self) -> None:
        views = {
            "a-1": _view("a-1", "2011a.jpg", 2),
            "c-1": _view("c-1", "2011c.jpg", 4),
            "b-1": _view("b-1", "2011b.jpg", 40),
        }
        by_file = {"2011a.jpg": ["a-1"], "2011c.jpg": ["c-1"], "2011b.jpg": ["b-1"]}
        fold_of = {"2011a.jpg": 1, "2011c.jpg": 1, "2011b.jpg": 0}
        caps, default = year_count_caps(views, by_file, fold_of)
        self.assertEqual(caps[2011], 3)
        self.assertLess(default, 40)

    def test_smallest_masks_are_dropped_first(self) -> None:
        small = np.zeros((8, 8), dtype=np.uint8)
        small[0:2, 0:2] = 1
        large = np.zeros((8, 8), dtype=np.uint8)
        large[0:6, 0:6] = 1
        kept = drop_smallest([small, large], cap=1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(int(kept[0].sum()), int(large.sum()))


if __name__ == "__main__":
    unittest.main()
