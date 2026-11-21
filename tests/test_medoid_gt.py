"""Medoid ground truth keeps one annotator and does not stack the others."""

from __future__ import annotations

import unittest

import numpy as np

from filament.coco_io import AnnotatorView, Filament
from filament.medoid_gt import medoid_gt_records


def _filament(x: int) -> Filament:
    polygon = np.array(
        [[x, 10], [x + 30, 10], [x + 30, 40], [x, 40]],
        dtype=np.float32,
    )
    return Filament(polygon=polygon, bbox=(x, 10, 30, 30), area=900, category_id=1)


def _view(image_id: str, x: int) -> AnnotatorView:
    view = AnnotatorView(image_id=image_id, file_name="20110101120000Mh.jpeg", annotator=image_id.split("-")[0])
    view.filaments = [_filament(x)]
    return view


class MedoidGtTest(unittest.TestCase):
    def test_one_annotator_is_returned_whole(self) -> None:
        views = {"a-sun": _view("a-sun", 100)}
        records = medoid_gt_records(views, ["a-sun"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["annotator_image"], "a-sun")
        self.assertEqual(records[0]["index"], 1)
        self.assertTrue(records[0]["rle"])

    def test_two_annotators_do_not_stack(self) -> None:
        views = {
            "a-sun": _view("a-sun", 100),
            "b-sun": _view("b-sun", 400),
        }
        records = medoid_gt_records(views, ["a-sun", "b-sun"])
        self.assertEqual(len(records), 1)
        self.assertIn(records[0]["annotator_image"], {"a-sun", "b-sun"})


if __name__ == "__main__":
    unittest.main()
