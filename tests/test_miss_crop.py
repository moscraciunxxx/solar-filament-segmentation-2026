"""Miss crops stay on the frame and drop polygons that miss the window."""

from __future__ import annotations

import unittest

import numpy as np

from filament.miss_crop import (
    CROP,
    MISS_IOU,
    NEG_PER_IMAGE,
    crop_origin,
    polygon_in_crop,
    quiet_origins,
    window_is_quiet,
    window_origins,
)


class MissCropTest(unittest.TestCase):
    def test_constants_are_fixed(self) -> None:
        self.assertEqual(CROP, 640)
        self.assertEqual(MISS_IOU, 0.5)

    def test_windows_cover_the_far_edge(self) -> None:
        origins = window_origins()
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 2048 - 640)
        self.assertIn(320, origins)

    def test_quiet_window_misses_filaments_and_stays_on_the_disk(self) -> None:
        self.assertEqual(NEG_PER_IMAGE, 2)
        union = np.zeros((2048, 2048), dtype=np.uint8)
        union[100:140, 100:140] = 1
        self.assertFalse(window_is_quiet(union, 0, 0))
        self.assertTrue(window_is_quiet(union, 704, 704))
        filled = np.ones((2048, 2048), dtype=np.uint8)
        self.assertEqual(quiet_origins(filled, np.random.default_rng(0)), [])
        picked = quiet_origins(union, np.random.default_rng(0), count=2)
        self.assertEqual(len(picked), 2)
        for x0, y0 in picked:
            self.assertTrue(window_is_quiet(union, x0, y0))

    def test_edge_centroid_clips_and_center_holds_the_point(self) -> None:
        self.assertEqual(crop_origin(10, 10), (0, 0))
        x0, y0 = crop_origin(2000, 1900)
        self.assertEqual((x0, y0), (2048 - 640, 2048 - 640))
        self.assertLessEqual(x0, 2000)
        self.assertLess(2000, x0 + 640)
        x1, y1 = crop_origin(1000, 1200)
        self.assertLessEqual(x1, 1000)
        self.assertLess(1000, x1 + 640)
        self.assertLessEqual(y1, 1200)
        self.assertLess(1200, y1 + 640)

    def test_polygon_outside_the_window_is_dropped(self) -> None:
        inside = np.array([[100, 100], [140, 100], [140, 160], [100, 160]], dtype=np.float32)
        outside = np.array([[900, 900], [940, 900], [940, 960]], dtype=np.float32)
        self.assertIsNotNone(polygon_in_crop(inside, 0, 0))
        self.assertIsNone(polygon_in_crop(outside, 0, 0))


if __name__ == "__main__":
    unittest.main()
