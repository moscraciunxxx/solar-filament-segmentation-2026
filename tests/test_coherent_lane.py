"""A dark aligned lane is proposed; a bright field and a blocked lane are not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.coherent_lane import COHERENCE_MIN, WINDOW, propose_coherent_masks


def _scene() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((256, 256), 140, dtype=np.uint8)
    image[120:128, 30:220] = 90
    return image, np.zeros((256, 256), dtype=np.uint8)


class CoherentLaneTest(unittest.TestCase):
    def test_scale_is_not_the_31_pixel_dip(self) -> None:
        self.assertEqual(WINDOW, 96)
        self.assertGreaterEqual(COHERENCE_MIN, 0.40)

    def test_dark_lane_is_kept(self) -> None:
        image, occupied = _scene()
        masks = propose_coherent_masks(
            image,
            occupied,
            response_min=8,
            min_area=40,
            max_area=5000,
            window=31,
            coherence_min=0.40,
            far_px=0,
            min_length=40,
        )
        self.assertEqual(len(masks), 1)
        self.assertGreater(int(masks[0].sum()), 40)

    def test_high_cut_and_blocked_lane_add_nothing(self) -> None:
        image, occupied = _scene()
        self.assertEqual(
            propose_coherent_masks(
                image,
                occupied,
                response_min=80,
                min_area=40,
                max_area=5000,
                window=31,
                coherence_min=0.40,
                far_px=0,
                min_length=40,
            ),
            [],
        )
        blocked = occupied.copy()
        blocked[100:150, 20:230] = 1
        self.assertEqual(
            propose_coherent_masks(
                image,
                blocked,
                response_min=8,
                min_area=40,
                max_area=5000,
                window=31,
                coherence_min=0.40,
                far_px=8,
                min_length=40,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
