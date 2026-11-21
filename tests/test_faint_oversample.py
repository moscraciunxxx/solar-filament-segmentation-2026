"""A training image is copied once when its faintest filament is below the median."""

from __future__ import annotations

import unittest

from filament.faint_oversample import image_is_faint


class FaintOversampleTest(unittest.TestCase):
    def test_below_median_is_copied_and_above_is_not(self) -> None:
        self.assertTrue(image_is_faint([12.0, 14.0], median=15.0))
        self.assertFalse(image_is_faint([12.0, 20.0], median=15.0))
        self.assertFalse(image_is_faint([16.0, 22.0], median=15.0))
        self.assertFalse(image_is_faint([None, None], median=15.0))
        self.assertTrue(image_is_faint([None, 14.9], median=15.0))


if __name__ == "__main__":
    unittest.main()
