"""An added mask stays only when the witness shares a pixel. The base stays."""

from __future__ import annotations

import unittest

import numpy as np

from filament.agreed_add import keep_added_if_touched


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class AddedWitnessTest(unittest.TestCase):
    def test_an_added_mask_the_witness_misses_is_dropped(self) -> None:
        base = _box(2, 10, 2, 10)
        added = _box(18, 28, 18, 28)
        far = _box(0, 4, 24, 28)
        kept, n_kept = keep_added_if_touched([base], [added], [far])
        self.assertEqual(n_kept, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))

    def test_an_added_mask_the_witness_touches_stays(self) -> None:
        base = _box(2, 10, 2, 10)
        added = _box(18, 28, 18, 28)
        touch = _box(22, 30, 22, 30)
        kept, n_kept = keep_added_if_touched([base], [added], [touch])
        self.assertEqual(n_kept, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[0], base))
        self.assertTrue(np.array_equal(kept[1], added))

    def test_a_witness_mask_is_not_appended(self) -> None:
        base = _box(2, 10, 2, 10)
        added = _box(18, 28, 18, 28)
        extra = _box(0, 6, 20, 28)
        touch = _box(20, 26, 20, 26)
        kept, n_kept = keep_added_if_touched([base], [added], [touch, extra])
        self.assertEqual(n_kept, 1)
        self.assertEqual(len(kept), 2)
        self.assertTrue(np.array_equal(kept[1], added))

    def test_a_base_mask_stays_when_the_witness_misses_it(self) -> None:
        base = _box(2, 10, 2, 10)
        far = _box(20, 28, 20, 28)
        kept, n_kept = keep_added_if_touched([base], [], [far])
        self.assertEqual(n_kept, 0)
        self.assertEqual(len(kept), 1)
        self.assertTrue(np.array_equal(kept[0], base))


if __name__ == "__main__":
    unittest.main()
