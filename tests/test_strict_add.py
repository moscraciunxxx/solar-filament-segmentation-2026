"""A candidate is added only when it is large, witnessed, and off the parent."""

from __future__ import annotations

import unittest

import numpy as np

from filament.paths import IMAGE_SIZE
from filament.rle import encode_rle
from filament.strict_add import strict_add


def _rect(y0: int, x0: int, h: int, w: int) -> str:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 1
    return encode_rle(mask)


class StrictAddTest(unittest.TestCase):
    def test_only_the_large_witnessed_miss_is_added(self) -> None:
        parent = [_rect(10, 10, 30, 20)]
        good = _rect(200, 200, 24, 16)
        small = _rect(400, 400, 4, 4)
        touching = _rect(12, 12, 24, 16)
        witness = [_rect(202, 202, 20, 14), _rect(402, 402, 4, 4), _rect(14, 14, 20, 14)]
        kept, added = strict_add(parent, [good, small, touching], witness)
        self.assertEqual(added, 1)
        self.assertEqual(kept[0], parent[0])
        self.assertEqual(kept[1], good)


if __name__ == "__main__":
    unittest.main()
