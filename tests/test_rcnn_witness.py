"""A disjoint donor is kept only when a witness mask overlaps it."""

from __future__ import annotations

import unittest

import numpy as np

from filament.rcnn_witness import WITNESS_IOU, rcnn_witness_add
from filament.rle import encode_rle


def _blob(x: int, y: int, width: int = 8) -> str:
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[y : y + width, x : x + width] = 1
    return encode_rle(mask)


class RcnnWitnessTest(unittest.TestCase):
    def test_witness_cut_is_one_half(self) -> None:
        self.assertEqual(WITNESS_IOU, 0.5)

    def test_confirmed_far_donor_is_kept(self) -> None:
        parent = _blob(100, 100)
        donor = _blob(400, 400)
        kept, n_added = rcnn_witness_add([parent], [donor], [donor])
        self.assertEqual(n_added, 1)
        self.assertEqual(kept, [parent, donor])

    def test_unconfirmed_or_overlapping_donor_is_dropped(self) -> None:
        parent = _blob(100, 100)
        far = _blob(400, 400)
        overlap = _blob(100, 100)
        stranger = _blob(800, 800)
        kept, n_added = rcnn_witness_add([parent], [far, overlap], [stranger])
        self.assertEqual(kept, [parent])
        self.assertEqual(n_added, 0)

    def test_empty_parent_still_needs_a_witness(self) -> None:
        donor = _blob(400, 400)
        kept, n_added = rcnn_witness_add([], [donor], [donor])
        self.assertEqual(n_added, 1)
        self.assertEqual(kept, [donor])
        kept, n_added = rcnn_witness_add([], [donor], [_blob(800, 800)])
        self.assertEqual(kept, [])
        self.assertEqual(n_added, 0)


if __name__ == "__main__":
    unittest.main()
