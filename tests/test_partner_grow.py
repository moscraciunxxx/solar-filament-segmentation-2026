"""A partial partner extends the parent by one clip. A copy and a far partner do not."""

from __future__ import annotations

import unittest

import numpy as np

from filament.partner_grow import CLIP, grow_masks, replace_partial_masks, union_partial_masks
from filament.paths import IMAGE_SIZE


class PartnerGrowTest(unittest.TestCase):
    def test_partial_partner_extends_only_inside_the_clip(self) -> None:
        parent = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        partner = np.zeros_like(parent)
        parent[500:504, 400:480] = 1
        partner[500:504, 460:560] = 1
        grown = grow_masks([parent], [partner])
        self.assertEqual(len(grown), 1)
        self.assertEqual(CLIP, 24)
        self.assertEqual(int(grown[0][501, 450]), 1)
        self.assertEqual(int(grown[0][501, 490]), 1)
        self.assertEqual(int(grown[0][501, 540]), 0)

    def test_copy_and_far_partner_leave_the_parent(self) -> None:
        parent = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        parent[500:504, 400:480] = 1
        far = np.zeros_like(parent)
        far[800:804, 800:900] = 1
        copied = grow_masks([parent], [parent.copy()])
        untouched = grow_masks([parent], [far])
        self.assertTrue(np.array_equal(copied[0], parent))
        self.assertTrue(np.array_equal(untouched[0], parent))

    def test_partial_replace_keeps_the_partner_tip_only(self) -> None:
        parent = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        partner = np.zeros_like(parent)
        parent[500:504, 400:480] = 1
        partner[500:504, 460:560] = 1
        swapped = replace_partial_masks([parent], [partner])
        self.assertEqual(len(swapped), 1)
        self.assertEqual(int(swapped[0][501, 420]), 0)
        self.assertEqual(int(swapped[0][501, 540]), 1)
        far = np.zeros_like(parent)
        far[800:804, 800:900] = 1
        self.assertTrue(np.array_equal(replace_partial_masks([parent], [parent.copy()])[0], parent))
        self.assertTrue(np.array_equal(replace_partial_masks([parent], [far])[0], parent))

    def test_partial_union_keeps_both_tips(self) -> None:
        parent = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        partner = np.zeros_like(parent)
        parent[500:504, 400:480] = 1
        partner[500:504, 460:560] = 1
        united = union_partial_masks([parent], [partner])
        self.assertEqual(len(united), 1)
        self.assertEqual(int(united[0][501, 420]), 1)
        self.assertEqual(int(united[0][501, 540]), 1)
        far = np.zeros_like(parent)
        far[800:804, 800:900] = 1
        self.assertTrue(np.array_equal(union_partial_masks([parent], [parent.copy()])[0], parent))
        self.assertTrue(np.array_equal(union_partial_masks([parent], [far])[0], parent))


if __name__ == "__main__":
    unittest.main()
