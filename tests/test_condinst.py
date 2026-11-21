"""CondInst controller, promote gate, and a tiny overfit of the dynamic mask."""

from __future__ import annotations

import unittest

import torch

from filament.condinst import (
    CONTROLLER_DIM,
    PARENT_TP,
    CondInst,
    apply_dynamic_mask,
    condinst_loss,
    controller_dim,
    should_promote,
)
from filament.paths import TEST_IMG_DIR


class CondInstTest(unittest.TestCase):
    def test_controller_width_matches_the_three_convs(self) -> None:
        self.assertEqual(controller_dim(), 169)
        self.assertEqual(CONTROLLER_DIM, 169)
        feat = torch.zeros(8, 4, 4)
        params = torch.zeros(CONTROLLER_DIM)
        out = apply_dynamic_mask(feat, (1, 2), params)
        self.assertEqual(tuple(out.shape), (4, 4))

    def test_promote_gate(self) -> None:
        self.assertFalse(should_promote(0.4387, PARENT_TP, same_hit_set=True))
        self.assertFalse(should_promote(0.0665, 86, same_hit_set=False))
        self.assertFalse(should_promote(0.47, PARENT_TP, same_hit_set=True))
        self.assertTrue(should_promote(0.47, 1000, same_hit_set=False))
        self.assertTrue(should_promote(0.47, PARENT_TP, same_hit_set=False))
        self.assertTrue(should_promote(0.670, 10, same_hit_set=True))

    def test_dynamic_mask_loss_drops_on_one_blob(self) -> None:
        torch.manual_seed(0)
        model = CondInst(backbone="tiny", pretrained=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        image = torch.zeros(1, 3, 64, 64)
        mask = torch.zeros(64, 64)
        mask[16:48, 16:48] = 1
        image[:, :, 16:48, 16:48] = 1
        inst = [{"mask": mask, "box": torch.tensor([16.0, 16.0, 48.0, 48.0])}]
        first = None
        last = None
        for _ in range(25):
            opt.zero_grad(set_to_none=True)
            outputs = model(image)
            unbatched = {key: value[0] for key, value in outputs.items()}
            losses = condinst_loss(unbatched, inst)
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()
            opt.step()
            value = float(losses["mask"])
            if first is None and value > 0:
                first = value
            last = value
        self.assertIsNotNone(first)
        self.assertLess(last, first)
        self.assertFalse(TEST_IMG_DIR.joinpath("used").exists())

    def test_full_mask_body_is_a_positive(self) -> None:
        torch.manual_seed(0)
        model = CondInst(backbone="tiny", pretrained=False)
        image = torch.zeros(1, 3, 64, 64)
        mask = torch.ones(64, 64)
        inst = [{"mask": mask, "box": torch.tensor([0.0, 0.0, 64.0, 64.0])}]
        outputs = model(image)
        unbatched = {key: value[0] for key, value in outputs.items()}
        losses = condinst_loss(unbatched, inst)
        self.assertGreaterEqual(float(losses["n_pos"]), 60)


if __name__ == "__main__":
    unittest.main()
