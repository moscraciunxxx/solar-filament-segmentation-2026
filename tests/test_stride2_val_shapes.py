"""Stride-2 validation masks must share one spatial size.

Stock validation assumes a stride-4 prototype. It resizes ground truth to
image // 4 and tells process_mask the image is 4 * proto. A stride-2
prototype emits image // 2, so the mask IoU matmul sees 520 against 1040.
"""

from __future__ import annotations

import importlib.util
import inspect
import unittest
from pathlib import Path

import torch

from filament.proto_grid import Stride2SegmentationValidator, validation_pair_hw
from ultralytics.utils import ops
from ultralytics.utils.metrics import mask_iou


class Stride2ValidationShapeTest(unittest.TestCase):
    def test_stride2_pair_matches_the_proto_grid(self) -> None:
        gt_hw, process_shape = validation_pair_hw((2080, 2080), (1040, 1040), 2)
        self.assertEqual(gt_hw, (1040, 1040))
        self.assertEqual(process_shape, (2080, 2080))
        self.assertNotEqual((2080 // 4, 2080 // 4), (1040, 1040))

    def test_prepare_batch_does_not_shrink_stride2_masks_to_a_quarter(self) -> None:
        validator = Stride2SegmentationValidator.__new__(Stride2SegmentationValidator)
        validator.mask_stride = 2
        validator.device = torch.device("cpu")
        validator.process = ops.process_mask
        validator.args = type("Args", (), {"overlap_mask": True})()
        image = torch.zeros(1, 3, 40, 40)
        masks = torch.zeros(1, 20, 20)
        masks[0, 4:16, 4:16] = 1
        batch = {
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "ori_shape": [(40, 40)],
            "img": image,
            "ratio_pad": [((1.0, 1.0), (0.0, 0.0))],
            "im_file": ["frame.jpg"],
            "masks": masks,
        }
        prepared = validator._prepare_batch(0, batch)
        self.assertEqual(tuple(prepared["masks"].shape[-2:]), (20, 20))

    def test_full_box_on_the_stride2_shape_keeps_the_proto_mask(self) -> None:
        proto = torch.ones(1, 32, 32)
        coeff = torch.ones(1, 1)
        boxes = torch.tensor([[0.0, 0.0, 64.0, 64.0]])
        _, shape = validation_pair_hw((64, 64), (32, 32), 2)
        kept = ops.process_mask(proto, coeff, boxes, shape=shape, upsample=False)
        stock = ops.process_mask(proto, coeff, boxes, shape=(128, 128), upsample=False)
        self.assertEqual(tuple(kept.shape[-2:]), (32, 32))
        self.assertEqual(int(kept.sum()), 32 * 32)
        self.assertLess(int(stock.sum()), 32 * 32)
        gt = torch.ones(1, 32, 32)
        iou = mask_iou(gt.flatten(1), kept.flatten(1).float())
        self.assertEqual(tuple(iou.shape), (1, 1))

    def test_postprocess_tells_process_mask_the_real_image_size(self) -> None:
        from ultralytics.models.yolo.detect import val as detect_val

        validator = Stride2SegmentationValidator.__new__(Stride2SegmentationValidator)
        validator.mask_stride = 2
        recorded: dict[str, tuple[int, int]] = {}

        def fake_process(proto, coeff, boxes, shape, upsample=False):
            recorded["shape"] = (int(shape[0]), int(shape[1]))
            return torch.ones(coeff.shape[0], int(proto.shape[-2]), int(proto.shape[-1]))

        validator.process = fake_process
        original = detect_val.DetectionValidator.postprocess

        def fake_nms(self, preds):
            return [
                {
                    "bboxes": torch.tensor([[0.0, 0.0, 16.0, 16.0]]),
                    "conf": torch.tensor([0.9]),
                    "cls": torch.tensor([0.0]),
                    "extra": torch.ones(1, 4),
                }
            ]

        detect_val.DetectionValidator.postprocess = fake_nms
        try:
            proto = torch.zeros(1, 4, 8, 8)
            out = validator.postprocess((torch.zeros(1, 16), proto))
        finally:
            detect_val.DetectionValidator.postprocess = original
        self.assertEqual(recorded["shape"], (16, 16))
        self.assertEqual(tuple(out[0]["masks"].shape[-2:]), (8, 8))

    def test_stride2_trainer_builds_the_aligned_validator(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "train_yolo.py"
        spec = importlib.util.spec_from_file_location("train_yolo_stride2", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        trainer = module.stride2_trainer()
        self.assertIn("get_validator", trainer.__dict__)
        self.assertIn("Stride2SegmentationValidator", inspect.getsource(trainer.get_validator))


if __name__ == "__main__":
    unittest.main()
