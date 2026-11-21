"""Resuming the stride-2 run keeps the stride-2 validator and the extra upsample."""

from __future__ import annotations

import importlib.util
import inspect
import unittest
from pathlib import Path

import torch

from filament.proto_grid import copy_stride2_upsample, patch_model_proto
from ultralytics.nn.modules.block import Proto


def _load_train():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_yolo.py"
    spec = importlib.util.spec_from_file_location("train_yolo_resume", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Head(torch.nn.Module):
    def __init__(self, proto: Proto) -> None:
        super().__init__()
        self.proto = proto


class _Detector(torch.nn.Module):
    def __init__(self, proto: Proto) -> None:
        super().__init__()
        self.model = torch.nn.ModuleList([_Head(proto)])


class ResumeStride2Test(unittest.TestCase):
    def test_resume_with_proto_stride_two_uses_the_aligned_validator(self) -> None:
        train = _load_train()
        args = train.build_parser().parse_args(["--resume", "--proto-stride", "2", "--name", "yolo11l_seg_full_2048_m2"])
        kwargs = train.resume_kwargs(args)
        self.assertTrue(kwargs["resume"])
        self.assertIn("trainer", kwargs)
        self.assertIn("Stride2SegmentationValidator", inspect.getsource(kwargs["trainer"].get_validator))

    def test_plain_resume_does_not_force_the_stride2_trainer(self) -> None:
        train = _load_train()
        args = train.build_parser().parse_args(["--resume"])
        kwargs = train.resume_kwargs(args)
        self.assertEqual(kwargs, {"resume": True})

    def test_copy_restores_the_trained_extra_upsample(self) -> None:
        source = _Detector(Proto(8, 4, 4))
        target = _Detector(Proto(8, 4, 4))
        patch_model_proto(source)
        patch_model_proto(target)
        with torch.no_grad():
            source.model[0].proto.upsample[1].weight.fill_(0.25)
            target.model[0].proto.upsample[1].weight.fill_(0.0)
        self.assertTrue(copy_stride2_upsample(source, target))
        self.assertTrue(torch.equal(
            target.model[0].proto.upsample[1].weight,
            source.model[0].proto.upsample[1].weight,
        ))


if __name__ == "__main__":
    unittest.main()
