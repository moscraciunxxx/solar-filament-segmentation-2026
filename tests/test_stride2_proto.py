"""A second transpose doubles the prototype grid and does not stack twice."""

from __future__ import annotations

import unittest

import torch

from filament.proto_grid import patch_model_proto, use_stride2_proto
from ultralytics.nn.modules.block import Proto


def _hw(proto: Proto, size: int = 8) -> tuple[int, int]:
    proto.eval()
    with torch.no_grad():
        out = proto(torch.zeros(1, proto.cv1.conv.in_channels, size, size))
    return int(out.shape[-2]), int(out.shape[-1])


class _Head(torch.nn.Module):
    def __init__(self, proto: Proto) -> None:
        super().__init__()
        self.proto = proto


class _Detector(torch.nn.Module):
    def __init__(self, proto: Proto) -> None:
        super().__init__()
        self.model = torch.nn.ModuleList([_Head(proto)])


class Stride2ProtoTest(unittest.TestCase):
    def test_one_extra_upsample_doubles_the_grid(self) -> None:
        proto = Proto(8, 4, 4)
        self.assertEqual(_hw(proto), (16, 16))
        use_stride2_proto(proto)
        self.assertEqual(_hw(proto), (32, 32))

    def test_a_second_call_does_not_add_another_upsample(self) -> None:
        proto = Proto(8, 4, 4)
        use_stride2_proto(proto)
        use_stride2_proto(proto)
        self.assertEqual(_hw(proto), (32, 32))

    def test_patch_finds_the_nested_segment_head(self) -> None:
        proto = Proto(8, 4, 4)
        patch_model_proto(_Detector(proto))
        self.assertEqual(_hw(proto), (32, 32))


if __name__ == "__main__":
    unittest.main()
