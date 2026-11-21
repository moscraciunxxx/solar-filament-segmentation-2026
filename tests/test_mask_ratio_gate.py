"""Finer mask grid stays shut until the current full-medoid run is scored."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MaskRatioGateTest(unittest.TestCase):
    def test_default_mask_ratio_matches_the_ultralytics_default(self) -> None:
        train = _load("train_yolo.py")
        args = train.build_parser().parse_args([])
        kwargs = train.train_kwargs(args, Path("data.yaml"), "cpu")
        self.assertEqual(args.mask_ratio, 4)
        self.assertEqual(args.proto_stride, 4)
        self.assertEqual(kwargs["mask_ratio"], 4)
        self.assertNotIn("proto_stride", kwargs)
        self.assertEqual(kwargs["imgsz"], 1024)
        self.assertEqual(kwargs["batch"], 4)

    def test_mask_ratio_two_is_what_the_trainer_receives(self) -> None:
        train = _load("train_yolo.py")
        args = train.build_parser().parse_args(["--mask-ratio", "2", "--imgsz", "2048", "--batch", "1"])
        kwargs = train.train_kwargs(args, Path("data.yaml"), "mps")
        self.assertEqual(kwargs["mask_ratio"], 2)
        self.assertEqual(kwargs["imgsz"], 2048)
        self.assertEqual(kwargs["batch"], 1)
        self.assertEqual(kwargs["device"], "mps")

    def test_finer_grid_does_not_start_over_the_live_run(self) -> None:
        starter = _load("start_yolo11l_seg_full_2048_m2_now.py")
        self.assertEqual(starter.DEST.name, "yolo11l_seg_full_2048_m2")
        self.assertNotEqual(starter.DEST.name, "yolo11l_seg_full_2048")
        command = starter.train_command()
        self.assertEqual(command[command.index("--mask-ratio") + 1], "2")
        self.assertEqual(command[command.index("--proto-stride") + 1], "2")
        self.assertEqual(command[command.index("--name") + 1], "yolo11l_seg_full_2048_m2")
        self.assertEqual(starter.launch_decision(None, "train_yolo.py", False), "mps_busy")
        self.assertEqual(starter.launch_decision(None, None, False), "await_lfull_score")
        self.assertEqual(
            starter.launch_decision({"public_forecast": 0.67}, None, False),
            "upload_lfull_instead",
        )
        self.assertEqual(
            starter.launch_decision({"public_forecast": 0.67}, "train_yolo.py", False),
            "upload_lfull_instead",
        )
        self.assertEqual(
            starter.launch_decision({"public_forecast": 0.388}, None, False),
            "start",
        )
        self.assertEqual(
            starter.launch_decision({"public_forecast": 0.6}, None, False),
            "start",
        )
        self.assertEqual(
            starter.launch_decision({"public_forecast": 0.388}, None, True),
            "already_trained",
        )


if __name__ == "__main__":
    unittest.main()
