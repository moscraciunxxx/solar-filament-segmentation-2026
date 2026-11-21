#!/usr/bin/env python3
"""Export Method 2 YOLO dataset then train yolo11s_seg_residual. Never submits.

Does not overwrite yolo_seg_phys / unet_3ch_disk / yolo_seg.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import FAST_CACHE, WEIGHTS_DIR, YOLO_DIR, YOLO_PHYS_DIR, YOLO_RESIDUAL_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
CLOSED = (
    "yolo11s_seg_phys",
    "yolo11s_seg_all",
    "yolo11s_seg_faint",
    "yolo11s_seg_shift",
    "yolo11s_seg_tiles",
)


def main() -> None:
    owner = live_mps_owner(ignore=frozenset({"run_residual_family.py"}))
    if owner:
        raise SystemExit(f"Refusing residual family: {owner} owns MPS.")
    if YOLO_RESIDUAL_DIR.resolve() == YOLO_PHYS_DIR.resolve():
        raise SystemExit("Refusing residual dest: equals yolo_seg_phys.")
    if not (YOLO_DIR / "data.yaml").exists():
        raise SystemExit(f"Need medoid export first: {YOLO_DIR / 'data.yaml'}")
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    export = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts" / "prepare_yolo.py"), "--residual-stack"],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    if export.returncode != 0:
        raise SystemExit(f"residual export failed: {export.returncode}")
    if not (YOLO_RESIDUAL_DIR / "data.yaml").exists():
        raise SystemExit(f"missing {YOLO_RESIDUAL_DIR / 'data.yaml'}")
    owner = live_mps_owner(ignore=frozenset({"run_residual_family.py"}))
    if owner:
        raise SystemExit(f"Refusing residual train: {owner} owns MPS.")
    name = "yolo11s_seg_residual"
    if name in CLOSED:
        raise SystemExit(f"Refusing residual train name {name}")
    train = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "train_yolo.py"),
            "--name",
            name,
            "--data",
            str(YOLO_RESIDUAL_DIR / "data.yaml"),
            "--model",
            "yolo11s-seg.pt",
            "--imgsz",
            "1024",
            "--batch",
            "8",
            "--epochs",
            "36",
            "--patience",
            "12",
        ],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    if train.returncode != 0:
        raise SystemExit(f"residual train failed: {train.returncode}")
    print({"best": str(WEIGHTS_DIR / name / "weights" / "best.pt"), "do_not_submit": True})


if __name__ == "__main__":
    main()
