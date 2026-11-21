#!/usr/bin/env python3
"""Prepare native-2048 YOLO11s-seg on thin erode_gt labels. Default is dry.

Dest is weights/yolo11s_seg_erode_gt_2048 only. Instance segmentation, not a
semantic U-Net. Does not start while another train owns MPS. Never submits.

A 1-step 2048 memory probe was not run in the session that wrote this file
when Track 2 (yolo11s_seg_pq_any) owns MPS. imgsz stays 2048. Closed imgsz
dests that must not be reused: 1024, 1280, 1408, 1536. If a later probe OOMs,
the fallback imgsz is 1920 (not 1408, 1536, 1280, or 1024).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, WEIGHTS_DIR, YOLO_ERODE_GT_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
RECEIPT = ARTIFACTS / "start_yolo11s_seg_2048_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048"
IMGSZ = 2048
FALLBACK_IMGSZ_IF_OOM = 1920
CLOSED_IMGSZ = (1024, 1280, 1408, 1536)
CLOSED = {
    "yolo11s_seg",
    "yolo11m_seg_1280",
    "yolo11l_seg_1280",
    "yolo11l_seg_erode_gt",
    "yolo11m_seg_erode_gt",
    "yolo11s_seg_erode_gt",
    "yolo11n_seg_erode_gt",
    "yolo11s_seg_erode_gt_1024",
    "yolo11s_seg_erode_gt_1408",
    "yolo11s_seg_erode_gt_1536",
    "yolo11s_seg_erode_gt_1408_ms",
    "yolo11s_seg_1408",
    "yolo11l_seg_shift",
    "yolo11s_seg_all",
    "yolo11s_seg_phys",
    "yolo11s_seg_shift",
    "yolo11s_seg_tiles",
    "yolo11s_seg_tiles_cm",
    "yolo11s_seg_faint",
    "yolo11s_seg_residual",
    "yolo11s_seg_rater",
    "yolo11s_seg_clahe",
    "yolo11s_seg_pq_any",
    "unet_resnet34",
    "unet_resnet34_1536",
    "unet_semantic",
    "unet_patch_dhebe",
    "unet_3ch_disk",
}
WATCH_NOT_MPS = frozenset(
    {
        "watch_yolo11l_then_score.py",
        "watch_erode_gt_then_score.py",
        "watch_yolo11l_erode_gt_then_score.py",
        "watch_yolo11l_shift_then_score.py",
        "watch_yolo11s_erode_gt_then_score.py",
        "watch_yolo11n_erode_gt_then_score.py",
        "watch_yolo11s_erode_gt_1024_then_score.py",
        "watch_yolo11s_erode_gt_1408_then_score.py",
        "watch_yolo11s_erode_gt_1536_then_score.py",
        "watch_yolo11s_erode_gt_1408_ms_then_score.py",
        "watch_yolo11s_seg_1408_then_score.py",
        "watch_yolo11s_clahe_then_score.py",
        "watch_yolo11s_pq_any_then_score.py",
        "watch_rater_then_score.py",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if IMGSZ in CLOSED_IMGSZ:
        raise SystemExit(f"refusing closed imgsz {IMGSZ}")
    if IMGSZ != IMAGE_SIZE:
        raise SystemExit(f"native instance imgsz must be {IMAGE_SIZE}, got {IMGSZ}")
    owner = live_mps_owner(ignore=WATCH_NOT_MPS)
    data = YOLO_ERODE_GT_DIR / "data.yaml"
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_flag": args.start,
        "auto_start": False,
        "started": False,
        "family": "yolo11s_seg_erode_gt_2048",
        "task": "native_2048_instance_not_semantic_unet",
        "data": str(data),
        "dest": str(DEST),
        "model": "yolo11s-seg.pt",
        "imgsz": IMGSZ,
        "batch": 4,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": 0.35,
        "label_protocol": "thin_erode_gt",
        "memory_probe": {
            "ran": False,
            "why": "Deferred while yolo11s_seg_pq_any is the one MPS job. Do not probe in parallel.",
            "oom": None,
            "fallback_imgsz_if_later_oom": FALLBACK_IMGSZ_IF_OOM,
            "fallback_not_in": list(CLOSED_IMGSZ),
        },
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "mps_owner": owner,
        "slot2_stays": 56148889,
        "pair_stays": [56173362, 56148889],
        "extras_or": False,
        "test_labels_used": False,
        "why": (
            "Native 2048 instance YOLO11s-seg on the thin erode_gt yaml. "
            "Dry until --start and until MPS is free. Not the banned semantic U-Net."
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.name in CLOSED or IMGSZ in CLOSED_IMGSZ:
        out["refused"] = f"dest {DEST.name} or imgsz {IMGSZ} is closed"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST / 'weights' / 'best.pt'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if owner:
        out["refused"] = f"Refusing yolo11s_seg_erode_gt_2048 start: {owner} owns MPS."
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        raise SystemExit(0 if not args.start else 1)
    if not data.exists():
        out["refused"] = f"missing {data}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11s_seg_erode_gt_2048_family.out"
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            "yolo11s_seg_erode_gt_2048",
            "--data",
            str(data),
            "--model",
            "yolo11s-seg.pt",
            "--imgsz",
            str(IMGSZ),
            "--batch",
            "4",
            "--epochs",
            "40",
            "--patience",
            "12",
        ],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=run_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    out.update({"started": True, "dry": False, "train_pid": run.pid, "train_log": str(run_log.relative_to(ROOT))})
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
