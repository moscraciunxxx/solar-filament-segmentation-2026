#!/usr/bin/env python3
"""Start dest-safe YOLO11s@1536 on eroded-GT labels. Default is dry.

Dest is weights/yolo11s_seg_erode_gt_1536 only. Never overwrites HOLD/closed dests.
Never submits. Conf 0.35 locked a priori. Standalone official PQ, not extras-OR.
Ban is m/l@1536 — this is s@1536, a new dest name.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR, YOLO_ERODE_GT_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
WATCH = ROOT / "scripts" / "watch_yolo11s_erode_gt_1536_then_score.py"
RECEIPT = ARTIFACTS / "start_yolo11s_erode_gt_1536_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_erode_gt_1536"
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
    "yolo11l_seg_shift",
    "yolo11s_seg_all",
    "yolo11s_seg_phys",
    "yolo11s_seg_shift",
    "yolo11s_seg_tiles",
    "yolo11s_seg_tiles_cm",
    "yolo11s_seg_faint",
    "yolo11s_seg_residual",
    "yolo11s_seg_rater",
    "unet_resnet34",
    "unet_resnet34_1536",
    "unet_semantic",
    "unet_patch_dhebe",
    "unet_3ch_disk",
}
LOCKED_CONF = 0.35
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
        "watch_rater_then_score.py",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner(ignore=WATCH_NOT_MPS)
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_flag": args.start,
        "auto_start": False,
        "family": "yolo11s_seg_erode_gt_1536",
        "data": str(YOLO_ERODE_GT_DIR / "data.yaml"),
        "dest": str(DEST),
        "model": "yolo11s-seg.pt",
        "imgsz": 1536,
        "batch": 4,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": LOCKED_CONF,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "mps_owner": owner,
        "slot2_stays": 56148889,
        "parent_bar": 0.4365,
        "leftover_invention": True,
        "extras_or": False,
        "user_banned_name": False,
        "user_ban_is_ml_1536_not_s": True,
        "can_certify_0.50": False,
        "native_image_size": 2048,
        "do_not_overwrite_s1408_hold": True,
        "why": (
            "Goal-continue after s@1408 0.4384 HOLD (hidden-exam / forecast 0.332). "
            "Resolution ladder on thin GT: 1024 0.3799 < 1280 0.4364 < 1408 0.4384. "
            "1536 is dest-safe s-only (ban is m/l@1536). Do not overwrite 1408 HOLD. "
            "Thick s/n@1280 stay parked. Cannot certify 0.50. Watcher scores only. "
            "Even if pq>0.4365: HOLD_FOR_PARENT, do not submit."
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.name in CLOSED:
        out["refused"] = f"dest {DEST.name} is a closed overwrite"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST / 'weights' / 'best.pt'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if owner:
        out["refused"] = f"Refusing yolo11s_erode_gt_1536 start: {owner} owns MPS."
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(0 if not args.start else 1)
    if not (YOLO_ERODE_GT_DIR / "data.yaml").exists():
        out["refused"] = f"missing {YOLO_ERODE_GT_DIR / 'data.yaml'}"
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
    run_log = ARTIFACTS / "yolo11s_erode_gt_1536_family.out"
    watch_log = ARTIFACTS / "yolo11s_erode_gt_1536_watch.out"
    watch = subprocess.Popen(
        [str(PYTHON), str(WATCH)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=watch_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            "yolo11s_seg_erode_gt_1536",
            "--data",
            str(YOLO_ERODE_GT_DIR / "data.yaml"),
            "--model",
            "yolo11s-seg.pt",
            "--imgsz",
            "1536",
            "--batch",
            "4",
            "--epochs",
            "40",
            "--patience",
            "12",
            "--copy-paste",
            "0.15",
            "--mixup",
            "0.05",
        ],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=run_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    out.update(
        {
            "started": True,
            "train_pid": run.pid,
            "watch_pid": watch.pid,
            "train_log": str(run_log.relative_to(ROOT)),
            "watch_log": str(watch_log.relative_to(ROOT)),
        }
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
