#!/usr/bin/env python3
"""Train YOLO11l-seg at 2048 on full medoids with a stride-2 prototype.

mask_ratio 2 only refines the label raster. The extra transpose is what
makes the prototype 1024 pixels at image size 2048. Default is dry.
Dest is weights/yolo11l_seg_full_2048_m2. The live stride-4 run is left
alone. Starts only after that run's fold-0 forecast is at or under 0.6
and nothing owns the GPU. Never submits.
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

from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, WEIGHTS_DIR, YOLO_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
RECEIPT = ARTIFACTS / "start_yolo11l_seg_full_2048_m2_now.json"
DEST = WEIGHTS_DIR / "yolo11l_seg_full_2048_m2"
MODEL = ROOT / "yolo11l-seg.pt"
DATA = YOLO_DIR / "data.yaml"
SCORE = ARTIFACTS / "stream_lfull_2048_fold0.json"
MASK_RATIO = 2


def train_command() -> list[str]:
    return [
        str(PYTHON),
        str(TRAIN),
        "--name",
        DEST.name,
        "--data",
        str(DATA),
        "--model",
        str(MODEL),
        "--imgsz",
        str(IMAGE_SIZE),
        "--batch",
        "1",
        "--epochs",
        "40",
        "--patience",
        "12",
        "--mask-ratio",
        str(MASK_RATIO),
        "--proto-stride",
        "2",
    ]


def launch_decision(score: dict | None, owner: str | None, dest_has_weights: bool) -> str:
    if dest_has_weights:
        return "already_trained"
    forecast = None
    if score is not None and "public_forecast" in score:
        forecast = float(score["public_forecast"])
    if forecast is not None and forecast > 0.6:
        return "upload_lfull_instead"
    if owner:
        return "mps_busy"
    if forecast is None:
        return "await_lfull_score"
    return "start"


def _read_score() -> dict | None:
    if not SCORE.is_file():
        return None
    return json.loads(SCORE.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner()
    score = _read_score()
    dest_has_weights = (DEST / "weights" / "best.pt").exists()
    decision = launch_decision(score, owner, dest_has_weights)
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started": False,
        "decision": decision,
        "family": DEST.name,
        "labels": "full_medoid_yolo_seg",
        "mask_ratio": MASK_RATIO,
        "proto_stride": 2,
        "data": str(DATA),
        "dest": str(DEST),
        "model": str(MODEL),
        "imgsz": IMAGE_SIZE,
        "batch": 1,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": 0.35,
        "score_file": str(SCORE),
        "public_forecast": None if score is None else score.get("public_forecast"),
        "why": (
            "A perfect stride-4 medoid round-trip forecasts 0.608, with almost "
            "no room under the 0.6 bar. The next run adds one prototype "
            "upsample so the grid is stride 2, and mask_ratio 2 matches that grid."
        ),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "test_labels_used": False,
        "mps_owner": owner,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if decision != "start":
        out["refused"] = decision
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        raise SystemExit(1)
    if not MODEL.is_file() or not DATA.is_file():
        out["refused"] = f"missing model or data: {MODEL} {DATA}"
        out["decision"] = "missing_inputs"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11l_seg_full_2048_m2_family.out"
    run = subprocess.Popen(
        train_command(),
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=run_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    out.update({"started": True, "train_pid": run.pid, "train_log": str(run_log.relative_to(ROOT))})
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
