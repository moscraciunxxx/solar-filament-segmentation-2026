#!/usr/bin/env python3
"""Build a test CSV: frozen Slot 1 NMS-merged with a specialist.

Default: refuse unless faint (or named) official gate is PROMOTE_LOOK.
Never uploads. Never reads MAGFiLO test labels. Test JPEGs only.
Do not run while a YOLO train owns MPS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.infer import pick_device, run_yolo_infer
from filament.merge import MERGE_IOU, merge_rles
from filament.metric import pred_to_df
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR

SLOT1_WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_1280" / "weights" / "best.pt"
SLOT1_CONF = 0.35
SLOT1_IMGSZ = 1280
FAINT_GATE = ARTIFACTS / "yolo11s_seg_faint_fold0_official.json"


def yolo_train_alive() -> bool:
    proc = subprocess.run(["pgrep", "-f", "train_yolo.py"], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return False
    return any(int(x) != os.getpid() for x in proc.stdout.split() if x.strip().isdigit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist-weights", required=True)
    parser.add_argument("--specialist-conf", type=float, default=0.35)
    parser.add_argument("--specialist-imgsz", type=int, default=1024)
    parser.add_argument("--gate-json", default=str(FAINT_GATE))
    parser.add_argument(
        "--research-csv",
        action="store_true",
        help="Write a do-not-upload CSV even if gate is not PROMOTE_LOOK.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if yolo_train_alive():
        raise SystemExit("Refusing test infer while train_yolo.py owns MPS.")

    gate_path = Path(args.gate_json)
    decision = None
    if gate_path.exists():
        decision = json.loads(gate_path.read_text())["gate"]["decision"]
    if decision != "PROMOTE_LOOK" and not args.research_csv:
        raise SystemExit(
            f"Gate is {decision!r} (need PROMOTE_LOOK). Pass --research-csv only for a local do-not-upload file."
        )

    spec_w = Path(args.specialist_weights)
    if not spec_w.exists():
        raise FileNotFoundError(spec_w)
    if not SLOT1_WEIGHTS.exists():
        raise FileNotFoundError(SLOT1_WEIGHTS)

    from ultralytics import YOLO

    device = pick_device(args.device)
    slot1 = YOLO(str(SLOT1_WEIGHTS))
    spec = YOLO(str(spec_w))
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(TEST_IMG_DIR)

    frames = []
    n_added = 0
    for path in tqdm(images, desc="test merge Slot1+specialist"):
        slot_rles = run_yolo_infer(slot1, path, conf=SLOT1_CONF, imgsz=SLOT1_IMGSZ, merge_gap=0, device=device)
        spec_rles = run_yolo_infer(
            spec, path, conf=args.specialist_conf, imgsz=args.specialist_imgsz, merge_gap=0, device=device
        )
        merged, added = merge_rles(slot_rles, spec_rles, iou_thr=MERGE_IOU)
        n_added += added
        if merged:
            frames.append(pred_to_df(path.stem, merged))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "PROMOTE" if decision == "PROMOTE_LOOK" else "DONOTUPLOAD"
    out = Path(args.out) if args.out else (
        FAST_SUBMISSION_DIR / f"submission_slot1_faint_merged_{tag}.csv"
    )
    df.to_csv(out, index=False)
    meta = {
        "do_not_upload": decision != "PROMOTE_LOOK",
        "gate": decision,
        "gate_json": str(gate_path),
        "slot1_weights": str(SLOT1_WEIGHTS),
        "slot1_conf": SLOT1_CONF,
        "specialist_weights": str(spec_w.resolve()),
        "specialist_conf": args.specialist_conf,
        "n_images": len(images),
        "n_rows": int(len(df)),
        "n_specialist_added": n_added,
        "merge_iou": MERGE_IOU,
        "path": str(out),
        "leakage": "test JPEGs only; no MAGFiLO public-test labels",
        "never_submits": True,
    }
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
