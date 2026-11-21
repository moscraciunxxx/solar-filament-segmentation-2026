#!/usr/bin/env python3
"""Build a test CSV: frozen Slot 1 NMS-merged with ResNet34 U-Net instances.

This is not the v1 TinyUNet promote path (make_unet_merged_submission.py).
refuse_resnet() is the RESEARCH_HOLD *start* gate — this CSV does not wait
on that family being in-training. It is only legal after the *ResNet merge*
official gate is PROMOTE_LOOK (eval_unet_resnet_merge.py).

Never uploads. Never reads MAGFiLO test labels. Test JPEGs only.
Refuse while live_mps_owner() is set. Heavy imports stay behind refuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR
from filament.slot1_csv import SLOT1_TEST_CSV, load_slot1_test_rles
from gate_unet_family import PROMOTE, live_mps_owner, read_gate

SLOT1_WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_1280" / "weights" / "best.pt"
SLOT1_CONF = 0.35
SLOT1_IMGSZ = 1280
UNET_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
GATE = ARTIFACTS / "unet_resnet34_slot1_merge_fold0_official.json"
FORBIDDEN_WEIGHT_MARKERS = ("unet_semantic", "TinyUNet", "tinyunet")


def refuse(gate_path: Path, *, research_csv: bool = False) -> str | None:
    """Owner-first, then ResNet merge PROMOTE_LOOK. Not refuse_resnet() / RESEARCH_HOLD."""
    owner = live_mps_owner()
    if owner:
        return f"Refusing ResNet34 merge test CSV: {owner} owns MPS."
    decision, _ = read_gate(gate_path)
    if decision != PROMOTE and not research_csv:
        if decision is None:
            return (
                f"Gate is None (need PROMOTE_LOOK). missing {gate_path}. "
                "Pass --research-csv only for a local do-not-upload file."
            )
        return (
            f"Gate is {decision!r} (need PROMOTE_LOOK). "
            "Pass --research-csv only for a local do-not-upload file."
        )
    return None


def _reject_tinyunet_weights(weights: Path) -> None:
    text = str(weights)
    if any(marker in text for marker in FORBIDDEN_WEIGHT_MARKERS):
        raise SystemExit(
            "Refusing TinyUNet / unet_semantic weights. This CSV is build_resnet34_unet only."
        )


def run_infer(args: argparse.Namespace, decision: str | None, gate_path: Path) -> None:
    """Real Slot1 + ResNet34 U-Net test merge. Never submits. Never reads test labels."""
    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances, infer_prob
    from filament.infer import pick_device, run_yolo_infer
    from filament.merge import MERGE_IOU, merge_rles
    from filament.metric import pred_to_df
    from filament.unet_resnet import build_resnet34_unet

    weights = Path(args.weights)
    _reject_tinyunet_weights(weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(TEST_IMG_DIR)

    slot1_by_stem = None
    slot1_source = "yolo_infer"
    if SLOT1_TEST_CSV.exists():
        slot1_by_stem = load_slot1_test_rles(images)
        slot1_source = str(SLOT1_TEST_CSV)
    elif not SLOT1_WEIGHTS.exists():
        raise FileNotFoundError(SLOT1_WEIGHTS)

    device = pick_device(args.device)
    slot1 = None
    if slot1_by_stem is None:
        from ultralytics import YOLO

        slot1 = YOLO(str(SLOT1_WEIGHTS))
    unet = build_resnet34_unet(in_ch=3, pretrained=False)
    unet.load_state_dict(torch.load(weights, map_location=device))
    unet.to(device)
    unet.eval()

    frames = []
    n_added = 0
    for path in tqdm(images, desc="test merge Slot1+ResNet34"):
        if slot1_by_stem is not None:
            slot_rles = slot1_by_stem[path.stem]
        else:
            slot_rles = run_yolo_infer(
                slot1, path, conf=SLOT1_CONF, imgsz=SLOT1_IMGSZ, merge_gap=0, device=device
            )
        extra = decode_instances(
            infer_prob(unet, path, args.imgsz, device),
            args.thr,
            min_area=int(args.min_area),
        )
        merged, added = merge_rles(slot_rles, extra, iou_thr=MERGE_IOU)
        n_added += added
        if merged:
            frames.append(pred_to_df(path.stem, merged))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "PROMOTE" if decision == PROMOTE else "DONOTUPLOAD"
    out = Path(args.out) if args.out else FAST_SUBMISSION_DIR / f"submission_slot1_unet_resnet34_merged_{tag}.csv"
    df.to_csv(out, index=False)
    meta = {
        "do_not_upload": decision != PROMOTE,
        "gate": decision,
        "gate_json": str(gate_path),
        "slot1_weights": str(SLOT1_WEIGHTS),
        "slot1_source": slot1_source,
        "slot1_re_inferred_yolo": slot1_by_stem is None,
        "unet_weights": str(weights.resolve()),
        "builder": "build_resnet34_unet",
        "not_tinyunet": True,
        "not_unet_semantic": True,
        "thr": args.thr,
        "min_area": int(args.min_area),
        "n_images": len(images),
        "n_rows": int(len(df)),
        "n_unet_added": n_added,
        "merge_iou": MERGE_IOU,
        "path": str(out),
        "leakage": "test JPEGs only; no MAGFiLO public-test labels",
        "never_submits": True,
        "kaggle_submit_called": False,
    }
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(UNET_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thr", type=float, default=0.45)
    parser.add_argument(
        "--min-area",
        type=int,
        default=1034,
        help="Must match locked_row.min_area. Default 1034 (hose-safe v1 replay floor).",
    )
    parser.add_argument("--gate-json", default=str(GATE))
    parser.add_argument(
        "--research-csv",
        action="store_true",
        help="Write a do-not-upload CSV even if gate is not PROMOTE_LOOK.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse reason (if any) and exit 0. Does not import torch.",
    )
    args = parser.parse_args()

    gate_path = Path(args.gate_json)
    reason = refuse(gate_path, research_csv=args.research_csv)
    if args.dry_refuse:
        if reason:
            print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(1)

    decision, payload = read_gate(gate_path)
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if isinstance(locked_row, dict):
        if locked_row.get("thr") is not None:
            args.thr = float(locked_row["thr"])
        if locked_row.get("min_area") is not None:
            args.min_area = int(locked_row["min_area"])
    else:
        locked = payload.get("locked") if isinstance(payload, dict) else None
        if locked is not None:
            args.thr = float(locked)

    run_infer(args, decision, gate_path)


if __name__ == "__main__":
    main()
