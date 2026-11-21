#!/usr/bin/env python3
"""Build a test CSV: frozen Slot 1 NMS-merged with patch-U-Net instances.

refuse_patch() first — this family may train only after the v1 merge gate is
KILL_NO_NEW_TP. The CSV itself is allowed only after the *patch merge* gate
(eval_unet_patch_merge.py → artifacts/unet_patch_slot1_merge_fold0_official.json)
is PROMOTE_LOOK, unless --research-csv (local do-not-upload).

Infer when allowed: CLAHE@2048 → 9 crops → stitch_max → one full-disk CC →
NMS-merge extras onto frozen Slot 1. Helpers come from eval_unet_patch
(infer_stitched_prob, decode_instances_stitched) AFTER refuse.

Weights: WEIGHTS_DIR/unet_patch_dhebe/best.pt — never unet_semantic.
Never uploads. Never reads MAGFiLO public-test labels. Test JPEGs only.
Refuse while live_mps_owner() is set. --dry-refuse exits 0 without torch.
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
from gate_unet_family import GATE, PROMOTE, live_mps_owner, read_gate, refuse_patch, resolve_start_gate

DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_patch_dhebe" / "best.pt"
WEIGHTS_NAME = "unet_patch_dhebe"
WEIGHTS_MISSING = "weights not trained yet"
MERGE_GATE = ARTIFACTS / "unet_patch_slot1_merge_fold0_official.json"
SLOT1_WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_1280" / "weights" / "best.pt"
SLOT1_CONF = 0.35
SLOT1_IMGSZ = 1280
# Copied from filament.merge — do not import that module on the refuse path.
MERGE_IOU = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse_patch() reason and exit 0. Do not infer.",
    )
    parser.add_argument(
        "--gate-json",
        default=str(GATE),
        help="v1 merge official JSON for refuse_patch() (need KILL_NO_NEW_TP).",
    )
    parser.add_argument(
        "--merge-gate-json",
        default=str(MERGE_GATE),
        help="Patch merge official JSON from eval_unet_patch_merge.py (need PROMOTE_LOOK).",
    )
    parser.add_argument(
        "--research-csv",
        action="store_true",
        help="Write a do-not-upload CSV even if the patch merge gate is not PROMOTE_LOOK.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--thr", type=float, default=0.45)
    parser.add_argument(
        "--min-area",
        type=int,
        default=1034,
        help="Must match locked_row.min_area. Default 1034 (hose-safe v1 replay floor).",
    )
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def refuse_weights(path: Path) -> str | None:
    if path.name and "unet_semantic" in path.parts:
        return "Refusing: weights must be unet_patch_dhebe, never unet_semantic."
    if path.parent.name != WEIGHTS_NAME:
        return "Refusing: weights dir must be unet_patch_dhebe, never unet_semantic."
    return None


def merge_gate_decision(path: Path) -> tuple[str | None, float | None]:
    decision, payload = read_gate(path)
    locked = payload.get("locked") if isinstance(payload, dict) else None
    thr = float(locked) if locked is not None and "error" not in payload else None
    return decision, thr


def run_test_merge(args: argparse.Namespace, decision: str | None, merge_gate: Path) -> None:
    """CLAHE@2048, 9 crops, stitch_max, one CC, NMS-merge onto frozen Slot1.

    First line re-checks refuse_patch() / live_mps_owner so torch is never
    imported on a closed gate.
    """
    reason = refuse_patch(resolve_start_gate(Path(args.gate_json)))
    if reason:
        raise SystemExit(reason)
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing patch U-Net start: {owner} owns MPS.")

    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet_patch import decode_instances_stitched, infer_stitched_prob
    from filament.infer import pick_device, run_yolo_infer
    from filament.merge import merge_rles
    from filament.metric import pred_to_df
    from filament.patch_grid import patch_boxes
    from filament.slot1_csv import SLOT1_TEST_CSV, load_slot1_test_rles
    from filament.unet import build_fulldisk_unet

    weights = Path(args.weights)
    bad = refuse_weights(weights)
    if bad:
        raise SystemExit(bad)
    if not weights.exists():
        raise SystemExit(WEIGHTS_MISSING)
    device = pick_device(args.device)
    boxes = patch_boxes()
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

    slot1 = None
    if slot1_by_stem is None:
        from ultralytics import YOLO

        slot1 = YOLO(str(SLOT1_WEIGHTS))
    unet = build_fulldisk_unet(base=args.base)
    unet.load_state_dict(torch.load(weights, map_location=device))
    unet.to(device)
    unet.eval()

    frames = []
    n_added = 0
    for path in tqdm(images, desc="test merge Slot1+patch-UNet"):
        if slot1_by_stem is not None:
            slot_rles = slot1_by_stem[path.stem]
        else:
            slot_rles = run_yolo_infer(
                slot1, path, conf=SLOT1_CONF, imgsz=SLOT1_IMGSZ, merge_gap=0, device=device
            )
        extra = decode_instances_stitched(
            infer_stitched_prob(unet, path, device, boxes),
            args.thr,
            min_area=int(args.min_area),
        )
        merged, added = merge_rles(slot_rles, extra, iou_thr=MERGE_IOU)
        n_added += added
        if merged:
            frames.append(pred_to_df(path.stem, merged))

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    )
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "PROMOTE" if decision == PROMOTE else "DONOTUPLOAD"
    out = Path(args.out) if args.out else FAST_SUBMISSION_DIR / f"submission_slot1_unet_patch_merged_{tag}.csv"
    df.to_csv(out, index=False)
    meta = {
        "do_not_upload": decision != PROMOTE,
        "gate": decision,
        "start_gate_json": str(Path(args.gate_json)),
        "gate_json": str(merge_gate),
        "slot1_weights": str(SLOT1_WEIGHTS),
        "slot1_source": slot1_source,
        "slot1_re_inferred_yolo": slot1_by_stem is None,
        "unet_weights": str(weights.resolve()),
        "weights_dir": WEIGHTS_NAME,
        "overwrote_unet_semantic": False,
        "thr": args.thr,
        "min_area": int(args.min_area),
        "n_images": len(images),
        "n_rows": int(len(df)),
        "n_unet_added": n_added,
        "n_patches_per_jpeg": len(boxes),
        "stitch": "stitch_max",
        "decode": "one full-disk CC after stitch",
        "merge_iou": MERGE_IOU,
        "path": str(out),
        "leakage": "test JPEGs only; no MAGFiLO public-test labels",
        "never_submits": True,
    }
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    start_gate = Path(args.gate_json)
    merge_gate = Path(args.merge_gate_json)

    start_gate = resolve_start_gate(start_gate)
    args.gate_json = str(start_gate)
    reason = refuse_patch(start_gate)
    if args.dry_refuse:
        if reason:
            print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(1)

    owner = live_mps_owner()
    if owner:
        print(f"Refusing patch U-Net start: {owner} owns MPS.", file=sys.stderr)
        raise SystemExit(1)

    decision, payload = read_gate(merge_gate)
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if isinstance(locked_row, dict):
        if locked_row.get("thr") is not None:
            args.thr = float(locked_row["thr"])
        if locked_row.get("min_area") is not None:
            args.min_area = int(locked_row["min_area"])
    else:
        _, locked = merge_gate_decision(merge_gate)
        if locked is not None:
            args.thr = locked
    if decision != PROMOTE and not args.research_csv:
        raise SystemExit(
            f"Gate is {decision!r} (need PROMOTE_LOOK). Pass --research-csv only for a local do-not-upload file."
        )
    if int(args.min_area) < 642:
        raise SystemExit(
            f"Refusing patch CSV: min_area {args.min_area} < 642 (hose floor)."
        )

    weights = Path(args.weights)
    bad = refuse_weights(weights)
    if bad:
        raise SystemExit(bad)
    if not weights.exists():
        print(WEIGHTS_MISSING, file=sys.stderr)
        raise SystemExit(1)

    run_test_merge(args, decision, merge_gate)


if __name__ == "__main__":
    main()
