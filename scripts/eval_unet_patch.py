#!/usr/bin/env python3
"""Official fold-0 instance PQ for native-2048 patch U-Net.

Pipeline when allowed: CLAHE@2048 → 9 crops → TinyUNet → stitch_max →
solar_disk_mask → one full-disk CC → score_pq. Not patch Dice.

Cannot infer until refuse_patch() is None (KILL_NO_NEW_TP and no MPS
owner) AND WEIGHTS_DIR/unet_patch_dhebe/best.pt exists. Never submits.
Never reads MAGFiLO public-test / Dataverse labels. Does not import
patch_grid from eval_unet.py / eval_unet_merge.py. Does not edit v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import GATE, refuse_patch, resolve_start_gate

DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_patch_dhebe" / "best.pt"
WEIGHTS_MISSING = "weights not trained yet"
# Copied from score_family — do not import that module on the refuse path (it pulls metric/torch).
M_1280 = {"pq": 0.410, "tp": 925, "fp": 443, "fn": 664}
S_MEDOID = {"pq": 0.357, "tp": 809, "fp": 475, "fn": 780}


def gray_as_rgb(gray):
    import numpy as np

    g = gray.astype(np.float32) / 255.0
    return np.stack([g, g, g], axis=0)


def decode_instances_stitched(prob, thr: float, min_area: int = 64) -> list[str]:
    """One full-disk CC after stitch_max. Not per-patch instances."""
    from filament.postprocess import connected_instances, resolve_overlaps
    from filament.rle import encode_rle

    binary = (prob >= thr).astype("uint8")
    masks = connected_instances(binary)
    masks = resolve_overlaps(masks)
    return [encode_rle(m) for m in masks if int(m.sum()) >= int(min_area)]


def infer_stitched_prob(model, path: Path, device: str, boxes):
    """CLAHE once, 9 native crops, sigmoid, stitch_max, zero off-limb."""
    import numpy as np
    import torch

    from filament.patch_grid import crop_patch, stitch_max
    from filament.preprocess import apply_clahe, read_gray, solar_disk_mask

    gray = read_gray(path)
    clahe = apply_clahe(gray, clip_limit=2.0, tile=8)
    patches = []
    for box in boxes:
        crop = crop_patch(clahe, box)
        tensor = torch.from_numpy(gray_as_rgb(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)[0, 0].detach().cpu().numpy()
        patches.append(1.0 / (1.0 + np.exp(-logit.astype(np.float32))))
    stitched = stitch_max(patches, boxes)
    disk = solar_disk_mask(gray)
    return stitched * disk.astype(np.float32)


def gate_unet_patch(best: dict) -> dict:
    """Standalone diagnosis vs Slot-1 numbers. Merge script owns replace-Slot-1."""
    pq = float(best["pq"])
    tp = int(best["tp"])
    fp = int(best["fp"])
    d_pq = pq - M_1280["pq"]
    d_tp = tp - M_1280["tp"]
    fp_exploded = fp >= 2 * M_1280["fp"]
    if pq >= 0.42 and not fp_exploded:
        decision = "PROMOTE_LOOK"
        next_action = (
            "Patch U-Net standalone cleared 0.42. Official replace is the "
            "Slot-1 merge script. Do not submit from here."
        )
    elif pq < 0.20 or d_pq <= -0.08:
        decision = "KILL"
        next_action = "Patch U-Net collapsed. Do not upload. Do not retune from public."
    elif d_tp <= 0 and pq <= M_1280["pq"]:
        decision = "FAIL_NO_NEW_PROPOSALS"
        next_action = "Did not add TPs vs Slot 1. Do not upload."
    elif 0.38 <= pq < 0.42:
        decision = "RESEARCH_HOLD_SPINE_V2"
        next_action = "PQ is real but below Slot-1 replace. Merge gate decides."
    else:
        decision = "RESEARCH_HOLD"
        next_action = "Diagnosis only. Do not upload. Year-shift stays Slot 2."
    return {
        "decision": decision,
        "next_action": next_action,
        "vs_m1280": {"d_pq": d_pq, "d_tp": d_tp, "fp_exploded": fp_exploded},
        "vs_s_medoid_d_pq": pq - S_MEDOID["pq"],
        "do_not_submit_from_this_script": True,
        "do_not_replace_slot2": True,
        "official_is_instance_pq_after_stitch_max_not_patch_dice": True,
    }


def run_infer_and_score(args: argparse.Namespace) -> None:
    """Fold-0 stitch + one CC + score_pq. Entered only after refuse is None and weights exist."""
    import pandas as pd
    import torch
    from tqdm import tqdm

    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.patch_grid import patch_boxes
    from filament.unet import build_fulldisk_unet
    from score_family import year_slices

    weights = Path(args.weights)
    device = pick_device(args.device)
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    boxes = patch_boxes()

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    model = build_fulldisk_unet(base=args.base)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc="patch stitch_max"):
        probs[file_name] = infer_stitched_prob(
            model, TRAIN_IMG_DIR / file_name, device, boxes
        )

    sweep = []
    pred_cache: dict[float, dict] = {}
    for thr in thrs:
        frames = {}
        for file_name in val_files:
            rles = decode_instances_stitched(probs[file_name], thr)
            frames[file_name] = pred_to_df(Path(file_name).stem, rles)
        pred_cache[thr] = frames
        pred_df = (
            pd.concat(frames.values(), ignore_index=True)
            if frames
            else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
        )
        br = score_pq(gt_df, pred_df)
        sweep.append(
            {
                "thr": thr,
                "pq": br.pq,
                "tp": br.tp,
                "fp": br.fp,
                "fn": br.fn,
                "mean_tp_iou": br.mean_tp_iou,
                "n_pred": int(len(pred_df)),
            }
        )
        print(sweep[-1], flush=True)

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45)))
    slices = year_slices(val_files, pred_cache[locked["thr"]], views, by_file)
    decision = gate_unet_patch(locked)

    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pred_csv = FAST_ARTIFACTS / f"unet_patch_dhebe_fold{args.fold}_thr{locked['thr']:.2f}_pred.csv"
    pd.concat(pred_cache[locked["thr"]].values(), ignore_index=True).to_csv(pred_csv, index=False)
    out = {
        "family": "unet_patch_native_1024_dhebe",
        "weights": str(weights.resolve()),
        "input": "CLAHE gray-as-RGB, native 2048 crops, stitch_max, disk mask, one CC",
        "device": device,
        "metric": "official host PQ after stitch_max + one full-disk CC (not patch Dice)",
        "fold": args.fold,
        "n_val": len(val_files),
        "n_patches_per_jpeg": len(boxes),
        "sweep": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "baselines": {"s_medoid": S_MEDOID, "m_1280": M_1280},
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
    }
    text = json.dumps(out, indent=2)
    (FAST_ARTIFACTS / f"unet_patch_dhebe_fold{args.fold}_official.json").write_text(text)
    repo = ARTIFACTS / f"unet_patch_dhebe_fold{args.fold}_official.json"
    repo.write_text(text)
    print(json.dumps({"path": str(repo), "locked": locked, "gate": decision}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse_patch() reason and exit 0. Do not infer.",
    )
    parser.add_argument("--gate-json", default=str(GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--thrs", default="0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    args = parser.parse_args()

    reason = refuse_patch(resolve_start_gate(Path(args.gate_json)))
    if args.dry_refuse:
        if reason:
            print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    weights = Path(args.weights)
    if not weights.exists():
        print(WEIGHTS_MISSING, file=sys.stderr)
        raise SystemExit(2)

    run_infer_and_score(args)


if __name__ == "__main__":
    main()
