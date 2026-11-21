#!/usr/bin/env python3
"""Gated spine-v2 U-Net decode eval (rank-3 family: unet_spine_watershed_v2).

Opens only if gate.decision == RESEARCH_HOLD_SPINE_V2.

Builders:
  tinyunet  — v1 GATE + unet_semantic/best.pt via build_fulldisk_unet (default).
              v1 merge is KILL_WORSE_OR_DESERT, so this path dry-refuses.
  resnet34  — ResNet merge GATE + unet_resnet34/best.pt via build_resnet34_unet.
              Never loads unet_semantic / unet_patch_dhebe / unet_3ch_disk.

If the gate opens: fold-0 logits, threshold, split_binary_on_ridge
(union_ridge_for_file from TRAIN_JSON), then score_pq.

Never submits. Never reads MAGFiLO public-test / Dataverse labels.
Does not import from eval_unet.py / eval_unet_merge.py / train_unet.py
(those stay spine-free). Does not start MPS from --dry-refuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, TRAIN_JSON, WEIGHTS_DIR
from gate_unet_family import GATE, RESNET_GATE

REQUIRED_DECISION = "RESEARCH_HOLD_SPINE_V2"
HOSE_FLOOR = 642
TINYUNET_WEIGHTS = WEIGHTS_DIR / "unet_semantic" / "best.pt"
RESNET_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
DEFAULT_WEIGHTS = TINYUNET_WEIGHTS
FORBIDDEN_RESNET_MARKERS = ("unet_semantic", "unet_patch_dhebe", "unet_3ch_disk")
if TRAIN_JSON.name != "MAGFiLO_1.0_Annotations_kaggle2026_train.json":
    raise SystemExit(f"Refusing spine-v2 decode: unexpected TRAIN_JSON {TRAIN_JSON}.")


def refuse_reason(gate_path: Path) -> str | None:
    """Why spine-v2 must not run. None means RESEARCH_HOLD_SPINE_V2 is set."""
    if not gate_path.exists():
        return (
            f"Refusing spine-v2 decode: missing {gate_path} "
            f"(need gate.decision == {REQUIRED_DECISION})."
        )
    try:
        payload = json.loads(gate_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"Refusing spine-v2 decode: cannot read {gate_path} ({exc})."
    gate = payload.get("gate") if isinstance(payload, dict) else None
    decision = gate.get("decision") if isinstance(gate, dict) else None
    if decision != REQUIRED_DECISION:
        return (
            f"Refusing spine-v2 decode: gate.decision is {decision!r} "
            f"(need {REQUIRED_DECISION})."
        )
    return None


def reject_resnet_spine_weights(weights: Path) -> str | None:
    """ResNet spine path may only load unet_resnet34. None means the path is legal."""
    text = str(weights)
    parts = set(Path(weights).parts)
    if any(marker in text for marker in FORBIDDEN_RESNET_MARKERS):
        return (
            f"Refusing spine-v2 ResNet decode: weights {weights} are not unet_resnet34 "
            f"(forbidden marker in path)."
        )
    if "unet_resnet34" not in parts and "unet_resnet34" not in text:
        return (
            f"Refusing spine-v2 ResNet decode: weights must be unet_resnet34 "
            f"(got {weights})."
        )
    return None


def parse_min_areas(min_areas: str, min_area: int, *, hose_floor: int = HOSE_FLOOR) -> list[int]:
    """Union of --min-areas and --min-area. Refuses any value below the hose floor."""
    vals = sorted(
        {int(x.strip()) for x in str(min_areas).split(",") if x.strip()} | {int(min_area)}
    )
    bad = [v for v in vals if v < hose_floor]
    if bad:
        raise SystemExit(
            f"Refusing spine-v2 min_area {bad} < {hose_floor} (hose floor)."
        )
    return vals


def decode_instances_spine_v2(prob, thr: float, ridge, min_area: int = 64) -> list[str]:
    """Threshold then cut CC on a train-JSON union ridge. Not v1 decode_instances."""
    from filament.rle import encode_rle
    from filament.spine_split import split_binary_on_ridge

    binary = (prob >= thr).astype("uint8")
    masks = split_binary_on_ridge(binary, ridge, min_area=int(min_area))
    return [encode_rle(m) for m in masks if int(m.sum()) >= int(min_area)]


def _infer_prob(model, path: Path, imgsz: int, device: str):
    """Gray-as-RGB @imgsz → native-2048 prob. Local copy; do not import eval_unet."""
    import cv2
    import numpy as np
    import torch

    from filament.paths import IMAGE_SIZE
    from filament.preprocess import read_gray
    from filament.semantic_dataset import gray_as_rgb

    gray = read_gray(path)
    small = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray_as_rgb(small)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)[0, 0].detach().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logit))
    return cv2.resize(prob.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def _build_model(builder: str, base: int):
    if builder == "resnet34":
        from filament.unet_resnet import build_resnet34_unet

        return build_resnet34_unet(in_ch=3, pretrained=False)
    if builder != "tinyunet":
        raise SystemExit(f"Refusing spine-v2 decode: unknown builder {builder!r}.")
    from filament.unet import build_fulldisk_unet

    return build_fulldisk_unet(base=base)


def resolve_spine_v2_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill gate/weights/min_area from --builder when the flags were omitted."""
    builder = args.builder
    if not args.gate_json:
        args.gate_json = str(RESNET_GATE if builder == "resnet34" else GATE)
    if not args.weights:
        args.weights = str(RESNET_WEIGHTS if builder == "resnet34" else TINYUNET_WEIGHTS)
    if getattr(args, "min_area", None) is None:
        args.min_area = HOSE_FLOOR if builder == "resnet34" else 64
    return args


def run_infer_and_score(args: argparse.Namespace, gate_payload: dict) -> None:
    """Fold-0 infer + ridge-split + score_pq. Never submit. Train-JSON spines only."""
    import pandas as pd
    import torch
    from tqdm import tqdm

    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.paths import FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR
    from filament.spine_io import load_spines_by_file, union_ridge_for_file
    from gate_unet_family import live_mps_owner
    from score_family import year_slices

    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing spine-v2 infer: {owner} owns MPS.")

    weights = Path(args.weights)
    if args.builder == "resnet34":
        bad = reject_resnet_spine_weights(weights)
        if bad:
            raise SystemExit(bad)
        if int(args.min_area) < HOSE_FLOOR:
            raise SystemExit(
                f"Refusing spine-v2 ResNet decode: min_area {args.min_area} < {HOSE_FLOOR} (hose floor)."
            )
    if not weights.exists():
        raise FileNotFoundError(weights)
    device = pick_device(getattr(args, "device", "auto"))
    thrs = [float(x) for x in args.thrs.split(",") if x.strip()]
    min_area = int(args.min_area)

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    spines = load_spines_by_file(TRAIN_JSON)
    ridges = {fn: union_ridge_for_file(fn, spines) for fn in val_files}

    model = _build_model(args.builder, args.base)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    probs = {}
    for file_name in tqdm(val_files, desc=f"spine-v2 {args.builder} probs"):
        probs[file_name] = _infer_prob(model, TRAIN_IMG_DIR / file_name, args.imgsz, device)

    sweep = []
    pred_cache = {}
    for thr in thrs:
        frames = {}
        for file_name in val_files:
            rles = decode_instances_spine_v2(
                probs[file_name], thr, ridges[file_name], min_area=min_area
            )
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
                "min_area": min_area,
            }
        )
        print(sweep[-1], flush=True)

    locked = max(sweep, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45)))
    slices = year_slices(val_files, pred_cache[locked["thr"]], views, by_file)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "resnet" if args.builder == "resnet34" else "v1"
    pred_csv = FAST_ARTIFACTS / f"unet_spine_v2_{tag}_fold{args.fold}_thr{locked['thr']:.2f}_pred.csv"
    pd.concat(pred_cache[locked["thr"]].values(), ignore_index=True).to_csv(pred_csv, index=False)
    dest_name = (
        f"unet_spine_v2_resnet_fold{args.fold}_official.json"
        if args.builder == "resnet34"
        else f"unet_spine_v2_fold{args.fold}_official.json"
    )
    out = {
        "family": (
            "unet_spine_watershed_v2_resnet"
            if args.builder == "resnet34"
            else "unet_spine_watershed_v2"
        ),
        "weights": str(weights.resolve()),
        "builder": "build_resnet34_unet" if args.builder == "resnet34" else "build_fulldisk_unet",
        "spine_source": str(TRAIN_JSON),
        "leakage": "train JSON spines only; no MAGFiLO public-test labels",
        "fold": args.fold,
        "n_val": len(val_files),
        "sweep": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "min_area": min_area,
        "year_slice": slices,
        "gate_payload_decision": (gate_payload.get("gate") or {}).get("decision"),
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
        "not_patch_stitch": True,
        "not_3ch_stack": True,
    }
    dest = ARTIFACTS / dest_name
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(dest), "locked": locked, "do_not_submit": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse_reason (None if RESEARCH_HOLD_SPINE_V2). Do not infer.",
    )
    parser.add_argument(
        "--builder",
        choices=("tinyunet", "resnet34"),
        default="tinyunet",
        help="tinyunet: v1 GATE + unet_semantic. resnet34: ResNet GATE + unet_resnet34.",
    )
    parser.add_argument(
        "--gate-json",
        default="",
        help="Default: v1 GATE (tinyunet) or ResNet merge GATE (resnet34).",
    )
    parser.add_argument("--weights", default="")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.35,0.40,0.45,0.50")
    parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help="Standalone decode floor. Default 64 (tinyunet) or 642 (resnet34 hose floor).",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    resolve_spine_v2_defaults(args)

    gate_path = Path(args.gate_json)
    reason = refuse_reason(gate_path)

    if args.dry_refuse:
        print(reason)
        return

    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    payload = json.loads(gate_path.read_text())
    run_infer_and_score(args, payload)


if __name__ == "__main__":
    main()
