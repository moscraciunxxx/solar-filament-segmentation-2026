"""Leak-free logistic rejector on 3ch extras only. Never submits.

Same recipe as resnet_extra_filter: fit on fold!=0 extras, lock threshold by
fold!=0 precision-first recall>=0.4, then one fold-0 Slot1 merge.
Decoder is unet_3ch_disk + infer_prob_3ch. Dest: WEIGHTS_DIR/ch3_extra_filter.
Does not overwrite unet_3ch_disk or resnet_extra_filter.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from filament.merge import MERGE_IOU
from filament.paths import FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import CH3_FILTER_WEIGHTS_NAME, HOSE_FLOOR, filter_dest_illegal
from resnet_extra_filter import (
    append_extra_rows,
    assert_safe_filter_dest,
    features_for_rles,
    fit_extra_rejector,
    gt_masks_for_file,
    jpeg_and_residual,
    keep_extra_rles,
    load_extra_rows,
    load_slot1_covering,
    mine_file_extras,
    parse_min_areas,
    rows_to_xy,
    save_rejector,
    _done_files,
)

FILTER_DEST = WEIGHTS_DIR / CH3_FILTER_WEIGHTS_NAME
EXTRAS_JSONL = FAST_ARTIFACTS / "ch3_extras_foldne0.jsonl"
EXTRAS_META = FAST_ARTIFACTS / "ch3_extras_foldne0_meta.json"


def assert_safe_ch3_filter_dest(dest: Path) -> Path:
    dest = assert_safe_filter_dest(dest)
    if dest.resolve() == (WEIGHTS_DIR / "resnet_extra_filter").resolve():
        raise SystemExit(f"Refusing 3ch extra-filter dest {dest}: must not overwrite resnet_extra_filter.")
    if dest.resolve() == (WEIGHTS_DIR / "unet_3ch_disk").resolve():
        raise SystemExit(f"Refusing 3ch extra-filter dest {dest}: must not overwrite unet_3ch_disk.")
    bad = filter_dest_illegal(dest)
    if bad:
        raise SystemExit(bad)
    return dest


def run_mine(
    *,
    weights: Path,
    filter_dest: Path,
    imgsz: int,
    thrs: list[float],
    min_areas: list[int],
    device: str,
    exclude_fold: int,
    seed: int,
    train_img_dir: Path,
) -> dict:
    from tqdm import tqdm

    import torch
    from eval_unet import decode_instances
    from eval_unet_3ch import infer_prob_3ch
    from filament.coco_io import grouped_folds, load_coco
    from filament.infer import pick_device
    from filament.unet import build_fulldisk_unet
    from gate_unet_family import live_mps_owner

    assert_safe_ch3_filter_dest(filter_dest)
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing 3ch extra-filter mine: {owner} owns MPS.")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=seed)
    mine_files = sorted(fn for fn, fold in fold_of.items() if fold != exclude_fold)
    if not mine_files:
        raise SystemExit("Refusing 3ch extra-filter mine: no fold!=0 files.")

    device = pick_device(device)
    slot1 = load_slot1_covering(mine_files, device)
    model = build_fulldisk_unet(in_ch=3, base=32)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    done = _done_files(EXTRAS_META)
    n_new = 0
    n_rows = 0
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for file_name in tqdm(mine_files, desc="mine fold!=0 3ch extras"):
        if file_name in done:
            continue
        prob = infer_prob_3ch(model, train_img_dir / file_name, imgsz, device)
        cand = {
            (thr, min_area): decode_instances(prob, thr, min_area=min_area)
            for min_area in min_areas
            for thr in thrs
        }
        slot_rles = slot1[file_name]["segmentation_rle"].tolist()
        image, residual = jpeg_and_residual(file_name)
        rows = mine_file_extras(
            file_name,
            slot_rles,
            cand,
            gt_masks_for_file(file_name, views, by_file),
            image,
            residual,
            fold_of[file_name],
        )
        append_extra_rows(EXTRAS_JSONL, rows)
        done.add(file_name)
        n_new += 1
        n_rows += len(rows)
        EXTRAS_META.write_text(
            json.dumps(
                {
                    "exclude_fold": exclude_fold,
                    "seed": seed,
                    "weights": str(Path(weights).resolve()),
                    "filter_dest": str(Path(filter_dest)),
                    "thrs": thrs,
                    "min_areas": min_areas,
                    "done_files": sorted(done),
                    "n_done": len(done),
                    "n_target": len(mine_files),
                    "decoder": "unet_3ch_disk",
                    "classical_new_seeds": False,
                    "do_not_submit": True,
                    "leakage": "fold!=0 train JPEGs + train GT only; no MAGFiLO test labels",
                },
                indent=2,
            )
            + "\n"
        )

    all_rows = load_extra_rows(EXTRAS_JSONL)
    n_pos = sum(int(r["label"] == 1) for r in all_rows)
    n_neg = sum(int(r["label"] == 0) for r in all_rows)
    return {
        "path": str(EXTRAS_JSONL),
        "meta": str(EXTRAS_META),
        "n_mine_files": len(mine_files),
        "n_done": len(done),
        "n_new_files": n_new,
        "n_new_rows": n_rows,
        "n_rows": len(all_rows),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "exclude_fold": exclude_fold,
        "seed": seed,
        "filter_dest": str(Path(filter_dest)),
        "ch3_weights_read_only": str(Path(weights).resolve()),
        "overwrote_forbidden_unet": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }


def run_fold0_merge(
    *,
    weights: Path,
    filter_dest: Path,
    imgsz: int,
    thrs: list[float],
    min_areas: list[int],
    device: str,
    fold: int,
    seed: int,
    train_img_dir: Path,
) -> dict:
    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances
    from eval_unet_3ch import infer_prob_3ch
    from eval_unet_merge import gate_unet_merge
    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.merge import extra_rles, merge_rles
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.paths import ARTIFACTS
    from filament.unet import build_fulldisk_unet
    from gate_unet_family import CH3_EXTRA_FILTER_GATE, live_mps_owner
    from score_faint_merge import load_or_infer_slot1
    from score_family import year_slices
    from resnet_extra_filter import REJECTOR_THR

    assert_safe_ch3_filter_dest(filter_dest)
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing 3ch extra-filter fold-0 merge: {owner} owns MPS.")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=seed)
    val_files = sorted(fn for fn, f in fold_of.items() if f == fold)
    all_rows = load_extra_rows(EXTRAS_JSONL)
    rows = [
        r
        for r in all_rows
        if int(fold_of.get(r.get("file_name"), -1)) != fold and int(r.get("fold", -1)) != fold
    ]
    if not rows:
        raise SystemExit(
            f"Refusing 3ch extra-filter fold-0 merge: no fold!={fold} extras at {EXTRAS_JSONL}. "
            "Run scripts/mine_3ch_extras.py --run first."
        )
    if any(int(fold_of.get(r.get("file_name"), -1)) == fold for r in rows):
        raise SystemExit("Refusing 3ch extra-filter: a fit row belongs to the official fold.")

    x, y = rows_to_xy(rows)
    clf, rejector_meta = fit_extra_rejector(x, y)
    rejector_path = save_rejector(filter_dest, clf, rejector_meta)
    thr_keep = float(rejector_meta.get("threshold", REJECTOR_THR))

    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    device = pick_device(device)
    slot1_frames = load_or_infer_slot1(val_files, device)
    slot1_df = pd.concat(slot1_frames.values(), ignore_index=True)
    slot1_br = score_pq(gt_df, slot1_df)
    slot1_row = {
        "pq": slot1_br.pq,
        "tp": slot1_br.tp,
        "fp": slot1_br.fp,
        "fn": slot1_br.fn,
        "mean_tp_iou": slot1_br.mean_tp_iou,
        "n_pred": int(len(slot1_df)),
    }
    slot1_slices = year_slices(val_files, slot1_frames, views, by_file)

    model = build_fulldisk_unet(in_ch=3, base=32)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    probs = {}
    jpeg_cache = {}
    for file_name in tqdm(val_files, desc="extra-filter fold-0 3ch probs"):
        probs[file_name] = infer_prob_3ch(model, train_img_dir / file_name, imgsz, device)
        jpeg_cache[file_name] = jpeg_and_residual(file_name)

    sweep = []
    pred_cache = {}
    for min_area in min_areas:
        for thr in thrs:
            frames = {}
            n_added = 0
            n_extras = 0
            n_kept = 0
            for file_name in val_files:
                cands = decode_instances(probs[file_name], thr, min_area=min_area)
                slot_rles = slot1_frames[file_name]["segmentation_rle"].tolist()
                extras = extra_rles(slot_rles, cands, iou_thr=MERGE_IOU)
                n_extras += len(extras)
                image, residual = jpeg_cache[file_name]
                feats = features_for_rles(extras, image, residual)
                kept = keep_extra_rles(extras, feats, clf, thr_keep)
                n_kept += len(kept)
                merged, added = merge_rles(slot_rles, kept)
                n_added += added
                frames[file_name] = pred_to_df(Path(file_name).stem, merged)
            pred_cache[(thr, min_area)] = frames
            pred_df = pd.concat(frames.values(), ignore_index=True)
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
                    "n_unet_added": n_added,
                    "n_extras_before_filter": n_extras,
                    "n_extras_kept": n_kept,
                    "min_area": min_area,
                    "rejector_thr": thr_keep,
                }
            )
            print(sweep[-1], flush=True)

    fp_cap = 2 * int(slot1_br.fp)
    non_exploded = [r for r in sweep if int(r["fp"]) < fp_cap]
    pool = non_exploded or sweep
    locked = max(pool, key=lambda r: (r["pq"], -abs(r["thr"] - 0.45), r["min_area"]))
    slices = year_slices(
        val_files, pred_cache[(locked["thr"], locked["min_area"])], views, by_file
    )
    decision = gate_unet_merge(locked, slot1_row, slices["2017-2020"], slot1_slices["2017-2020"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = CH3_EXTRA_FILTER_GATE
    out = {
        "family": "unet_3ch_extra_filter_slot1_merge",
        "scored": True,
        "status": "fold0_official",
        "weights": str(Path(weights).resolve()),
        "builder": "build_fulldisk_unet + logistic_morphometrics extras rejector",
        "decode": "3ch CC extras, fold!=0-trained rejector, then Slot1 NMS-merge",
        "filter_dest": str(Path(filter_dest).resolve()),
        "rejector_path": str(rejector_path),
        "rejector": rejector_meta,
        "ch3_weights_read_only": True,
        "overwrote_forbidden_unet": False,
        "classical_new_seeds": False,
        "not_resnet_extra_filter_retune": True,
        "leakage": (
            "fold!=0 3ch extras + train GT labels only; fold-0 is the official merge. "
            "No MAGFiLO test labels. Rejector lock is fold!=0 precision-first, not fold-0 PQ."
        ),
        "fold": fold,
        "fit_folds": f"!={fold}",
        "seed": seed,
        "n_val": len(val_files),
        "n_fit_extras": int(len(rows)),
        "slot1": slot1_row,
        "sweep_merged": sweep,
        "locked": locked["thr"],
        "locked_row": locked,
        "year_slice": slices,
        "gate": decision,
        "min_areas": min_areas,
        "min_area": locked["min_area"],
        "hose_floor": HOSE_FLOOR,
        "lock_rule": "max PQ among FP < 2*Slot1; else max PQ (honest hose fail)",
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    text = json.dumps(out, indent=2) + "\n"
    dest.write_text(text)
    (FAST_ARTIFACTS / dest.name).write_text(text)
    return out
