#!/usr/bin/env python3
"""Inventory unused spine + chirality fields in the COMPETITION TRAIN JSON.

CPU-only. Train JSON only. Does not train, submit, read test labels, or touch
yolo_seg / yolo_seg_all / yolo_seg_phys / yolo_seg_rater / live weights.
Does not edit yolo_export.py or prepare_yolo.py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id, polygon_to_xy, rasterize_polygons
from filament.paths import ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_JSON

CAT_NAME = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}
CHIRALITY = {1: "L", 2: "R", 3: "unidentifiable", 4: "ambiguous"}
FULL_MAGFILO_PAPER = {"L": 3128, "R": 3273, "unidentifiable": 3843, "note": "Ahmadzadeh Sci Data 2024 full MAGFiLO, not this Kaggle train slice"}
SAMPLE_N = 50
SAMPLE_MIN_VERTICES = 3  # multi-point polyline, not a 2-vertex stub


def spine_to_xy(spine) -> np.ndarray:
    arr = np.asarray(spine, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return arr.reshape(-1, 2)


def numeric_summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def points_inside_mask(pts: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    h, w = mask.shape
    n_in = 0
    n_oob = 0
    for x, y in np.round(pts).astype(np.int32):
        if x < 0 or y < 0 or x >= w or y >= h:
            n_oob += 1
            continue
        if mask[y, x]:
            n_in += 1
    return n_in, int(len(pts)), n_oob


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def cat_hist(ids) -> dict:
    c = Counter(ids)
    return {
        "1_Left": int(c.get(1, 0)),
        "2_Right": int(c.get(2, 0)),
        "3_Unidentifiable": int(c.get(3, 0)),
        "4_Ambiguous": int(c.get(4, 0)),
        "other": int(sum(v for k, v in c.items() if k not in (1, 2, 3, 4))),
        "n": int(sum(c.values())),
        "L_share": (c.get(1, 0) / sum(c.values())) if c else None,
        "R_share": (c.get(2, 0) / sum(c.values())) if c else None,
        "unidentifiable_share": (c.get(3, 0) / sum(c.values())) if c else None,
        "identifiable_LR": int(c.get(1, 0) + c.get(2, 0)),
    }


def split_block(rows: list[dict]) -> dict:
    n = len(rows)
    n_spine = sum(1 for r in rows if r["n_spine_vertices"] >= 2)
    n_multi = sum(1 for r in rows if r["n_spine_vertices"] >= SAMPLE_MIN_VERTICES)
    n_odd = sum(1 for r in rows if r["spine_list_len"] % 2)
    return {
        "n_filaments": n,
        "n_with_spine_key": n,
        "n_spine_ge2_vertices": n_spine,
        "n_spine_ge3_vertices_multipoint": n_multi,
        "n_odd_length_spine_lists": n_odd,
        "n_physical_jpegs": len({r["file_name"] for r in rows}),
        "n_annotator_images": len({r["image_id"] for r in rows}),
        "category_id_histogram": cat_hist(r["category_id"] for r in rows),
        "chirality_from_category_id": {
            "L": sum(1 for r in rows if r["category_id"] == 1),
            "R": sum(1 for r in rows if r["category_id"] == 2),
            "unidentifiable": sum(1 for r in rows if r["category_id"] == 3),
            "ambiguous": sum(1 for r in rows if r["category_id"] == 4),
            "missing_category": sum(1 for r in rows if r["category_id"] is None),
        },
        "spine_vertices": numeric_summary([r["n_spine_vertices"] for r in rows]),
        "spine_list_len": numeric_summary([r["spine_list_len"] for r in rows]),
    }


def pairwise_category_agree(ann_lists: list[list[dict]], iou_thr: float = 0.3) -> dict:
    """Greedy bbox match; train-only multi-annotator chirality noise probe."""
    agree = 0
    matched = 0
    pairs = 0
    for i in range(len(ann_lists)):
        for j in range(i + 1, len(ann_lists)):
            A, B = ann_lists[i], ann_lists[j]
            used_b = set()
            pairs += 1
            for a in A:
                best_k, best_iou = None, iou_thr
                for k, b in enumerate(B):
                    if k in used_b:
                        continue
                    iou = bbox_iou(a["bbox"], b["bbox"])
                    if iou > best_iou:
                        best_k, best_iou = k, iou
                if best_k is None:
                    continue
                used_b.add(best_k)
                matched += 1
                if a["category_id"] == B[best_k]["category_id"]:
                    agree += 1
    return {
        "n_annotator_pairs": pairs,
        "n_matched_bbox_iou_ge_0.3": matched,
        "n_category_agree": agree,
        "category_agree_rate": (agree / matched) if matched else None,
        "note": "Chirality is noisy even when masks overlap; do not treat L/R as a hard PQ target.",
    }


def main() -> None:
    assert TRAIN_JSON.name == "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    with TRAIN_JSON.open() as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    img_by_id = {im["id"]: im for im in images}

    # Schema / coverage (train JSON only)
    ann_key_sets = Counter(tuple(sorted(a.keys())) for a in annotations)
    img_key_sets = Counter(tuple(sorted(im.keys())) for im in images)
    chirality_like = [k for k in ("chirality", "handed", "handedness", "barb", "barbs") if any(k in a for a in annotations)]

    rows = []
    for a in annotations:
        im = img_by_id[a["image_id"]]
        spine = a.get("spine")
        pts = spine_to_xy(spine) if spine is not None else np.zeros((0, 2), dtype=np.float32)
        rows.append(
            {
                "id": a["id"],
                "image_id": a["image_id"],
                "file_name": im["file_name"],
                "category_id": a.get("category_id"),
                "spine_present": spine is not None,
                "spine_list_len": len(spine) if isinstance(spine, list) else 0,
                "n_spine_vertices": int(len(pts)),
                "spine": pts,
                "polygon": polygon_to_xy(a["segmentation"]),
                "bbox": a["bbox"],
            }
        )

    views, by_file = load_coco(TRAIN_JSON)
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    assert len(val_files) + len(train_files) == len(by_file)

    medoid_of = {fn: medoid_view_id(views, by_file[fn]) for fn in by_file}
    for r in rows:
        r["fold"] = int(fold_of[r["file_name"]])
        r["is_fold0"] = r["fold"] == 0
        r["is_medoid"] = r["image_id"] == medoid_of[r["file_name"]]

    all_rows = rows
    fold0_rows = [r for r in rows if r["is_fold0"]]
    rest_rows = [r for r in rows if not r["is_fold0"]]
    medoid_rows = [r for r in rows if r["is_medoid"]]
    fold0_medoid = [r for r in medoid_rows if r["is_fold0"]]
    rest_medoid = [r for r in medoid_rows if not r["is_fold0"]]

    # Multi-annotator chirality noise: train physicals only (no val leak into this probe)
    anns_by_view = defaultdict(list)
    for r in rest_rows:
        anns_by_view[r["image_id"]].append(r)
    train_multi_agree = []
    n_train_multi = 0
    for fn in train_files:
        ids = by_file[fn]
        if len(ids) < 2:
            continue
        n_train_multi += 1
        train_multi_agree.append(pairwise_category_agree([anns_by_view[i] for i in ids]))
    chir_noise = {
        "n_train_multi_annotator_physicals": n_train_multi,
        "n_annotator_pairs": sum(x["n_annotator_pairs"] for x in train_multi_agree),
        "n_matched_bbox_iou_ge_0.3": sum(x["n_matched_bbox_iou_ge_0.3"] for x in train_multi_agree),
        "n_category_agree": sum(x["n_category_agree"] for x in train_multi_agree),
        "category_agree_rate": (
            sum(x["n_category_agree"] for x in train_multi_agree)
            / sum(x["n_matched_bbox_iou_ge_0.3"] for x in train_multi_agree)
            if train_multi_agree and sum(x["n_matched_bbox_iou_ge_0.3"] for x in train_multi_agree)
            else None
        ),
        "split": "train_physicals_fold_ne_0_only",
        "note": "Do not compute this on fold-0 if it would be used to pick an aux loss weight.",
    }

    # Sample 50 multi-point spines on MEDOID views (the masks a later YOLO would train)
    rng = np.random.default_rng(SEED)
    pool = [r for r in medoid_rows if r["n_spine_vertices"] >= SAMPLE_MIN_VERTICES]
    sample_idx = rng.choice(len(pool), size=min(SAMPLE_N, len(pool)), replace=False)
    sample = [pool[int(i)] for i in sample_idx]

    sample_stats = []
    own_fracs = []
    union_fracs = []
    dilate_fracs = []
    n_all_in_own = 0
    n_all_in_union = 0
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Cache per-file medoid union for sampled files only
    union_cache: dict[str, np.ndarray] = {}
    for r in sample:
        fn = r["file_name"]
        if fn not in union_cache:
            sibs = [x for x in medoid_rows if x["file_name"] == fn]
            union_cache[fn] = rasterize_polygons([x["polygon"] for x in sibs], size=IMAGE_SIZE)
        own = rasterize_polygons([r["polygon"]], size=IMAGE_SIZE)
        own_d = cv2.dilate(own, kernel, iterations=1)
        n_in, n_pts, n_oob = points_inside_mask(r["spine"], own)
        n_in_u, _, _ = points_inside_mask(r["spine"], union_cache[fn])
        n_in_d, _, _ = points_inside_mask(r["spine"], own_d)
        own_f = n_in / n_pts if n_pts else 0.0
        union_f = n_in_u / n_pts if n_pts else 0.0
        dil_f = n_in_d / n_pts if n_pts else 0.0
        own_fracs.append(own_f)
        union_fracs.append(union_f)
        dilate_fracs.append(dil_f)
        if n_in == n_pts:
            n_all_in_own += 1
        if n_in_u == n_pts:
            n_all_in_union += 1
        sample_stats.append(
            {
                "id": r["id"],
                "file_name": fn,
                "image_id": r["image_id"],
                "fold": r["fold"],
                "is_fold0": r["is_fold0"],
                "category_id": r["category_id"],
                "n_spine_vertices": n_pts,
                "n_inside_own_polygon": n_in,
                "n_inside_medoid_union": n_in_u,
                "n_inside_own_dilate1px": n_in_d,
                "n_oob": n_oob,
                "frac_inside_own_polygon": own_f,
                "frac_inside_medoid_union": union_f,
                "frac_inside_own_dilate1px": dil_f,
            }
        )

    # Full medoid confirmation (same raster test; cheap one-mask-at-a-time)
    full_own = []
    full_dil = []
    for r in medoid_rows:
        if r["n_spine_vertices"] < 2:
            continue
        own = rasterize_polygons([r["polygon"]], size=IMAGE_SIZE)
        n_in, n_pts, _ = points_inside_mask(r["spine"], own)
        full_own.append(n_in / n_pts if n_pts else 0.0)
        own_d = cv2.dilate(own, kernel, iterations=1)
        n_in_d, _, _ = points_inside_mask(r["spine"], own_d)
        full_dil.append(n_in_d / n_pts if n_pts else 0.0)

    # All 8199 own-polygon (same test) so we know non-medoid views are not worse
    all_own = []
    for r in all_rows:
        if r["n_spine_vertices"] < 2:
            continue
        own = rasterize_polygons([r["polygon"]], size=IMAGE_SIZE)
        n_in, n_pts, _ = points_inside_mask(r["spine"], own)
        all_own.append(n_in / n_pts if n_pts else 0.0)

    mean_sample_own = float(np.mean(own_fracs)) if own_fracs else 0.0
    mean_sample_dil = float(np.mean(dilate_fracs)) if dilate_fracs else 0.0
    mean_medoid_own = float(np.mean(full_own)) if full_own else 0.0

    # Concrete later recipe — gated, not a Slot-1 clone, not a Slot-2 replace
    if mean_sample_own >= 0.90:
        heatmap_target = "strict_gaussian_on_polyline"
        heatmap_note = "Spines sit in the body; a thin Gaussian ridge on the polyline is a clean aux target."
        recommended = "sidecar_spine_heatmap_then_optional_yolo11s_aux"
    elif mean_sample_dil >= 0.90 or mean_sample_own >= 0.70:
        heatmap_target = "distance_transform_or_dilate1_ridge"
        heatmap_note = "Some vertices snap to the raster edge; supervise a 1-3 px ridge / soft distance map, not hard 1-px pixels."
        recommended = "sidecar_spine_heatmap_dilated_target"
    else:
        heatmap_target = "do_not_supervise_hard_inside"
        heatmap_note = "Too many spine vertices fall outside the rasterized polygon; kill hard heatmap or restrict loss to interior vertices."
        recommended = "hold_or_kill_pending_manual_review"

    later_train_recipe = {
        "status": "WRITE_PLAN_ONLY",
        "do_not_start_until": [
            "yolo11s_seg_all official fold-0 PQ exists (scripts/score_family.py on the 142 grouped val JPEGs; independent conf lock).",
            "phys fail-path is decided: start phys_stack_s1024 only if all-annotator FAIL/KILL; skip or demote phys if all-annotator is the Slot-1 climber success.",
        ],
        "explicit_gate": "This does not start until yolo11s_seg_all official PQ exists and phys fail-path is decided.",
        "not_a_train_this_turn": True,
        "not_slot1_clone": {
            "slot1": "yolo11m_1280_medoid public 0.29 / local official PQ 0.410 (ref 56148686)",
            "this_family": "YOLO11s-seg@1024 with unused train-JSON spine/chirality aux, OR a sidecar heatmap on frozen then-incumbent masks. Different objects (instance identity / barb-consistent splits), not capacity+1280 on the same gray medoid masks.",
        },
        "not_slot2_replace": {
            "slot2": "yolo11s_year_shift public 0.24 / local 0.352 (ref 56148889) STAYS",
            "why": "Year-shift errors differ on late-cycle years. Spine/chirality aux is a Slot-1 climber / research look. Do not retire the 0.05 public haircut.",
        },
        "recommended_path": recommended,
        "heatmap_target": heatmap_target,
        "heatmap_target_note": heatmap_note,
        "option_A_sidecar_heatmap": {
            "when": "First probe after the gate. No ultralytics fork. No yolo_seg* rmtree. CPU-ok or short MPS after phys is decided.",
            "labels": "Train-only 565 physical JPEGs, medoid view spines (fold != 0). Never paint fold-0 spines into a loss that selects the PQ cut.",
            "input": "Same pixels the then-incumbent uses (gray JPEG, or phys stack IFF phys already won). Do not invent a third input.",
            "targets": {
                "spine": f"1024x1024 1-ch Gaussian / distance ridge on the flat [x,y,...] polyline ({heatmap_target})",
                "chirality": "instance-level 3-way CE on category_id {1:L,2:R,3:U}; downweight or ignore class 3; class 4 is empty in this JSON",
            },
            "arch": "tiny U-Net / 3-level residual CNN at 1024, batch 4-8, ~20-40 ep. Not YOLO11m@1280.",
            "decode": "Freeze Slot 1 (or then-incumbent) instance masks. Use the spine ridge to (1) split a YOLO mask that covers two GT bodies along the valley between two ridges, (2) merge two YOLO fragments whose ridges are one polyline. Chirality is a barb-consistency regularizer, not a submit class.",
            "eval": "Official per-annotator PQ on fold-0 142, independent conf lock. Aux heatmap Dice/CE is diagnostic only.",
            "mps_hours": "0.4-0.8 sidecar; decode+PQ ~15-25 min. Far cheaper than a custom YOLO head.",
            "new_cache_dir_if_any": "yolo_seg_spine_heat (NEW). Never rmtree yolo_seg, yolo_seg_all, yolo_seg_phys, yolo_seg_rater.",
        },
        "option_B_yolo11s_1024_aux_heads": {
            "when": "Only if sidecar shows a real RQ/SQ lift (fewer fragment FPs or mean TP IoU +>=0.01) OR sidecar is infeasible and the gate is green.",
            "model": "YOLO11s-seg imgsz=1024, 36 ep, copy-paste 0.20 but paste along the spine (broken fragments), NOT year copies.",
            "labels": "Medoid yolo_seg 565/142 PLUS aux maps written to a NEW dir. If all-annotator official PQ already lifted TPs without FP explosion, may use those train views — still grouped by physical JPEG.",
            "heads": {
                "seg": "stock proto-seg, class_id=0 filament (host PQ is class-agnostic)",
                "spine_heatmap": "1-ch aux on the proto / mask branch",
                "chirality": "3-way instance head; loss only on identifiable L/R or with U weight <=0.3",
            },
            "loss": "L = L_seg + 0.5 L_spine + 0.25 L_chir. Do not let aux dominate the mask head.",
            "val": "Fold-0 142 physicals. Official PQ ignores aux. Do not pick λ from fold-0 PQ (one preset, then one look).",
            "slot_role": "Slot-1 climber research look; never Slot-2; never a conf twin of m@1280.",
            "replace_slot1_when": "Independent fold-0 official PQ >= 0.42 OR +0.02 vs then-incumbent, AND not an FP dump (dPQ > -0.03 with TP gain). Then at most one public look after quota reset.",
            "mps_hours": "1.5-2.5 (s@1024 custom head). m@1280 aux ONLY if s aux proves >=+0.02 official PQ.",
            "forbidden": [
                "Clone Slot 1 by attaching aux to frozen m@1280 and uploading as a sibling final",
                "Replace year-shift Slot 2",
                "Year-upsample copies",
                "Train aux on fold-0 spines",
                "Edit src/filament/yolo_export.py or scripts/prepare_yolo.py while the other agent owns them — new writer only",
            ],
        },
        "leak_guard": {
            "group": "physical JPEG file_name, seed 20260910, val_fold=0 (142 / 565)",
            "never": "010101 and 010102 of a val JPEG in train; spine/chirality loss on val; MAGFiLO public-test / Dataverse test labels",
            "supervise_aux_on": "rest_medoid filaments only (fold != 0)",
            "val_labels_for_pq": "official all-annotator-images on the 142 JPEGs (same as score_family.py)",
        },
    }

    kill_criteria = {
        "do_not_start": "yolo11s_seg_all official PQ missing, OR phys fail-path not yet decided",
        "illegal": [
            "Any file that read MAGFiLO public-test / Harvard Dataverse test annotations",
            "Champion / lamhuy RLEs",
            "Empty-mask PQ hack",
        ],
        "collapsed": [
            "Fold-0 official PQ < 0.20",
            "PQ >= 0.08 worse than s-medoid 0.357 with no TP/recall gain",
            "Classical-like FP dump (hybrid 0.327 / FP tripled is the template)",
        ],
        "no_lift": [
            "Official PQ flat vs then-incumbent (dPQ > -0.01 and < +0.01) AND fragment FP count not down AND mean TP IoU not +>=0.01 — aux heads did not buy instance identity",
            "Chirality head accuracy high but mask PQ unchanged — do not promote a chirality toy",
        ],
        "leaky": [
            "Public jump into the 0.55 cluster with no morphology story",
            "Identical public/private PQ to many decimals after using test GT",
        ],
        "slot_discipline": [
            "Uploading this family as Slot 2 (year-shift stays)",
            "Pairing this file with m@1280 as two finals while year-shift is alive (same-label-family hedge)",
            "Conf / TTA twin of the aux run as the second final",
        ],
        "family_specific": "Kill the aux/sidecar if official PQ does not lift AND RQ (fragments/merges) does not improve. A chirality CE drop alone is not a climb.",
    }

    unused_today = {
        "yolo_export_polygon_to_yolo_line": "hard-codes class_id=0; drops category_id Left/Right/Unidentifiable",
        "yolo_export_write_paths": "never writes spine polylines or heatmaps",
        "coco_io_Filament": "keeps category_id in memory but spine is discarded at load",
        "host_pq": "class-agnostic instance masks; chirality is unused at score time",
        "implication": "These fields are legal train-only metadata. Using them later is not a test-GT leak. Using them on fold-0 for λ/cut search would be a local leak.",
    }

    headline = {
        "n_train_filaments": 8199,
        "n_with_spine": sum(1 for r in all_rows if r["n_spine_vertices"] >= 2),
        "n_spine_ge3_multipoint": sum(1 for r in all_rows if r["n_spine_vertices"] >= SAMPLE_MIN_VERTICES),
        "chirality_L": sum(1 for r in all_rows if r["category_id"] == 1),
        "chirality_R": sum(1 for r in all_rows if r["category_id"] == 2),
        "chirality_unidentifiable": sum(1 for r in all_rows if r["category_id"] == 3),
        "chirality_ambiguous": sum(1 for r in all_rows if r["category_id"] == 4),
        "category_id_histogram": cat_hist(r["category_id"] for r in all_rows),
        "n_fold0_physical": len(val_files),
        "n_rest_physical": len(train_files),
        "n_fold0_filaments_all_raters": len(fold0_rows),
        "n_rest_filaments_all_raters": len(rest_rows),
        "n_fold0_medoid_filaments": len(fold0_medoid),
        "n_rest_medoid_filaments": len(rest_medoid),
        "sample50_mean_frac_spine_pts_inside_own_raster": mean_sample_own,
        "sample50_mean_frac_inside_medoid_union": float(np.mean(union_fracs)) if union_fracs else None,
        "sample50_mean_frac_inside_own_dilate1px": mean_sample_dil,
        "sample50_n_spines_all_pts_inside_own": n_all_in_own,
        "medoid_all_mean_frac_inside_own_raster": mean_medoid_own,
        "all8199_mean_frac_inside_own_raster": float(np.mean(all_own)) if all_own else None,
        "train_multi_rater_category_agree_rate": chir_noise["category_agree_rate"],
    }

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "spine_chirality_multitask_instance_id",
        "method_index_in_legal_climb_methods": 5,
        "task": "diagnosis_plus_write_plan",
        "trained": False,
        "submitted": False,
        "used_mps": False,
        "read_magfilo_public_test_or_dataverse_test_labels": False,
        "touched_yolo_seg_dirs": False,
        "edited_yolo_export_or_prepare_yolo": False,
        "python": "~/.venvs/filament/bin/python",
        "train_json": str(TRAIN_JSON),
        "seed": SEED,
        "val_fold": 0,
        "grouped_split": {
            "method": "filament.coco_io.grouped_folds",
            "seed": SEED,
            "n_physical_train": len(train_files),
            "n_physical_val_fold0": len(val_files),
            "matches_yolo_split_json": len(train_files) == 565 and len(val_files) == 142,
        },
        "schema": {
            "top_keys": list(coco.keys()),
            "categories": coco["categories"],
            "image_key_sets": {str(k): v for k, v in img_key_sets.items()},
            "annotation_key_sets": {str(k): v for k, v in ann_key_sets.items()},
            "annotation_keys": ["area", "bbox", "category_id", "id", "image_id", "iscrowd", "segmentation", "spine"],
            "chirality_field": None,
            "chirality_is": "category_id via categories[].name (Left/Right/Unidentifiable/Ambiguous). No separate chirality/handed/barb key on any of 8199 annotations.",
            "separate_chirality_like_keys_found": chirality_like,
            "spine_format": "flat list of floats [x1,y1,x2,y2,...] native 2048 px; all 8199 annotations; all elements float; even length",
            "yolo_today": unused_today,
        },
        "coverage_all_8199": split_block(all_rows),
        "vs_full_magfilo_paper": {
            "paper": FULL_MAGFILO_PAPER,
            "this_train_json": {
                "L": headline["chirality_L"],
                "R": headline["chirality_R"],
                "unidentifiable": headline["chirality_unidentifiable"],
                "ambiguous": headline["chirality_ambiguous"],
                "n": 8199,
            },
            "fraction_of_paper_counts": {
                "L": headline["chirality_L"] / FULL_MAGFILO_PAPER["L"],
                "R": headline["chirality_R"] / FULL_MAGFILO_PAPER["R"],
                "unidentifiable": headline["chirality_unidentifiable"] / FULL_MAGFILO_PAPER["unidentifiable"],
            },
            "note": "Competition train is a subset. Do not pull the missing paper filaments from Dataverse.",
        },
        "fold0_vs_rest_no_val_leak": {
            "why": "A later aux-head train must supervise spine/chirality only on fold!=0 physicals so fold-0 official PQ stays an unseen JPEG group.",
            "fold0_val_all_raters": split_block(fold0_rows),
            "rest_train_all_raters": split_block(rest_rows),
            "fold0_val_medoid": split_block(fold0_medoid),
            "rest_train_medoid": split_block(rest_medoid),
            "supervise_aux_on": "rest_train_medoid (or rest_train_all_raters if all-annotator already won)",
            "hold_out_of_aux_loss": "fold0_val_* ",
        },
        "spine_inside_medoid_masks": {
            "definition": "Rasterize the filament polygon with cv2.fillPoly at 2048 (same as coco_io.rasterize_polygons). A polyline vertex is inside if mask[round(y), round(x)]==1. Medoid = coco_io.medoid_view_id per physical JPEG.",
            "sample": {
                "n": len(sample),
                "seed": SEED,
                "pool": "medoid-view filaments with >=3 spine vertices",
                "pool_size": len(pool),
                "n_from_fold0": sum(1 for s in sample_stats if s["is_fold0"]),
                "n_from_rest": sum(1 for s in sample_stats if not s["is_fold0"]),
                "mean_frac_points_inside_own_polygon": mean_sample_own,
                "mean_frac_points_inside_medoid_union": float(np.mean(union_fracs)) if union_fracs else None,
                "mean_frac_points_inside_own_dilate1px": mean_sample_dil,
                "n_spines_with_all_points_inside_own": n_all_in_own,
                "n_spines_with_all_points_inside_union": n_all_in_union,
                "per_spine_own_frac": numeric_summary(own_fracs),
                "per_spine_union_frac": numeric_summary(union_fracs),
                "per_spine_dilate1_frac": numeric_summary(dilate_fracs),
                "rows": sample_stats,
            },
            "full_medoid_confirmation": {
                "n_medoid_filaments_ge2": len(full_own),
                "mean_frac_points_inside_own_polygon": mean_medoid_own,
                "mean_frac_points_inside_own_dilate1px": float(np.mean(full_dil)) if full_dil else None,
                "per_spine_own_frac": numeric_summary(full_own),
            },
            "all_8199_own_polygon": {
                "n": len(all_own),
                "mean_frac_points_inside_own_polygon": float(np.mean(all_own)) if all_own else None,
                "per_spine_own_frac": numeric_summary(all_own),
            },
        },
        "chirality_label_noise_train_only": chir_noise,
        "later_train_recipe": later_train_recipe,
        "kill_criteria": kill_criteria,
        "hidden_exam": {
            "public_best": "Slot 1 = m@1280 medoid 0.29",
            "hedge": "Slot 2 = year-shift 0.24 — keep",
            "this_file_if_trained_later": "Different-error Slot-1 climber (instance ID / barb consistency). One public look only after its own official PQ. Not a public-invert builder.",
        },
        "headline_counts": headline,
        "most_important_counts": [
            {"key": "n_train_filaments", "value": 8199},
            {"key": "n_with_spine", "value": headline["n_with_spine"]},
            {"key": "chirality_L", "value": headline["chirality_L"]},
            {"key": "chirality_R", "value": headline["chirality_R"]},
            {"key": "chirality_unidentifiable", "value": headline["chirality_unidentifiable"]},
            {"key": "n_rest_medoid_filaments_aux_ok", "value": headline["n_rest_medoid_filaments"]},
            {"key": "n_fold0_medoid_filaments_hold_out", "value": headline["n_fold0_medoid_filaments"]},
            {"key": "sample50_mean_frac_inside_own_raster", "value": round(mean_sample_own, 4)},
            {"key": "medoid_all_mean_frac_inside_own_raster", "value": round(mean_medoid_own, 4)},
            {"key": "train_multi_rater_category_agree_rate", "value": chir_noise["category_agree_rate"]},
            {"key": "n_physical_train_val", "value": [len(train_files), len(val_files)]},
            {"key": "category_4_ambiguous", "value": headline["chirality_ambiguous"]},
        ],
    }

    # Drop heavy arrays before JSON
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "spine_chirality_inventory.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": str(dest), "headline": headline, "recommended": recommended}, indent=2))


if __name__ == "__main__":
    main()
