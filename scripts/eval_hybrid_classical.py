#!/usr/bin/env python3
"""Legal Method-4 hybrid: frozen YOLO ∪ filtered classical dark-residual seeds.

Keep a classical seed iff a train-only morphometrics rejector says filament
OR it IoU-matches a frozen YOLO mask. YOLO weights are never trained. MAGFiLO
test labels are never read. MPS is never requested.

Official score is host PQ (IoU>0.5, not Hungarian) on ALL annotators of the
grouped holdout (seed 20260910). Do not invert 2017-2020 density toward 7/img.
Do not retune cuts from public 0.29.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.classical import ClassicalConfig, dark_residual_channel, detect_classical
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco, rasterize_polygons
from filament.infer import masks_to_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle

# A priori locks. Not searched on fold-0 PQ. Not taken from public 0.29.
MATCH_IOU = 0.30
DEDUP_IOU = 0.50
REJECTOR_THR = 0.50
HARD_NEG_IOU = 0.10
HARD_POS_IOU = 0.50
MINE_IOU_SIZE = 512
FEATURE_NAMES = (
    "log_area",
    "aspect",
    "elongation",
    "circularity",
    "fill",
    "solidity",
    "eccentricity",
    "radial_frac",
    "mean_int",
    "std_int",
    "mean_residual",
    "rel_dark",
    "major_frac",
)


@dataclass
class PQRow:
    pq: float
    tp: int
    fp: int
    fn: int
    mean_tp_iou: float
    n_pred: int
    pred_per_img: float
    pred_per_img_2017_2020: float


def _empty_pred_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["filament_id", "segmentation_rle"])


def stem_of(file_name: str) -> str:
    return Path(file_name).stem


def year_of(file_name: str) -> int:
    return int(file_name[:4])


def late_year(file_name: str) -> bool:
    return year_of(file_name) >= 2017 and year_of(file_name) <= 2020


def pred_df_from_rles(per_file: dict[str, list[str]]) -> pd.DataFrame:
    frames = [pred_to_df(stem_of(fn), rles) for fn, rles in sorted(per_file.items())]
    return pd.concat(frames, ignore_index=True) if frames else _empty_pred_df()


def pred_per_img(per_file: dict[str, list[str]], file_names: list[str]) -> float:
    if not file_names:
        return 0.0
    return float(np.mean([len(per_file.get(fn, [])) for fn in file_names]))


def group_yolo_rles(pred_df: pd.DataFrame) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for filament_id, rle in pred_df[["filament_id", "segmentation_rle"]].itertuples(index=False):
        stem = str(filament_id).split("_", 1)[0]
        grouped[stem].append(str(rle))
    return dict(grouped)


def instance_features(mask: np.ndarray, image: np.ndarray, residual: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    area = float(mask_u8.sum())
    if area < 8:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    ys, xs = np.nonzero(mask_u8)
    height, width = mask_u8.shape
    radial = math.hypot(float(ys.mean()) - (height - 1) / 2.0, float(xs.mean()) - (width - 1) / 2.0) / 1000.0
    _x, _y, bw, bh = cv2.boundingRect(mask_u8)
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    fill = area / max(float(bw * bh), 1.0)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = cv2.arcLength(contours[0], True) if contours else 0.0
    circularity = (4.0 * math.pi * area / (peri * peri)) if peri > 0 else 0.0
    hull_area = area
    if contours and len(contours[0]) >= 3:
        hull = cv2.convexHull(contours[0])
        if len(hull) >= 3:
            hull_area = max(float(cv2.contourArea(hull)), area)
    solidity = area / hull_area
    elongation = aspect
    eccentricity = 0.0
    major = float(max(bw, bh))
    if len(xs) >= 5:
        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        try:
            _center, (ma, MA), _ang = cv2.fitEllipse(pts)
            short, long_ax = (ma, MA) if MA >= ma else (MA, ma)
            short = max(float(short), 1e-3)
            long_ax = max(float(long_ax), short)
            elongation = long_ax / short
            eccentricity = math.sqrt(max(0.0, 1.0 - (short / long_ax) ** 2))
            major = long_ax
        except cv2.error:
            pass
    pix = image[mask_u8 > 0].astype(np.float32)
    res = residual[mask_u8 > 0].astype(np.float32)
    mean_int = float(pix.mean()) / 255.0
    std_int = float(pix.std()) / 255.0
    mean_res = float(res.mean()) / 255.0
    disk_mean = float(image.mean()) / 255.0
    rel_dark = max(0.0, disk_mean - mean_int) / max(disk_mean, 1e-3)
    return np.array(
        [
            math.log(area),
            aspect,
            elongation,
            circularity,
            fill,
            solidity,
            eccentricity,
            radial,
            mean_int,
            std_int,
            mean_res,
            rel_dark,
            major / float(max(height, width)),
        ],
        dtype=np.float32,
    )


def morph_filament_like(feat: np.ndarray) -> bool:
    """Qu/Diercke-shaped thin-lane gate. A priori, not fold-0 searched."""
    feat = np.asarray(feat, dtype=np.float32)
    log_area, aspect, elongation, circularity, fill, solidity, _ecc, radial = feat[:8]
    area = math.exp(float(log_area))
    if area < 200 or area > 50_000:
        return False
    if radial < 0.04 or radial > 0.96:
        return False
    if circularity > 0.70 and fill > 0.78:
        return False
    thin = elongation >= 1.8 or aspect >= 2.0
    open_shape = solidity <= 0.90 and fill <= 0.80
    return bool(thin and open_shape and circularity <= 0.72)


def _resize_bin(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return (mask > 0).astype(np.uint8)
    return cv2.resize((mask > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def max_iou(mask: np.ndarray, others: list[np.ndarray], size: int | None = None) -> float:
    if not others:
        return 0.0
    probe = _resize_bin(mask, size) if size else (mask > 0)
    area = int(probe.sum())
    if area == 0:
        return 0.0
    best = 0.0
    for other in others:
        other_b = _resize_bin(other, size) if size else (other > 0)
        inter = int(np.logical_and(probe, other_b).sum())
        if inter == 0:
            continue
        union = area + int(other_b.sum()) - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou > best:
            best = float(iou)
    return best


def decode_masks(rles: list[str]) -> list[np.ndarray]:
    return [decode_rle(rle) for rle in rles]


def breakdown_to_row(br, per_file: dict[str, list[str]], file_names: list[str]) -> PQRow:
    late = [fn for fn in file_names if late_year(fn)]
    return PQRow(
        pq=float(br.pq),
        tp=int(br.tp),
        fp=int(br.fp),
        fn=int(br.fn),
        mean_tp_iou=float(br.mean_tp_iou),
        n_pred=int(sum(len(v) for v in per_file.values())),
        pred_per_img=pred_per_img(per_file, file_names),
        pred_per_img_2017_2020=pred_per_img(per_file, late),
    )


def score_variant(gt_df: pd.DataFrame, per_file: dict[str, list[str]], file_names: list[str]) -> PQRow:
    pred_df = pred_df_from_rles(per_file)
    return breakdown_to_row(score_pq(gt_df, pred_df), per_file, file_names)


def filter_year_dfs(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    file_names: list[str],
    lo: int = 2017,
    hi: int = 2020,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stems = {stem_of(fn) for fn in file_names if lo <= year_of(fn) <= hi}
    if not stems:
        return gt_df.iloc[0:0], pred_df.iloc[0:0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    gt_stems = gt_keys.str.split("-", n=1).str[1]
    return gt_df.loc[gt_stems.isin(stems)].copy(), pred_df.loc[pred_keys.isin(stems)].copy()


def _detect_one(file_name: str, cfg: ClassicalConfig) -> tuple[str, list[str], list[list[float]]]:
    cv2.setNumThreads(1)
    image = read_gray(TRAIN_IMG_DIR / file_name)
    residual = dark_residual_channel(image, cfg)
    masks = detect_classical(image, cfg)
    feats = [instance_features(mask, image, residual).tolist() for mask in masks]
    return file_name, masks_to_rles(masks), feats


def detect_files(file_names: list[str], cfg: ClassicalConfig, workers: int) -> dict[str, tuple[list[str], np.ndarray]]:
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    if not file_names:
        return out
    if workers <= 1:
        for file_name in tqdm(file_names, desc="classical detect"):
            fn, rles, feats = _detect_one(file_name, cfg)
            out[fn] = (rles, np.asarray(feats, dtype=np.float32).reshape(len(rles), -1))
        return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_detect_one, fn, cfg) for fn in file_names]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="classical detect"):
            fn, rles, feats = fut.result()
            out[fn] = (rles, np.asarray(feats, dtype=np.float32).reshape(len(rles), -1))
    return out


def _mine_one(
    file_name: str,
    polygons_per_view: list[list[np.ndarray]],
    cfg: ClassicalConfig,
) -> tuple[list[list[float]], list[int]]:
    cv2.setNumThreads(1)
    image = read_gray(TRAIN_IMG_DIR / file_name)
    residual = dark_residual_channel(image, cfg)
    seeds = detect_classical(image, cfg)
    gt_masks: list[np.ndarray] = []
    for polys in polygons_per_view:
        for poly in polys:
            gt_masks.append(rasterize_polygons([poly]))
    feats: list[list[float]] = []
    labels: list[int] = []
    for gt in gt_masks:
        feats.append(instance_features(gt, image, residual).tolist())
        labels.append(1)
    for seed in seeds:
        iou = max_iou(seed, gt_masks, size=MINE_IOU_SIZE)
        if iou < HARD_NEG_IOU:
            feats.append(instance_features(seed, image, residual).tolist())
            labels.append(0)
        elif iou >= HARD_POS_IOU:
            feats.append(instance_features(seed, image, residual).tolist())
            labels.append(1)
    return feats, labels


def fit_rejector(
    file_names: list[str],
    views,
    by_file: dict[str, list[str]],
    cfg: ClassicalConfig,
    workers: int,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    xs: list[list[float]] = []
    ys: list[int] = []

    def job(fn: str):
        polys = [[filament.polygon for filament in views[image_id].filaments] for image_id in by_file[fn]]
        return _mine_one(fn, polys, cfg)

    if workers <= 1:
        iterator = (job(fn) for fn in file_names)
        wrapped = tqdm(iterator, total=len(file_names), desc="rejector mine")
        for feats, labels in wrapped:
            xs.extend(feats)
            ys.extend(labels)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(job, fn) for fn in file_names]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="rejector mine"):
                feats, labels = fut.result()
                xs.extend(feats)
                ys.extend(labels)

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 8 or n_neg < 8:
        return None, {"status": "fallback_morph", "n_pos": n_pos, "n_neg": n_neg, "n_rows": int(len(y))}
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=400,
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(x, y)
    train_acc = float((clf.predict(x) == y).mean())
    return clf, {
        "status": "fitted",
        "type": "logistic_morphometrics",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rows": int(len(y)),
        "train_acc": train_acc,
        "feature_names": list(FEATURE_NAMES),
        "threshold": REJECTOR_THR,
    }


def rejector_probas(feats: np.ndarray, clf) -> np.ndarray:
    if feats.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if clf is None:
        return np.array([1.0 if morph_filament_like(row) else 0.0 for row in feats], dtype=np.float32)
    return clf.predict_proba(feats)[:, 1].astype(np.float32)


def keep_seed(yolo_iou: float, proba: float) -> tuple[bool, str]:
    if yolo_iou >= MATCH_IOU:
        return True, "yolo_match"
    if proba >= REJECTOR_THR:
        return True, "filament_like"
    return False, "reject"


def decide(yolo: PQRow, hybrid: PQRow) -> tuple[str, str]:
    fp_doubled = hybrid.fp >= 2 * max(yolo.fp, 1)
    pq_drop = hybrid.pq < yolo.pq - 1e-12
    tp_gain = hybrid.tp - yolo.tp
    fp_ok = (not fp_doubled) and hybrid.fp <= yolo.fp + max(80, yolo.fp)
    if pq_drop or fp_doubled:
        why = []
        if pq_drop:
            why.append(f"PQ {hybrid.pq:.4f} < yolo {yolo.pq:.4f}")
        if fp_doubled:
            why.append(f"FP doubled {yolo.fp}->{hybrid.fp}")
        return "KILL", "; ".join(why)
    if hybrid.pq >= 0.420:
        return (
            "ATTACH_SLOT1_AFTER_QUOTA_RESET",
            f"hybrid official PQ {hybrid.pq:.4f} >= 0.420 (TP {hybrid.tp}/{yolo.tp}, FP {hybrid.fp}/{yolo.fp})",
        )
    if tp_gain >= 20 and fp_ok:
        return (
            "ATTACH_SLOT1_AFTER_QUOTA_RESET",
            f"TP gain {tp_gain} >= 20 with FP {hybrid.fp} (yolo {yolo.fp}) not exploding",
        )
    return (
        "KILL",
        f"PQ {hybrid.pq:.4f} < 0.420 and TP gain {tp_gain} < 20 (FP {hybrid.fp} vs {yolo.fp})",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0, help="0 = full grouped fold")
    parser.add_argument(
        "--yolo-pred",
        default=str(
            Path.home()
            / ".cache"
            / "filament-segmentation-2026"
            / "artifacts"
            / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
        ),
    )
    parser.add_argument("--out", default="", help="JSON path. Default artifacts/classical_hybrid_fold{fold}_pq.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mine-max", type=int, default=0, help="Cap other-fold mine images. 0 = all")
    parser.add_argument("--skip-fit", action="store_true", help="Use morphometric gate only")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ClassicalConfig()
    cv2.setNumThreads(1)

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=args.seed)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == args.fold)
    if args.max_val:
        val_files = val_files[: args.max_val]
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != args.fold)
    if args.mine_max:
        rng = np.random.default_rng(args.seed)
        years: dict[int, list[str]] = defaultdict(list)
        for fn in train_files:
            years[year_of(fn)].append(fn)
        picked: list[str] = []
        leftover = list(train_files)
        for bucket in years.values():
            take = max(1, int(round(args.mine_max * len(bucket) / max(len(train_files), 1))))
            choose = list(bucket)
            rng.shuffle(choose)
            picked.extend(choose[:take])
        leftover = [fn for fn in leftover if fn not in set(picked)]
        rng.shuffle(leftover)
        train_files = (picked + leftover)[: args.mine_max]
        train_files.sort()

    yolo_raw = pd.read_csv(args.yolo_pred)
    yolo_by_stem = group_yolo_rles(yolo_raw)
    yolo_per_file = {fn: list(yolo_by_stem.get(stem_of(fn), [])) for fn in val_files}
    missing = [fn for fn in val_files if stem_of(fn) not in yolo_by_stem]
    extra_stems = sorted(set(yolo_by_stem) - {stem_of(fn) for fn in val_files})

    print("building all-annotator GT…", flush=True)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    clf, rejector_meta = None, {"status": "skipped", "type": "morphometric_only"}
    if not args.skip_fit:
        clf, rejector_meta = fit_rejector(train_files, views, by_file, cfg, args.workers)
    else:
        rejector_meta["n_train_images"] = 0
    rejector_meta["n_train_images"] = len(train_files)
    rejector_meta["match_iou"] = MATCH_IOU
    rejector_meta["dedup_iou"] = DEDUP_IOU
    rejector_meta["hard_neg_iou"] = HARD_NEG_IOU
    rejector_meta["mine_unit"] = "instance_masks_not_semantic_union"

    detected = detect_files(val_files, cfg, args.workers)

    classical_per_file: dict[str, list[str]] = {}
    union_per_file: dict[str, list[str]] = {}
    hybrid_per_file: dict[str, list[str]] = {}
    n_seeds = 0
    n_kept_filament = 0
    n_kept_yolo = 0
    n_added = 0
    n_dup_suppressed = 0
    keep_reasons: dict[str, int] = defaultdict(int)

    for file_name in tqdm(val_files, desc="filter hybrid"):
        yolo_rles = yolo_per_file[file_name]
        class_rles, feats = detected.get(file_name, ([], np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)))
        yolo_masks = decode_masks(yolo_rles)
        class_masks = decode_masks(class_rles)
        ious = np.array([max_iou(mask, yolo_masks) for mask in class_masks], dtype=np.float32)
        probas = rejector_probas(feats, clf)
        classical_per_file[file_name] = list(class_rles)
        union_keep = list(yolo_rles)
        hybrid_keep = list(yolo_rles)
        for rle, iou, proba in zip(class_rles, ious, probas):
            n_seeds += 1
            if iou < DEDUP_IOU:
                union_keep.append(rle)
            else:
                n_dup_suppressed += 1
            kept, reason = keep_seed(float(iou), float(proba))
            keep_reasons[reason] += 1
            if not kept:
                continue
            if reason == "filament_like":
                n_kept_filament += 1
            elif reason == "yolo_match":
                n_kept_yolo += 1
            if iou < DEDUP_IOU:
                hybrid_keep.append(rle)
                n_added += 1
        union_per_file[file_name] = union_keep
        hybrid_per_file[file_name] = hybrid_keep
        del yolo_masks, class_masks

    print("scoring yolo-only…", flush=True)
    yolo_row = score_variant(gt_df, yolo_per_file, val_files)
    print("scoring classical-only…", flush=True)
    classical_row = score_variant(gt_df, classical_per_file, val_files)
    print("scoring union-all-classical…", flush=True)
    union_row = score_variant(gt_df, union_per_file, val_files)
    print("scoring hybrid-filtered…", flush=True)
    hybrid_row = score_variant(gt_df, hybrid_per_file, val_files)

    yolo_pred_df = pred_df_from_rles(yolo_per_file)
    hybrid_pred_df = pred_df_from_rles(hybrid_per_file)
    gt_late, yolo_late = filter_year_dfs(gt_df, yolo_pred_df, val_files)
    _, hybrid_late = filter_year_dfs(gt_df, hybrid_pred_df, val_files)
    late_yolo = score_pq(gt_late, yolo_late) if len(gt_late) else None
    late_hybrid = score_pq(gt_late, hybrid_late) if len(gt_late) else None

    decision, reason = decide(yolo_row, hybrid_row)
    rejector_meta.update(
        {
            "n_classical_seeds": n_seeds,
            "n_kept_filament_like": n_kept_filament,
            "n_kept_yolo_match": n_kept_yolo,
            "n_added_to_yolo": n_added,
            "n_dup_suppressed": n_dup_suppressed,
            "keep_reasons": dict(keep_reasons),
        }
    )

    n_annotator_images = int(gt_df["filament_id"].str.split("_", n=1).str[0].nunique())
    out = {
        "written": str(date.today()),
        "method": "classical_seed_learned_sunspot_filter",
        "legal": True,
        "fold": args.fold,
        "seed": args.seed,
        "n_val_images": len(val_files),
        "n_annotator_images": n_annotator_images,
        "n_gt": int(len(gt_df)),
        "yolo_pred_csv": str(args.yolo_pred),
        "yolo_stems_missing_on_val": missing,
        "yolo_extra_stems": extra_stems[:12],
        "n_yolo_extra_stems": len(extra_stems),
        "classical_config": asdict(cfg),
        "metric": "official host PQ (IoU>0.5, not Hungarian, all annotators)",
        "hidden_exam": {
            "do_not_invert_2017_2020_density": True,
            "do_not_retune_from_public_029": True,
            "note": "2017-2020 pred/img is diagnosis only. 2.3 is solar-min, not a 7/img target.",
        },
        "yolo_only": asdict(yolo_row),
        "classical_only": asdict(classical_row),
        "union_all_classical": asdict(union_row),
        "hybrid_filtered": asdict(hybrid_row),
        "hybrid_delta_vs_yolo": {
            "dq": hybrid_row.pq - yolo_row.pq,
            "dtp": hybrid_row.tp - yolo_row.tp,
            "dfp": hybrid_row.fp - yolo_row.fp,
            "dfn": hybrid_row.fn - yolo_row.fn,
        },
        "year_2017_2020": {
            "n_images": sum(1 for fn in val_files if late_year(fn)),
            "yolo": None
            if late_yolo is None
            else {
                "pq": float(late_yolo.pq),
                "tp": int(late_yolo.tp),
                "fp": int(late_yolo.fp),
                "fn": int(late_yolo.fn),
            },
            "hybrid": None
            if late_hybrid is None
            else {
                "pq": float(late_hybrid.pq),
                "tp": int(late_hybrid.tp),
                "fp": int(late_hybrid.fp),
                "fn": int(late_hybrid.fn),
            },
            "yolo_pred_per_img": yolo_row.pred_per_img_2017_2020,
            "hybrid_pred_per_img": hybrid_row.pred_per_img_2017_2020,
        },
        "rejector": rejector_meta,
        "decision": decision,
        "decision_reason": reason,
        "attach_if": "hybrid official PQ >= 0.420 OR TP gain >= 20 with FP not exploding",
        "kill_if": "official PQ drops or FP doubles",
        "do_not_submit": True,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else ARTIFACTS / f"classical_hybrid_fold{args.fold}_pq.json"
    out_path.write_text(json.dumps(out, indent=2))
    out["path"] = str(out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
