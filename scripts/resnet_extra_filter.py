"""Leak-free logistic rejector on ResNet extras only. CPU fit. Never submits.

Train rows: fold!=0 extras (grouped_folds, SEED) labeled vs train GT.
Features: extra mask + JPEG crop / dark residual. No MAGFiLO test labels.
Not classical detect_classical seeds (that family is KILL 0.327).
Filter weights: WEIGHTS_DIR/resnet_extra_filter — never unet_resnet34 /
unet_semantic / unet_patch_dhebe / unet_3ch_disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from filament.classical import ClassicalConfig, dark_residual_channel
from filament.merge import MERGE_IOU, extra_rles, mask_iou
from filament.paths import FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.preprocess import read_gray
from filament.rle import decode_rle
from gate_unet_family import (
    FILTER_WEIGHTS_NAME,
    FORBIDDEN_WEIGHT_DIRS,
    HOSE_FLOOR,
    filter_dest_illegal,
)

FILTER_DEST = WEIGHTS_DIR / FILTER_WEIGHTS_NAME
EXTRAS_JSONL = FAST_ARTIFACTS / "resnet_extras_foldne0.jsonl"
EXTRAS_META = FAST_ARTIFACTS / "resnet_extras_foldne0_meta.json"
SLOT1_FOLDNE0_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_foldne0_conf0.35_sz1280_m0_base.csv"
LABEL_IOU = 0.5
REJECTOR_THR = 0.50
MIN_RECALL_FOR_PREC_LOCK = 0.40
FEATURE_MIN_AREA = 8


def parse_min_areas(min_areas: str, min_area: int, *, hose_floor: int = HOSE_FLOOR) -> list[int]:
    vals = sorted(
        {int(x.strip()) for x in str(min_areas).split(",") if x.strip()} | {int(min_area)}
    )
    bad = [v for v in vals if v < hose_floor]
    if bad:
        raise SystemExit(f"Refusing extra-filter min_area {bad} < {hose_floor} (hose floor).")
    return vals


def assert_safe_filter_dest(dest: Path) -> Path:
    dest = Path(dest)
    bad = filter_dest_illegal(dest)
    if bad:
        raise SystemExit(bad)
    if dest.resolve() == (WEIGHTS_DIR / "unet_resnet34").resolve():
        raise SystemExit(f"Refusing extra-filter dest {dest}: must not overwrite unet_resnet34.")
    return dest


def extra_features(mask: np.ndarray, image: np.ndarray, residual: np.ndarray) -> np.ndarray:
    from eval_hybrid_classical import instance_features

    return instance_features(mask, image, residual)


def extra_feature_names() -> tuple[str, ...]:
    from eval_hybrid_classical import FEATURE_NAMES

    return FEATURE_NAMES


def max_gt_iou(mask: np.ndarray, gt_masks: list[np.ndarray]) -> float:
    if not gt_masks:
        return 0.0
    return max(mask_iou(mask, other) for other in gt_masks)


def label_extra(mask: np.ndarray, gt_masks: list[np.ndarray], iou_thr: float = LABEL_IOU) -> int:
    return 1 if max_gt_iou(mask, gt_masks) > iou_thr else 0


def features_for_rles(rles: list[str], image: np.ndarray, residual: np.ndarray) -> np.ndarray:
    names = extra_feature_names()
    if not rles:
        return np.zeros((0, len(names)), dtype=np.float32)
    rows = [extra_features(decode_rle(rle), image, residual) for rle in rles]
    return np.asarray(rows, dtype=np.float32).reshape(len(rles), -1)


def context_feature_names() -> tuple[str, ...]:
    return (
        "max_iou_slot1",
        "min_centroid_dist",
        "log_min_centroid_dist",
        "log_area_ratio_nearest_slot1",
        "log1p_n_slot1",
    )


def _centroid_area(mask: np.ndarray) -> tuple[float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean()), float(ys.size)


def context_features(mask: np.ndarray, slot_masks: list[np.ndarray]) -> np.ndarray:
    """Slot1-relative extras. Isolation vs near-miss fragment, not a new cut."""
    extra = _centroid_area(mask)
    n_slot = len(slot_masks)
    if extra is None or n_slot == 0:
        return np.zeros(len(context_feature_names()), dtype=np.float32)
    ey, ex, ea = extra
    best_iou = 0.0
    best_dist = 1e9
    nearest_area = 0.0
    for sm in slot_masks:
        iou = mask_iou(mask, sm)
        if iou > best_iou:
            best_iou = float(iou)
        sc = _centroid_area(sm)
        if sc is None:
            continue
        sy, sx, sa = sc
        dist = math.hypot(ey - sy, ex - sx) / 2048.0
        if dist < best_dist:
            best_dist = float(dist)
            nearest_area = sa
    if best_dist >= 1e8:
        best_dist = 1.0
    return np.array(
        [
            best_iou,
            best_dist,
            math.log1p(best_dist * 2048.0),
            math.log1p(ea / max(nearest_area, 1.0)),
            math.log1p(float(n_slot)),
        ],
        dtype=np.float32,
    )


def features_for_rles_context(
    rles: list[str],
    image: np.ndarray,
    residual: np.ndarray,
    slot_masks: list[np.ndarray],
) -> np.ndarray:
    morph = features_for_rles(rles, image, residual)
    if not rles:
        return np.zeros((0, morph.shape[1] + len(context_feature_names())), dtype=np.float32)
    ctx = np.stack([context_features(decode_rle(rle), slot_masks) for rle in rles], axis=0)
    return np.concatenate([morph, ctx], axis=1)


def rows_to_xy_context(
    rows: list[dict],
    slot1_by_file: dict,
) -> tuple[np.ndarray, np.ndarray]:
    names = extra_feature_names() + context_feature_names()
    if not rows:
        return np.zeros((0, len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    mask_cache: dict[str, list[np.ndarray]] = {}
    xs = []
    ys = []
    for row in rows:
        file_name = str(row["file_name"])
        if file_name not in mask_cache:
            frame = slot1_by_file.get(file_name)
            rles = []
            if frame is not None and len(frame):
                rles = frame["segmentation_rle"].tolist()
            mask_cache[file_name] = [decode_rle(rle) for rle in rles]
        morph = np.asarray(row["features"], dtype=np.float32).reshape(-1)
        ctx = context_features(decode_rle(row["rle"]), mask_cache[file_name])
        xs.append(np.concatenate([morph, ctx], axis=0))
        ys.append(int(row["label"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def jpeg_and_residual(file_name: str, cfg: ClassicalConfig | None = None):
    image = read_gray(TRAIN_IMG_DIR / file_name)
    residual = dark_residual_channel(image, cfg or ClassicalConfig())
    return image, residual


def fit_extra_rejector(features: np.ndarray, labels: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=np.int32)
    x = np.asarray(features, dtype=np.float32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    meta = {
        "status": "fallback_morph",
        "type": "logistic_morphometrics",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rows": int(len(y)),
        "feature_names": list(extra_feature_names()),
        "threshold": REJECTOR_THR,
        "fit_folds": "fold!=0 only",
        "never_fit_fold0": True,
        "classical_new_seeds": False,
        "seed": SEED,
    }
    if n_pos < 8 or n_neg < 8 or len(y) < 16:
        meta["status"] = "fallback_morph"
        return None, meta
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
    proba = clf.predict_proba(x)[:, 1].astype(np.float32)
    thr, lock = lock_rejector_thr(y, proba)
    meta.update(
        {
            "status": "fitted",
            "train_acc": float((clf.predict(x) == y).mean()),
            "threshold": thr,
            "foldne0_lock": lock,
            "fallback_morph_used": False,
        }
    )
    return clf, meta


def fit_extra_rejector_hgb(features: np.ndarray, labels: np.ndarray):
    """Nonlinear morphometric rejector. Same fold!=0 precision-first lock as logistic."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = np.asarray(labels, dtype=np.int32)
    x = np.asarray(features, dtype=np.float32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    meta = {
        "status": "fallback_morph",
        "type": "hgb_morphometrics",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rows": int(len(y)),
        "feature_names": list(extra_feature_names()),
        "threshold": REJECTOR_THR,
        "fit_folds": "fold!=0 only",
        "never_fit_fold0": True,
        "classical_new_seeds": False,
        "seed": SEED,
        "not_logistic_cut_retune": True,
    }
    if n_pos < 8 or n_neg < 8 or len(y) < 16:
        return None, meta
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.08,
        max_iter=120,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=int(SEED),
    )
    clf.fit(x, y)
    proba = clf.predict_proba(x)[:, 1].astype(np.float32)
    thr, lock = lock_rejector_thr(y, proba)
    meta.update(
        {
            "status": "fitted",
            "train_acc": float((clf.predict(x) == y).mean()),
            "threshold": thr,
            "foldne0_lock": lock,
            "fallback_morph_used": False,
        }
    )
    return clf, meta


def lock_rejector_thr(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    min_recall: float = MIN_RECALL_FOR_PREC_LOCK,
) -> tuple[float, dict]:
    """Precision-first cut on fold!=0 extra labels. Not fold-0 PQ. Not public invert."""
    best_thr = REJECTOR_THR
    best_prec = -1.0
    best_rec = 0.0
    rows = []
    for thr in np.linspace(0.30, 0.80, 11):
        pred = proba >= thr
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rows.append({"thr": float(thr), "precision": float(prec), "recall": float(rec), "tp": tp, "fp": fp})
        if rec + 1e-12 >= min_recall and prec > best_prec:
            best_thr, best_prec, best_rec = float(thr), float(prec), float(rec)
    if best_prec < 0:
        return REJECTOR_THR, {
            "status": "fallback_0.5",
            "min_recall": min_recall,
            "sweep": rows,
            "why": "no cut kept fold!=0 extra recall; a priori 0.5",
        }
    return best_thr, {
        "status": "locked_foldne0",
        "min_recall": min_recall,
        "precision": best_prec,
        "recall": best_rec,
        "threshold": best_thr,
        "sweep": rows,
    }


def rejector_probas(feats: np.ndarray, clf) -> np.ndarray:
    from eval_hybrid_classical import morph_filament_like, rejector_probas as _probas

    if clf is None:
        if feats.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.array([1.0 if morph_filament_like(row) else 0.0 for row in feats], dtype=np.float32)
    return _probas(feats, clf)


def keep_extra_rles(rles: list[str], feats: np.ndarray, clf, thr: float) -> list[str]:
    if not rles:
        return []
    proba = rejector_probas(feats, clf)
    return [rle for rle, p in zip(rles, proba) if float(p) >= float(thr)]


def save_rejector(dest: Path, clf, meta: dict) -> Path:
    dest = assert_safe_filter_dest(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in FORBIDDEN_WEIGHT_DIRS:
        if name in dest.parts:
            raise SystemExit(f"Refusing extra-filter dest {dest}: must not overwrite {name}.")
    path = dest / "rejector.joblib"
    import joblib

    joblib.dump({"clf": clf, "meta": meta}, path)
    (dest / "rejector_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_rejector(dest: Path):
    dest = assert_safe_filter_dest(dest)
    path = dest / "rejector.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    import joblib

    payload = joblib.load(path)
    return payload["clf"], payload["meta"]


def append_extra_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_extra_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def rows_to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    names = extra_feature_names()
    if not rows:
        return np.zeros((0, len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    return x, y


def load_slot1_covering(file_names: list[str], device: str) -> dict:
    """Fold-0 Slot1 cache plus a separate fold!=0 cache. Never overwrites fold-0 CSV."""
    import pandas as pd
    from filament.infer import run_yolo_infer
    from filament.metric import pred_to_df
    from score_faint_merge import (
        EMPTY_SLOT1_STEMS,
        SLOT1_CONF,
        SLOT1_IMGSZ,
        SLOT1_PRED_CSV,
        SLOT1_WEIGHTS,
        _stem_frames,
    )

    frames: dict = {}
    known: set[str] = set()
    if SLOT1_PRED_CSV.exists():
        df0 = pd.read_csv(SLOT1_PRED_CSV)
        fold0_stems = set(df0["filament_id"].str.split("_", n=1).str[0]) | set(EMPTY_SLOT1_STEMS)
        part = _stem_frames(df0, file_names)
        for file_name, sub in part.items():
            if Path(file_name).stem in fold0_stems:
                frames[file_name] = sub
                known.add(file_name)
    if SLOT1_FOLDNE0_CSV.exists():
        dfn = pd.read_csv(SLOT1_FOLDNE0_CSV)
        ne0_stems = set(dfn["filament_id"].str.split("_", n=1).str[0])
        part = _stem_frames(dfn, file_names)
        for file_name, sub in part.items():
            if file_name in known:
                continue
            if Path(file_name).stem in ne0_stems:
                frames[file_name] = sub
                known.add(file_name)
    missing = [fn for fn in file_names if fn not in known]
    if missing:
        if not SLOT1_WEIGHTS.exists():
            raise FileNotFoundError(SLOT1_WEIGHTS)
        from ultralytics import YOLO
        from tqdm import tqdm

        model = YOLO(str(SLOT1_WEIGHTS))
        new_frames = []
        for file_name in tqdm(missing, desc="slot1 fold!=0 m@1280"):
            rles = run_yolo_infer(
                model,
                TRAIN_IMG_DIR / file_name,
                conf=SLOT1_CONF,
                imgsz=SLOT1_IMGSZ,
                merge_gap=0,
                device=device,
            )
            frame = pred_to_df(Path(file_name).stem, rles)
            frames[file_name] = frame
            new_frames.append(frame)
        FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        added = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame(
            columns=["filament_id", "segmentation_rle"]
        )
        if SLOT1_FOLDNE0_CSV.exists():
            prev = pd.read_csv(SLOT1_FOLDNE0_CSV)
            added = pd.concat([prev, added], ignore_index=True)
        added.to_csv(SLOT1_FOLDNE0_CSV, index=False)
    empty = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return {fn: frames.get(fn, empty) for fn in file_names}


def mine_file_extras(
    file_name: str,
    slot_rles: list[str],
    cand_by_setting: dict[tuple[float, int], list[str]],
    gt_masks: list[np.ndarray],
    image: np.ndarray,
    residual: np.ndarray,
    fold: int,
) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for (thr, min_area), cands in cand_by_setting.items():
        extras = extra_rles(slot_rles, cands, iou_thr=MERGE_IOU)
        for rle in extras:
            key = rle
            if key in seen:
                continue
            seen.add(key)
            mask = decode_rle(rle)
            if int(mask.sum()) < FEATURE_MIN_AREA:
                continue
            rows.append(
                {
                    "file_name": file_name,
                    "fold": int(fold),
                    "thr": float(thr),
                    "min_area": int(min_area),
                    "rle": rle,
                    "features": extra_features(mask, image, residual).tolist(),
                    "label": label_extra(mask, gt_masks),
                    "area": int(mask.sum()),
                }
            )
    return rows


def gt_masks_for_file(file_name: str, views, by_file) -> list[np.ndarray]:
    from filament.coco_io import gt_records_for_files

    records = gt_records_for_files(views, by_file, [file_name])
    return [decode_rle(rec["rle"]) for rec in records]


def _done_files(meta_path: Path) -> set[str]:
    if not meta_path.exists():
        return set()
    payload = json.loads(meta_path.read_text())
    return set(payload.get("done_files") or [])


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
    """Infer ResNet + Slot1 on fold!=exclude_fold. Write extras jsonl. No fold-0 fit."""
    from tqdm import tqdm

    import torch
    from eval_unet import decode_instances, infer_prob
    from filament.coco_io import grouped_folds, load_coco
    from filament.infer import pick_device
    from filament.unet_resnet import build_resnet34_unet
    from gate_unet_family import live_mps_owner

    assert_safe_filter_dest(filter_dest)
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing extra-filter mine: {owner} owns MPS.")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=seed)
    mine_files = sorted(fn for fn, fold in fold_of.items() if fold != exclude_fold)
    if not mine_files:
        raise SystemExit("Refusing extra-filter mine: no fold!=0 files.")

    device = pick_device(device)
    slot1 = load_slot1_covering(mine_files, device)
    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    done = _done_files(EXTRAS_META)
    n_new = 0
    n_rows = 0
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for file_name in tqdm(mine_files, desc="mine fold!=0 ResNet extras"):
        if file_name in done:
            continue
        prob = infer_prob(model, train_img_dir / file_name, imgsz, device)
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
        "resnet_weights_read_only": str(Path(weights).resolve()),
        "overwrote_forbidden_unet": False,
        "classical_new_seeds": False,
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
    fit_fn=None,
    official_dest: Path | None = None,
    family: str = "unet_resnet_extra_filter_slot1_merge",
    builder: str = "build_resnet34_unet + logistic_morphometrics extras rejector",
    decode: str = "ResNet CC extras, fold!=0-trained rejector, then Slot1 NMS-merge",
    use_context: bool = False,
    use_crop: bool = False,
) -> dict:
    """Fit on cached fold!=0 extras, filter fold-0 extras, Slot1 NMS-merge, gate."""
    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances, infer_prob
    from eval_unet_merge import gate_unet_merge
    from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
    from filament.infer import pick_device
    from filament.merge import merge_rles
    from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
    from filament.paths import ARTIFACTS
    from filament.unet_resnet import build_resnet34_unet
    from gate_unet_family import EXTRA_FILTER_GATE, live_mps_owner
    from score_faint_merge import load_or_infer_slot1
    from score_family import year_slices

    assert_safe_filter_dest(filter_dest)
    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing extra-filter fold-0 merge: {owner} owns MPS.")

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
            f"Refusing extra-filter fold-0 merge: no fold!={fold} extras at {EXTRAS_JSONL}. "
            "Run scripts/mine_resnet_extras.py --run first."
        )
    if any(int(fold_of.get(r.get("file_name"), -1)) == fold for r in rows):
        raise SystemExit("Refusing extra-filter: a fit row belongs to the official fold.")

    need_slot = use_context or use_crop
    if use_crop and fit_fn is None:
        from resnet_extra_crop import fit_extra_rejector_crop

        fit_fn = fit_extra_rejector_crop
    fitter = fit_fn or fit_extra_rejector
    if need_slot:
        fit_files = sorted({str(r["file_name"]) for r in rows})
        slot1_fit = load_slot1_covering(fit_files, device="cpu")
        if use_crop:
            from resnet_extra_crop import rows_to_xy_crop

            x, y = rows_to_xy_crop(rows, slot1_fit)
        else:
            x, y = rows_to_xy_context(rows, slot1_fit)
    else:
        x, y = rows_to_xy(rows)
    clf, rejector_meta = fitter(x, y)
    if use_crop:
        from resnet_extra_crop import crop_pixel_names

        rejector_meta["feature_names"] = (
            list(extra_feature_names()) + list(context_feature_names()) + list(crop_pixel_names())
        )
        rejector_meta["type"] = "tiny_crop_cnn_plus_morph_ctx"
        rejector_meta["not_logistic_cut_retune"] = True
        rejector_meta["not_hgb_morph_only"] = True
    elif use_context:
        rejector_meta["feature_names"] = list(extra_feature_names()) + list(context_feature_names())
        rejector_meta["type"] = "logistic_morphometrics_plus_slot1_context"
        rejector_meta["not_logistic_cut_retune"] = True
        rejector_meta["not_hgb_morph_only"] = True
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

    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    probs = {}
    jpeg_cache = {}
    slot_mask_cache = {}
    for file_name in tqdm(val_files, desc="extra-filter fold-0 ResNet probs"):
        probs[file_name] = infer_prob(model, train_img_dir / file_name, imgsz, device)
        jpeg_cache[file_name] = jpeg_and_residual(file_name)
        if need_slot:
            slot_mask_cache[file_name] = [
                decode_rle(rle) for rle in slot1_frames[file_name]["segmentation_rle"].tolist()
            ]

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
                if use_crop:
                    from resnet_extra_crop import features_for_rles_crop

                    feats = features_for_rles_crop(
                        extras, image, residual, slot_mask_cache[file_name]
                    )
                elif use_context:
                    feats = features_for_rles_context(
                        extras, image, residual, slot_mask_cache[file_name]
                    )
                else:
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
    dest = Path(official_dest) if official_dest is not None else EXTRA_FILTER_GATE
    out = {
        "family": family,
        "scored": True,
        "status": "fold0_official",
        "weights": str(Path(weights).resolve()),
        "builder": builder,
        "decode": decode,
        "filter_dest": str(Path(filter_dest).resolve()),
        "rejector_path": str(rejector_path),
        "rejector": rejector_meta,
        "resnet_weights_read_only": True,
        "overwrote_forbidden_unet": False,
        "classical_new_seeds": False,
        "leakage": (
            "fold!=0 extras + train GT labels only; fold-0 is the official merge. "
            "Features from extra mask + JPEG / dark residual. No MAGFiLO test labels."
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
        "lock_rule": (
            "max PQ among FP < 2*Slot1 (886 when Slot1 FP=443); "
            "else max PQ (honest hose fail)"
        ),
        "v1_fp_hose_lesson": (
            "TinyUNet min_area=64: FP 443->2301. Replay: 642 still fp_exploded "
            "(FP 1056>=886, +145 TP); 1034 first clean hose (FP 806, +93 TP). "
            "Sweep 642+1034 after extra filter; do not invert public."
        ),
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    text = json.dumps(out, indent=2) + "\n"
    dest.write_text(text)
    (FAST_ARTIFACTS / dest.name).write_text(text)
    return out
