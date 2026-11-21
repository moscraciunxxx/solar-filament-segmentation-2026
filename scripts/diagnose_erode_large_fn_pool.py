#!/usr/bin/env python3
"""Expand unique large-FN cover to every on-disk pred CSV except Merckle.

Diagnosis only. extras-OR stays vetoed even if unique >= 220.
Also characterize the still-uncovered large identifiable FNs.
No train, no MPS, no submit.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_erode_large_fn_overlap import CANDIDATES, ERODE_CSV, _by_stem, _layers
from diagnose_slot1_large_fn_for_unet import (
    AREA_P25,
    IOU_THR,
    YEAR_KEYS,
    is_large_identifiable,
    jpeg_year,
    quartiles,
    year_partition,
)
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

OUT = ARTIFACTS / "erode_large_fn_pool_after_d.json"
SKIP_STEMS = {
    "slot1_erode_3x1_fold0_pred",
    "slot1_identity_sanity_erode3x1_fold0_pred",
    "merckle046_fold0_pred",
}
WAVE16_NAMES = {name for name, _ in CANDIDATES}
ALIAS = {path.resolve(): name for name, path in CANDIDATES}


def _name_for(path: Path) -> str:
    return ALIAS.get(path.resolve(), path.stem.replace("_fold0_pred", "").replace("_pred", ""))


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    erode = _by_stem(ERODE_CSV)
    csvs = sorted(
        p for p in FAST_ARTIFACTS.glob("*_pred.csv") if p.stem not in SKIP_STEMS
    )
    loaded = [(_name_for(p), p, _by_stem(p)) for p in csvs]
    names = [n for n, _, _ in loaded]
    n_large = 0
    any_hit = 0
    wave16_hit = 0
    expanded_only = 0
    pattern_counts: Counter[str] = Counter()
    uncovered_year: Counter[str] = Counter()
    uncovered_cat: Counter[str] = Counter()
    uncovered_areas: list[float] = []
    hit_year: Counter[str] = Counter()

    for file_name in val_files:
        stem = Path(file_name).stem
        year_key = year_partition(jpeg_year(file_name))
        erode_layers = _layers(erode.get(stem, []))
        other_layers = {n: _layers(mp.get(stem, [])) for n, _, mp in loaded}
        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_layers = torch.from_numpy(
                rles_to_layers([encode_rle(rasterize_polygons([fil.polygon])) for fil in fils])
            )
            if erode_layers.shape[0] == 0:
                erode_max = torch.zeros(len(fils))
            else:
                iou, _ = overlap_matrices(gt_layers, erode_layers)
                erode_max = iou.max(dim=1).values
            other_max = {}
            for name, layers in other_layers.items():
                if layers.shape[0] == 0:
                    other_max[name] = torch.zeros(len(fils))
                else:
                    oi, _ = overlap_matrices(gt_layers, layers)
                    other_max[name] = oi.max(dim=1).values
            for i, fil in enumerate(fils):
                if float(erode_max[i]) > IOU_THR:
                    continue
                if not is_large_identifiable(fil):
                    continue
                n_large += 1
                hitters = tuple(n for n in names if float(other_max[n][i]) > IOU_THR)
                wave16 = tuple(n for n in hitters if n in WAVE16_NAMES)
                if hitters:
                    any_hit += 1
                    hit_year[year_key] += 1
                    if wave16:
                        wave16_hit += 1
                    else:
                        expanded_only += 1
                else:
                    uncovered_year[year_key] += 1
                    uncovered_cat[str(int(fil.category_id))] += 1
                    uncovered_areas.append(float(fil.area))
                pattern_counts["+".join(hitters) if hitters else "none"] += 1

    unique_none = n_large - any_hit
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_erode_large_fn_full_pool",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "did_not_train": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "recommend_start": False,
        "next_id": None,
        "csvs": names,
        "skipped": sorted(SKIP_STEMS),
        "erode_large_identifiable_fn": n_large,
        "unique_any_csv": any_hit,
        "unique_none": unique_none,
        "unique_wave16_six": wave16_hit,
        "unique_expanded_only": expanded_only,
        "k_needed_local_0.50": 220,
        "unique_any_reaches_220": any_hit >= 220,
        "can_supply_k_220_zero_fp": False,
        "extras_or_veto_even_if_unique_ge_220": True,
        "uncovered": {
            "n": unique_none,
            "year": {k: uncovered_year.get(k, 0) for k in YEAR_KEYS},
            "category_id": dict(sorted(uncovered_cat.items())),
            "area_gt": AREA_P25,
            "area": quartiles(uncovered_areas) if uncovered_areas else {},
        },
        "covered_year": {k: hit_year.get(k, 0) for k in YEAR_KEYS},
        "pattern_counts_top": dict(pattern_counts.most_common(15)),
        "why": (
            "Full on-disk pred pool unique hits on erode large FNs. "
            "OR onto public-best stays vetoed. Zero-FP supply is false. "
            "next_id stays null."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "erode_large_identifiable_fn",
        "unique_any_csv",
        "unique_none",
        "unique_expanded_only",
        "unique_any_reaches_220",
        "can_supply_k_220_zero_fp",
        "next_id",
        "uncovered",
    )}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
