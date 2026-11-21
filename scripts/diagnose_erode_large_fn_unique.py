#!/usr/bin/env python3
"""CPU unique cover of Slot1-erode large-identifiable FNs across on-disk CSVs.

Diagnosis only. extras-OR stays vetoed even if unique >= 220.
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
from diagnose_slot1_large_fn_for_unet import IOU_THR, is_large_identifiable
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

OUT = ARTIFACTS / "erode_large_fn_unique_after_d.json"


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    erode = _by_stem(ERODE_CSV)
    loaded = [(n, p, _by_stem(p)) for n, p in CANDIDATES if p.exists()]
    names = [n for n, _, _ in loaded]
    n_large = 0
    any_hit = 0
    pattern_counts: Counter[str] = Counter()
    for file_name in val_files:
        stem = Path(file_name).stem
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
                if hitters:
                    any_hit += 1
                pattern_counts["+".join(hitters) if hitters else "none"] += 1

    yolo = {"A_yolo11l_identity", "B_yolo11l_erode3x1", "C_yolo11m_seg_erode_gt", "D_yolo11l_seg_erode_gt"}
    yolo_or_unet = yolo | {"unet_resnet_slot1_merge"}
    unique_yolo = sum(
        c for pat, c in pattern_counts.items() if pat != "none" and set(pat.split("+")) & yolo
    )
    unique_yolo_unet = sum(
        c for pat, c in pattern_counts.items() if pat != "none" and set(pat.split("+")) & yolo_or_unet
    )
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_erode_large_fn_unique",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "did_not_train": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "recommend_start": False,
        "next_id": None,
        "erode_large_identifiable_fn": n_large,
        "unique_any_csv": any_hit,
        "unique_none": n_large - any_hit,
        "unique_yolo_A_B_C_D": unique_yolo,
        "unique_yolo_or_unet": unique_yolo_unet,
        "k_needed_local_0.50": 220,
        "unique_any_reaches_220": any_hit >= 220,
        "can_supply_k_220_zero_fp": False,
        "extras_or_veto_even_if_unique_ge_220": True,
        "pattern_counts": dict(pattern_counts.most_common(20)),
        "why": (
            "Unique hits of already-scored CSVs on erode large FNs. "
            "OR onto public-best stays vetoed. Zero-FP supply is false because "
            "those CSVs add FPs (B union +119 FP). next_id stays null."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "erode_large_identifiable_fn",
        "unique_any_csv",
        "unique_none",
        "unique_yolo_A_B_C_D",
        "unique_any_reaches_220",
        "can_supply_k_220_zero_fp",
        "next_id",
    )}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
