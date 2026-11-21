#!/usr/bin/env python3
"""A-priori per-image expert gates on grouped fold 0. Official PQ only.

Rules below are fixed before any gate PQ is computed. One expert's rows per
image. Never union RLEs. CPU only. Does not train, does not submit, does not
read MAGFiLO test labels.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

_tensor_to = torch.Tensor.to


def _cpu_only_to(self, *args, **kwargs):
    device = args[0] if args else kwargs.get("device")
    if device is not None and "mps" in str(device).lower():
        raise RuntimeError("MPS is forbidden for this run")
    return _tensor_to(self, *args, **kwargs)


torch.Tensor.to = _cpu_only_to

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED

SLOT1_REF = 0.4365241382777018
BAR_4DP = 0.4365

# Declared before scoring. Do not reorder or retarget from fold-0 PQ.
# Pre-2017 local expert is Slot1 erode: the fold!=0-locked thick local file.
# thin@1408 is only the empty-count fallback, not a wholesale local replacement.
# CLAHE is present and unused. Year-shift is used only on filename year >= 2017.
GATES = (
    "always_slot1",
    "year_pre2017_slot1_2017plus_shift",
    "density_slot1_zero_then_thin1408",
)

EXPERTS = {
    "slot1_erode": FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv",
    "thin_s1408": FAST_ARTIFACTS / "yolo11s_seg_erode_gt_1408_fold0_conf0.35_pred.csv",
    "clahe_matched": FAST_ARTIFACTS / "yolo11s_seg_clahe_matched_fold0_conf0.35_pred.csv",
    "year_shift": FAST_ARTIFACTS
    / "yolo_yolo11s_seg_shift_fold0_conf0.35_sz1024_m0_base_shift_pq.csv",
}

OUT = ARTIFACTS / "moe_per_image_gate.json"


def jpeg_year(file_name: str) -> int:
    return int(Path(file_name).stem[:4])


def frames_from_csv(path: Path, fold0: list[str]) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        raise SystemExit(f"unexpected columns in {path}: {list(df.columns)}")
    stems = df["filament_id"].str.split("_", n=1).str[0]
    grouped = {stem: part.reset_index(drop=True) for stem, part in df.groupby(stems, sort=False)}
    empty = df.iloc[0:0].copy()
    return {fn: grouped.get(Path(fn).stem, empty).copy() for fn in fold0}


def assemble(chosen: dict[str, pd.DataFrame], files: list[str]) -> pd.DataFrame:
    parts = [chosen[fn] for fn in files if len(chosen[fn])]
    if not parts:
        return pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    return pd.concat(parts, ignore_index=True)


def route(gate: str, file_name: str, experts: dict[str, dict[str, pd.DataFrame]]) -> str:
    if gate == "always_slot1":
        return "slot1_erode"
    if gate == "year_pre2017_slot1_2017plus_shift":
        if jpeg_year(file_name) < 2017:
            return "slot1_erode"
        return "year_shift"
    if gate == "density_slot1_zero_then_thin1408":
        if len(experts["slot1_erode"][file_name]) == 0:
            return "thin_s1408"
        return "slot1_erode"
    raise SystemExit(f"unknown gate {gate}")


def slice_note(name: str, beats: bool) -> str:
    if not beats:
        return (
            "No a-priori gate beat Slot1. Nothing to upload. "
            "Grouped fold-0 PQ is in-sample and is not a public certificate."
        )
    if name == "always_slot1":
        return "Winner is the Slot1 incumbent, not a new cut."
    if name == "year_pre2017_slot1_2017plus_shift":
        return (
            "In-sample fold-0 selection of a predeclared year switch. "
            "Public-slice bet: the gain is measured on the same grouped fold-0 "
            "that produced Slot1 0.4365. Not extras-OR. Not a threshold edit of "
            "Slot1 masks: each 2017+ image is the year-shift expert alone. Do not upload."
        )
    return (
        "In-sample fold-0 operating rule: images with zero Slot1 instances take "
        "thin@1408 wholesale. Public-slice selection of when to swap experts, "
        "closer to a cut edit than a hidden-exam representation. Not extras-OR. Do not upload."
    )


def main() -> None:
    for name, path in EXPERTS.items():
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")

    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    experts = {name: frames_from_csv(path, fold0) for name, path in EXPERTS.items()}

    print(f"encoding fold-0 GT for {len(fold0)} images", flush=True)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, fold0))

    results = {}
    routed_counts = {}
    for gate in GATES:
        chosen = {}
        counts: dict[str, int] = {}
        for fn in fold0:
            expert = route(gate, fn, experts)
            chosen[fn] = experts[expert][fn]
            counts[expert] = counts.get(expert, 0) + 1
        pred_df = assemble(chosen, fold0)
        # One expert per image: the assembled ids must be exactly that expert's rows.
        print(f"scoring {gate} n_pred={len(pred_df)}", flush=True)
        br = score_pq(gt_df, pred_df)
        results[gate] = {
            "pq": br.pq,
            "tp": br.tp,
            "fp": br.fp,
            "fn": br.fn,
            "mean_tp_iou": br.mean_tp_iou,
            "n_pred": int(len(pred_df)),
            "n_images": len(fold0),
            "n_images_by_expert": counts,
            "d_pq_vs_slot1": br.pq - SLOT1_REF,
        }
        routed_counts[gate] = counts
        print(
            f"{gate} pq={br.pq:.12f} tp={br.tp} fp={br.fp} fn={br.fn}",
            flush=True,
        )

    slot1_pq = results["always_slot1"]["pq"]
    if abs(slot1_pq - SLOT1_REF) > 1e-9:
        raise SystemExit(
            f"Slot1 reproduce failed: got {slot1_pq} expected {SLOT1_REF}. "
            "Not writing a gate decision."
        )

    def sort_key(name: str) -> tuple:
        # Higher PQ wins. Exact tie keeps the earlier a-priori gate (Slot1 first).
        return (results[name]["pq"], -GATES.index(name))

    winner = max(GATES, key=sort_key)
    winner_pq = results[winner]["pq"]
    beats_bar = bool(winner_pq > SLOT1_REF and winner_pq > BAR_4DP)
    parent = "HOLD_FOR_PARENT" if beats_bar else "NO_LOOK"

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cpu_only": True,
        "mps_used": False,
        "did_not_train": True,
        "kaggle_submit_called": False,
        "extras_or": False,
        "per_image_one_expert": True,
        "magfilo_test_labels_read": False,
        "remaining_looks": 2,
        "pair_stays": [56173362, 56148889],
        "slot1_ref_pq": SLOT1_REF,
        "bar_4dp": BAR_4DP,
        "beats_bar_rule": (
            "true only if the winning a-priori gate pq is strictly above "
            "Slot1 0.4365241382777018 and above 0.4365. Reproducing Slot1 is not a beat."
        ),
        "fold": 0,
        "n_images": len(fold0),
        "n_pre2017": sum(1 for fn in fold0 if jpeg_year(fn) < 2017),
        "n_2017plus": sum(1 for fn in fold0 if jpeg_year(fn) >= 2017),
        "a_priori_declaration": {
            "declared_before_scoring": True,
            "pre2017_local_expert": "slot1_erode",
            "pre2017_reason": (
                "Slot1 erode is the fold!=0-locked thick local expert. "
                "thin@1408 is not substituted for every pre-2017 image. "
                "CLAHE is not the year-local expert."
            ),
            "year_2017plus_expert": "year_shift",
            "year_shift_csv_exists": True,
            "density_rule": "if Slot1 pred count on that image is 0, take thin_s1408; else Slot1",
            "density_is_not_a_learned_count_ratio": True,
            "clahe_available_unused": True,
            "tie_break": "higher pq; exact tie keeps earlier declared gate, Slot1 first",
        },
        "experts": {name: str(path) for name, path in EXPERTS.items()},
        "gates": results,
        "winning_gate": winner,
        "d_pq": results[winner]["d_pq_vs_slot1"],
        "beats_bar": beats_bar,
        "pq_gt_0_4365": bool(winner_pq > BAR_4DP),
        "parent_decision": parent,
        "public_slice_note": slice_note(winner, beats_bar),
        "in_sample_fold0_is_not_a_public_certificate": True,
        "do_not_submit": True,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "winning_gate": winner, "beats_bar": beats_bar, "parent_decision": parent}, indent=2))


if __name__ == "__main__":
    main()
