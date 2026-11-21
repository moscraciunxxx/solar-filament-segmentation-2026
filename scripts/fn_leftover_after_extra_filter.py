#!/usr/bin/env python3
"""CPU-only FN leftover after ctx extra-filter. Reads official JSONs only.

No train, no infer, no MPS, no rejector fit, no submit, no upload.
fold-0 extras were never written (jsonl is fold!=0); counts are locked-row
arithmetic (+82 vs +59 vs 215), not instance overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "fn_leftover_after_extra_filter.json"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def pq_of(tp: int, fp: int, fn: int, sq: float) -> dict:
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "SQ": sq,
        "RQ": rq,
        "denom": denom,
        "pq": sq * rq,
    }


def min_new_tps(slot1_tp: int, n_gt: int, fp: int, sq: float, target: float) -> dict:
    """Smallest k>=0 s.t. PQ(slot1_tp+k, fp, n_gt-(slot1_tp+k), sq) >= target."""
    algebraic = None
    needed = None
    just_below = None
    at = None
    for k in range(0, n_gt - slot1_tp + 1):
        tp = slot1_tp + k
        row = pq_of(tp, fp, n_gt - tp, sq)
        if row["pq"] >= target:
            needed = k
            at = {"new_tps": k, **row}
            break
        just_below = {"new_tps": k, **row}
    # closed form: target = 2*sq*tp / (tp + fp + n_gt) => tp = target*(fp+n_gt) / (2*sq - target)
    denom_a = 2.0 * sq - target
    if denom_a > 0:
        algebraic = target * (fp + n_gt) / denom_a - slot1_tp
    return {
        "target_pq": target,
        "SQ": sq,
        "FP": fp,
        "algebraic_new_tps": algebraic,
        "new_tps_needed": needed,
        "just_below": just_below,
        "at_threshold": at,
    }


def main() -> None:
    ctx = load("artifacts/unet_resnet_extra_filter_ctx_slot1_merge_fold0_official.json")
    unf = load("artifacts/unet_resnet34_slot1_merge_fold0_official.json")
    logistic = load("artifacts/unet_resnet_extra_filter_slot1_merge_fold0_official.json")
    leftover_ctx = load("artifacts/leftover_after_ctx.json")
    leftover_crop = load("artifacts/leftover_after_crop.json")
    leftover_spine = load("artifacts/leftover_after_spine.json")
    leftover_if_crop = load("artifacts/leftover_if_crop_holds.json")
    large_fn = load("artifacts/slot1_large_fn_for_unet.json")
    hose = load("artifacts/unet_merge_fp_hose_math.json")
    phys = load("artifacts/yolo11s_seg_phys_fold0_official.json")
    ch3 = load("artifacts/unet_3ch_disk_slot1_merge_fold0_official.json")
    extras_meta = json.loads(
        Path(
            "~/.cache/filament-segmentation-2026"
            "/artifacts/resnet_extras_foldne0_meta.json"
        ).read_text()
    )

    slot1 = ctx["slot1"]
    ctx_lock = ctx["locked_row"]
    unf_lock = unf["locked_row"]
    unf_06 = next(
        r
        for r in unf["sweep_merged"]
        if r["thr"] == 0.6 and r.get("min_area") == 1034
    )
    ctx_05 = next(
        r
        for r in ctx["sweep_merged"]
        if r["thr"] == 0.5 and r.get("min_area") == 1034
    )

    n_gt = large_fn["n_gt"]
    assert slot1["tp"] + slot1["fn"] == n_gt

    # Canonical project story: unfiltered locked +82; ctx HOLD +59; 215 extras @1034.
    extras_pool_tp = unf["gate"]["vs_slot1"]["d_tp"]  # 82
    filter_kept_tp = ctx["gate"]["vs_slot1"]["d_tp"]  # 59
    in_extra_killed_tp = extras_pool_tp - filter_kept_tp  # 23
    fn_never_proposed = slot1["fn"] - extras_pool_tp  # 582
    assert ctx_lock["fn"] == fn_never_proposed + in_extra_killed_tp

    # Same-cut (thr 0.6, min_area 1034, 215 extras) — one extra / one TP off.
    same_cut_pool_tp = unf_06["tp"] - slot1["tp"]
    same_cut_killed = same_cut_pool_tp - filter_kept_tp
    same_cut_never = slot1["fn"] - same_cut_pool_tp

    sq_unf = unf_lock["mean_tp_iou"]
    sq_slot1 = slot1["mean_tp_iou"]
    sq_plan = hose["slot1_frozen_fold0"]["planning_SQ"]

    oracle_unf_sq = pq_of(unf_lock["tp"], slot1["fp"], unf_lock["fn"], sq_unf)
    oracle_plan = pq_of(unf_lock["tp"], slot1["fp"], unf_lock["fn"], sq_plan)
    oracle_slot1_sq = pq_of(unf_lock["tp"], slot1["fp"], unf_lock["fn"], sq_slot1)
    recover_killed_same_fp = pq_of(
        ctx_lock["tp"] + in_extra_killed_tp,
        ctx_lock["fp"],
        ctx_lock["fn"] - in_extra_killed_tp,
        ctx_lock["mean_tp_iou"],
    )

    need_048 = min_new_tps(slot1["tp"], n_gt, slot1["fp"], sq_plan, 0.48)
    need_042 = min_new_tps(slot1["tp"], n_gt, slot1["fp"], sq_plan, 0.42)
    pool_157 = pq_of(slot1["tp"] + 157, slot1["fp"], slot1["fn"] - 157, sq_plan)

    extras_ne0_pos = ctx["rejector"]["n_pos"]
    extras_ne0_neg = ctx["rejector"]["n_neg"]
    extras_ne0_n = extras_ne0_pos + extras_ne0_neg
    fold0_extra_prec = extras_pool_tp / ctx_lock["n_extras_before_filter"]

    out = {
        "written": "2026-09-11T15:55:00Z",
        "cpu_only": True,
        "did_not_train": True,
        "did_not_infer": True,
        "did_not_infer_mps": True,
        "did_not_fit_rejector": True,
        "did_not_fit_rejector_on_fold0": True,
        "did_not_submit": True,
        "did_not_upload": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "instance_overlap_counted": False,
        "instance_overlap_why_impossible_without_infer": (
            "resnet_extras_foldne0.jsonl / meta exclude fold 0 (exclude_fold=0, n_done=565). "
            "No fold-0 extra masks, RLEs, or per-FN IoU table exist in artifacts or cache. "
            "Matching leftover FNs to the 215 extras requires remine/infer of ResNet CC on fold-0. "
            "This file estimates from locked rows only."
        ),
        "estimate_method": "+82 vs +59 vs 215 extras (locked-row TP arithmetic)",
        "sources": {
            "ctx_official": "artifacts/unet_resnet_extra_filter_ctx_slot1_merge_fold0_official.json",
            "unfiltered_resnet_merge": "artifacts/unet_resnet34_slot1_merge_fold0_official.json",
            "logistic_official": "artifacts/unet_resnet_extra_filter_slot1_merge_fold0_official.json",
            "leftover_after_ctx": "artifacts/leftover_after_ctx.json",
            "leftover_after_crop": "artifacts/leftover_after_crop.json",
            "leftover_after_spine": "artifacts/leftover_after_spine.json",
            "leftover_if_crop_holds": "artifacts/leftover_if_crop_holds.json",
            "slot1_large_fn": "artifacts/slot1_large_fn_for_unet.json",
            "hose_math": "artifacts/unet_merge_fp_hose_math.json",
            "phys_yolo": "artifacts/yolo11s_seg_phys_fold0_official.json",
            "ch3_unfiltered_merge": "artifacts/unet_3ch_disk_slot1_merge_fold0_official.json",
            "extras_jsonl_meta": "WEIGHTS cache artifacts/resnet_extras_foldne0_meta.json",
        },
        "locked_rows": {
            "slot1": {
                "tp": slot1["tp"],
                "fp": slot1["fp"],
                "fn": slot1["fn"],
                "pq": slot1["pq"],
                "mean_tp_iou": slot1["mean_tp_iou"],
                "n_pred": slot1["n_pred"],
            },
            "unfiltered_resnet_locked_0.5_1034": {
                "thr": unf_lock["thr"],
                "min_area": unf_lock["min_area"],
                "tp": unf_lock["tp"],
                "fp": unf_lock["fp"],
                "fn": unf_lock["fn"],
                "pq": unf_lock["pq"],
                "n_unet_added": unf_lock["n_unet_added"],
                "d_tp": extras_pool_tp,
            },
            "unfiltered_resnet_0.6_1034": {
                "thr": unf_06["thr"],
                "min_area": unf_06["min_area"],
                "tp": unf_06["tp"],
                "fp": unf_06["fp"],
                "fn": unf_06["fn"],
                "pq": unf_06["pq"],
                "n_unet_added": unf_06["n_unet_added"],
                "d_tp": same_cut_pool_tp,
            },
            "ctx_extra_filter_hold_0.6_1034": {
                "thr": ctx_lock["thr"],
                "min_area": ctx_lock["min_area"],
                "tp": ctx_lock["tp"],
                "fp": ctx_lock["fp"],
                "fn": ctx_lock["fn"],
                "pq": ctx_lock["pq"],
                "n_extras_before_filter": ctx_lock["n_extras_before_filter"],
                "n_extras_kept": ctx_lock["n_extras_kept"],
                "d_tp": filter_kept_tp,
                "rejector_thr": ctx_lock["rejector_thr"],
            },
            "ctx_at_unfiltered_cut_0.5_1034": {
                "tp": ctx_05["tp"],
                "fp": ctx_05["fp"],
                "fn": ctx_05["fn"],
                "n_extras_before_filter": ctx_05["n_extras_before_filter"],
                "n_extras_kept": ctx_05["n_extras_kept"],
                "d_tp": ctx_05["tp"] - slot1["tp"],
            },
            "logistic_hold_same_215": {
                "tp": logistic["locked_row"]["tp"],
                "fp": logistic["locked_row"]["fp"],
                "fn": logistic["locked_row"]["fn"],
                "pq": logistic["locked_row"]["pq"],
                "n_extras_before_filter": logistic["locked_row"]["n_extras_before_filter"],
                "n_extras_kept": logistic["locked_row"]["n_extras_kept"],
                "d_tp": logistic["gate"]["vs_slot1"]["d_tp"],
            },
        },
        "fn_split_of_remaining_605": {
            "remaining_fn": ctx_lock["fn"],
            "in_extra_pool_killed_tps": in_extra_killed_tp,
            "fns_never_in_extras": fn_never_proposed,
            "identity": "605 = 23 killed extra-TPs + 582 never proposed (Slot1 FN 664 − 82)",
            "extras_pool_size": ctx_lock["n_extras_before_filter"],
            "extras_kept": ctx_lock["n_extras_kept"],
            "extras_rejected": ctx_lock["n_extras_before_filter"] - ctx_lock["n_extras_kept"],
            "unfiltered_extra_tps": extras_pool_tp,
            "filter_kept_extra_tps": filter_kept_tp,
            "unfiltered_extra_fps_approx": ctx_lock["n_extras_before_filter"] - extras_pool_tp,
            "fold0_extra_precision_approx": fold0_extra_prec,
            "same_cut_0.6_1034_note": (
                "Unfiltered locked is thr 0.5 / 214 extras / +82 TP / FN 582. "
                "Ctx HOLD is thr 0.6 / 215 extras / +59 TP / FN 605. "
                "Same-cut 0.6/1034 unfiltered is +83 TP / FN 581 → killed 24 / never 581. "
                "Primary table uses the project's +82 vs +59 locked story."
            ),
            "same_cut_0.6_1034": {
                "unfiltered_d_tp": same_cut_pool_tp,
                "in_extra_killed_tps": same_cut_killed,
                "fns_never_in_extras": same_cut_never,
            },
            "near_miss_caveat": (
                "Count is official IoU>0.5 TP movement, not geometric near-overlap. "
                "Some of the ~133 extra-FPs could sit at IoU in (0, 0.5] on the 582; "
                "that cannot be counted without remine."
            ),
        },
        "extras_pool_ceiling": {
            "definition": "oracle keep-TP-only ResNet extras (add 82 TPs, 0 extra FPs)",
            "approx_quoted_in_leftovers": 0.43,
            "pq_unfiltered_measured_sq": oracle_unf_sq["pq"],
            "pq_planning_sq_0.655": oracle_plan["pq"],
            "pq_slot1_sq": oracle_slot1_sq["pq"],
            "rounded": 0.43,
            "unfiltered_measured_sq": oracle_unf_sq,
            "planning_sq_0.655": oracle_plan,
            "if_return_23_killed_keep_ctx_fp": recover_killed_same_fp,
            "source_leftover_after_spine": leftover_spine["next_legal_leftover"]["why"],
            "source_leftover_after_ctx": leftover_ctx["hand_feature_plateau"],
        },
        "fns_never_in_extras": fn_never_proposed,
        "proposal_leftover": {
            "n": fn_never_proposed,
            "of_slot1_fn": slot1["fn"],
            "of_remaining_fn": ctx_lock["fn"],
            "slot1_large_identifiable_fn": large_fn["count"],
            "true_miss_iou_le_0.3_vs_slot1": large_fn["iou_vs_slot1"]["true_miss_iou_le_0.3"],
            "near_miss_0.3_to_0.5_vs_slot1": large_fn["iou_vs_slot1"][
                "near_miss_iou_in_0.3_exclusive_to_0.5_inclusive"
            ],
            "how_many_of_157_are_in_the_82": (
                "Unknown without instance match. Bounds: at most 82 of 157 recovered "
                "(>=75 true-miss still never proposed); at least 0 of 157 recovered "
                "(all 82 from near-miss / other FNs)."
            ),
            "best_other_dense_proposal_fn_leftover": {
                "family": "unet_3ch_disk_slot1_merge unfiltered",
                "d_tp": ch3["gate"]["vs_slot1"]["d_tp"],
                "fn": ch3["locked_row"]["fn"],
                "note": "3ch leftover FN 566 vs ResNet unfiltered 582 — still ~560 never proposed by that family.",
            },
        },
        "new_proposal_family_required_for_local_0.48": True,
        "new_proposal_family_required_why": (
            "Oracle extras ceiling is ~0.43. Returning the 23 filter-killed TPs at ctx FP "
            "is ~0.421. The 157 true-miss pool at Slot1 FP/SQ is ~0.455. Local 0.48 at "
            "SQ=0.655 FP=443 needs +251 TP (TP 1176). Extras supply only +82. "
            "EdgeAtt is same gray as ResNet HOLD. Extra-filter (logistic/HGB/ctx/crop) "
            "is exhausted on this 215-pool. A different proposal family is required."
        ),
        "not_EdgeAtt": True,
        "not_extra_filter": True,
        "local_0.48_math": {
            "formula": "PQ = SQ * TP / (TP + 0.5*FP + 0.5*FN); n_gt=1589",
            "sq_planning": sq_plan,
            "fp_held_slot1": slot1["fp"],
            "new_tps_for_0.42": need_042,
            "new_tps_for_0.48": need_048,
            "extras_supply": extras_pool_tp,
            "still_need_from_never_proposed_for_0.48": (
                None
                if need_048["new_tps_needed"] is None
                else need_048["new_tps_needed"] - extras_pool_tp
            ),
            "true_miss_157_ceiling_sq655_fp443": pool_157,
            "true_miss_157_ceiling_lt_0.48": pool_157["pq"] < 0.48,
        },
        "can_a_new_yolo_residual_stack_help": False,
        "yolo_residual_stack": {
            "verdict": "no",
            "helps_killed_23": False,
            "helps_never_proposed_582_enough_for_0.48": False,
            "why": (
                "legal_climb_methods local_dark_residual_stack as YOLO is yolo11s_seg_phys: "
                "locked PQ 0.357, TP 798, FN 791, d_tp -11 vs s_medoid. Closed. "
                "The same stack as U-Net channels is 3ch (unfiltered +101 TP, extra-filter "
                "HOLD 0.4095). leftover_if_crop_holds / post_patch: do not retrain "
                "yolo_seg_phys; residual stack was Method 3 U-Net, already scored. "
                "Another YOLO residual cannot invent the 582 boxes m@1280 never fired "
                "and does not return filter-killed extras (that is keep-set, crop already HOLD)."
            ),
            "phys_locked": phys["locked_row"],
            "phys_gate": phys["gate"]["decision"],
            "playbook": "do not restart phys / tiles / another gray-medoid YOLO",
        },
        "table": [
            {
                "row": "in-extra killed TPs",
                "n": in_extra_killed_tp,
                "meaning": "of 605 leftover FNs, extras had already proposed them (+82 unfiltered); ctx rejector dropped them (+59 kept)",
                "can_extra_filter_recover": "crop-CNN already tried; HOLD 0.4125 worse than ctx",
            },
            {
                "row": "never-proposed FNs",
                "n": fn_never_proposed,
                "meaning": "Slot1 FN 664 minus unfiltered extra TPs 82; no ResNet CC extra at IoU>0.5",
                "can_extra_filter_recover": False,
            },
            {
                "row": "can-a-new-YOLO-residual-stack-help",
                "n": 0,
                "meaning": "phys_stack YOLO already died 0.357; 3ch residual U-Net already HOLD",
                "can_extra_filter_recover": False,
                "verdict": False,
            },
        ],
        "foldne0_extras_prior_not_a_fold0_fit": {
            "jsonl_is_fold_ne0": True,
            "exclude_fold": extras_meta["exclude_fold"],
            "n_done": extras_meta["n_done"],
            "n_pos": extras_ne0_pos,
            "n_neg": extras_ne0_neg,
            "n_rows": extras_ne0_n,
            "pool_precision": extras_ne0_pos / extras_ne0_n,
            "note": "Sanity only (fold!=0 extra precision ~0.32 vs fold-0 82/215~0.38). Did not fit or retune a rejector.",
        },
        "extra_filter_family_exhausted": leftover_crop.get("extra_filter_family_exhausted", True),
        "next_legal_leftover": leftover_crop.get("next_legal_leftover"),
        "closed": leftover_if_crop.get("closed"),
        "do_not_start": leftover_crop.get("do_not_start"),
        "quota_remaining": leftover_crop.get("quota_remaining"),
        "slot2_stays": leftover_crop.get("slot2_stays"),
        "certified_top10": False,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "wrote": str(OUT),
        "in_extra_killed_tps": in_extra_killed_tp,
        "fns_never_in_extras": fn_never_proposed,
        "extras_pool_ceiling": round(oracle_unf_sq["pq"], 4),
        "new_proposal_family_required_for_local_0.48": True,
        "can_a_new_yolo_residual_stack_help": False,
        "do_not_submit": True,
    }, indent=2))


if __name__ == "__main__":
    main()
