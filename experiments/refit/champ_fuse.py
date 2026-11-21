#!/usr/bin/env python3
"""Champion 72-voter fusion -> submission CSV. VERBATIM port of scripts/make_ensemble3_tta_submission.py
(fuse_masks_fast, IoU 0.5, min votes 24, vote >= (n+1)//2, area >= 96, then filament.infer.yolo_masks_to_instances(merge_gap=0):
96..80000 px + resolve_overlaps) and eval_parent_tta._decode_all/_box. Payload order = seed order, then each cache's key order.
usage: champ_fuse.py OUT.csv TEST_IMGDIR CACHE_SEED10 CACHE_SEED11 CACHE_SEED12"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
KIT = Path(__file__).resolve().parent; sys.path.insert(0, str(KIT))
from filament.infer import yolo_masks_to_instances
from filament.rle import decode_rle, encode_rle
IMAGE_SIZE = 2048; IOU = 0.5; VOTES = 24
def _decode_all(payload):
    masks = []
    for rles in payload.values():
        for rle in rles:
            masks.append(decode_rle(rle))
    return masks
def _box(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
def fuse_masks_fast(masks, iou_thr, min_votes):
    if not masks:
        return []
    boxes = [_box(m) for m in masks]
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks)
    out = []
    for i in order:
        if used[i] or boxes[i] is None:
            continue
        group = [i]
        used[i] = True
        ax0, ay0, ax1, ay1 = boxes[i]
        ai = areas[i]
        mi = masks[i]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            ix0 = max(ax0, bx0); iy0 = max(ay0, by0); ix1 = min(ax1, bx1) + 1; iy1 = min(ay1, by1) + 1
            inter = int(np.logical_and(mi[iy0:iy1, ix0:ix1], masks[j][iy0:iy1, ix0:ix1]).sum())
            union = ai + areas[j] - inter
            if union and (inter / union) >= iou_thr:
                group.append(j)
                used[j] = True
        if len(group) < min_votes:
            continue
        vote = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for k in group:
            vote += masks[k]
        merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
        if int(merged.sum()) >= 96:
            out.append(merged)
    return yolo_masks_to_instances(out, merge_gap=0)
def pred_to_df(stem, rles):
    return pd.DataFrame([{"filament_id": f"{stem}_{i + 1}", "segmentation_rle": r} for i, r in enumerate(rles)], columns=["filament_id", "segmentation_rle"])
CACHES = NAMES = None
def _one(stem):
    payload = {}
    for name, c in zip(NAMES, CACHES):
        one = json.loads((c / f"{stem}.json").read_text())
        assert len(one) == 24, (c, stem, len(one))
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    masks = fuse_masks_fast(_decode_all(payload), IOU, VOTES)
    return stem, [encode_rle(m) for m in masks]
def main():
    out, td = Path(sys.argv[1]), Path(sys.argv[2]); caches = [Path(c) for c in sys.argv[3:6]]
    names = ("s20260910", "s20260911", "s20260912")
    images = sorted(td.glob("*.jpeg")); assert len(images) == 180, len(images)
    global CACHES, NAMES; CACHES, NAMES = caches, names
    import os
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    stems = [p.stem for p in images]; res = {}; t0 = time.time(); nproc = int(os.environ.get("NPROC", "3"))
    def _log(i):
        if i == 1 or i % 25 == 0 or i == len(stems): print(f"test_fused {i}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    try:   # per-image work is independent; a killed worker (e.g. OOM) raises BrokenProcessPool instead of hanging
        with ProcessPoolExecutor(nproc) as ex:
            for i, (stem, rles) in enumerate(ex.map(_one, stems), 1):
                res[stem] = rles; _log(i)
    except BrokenProcessPool as e:
        print(f"FUSE_POOL_BROKEN after {len(res)} images ({e!r}); finishing the rest serially", flush=True)
        for stem in stems:
            if stem not in res:
                res[stem] = _one(stem)[1]; _log(len(res))
    assert len(res) == len(stems)
    frames = [pred_to_df(stem, res[stem]) for stem in stems if res[stem]]   # sorted image order, as in the original
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    df.to_csv(out, index=False); print("FUSED_ROWS", len(df), flush=True)
if __name__ == "__main__":
    main()
