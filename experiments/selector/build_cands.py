#!/usr/bin/env python3
"""Candidate table builder (CPU only). For one image: cluster all TTA view masks exactly like the champion
fuse_masks_fast (IoU 0.5, area-desc greedy), keep every group with vote fraction >= 4/24, add U-Net-only components,
compute features. Writes cands/<stem>.pkl. No labels here."""
import sys, json, pickle, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path.home() / "filament_run/src_full"))
from filament.rle import decode_rle, encode_rle
from filament.infer import yolo_masks_to_instances
S = 2048
FA = Path.home() / ".cache/filament-segmentation-2026/artifacts"
UNET = Path.home() / "filament_run/probe_out/unet_oof_prob"
IMGS = [Path.home() / "filament_run/kaggle_d4_dataset/yolo_seg_erode_gt/images" / s for s in ("train", "val")]
OUT = Path(os.environ.get("CANDS", str(Path.home() / "filament_run/keep/cands"))); OUT.mkdir(exist_ok=True)
MIN_FRAC = 4 / 24
EXT = os.environ.get("EXT") == "1"

def box(m):
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None

def groups_from(masks):
    boxes = [box(m) for m in masks]; areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks); groups = []
    for i in order:
        if used[i] or boxes[i] is None: continue
        g = [i]; used[i] = True; ax0, ay0, ax1, ay1 = boxes[i]; ai = areas[i]; mi = masks[i]
        for j in order:
            if used[j] or boxes[j] is None: continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0: continue
            ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1) + 1, min(ay1, by1) + 1
            inter = int(np.logical_and(mi[iy0:iy1, ix0:ix1], masks[j][iy0:iy1, ix0:ix1]).sum())
            union = ai + areas[j] - inter
            if union and inter / union >= 0.5: g.append(j); used[j] = True
        groups.append(g)
    return groups

def shape_feats(m, img, disk_c, disk_r, prob):
    ys, xs = np.nonzero(m); a = len(xs)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea); (w, h) = cv2.minAreaRect(c)[1]
    elong = max(w, h) / max(min(w, h), 1.0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    pad = 12; X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(S, x1 + pad), min(S, y1 + pad)
    sub = m[Y0:Y1, X0:X1].astype(np.uint8); ring = (cv2.dilate(sub, np.ones((15, 15), np.uint8)) > 0) & (sub == 0)
    im = img[Y0:Y1, X0:X1].astype(np.float32); inside = im[sub > 0].mean(); outside = im[ring].mean() if ring.any() else inside
    cy, cx = ys.mean(), xs.mean(); rr = np.hypot(cx - disk_c[0], cy - disk_c[1]) / max(disk_r, 1)
    p = prob[Y0:Y1, X0:X1][sub > 0]
    return {"area": a, "elong": elong, "contrast": float(outside - inside), "rel_contrast": float((outside - inside) / max(outside, 1)),
            "inside_int": float(inside), "radius": float(rr), "u_mean": float(p.mean()), "u_peak": float(p.max()),
            "u_p90": float(np.percentile(p, 90)), "u_frac50": float((p > 0.5).mean()), "fill": a / max(w * h, 1.0)}

def iou(a, b):
    i = np.logical_and(a, b).sum(); return i / (a.sum() + b.sum() - i) if i else 0.0

def build(stem, caches):
    payload = []
    for c in caches:
        d = json.loads((c / f"{stem}.json").read_text())
        for k in sorted(d): payload.extend(d[k])
    nvot = 24 * len(caches)
    masks = [decode_rle(r) for r in payload]
    img = None
    for d in IMGS:
        p = d / f"{stem}.jpeg"
        if p.exists(): img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    disk = img > 20; disk_r = float(np.sqrt(disk.sum() / np.pi)); yy, xx = np.nonzero(disk); disk_c = (xx.mean(), yy.mean())
    prob = cv2.imread(str(UNET / f"{stem}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    cands = []
    for g in groups_from(masks):
        votes = len(g)
        if votes < max(1, int(np.ceil(MIN_FRAC * nvot - 1e-9))): continue
        vote = np.zeros((S, S), np.uint8)
        for k in g: vote += masks[k]
        tmid = max(1, (votes + 1) // 2)
        merged = (vote >= tmid).astype(np.uint8)
        if int(merged.sum()) < 96: continue
        union = (vote > 0).sum(); unanim = (vote >= votes).sum(); amid = int(merged.sum())
        gid = len(set(cf.get("gid", -1) for _, cf in cands))
        exts = [(0, tmid)]
        if EXT:  # BarrelAged-style multi-extent pool: looser and tighter pixel-vote thresholds of the same cluster
            tlo = max(1, int(round(0.25 * votes))); thi = min(votes, max(tmid + 1, int(round(0.8 * votes))))
            if tlo < tmid: exts.append((-1, tlo))
            if thi > tmid: exts.append((1, thi))
        for ext, t in exts:
            mm = merged if ext == 0 else (vote >= t).astype(np.uint8)
            if int(mm.sum()) < 96: continue
            f = shape_feats(mm, img, disk_c, disk_r, prob)
            f.update(src=0, votes=votes, vfrac=votes / nvot, nvoters=nvot, spread=union / mm.sum(), core=unanim / mm.sum(),
                     baseline=int(votes >= nvot / 3 and ext == 0), gid=gid, ext=ext, ext_ratio=int(mm.sum()) / amid, pix_thr=t / votes)
            cands.append((mm, f))
    # U-Net components not covered by a YOLO candidate
    n, lab = cv2.connectedComponents((prob > 0.35).astype(np.uint8), connectivity=8)
    for j in range(1, n):
        m = (lab == j).astype(np.uint8)
        if m.sum() < 96: continue
        best, bv = 0.0, 0.0
        for cm, cf in cands:
            if cf["src"] != 0: continue
            v = iou(m, cm)
            if v > best: best, bv = v, cf["vfrac"]
        for cm, cf in cands:
            if cf["src"] == 0 and cf.get("u_iou", 0) < iou(m, cm): cf["u_iou"] = iou(m, cm)
        if best >= 0.3: continue
        f = shape_feats(m, img, disk_c, disk_r, prob)
        f.update(src=1, votes=0, vfrac=bv, nvoters=nvot, spread=1.0, core=1.0, baseline=0, u_iou=best,
                 gid=10000 + j, ext=0, ext_ratio=1.0, pix_thr=0.35)
        cands.append((m, f))
    for cm, cf in cands: cf.setdefault("u_iou", 0.0)
    rec = {"stem": stem, "nvoters": nvot, "disk_r": disk_r,
           "cands": [(encode_rle(m), f) for m, f in cands]}
    with open(OUT / f"{stem}.pkl", "wb") as h: pickle.dump(rec, h)
    return len(cands)

if __name__ == "__main__":
    import csv, time
    from multiprocessing import Pool
    which = sys.argv[1]  # "fold0" or "k" for OOF fold k with cache dir argv[2]
    folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(Path.home() / "filament_run/oof_kit/folds_ours.csv"))}
    if which == "fold0":
        stems = sorted(s for s, v in folds.items() if v == 0)
        caches = [FA / "tta_parent_2048", FA / "tta_s20260911", FA / "tta_s20260912"]
    elif which.startswith("insample"):
        k = int(which[-1]); stems = sorted(s for s, v in folds.items() if v == k)
        caches = [FA / "tta_parent_2048", FA / "tta_s20260911", FA / "tta_s20260912"]
    else:
        k = int(which); stems = sorted(s for s, v in folds.items() if v == k); caches = [Path(sys.argv[2])]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if lim: stems = stems[:lim]
    todo = [s for s in stems if not (OUT / f"{s}.pkl").exists()]
    t = time.time()
    with Pool(int(os.environ.get("NPROC", "6"))) as p:
        for i, n in enumerate(p.starmap(build, [(s, caches) for s in todo])): pass
    print(which, "built", len(todo), "of", len(stems), "sec", round(time.time() - t, 1), flush=True)
