#!/usr/bin/env python3
"""P11 (pre-registered Nov 12 ~3:10 PM PT): U-Net instances as weighted voters inside the champion vote. Crop-based exact port of
pool_score.decode/score. usage: p11.py oof|fold0 OUT.json"""
import sys, os, json, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
R = Path(os.environ.get("P11_ROOT", "/workspace/p10")); sys.path.insert(0, str(R / "src_full"))
from filament.rle import decode_rle
from filament.coco_io import load_coco, polygon_to_coco_rle
FOLDS = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(R / "oof_kit/folds_ours.csv"))}
WS = [int(x) for x in os.environ.get("P11_WS", "0,6,12,18,24,36").split(",")]
V = B = None; MODE = None
def init(mode):
    global V, B, MODE
    V, B = load_coco(R / "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"); MODE = mode
def caches():
    if MODE == "oof": return [R / "oof_all4_out/tta/tta", R / "ls/oofLS/tta", R / "d4b/oofD4/tta"]
    return [R / "p10pack/champf0/s10", R / "p10pack/champf0/s11", R / "p10pack/champf0/s12"]
def crop(m):
    ys, xs = np.nonzero(m)
    if not len(xs): return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (int(y0), int(y1), int(x0), int(x1), m[y0:y1, x0:x1].astype(bool), int(len(xs)))
def unet_inst(stem):
    d = R / ("unet_maps/held" if MODE == "oof" else "unet_maps/f0/maps_f0")
    u = cv2.imread(str(d / f"{stem}.png"), cv2.IMREAD_GRAYSCALE); assert u is not None and u.shape == (2048, 2048)
    n, lab, st, _ = cv2.connectedComponentsWithStats((u > 0.35 * 255).astype(np.uint8), connectivity=8); out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sub = lab[y:y + h, x:x + w] == i; vals = u[y:y + h, x:x + w][sub]
        if vals.max() > 0.995 * 255 and vals.mean() >= 0.85 * 255 and a >= 32:
            out.append((int(y), int(y + h), int(x), int(x + w), sub, int(a)))
    return out
def inter(a, b):
    y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    if y0 >= y1 or x0 >= x1: return 0
    return int(np.logical_and(a[4][y0 - a[0]:y1 - a[0], x0 - a[2]:x1 - a[2]], b[4][y0 - b[0]:y1 - b[0], x0 - b[2]:x1 - b[2]]).sum())
def groups(ms):
    order = sorted(range(len(ms)), key=lambda i: ms[i][5], reverse=True); used = [False] * len(ms); gs = []
    for i in order:
        if used[i]: continue
        g = [i]; used[i] = True
        for j in order:
            if used[j]: continue
            it = inter(ms[i], ms[j])
            if it and it / (ms[i][5] + ms[j][5] - it) >= 0.5: g.append(j); used[j] = True
        gs.append(g)
    return gs
def decode(ms, wts, gs):
    preds = []
    for g in gs:
        n = sum(wts[k] for k in g)
        if n < 24 or n == 0: continue
        y0 = min(ms[k][0] for k in g); y1 = max(ms[k][1] for k in g); x0 = min(ms[k][2] for k in g); x1 = max(ms[k][3] for k in g)
        vote = np.zeros((y1 - y0, x1 - x0), np.int32)
        for k in g:
            if wts[k]: m = ms[k]; vote[m[0] - y0:m[1] - y0, m[2] - x0:m[3] - x0] += wts[k] * m[4]
        mm = vote >= (n + 1) // 2; a = int(mm.sum())
        if 96 <= a <= 80000: preds.append(((y0, x0), mm, a))
    preds.sort(key=lambda t: -t[2]); claimed = np.zeros((2048, 2048), bool); res = []
    for (y0, x0), mm, a in preds:
        h, w = mm.shape; ex = mm & ~claimed[y0:y0 + h, x0:x0 + w]; ea = int(ex.sum())
        if ea < 32: continue
        claimed[y0:y0 + h, x0:x0 + w] |= ex; c = crop(ex); res.append((c[0] + y0, c[1] + y0, c[2] + x0, c[3] + x0, c[4], c[5]))
    return res
def score(preds, gts):
    out = np.zeros(4)
    for gl in gts:
        hp = np.zeros(len(preds), bool); hg = np.zeros(len(gl), bool)
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gl):
                if g is None: continue
                it = inter(p, g)
                if it:
                    v = it / (p[5] + g[5] - it)
                    if v > 0.5: out[0] += v; out[1] += 1; hp[pi] = True; hg[gi] = True
        out[2] += int((~hp).sum()); out[3] += int((~hg).sum())
    return out
def one(stem):
    gts = [[crop(decode_rle(polygon_to_coco_rle(f.polygon))) for f in V[iid].filaments] for iid in B[stem + ".jpeg"]]
    ys = []
    for c in caches():
        d = json.loads((Path(c) / f"{stem}.json").read_text()); assert len(d) == 24
        for k in sorted(d):
            for r in d[k]:
                cm = crop(decode_rle(r))
                ys.append(cm)
    ys = [m for m in ys if m is not None]  # pool_score skips empty boxes in groups_from too
    us = unet_inst(stem); out = {}
    g0 = groups(ys); out["w0"] = score(decode(ys, [1] * len(ys), g0), gts).tolist()
    ms = ys + us; g1 = groups(ms)
    for w in WS[1:]:
        out[f"w{w}"] = score(decode(ms, [1] * len(ys) + [w] * len(us), g1), gts).tolist()
    out["n_unet"] = len(us)
    return stem, out
if __name__ == "__main__":
    mode, outp = sys.argv[1], sys.argv[2]
    stems = sorted(s for s, v in FOLDS.items() if (v == 0) == (mode == "fold0"))
    if mode == "fold0" and os.environ.get("P11_LOOK") != "1": raise SystemExit("fold-0 refused without P11_LOOK=1")
    if os.environ.get("LIMIT"): stems = stems[: int(os.environ["LIMIT"])]
    res = {}
    with Pool(int(os.environ.get("NPROC", "4")), initializer=init, initargs=(mode,)) as p:
        for s, r in p.imap_unordered(one, stems):
            res[s] = r
            if len(res) % 25 == 0: print("progress", len(res), flush=True)
    json.dump({"split": mode, "stems": stems, "per_image": res}, open(outp, "w"))
    for k in [f"w{w}" for w in WS]:
        a = np.sum([res[s][k] for s in stems], 0); print(k, "PQ", round(a[0] / (a[1] + .5 * a[2] + .5 * a[3]), 6), a[1:].astype(int).tolist(), flush=True)
