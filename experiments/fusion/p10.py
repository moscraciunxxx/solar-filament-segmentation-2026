#!/usr/bin/env python3
"""P10 dense pixel fusion (pre-registered Nov 12 2:15 PM PT). usage: p10.py oof|fold0 OUT.npz"""
import sys, os, json, csv, itertools
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
R = Path("/workspace/p10"); sys.path.insert(0, str(R / "src_full")); sys.path.insert(0, str(R / "keep"))
from filament.rle import decode_rle
from filament.coco_io import load_coco, polygon_to_coco_rle
FOLDS = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(R / "oof_kit/folds_ours.csv"))}
A = [0, .25, .4, .5, .6, .75, 1.0]; TLO = [.3, .35, .4, .5]; THI = [.6, .7, .8, .9, .95, .995]; MM = [.5, .6, .7, .8, .85]; AMIN = [0, 96, 200]
RULES = list(itertools.product(range(len(A)), range(len(TLO)), THI, MM, AMIN))
V = B = None; MODE = None
def init(mode):
    global V, B, MODE
    V, B = load_coco(R / "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"); MODE = mode
def caches(stem):
    if MODE == "oof": return [R / "oof_all4_out/tta/tta", R / "ls/oofLS/tta", R / "d4b/oofD4/tta"]
    return [R / "p10pack/champf0/s10", R / "p10pack/champf0/s11", R / "p10pack/champf0/s12"]
def umap(stem):
    d = R / ("unet_maps/held" if MODE == "oof" else "unet_maps/f0/maps_f0")
    u = cv2.imread(str(d / f"{stem}.png"), cv2.IMREAD_GRAYSCALE); assert u is not None and u.shape == (2048, 2048), stem
    return u.astype(np.float32) / 255.0
def bbox(m):
    ys, xs = np.nonzero(m); return (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
def one(stem):
    gts = []  # list of annotators: list of (mask, area, bbox)
    for iid in B[stem + ".jpeg"]:
        gl = []
        for f in V[iid].filaments:
            g = decode_rle(polygon_to_coco_rle(f.polygon)).astype(bool)
            if g.any(): gl.append((g, int(g.sum()), bbox(g)))
            else: gl.append((g, 0, (0, 0, 0, 0)))
        gts.append(gl)
    Y = np.zeros((2048, 2048), np.float32); n = 0
    for c in caches(stem):
        d = json.loads((Path(c) / f"{stem}.json").read_text()); assert len(d) == 24
        for k in sorted(d):
            for r in d[k]: Y += decode_rle(r)
            n += 1
    Y /= n; U = umap(stem)
    res = np.zeros((len(RULES), 4))
    for ai, a in enumerate(A):
        F = a * Y + (1 - a) * U
        for ti, t in enumerate(TLO):
            nc, lab, st, _ = cv2.connectedComponentsWithStats((F > t).astype(np.uint8), connectivity=8)
            comps = []  # (max, mean, area, pairs[(annot, gi, iou)])
            for i in range(1, nc):
                x, y, w, h, ar = st[i]
                if ar < 32: continue
                sub = lab[y:y + h, x:x + w] == i; vals = F[y:y + h, x:x + w][sub]; mx = float(vals.max())
                if mx <= min(THI): continue
                pairs = []
                for an, gl in enumerate(gts):
                    for gi, (g, ga, (y0, y1, x0, x1)) in enumerate(gl):
                        if ga == 0 or y1 <= y or y0 >= y + h or x1 <= x or x0 >= x + w: continue
                        inter = int(np.logical_and(sub, g[y:y + h, x:x + w]).sum())
                        if inter and inter / (ar + ga - inter) > 0.5: pairs.append((an, gi, inter / (ar + ga - inter)))
                comps.append((mx, float(vals.mean()), int(ar), pairs))
            ngt = [len(gl) for gl in gts]
            for ri, (rai, rti, th, mm, am) in enumerate(RULES):
                if rai != ai or rti != ti: continue
                kept = [c for c in comps if c[0] > th and c[1] >= mm and c[2] >= am]
                s = tp = fp = fn = 0.0
                for an in range(len(gts)):
                    hitg = set(); 
                    for c in kept:
                        pp = [p for p in c[3] if p[0] == an]
                        if pp:
                            for p in pp: s += p[2]; tp += 1; hitg.add(p[1])
                        else: fp += 1
                    fn += ngt[an] - len(hitg)
                res[ri] = (s, tp, fp, fn)
    # baseline: pseudo-champion / champion decode (pool_score logic) on the same caches
    sys.path.insert(0, str(R)); import pool_score as ps
    preds = ps.decode(stem, caches(stem)); gd = {an: [(g, ga) for g, ga, _ in gl] for an, gl in enumerate(gts)}
    base = ps.score(stem, preds, gd)
    return stem, res, base
if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    stems = sorted(s for s, v in FOLDS.items() if (v == 0) == (mode == "fold0"))
    if mode == "fold0" and os.environ.get("P10_LOOK") != "1": raise SystemExit("fold-0 refused without P10_LOOK=1")
    if os.environ.get("LIMIT"): stems = stems[: int(os.environ["LIMIT"])]
    with Pool(int(os.environ.get("NPROC", "5")), initializer=init, initargs=(mode,)) as p:
        rs = []
        for r in p.imap(one, stems, chunksize=1):
            rs.append(r)
            if len(rs) % 25 == 0: print('progress', len(rs), flush=True)
    np.savez(out, stems=np.array([r[0] for r in rs]), res=np.stack([r[1] for r in rs]), base=np.stack([r[2] for r in rs]))
    print("DONE", len(rs), flush=True)
