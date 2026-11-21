#!/usr/bin/env python3
"""P6 (pre-registered 8:35 PM Nov 10) U-Net scoring on OOF folds 1-4 (only stems that have a map).
usage: unet_score.py OUT.json MAPDIR ENC     MAPDIR/<stem>.png = uint8 8-view TTA map of the fold model that never saw stem.
names: base (YOLO fold models, exact 0.438046 decode), unet (frozen single-family keep rule), blend (B1 add-disjoint), oracle (per-image better of base/unet).
Also writes GT coverage counts (YOLO only / U-Net only / both / neither)."""
import sys, os, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
import pool_score as ps
RULES = {"tu-tf_efficientnetv2_s": (0.35, 0.995, 0, 0.85), "resnet34": (0.15, 0.97, 250, 0.70)}
MAPDIR = ENC = None
def init2(md, enc):
    global MAPDIR, ENC; MAPDIR, ENC = Path(md), enc; ps.init({})
def unet_preds(stem):
    t_lo, t_hi, amin, mmin = RULES[ENC]; u = cv2.imread(str(MAPDIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    assert u is not None and u.shape == (2048, 2048), stem
    n, lab, st, _ = cv2.connectedComponentsWithStats((u > t_lo * 255).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sub = lab[y:y + h, x:x + w] == i; vals = u[y:y + h, x:x + w][sub]
        if vals.max() > t_hi * 255 and a >= amin and vals.mean() >= mmin * 255:
            m = np.zeros((2048, 2048), bool); m[y:y + h, x:x + w] = sub; out.append((m, int(a)))
    return out
def resolve(preds, claimed=None):
    claimed = np.zeros((2048, 2048), bool) if claimed is None else claimed.copy(); res = []
    for m, a in sorted(preds, key=lambda t: -t[1]):
        ex = m & ~claimed; ea = int(ex.sum())
        if ea < 32: continue
        claimed |= ex; res.append((ex, ea))
    return res, claimed
def score_hits(preds, gts):
    out = np.zeros(4); hits = {}
    for iid, gl in gts.items():
        hp = np.zeros(len(preds), bool); hg = np.zeros(len(gl), bool)
        for pi, (m, a) in enumerate(preds):
            for gi, (g, ga) in enumerate(gl):
                inter = int(np.logical_and(m, g).sum())
                if inter and inter / (a + ga - inter) > 0.5:
                    out[0] += inter / (a + ga - inter); out[1] += 1; hp[pi] = True; hg[gi] = True
        out[2] += int((~hp).sum()); out[3] += int((~hg).sum()); hits[iid] = hg
    return out, hits
def one(stem):
    gts = {iid: [] for iid in ps.B[stem + ".jpeg"]}
    for iid in gts:
        for f in ps.V[iid].filaments:
            g = ps.decode_rle(ps.polygon_to_coco_rle(f.polygon)).astype(bool); gts[iid].append((g, int(g.sum())))
    Y = ps.decode(stem, [Path.home() / "filament_run/oof_all4_out/tta/tta"])  # already resolved (champion decode)
    U, _ = resolve(unet_preds(stem))
    yu = np.zeros((2048, 2048), bool)
    for m, a in Y: yu |= m
    add = [(m, a) for m, a in U if (m & yu).sum() < 0.2 * a]
    addr, _ = resolve(add, yu); Bl = Y + addr
    sy, hy = score_hits(Y, gts); su, hu = score_hits(U, gts); sb, _ = score_hits(Bl, gts)
    pq = lambda x: x[0] / max(1e-9, x[1] + 0.5 * x[2] + 0.5 * x[3])
    so = su if pq(su) > pq(sy) else sy
    cov = [0, 0, 0, 0]  # yolo_only, unet_only, both, neither
    for iid in gts:
        for a_, b_ in zip(hy[iid], hu[iid]): cov[0 if (a_ and not b_) else 1 if (b_ and not a_) else 2 if a_ else 3] += 1
    return stem, {"base": sy.tolist(), "unet": su.tolist(), "blend": sb.tolist(), "oracle": so.tolist(), "cov": cov, "n_unet": len(U), "n_add": len(addr)}
if __name__ == "__main__":
    out, md, enc = sys.argv[1], sys.argv[2], sys.argv[3]
    stems = sorted(s for s, v in ps.FOLDS.items() if v in (1, 2, 3, 4) and (Path(md) / f"{s}.png").exists())
    with Pool(int(os.environ.get("NPROC", "4")), initializer=init2, initargs=(md, enc)) as p:
        res = dict(p.imap_unordered(one, stems))
    json.dump({"split": "oof_unet", "stems": stems, "per_image": res}, open(out, "w"))
    for name in ("base", "unet", "blend", "oracle"):
        a = np.sum([res[s][name] for s in stems], axis=0)
        print(name, "PQ", round(a[0] / (a[1] + 0.5 * a[2] + 0.5 * a[3]), 6), "TP/FP/FN", a[1:].astype(int).tolist(), flush=True)
    print("coverage yolo_only/unet_only/both/neither", np.sum([res[s]["cov"] for s in stems], 0).tolist(), "stems", len(stems), flush=True)
