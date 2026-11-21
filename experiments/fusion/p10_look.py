#!/usr/bin/env python3
"""P10 SINGLE fold-0 look (frozen rule from all folds 1-4). Refuses without P10_LOOK=1 or if the result exists."""
import sys, os, json, datetime as dt
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
R = Path.home() / "filament_run/p10"; sys.path.insert(0, str(R))
import p10
RULE = dict(a=0.4, t_lo=0.3, t_hi=0.6, m=0.6, amin=200)
OUT = R / "p10_fold0_look.json"
def one(stem):
    gts = []
    for iid in p10.B[stem + ".jpeg"]:
        gl = []
        for f in p10.V[iid].filaments:
            g = p10.decode_rle(p10.polygon_to_coco_rle(f.polygon)).astype(bool); gl.append((g, int(g.sum())))
        gts.append(gl)
    Y = np.zeros((2048, 2048), np.float32); n = 0
    for c in p10.caches(stem):
        d = json.loads((Path(c) / f"{stem}.json").read_text()); assert len(d) == 24
        for k in sorted(d):
            for r in d[k]: Y += p10.decode_rle(r)
            n += 1
    assert n == 72; Y /= n; U = p10.umap(stem); F = RULE["a"] * Y + (1 - RULE["a"]) * U
    nc, lab, st, _ = cv2.connectedComponentsWithStats((F > RULE["t_lo"]).astype(np.uint8), connectivity=8); preds = []
    for i in range(1, nc):
        x, y, w, h, ar = st[i]
        if ar < 32 or ar < RULE["amin"]: continue
        sub = lab[y:y + h, x:x + w] == i; vals = F[y:y + h, x:x + w][sub]
        if vals.max() > RULE["t_hi"] and vals.mean() >= RULE["m"]:
            m = np.zeros((2048, 2048), bool); m[y:y + h, x:x + w] = sub; preds.append((m, int(ar)))
    import pool_score as ps
    gd = {an: gl for an, gl in enumerate(gts)}
    new = ps.score(stem, preds, gd); base = ps.score(stem, ps.decode(stem, p10.caches(stem)), gd)
    return stem, {"p10": new.tolist(), "champ": base.tolist(), "n_pred": len(preds)}
if __name__ == "__main__":
    if os.environ.get("P10_LOOK") != "1" or OUT.exists(): raise SystemExit("refused (needs P10_LOOK=1 and no previous look)")
    stems = sorted(s for s, v in p10.FOLDS.items() if v == 0); assert len(stems) == 142
    with Pool(int(os.environ.get("NPROC", "6")), initializer=p10.init, initargs=("fold0",)) as p:
        res = dict(p.imap_unordered(one, stems))
    pq = lambda x: x[..., 0] / (x[..., 1] + .5 * x[..., 2] + .5 * x[..., 3])
    a = np.array([res[s]["p10"] for s in stems]); b = np.array([res[s]["champ"] for s in stems])
    rng = np.random.default_rng(20261110); N = len(stems)
    ib = np.array([pq(a[i].sum(0)) - pq(b[i].sum(0)) for i in [rng.integers(0, N, N) for _ in range(2000)]])
    dd = [dt.datetime.strptime(s[:8], "%Y%m%d").date() for s in stems]; order = np.argsort(dd); cl = np.zeros(N, int); c = 0
    for i in range(1, N):
        if (dd[order[i]] - dd[order[i - 1]]).days > 1: c += 1
        cl[order[i]] = c
    G = [np.nonzero(cl == g)[0] for g in range(c + 1)]; rng = np.random.default_rng(20261110)
    db = np.array([pq(a[i].sum(0)) - pq(b[i].sum(0)) for i in [np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))]) for _ in range(2000)]])
    rep = {"rule": RULE, "fold0_p10": float(pq(a.sum(0))), "fold0_champ": float(pq(b.sum(0))), "delta": float(pq(a.sum(0)) - pq(b.sum(0))),
           "TPFPFN_p10": a.sum(0)[1:].astype(int).tolist(), "TPFPFN_champ": b.sum(0)[1:].astype(int).tolist(),
           "img_ci95": [float(np.percentile(ib, 2.5)), float(np.percentile(ib, 97.5))], "img_p_le_0": float((ib <= 0).mean()),
           "date_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
           "slot2_keep": bool(pq(a.sum(0)) > 0.465553), "slot1_promote": bool(pq(a.sum(0)) >= 0.475553),
           "champ_reproduced": bool(abs(pq(b.sum(0)) - 0.465553) < 1e-5), "per_image": res}
    json.dump(rep, open(OUT, "w"))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_image"}, indent=1))
