import json, csv, datetime as dt, numpy as np
r = json.load(open("/workspace/p10/p11_oof.json")); S = r["stems"]; P = r["per_image"]
F = {x["stem"]: int(x["fold"]) for x in csv.DictReader(open("/workspace/p10/oof_kit/folds_ours.csv"))}
WS = [6, 12, 18, 24, 36]; pq = lambda x: x[..., 0] / (x[..., 1] + .5 * x[..., 2] + .5 * x[..., 3])
M = {w: np.array([P[s][f"w{w}"] for s in S]) for w in [0] + WS}; fk = np.array([F[s] for s in S]); base = M[0]
for w in [0] + WS: print("w", w, round(float(pq(M[w].sum(0))), 6), M[w].sum(0)[1:].astype(int).tolist())
nest = np.zeros_like(base); ch = {}
for k in (1, 2, 3, 4):
    w = max(WS, key=lambda w: pq(M[w][fk != k].sum(0))); ch[k] = w; nest[fk == k] = M[w][fk == k]
rng = np.random.default_rng(20261110); n = len(S)
ib = [pq(nest[i].sum(0)) - pq(base[i].sum(0)) for i in [rng.integers(0, n, n) for _ in range(2000)]]
dd = [dt.datetime.strptime(s[:8], "%Y%m%d").date() for s in S]; o = np.argsort(dd); cl = np.zeros(n, int); c = 0
for i in range(1, n):
    if (dd[o[i]] - dd[o[i - 1]]).days > 1: c += 1
    cl[o[i]] = c
G = [np.nonzero(cl == g)[0] for g in range(c + 1)]; rng = np.random.default_rng(20261110)
db = [pq(nest[i].sum(0)) - pq(base[i].sum(0)) for i in [np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))]) for _ in range(2000)]]
res = {"chosen": ch, "nested": float(pq(nest.sum(0))), "base": float(pq(base.sum(0))), "delta": float(pq(nest.sum(0)) - pq(base.sum(0))),
       "folds": {k: float(pq(nest[fk == k].sum(0)) - pq(base[fk == k].sum(0))) for k in (1, 2, 3, 4)},
       "img_ci95": [float(np.percentile(ib, 2.5)), float(np.percentile(ib, 97.5))], "date_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}
res["PASS"] = bool(res["delta"] >= .006 and res["img_ci95"][0] > 0 and res["date_ci95"][0] > 0 and sum(v > 0 for v in res["folds"].values()) >= 3)
print(json.dumps(res, indent=1)); json.dump(res, open("/workspace/p10/p11_gate.json", "w"), indent=1)
