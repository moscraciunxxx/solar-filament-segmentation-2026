import sys, json, csv, datetime as dt, numpy as np
sys.path.insert(0, "/workspace/p10"); from p10 import RULES, A, TLO
d = np.load(sys.argv[1]); stems = list(d["stems"]); R = d["res"]; Bs = d["base"]
F = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open("/workspace/p10/oof_kit/folds_ours.csv"))}
fk = np.array([F[s] for s in stems])
pq = lambda x: x[..., 0] / np.maximum(1e-9, x[..., 1] + .5 * x[..., 2] + .5 * x[..., 3])
def rname(i): a, t, th, m, am = RULES[i]; return dict(a=A[a], t_lo=TLO[t], t_hi=th, m=m, amin=am)
allpq = pq(R.sum(0)); best = int(np.argmax(allpq))
print("baseline (72-voter pseudo-champion) OOF", round(float(pq(Bs.sum(0))), 6), Bs.sum(0)[1:].astype(int).tolist())
# sanity: a=0, t_lo .35, t_hi .995, m .85, amin 0 == U-Net P6
san = [i for i, r in enumerate(RULES) if A[r[0]] == 0 and TLO[r[1]] == .35 and r[2] == .995 and r[3] == .85 and r[4] == 0][0]
print("sanity U-Net P6 rule", round(float(allpq[san]), 6), R[:, san].sum(0)[1:].astype(int).tolist())
print("in-sample best", rname(best), round(float(allpq[best]), 6), R[:, best].sum(0)[1:].astype(int).tolist())
for a_i, a in enumerate(A):
    idx = [i for i, r in enumerate(RULES) if r[0] == a_i]; j = idx[int(np.argmax(allpq[idx]))]
    print("  best at a=%.2f" % a, rname(j), round(float(allpq[j]), 6))
nest = np.zeros((len(stems), 4)); chosen = {}
for k in (1, 2, 3, 4):
    tr = fk != k; j = int(np.argmax(pq(R[tr].sum(0)))); chosen[k] = rname(j); nest[fk == k] = R[fk == k, j]
print("nested choices", json.dumps(chosen))
res = {"delta": float(pq(nest.sum(0)) - pq(Bs.sum(0))), "nested": float(pq(nest.sum(0))), "base": float(pq(Bs.sum(0))), "TPFPFN": nest.sum(0)[1:].astype(int).tolist()}
res["folds"] = {k: float(pq(nest[fk == k].sum(0)) - pq(Bs[fk == k].sum(0))) for k in (1, 2, 3, 4)}
rng = np.random.default_rng(20261110); n = len(stems)
ib = [pq(nest[i].sum(0)) - pq(Bs[i].sum(0)) for i in [rng.integers(0, n, n) for _ in range(2000)]]
dd = [dt.datetime.strptime(s[:8], "%Y%m%d").date() for s in stems]; order = np.argsort(dd); cl = np.zeros(n, int); c = 0
for i in range(1, n):
    if (dd[order[i]] - dd[order[i - 1]]).days > 1: c += 1
    cl[order[i]] = c
groups = [np.nonzero(cl == g)[0] for g in range(c + 1)]; rng = np.random.default_rng(20261110)
db = []
for _ in range(2000):
    i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]); db.append(pq(nest[i].sum(0)) - pq(Bs[i].sum(0)))
res["img_ci95"] = [float(np.percentile(ib, 2.5)), float(np.percentile(ib, 97.5))]; res["date_ci95"] = [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]
res["PASS"] = bool(res["delta"] >= 0.006 and res["img_ci95"][0] > 0 and res["date_ci95"][0] > 0 and sum(v > 0 for v in res["folds"].values()) >= 3)
res["frozen_rule_all_folds"] = rname(best)
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1].replace(".npz", "_gate.json"), "w"), indent=1)
