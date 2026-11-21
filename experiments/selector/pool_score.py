#!/usr/bin/env python3
"""Pre-registered pooled decode + pycocotools PQ scorer (see /workspace/filament_plan_exec.md, PRE-REGISTRATION).
Decode = champion fusion generalised to N voters (zero-shift path of keep/shift_tune.py, which reproduced score_pq exactly):
groups_from (IoU 0.5, largest-first) -> keep n >= N/3 -> pixel vote >= (n+1)//2 -> area 96..80000 -> area-desc exclusive, drop < 32.
usage: pool_score.py OUT.json SPLIT(fold0|oof) CONFIG_JSON   (CONFIG_JSON: {name: [cache_dir_or_'OOF', ...]})"""
import sys, os, json, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path.home() / "filament_run/src_full")); sys.path.insert(0, str(Path.home() / "filament_run/keep"))
from filament.rle import decode_rle
from filament.coco_io import load_coco, polygon_to_coco_rle
FOLDS = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(Path.home() / "filament_run/oof_kit/folds_ours.csv"))}
V = B = None; CFG = None
def init(cfg):
    global V, B, CFG
    V, B = load_coco(Path.home() / "filament_run/data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"); CFG = cfg
def decode(stem, caches):
    from build_cands import groups_from
    payload = []
    for c in caches:
        d = json.loads((Path(c) / f"{stem}.json").read_text())
        assert len(d) == 24, (c, stem, len(d))
        for k in sorted(d): payload.extend(d[k])
    masks = [decode_rle(r).astype(np.uint8) for r in payload]; N = 24 * len(caches)
    preds = []
    for g in groups_from(masks):
        n = len(g)
        if n < N / 3: continue
        vote = np.zeros((2048, 2048), np.uint16)
        for k in g: vote += masks[k]
        m = vote >= (n + 1) // 2; a = int(m.sum())
        if 96 <= a <= 80000: preds.append((m, a))
    preds.sort(key=lambda t: -t[1]); claimed = np.zeros((2048, 2048), bool); res = []
    for m, a in preds:
        ex = m & ~claimed; ea = int(ex.sum())
        if ea < 32: continue
        claimed |= ex; res.append((ex, ea))
    return res
def score(stem, preds, gts):
    out = np.zeros(4)  # sumIoU, TP, FP, FN
    for iid, gl in gts.items():
        hitp = np.zeros(len(preds), bool); hitg = np.zeros(len(gl), bool)
        for pi, (m, a) in enumerate(preds):
            for gi, (g, ga) in enumerate(gl):
                inter = int(np.logical_and(m, g).sum())
                if not inter: continue
                v = inter / (a + ga - inter)
                if v > 0.5: out[0] += v; out[1] += 1; hitp[pi] = True; hitg[gi] = True
        out[2] += int((~hitp).sum()); out[3] += int((~hitg).sum())
    return out
def one(stem):
    gts = {}
    for iid in B[stem + ".jpeg"]:
        gts[iid] = []
        for f in V[iid].filaments:
            g = decode_rle(polygon_to_coco_rle(f.polygon)).astype(bool); gts[iid].append((g, int(g.sum())))
    r = {}
    for name, caches in CFG.items():
        cs = [Path.home() / "filament_run/oof_all4_out/tta/tta" if c == "OOF" else c for c in caches]
        r[name] = score(stem, decode(stem, cs), gts).tolist()
    return stem, r
if __name__ == "__main__":
    out, split, cfg = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
    if split == "fold0":
        if os.environ.get("FOLD0_PREREG") != "1": raise SystemExit("fold 0 refused without FOLD0_PREREG=1")
        stems = sorted(s for s, v in FOLDS.items() if v == 0)
    else:
        stems = sorted(s for s, v in FOLDS.items() if v in (1, 2, 3, 4))
    if os.environ.get("LIMIT"): stems = stems[: int(os.environ["LIMIT"])]
    with Pool(int(os.environ.get("NPROC", "4")), initializer=init, initargs=(cfg,)) as p:
        res = dict(p.imap_unordered(one, stems))
    json.dump({"split": split, "stems": stems, "per_image": res}, open(out, "w"))
    for name in cfg:
        a = np.sum([res[s][name] for s in stems], axis=0)
        print(name, "PQ", round(a[0] / (a[1] + 0.5 * a[2] + 0.5 * a[3]), 6), "TP/FP/FN", a[1:].astype(int).tolist(), flush=True)
