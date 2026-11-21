#!/usr/bin/env python3
"""Our own retrain of the dariushafshar dense U-Net recipe (rule 2.8: recipe only, no foreign weights).
usage: worker_unet.py FOLD ENCODER MODE OUTROOT TRAIN_BUDGET_SEC
  FOLD k in 1..4: train on our folds {1,2,3,4}\\{k} (fold 0 never seen), held-out = fold k.
  MODE learn: real schedule, stop at LEARN_STEPS, collapse alarm, quick held-out check, exit.
  MODE full : train (steps = min(10500, budget-fitted at step 200)), save weights, 8-view TTA uint8 maps on fold k, fold 0, test.
Collapse alarm: NaN/inf loss -> exit 3; at step 500 loss(401-500) must be < 0.85*loss(1-50) and mean p(target>=.5) - mean p(target==0) > 0.25;
effective batch asserted == 16 every step (no silent halving; OOM raises)."""
import os, sys, json, time, math, csv, random
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
k = int(sys.argv[1]); ENC = sys.argv[2]; MODE = sys.argv[3]; OUTROOT = Path(sys.argv[4]); FIXED = int(sys.argv[5].split(":")[1]) if sys.argv[5].startswith("steps:") else 0; BUDGET = 0.0 if FIXED else float(sys.argv[5]); SETS = (sys.argv[6] if len(sys.argv) > 6 else "held").split(",")
SEEDX = int(os.environ.get("UNET_SEED", "0")); TAG = f"{ENC.replace('tu-','')}_f{k}" + (f"_s{SEEDX}" if SEEDX else ""); OUT = OUTROOT / TAG; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time(); RES = {"fold": k, "encoder": ENC, "mode": MODE}
def log(*a):
    print(f"[{TAG} {time.time()-T0:7.0f}s]", *a, flush=True)
def save(): (OUT / "result.json").write_text(json.dumps(RES, indent=1))
SEED = 20261110 + k + 1000 * SEEDX; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
INP = Path(os.environ.get("UNET_INP", "/kaggle/input"))
import segmentation_models_pytorch as smp
from pycocotools import mask as mu
KIT = Path(__file__).parent
folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(KIT / "folds_ours.csv"))}
ann = next(INP.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json")); IMG = ann.parent / "train_images"
TEST = next(p for p in INP.rglob("test_images") if p.is_dir())
coco = json.loads(ann.read_text())
views = {}  # stem -> {view_id: [segs]}
for im in coco["images"]:
    views.setdefault(Path(im["file_name"]).stem, {})[im["id"]] = []
id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
for a in coco["annotations"]:
    views[id2stem[a["image_id"]]][a["image_id"]].append(a["segmentation"])
assert set(views) == set(folds), (len(views), len(folds))
train = sorted(folds) if k == 0 else sorted(s for s, f in folds.items() if f in (1, 2, 3, 4) and f != k)
held = [] if k == 0 else sorted(s for s, f in folds.items() if f == k); f0 = [] if k == 0 else sorted(s for s, f in folds.items() if f == 0)
assert not set(train) & set(held) and not set(train) & set(f0)
SMOKE = os.environ.get("UNET_SMOKE") == "1"
if SMOKE: train, held, f0 = train[:4], held[:2], f0[:1]
RES.update(n_train=len(train), n_held=len(held), train_folds=sorted({folds[s] for s in train}))
yy, xx = np.mgrid[:2048, :2048]; DISK = (yy - 1023.5) ** 2 + (xx - 1023.5) ** 2 <= 950 ** 2
def norm_stats(img):
    x = img[DISK].astype(np.float32) / 255.0; return float(x.mean()), float(x.std() + 1e-6)
def soft_target(stem):
    acc = np.zeros((2048, 2048), np.float32); vs = views[stem]
    for vid, segs in vs.items():
        u = np.zeros((2048, 2048), np.uint8)
        for seg in segs:
            polys = seg if (seg and isinstance(seg[0], list)) else [seg]
            polys = [list(map(float, p)) for p in polys if len(p) >= 6]
            if polys: u |= mu.decode(mu.merge(mu.frPyObjects(polys, 2048, 2048))).astype(np.uint8)
        acc += u
    return np.round(acc / max(1, len(vs)) * 255).astype(np.uint8)
t = time.time(); IM, TG, ST, POS = [], [], [], []
for s in train:
    img = cv2.imread(str(IMG / (s + ".jpeg")), cv2.IMREAD_GRAYSCALE); assert img is not None and img.shape == (2048, 2048), s
    tg = soft_target(s); IM.append(img); TG.append(tg); ST.append(norm_stats(img))
    ys, xs = np.nonzero(tg[::8, ::8]); POS.append(np.stack([ys * 8, xs * 8], 1) if len(ys) else None)
log("loaded", len(IM), "train imgs + soft targets in", round(time.time() - t), "s; mean target>0 frac", round(float(np.mean([(g > 0).mean() for g in TG])), 4))
CROP = 512; BATCH = 16; MICRO = 8 if "efficientnet" in ENC else 16; ACC = BATCH // MICRO
def sample(n):
    xb = np.empty((n, 1, CROP, CROP), np.float32); yb = np.empty((n, 1, CROP, CROP), np.float32)
    for i in range(n):
        j = random.randrange(len(IM))
        if random.random() < 0.7 and POS[j] is not None:
            cy, cx = POS[j][random.randrange(len(POS[j]))]; cy += random.randint(-128, 128); cx += random.randint(-128, 128)
            y0 = int(np.clip(cy - CROP // 2, 0, 2048 - CROP)); x0 = int(np.clip(cx - CROP // 2, 0, 2048 - CROP))
        else:
            y0 = random.randrange(0, 2048 - CROP + 1); x0 = random.randrange(0, 2048 - CROP + 1)
        x = IM[j][y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255.0; y = TG[j][y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255.0
        g = math.exp(random.uniform(math.log(0.8), math.log(1.25))); x = np.power(np.clip(x, 0, 1), g)
        m, sd = ST[j]; x = (x - m) / sd
        x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)   # contrast / brightness jitter (normalised units)
        r = random.randrange(4); x = np.rot90(x, r); y = np.rot90(y, r)
        if random.random() < 0.5: x = x[:, ::-1]; y = y[:, ::-1]
        xb[i, 0] = x; yb[i, 0] = y
    return torch.from_numpy(xb), torch.from_numpy(yb)
dev = "cpu" if SMOKE else "cuda"
model = smp.Unet(encoder_name=ENC, encoder_weights="imagenet", in_channels=1, classes=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
TOTAL = [FIXED or 10500]; WARM = 300
def lr_at(s):
    if s < WARM: return (s + 1) / WARM
    p = min(1.0, (s - WARM) / max(1, TOTAL[0] - WARM)); return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
scaler = torch.amp.GradScaler("cuda", enabled=not SMOKE)
AC = dict(device_type="cpu", dtype=torch.bfloat16) if SMOKE else dict(device_type="cuda", dtype=torch.float16)
def loss_fn(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y); p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1); return bce + dice, p
LEARN_STEPS = 3 if SMOKE else 700; losses = []; sep = []; t_train = time.time(); alarm = None; RES["log"] = []
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(2); fut = pool.submit(sample, BATCH)
step = 0
model.train()
while step < (LEARN_STEPS if MODE == "learn" else TOTAL[0]):
    xb, yb = fut.result(); fut = pool.submit(sample, BATCH)
    for g in opt.param_groups: g["lr"] = 3e-4 * lr_at(step)
    opt.zero_grad(set_to_none=True); seen = 0; ltot = 0.0
    for a in range(ACC):
        xm = xb[a * MICRO:(a + 1) * MICRO].to(dev, non_blocking=True); ym = yb[a * MICRO:(a + 1) * MICRO].to(dev, non_blocking=True)
        with torch.autocast(**AC):
            logit = model(xm)
        l, p = loss_fn(logit.float(), ym)
        scaler.scale(l / ACC).backward(); seen += xm.shape[0]; ltot += l.item() / ACC
    assert seen == BATCH, f"effective batch {seen} != {BATCH}"
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
    if not math.isfinite(ltot): alarm = f"non-finite loss at step {step}"; break
    losses.append(ltot)
    with torch.no_grad():
        pos = ym >= 0.5; neg = ym == 0
        if pos.any() and neg.any(): sep.append(float(p[pos].mean() - p[neg].mean()))
    step += 1
    if step == (2 if SMOKE else 200) and MODE == "full":
        sps = (time.time() - t_train) / step
        if FIXED:
            TOTAL[0] = 4 if SMOKE else FIXED
            if not SMOKE and sps > float(os.environ.get("UNET_MAX_SPS", "0.80")): alarm = f"speed alarm: {sps:.3f} s/step > max (session would overrun)"; break
        else:
            fit = int(BUDGET / sps); TOTAL[0] = 4 if SMOKE else max(WARM + 1000, min(10500, fit))
        RES["sec_per_step_200"] = sps; RES["total_steps"] = TOTAL[0]; log("s/step", round(sps, 3), "-> total steps", TOTAL[0]); save()
    if step % 100 == 0:
        e = {"step": step, "loss100": float(np.mean(losses[-100:])), "sep100": float(np.mean(sep[-100:])) if sep else None,
             "lr": opt.param_groups[0]["lr"], "s_per_step": (time.time() - t_train) / step, "eff_batch": seen, "micro": MICRO,
             "mem_gb": (0 if SMOKE else torch.cuda.max_memory_allocated() / 2**30)}
        RES["log"].append(e); log(json.dumps(e)); save()
    if step == 500:
        l0 = float(np.mean(losses[:50])); l1 = float(np.mean(losses[400:500])); s1 = float(np.mean(sep[-100:]))
        ok = (l1 < 0.85 * l0) and (s1 > 0.25)
        RES["alarm500"] = {"loss_1_50": l0, "loss_401_500": l1, "ratio": l1 / l0, "sep_401_500": s1, "ok": ok}; log("ALARM_CHECK", RES["alarm500"]); save()
        if not ok: alarm = "collapse alarm at step 500"; break
    if step == 2500 and MODE == "full":   # ~30-min learning check inside the real run
        l2 = float(np.mean(losses[2400:2500])); l1 = RES["alarm500"]["loss_401_500"]; s2 = float(np.mean(sep[-100:]))
        ok2 = (l2 < 0.9 * l1) and (s2 > 0.25) and all(e["eff_batch"] == BATCH for e in RES["log"])
        RES["alarm2500"] = {"loss_2401_2500": l2, "loss_401_500": l1, "sep": s2, "ok": ok2}; log("ALARM_CHECK_2500", RES["alarm2500"]); save()
        if not ok2: alarm = "collapse alarm at step 2500"; break
RES["train_steps"] = step; RES["train_sec"] = time.time() - t_train
if alarm:
    RES["ALARM"] = alarm; save(); log("ALARM", alarm); sys.exit(3)
model.eval()
@torch.no_grad()
def predict(img, views=8):
    views = 1 if SMOKE else views
    m, sd = norm_stats(img); x = torch.from_numpy((img.astype(np.float32) / 255.0 - m) / sd)[None, None].to(dev)
    vs = []
    for t_ in range(views):
        v = torch.rot90(x, t_ % 4, (2, 3)); vs.append(torch.flip(v, (3,)) if t_ >= 4 else v)
    acc = torch.zeros((1, 1, 2048, 2048), device=dev)
    for i in range(0, views, 4):
        b = torch.cat(vs[i:i + 4])
        with torch.autocast(**AC):
            o = torch.sigmoid(model(b).float())
        for jj in range(o.shape[0]):
            t_ = i + jj; oo = o[jj:jj + 1]
            if t_ >= 4: oo = torch.flip(oo, (3,))
            acc += torch.rot90(oo, -(t_ % 4), (2, 3))
    return (acc / views)[0, 0].cpu().numpy()
if MODE == "learn":
    # quick held-out sanity: 1-view soft Dice / IoU@0.5 vs soft target on 16 held-out fold-k images
    d, iou = [], []
    for s in held[:2 if SMOKE else 16]:
        img = cv2.imread(str(IMG / (s + ".jpeg")), cv2.IMREAD_GRAYSCALE); p = predict(img, 1); y = soft_target(s) / 255.0
        d.append(float(2 * (p * y).sum() / (p.sum() + y.sum() + 1e-6))); a = p > 0.5; b = y >= 0.5; iou.append(float((a & b).sum() / max(1, (a | b).sum())))
    RES["learn_heldout16"] = {"soft_dice": float(np.mean(d)), "iou_p05_vs_majority": float(np.mean(iou))}; log("LEARN_HELDOUT", RES["learn_heldout16"])
    RES["peak_mem_gb"] = (0 if SMOKE else torch.cuda.max_memory_allocated() / 2**30); save(); log("LEARN_DONE"); sys.exit(0)
torch.save({kk: v.half() for kk, v in model.state_dict().items()}, OUT / f"unet_{TAG}.pt"); log("saved weights")
t = time.time()
for name, stems, d in (("held", held, IMG), ("f0", f0, IMG), ("test", sorted(p.stem for p in TEST.glob("*.jpeg"))[:1 if SMOKE else None], TEST)):
    if name not in SETS: continue
    od = OUT / name; od.mkdir(exist_ok=True)
    for s in stems:
        img = cv2.imread(str(d / (s + ".jpeg")), cv2.IMREAD_GRAYSCALE); assert img is not None and img.shape == (2048, 2048), s
        cv2.imwrite(str(od / (s + ".png")), np.clip(predict(img) * 255 + 0.5, 0, 255).astype(np.uint8))
    RES[f"n_{name}"] = len(stems); log("TTA", name, len(stems), round(time.time() - t), "s"); save()
RES["infer_sec"] = time.time() - t; RES["peak_mem_gb"] = (0 if SMOKE else torch.cuda.max_memory_allocated() / 2**30); RES["total_sec"] = time.time() - T0; save(); log("DONE")
