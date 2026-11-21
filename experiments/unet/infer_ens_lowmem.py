#!/usr/bin/env python3
"""Same math as kit/infer_ens.py (mean of N models, each 8-view D4 TTA fp16, round once to uint8).
Differences vs the Kaggle script, only to fit a 6 GB card next to casmi-ice (~0.26 GB):
  - one model on GPU at a time (weights stay on CPU)
  - TTA view-batch = 2 instead of 4 (probe: 4 views peaked at 6.01 GB and was slower due to paging;
    2 views peaked at 3.17 GB; 1 view 1.74 GB). 8 views are still all applied.
Mean values of one image differed by ~2e-8 across view-batch 1/2/4.
usage: infer_ens_lowmem.py OUTDIR IMGDIR STEMS_FILE|ALL W1.pt [W2.pt ...]
"""
import sys, time, json
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
OUT, IMGDIR, SF = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; WS = sys.argv[4:]; OUT.mkdir(parents=True, exist_ok=True)
stems = sorted(p.stem for p in IMGDIR.glob("*.jpeg")) if SF == "ALL" else Path(SF).read_text().split()
yy, xx = np.mgrid[:2048, :2048]; DISK = (yy - 1023.5) ** 2 + (xx - 1023.5) ** 2 <= 950 ** 2
VIEW_BATCH = 2
t0 = time.time()
models = []
for w in WS:
    m = smp.Unet(encoder_name="tu-tf_efficientnetv2_s", encoder_weights=None, in_channels=1, classes=1)
    sd = torch.load(w, map_location="cpu", weights_only=True)
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    models.append(m.eval())
    print("CPU_LOADED", Path(w).name, flush=True)

@torch.no_grad()
def tta(model, x):
    acc = torch.zeros((1, 1, 2048, 2048), device="cuda"); vs = []
    for t in range(8):
        v = torch.rot90(x, t % 4, (2, 3)); vs.append(torch.flip(v, (3,)) if t >= 4 else v)
    for i in range(0, 8, VIEW_BATCH):
        with torch.autocast("cuda", dtype=torch.float16):
            o = torch.sigmoid(model(torch.cat(vs[i:i + VIEW_BATCH])).float())
        for j in range(o.shape[0]):
            t = i + j; oo = o[j:j + 1]
            if t >= 4: oo = torch.flip(oo, (3,))
            acc += torch.rot90(oo, -(t % 4), (2, 3))
    return acc / 8

for n, s in enumerate(stems):
    img = cv2.imread(str(IMGDIR / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE); assert img is not None and img.shape == (2048, 2048), s
    x = img.astype(np.float32) / 255.0; mu, sd = x[DISK].mean(), x[DISK].std() + 1e-6
    x = torch.from_numpy((x - mu) / sd)[None, None].cuda()
    acc = None
    for m in models:
        m.cuda()
        p = tta(m, x)
        acc = p if acc is None else acc + p
        m.cpu()
        torch.cuda.empty_cache()
    p = acc / len(models)
    cv2.imwrite(str(OUT / f"{s}.png"), np.clip(p[0, 0].cpu().numpy() * 255 + 0.5, 0, 255).astype(np.uint8))
    if n % 10 == 0:
        print("INFER", n, len(stems), round(time.time() - t0), "s",
              "max_alloc_gb", round(torch.cuda.max_memory_allocated() / 2**30, 3), flush=True)
        torch.cuda.reset_peak_memory_stats()
json.dump({"n": len(stems), "models": WS, "sec": time.time() - t0, "mode": "one_model_gpu", "view_batch": VIEW_BATCH}, open(OUT / "_infer.json", "w"))
print("INFER_DONE", len(stems), round(time.time() - t0), flush=True)
