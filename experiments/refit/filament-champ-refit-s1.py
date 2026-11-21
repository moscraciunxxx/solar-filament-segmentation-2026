# Champion refit on ALL 707 images, session 1: seeds 20260910 (GPU0) and 20260911 (GPU1) in parallel (T4x2 fits: the same recipe ran 2 folds
# in parallel per session in the S1/LS/D4 kernels, peak ~14 GiB per T4 with the v3 patch). Training only; outputs last_s<seed>.pt. NO submission.
import os, sys, subprocess, time
from pathlib import Path
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics==8.4.146"])
INP = Path("/kaggle/input"); W = Path("/kaggle/working"); t0 = time.time()
wk = next(INP.rglob("champ_refit_worker.py")); py = sys.executable
import torch; print("gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
assert torch.cuda.device_count() == 2, "need T4x2"
procs = [subprocess.Popen(["bash", "-c", f"CUDA_VISIBLE_DEVICES={i} {py} {wk} train {s}"], env=dict(os.environ)) for i, s in enumerate((20260910, 20260911))]
codes = [p.wait() for p in procs]; print("WORKER_EXIT", codes, "hours", round((time.time() - t0) / 3600, 2), flush=True)
for s in (20260910, 20260911): print(s, (W / f"s{s}" / f"last_s{s}.pt").exists(), flush=True)
print("ALL_DONE", flush=True)
