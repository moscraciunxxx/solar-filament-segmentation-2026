# Slot-2 U-Net refit on ALL 707 images (P7 recipe = P6 recipe), 2 seeds in parallel (GPU0 seed 1, GPU1 seed 2), fixed 9,850 steps each
# (median of the 9,576-9,979 steps the fold models ran). Built-in learning test: speed alarm at step 200 (> 0.80 s/step aborts),
# collapse alarm at step 500 (P6 rule) and at step 2500 (~30 min: loss < 0.9 x loss@500, sep > 0.25, batch 16 every step).
# Then test maps = mean of the 2 seeds (8-view TTA each) and an RLE CSV with format checks. NO submission. Expected ~2.25 GPU-h.
import os, sys, subprocess, time, glob, json
from pathlib import Path
INP = Path("/kaggle/input"); W = Path("/kaggle/working"); t0 = time.time()
whl = next(INP.rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl"))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl)])
kit = next(INP.rglob("worker_unet.py")).parent; py = sys.executable
ted = next(p for p in INP.rglob("test_images") if p.is_dir())
procs = [subprocess.Popen(["bash", "-c", f"UNET_SEED={s} CUDA_VISIBLE_DEVICES={i} {py} {kit}/worker_unet.py 0 tu-tf_efficientnetv2_s full {W}/out steps:9850 none"]) for i, s in enumerate((1, 2))]
codes = [p.wait() for p in procs]; print("WORKER_EXIT", codes, round(time.time() - t0), flush=True)
for r in sorted((W / "out").rglob("result.json")): print("RESULT", r.parent.name, r.read_text()[:1500], flush=True)
ws = sorted(str(p) for p in (W / "out").rglob("unet_tf_efficientnetv2_s_f0_s*.pt")); print("WEIGHTS", ws, flush=True)
if codes != [0, 0] or len(ws) != 2: print("REFIT_FAILED - no CSV", flush=True); sys.exit(0)
rc = subprocess.call(["bash", "-c", f"CUDA_VISIBLE_DEVICES=0 {py} {kit}/infer_ens.py {W}/maps_test {ted} ALL " + " ".join(ws)]); print("INFER_RC", rc, flush=True)
rc = subprocess.call([py, str(kit / "make_csv.py"), str(W / "maps_test"), str(ted), str(W / "submission_unet_refit707_2seed.csv")]); print("CSV_RC", rc, flush=True)
subprocess.call(["tar", "czf", str(W / "maps_test.tgz"), "-C", str(W), "maps_test"]); subprocess.call(["rm", "-rf", str(W / "maps_test")])
print("ALL_DONE", round(time.time() - t0), flush=True)
