# P7 (frozen Nov 10 11:45 PM): fold-0 and test maps = mean of the 4 fold U-Nets (none saw fold 0), each 8-view TTA. Also a fallback test CSV (frozen P6 decode). NO submission.
# GPU0 -> fold 0 (142 imgs), GPU1 -> test (180 imgs). Expected ~0.45 GPU-h.
import os, sys, subprocess, time, csv, glob
from pathlib import Path
INP = Path("/kaggle/input"); W = Path("/kaggle/working"); t0 = time.time()
whl = next(INP.rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl"))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl)])
kit = next(INP.rglob("infer_ens.py")).parent; py = sys.executable
ws = sorted(str(p) for p in INP.rglob("unet_tf_efficientnetv2_s_f[1-4].pt")); print("WEIGHTS", ws, flush=True); assert len(ws) == 4, ws
ann = next(INP.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json")); trd = ann.parent / "train_images"
ted = next(p for p in INP.rglob("test_images") if p.is_dir())
f0 = sorted(r["stem"] for r in csv.DictReader(open(kit / "folds_ours.csv")) if r["fold"] == "0"); assert len(f0) == 142
(W / "f0.txt").write_text("\n".join(f0))
procs = [subprocess.Popen(["bash", "-c", f"CUDA_VISIBLE_DEVICES=0 {py} {kit}/infer_ens.py {W}/maps_f0 {trd} {W}/f0.txt " + " ".join(ws)]),
         subprocess.Popen(["bash", "-c", f"CUDA_VISIBLE_DEVICES=1 {py} {kit}/infer_ens.py {W}/maps_test {ted} ALL " + " ".join(ws)])]
print("EXIT", [p.wait() for p in procs], round(time.time() - t0), flush=True)
rc = subprocess.call([py, str(kit / "make_csv.py"), str(W / "maps_test"), str(ted), str(W / "submission_unet_fold4avg.csv")]); print("CSV_RC", rc, flush=True)
for d in ("maps_f0", "maps_test"):
    subprocess.call(["tar", "czf", str(W / f"{d}.tgz"), "-C", str(W), d]); subprocess.call(["rm", "-rf", str(W / d)])
print("ALL_DONE", round(time.time() - t0), flush=True)
