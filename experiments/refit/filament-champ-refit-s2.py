# Champion refit session 2: GPU0 trains seed 20260912 on all 707; GPU1 meanwhile runs the 24-view test TTA of seeds 20260910/11 (weights from
# session 1 = kernel input filament-champ-refit-s1); then seed 12's test TTA split over both GPUs; then the champion 72-voter fusion
# (24/72 votes, verbatim port proven byte-identical on the box) -> submission_champ_refit707_ens3.csv + format checks. NO submission.
import os, sys, subprocess, time, json
from pathlib import Path
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics==8.4.146"])
INP = Path("/kaggle/input"); W = Path("/kaggle/working"); t0 = time.time()
kit = next(INP.rglob("champ_refit_worker.py")).parent; wk = kit / "champ_refit_worker.py"; py = sys.executable
w10 = next(INP.rglob("last_s20260910.pt")); w11 = next(INP.rglob("last_s20260911.pt")); print("S1 WEIGHTS", w10, w11, flush=True)
import torch; assert torch.cuda.device_count() == 2, "need T4x2"
C = W / "caches"; w12 = W / "s20260912" / "last_s20260912.pt"
lane0 = f"CUDA_VISIBLE_DEVICES=0 {py} {wk} train 20260912 && CUDA_VISIBLE_DEVICES=0 {py} {wk} tta 20260912 {w12} {C}/s12 0 2"
lane1 = (f"CUDA_VISIBLE_DEVICES=1 {py} {wk} tta 20260910 {w10} {C}/s10 0 1 && CUDA_VISIBLE_DEVICES=1 {py} {wk} tta 20260911 {w11} {C}/s11 0 1 && "
         f"T=0; while [ ! -f {w12} ] && [ ! -f {W}/s20260912/result.json ] && [ $T -lt 12600 ]; do sleep 30; T=$((T+30)); done; sleep 20; [ -f {w12} ] && CUDA_VISIBLE_DEVICES=1 {py} {wk} tta 20260912 {w12} {C}/s12 1 2")
procs = [subprocess.Popen(["bash", "-c", l], env=dict(os.environ)) for l in (lane0, lane1)]
codes = [p.wait() for p in procs]; print("LANES_EXIT", codes, "hours", round((time.time() - t0) / 3600, 2), flush=True)
n = {s: len(list((C / s).glob("*.json"))) if (C / s).exists() else 0 for s in ("s10", "s11", "s12")}; print("CACHE_COUNTS", n, flush=True)
if codes != [0, 0] or any(v != 180 for v in n.values()): print("REFIT_INCOMPLETE - no CSV", flush=True); sys.exit(0)
ted = next(p for p in INP.rglob("test_images") if p.is_dir()); sub = W / "submission_champ_refit707_ens3.csv"
rc = subprocess.call(["bash", "-c", f"NPROC=3 {py} {kit}/champ_fuse.py {sub} {ted} {C}/s10 {C}/s11 {C}/s12"]); print("FUSE_RC", rc, flush=True)
rc = subprocess.call([py, str(kit / "csv_check.py"), str(sub), str(ted)]); print("CHECK_RC", rc, flush=True)
subprocess.call(["tar", "czf", str(W / "test_caches_refit707.tgz"), "-C", str(W), "caches"]); subprocess.call(["rm", "-rf", str(C)])
print("ALL_DONE hours", round((time.time() - t0) / 3600, 2), flush=True)
