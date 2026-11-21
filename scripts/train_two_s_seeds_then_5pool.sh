#!/usr/bin/env bash
# Wait until the 96-voter pool releases MPS, then train seeds 20260913 and
# 20260914 on the parent yolo11s erode-gt 2048 recipe, then score the 5-seed pool.
set -u
ROOT="."
POOL_PID="${1:?pool pid required}"
FLAG="$ROOT/artifacts/pool_3s_plus_m_mps_free"
PY="$HOME/.venvs/filament/bin/python"
DATA="$HOME/.cache/filament-segmentation-2026/yolo_seg_erode_gt/data.yaml"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR="$HOME/.cache/filament-segmentation-2026/yolo"
export MPLCONFIGDIR="$HOME/.cache/filament-segmentation-2026/mpl"
cd "$ROOT"

echo "waiting for MPS flag; pool pid $POOL_PID"
while [ ! -f "$FLAG" ]; do
  if ! kill -0 "$POOL_PID" 2>/dev/null; then
    echo "FAILED: pool pid $POOL_PID exited before MPS was released"
    exit 1
  fi
  sleep 30
done
echo "MPS flag present, starting seed 20260913"

train_one() {
  local seed="$1"
  local name="yolo11s_seg_erode_gt_2048_s${seed}"
  local weights="$HOME/.cache/filament-segmentation-2026/weights/${name}"
  local best="${weights}/weights/best.pt"
  local last="${weights}/weights/last.pt"
  local results="${weights}/results.csv"
  local epochs=0
  if [ -f "$results" ]; then
    epochs=$(tail -n +2 "$results" | wc -l | tr -d ' ')
  fi
  if [ "$epochs" -ge 40 ]; then
    echo "seed $seed already finished $epochs epochs"
    return 0
  fi
  if [ -f "$last" ] && [ "$epochs" -gt 0 ]; then
    echo "resuming seed $seed from epoch $epochs"
    "$PY" -u scripts/train_yolo.py --name "$name" --resume || { echo "FAILED: seed $seed resume exit $?"; exit 1; }
  else
    "$PY" -u scripts/train_yolo.py \
      --name "$name" \
      --data "$DATA" \
      --model "$ROOT/yolo11s-seg.pt" \
      --imgsz 2048 \
      --batch 4 \
      --epochs 40 \
      --patience 12 \
      --seed "$seed" || { echo "FAILED: seed $seed train exit $?"; exit 1; }
  fi
  if [ ! -f "$best" ]; then
    echo "FAILED: seed $seed missing $best"
    exit 1
  fi
  echo "seed $seed best.pt ready"
}

train_one 20260913
train_one 20260914
echo "starting 5-seed score"
"$PY" -u scripts/eval_5seed_s_tta.py || { echo "FAILED: 5-seed score exit $?"; exit 1; }
echo "DONE chain"
