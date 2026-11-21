#!/usr/bin/env python3
"""Out-of-fold Mask R-CNN instances for one grouped fold.

Fold 4 uses the model trained on folds 1-3. Folds 1-3 use the matching
holdout checkpoint. Does not score fold 0. Does not submit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR
from filament.rle import encode_rle

spec = importlib.util.spec_from_file_location(
    "train_maskrcnn_convnext", ROOT / "scripts" / "train_maskrcnn_convnext.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

OUT_ROOT = FAST_ARTIFACTS / "maskrcnn_oof"
SCORE_FLOOR = 0.05


def main() -> None:
    fold = int(sys.argv[1])
    if fold not in (0, 1, 2, 3, 4):
        raise SystemExit("fold must be 0, 1, 2, 3, or 4")
    # Fold 0 was never a training fold. The folds 1-3 model is the one that
    # did not see it. Holdout folders 1-3 use their own checkpoints.
    weight_fold = 4 if fold == 0 else fold
    weights = trainer.WEIGHTS_DIR / trainer.family_for_holdout(weight_fold) / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")
    out_dir = OUT_ROOT / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = trainer.build_model()
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(name for name, got in fold_of.items() if got == fold)
    print(f"fold {fold} images {len(files)} weights {weights}", flush=True)
    for index, file_name in enumerate(files, start=1):
        dest = out_dir / f"{Path(file_name).stem}.json"
        if dest.exists():
            continue
        image = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(file_name)
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            pred = model([tensor.to(device)])[0]
        scores = pred["scores"].detach().cpu().numpy()
        masks = pred["masks"].detach().cpu().numpy()
        rows = []
        for score, mask in zip(scores, masks):
            if float(score) < SCORE_FLOOR:
                continue
            prob = mask[0]
            binary = (prob > 0.5).astype(np.uint8)
            if binary.shape != (IMAGE_SIZE, IMAGE_SIZE):
                binary = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                prob = cv2.resize(prob, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            if int(binary.sum()) < 96:
                continue
            inside = prob[binary > 0]
            rows.append(
                {
                    "rle": encode_rle(binary),
                    "score": float(score),
                    "mean_prob": float(inside.mean()) if inside.size else 0.0,
                }
            )
        dest.write_text(json.dumps(rows))
        if index % 20 == 0:
            print(f"inferred {index}/{len(files)}", flush=True)
    print(f"DONE infer_maskrcnn_oof fold {fold}", flush=True)


if __name__ == "__main__":
    main()
