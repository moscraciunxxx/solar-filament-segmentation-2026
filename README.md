# Solar Filament Segmentation 2026

Code and experimental records for Kaggle **Solar Filament Segmentation 2026** (`filament-segmentation-2026`), a 2048×2048 solar H-alpha instance segmentation challenge scored by **Panoptic Quality (PQ) at IoU > 0.5** pooled across multi-annotator ground-truth masks via `pycocotools`.

> **Data & Weights Notice:** Competition images, ground-truth masks, model checkpoints (`.pt`/`.pth`), and Kaggle submissions are **not** included. No MAGFiLO external test labels were ever used.

---

## Approach Overview

1. **Exact Scorer & Medoid-Eroded Targets (`src/filament/`)**
   - Multi-annotator training images can have up to several annotators per frame, while PQ pools all-pairs IoU > 0.5 matches across annotator-images (`TP / (TP + 0.5*FP + 0.5*FN)`).
   - We select the **medoid annotator** per training image (`src/filament/medoid_gt.py`) and apply a **1-iteration 3×3 elliptical erosion** to align polygon boundaries with crisp physical filament edges.
   - All validation uses `pycocotools.mask.frPyObjects` (`src/filament/coco_io.py`, `src/filament/metric.py`, `scripts/rescore_champion_coco_raster.py`) matching the host scorer (avoiding the ~0.008–0.009 PQ inflation caused by `cv2.fillPoly` rasterization discrepancies on thin filaments).

2. **Champion Detector (`yolo11s-seg` @ 2048px, 3 Seeds × 24-View TTA)**
   - Three `yolo11s-seg` seeds (`20260910`, `20260911`, `20260912`) trained on 565 non-holdout images (folds 1–4, 2048×2048, `mask_ratio=4`) plus a 707-image refit pipeline (`experiments/refit/`).
   - **72-view TTA + IoU-0.5 Clustering (`scripts/solution/infer_ens3.py`, `decode_ens3.py`):** 8 D4 geometric views × 3 scales (`1792`, `2048`, `2304`) across 3 seeds, merged by greedy IoU ≥ 0.5 vote clustering (`vote >= 24/72`, pixel majority vote, disjoint owner resolution).

3. **Dense Cross-Family U-Net (`tu-tf_efficientnetv2_s`, `experiments/unet/`)**
   - Clean reimplementation of `dariushafshar`'s `smp.Unet` (`tu-tf_efficientnetv2_s`, ImageNet-pretrained) trained on 512×512 crops (70% filament-centered), per-disk z-score normalization, soft multi-rater union targets, BCE + soft-Dice + boundary/focal-tversky loss, and 8-view D4 TTA inference.

4. **P10 Dense Pixel-Level Fusion (`experiments/fusion/`)**
   - Blends calibrated YOLO vote-fraction maps `Y` (from 72-view / 24-view clusters) with dense U-Net probability maps `U`:
     $$P_{10} = 0.4 \cdot Y + 0.6 \cdot U$$
   - Hysteresis connected-component extraction (`t_lo=0.30`, `t_hi=0.60`, `mean >= 0.60`, `area >= 200 px`) followed by disjoint resolution.

5. **Strict Two-Stage Gating (Gate A + Single Holdout Look)**
   - Every candidate change had to clear **Gate A** on true 4-fold OOF (`folds 1–4`, 565 images): `ΔPQ >= +0.006`, positive 95% CI lower bounds under both **image bootstrap** and **date-cluster bootstrap** (solar rotation persists across adjacent days), and `>= 3/4` folds positive—before spending a single locked look on held-out `fold 0` (142 images / 228 annotator-images).

---

## Verified Results

### Final Submissions & Main Pipeline

| Model / Experiment | Folds 1–4 OOF PQ (`pycocotools`) | Fold-0 PQ (`pycocotools`) | Public LB | Private LB | Notes |
| --- | --- | --- | --- | --- | --- |
| `yolo11s-seg` match-any @ 1280 (`56444509`) | — | — | 0.29 | 0.27 | Early baseline |
| `yolo11s-seg` eroded-medoid @ 2048, 1 seed (`56456485`) | — | 0.438724 | 0.37 | 0.35 | Parent single model (`TP 949, FP 410, FN 640`) |
| `yolo11s-seg` 1-seed 24-view TTA, vote ≥ 8 (`57157301`) | 0.438046 *(4-fold OOF)* | 0.465188 *(cv2)* | 0.38 | 0.36 | Single model TTA (`1018 / 409 / 571` cv2) |
| **Champion:** 3-seed `yolo11s-seg` 72-view TTA, vote ≥ 24 (`57162191`) | 0.467362 *(in-sample)* / 0.443759 *(72v OOF)* | **0.465553** *(0.474168 cv2)* | **0.39** | **0.37** | `TP 1018, FP 378, FN 571`, SQ 0.6826, RQ 0.6821 |
| **U-Net** (`EffNetV2-S`, 8-view TTA, P6/P7) (`58259101` 2-seed 707 refit) | **0.445502** | 0.456299 | 0.38 | **0.37** | 4-fold OOF `+0.00746` vs YOLO fold OOF; 762 U-Net-only GT hits |
| **P10 Dense Pixel Fusion** (`0.4*Y + 0.6*U`) (`58287237`) | **0.455093** *(nested OOF)* | **0.468361** | **0.39** | **0.37** | Nested OOF `+0.01133` vs 72v OOF (4/4 folds+, CI `[+0.0047, +0.0182]`); Fold-0 `+0.00281` |

*Note on leaderboard cleanup:* Before deadline, >70 entries clustered at `0.55–0.70` public LB from suspected MAGFiLO public-test mask leakage. After post-deadline host verification/cleanup, the top private LB score across all valid teams was `0.42`, placing our `0.37` private (`0.39` public) cleanly in the legitimate competitive tier.

---

## What Didn't Work (Killed Experiments)

All candidates below were killed cleanly via OOF Gate A or a locked Fold-0 check:

| Experiment | Validation Result | Why Killed |
| --- | --- | --- |
| **BarrelAged-style candidate selector** (`experiments/selector/`) | 1-model OOF: **0.4513** (`+0.0133` vs `0.4380`, CI `[+0.0075, +0.0194]`); Fold-0 vs 72v champ: **0.4587** (`-0.0069`) | Overfit single-model OOF; failed against 72-view champion |
| **P9 Ensemble-level selector** (on 72-voter OOF) | Nested OOF: **0.446308** (`+0.00255` vs `0.443759`, img CI95 `[-0.00194, +0.00731]`) | Failed Gate A (`< +0.006`, CI crossed 0) |
| **Label Transfer (`LT4`, ±5 min GONG neighbours)** (`experiments/label_transfer/`) | OOF: **0.434955** (`4237 / 2389 / 2373`) vs U-Net `0.445502` (**-0.01055**) | Near-duplicate frames shifted calibration upward (`+384` FP) |
| **Per-annotator Label Sampling (`LS`)** (`experiments/oof_variants/`) | Gate A OOF: **0.443118** (`+0.00507` vs `0.438046`, fold 2 `-0.0105`) | Below `+0.006` Gate A bar |
| **D4 `RandomTranspose(p=0.5)`** (`experiments/oof_variants/`) | Gate A OOF: **0.442949** (`+0.00490` vs `0.438046`, fold 2 `-0.0069`) | Below `+0.006` Gate A bar |
| **P11 U-Net as weighted voter** (`w=12` in YOLO pool) | Nested OOF: **0.449888** (`+0.00613`); Fold-0: **0.464787** (`-0.00077`) | Hurt Fold-0 vs Champion |
| **Ownership decode** (`decode_ens3_own.py`) | Folds 1–4: `0.475429`; Fold-0: **0.468760** | Missed `+0.01` slot-1 bar (`0.475553`) |
| **Larger YOLO backbones @ 2048** (`yolo11m`, `yolo11l`, `5-seed s`) | `yolo11m` TTA `0.461333`; `yolo11l` `0.419579` solo / `0.444512` TTA; `3s+m` `0.469981` | No improvement over 3-seed `yolo11s` (`0.4742` cv2) |
| **CLAHE preprocessing** (`yolo11s_seg_clahe` @ 1280) | Fold-0: **0.357111** (unmatched) / **0.398684** (matched) vs `0.410` | Degraded detector precision & recall |
| **Flat-field CLV input** (`yolo11s_seg_erode_gt_2048_clv`) | Fold-0: **0.406468** (`795 / 289 / 794`) vs `0.438724` | Hurt native limb/disk contrast cues |
| **Component-tree decoding** (on YOLO max-logits) | **0.164409** (uncapped) / **0.275900** (capped) / **0.348698** (in-parent) | Detector logits uncalibrated for tree extraction |
| **CondInst (ResNet-18 stride 8 @ 1024)** | Fold-0: **0.0** (vanilla) / **0.036178** (body-pos + centerness) | Failed to converge on thin elongated filaments |
| **Stride-2 YOLO (`yolo11l_seg_full_2048_m2`)** | Epoch-1 Fold-0: **0.066462** (epoch-2 mAP50 collapsed `0.287 -> 0.0248`) | Pretrained head stride surgery collapsed |
| **Mask R-CNN ConvNeXt / Soft-CLV U-Net / ResNet34-1536 U-Net** | Fold-0 / OOF: `0.369197` / `0.370578` / `0.3537` | Substantially behind `yolo11s` and `EffNetV2-S` U-Net |

---

## Repository Structure

```text
.
├── src/filament/           # Core PQ metric, pycocotools RLE/IO, medoid GT, postprocessing, CondInst/U-Net modules
├── scripts/
│   ├── solution/           # Packaged 3-seed 72-view TTA inference & decode_ens3.py
│   └── *.py                # Full training, TTA evaluation, rescoring, and ablation scripts
├── experiments/
│   ├── refit/              # 707-image YOLO champion refit workers, fusion & CSV checks
│   ├── unet/               # EffNetV2-S U-Net workers, 8-view TTA inference, scoring & Kaggle runners
│   ├── fusion/             # P10 dense pixel fusion, P11 weighted voter, Gate A & Fold-0 look scripts
│   ├── selector/           # BarrelAged-style candidate extraction, pool scoring, folds_ours.csv
│   ├── label_transfer/     # GONG +/-5m neighbour downloader, limb registration & LT4 U-Net worker
│   ├── oof_variants/       # 4-fold YOLO OOF, LS, and D4 experiment workers & Kaggle scripts
│   └── rfdetr_prep/        # Unlaunched RF-DETR tile prep & scoring scripts
├── tests/                  # Unit & regression gate tests
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── THIRD_PARTY_NOTICES.md  # Apache-2.0 / CC BY 4.0 / CC BY-NC 4.0 attributions
```

---

## License & Credits

- Original code in this repository is released under the **MIT License** (`LICENSE`).
- See `THIRD_PARTY_NOTICES.md` for attributions to `Barrel-Aged/filament-segmentation-2026` (Apache-2.0), `dariushafshar` (CC BY 4.0 / `smp` MIT), and the MAGFiLO dataset (CC BY-NC 4.0).
