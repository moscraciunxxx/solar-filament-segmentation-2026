# Third-Party Notices & Attributions

This repository contains original competition code (MIT licensed) alongside ideas,
recipes, and adaptations inspired by or derived from the following third-party works:

1. **Barrel-Aged (`Barrel-Aged/filament-segmentation-2026`)**
   - License: Apache License 2.0
   - Usage: Reference pipeline architecture and candidate-selector experiments
     (`experiments/selector/`).

2. **Dariush Afshar (`dariushafshar`) — Dense U-Net Recipe**
   - Reference: Public Kaggle baseline/notebook for `filament-segmentation-2026`
     (CC BY 4.0) using `segmentation_models_pytorch` (MIT License) with
     `tu-tf_efficientnetv2_s` and per-disk z-score normalization on 512x512 crops.
   - Usage: Clean-room reimplementation and 4-fold/refit training/inference scripts
     in `experiments/unet/`.

3. **MAGFiLO 1.0 Dataset (NSO GONG H-alpha Solar Filaments)**
   - License: CC BY-NC 4.0 (not redistributed in this repository).
   - No competition images, masks, or external MAGFiLO test labels are included.

4. **Ultralytics YOLO (`yolo11s-seg`) & PyTorch / pycocotools**
   - Standard open-source dependencies listed in `requirements.txt` (not bundled).
