from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test" / "test_images"
ARTIFACTS = PROJECT_ROOT / "artifacts"
# YOLO images/weights live on the internal SSD so training is not Seagate-bound.
FAST_CACHE = Path.home() / ".cache" / "filament-segmentation-2026"
YOLO_DIR = FAST_CACHE / "yolo_seg"
YOLO_ALL_DIR = FAST_CACHE / "yolo_seg_all"
YOLO_SHIFT_DIR = FAST_CACHE / "yolo_seg_shift"
YOLO_CLAHE_DIR = FAST_CACHE / "yolo_seg_clahe"
YOLO_PHYS_DIR = FAST_CACHE / "yolo_seg_phys"
# Method 2: flatten+raw+Sobel. Never rmtree yolo_seg or yolo_seg_phys.
YOLO_RESIDUAL_DIR = FAST_CACHE / "yolo_seg_residual"
# Random-rater / expected-PQ labels. Never rmtree yolo_seg or yolo_seg_all for this.
YOLO_RATER_DIR = FAST_CACHE / "yolo_seg_rater"
# Native-2048 overlapping tiles for later tile_native_s640. Never rmtree live yolo_seg*.
YOLO_TILE_DIR = FAST_CACHE / "yolo_seg_tiles"
# Small/unidentifiable specialist. Never rmtree yolo_seg or yolo_seg_all.
YOLO_FAINT_DIR = FAST_CACHE / "yolo_seg_faint"
# Medoid labels after Slot1 erode 3x3x1. Never rmtree live yolo_seg*.
YOLO_ERODE_GT_DIR = FAST_CACHE / "yolo_seg_erode_gt"
# One real rater outline per agreed filament (match-any). Never a union dump.
YOLO_PQ_ANY_DIR = FAST_CACHE / "yolo_seg_pq_any"
WEIGHTS_DIR = FAST_CACHE / "weights"
# Inference JSON + Kaggle CSVs stay on the internal SSD (Seagate is the data disk).
FAST_ARTIFACTS = FAST_CACHE / "artifacts"
FAST_SUBMISSION_DIR = FAST_CACHE / "submissions"
SUBMISSION_DIR = FAST_SUBMISSION_DIR

IMAGE_SIZE = 2048
SEED = 20260910
