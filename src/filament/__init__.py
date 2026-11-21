"""MAGFiLO filament instance-segmentation utilities.

All training and inference use only the official competition train JSON
and the provided H-alpha JPEGs. Public MAGFiLO test annotations are
never loaded.
"""

from .paths import PROJECT_ROOT, DATA_ROOT, TRAIN_JSON, TRAIN_IMG_DIR, TEST_IMG_DIR

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "TRAIN_JSON",
    "TRAIN_IMG_DIR",
    "TEST_IMG_DIR",
]
