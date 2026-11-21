#!/usr/bin/env python3
"""Prove spine_io reads TRAIN_JSON only and is not imported by U-Net v1 eval.

Never submits. Never opens test / Dataverse labels.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import ARTIFACTS, IMAGE_SIZE, TRAIN_JSON
from filament.spine_io import load_spines_by_file, rasterize_spine_ridge, union_ridge_for_file
from filament.spine_split import split_binary_on_ridge

FORBIDDEN_IMPORTS = {"filament.spine_io", "filament.spine_split"}
V1_EVALS = [
    ROOT / "scripts" / "eval_unet.py",
    ROOT / "scripts" / "eval_unet_merge.py",
    ROOT / "scripts" / "train_unet.py",
    ROOT / "scripts" / "make_unet_merged_submission.py",
    ROOT / "src" / "filament" / "semantic_dataset.py",
    ROOT / "src" / "filament" / "unet.py",
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _synthetic_split() -> dict:
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[20:44, 8:56] = 1
    ridge = np.zeros((64, 64), dtype=np.uint8)
    ridge[20:44, 31:33] = 1
    parts = split_binary_on_ridge(binary, ridge, min_area=8, max_area=80_000)
    return {
        "n_parts": len(parts),
        "split_into_two": len(parts) >= 2,
        "areas": [int(p.sum()) for p in parts],
    }


def main() -> None:
    if TRAIN_JSON.name != "MAGFiLO_1.0_Annotations_kaggle2026_train.json":
        raise SystemExit(f"refused: unexpected train json {TRAIN_JSON}")
    if "test" in str(TRAIN_JSON).lower() and "train" not in str(TRAIN_JSON):
        raise SystemExit("refused: train path looks like test")

    v1_imports = {str(p.relative_to(ROOT)): sorted(_imports(p) & FORBIDDEN_IMPORTS) for p in V1_EVALS}
    leaked = {k: v for k, v in v1_imports.items() if v}
    if leaked:
        raise SystemExit(f"v1 imported spine modules: {leaked}")

    by_file = load_spines_by_file(TRAIN_JSON)
    n_spines = sum(len(v) for v in by_file.values())
    sample_name = next(iter(by_file))
    sample_pts = by_file[sample_name][0]
    ridge = rasterize_spine_ridge(sample_pts, size=IMAGE_SIZE, width=1)
    union = union_ridge_for_file(sample_name, by_file, size=IMAGE_SIZE, width=1)
    synthetic = _synthetic_split()

    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "source": str(TRAIN_JSON),
        "source_is_train_json": True,
        "opened_test_labels": False,
        "n_files_with_spine": len(by_file),
        "n_spines": n_spines,
        "image_size": IMAGE_SIZE,
        "sample_file": sample_name,
        "sample_pts": int(len(sample_pts)),
        "sample_ridge_px": int(ridge.sum()),
        "sample_union_px": int(union.sum()),
        "v1_does_not_import_spine": leaked == {},
        "v1_imports_checked": v1_imports,
        "synthetic_split": synthetic,
        "wired_into_v1_eval": False,
        "next": "Use only after RESEARCH_HOLD_SPINE_V2. Do not import from eval_unet.py.",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "spine_io_verify.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"path": str(path), **{k: out[k] for k in (
        "n_files_with_spine", "n_spines", "sample_ridge_px", "v1_does_not_import_spine", "wired_into_v1_eval"
    )}}, indent=2))


if __name__ == "__main__":
    main()
