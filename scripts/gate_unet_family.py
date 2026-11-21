"""Shared v1 merge-gate reader. Never submits. Never starts MPS."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from filament.paths import ARTIFACTS, WEIGHTS_DIR

GATE = ARTIFACTS / "unet_slot1_merge_fold0_official.json"
RESNET_GATE = ARTIFACTS / "unet_resnet34_slot1_merge_fold0_official.json"
PATCH_GATE = ARTIFACTS / "unet_patch_slot1_merge_fold0_official.json"
CH3_GATE = ARTIFACTS / "unet_3ch_disk_slot1_merge_fold0_official.json"
SPINE_V2_RESNET_GATE = ARTIFACTS / "unet_spine_v2_resnet_slot1_merge_fold0_official.json"
EXTRA_FILTER_GATE = ARTIFACTS / "unet_resnet_extra_filter_slot1_merge_fold0_official.json"
EXTRA_FILTER_HGB_GATE = ARTIFACTS / "unet_resnet_extra_filter_hgb_slot1_merge_fold0_official.json"
EXTRA_FILTER_CTX_GATE = ARTIFACTS / "unet_resnet_extra_filter_ctx_slot1_merge_fold0_official.json"
EXTRA_FILTER_CROP_GATE = ARTIFACTS / "unet_resnet_extra_filter_crop_slot1_merge_fold0_official.json"
CH3_EXTRA_FILTER_GATE = ARTIFACTS / "unet_3ch_extra_filter_slot1_merge_fold0_official.json"
FILTER_WEIGHTS_NAME = "resnet_extra_filter"
FILTER_HGB_WEIGHTS_NAME = "resnet_extra_filter_hgb"
FILTER_CTX_WEIGHTS_NAME = "resnet_extra_filter_ctx"
FILTER_CROP_WEIGHTS_NAME = "resnet_extra_filter_crop"
CH3_FILTER_WEIGHTS_NAME = "ch3_extra_filter"
FORBIDDEN_WEIGHT_DIRS = (
    "unet_resnet34",
    "unet_semantic",
    "unet_patch_dhebe",
    "unet_3ch_disk",
)
HOSE_FLOOR = 642
START_RESNET = frozenset(
    {"RESEARCH_HOLD", "RESEARCH_HOLD_SPINE_V2", "KILL_WORSE_OR_DESERT"}
)
START_PATCH = frozenset({"KILL_NO_NEW_TP"})
START_PATCH_AFTER_RESNET = frozenset(
    {
        "KILL_NO_NEW_TP",
        "KILL_WORSE_OR_DESERT",
        "RESEARCH_HOLD",
        "RESEARCH_HOLD_SPINE_V2",
    }
)
PROMOTE = "PROMOTE_LOOK"


def active_merge_gate() -> Path:
    """Prefer the ResNet merge gate once it exists; else v1."""
    if RESNET_GATE.exists():
        return RESNET_GATE
    return GATE


def resolve_start_gate(path: Path | None = None) -> Path:
    """Default --gate-json=v1 GATE remaps to active_merge_gate() after ResNet."""
    if path is None or path == GATE:
        return active_merge_gate()
    return path


def pgrep(pattern: str) -> list[int]:
    """Real python running this script. Ignore zsh wrappers, pgrep, compile."""
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return []
    kept = []
    for raw in proc.stdout.split():
        if not raw.strip().isdigit():
            continue
        pid = int(raw)
        if pid == os.getpid():
            continue
        ps = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
        )
        cmd = (ps.stdout or "").strip()
        if not cmd or pattern not in cmd:
            continue
        exe = cmd.split()[0]
        if "pgrep" in exe or exe.endswith("/pgrep"):
            continue
        if exe.endswith("/zsh") or exe == "zsh" or "builtin eval" in cmd:
            continue
        if "-m py_compile" in cmd or "refresh_orchestra_status.py" in cmd:
            continue
        if "bin/python" not in cmd and "python " not in cmd:
            continue
        tokens = (
            cmd.replace("(", " ")
            .replace(")", " ")
            .replace('"', " ")
            .replace("'", " ")
            .split()
        )
        if not any(tok == pattern or tok.endswith("/" + pattern) for tok in tokens):
            continue
        kept.append(pid)
    return kept


def live_mps_owner(*, ignore: frozenset[str] | None = None) -> str | None:
    # Specific names first: pgrep -f is substring match.
    skip = ignore or frozenset()
    for pattern, label in (
        ("train_unet_resnet1536.py", "train_unet_resnet1536.py"),
        ("eval_unet_resnet1536_merge.py", "eval_unet_resnet1536_merge"),
        ("eval_unet_resnet1536.py", "eval_unet_resnet1536"),
        ("measure_resnet_prob_on_agreed_fn.py", "measure_resnet_prob_on_agreed_fn.py"),
        ("measure_resnet_cc_on_agreed_fn.py", "measure_resnet_cc_on_agreed_fn.py"),
        ("measure_3ch_resnet_xor_agreed_fn.py", "measure_3ch_resnet_xor_agreed_fn.py"),
        ("measure_existing_csv_cover_agreed_fn.py", "measure_existing_csv_cover_agreed_fn.py"),
        ("train_unet_resnet.py", "train_unet_resnet.py"),
        ("train_unet_patch.py", "train_unet_patch.py"),
        ("train_unet_3ch.py", "train_unet_3ch.py"),
        ("train_unet.py", "train_unet.py"),
        ("run_residual_family.py", "run_residual_family.py"),
        ("score_family.py", "score_family.py"),
        ("eval_erode_slot1_ctx_merge.py", "eval_erode_slot1_ctx_merge.py"),
        ("eval_slot1_resnet_box_refine.py", "eval_slot1_resnet_box_refine.py"),
        ("train_yolo.py", "train_yolo.py"),
        # Watch scripts are CPU. Counting them blocked yolo11l → erode-GT handoff.
        ("eval_unet_resnet_extra_filter_crop_merge.py", "eval_unet_resnet_extra_filter_crop_merge"),
        ("make_unet_resnet_extra_filter_crop_merged_submission.py", "make_unet_resnet_extra_filter_crop_merged_submission"),
        ("eval_unet_resnet_extra_filter_ctx_merge.py", "eval_unet_resnet_extra_filter_ctx_merge"),
        ("make_unet_resnet_extra_filter_ctx_merged_submission.py", "make_unet_resnet_extra_filter_ctx_merged_submission"),
        ("eval_unet_resnet_extra_filter_hgb_merge.py", "eval_unet_resnet_extra_filter_hgb_merge"),
        ("make_unet_resnet_extra_filter_hgb_merged_submission.py", "make_unet_resnet_extra_filter_hgb_merged_submission"),
        ("eval_unet_resnet_extra_filter_merge.py", "eval_unet_resnet_extra_filter_merge"),
        ("mine_resnet_extras.py", "mine_resnet_extras.py"),
        ("make_unet_resnet_extra_filter_merged_submission.py", "make_unet_resnet_extra_filter_merged_submission"),
        ("eval_unet_3ch_extra_filter_merge.py", "eval_unet_3ch_extra_filter_merge"),
        ("mine_3ch_extras.py", "mine_3ch_extras.py"),
        ("eval_unet_resnet_merge.py", "eval_unet_resnet_merge"),
        ("eval_unet_resnet.py", "eval_unet_resnet"),
        ("eval_unet_patch_merge.py", "eval_unet_patch_merge"),
        ("eval_unet_patch.py", "eval_unet_patch"),
        ("eval_unet_3ch_merge.py", "eval_unet_3ch_merge"),
        ("eval_unet_3ch.py", "eval_unet_3ch"),
        ("eval_unet_spine_v2_resnet_merge.py", "eval_unet_spine_v2_resnet_merge"),
        ("eval_unet_spine_v2.py", "eval_unet_spine_v2"),
        ("eval_unet_merge.py", "eval_unet_merge"),
        ("eval_unet.py", "eval_unet"),
    ):
        if pattern in skip or label in skip:
            continue
        if pgrep(pattern):
            return label
    return None


def read_gate(path: Path = GATE) -> tuple[str | None, dict]:
    if not path.exists():
        return None, {"error": f"missing {path}"}
    payload = json.loads(path.read_text())
    gate = payload.get("gate") if isinstance(payload, dict) else None
    decision = gate.get("decision") if isinstance(gate, dict) else None
    return decision, payload


def filter_dest_illegal(dest: Path) -> str | None:
    """None if dest is a new rejector dir. Never unet_resnet34 / semantic / patch / 3ch."""
    dest = Path(dest)
    parts = set(dest.parts)
    for name in FORBIDDEN_WEIGHT_DIRS:
        if name in parts:
            return f"Refusing extra-filter dest {dest}: must not overwrite {name}."
    return None


def refuse_extra_filter(
    path: Path | None = None,
    *,
    min_area: int = HOSE_FLOOR,
    filter_dest: Path | None = None,
    resnet_weights: Path | None = None,
) -> str | None:
    """Legal to mine/fit extras. No MPS check — leftover is refuse None AND MPS free."""
    if int(min_area) < HOSE_FLOOR:
        return f"Refusing extra-filter start: min_area {min_area} < {HOSE_FLOOR} (hose floor)."
    dest = Path(filter_dest) if filter_dest is not None else WEIGHTS_DIR / FILTER_WEIGHTS_NAME
    bad = filter_dest_illegal(dest)
    if bad:
        return bad
    weights = Path(resnet_weights) if resnet_weights is not None else WEIGHTS_DIR / "unet_resnet34" / "best.pt"
    if "unet_resnet34" not in Path(weights).parts:
        return f"Refusing extra-filter start: extras decoder must be unet_resnet34 (got {weights})."
    if not weights.exists():
        return f"Refusing extra-filter start: missing ResNet weights {weights}."
    gate_path = path if path is not None else RESNET_GATE
    if not Path(gate_path).exists():
        return f"Refusing extra-filter start: missing {gate_path}."
    return None


def refuse_extra_filter_csv(
    gate_path: Path | None = None,
    min_area: int = HOSE_FLOOR,
) -> str | None:
    """Test CSV only after a scored PROMOTE_LOOK. Hose floor. Never submits."""
    owner = live_mps_owner()
    if owner:
        return f"Refusing extra-filter test CSV: {owner} owns MPS."
    if int(min_area) < HOSE_FLOOR:
        return f"Refusing extra-filter test CSV: min_area {min_area} < {HOSE_FLOOR} (hose floor)."
    path = Path(gate_path) if gate_path is not None else EXTRA_FILTER_GATE
    decision, payload = read_gate(path)
    if not isinstance(payload, dict) or not payload.get("scored"):
        return f"Gate is not scored (need PROMOTE_LOOK). {path}."
    if decision != PROMOTE:
        if decision is None:
            return f"Gate is None (need PROMOTE_LOOK). {path}."
        return f"Gate is {decision!r} (need PROMOTE_LOOK)."
    return None


def refuse_3ch_extra_filter(
    path: Path | None = None,
    *,
    min_area: int = HOSE_FLOOR,
    filter_dest: Path | None = None,
    ch3_weights: Path | None = None,
) -> str | None:
    """Legal to mine/fit 3ch extras. Same hose/dest rules. Decoder is unet_3ch_disk."""
    if int(min_area) < HOSE_FLOOR:
        return f"Refusing 3ch extra-filter start: min_area {min_area} < {HOSE_FLOOR} (hose floor)."
    dest = Path(filter_dest) if filter_dest is not None else WEIGHTS_DIR / CH3_FILTER_WEIGHTS_NAME
    bad = filter_dest_illegal(dest)
    if bad:
        return bad
    if dest.name == FILTER_WEIGHTS_NAME or FILTER_WEIGHTS_NAME in dest.parts:
        return f"Refusing 3ch extra-filter dest {dest}: must not overwrite resnet_extra_filter."
    weights = Path(ch3_weights) if ch3_weights is not None else WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"
    if "unet_3ch_disk" not in Path(weights).parts:
        return f"Refusing 3ch extra-filter start: extras decoder must be unet_3ch_disk (got {weights})."
    if not weights.exists():
        return f"Refusing 3ch extra-filter start: missing 3ch weights {weights}."
    gate_path = path if path is not None else CH3_GATE
    if not Path(gate_path).exists():
        return f"Refusing 3ch extra-filter start: missing {gate_path}."
    return None


def refuse_3ch_extra_filter_csv(
    gate_path: Path | None = None,
    min_area: int = HOSE_FLOOR,
) -> str | None:
    owner = live_mps_owner()
    if owner:
        return f"Refusing 3ch extra-filter test CSV: {owner} owns MPS."
    if int(min_area) < HOSE_FLOOR:
        return f"Refusing 3ch extra-filter test CSV: min_area {min_area} < {HOSE_FLOOR} (hose floor)."
    path = Path(gate_path) if gate_path is not None else CH3_EXTRA_FILTER_GATE
    decision, payload = read_gate(path)
    if not isinstance(payload, dict) or not payload.get("scored"):
        return f"Gate is not scored (need PROMOTE_LOOK). {path}."
    if decision != PROMOTE:
        if decision is None:
            return f"Gate is None (need PROMOTE_LOOK). {path}."
        return f"Gate is {decision!r} (need PROMOTE_LOOK)."
    return None


def refuse_resnet(path: Path = GATE) -> str | None:
    owner = live_mps_owner()
    if owner:
        return f"Refusing ResNet34 start: {owner} owns MPS."
    decision, _ = read_gate(path)
    if decision is None:
        return f"Refusing ResNet34 start: missing {path}."
    if decision == PROMOTE:
        return "Refusing ResNet34 start: PROMOTE_LOOK — one merged look, no second family."
    if decision in START_PATCH:
        return "Refusing ResNet34 start: KILL_NO_NEW_TP — next family is patch U-Net, not capacity."
    if decision not in START_RESNET:
        return f"Refusing ResNet34 start: gate.decision is {decision!r}."
    return None


def refuse_patch(
    path: Path | None = None,
    *,
    ignore_owners: frozenset[str] | tuple[str, ...] = (),
) -> str | None:
    owner = live_mps_owner(ignore=frozenset(ignore_owners))
    if owner:
        return f"Refusing patch U-Net start: {owner} owns MPS."
    if path is None:
        path = active_merge_gate()
    decision, _ = read_gate(path)
    if decision is None:
        return f"Refusing patch U-Net start: missing {path}."
    if decision == PROMOTE:
        return "Refusing patch U-Net start: PROMOTE_LOOK — one merged look, no second family."
    after_resnet = path.resolve() == RESNET_GATE.resolve()
    allowed = START_PATCH_AFTER_RESNET if after_resnet else START_PATCH
    if decision not in allowed:
        need = "KILL_NO_NEW_TP or ResNet-gate HOLD/hose" if after_resnet else "KILL_NO_NEW_TP"
        return f"Refusing patch U-Net start: gate.decision is {decision!r} (need {need})."
    return None


def dispatch_plan_resnet(decision: str | None) -> dict:
    if decision == PROMOTE:
        return {
            "next_train": None,
            "public_look": "scripts/make_unet_resnet_merged_submission.py; at most one",
            "slot2": "year-shift 56148889 stays",
        }
    if decision in START_PATCH_AFTER_RESNET:
        return {
            "next_train": "unet_patch_native_1024_dhebe",
            "script": "scripts/train_unet_patch.py",
            "start_only_after": "refuse_patch() is None (ResNet train/eval gone)",
        }
    return {"next_train": None, "wait": "resnet gate missing or unknown"}


def refuse_3ch(
    path: Path | None = None,
    *,
    ignore_owners: frozenset[str] | tuple[str, ...] = (),
) -> str | None:
    """3ch starts only after the patch merge gate exists and is not PROMOTE_LOOK."""
    owner = live_mps_owner(ignore=frozenset(ignore_owners))
    if owner:
        return f"Refusing 3ch U-Net start: {owner} owns MPS."
    if path is None:
        path = PATCH_GATE
    if not path.exists():
        return f"Refusing 3ch U-Net start: missing {path}. Patch is next after ResNet."
    decision, _ = read_gate(path)
    if decision is None:
        return f"Refusing 3ch U-Net start: missing decision in {path}."
    if decision == PROMOTE:
        return "Refusing 3ch U-Net start: PROMOTE_LOOK — one merged look, no next family."
    return None


def dispatch_plan_patch(decision: str | None) -> dict:
    if decision == PROMOTE:
        return {
            "next_train": None,
            "public_look": "scripts/make_unet_patch_merged_submission.py then upload_if_patch_promote.py; at most one",
            "slot2": "year-shift 56148889 stays",
        }
    return {
        "next_train": None,
        "auto_start": False,
        "note": "Patch family closed or held. Overwatch picks the next legal representation. No auto-start. No upload.",
    }


def dispatch_plan_3ch(decision: str | None) -> dict:
    if decision == PROMOTE:
        return {
            "next_train": None,
            "public_look": "scripts/make_unet_3ch_merged_submission.py then upload_if_3ch_promote.py; at most one",
            "slot2": "year-shift 56148889 stays",
        }
    return {
        "next_train": None,
        "auto_start": False,
        "note": "3ch closed or held. Overwatch picks any leftover legal family. No auto-start. No upload.",
    }


def dispatch_plan(decision: str | None) -> dict:
    if decision == PROMOTE:
        return {
            "next_train": None,
            "public_look": "scripts/make_unet_merged_submission.py after quota reset; at most one",
            "slot2": "year-shift 56148889 stays",
        }
    if decision == "KILL_NO_NEW_TP":
        return {"next_train": "unet_patch_native_1024_dhebe", "script": "scripts/train_unet_patch.py"}
    if decision in START_RESNET:
        return {
            "next_train": "unet_resnet34_or_edgeatt_1024",
            "script": "scripts/train_unet_resnet.py",
            "parallel_if_spine": (
                "scripts/eval_unet_spine_v2.py --builder resnet34 "
                "or scripts/eval_unet_spine_v2_resnet_merge.py only if RESEARCH_HOLD_SPINE_V2"
            ),
        }
    return {"next_train": None, "wait": "gate missing or unknown"}
