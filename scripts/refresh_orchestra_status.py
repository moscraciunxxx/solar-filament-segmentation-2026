#!/usr/bin/env python3
"""Write artifacts/orchestra_status.json from live pgrep + curves. Never trains/submits."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

WATCH = [
    ("scripts/train_unet.py", "MPS TinyUNet 40ep"),
    ("scripts/watch_unet_then_score.py", "score standalone then Slot-1 merge after train"),
    ("scripts/wait_quota_reset.py", "done after 00:05Z; absent is ok"),
    ("scripts/watch_quota_then_parse.py", "done after parse; absent is ok"),
    ("scripts/apply_quota_remaining_to_slots.py", "done after patch; absent is ok"),
    ("scripts/dispatch_after_unet_gate.py", "record dispatch after merge gate exists"),
    ("scripts/promote_look_if_ready.py", "build CSV only if gate is PROMOTE_LOOK"),
    ("scripts/watch_dispatch_then_announce.py", "write overwatch_next_action after dispatch"),
    ("scripts/watch_promote_then_announce.py", "announce PROMOTE CSV; never upload"),
    ("scripts/wait_upload_then_confirm_rank.py", "snapshot rank after a real --upload"),
    ("scripts/train_unet_resnet.py", "MPS ResNet34 40ep after v1 KILL_WORSE_OR_DESERT"),
    ("scripts/watch_resnet_then_score.py", "score ResNet34 standalone then Slot-1 merge"),
    ("scripts/dispatch_after_resnet_gate.py", "record dispatch after ResNet merge gate"),
    ("scripts/promote_resnet_look_if_ready.py", "build ResNet CSV only if PROMOTE_LOOK"),
    ("scripts/watch_resnet_dispatch_then_announce.py", "write overwatch_resnet_next_action"),
    ("scripts/watch_resnet_promote_then_dry.py", "dry ResNet upload only; never --upload"),
    ("scripts/train_unet_patch.py", "MUST be absent unless refuse_patch() is None"),
    ("scripts/train_unet_3ch.py", "MUST be absent until patch merge exists and is not PROMOTE"),
    ("scripts/watch_unet_3ch_then_score.py", "absent until 3ch train starts"),
    ("scripts/watch_patch_then_score.py", "absent until patch train starts"),
    ("scripts/record_patch_start_ready.py", "record may_start; never starts train"),
    ("scripts/dispatch_after_patch_gate.py", "record dispatch after patch merge gate"),
    ("scripts/promote_patch_look_if_ready.py", "build patch CSV only if PROMOTE_LOOK"),
    ("scripts/watch_patch_dispatch_then_announce.py", "write overwatch_patch_next_action"),
    ("scripts/watch_patch_promote_then_dry.py", "dry patch upload only; never --upload"),
    ("scripts/dispatch_after_3ch_gate.py", "record dispatch after 3ch merge gate"),
    ("scripts/promote_3ch_look_if_ready.py", "build 3ch CSV only if PROMOTE_LOOK"),
    ("scripts/watch_3ch_dispatch_then_announce.py", "write overwatch_3ch_next_action"),
    ("scripts/watch_3ch_promote_then_dry.py", "dry 3ch upload only; never --upload"),
]

EXPECTED_ALIVE = {
    "scripts/train_unet_patch.py",
    "scripts/watch_patch_then_score.py",
    "scripts/wait_upload_then_confirm_rank.py",
    "scripts/dispatch_after_patch_gate.py",
    "scripts/promote_patch_look_if_ready.py",
    "scripts/watch_patch_dispatch_then_announce.py",
    "scripts/watch_patch_promote_then_dry.py",
    "scripts/dispatch_after_3ch_gate.py",
    "scripts/promote_3ch_look_if_ready.py",
    "scripts/watch_3ch_dispatch_then_announce.py",
    "scripts/watch_3ch_promote_then_dry.py",
}
# train_unet_resnet.py / watch_resnet_then_score.py are done (HOLD_SPINE_V2).
# train_unet.py absent is success after v1 closed.

STEAL = {
    "scripts/train_unet.py",
    "scripts/train_unet_resnet.py",
    "scripts/train_unet_3ch.py",
}


def pgrep(pattern: str) -> tuple[list[int], str | None]:
    """Match python running this script. Ignore compile/refresh wrappers that quote the name."""
    proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        err = (proc.stderr or "").strip() or f"pgrep rc={proc.returncode}"
        return [], err
    pids = [int(x) for x in proc.stdout.split() if x.strip().isdigit()]
    kept = []
    for pid in pids:
        ps = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
        cmd = (ps.stdout or "").strip()
        if not cmd or pattern not in cmd:
            continue
        if "-m py_compile" in cmd or "refresh_orchestra_status.py" in cmd:
            continue
        if "bin/python" not in cmd and "python " not in cmd:
            continue
        kept.append(pid)
    return kept, None


def main() -> None:
    processes = {}
    watchers_down = []
    steal = []
    pgrep_errors = []
    for script, role in WATCH:
        pids, err = pgrep(Path(script).name)
        if err:
            pgrep_errors.append({"script": script, "error": err})
        status = "alive" if pids else "absent"
        processes[script] = {"status": status, "pids": pids, "role": role}
        if script in EXPECTED_ALIVE and not pids:
            watchers_down.append(script)
        if script == "scripts/train_unet.py" and not pids:
            meta_done = (ARTIFACTS / "unet_train_meta.json").exists()
            if not meta_done:
                watchers_down.append(script)
        if script in STEAL and pids:
            steal.append(script)
    pgrep_ok = not pgrep_errors
    if not pgrep_ok:
        # Sandbox pgrep returns empty and "Cannot get process list". Do not
        # treat that as watchers_down or overwatch will restart live jobs.
        watchers_down = []
        steal = []

    def _curve(name: str) -> dict:
        cpath = ARTIFACTS / name
        if not cpath.exists():
            return {}
        blob = json.loads(cpath.read_text())
        hist = blob.get("history") or []
        return {
            "path": f"artifacts/{name}",
            "best_epoch": blob.get("best_epoch"),
            "last_epoch": hist[-1]["epoch"] if hist else None,
            "best_val_loss": blob.get("best_val_loss"),
            "val_loss_is_not_official_pq": True,
        }

    curve = _curve("unet_train_curve.json")
    resnet_curve = _curve("unet_resnet34_train_curve.json")

    gate = ARTIFACTS / "unet_slot1_merge_fold0_official.json"
    remaining = None
    rem_path = ARTIFACTS / "kaggle_quota_remaining.json"
    if rem_path.exists():
        remaining = json.loads(rem_path.read_text()).get("remaining")

    lb = ARTIFACTS / "kaggle_lb_latest.json"
    if not lb.exists():
        lb = ARTIFACTS / "kaggle_lb_2026-09-10T2344Z.json"
    public = {}
    if lb.exists():
        blob = json.loads(lb.read_text())
        us = blob.get("us") or {}
        public = {
            "source": str(lb.relative_to(ROOT)),
            "rank": us.get("rank"),
            "n_teams": blob.get("n_teams"),
            "public": us.get("public"),
            "team": us.get("team"),
        }

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "role": "cpu_only_overwatch",
        "competition": "filament-segmentation-2026",
        "do_not_submit": True,
        "do_not": [
            "kill any process",
            "start train/eval/submit while v1 owns MPS",
            "kaggle competitions submit unless PROMOTE_LOOK + --upload",
        ],
        "pgrep_ok": pgrep_ok,
        "pgrep_errors": pgrep_errors,
        "watchers_down": watchers_down,
        "mps_steal_detected": steal,
        "all_expected_alive": pgrep_ok and not watchers_down and not steal,
        "processes": processes,
        "unet_train_curve": curve,
        "unet_resnet34_train_curve": resnet_curve,
        "unet_slot1_merge_fold0_official.json": gate.exists(),
        "unet_resnet34_slot1_merge_fold0_official.json": (
            ARTIFACTS / "unet_resnet34_slot1_merge_fold0_official.json"
        ).exists(),
        "quota_remaining": remaining,
        "public_rank": public,
        "next_real_number": "official fold-0 merge PQ after train ends",
        "kaggle_submit_called": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "orchestra_status.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
