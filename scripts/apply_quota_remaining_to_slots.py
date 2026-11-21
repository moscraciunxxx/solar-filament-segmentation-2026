#!/usr/bin/env python3
"""Patch hidden_exam_slots remaining_today from parsed quota. Never submits.

Waits for artifacts/kaggle_quota_remaining.json (scripts/parse_kaggle_quota.py).
Reads remaining, writes that integer to
artifacts/hidden_exam_slots.json daily_slot_budget.remaining_today, and keeps
the standing rule that looks are reserved for PROMOTE_LOOK only.

Does not call kaggle. Does not submit. Does not start train.
Does not upload faint 0.412 or merckle.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REMAINING = ARTIFACTS / "kaggle_quota_remaining.json"
SLOTS = ARTIFACTS / "hidden_exam_slots.json"
APPLIED = ARTIFACTS / "quota_remaining_applied.json"

PROMOTE_RULE = (
    "After 2026-09-11 00:00Z reset, 5 looks exist but they are reserved for a "
    "PROMOTE_LOOK family only. Standing policy: artifacts/quota_reset_look_policy.json. "
    "Do not spend on faint 0.412, merckle 0.036, U-Net standalone, conf clones, "
    "phys/tiles/all-annotator, or specialist-alone. If eval_unet_merge gate is "
    "missing or not PROMOTE_LOOK, wait; do not burn a diagnostic look."
)


def _load_json(path: Path) -> dict:
    while True:
        try:
            blob = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(2)
            continue
        if isinstance(blob, dict):
            return blob
        time.sleep(2)


def _read_remaining() -> int:
    print(f"waiting for {REMAINING}", flush=True)
    while not REMAINING.exists():
        time.sleep(10)
    while True:
        blob = _load_json(REMAINING)
        if "remaining" in blob:
            return int(blob["remaining"])
        time.sleep(2)


def main() -> None:
    remaining = _read_remaining()
    slots = _load_json(SLOTS) if SLOTS.exists() else {}
    budget = slots.setdefault("daily_slot_budget", {})
    if not isinstance(budget, dict):
        budget = {}
        slots["daily_slot_budget"] = budget
    previous = budget.get("remaining_today")
    budget["remaining_today"] = remaining
    rule = budget.get("rule")
    if not isinstance(rule, str) or "PROMOTE_LOOK" not in rule:
        budget["rule"] = PROMOTE_RULE
        rule = budget["rule"]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SLOTS.write_text(json.dumps(slots, indent=2) + "\n")
    applied = {
        "written": datetime.now(timezone.utc).isoformat(),
        "source": str(REMAINING.relative_to(ROOT)),
        "slots": str(SLOTS.relative_to(ROOT)),
        "patched_field": "daily_slot_budget.remaining_today",
        "previous_remaining_today": previous,
        "remaining": remaining,
        "remaining_today": remaining,
        "rule_kept": rule,
        "looks_reserved_for": "PROMOTE_LOOK",
        "spend_only_if": "artifacts/unet_slot1_merge_fold0_official.json decision == PROMOTE_LOOK",
        "never_spend_on": [
            "faint merge 0.412",
            "merckle 0.036",
            "U-Net standalone",
            "conf clones",
            "phys / tiles / all-annotator",
        ],
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    APPLIED.write_text(json.dumps(applied, indent=2) + "\n")
    print(json.dumps(applied, indent=2), flush=True)


if __name__ == "__main__":
    main()
