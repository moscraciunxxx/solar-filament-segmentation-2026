#!/usr/bin/env python3
"""Decide whether a full-medoid fold score may be uploaded. Default is dry.

Upload only when the fold JSON says promote and the public forecast is
above 0.6. A public score counts only when two reads of the same Kaggle
ref are equal and above 0.6. Does not submit unless --upload is passed,
and this dry entry never passes it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "artifacts" / "stream_lfull_2048_fold0.json"
PUBLIC_BAR = 0.6


def upload_decision(score: dict | None) -> str:
    if not score or "public_forecast" not in score or "promote" not in score:
        return "await_score"
    forecast = float(score["public_forecast"])
    if score["promote"] is True and forecast > PUBLIC_BAR:
        return "upload"
    return "start_m2"


def _public_score(rows: list[dict], ref: int) -> float | None:
    for row in rows:
        if int(row.get("ref", -1)) != int(ref):
            continue
        if row.get("status") != "COMPLETE":
            return None
        raw = row.get("publicScore")
        if raw in (None, ""):
            return None
        return float(raw)
    return None


def confirmed_public_score(first: list[dict], second: list[dict], ref: int) -> float | None:
    left = _public_score(first, ref)
    right = _public_score(second, ref)
    if left is None or right is None or left != right or not left > PUBLIC_BAR:
        return None
    return left


def main() -> None:
    score = json.loads(SCORE.read_text()) if SCORE.is_file() else None
    decision = upload_decision(score)
    print(json.dumps({"decision": decision, "kaggle_submit_called": False, "score_file": str(SCORE)}))
    if decision != "upload":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
