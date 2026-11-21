#!/usr/bin/env python3
"""Snapshot official public LB rank. Never submits.

Paginates `kaggle competitions leaderboard -s --format json` for
filament-segmentation-2026, finds team Vitalie Cervinschi (teamId 16865067),
writes artifacts/kaggle_lb_latest.json. Does not call kaggle competitions submit.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
KAGGLE = Path("~/.local/bin/kaggle")
COMP = "filament-segmentation-2026"
TEAM = "Vitalie Cervinschi"
TEAM_ID = "16865067"
TOKEN_RE = re.compile(r"Next Page Token\s*=\s*(\S+)")


def _fetch_page(token: str | None) -> tuple[list[dict], str | None, int]:
    cmd = [
        str(KAGGLE),
        "competitions",
        "leaderboard",
        "-c",
        COMP,
        "-s",
        "--format",
        "json",
        "--page-size",
        "200",
    ]
    if token:
        cmd.extend(["--page-token", token])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = proc.stdout or ""
    nxt = None
    m = TOKEN_RE.search(text)
    if m:
        nxt = m.group(1)
    start = text.find("[")
    rows: list[dict] = []
    if start >= 0:
        try:
            parsed = json.loads(text[start:])
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            rows = [r for r in parsed if isinstance(r, dict)]
    return rows, nxt, int(proc.returncode)


def _score(row: dict) -> float | None:
    raw = row.get("score")
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def main() -> None:
    rows: list[dict] = []
    token = None
    last_rc = 0
    for _ in range(20):
        page, token, last_rc = _fetch_page(token)
        if last_rc != 0 and not page:
            break
        rows.extend(page)
        if not token:
            break

    scored: list[tuple[float, dict]] = []
    us = None
    us_score = None
    for row in rows:
        sc = _score(row)
        if sc is None:
            continue
        scored.append((sc, row))
        if str(row.get("teamId")) == TEAM_ID or row.get("teamName") == TEAM:
            us = row
            us_score = sc
    scored.sort(key=lambda x: x[0], reverse=True)
    scores = [s for s, _ in scored]
    rank = None
    if us is not None:
        rank = next(
            (
                i + 1
                for i, (_, r) in enumerate(scored)
                if str(r.get("teamId")) == TEAM_ID or r.get("teamName") == TEAM
            ),
            None,
        )
    goal_ok = rank is not None and rank <= 10
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "source": "kaggle competitions leaderboard -s --format json filament-segmentation-2026",
        "n_teams": len(scored),
        "live_returncode": last_rc,
        "us": {
            "rank": rank,
            "team": TEAM,
            "teamId": int(TEAM_ID),
            "public": us_score,
        },
        "top10": scores[:10],
        "top10_are_0.55_cluster": bool(scores) and scores[min(9, len(scores) - 1)] >= 0.55,
        "n_ge_0.55": sum(s >= 0.55 for s in scores),
        "first_le_0.46_rank": next((i + 1 for i, s in enumerate(scores) if s <= 0.46), None),
        "n_ge_0.46": sum(s >= 0.46 for s in scores),
        "n_ge_0.40": sum(s >= 0.40 for s in scores),
        "do_not_chase_0.55": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "goal_top10": "verified" if goal_ok else "unproven",
        "rank_le_10": goal_ok,
        "current_objective": (
            "verified rank <=10."
            if goal_ok
            else "verified rank <=10. Unproven. Do not upload until PROMOTE_LOOK."
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "kaggle_lb_latest.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    stamp = ARTIFACTS / f"kaggle_lb_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%MZ')}.json"
    stamp.write_text(json.dumps(out, indent=2) + "\n")
    verified = ARTIFACTS / "quota_reset_verified.json"
    if verified.exists():
        try:
            blob = json.loads(verified.read_text())
            blob["rank"] = rank
            blob["public"] = us_score
            blob["rank_le_10"] = goal_ok
            blob["goal_top10"] = out["goal_top10"]
            blob["confirm_returncode"] = 0 if scored else 1
            blob["lb"] = str(dest.relative_to(ROOT))
            verified.write_text(json.dumps(blob, indent=2) + "\n")
        except (OSError, json.JSONDecodeError):
            pass
    print(json.dumps({"path": str(dest), "rank": rank, "public": us_score, "rank_le_10": goal_ok}, indent=2))
    if not scored:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
