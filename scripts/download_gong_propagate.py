#!/usr/bin/env python3
"""Download GONG H-alpha JPEGs 1-6 hours from each train observation.

The nearest archive frame to each whole hour is kept. Any frame within
one day of a test observation is skipped and not downloaded. Does not
read test labels or MAGFiLO ground truth.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import TEST_IMG_DIR, TRAIN_IMG_DIR

BASE = "https://gong2.nso.edu/HA/hag"
OUT = ROOT / "data" / "gong_propagate"
JPG_DIR = OUT / "jpg"
INDEX_DIR = OUT / "day_index"
MANIFEST = ROOT / "artifacts" / "gong_propagate_manifest.jsonl"
OFFSETS = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
TOLERANCE = timedelta(minutes=8)
EXCLUSION = timedelta(days=1)
WORKERS = 6
UA = {"User-Agent": "filament-research"}


def parse_stamp(name: str) -> datetime | None:
    stem = Path(name).stem
    if len(stem) < 14 or not stem[:14].isdigit():
        return None
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def site_of(name: str) -> str:
    stem = Path(name).stem
    if len(stem) > 15 and stem.endswith("h"):
        return stem[14:-1]
    return ""


def day_url(day: datetime) -> str:
    month = day.strftime("%Y%m")
    return f"{BASE}/{month}/{day.strftime('%Y%m%d')}/"


def file_url(when: datetime, filename: str) -> str:
    return f"{day_url(when)}{filename}"


def list_day(day: datetime) -> list[tuple[datetime, str, str]]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = INDEX_DIR / f"{day.strftime('%Y%m%d')}.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        return [
            (datetime.fromisoformat(item["time"]), item["site"], item["name"])
            for item in raw
        ]
    try:
        req = urllib.request.Request(day_url(day), headers=UA)
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    except Exception:
        cache.write_text("[]\n")
        return []
    found = []
    for name in re.findall(r'href="(\d{14}[A-Za-z]h\.jpg)"', html):
        when = parse_stamp(name)
        if when is None:
            continue
        found.append({"time": when.isoformat(), "site": site_of(name), "name": name})
    found.sort(key=lambda item: item["time"])
    cache.write_text(json.dumps(found))
    return [
        (datetime.fromisoformat(item["time"]), item["site"], item["name"])
        for item in found
    ]


def nearest(frames: list[tuple[datetime, str, str]], target: datetime):
    best = None
    best_gap = None
    for when, site, name in frames:
        gap = abs(when - target)
        if gap > TOLERANCE:
            continue
        if best_gap is None or gap < best_gap:
            best = (when, site, name)
            best_gap = gap
    return best, best_gap


def excluded(when: datetime, test_times: list[datetime]) -> str | None:
    for test_time in test_times:
        if abs(when - test_time) <= EXCLUSION:
            return test_time.strftime("%Y%m%d%H%M%S")
    return None


def download(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest.stat().st_size
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as response:
        data = response.read()
    dest.write_bytes(data)
    return len(data)


def main() -> None:
    JPG_DIR.mkdir(parents=True, exist_ok=True)
    train_files = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpeg"))
    test_files = sorted(p.name for p in TEST_IMG_DIR.glob("*.jpeg"))
    test_times = [t for name in test_files if (t := parse_stamp(name))]
    jobs = []
    for name in train_files:
        origin = parse_stamp(name)
        if origin is None:
            continue
        for hour in OFFSETS:
            target = origin + timedelta(hours=hour)
            reason = excluded(target, test_times)
            jobs.append(
                {
                    "train_file": name,
                    "train_time": origin.isoformat(),
                    "offset_hours": hour,
                    "target_time": target.isoformat(),
                    "excluded_by_test": reason,
                }
            )
    n_excluded = sum(1 for job in jobs if job["excluded_by_test"])
    print(
        f"train {len(train_files)} test {len(test_files)} targets {len(jobs)} "
        f"excluded_near_test {n_excluded}",
        flush=True,
    )
    days = sorted({datetime.fromisoformat(job["target_time"]).date() for job in jobs if not job["excluded_by_test"]})
    print(f"days_to_index {len(days)}", flush=True)
    day_frames: dict = {}
    lock = threading.Lock()
    manifest_lock = threading.Lock()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done_keys.add((row["train_file"], row["offset_hours"]))

    def index_one(day):
        when = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        frames = list_day(when)
        with lock:
            day_frames[day] = frames
        return day, len(frames)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(index_one, day) for day in days]
        for count, future in enumerate(as_completed(futures), start=1):
            day, n = future.result()
            if count % 50 == 0 or count == len(days):
                print(f"indexed {count}/{len(days)} {day} frames {n}", flush=True)

    def one(job):
        key = (job["train_file"], job["offset_hours"])
        if key in done_keys:
            return "skip"
        row = dict(job)
        row["test_labels_used"] = False
        if job["excluded_by_test"]:
            row["status"] = "excluded_near_test"
            with manifest_lock:
                with MANIFEST.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            return "excluded"
        target = datetime.fromisoformat(job["target_time"])
        frames = day_frames.get(target.date(), [])
        chosen, gap = nearest(frames, target)
        if chosen is None:
            row["status"] = "no_frame_within_8min"
            with manifest_lock:
                with MANIFEST.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            return "miss"
        when, site, filename = chosen
        url = file_url(when, filename)
        dest = JPG_DIR / filename
        try:
            nbytes = download(url, dest)
        except Exception as exc:
            row["status"] = "download_error"
            row["error"] = type(exc).__name__
            with manifest_lock:
                with MANIFEST.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            return "error"
        row.update(
            {
                "status": "downloaded",
                "gong_file": filename,
                "gong_time": when.isoformat(),
                "gong_site": site,
                "gap_seconds": gap.total_seconds(),
                "url": url,
                "bytes": nbytes,
                "path": str(dest),
            }
        )
        with manifest_lock:
            with MANIFEST.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return "ok"

    counts = {"ok": 0, "miss": 0, "excluded": 0, "error": 0, "skip": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            counts[future.result()] += 1
            if count % 200 == 0 or count == len(jobs):
                print(f"progress {count}/{len(jobs)} {json.dumps(counts)}", flush=True)
    print("DONE gong_download", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
