#!/usr/bin/env python3
"""Exhaustive EDA for MAGFiLO 1.0 Kaggle 2026 (train annotations + train/test jpegs).

Uses stdlib only (json, pathlib, collections, math, statistics, re, struct).
Does not load any leaked public-test labels.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path(
    "."
    "/data/MAGFiLO_1.0_Kaggle_2026"
)
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMG_DIR = DATA_ROOT / "test" / "test_images"
OUT_JSON = Path(
    "."
    "/artifacts/eda_stats.json"
)

FN_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})([A-Za-z]{2})\.jpe?g$", re.I)
ID_RE = re.compile(r"^(\d{6})-(.+)$")
STATION_NAME = {
    "Bh": "Big Bear (GONG)",
    "Ch": "Cerro Tololo (GONG)",
    "Lh": "Learmonth (GONG)",
    "Mh": "Mauna Loa (GONG)",
    "Th": "Teide / Tenerife (GONG)",
    "Uh": "Udaipur (GONG)",
}


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(sorted_vals) - 1)
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def summarize_numeric(vals):
    if not vals:
        return {"n": 0}
    s = sorted(float(x) for x in vals)
    return {
        "n": len(s),
        "min": s[0],
        "p10": percentile(s, 10),
        "p25": percentile(s, 25),
        "median": percentile(s, 50),
        "p75": percentile(s, 75),
        "p90": percentile(s, 90),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


def parse_filename(name):
    m = FN_RE.match(name)
    if not m:
        return None
    y, mo, d, h, mi, se, st = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(se))
    except ValueError:
        dt = None
    return {
        "year": int(y),
        "month": int(mo),
        "day": int(d),
        "hour": int(h),
        "minute": int(mi),
        "second": int(se),
        "station": st,
        "datetime_ok": dt is not None,
        "iso": dt.isoformat(sep=" ") if dt else None,
    }


def jpeg_size(path):
    """Read SOF0/2 dimensions without PIL. Returns (w, h) or None."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                while marker[1] == 0xFF:
                    marker = b"\xff" + f.read(1)
                    if len(marker) < 2:
                        return None
                m = marker[1]
                if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:
                    continue
                seglen = f.read(2)
                if len(seglen) < 2:
                    return None
                length = struct.unpack(">H", seglen)[0]
                if length < 2:
                    return None
                payload = f.read(length - 2)
                if len(payload) < length - 2:
                    return None
                if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    if len(payload) < 5:
                        return None
                    h, w = struct.unpack(">HH", payload[1:5])
                    return int(w), int(h)
    except OSError:
        return None


def shoelace(coords):
    if len(coords) < 6 or len(coords) % 2:
        return None
    n = len(coords) // 2
    area = 0.0
    for i in range(n):
        x1, y1 = coords[2 * i], coords[2 * i + 1]
        x2, y2 = coords[2 * ((i + 1) % n)], coords[2 * ((i + 1) % n) + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def poly_bounds(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0.0, xb - xa), max(0.0, yb - ya)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def segments_intersect(a, b, c, d, eps=1e-9):
    """Proper intersection of segments ab and cd (shared vertices ignored)."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if abs(o1) < eps or abs(o2) < eps or abs(o3) < eps or abs(o4) < eps:
        return False
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        if a == c or a == d or b == c or b == d:
            return False
        return True
    return False


def count_self_intersections(coords, max_edges=80):
    n = len(coords) // 2
    if n < 4:
        return 0
    pts = [(coords[2 * i], coords[2 * i + 1]) for i in range(n)]
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    if n > max_edges:
        step = max(1, n // max_edges)
        idxs = list(range(0, n, step))
        edges = [(pts[i], pts[(i + 1) % n]) for i in idxs]
    hits = 0
    for i in range(len(edges)):
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                hits += 1
                if hits >= 3:
                    return hits
    return hits


def point_in_poly(x, y, coords):
    n = len(coords) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = coords[2 * i], coords[2 * i + 1]
        xj, yj = coords[2 * j], coords[2 * j + 1]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def raster_poly_iou(coords_a, coords_b, target_cells=12000):
    """Approximate polygon IoU via even-odd sampling on a regular grid."""
    ax0, ay0, ax1, ay1 = poly_bounds(coords_a)
    bx0, by0, bx1, by1 = poly_bounds(coords_b)
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    union_x0, union_y0 = min(ax0, bx0), min(ay0, by0)
    union_x1, union_y1 = max(ax1, bx1), max(ay1, by1)
    uw = max(union_x1 - union_x0, 1e-6)
    uh = max(union_y1 - union_y0, 1e-6)
    aspect = uw / uh
    ny = max(8, int(round(math.sqrt(target_cells / aspect))))
    nx = max(8, int(round(target_cells / ny)))
    dx, dy = uw / nx, uh / ny
    inter = only_a = only_b = 0
    if x1 <= x0 or y1 <= y0:
        # disjoint bboxes: still estimate union via own areas
        for iy in range(ny):
            y = union_y0 + (iy + 0.5) * dy
            for ix in range(nx):
                x = union_x0 + (ix + 0.5) * dx
                ina = point_in_poly(x, y, coords_a)
                inb = point_in_poly(x, y, coords_b)
                if ina and inb:
                    inter += 1
                elif ina:
                    only_a += 1
                elif inb:
                    only_b += 1
    else:
        for iy in range(ny):
            y = union_y0 + (iy + 0.5) * dy
            for ix in range(nx):
                x = union_x0 + (ix + 0.5) * dx
                ina = point_in_poly(x, y, coords_a)
                inb = point_in_poly(x, y, coords_b)
                if ina and inb:
                    inter += 1
                elif ina:
                    only_a += 1
                elif inb:
                    only_b += 1
    denom = inter + only_a + only_b
    return (inter / denom) if denom else 0.0


def greedy_match(items_a, items_b, iou_fn, thresh=0.1):
    """Greedy 1-1 match maximizing pairwise IoU."""
    pairs = []
    for i, a in enumerate(items_a):
        for j, b in enumerate(items_b):
            pairs.append((iou_fn(a, b), i, j))
    pairs.sort(reverse=True)
    used_a, used_b = set(), set()
    matches = []
    for iou, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        if iou < thresh:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append({"i": i, "j": j, "iou": iou})
    unmatched_a = [i for i in range(len(items_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(items_b)) if j not in used_b]
    return matches, unmatched_a, unmatched_b


def dist_table(counter):
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def main():
    print("Loading train JSON...", flush=True)
    with TRAIN_JSON.open() as f:
        coco = json.load(f)

    train_files = sorted(p.name for p in TRAIN_IMG_DIR.glob("*") if p.is_file())
    test_files = sorted(p.name for p in TEST_IMG_DIR.glob("*") if p.is_file())
    train_set, test_set = set(train_files), set(test_files)

    info = coco.get("info", {})
    licenses = coco.get("licenses")
    categories = coco.get("categories", [])
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    cat_by_id = {c["id"]: c for c in categories}

    # ---- schema ----
    image_key_sets = Counter(tuple(sorted(im.keys())) for im in images)
    ann_key_sets = Counter(tuple(sorted(a.keys())) for a in annotations)
    image_field_types = {}
    if images:
        for k, v in images[0].items():
            image_field_types[k] = type(v).__name__
    ann_field_types = {}
    if annotations:
        for k, v in annotations[0].items():
            ann_field_types[k] = type(v).__name__

    # ---- images / annotators ----
    image_ids = [im["id"] for im in images]
    dup_image_ids = [i for i, n in Counter(image_ids).items() if n > 1]
    json_filenames = [im["file_name"] for im in images]
    unique_physical = sorted(set(json_filenames))
    extra_json_not_on_disk = sorted(set(json_filenames) - train_set)
    extra_disk_not_in_json = sorted(train_set - set(json_filenames))

    id_parse_fail = []
    annotator_codes = []
    annotator_aa = Counter()
    annotator_bb = Counter()
    annotator_cc = Counter()
    physical_to_ann_ids = defaultdict(list)
    physical_to_codes = defaultdict(set)
    code_to_physicals = defaultdict(set)
    id_stem_mismatch = []
    heights = Counter()
    widths = Counter()
    licenses_used = Counter()
    date_parse_fail = []
    url_exts = Counter()

    for im in images:
        heights[im.get("height")] += 1
        widths[im.get("width")] += 1
        licenses_used[im.get("license")] += 1
        url = im.get("url") or ""
        url_exts[Path(url).suffix.lower() or "(none)"] += 1
        m = ID_RE.match(str(im["id"]))
        stem = Path(im["file_name"]).stem
        if not m:
            id_parse_fail.append(im["id"])
            code = None
        else:
            code, id_stem = m.group(1), m.group(2)
            if id_stem != stem:
                id_stem_mismatch.append({"id": im["id"], "file_name": im["file_name"]})
            annotator_codes.append(code)
            annotator_aa[code[:2]] += 1
            annotator_bb[code[2:4]] += 1
            annotator_cc[code[4:6]] += 1
        physical_to_ann_ids[im["file_name"]].append(im["id"])
        if code:
            physical_to_codes[im["file_name"]].add(code)
            code_to_physicals[code].add(im["file_name"])
        parsed = parse_filename(im["file_name"])
        if parsed is None or not parsed["datetime_ok"]:
            date_parse_fail.append(im["file_name"])

    n_ann_ids_per_physical = Counter(len(v) for v in physical_to_ann_ids.values())
    n_codes_per_physical = Counter(len(v) for v in physical_to_codes.values())
    physicals_by_n_annot = defaultdict(list)
    for fn, codes in physical_to_codes.items():
        physicals_by_n_annot[len(codes)].append(fn)

    # same physical, codes differ only in last 2 digits?
    suffix_pattern = Counter()
    mixed_prefix_same_file = 0
    for fn, codes in physical_to_codes.items():
        if len(codes) < 2:
            continue
        prefs = {c[:4] for c in codes}
        if len(prefs) == 1:
            suffix_pattern["same_AABB_diff_CC"] += 1
        else:
            suffix_pattern["mixed_AABB_on_same_file"] += 1
            mixed_prefix_same_file += 1

    # ---- annotations ----
    ann_ids = [a["id"] for a in annotations]
    dup_ann_ids = [i for i, n in Counter(ann_ids).items() if n > 1]
    anns_by_image = defaultdict(list)
    orphan_image_ids = []
    image_id_set = set(image_ids)
    cat_counts = Counter()
    iscrowd_counts = Counter()
    nseg_counts = Counter()
    empty_segs = []
    odd_len_segs = []
    short_polys = []
    zero_area_reported = []
    area_vs_shoelace = []
    bbox_vs_poly = []
    oob_coords = []
    self_x = []
    multi_poly = []
    spine_lens = []
    spine_missing = 0
    areas = []
    bbox_ws, bbox_hs, bbox_diags, bbox_areas = [], [], [], []
    n_vertices = []
    cat_by_physical = Counter()

    img_by_id = {im["id"]: im for im in images}

    print("Scanning annotations...", flush=True)
    for a in annotations:
        iid = a.get("image_id")
        if iid not in image_id_set:
            orphan_image_ids.append(a.get("id"))
        anns_by_image[iid].append(a)
        cat_counts[a.get("category_id")] += 1
        iscrowd_counts[a.get("iscrowd")] += 1
        segs = a.get("segmentation")
        if not segs:
            empty_segs.append(a.get("id"))
            nseg_counts[0] += 1
            continue
        nseg_counts[len(segs)] += 1
        if len(segs) > 1:
            multi_poly.append(a.get("id"))
        if a.get("area") == 0:
            zero_area_reported.append(a.get("id"))
        spine = a.get("spine")
        if not spine:
            spine_missing += 1
        else:
            spine_lens.append(len(spine) if isinstance(spine, list) else -1)

        im = img_by_id.get(iid, {})
        iw, ih = im.get("width") or 2048, im.get("height") or 2048
        poly_area_sum = 0.0
        all_xs, all_ys = [], []
        verts = 0
        for si, seg in enumerate(segs):
            if not isinstance(seg, (list, tuple)):
                empty_segs.append(a.get("id"))
                continue
            if len(seg) == 0:
                empty_segs.append(a.get("id"))
                continue
            if len(seg) % 2:
                odd_len_segs.append({"id": a.get("id"), "poly": si, "n": len(seg)})
                continue
            if len(seg) < 6:
                short_polys.append({"id": a.get("id"), "poly": si, "n": len(seg)})
            verts += len(seg) // 2
            sa = shoelace(seg)
            if sa is not None:
                poly_area_sum += sa
            xs, ys = seg[0::2], seg[1::2]
            all_xs.extend(xs)
            all_ys.extend(ys)
            if any(x < -1 or x > iw + 1 or y < -1 or y > ih + 1 for x, y in zip(xs, ys)):
                oob_coords.append(a.get("id"))
            if len(seg) >= 8:
                hits = count_self_intersections(seg)
                if hits:
                    self_x.append({"id": a.get("id"), "hits": hits, "n_verts": len(seg) // 2})

        n_vertices.append(verts)
        reported = float(a.get("area") or 0)
        areas.append(reported)
        if poly_area_sum > 0:
            rel = abs(reported - poly_area_sum) / max(reported, poly_area_sum, 1e-9)
            area_vs_shoelace.append(rel)

        bbox = a.get("bbox") or [0, 0, 0, 0]
        if len(bbox) == 4:
            x, y, w, h = [float(t) for t in bbox]
            bbox_ws.append(w)
            bbox_hs.append(h)
            bbox_diags.append(math.hypot(w, h))
            bbox_areas.append(max(w, 0) * max(h, 0))
            if all_xs and all_ys:
                px0, py0, px1, py1 = min(all_xs), min(all_ys), max(all_xs), max(all_ys)
                pw, ph = px1 - px0, py1 - py0
                if abs(x - px0) > 2 or abs(y - py0) > 2 or abs(w - pw) > 3 or abs(h - ph) > 3:
                    bbox_vs_poly.append(a.get("id"))

        fn = im.get("file_name")
        if fn:
            cat_by_physical[(fn, a.get("category_id"))] += 1

    empty_ann_images = [iid for iid in image_ids if iid not in anns_by_image]
    anns_per_ann_image = [len(anns_by_image[iid]) for iid in image_ids]
    anns_per_physical = []
    filaments_median_per_physical = {}
    filaments_mean_per_physical = {}
    year_per_physical = {}
    station_per_physical = {}
    for fn, iids in physical_to_ann_ids.items():
        counts = [len(anns_by_image[i]) for i in iids]
        anns_per_physical.append(sum(counts))
        sc = sorted(counts)
        filaments_median_per_physical[fn] = percentile(sc, 50)
        filaments_mean_per_physical[fn] = sum(counts) / len(counts)
        parsed = parse_filename(fn)
        year_per_physical[fn] = parsed["year"] if parsed else None
        station_per_physical[fn] = parsed["station"] if parsed else None

    # ---- train filename encoding ----
    train_parsed, train_unparsed = [], []
    train_stations, train_years, train_months, train_hours = Counter(), Counter(), Counter(), Counter()
    for fn in train_files:
        p = parse_filename(fn)
        if p is None:
            train_unparsed.append(fn)
            continue
        train_parsed.append((fn, p))
        train_stations[p["station"]] += 1
        train_years[p["year"]] += 1
        train_months[p["month"]] += 1
        train_hours[p["hour"]] += 1

    test_parsed, test_unparsed = [], []
    test_stations, test_years, test_months, test_hours = Counter(), Counter(), Counter(), Counter()
    for fn in test_files:
        p = parse_filename(fn)
        if p is None:
            test_unparsed.append(fn)
            continue
        test_parsed.append((fn, p))
        test_stations[p["station"]] += 1
        test_years[p["year"]] += 1
        test_months[p["month"]] += 1
        test_hours[p["hour"]] += 1

    overlap = sorted(train_set & test_set)
    train_stems = {Path(f).stem for f in train_files}
    test_stems = {Path(f).stem for f in test_files}
    stem_overlap = sorted(train_stems & test_stems)

    def shift_props(train_c, test_c):
        keys = sorted(set(train_c) | set(test_c))
        nt, nte = sum(train_c.values()) or 1, sum(test_c.values()) or 1
        rows = []
        for k in keys:
            pt, pe = train_c[k] / nt, test_c[k] / nte
            rows.append(
                {
                    "key": k,
                    "train_n": train_c[k],
                    "test_n": test_c[k],
                    "train_p": round(pt, 4),
                    "test_p": round(pe, 4),
                    "delta_pp": round((pe - pt) * 100, 2),
                }
            )
        rows.sort(key=lambda r: -abs(r["delta_pp"]))
        return rows

    year_shift = shift_props(train_years, test_years)
    station_shift = shift_props(train_stations, test_stations)

    # JPEG headers (sample all; SOF is cheap)
    print("Reading JPEG SOF headers...", flush=True)
    train_jpeg_sizes = Counter()
    test_jpeg_sizes = Counter()
    train_bad_jpeg = []
    test_bad_jpeg = []
    train_size_mismatch = []
    for fn in train_files:
        sz = jpeg_size(TRAIN_IMG_DIR / fn)
        if sz is None:
            train_bad_jpeg.append(fn)
        else:
            train_jpeg_sizes[f"{sz[0]}x{sz[1]}"] += 1
            # compare vs any json record
            recs = [im for im in images if im["file_name"] == fn]
            if recs and (recs[0]["width"], recs[0]["height"]) != sz:
                train_size_mismatch.append({"file": fn, "jpeg": sz, "json": (recs[0]["width"], recs[0]["height"])})
    for fn in test_files:
        sz = jpeg_size(TEST_IMG_DIR / fn)
        if sz is None:
            test_bad_jpeg.append(fn)
        else:
            test_jpeg_sizes[f"{sz[0]}x{sz[1]}"] += 1

    # ---- sample 3 physical images for agreement ----
    print("Computing annotator agreement on 3 samples...", flush=True)
    candidates = []
    for fn, codes in physical_to_codes.items():
        n = len(codes)
        med = filaments_median_per_physical.get(fn, 0) or 0
        candidates.append((n, med, year_per_physical.get(fn) or 0, fn))
    # prefer 3, then 2, then 1; spread years
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    picked = []
    used_n = set()
    used_years = set()
    for n, med, y, fn in candidates:
        key = (n, y)
        if n >= 2 and n not in used_n:
            picked.append(fn)
            used_n.add(n)
            used_years.add(y)
        if len(picked) >= 3:
            break
    if len(picked) < 3:
        for n, med, y, fn in candidates:
            if fn not in picked:
                picked.append(fn)
            if len(picked) >= 3:
                break

    def first_poly(ann):
        segs = ann.get("segmentation") or []
        return segs[0] if segs else None

    sample_reports = []
    for fn in picked:
        iids = physical_to_ann_ids[fn]
        per_ann = []
        for iid in iids:
            code = iid.split("-", 1)[0]
            anns = anns_by_image.get(iid, [])
            per_ann.append(
                {
                    "image_id": iid,
                    "annotator_code": code,
                    "n_filaments": len(anns),
                    "category_counts": dist_table(Counter(a.get("category_id") for a in anns)),
                    "areas": [float(a.get("area") or 0) for a in anns],
                }
            )
        agreement = []
        # pairwise if 2+
        if len(iids) >= 2:
            for ia in range(len(iids)):
                for ib in range(ia + 1, len(iids)):
                    A = anns_by_image[iids[ia]]
                    B = anns_by_image[iids[ib]]
                    bboxes_a = [a.get("bbox") for a in A]
                    bboxes_b = [b.get("bbox") for b in B]
                    m_bbox, ua, ub = greedy_match(bboxes_a, bboxes_b, bbox_iou, thresh=0.1)
                    poly_ious = []
                    for m in m_bbox:
                        pa = first_poly(A[m["i"]])
                        pb = first_poly(B[m["j"]])
                        if pa and pb and len(pa) >= 6 and len(pb) >= 6 and len(pa) % 2 == 0 and len(pb) % 2 == 0:
                            poly_ious.append(raster_poly_iou(pa, pb))
                        else:
                            poly_ious.append(None)
                    bbox_ious = [m["iou"] for m in m_bbox]
                    cat_agree = 0
                    for k, m in enumerate(m_bbox):
                        if A[m["i"]].get("category_id") == B[m["j"]].get("category_id"):
                            cat_agree += 1
                    agreement.append(
                        {
                            "a": iids[ia],
                            "b": iids[ib],
                            "n_a": len(A),
                            "n_b": len(B),
                            "n_matched_bbox_iou>=0.1": len(m_bbox),
                            "unmatched_a": len(ua),
                            "unmatched_b": len(ub),
                            "bbox_iou_matched": summarize_numeric(bbox_ious) if bbox_ious else {"n": 0},
                            "poly_iou_matched": summarize_numeric([x for x in poly_ious if x is not None]),
                            "category_agree_among_matched": cat_agree,
                            "category_agree_rate": (cat_agree / len(m_bbox)) if m_bbox else None,
                        }
                    )
        parsed = parse_filename(fn)
        sample_reports.append(
            {
                "physical_file": fn,
                "station": parsed["station"] if parsed else None,
                "year": parsed["year"] if parsed else None,
                "n_annotators": len(iids),
                "annotators": per_ann,
                "pairwise_agreement": agreement,
            }
        )

    # ---- grouped CV recommendation ----
    # bins: year, n_filaments (median across annotators), station
    def fil_bin(n):
        if n is None:
            return "unk"
        if n <= 3:
            return "0-3"
        if n <= 6:
            return "4-6"
        if n <= 10:
            return "7-10"
        return "11+"

    # 5-fold grouped by physical jpeg, greedy stratify on (year, fil_bin)
    physicals = sorted(unique_physical)
    strata = []
    for fn in physicals:
        y = year_per_physical.get(fn)
        fb = fil_bin(filaments_median_per_physical.get(fn))
        st = station_per_physical.get(fn)
        strata.append((fn, y, fb, st, filaments_median_per_physical.get(fn) or 0))

    # balance folds by stratum counts via greedy assignment to smallest fold
    # within same (year, fil_bin)
    fold_of = {}
    fold_n = [0] * 5
    fold_fil = [0.0] * 5
    by_stratum = defaultdict(list)
    for fn, y, fb, st, med in strata:
        by_stratum[(y, fb)].append((fn, med, st))
    # assign larger groups first
    for key in sorted(by_stratum, key=lambda k: -len(by_stratum[k])):
        items = sorted(by_stratum[key], key=lambda t: (-t[1], t[0]))
        for i, (fn, med, st) in enumerate(items):
            # pick fold with fewest items, then lowest filament sum
            fold = min(range(5), key=lambda f: (fold_n[f], fold_fil[f], f))
            fold_of[fn] = fold
            fold_n[fold] += 1
            fold_fil[fold] += med

    fold_summary = []
    for f in range(5):
        members = [fn for fn in physicals if fold_of[fn] == f]
        ys = Counter(year_per_physical[fn] for fn in members)
        sts = Counter(station_per_physical[fn] for fn in members)
        fbs = Counter(fil_bin(filaments_median_per_physical[fn]) for fn in members)
        n_ann_images = sum(len(physical_to_ann_ids[fn]) for fn in members)
        n_anns = sum(len(anns_by_image[iid]) for fn in members for iid in physical_to_ann_ids[fn])
        fold_summary.append(
            {
                "fold": f,
                "n_physical": len(members),
                "n_annotator_images": n_ann_images,
                "n_annotations": n_anns,
                "median_filaments_mean": (sum(filaments_median_per_physical[fn] or 0 for fn in members) / len(members))
                if members
                else 0,
                "years": dist_table(ys),
                "stations": dist_table(sts),
                "filament_bins": dist_table(fbs),
            }
        )

    leakage_risks = [
        "Splitting by annotator-image id (e.g. 010101-STEM vs 010102-STEM) leaks the same JPEG pixels into train and val.",
        "Random annotation-level split is worse: multiple polygons from the same disk image can land in both sides.",
        "Same-day / adjacent-day GONG frames of the same station can share slowly evolving filaments (temporal leakage).",
        "Year shift train->test is a hidden-exam risk: a year-unbalanced CV will overfit the public-looking years.",
        "Station shift (site PSF, seeing, contrast) is a second shift axis; do not treat station as i.i.d.",
        "Multi-annotator labels are not independent replicates of a single truth; training on all and validating on one annotator still shares the image.",
        "Do not use any leaked MAGFiLO public-test JSON/labels for CV, feature selection, or threshold fitting.",
        "file_name is .jpeg on disk while url is .jpg; do not key models on the remote URL stem alone.",
        "Category 'Unidentifiable'/'Ambiguous' vs Left/Right chirality is a label-noise axis; agreement may be low even when masks overlap.",
    ]

    # category names for stats
    cat_named = {str(c["id"]) + "_" + c["name"]: cat_counts[c["id"]] for c in categories}

    stats = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(DATA_ROOT),
        "used_leaked_public_test_labels": False,
        "schema": {
            "top_keys": list(coco.keys()),
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "image_fields": image_field_types,
            "image_key_sets": {str(k): v for k, v in image_key_sets.items()},
            "annotation_fields": ann_field_types,
            "annotation_key_sets": {str(k): v for k, v in ann_key_sets.items()},
            "image_record_example": {
                k: (images[0][k] if k != "url" else images[0][k]) for k in images[0]
            }
            if images
            else None,
            "annotation_record_example_meta": {
                "id": annotations[0]["id"],
                "image_id": annotations[0]["image_id"],
                "category_id": annotations[0]["category_id"],
                "area": annotations[0]["area"],
                "bbox": annotations[0]["bbox"],
                "iscrowd": annotations[0]["iscrowd"],
                "n_seg_polys": len(annotations[0]["segmentation"]),
                "n_seg_coords": len(annotations[0]["segmentation"][0]) if annotations[0]["segmentation"] else 0,
                "n_spine": len(annotations[0]["spine"]) if annotations[0].get("spine") else 0,
            }
            if annotations
            else None,
        },
        "counts": {
            "json_images": len(images),
            "json_annotations": len(annotations),
            "unique_physical_filenames_in_json": len(unique_physical),
            "unique_annotator_image_ids": len(set(image_ids)),
            "duplicate_image_ids": dup_image_ids,
            "duplicate_annotation_ids": dup_ann_ids[:20],
            "n_duplicate_annotation_ids": len(dup_ann_ids),
            "train_jpegs_on_disk": len(train_files),
            "test_jpegs_on_disk": len(test_files),
            "unique_annotator_codes": len(set(annotator_codes)),
            "annotator_codes": dist_table(Counter(annotator_codes)),
            "annotator_AA": dist_table(annotator_aa),
            "annotator_BB": dist_table(annotator_bb),
            "annotator_CC": dist_table(annotator_cc),
            "annotators_per_physical_image": dist_table(n_codes_per_physical),
            "annotator_image_records_per_physical": dist_table(n_ann_ids_per_physical),
            "physicals_with_n_annotators": {str(k): len(v) for k, v in sorted(physicals_by_n_annot.items())},
            "suffix_pattern_on_multi_annotator_files": dist_table(suffix_pattern),
            "mixed_AABB_same_physical": mixed_prefix_same_file,
            "images_with_zero_annotations": len(empty_ann_images),
            "orphan_annotations": len(orphan_image_ids),
            "annotations_per_annotator_image": summarize_numeric(anns_per_ann_image),
            "annotation_instances_summed_per_physical": summarize_numeric(anns_per_physical),
            "median_filaments_per_physical_across_annotators": summarize_numeric(
                list(filaments_median_per_physical.values())
            ),
            "category_counts": dist_table(cat_counts),
            "category_named": cat_named,
            "iscrowd": dist_table(iscrowd_counts),
        },
        "filename_encoding": {
            "pattern": "YYYYMMDDHHMMSS + 2-letter GONG station + .jpeg",
            "station_names": STATION_NAME,
            "train_unparsed": train_unparsed,
            "test_unparsed": test_unparsed,
            "id_pattern": "{AABBCC}-{YYYYMMDDHHMMSS}{Station}",
            "id_parse_failures": id_parse_fail,
            "id_stem_mismatches": id_stem_mismatch[:20],
            "n_id_stem_mismatches": len(id_stem_mismatch),
            "multi_annotator_example": "010101-STEM vs 010102-STEM share file_name STEM.jpeg; CC is the annotator replica.",
            "train_stations": dist_table(train_stations),
            "train_years": dist_table(train_years),
            "train_months": dist_table(train_months),
            "train_hours": dist_table(train_hours),
            "test_stations": dist_table(test_stations),
            "test_years": dist_table(test_years),
            "test_months": dist_table(test_months),
            "test_hours": dist_table(test_hours),
            "codes_appearing_on_n_physicals": summarize_numeric([len(v) for v in code_to_physicals.values()]),
        },
        "size_stats": {
            "reported_area_px": summarize_numeric(areas),
            "bbox_width": summarize_numeric(bbox_ws),
            "bbox_height": summarize_numeric(bbox_hs),
            "bbox_diagonal": summarize_numeric(bbox_diags),
            "bbox_area": summarize_numeric(bbox_areas),
            "polygon_vertices": summarize_numeric(n_vertices),
            "spine_list_len": summarize_numeric(spine_lens),
            "rel_abs_area_vs_shoelace": summarize_numeric(area_vs_shoelace),
        },
        "train_vs_test": {
            "filename_overlap": overlap,
            "stem_overlap": stem_overlap,
            "n_overlap": len(overlap),
            "year_shift": year_shift,
            "station_shift": station_shift,
            "train_year_span": [min(train_years) if train_years else None, max(train_years) if train_years else None],
            "test_year_span": [min(test_years) if test_years else None, max(test_years) if test_years else None],
        },
        "samples": sample_reports,
        "quality": {
            "json_filenames_missing_on_disk": extra_json_not_on_disk,
            "disk_train_files_missing_from_json": extra_disk_not_in_json,
            "empty_segmentations": empty_segs[:50],
            "n_empty_segmentations": len(set(empty_segs)),
            "odd_length_polygons": odd_len_segs[:20],
            "n_odd_length_polygons": len(odd_len_segs),
            "short_polygons_lt_3_pts": short_polys[:20],
            "n_short_polygons": len(short_polys),
            "zero_reported_area": zero_area_reported[:20],
            "n_zero_reported_area": len(zero_area_reported),
            "iscrowd_nonzero": int(sum(v for k, v in iscrowd_counts.items() if k != 0)),
            "multi_polygon_annotations": len(multi_poly),
            "coords_out_of_image": list(dict.fromkeys(oob_coords))[:20],
            "n_coords_out_of_image": len(set(oob_coords)),
            "self_intersecting_flagged": self_x[:20],
            "n_self_intersecting_flagged": len(self_x),
            "bbox_vs_polygon_extent_mismatch": bbox_vs_poly[:20],
            "n_bbox_vs_polygon_mismatch": len(bbox_vs_poly),
            "spine_missing": spine_missing,
            "date_parse_fail": date_parse_fail[:20],
            "n_date_parse_fail": len(date_parse_fail),
            "json_height_dist": dist_table(heights),
            "json_width_dist": dist_table(widths),
            "json_license_dist": dist_table(licenses_used),
            "url_extensions": dist_table(url_exts),
            "train_jpeg_sizes": dist_table(train_jpeg_sizes),
            "test_jpeg_sizes": dist_table(test_jpeg_sizes),
            "train_unreadable_jpeg": train_bad_jpeg,
            "test_unreadable_jpeg": test_bad_jpeg,
            "train_jpeg_vs_json_size_mismatch": train_size_mismatch,
            "nseg_dist": dist_table(nseg_counts),
            "empty_annotation_image_ids": empty_ann_images[:20],
        },
        "cv": {
            "recommendation": (
                "GroupKFold on physical jpeg file_name (707 groups). Never split 010101/010102 "
                "of the same stem. Stratify groups by year and median filament-count bin. "
                "Optional secondary check: leave-one-station-out or leave-late-years-out as a "
                "shift probe, not as the only selection metric."
            ),
            "n_folds": 5,
            "n_groups": len(physicals),
            "filament_bins": ["0-3", "4-6", "7-10", "11+"],
            "fold_summary": fold_summary,
            "leakage_risks": leakage_risks,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)", flush=True)

    # Human report dump (also used by the caller)
    print("\n========== EDA SUMMARY ==========")
    print("images_json", len(images), "annotations", len(annotations))
    print("unique_physical_json", len(unique_physical), "train_disk", len(train_files), "test_disk", len(test_files))
    print("annotator_image_ids", len(set(image_ids)))
    print("annotators_per_physical", dict(n_codes_per_physical))
    print("overlap", overlap)
    print("empty_segs", len(set(empty_segs)), "iscrowd_nz", stats["quality"]["iscrowd_nonzero"])
    print("missing_disk", extra_json_not_on_disk, "extra_disk", extra_disk_not_in_json)
    print("picked_samples", picked)
    print("area", stats["size_stats"]["reported_area_px"])
    print("bbox_w", stats["size_stats"]["bbox_width"])
    print("bbox_h", stats["size_stats"]["bbox_height"])


if __name__ == "__main__":
    main()
