"""
Count hard-braking events (HBEs) per CTN segment across a full GCS export.

Two stages, so the expensive part only runs once:

  1. extract   — streams every daily CSV from the bucket, keeps braking events inside
                 the CTN bounding box, assigns each event to a CTN segment, and writes
                 one small parquet of matched events per input file, plus a per-file
                 QA row. Re-running skips files already done.

  2. aggregate — reads the match parquets, de-duplicates on (event id, segment) across
                 ALL files, drops any excluded dates, and writes segment-level counts
                 and (optionally) rates per million VMT.

Matching (--match heading, the default):
  - candidates = every CTN segment whose 75 ft flat-cap buffer contains the event's
    start point (same buffers as the original notebook)
  - for each candidate, the segment's direction is taken from the exact polyline edge
    nearest the event, so curves, ramps and corners use their local direction
  - keep candidates within --heading-tol degrees of the event's startHeading, ignoring
    direction of travel (a northbound event matches a southbound-digitized line)
  - of those, the nearest segment wins: one segment per event
  - events with no candidate inside the tolerance are dropped (--fallback drop) or
    given to the nearest candidate (--fallback nearest); both counts are in the QA
--match all reproduces the earlier behavior: an event counts on every buffer it is in.

Examples
--------
python hbe_counts.py extract \
    --bucket stl-willow-production-toclient-uta-otb \
    --prefix hard-braking-exports/ --creds creds.json \
    --ctn data/ctn_within_boundary.geojson \
    --out hbe_work --workers 8

python hbe_counts.py aggregate --out hbe_work \
    --ctn data/ctn_within_boundary.geojson \
    --aadt data/24_aadt_manual.csv \
    --exclude 2026-08-25:2026-08-28

Requires: pandas, geopandas>=0.14, shapely>=2, pyarrow, google-cloud-storage

For testing against a local copy, use --local-dir PATH instead of --bucket.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from google.cloud import storage

CRS_WORK = 2277  # NAD83 / Texas Central (ftUS) — same as the notebook
BUFFER_FT = 75
BRAKING = ["Severe Braking", "Very Hard Braking", "Hard Braking"]
USECOLS = ["id", "startCoords", "endCoords", "date", "severity", "startHeading"]
SEG_ID = "CTN_SEGMENT_ID"
AADT_COL = "Estimated 2024 AADT (initial release)"
PART_RE = re.compile(r"region=([^/]+)/year=(\d{4})/month=(\d{2})/day=(\d{2})")
WKT_RE = r"POINT\s*\(\s*(\S+)\s+(\S+)\s*\)"


# --------------------------------------------------------------------------- helpers
def load_ctn(path: str) -> gpd.GeoDataFrame:
    ctn = gpd.read_file(path).to_crs(epsg=CRS_WORK)
    ctn = ctn[ctn.geometry.notna() & ~ctn.geometry.is_empty].reset_index(drop=True)
    return ctn[[SEG_ID, "geometry"]]


def parse_points(s: pd.Series) -> pd.DataFrame:
    """'POINT(-97.74 30.27)' -> lon, lat floats (vectorized)."""
    xy = s.astype(str).str.extract(WKT_RE)
    return pd.DataFrame({"lon": pd.to_numeric(xy[0], errors="coerce"),
                         "lat": pd.to_numeric(xy[1], errors="coerce")}, index=s.index)


def axial_diff(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Angle between two headings ignoring direction of travel, 0-90 degrees."""
    return np.abs((h1 - h2 + 90) % 180 - 90)


class Matcher:
    """CTN parts, buffers, spatial index, and per-edge bearings; thread-safe for reads."""

    def __init__(self, ctn: gpd.GeoDataFrame, mode: str, heading_tol: float, fallback: str):
        self.mode, self.heading_tol, self.fallback = mode, heading_tol, fallback

        # Split multi-part segments into single LineStrings, remembering the parent segment
        parts = ctn[[SEG_ID, "geometry"]].explode(index_parts=False, ignore_index=True)
        parts = parts[parts.geometry.geom_type == "LineString"].reset_index(drop=True)
        self.part_seg = parts[SEG_ID].to_numpy()
        self.part_geom = np.asarray(parts.geometry.values)
        self.buffers = gpd.GeoSeries(self.part_geom, crs=CRS_WORK).buffer(BUFFER_FT, cap_style="flat")
        self.tree = shapely.STRtree(np.asarray(self.buffers.values))

        # Flat vertex table: (part, cumulative distance) -> lets us find the edge under
        # any point along a part with one vectorized searchsorted.
        xy, part_of_v = shapely.get_coordinates(self.part_geom, return_index=True)
        seg_len = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)]
        new_part = np.r_[True, part_of_v[1:] != part_of_v[:-1]]
        seg_len[new_part] = 0
        cum = np.cumsum(seg_len)
        cum -= np.repeat(cum[new_part], np.diff(np.r_[np.flatnonzero(new_part), len(cum)]))
        self.big = float(cum.max() + 1.0)
        self.v_key = part_of_v * self.big + cum
        starts = np.flatnonzero(new_part)
        self.part_first_v = starts
        self.part_last_v = np.r_[starts[1:], len(cum)] - 1

        # True (not grid) compass bearing of the edge starting at each vertex.
        # Computed in lon/lat so Texas Central's ~1.3 degree grid convergence drops out.
        ll = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS_WORK).to_crs(4326)
        lon, lat = ll.x.to_numpy(), ll.y.to_numpy()
        dx = np.diff(lon, append=lon[-1]) * np.cos(np.radians(lat))
        dy = np.diff(lat, append=lat[-1])
        self.edge_bearing = np.degrees(np.arctan2(dx, dy)) % 360

        x0, y0, x1, y1 = self.buffers.to_crs(4326).total_bounds
        pad = 0.002
        self.bbox = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    def local_bearing(self, part_idx: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Bearing of the polyline edge nearest each point on its candidate part."""
        d = shapely.line_locate_point(self.part_geom[part_idx], pts)
        v = np.searchsorted(self.v_key, part_idx * self.big + d, side="right") - 1
        v = np.clip(v, self.part_first_v[part_idx], np.maximum(self.part_last_v[part_idx] - 1,
                                                               self.part_first_v[part_idx]))
        return self.edge_bearing[v]

    def match(self, pts: np.ndarray, heading: np.ndarray):
        """Return a DataFrame of matches: event, part, dist_ft, heading_diff, n_candidates,
        plus counts for QA."""
        ev, part = self.tree.query(pts, predicate="within")
        cand = pd.DataFrame({"event": ev, "part": part})
        cand["seg"] = self.part_seg[part]
        cand["dist_ft"] = shapely.distance(pts[ev], self.part_geom[part])
        cand["heading_diff"] = axial_diff(heading[ev], self.local_bearing(part, pts[ev]))
        # a segment with several parts near the point: keep its closest part
        cand = cand.sort_values(["event", "dist_ft"]).drop_duplicates(["event", "seg"])
        cand["n_candidates"] = cand.groupby("event")["seg"].transform("size")

        stats = {"in_buffer": cand["event"].nunique(),
                 "multi_candidate": int((cand.groupby("event").size() > 1).sum())}
        if self.mode == "all":
            stats["no_heading_match"] = 0
            return cand, stats

        ok = cand["heading_diff"] <= self.heading_tol  # NaN heading -> False
        best = cand[ok].drop_duplicates("event")      # already sorted by distance
        missing = np.setdiff1d(cand["event"].unique(), best["event"].to_numpy())
        stats["no_heading_match"] = len(missing)
        if self.fallback == "nearest" and len(missing):
            extra = cand[cand["event"].isin(missing)].drop_duplicates("event")
            best = pd.concat([best, extra]).sort_values("event")
        return best, stats


# --------------------------------------------------------------------------- extract
class GCSSource:
    """Lists and reads CSVs from a bucket using a service-account key file."""

    def __init__(self, bucket_name: str, prefix: str, creds: str):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
        self.bucket_name = bucket_name
        self.prefix = prefix
        self._local = threading.local()  # one client per worker thread

    def _bucket(self):
        if not hasattr(self._local, "bucket"):
            self._local.bucket = storage.Client().bucket(self.bucket_name)
        return self._local.bucket

    def list_csvs(self) -> list[str]:
        client = storage.Client()
        names = [b.name for b in client.list_blobs(self.bucket_name, prefix=self.prefix)
                 if b.name.lower().endswith(".csv")]
        return sorted(names)

    def read_bytes(self, name: str) -> bytes:
        return self._bucket().blob(name).download_as_bytes()


class LocalSource:
    """Same interface over a local folder (for testing or a downloaded copy)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def list_csvs(self) -> list[str]:
        return sorted(str(p) for p in self.root.rglob("*") if p.suffix.lower() == ".csv")

    def read_bytes(self, name: str) -> bytes:
        return Path(name).read_bytes()


def process_file(source, path: str, matcher: Matcher, out_dir: Path, sep: str) -> dict:
    m = PART_RE.search(path)
    region, y, mo, d = m.groups() if m else ("unknown", "0000", "00", "00")
    stem = f"{region}_{y}_{mo}_{d}_{Path(path).stem}"
    out_path = out_dir / "matched" / f"{stem}.parquet"
    qa = {"file": path, "region": region, "partition_date": f"{y}-{mo}-{d}"}

    df = pd.read_csv(io.BytesIO(source.read_bytes(path)), sep=sep,
                     usecols=lambda c: c in USECOLS, dtype={"id": str})
    qa["rows"] = len(df)
    qa["unique_ids"] = df["id"].nunique()
    qa["severity_counts"] = df["severity"].value_counts().to_dict().__repr__()

    df = df[df["severity"].isin(BRAKING)].drop_duplicates("id")
    qa["braking_rows"] = len(df)

    xy = parse_points(df["startCoords"])
    x0, y0, x1, y1 = matcher.bbox
    keep = xy["lon"].between(x0, x1) & xy["lat"].between(y0, y1)
    df, xy = df[keep], xy[keep]
    qa["in_bbox"] = len(df)

    pts = np.asarray(gpd.GeoSeries(gpd.points_from_xy(xy["lon"], xy["lat"]), crs=4326)
                     .to_crs(CRS_WORK).values)
    heading = pd.to_numeric(df["startHeading"], errors="coerce").to_numpy(dtype=float)
    m, stats = matcher.match(pts, heading)
    qa.update(stats)

    ev = m["event"].to_numpy()
    matched = df.iloc[ev]
    exy = parse_points(matched["endCoords"])
    end_pts = np.asarray(gpd.GeoSeries(gpd.points_from_xy(exy["lon"], exy["lat"]), crs=4326)
                         .to_crs(CRS_WORK).values)
    out = pd.DataFrame({
        "id": matched["id"].to_numpy(),
        SEG_ID: m["seg"].to_numpy(),
        "date": pd.to_datetime(matched["date"], format="mixed").to_numpy(),
        "severity": matched["severity"].to_numpy(),
        "lon": xy["lon"].to_numpy()[ev],
        "lat": xy["lat"].to_numpy()[ev],
        "heading": heading[ev],
        "dist_ft": m["dist_ft"].round(1).to_numpy(),
        "heading_diff": m["heading_diff"].round(1).to_numpy(),
        "n_candidates": m["n_candidates"].to_numpy(),
        "start_end_ft": np.round(shapely.distance(pts[ev], end_pts), 0),
    })
    # Pin dtypes so files with zero matches don't write null-typed columns.
    out = out.astype({"id": "string", "severity": "string", "lon": "float64", "lat": "float64",
                      "heading": "float64", "dist_ft": "float64", "heading_diff": "float64",
                      "n_candidates": "int64", "start_end_ft": "float64"})
    out["date"] = pd.to_datetime(out["date"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    qa["matched_events"] = out["id"].nunique()
    qa["matched_pairs"] = len(out)
    qa["median_start_end_ft"] = float(out.drop_duplicates("id")["start_end_ft"].median()) if len(out) else np.nan
    return qa


def cmd_extract(a):
    out_dir = Path(a.out)
    (out_dir / "qa").mkdir(parents=True, exist_ok=True)
    matcher = Matcher(load_ctn(a.ctn), a.match, a.heading_tol, a.fallback)
    source = GCSSource(a.bucket, a.prefix, a.creds) if a.bucket else LocalSource(a.local_dir)
    files = source.list_csvs()
    done = {p.stem for p in (out_dir / "qa").glob("*.parquet")}

    def qa_stem(path):
        m = PART_RE.search(path)
        region, y, mo, d = m.groups() if m else ("unknown", "0000", "00", "00")
        return f"{region}_{y}_{mo}_{d}_{Path(path).stem}"

    todo = [p for p in files if qa_stem(p) not in done]
    print(f"{len(files)} files found, {len(files) - len(todo)} already done, {len(todo)} to process")

    def run(path):
        qa = process_file(source, path, matcher, out_dir, a.sep)
        pd.DataFrame([qa]).to_parquet(out_dir / "qa" / f"{qa_stem(path)}.parquet", index=False)
        return qa

    t0, n, failures = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run, p): p for p in todo}
        for fut in as_completed(futures):
            n += 1
            try:
                qa = fut.result()
                if n % 25 == 0 or n == len(todo):
                    print(f"[{n}/{len(todo)}] {time.time() - t0:6.0f}s  last: {qa['region']} "
                          f"{qa['partition_date']} rows={qa['rows']:,} matched={qa['matched_events']:,}")
            except Exception as e:  # keep going; failed files simply get retried next run
                failures.append((futures[fut], repr(e)))
                print(f"FAILED {futures[fut]}: {e!r}", file=sys.stderr)

    qa_all = pd.concat([pd.read_parquet(p) for p in (out_dir / "qa").glob("*.parquet")])
    qa_all.sort_values(["region", "partition_date"]).to_csv(out_dir / "qa_by_file.csv", index=False)
    print(f"Done. {len(failures)} failures. QA table: {out_dir / 'qa_by_file.csv'}")


# --------------------------------------------------------------------------- aggregate
def parse_excludes(specs):
    days = set()
    for s in specs or []:
        lo, _, hi = s.partition(":")
        for d in pd.date_range(lo, hi or lo):
            days.add(d.date())
    return days


def cmd_aggregate(a):
    out_dir = Path(a.out)
    parts = [pd.read_parquet(p) for p in sorted((out_dir / "matched").glob("*.parquet"))]
    ev = pd.concat([p for p in parts if len(p)], ignore_index=True)
    n_raw = len(ev)
    ev = ev.drop_duplicates(["id", SEG_ID])  # catches duplicate events that span files/days
    print(f"{n_raw:,} matched rows, {len(ev):,} after cross-file de-dup on (id, segment); "
          f"{ev['id'].nunique():,} distinct events")

    excl = parse_excludes(a.exclude)
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    all_days = sorted(set(ev["date"]))
    if a.start:
        all_days = [d for d in all_days if d >= pd.Timestamp(a.start).date()]
    if a.end:
        all_days = [d for d in all_days if d <= pd.Timestamp(a.end).date()]
    days = [d for d in all_days if d not in excl]
    ev = ev[ev["date"].isin(set(days))]
    n_days = len(days)
    print(f"{n_days} days kept ({min(days)} to {max(days)}), {len(excl & set(all_days))} excluded; "
          f"{len(ev):,} event-segment matches counted")

    daily = ev.groupby("date")["id"].nunique().rename("matched_events")
    daily.to_csv(out_dir / "daily_matched_events.csv")

    counts = (ev.groupby([SEG_ID, "severity"]).size().unstack(fill_value=0)
              .reindex(columns=BRAKING, fill_value=0))
    counts["hbe_count"] = counts.sum(axis=1)
    counts = counts.reset_index()

    if a.ctn:
        ctn = load_ctn(a.ctn)
        ctn["length_mi"] = ctn.length / 5280
        seg = ctn.drop(columns="geometry").merge(counts, on=SEG_ID, how="left")
        seg[counts.columns.drop(SEG_ID)] = seg[counts.columns.drop(SEG_ID)].fillna(0).astype(int)
        seg["days_covered"] = n_days
        if a.aadt:
            aadt = pd.read_csv(a.aadt)[["Zone Name", AADT_COL]]
            seg = seg.merge(aadt, left_on=SEG_ID, right_on="Zone Name", how="left").drop(columns="Zone Name")
            vmt = seg[AADT_COL] * seg["length_mi"] * n_days
            seg["hbe_per_mvmt"] = (seg["hbe_count"] / vmt * 1e6).replace([np.inf, -np.inf], np.nan)
        counts = seg

    counts.to_csv(out_dir / "segment_hbe_counts.csv", index=False)
    print(f"Wrote {out_dir / 'segment_hbe_counts.csv'} ({len(counts):,} segments)")


# --------------------------------------------------------------------------- cli
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--bucket", help="GCS bucket name")
    src.add_argument("--local-dir", help="local folder with the same layout (instead of --bucket)")
    e.add_argument("--prefix", default="hard-braking-exports/", help="object prefix inside the bucket")
    e.add_argument("--creds", default="creds.json", help="service-account key file")
    e.add_argument("--ctn", required=True, help="CTN line layer (geojson/gpkg/shp)")
    e.add_argument("--out", required=True, help="working directory for outputs")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--match", choices=["heading", "all"], default="heading",
                   help="heading: one best segment per event; all: every buffer it falls in")
    e.add_argument("--heading-tol", type=float, default=30.0,
                   help="max degrees between startHeading and the segment's local direction")
    e.add_argument("--fallback", choices=["drop", "nearest"], default="drop",
                   help="events with no candidate inside --heading-tol: drop, or take the nearest")
    e.add_argument("--sep", default=",", help="CSV delimiter (use '\\t' for TSV)")
    e.set_defaults(func=cmd_extract)

    g = sub.add_parser("aggregate")
    g.add_argument("--out", required=True)
    g.add_argument("--ctn", help="CTN layer, to include zero-count segments and lengths")
    g.add_argument("--aadt", help="AADT csv (Zone Name, Estimated 2024 AADT ...) for rates")
    g.add_argument("--exclude", nargs="*", help="dates or ranges to drop, e.g. 2026-08-25:2026-08-28")
    g.add_argument("--start", help="first date to include (YYYY-MM-DD)")
    g.add_argument("--end", help="last date to include (YYYY-MM-DD)")
    g.set_defaults(func=cmd_aggregate)

    a = p.parse_args()
    if getattr(a, "sep", None) == "\\t":
        a.sep = "\t"
    a.func(a)


if __name__ == "__main__":
    main()