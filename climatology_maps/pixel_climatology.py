"""Per-pixel flood climatology for the five Aweil counties (Northern Bahr el Ghazal).

Builds the *historical* per-pixel flood pattern that the LightGBM forecast maps
downscale: for every ~232 m MODIS/VIIRS grid cell inside the five Aweil counties
and every calendar month, count how often the satellite detected that pixel as
flooded across the 2000-2025 baseline.

Three derived quantities per (pixel, month):

* `freq`        = hit_weeks / total_weeks
                  -> P(pixel is wet in a randomly chosen week of that month).
* `freq_given`  = hit_weeks / county_flood_weeks
                  -> P(pixel is wet | the county has a flood that month). This is
                     the *spatial signature*: the places where the water actually
                     goes when the county floods, normalised to 1 across pixels.
* `clim_det_prob` = (k + 0.5) / (n + 1) county-month smoothed flood probability
                  -> the same reference used by `LightGBM_v1/baselines.py`.

Everything is a pure count over observed weeks; nothing is trained and no forecast
is made here. Weeks without a satellite detection count as zero area (the same
censoring convention as the EDA and the LightGBM labels).

Raw data requirement
--------------------
Reads the committed compact flood-mask Parquet files that the EDA notebooks use:

    raw_data/flood_masks/compact_recurring/flood_events_h20v08_YYYY.parquet
    raw_data/flood_masks/compact_unusual/flood_events_h20v08_YYYY.parquet

(tile h20v08 covers the Aweil region; 2000-2025). If they are missing, a
FileNotFoundError with instructions is raised.

Caching / outputs
-----------------
The build result is cached under
`climatology_maps/outputs/climatology_data/` (gitignored):

    pixel_grid.csv                     every grid cell inside the 5 counties
    pixel_month_climatology.csv        per (county, pixel, month) hit statistics
    county_month_climatology.csv       per (county, month) totals + references

Expected runtime on a laptop: ~1-3 minutes, dominated by the 52 filtered Parquet
reads.

Example
-------
    python -m climatology_maps.pixel_climatology          # build + cache
    python -m climatology_maps.pixel_climatology --rebuild  # force rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as pds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ADMIN2_PATH = PROJECT_ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
FLOOD_ROOT = PROJECT_ROOT / "raw_data" / "flood_masks"

OUT_DIR = PROJECT_ROOT / "climatology_maps" / "outputs" / "climatology_data"

STATE = "Northern Bahr el Ghazal"
TILE = "h20v08"                       # tile covering the Aweil region
YEARS = range(2000, 2026)
FLOOD_TYPES = ("recurring", "unusual")

# NASA MCDWD/VCDWD grid (User Guide Rev. F, Table 5, p. 28).
GRID_STEP_DEGREES = 10 / 4800
PRODUCT_RADIUS_M = 6371007.181
PIXEL_AREA_KM2_REF = 0.53  # approximate comment value; exact value is latitude-dependent


def pixel_area_km2(latitude) -> np.ndarray:
    """Area (km2) of a MODIS/VCDWD grid cell centred at the given latitude.

    Same formula as `EDA_flood_masks.flood_eda.pixel_area_km2`, kept here so the
    forecast-map module does not have to import the EDA package.
    """
    latitude = np.asarray(latitude, dtype=float)
    half_step = GRID_STEP_DEGREES / 2
    if not np.isfinite(latitude).all() or (np.abs(latitude) + half_step > 90).any():
        raise ValueError("Grid-cell centres must be finite and inside the poles")
    south = np.deg2rad(latitude - half_step)
    north = np.deg2rad(latitude + half_step)
    return (
        PRODUCT_RADIUS_M**2 * np.deg2rad(GRID_STEP_DEGREES)
        * (np.sin(north) - np.sin(south)) / 1e6
    )


def snap_to_grid(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    """Snap coordinates to the product's canonical cell centres.

    The compact Parquet stores cell centres at (k + 0.5) * GRID_STEP_DEGREES
    with float32 precision (~7 significant decimals), so the stored values can
    differ from the ideal centre in the 6th decimal. Rounding in place would
    put neighbouring cells on *different* canonical keys than the render grid
    (built from the ideal formula) and they would never merge. Instead we
    recover the cell index k = round(v/step - 0.5) and rebuild the centre with
    the same formula the grid uses — both paths then agree exactly.
    """
    lo_k = np.round(np.asarray(longitude, dtype=float) / GRID_STEP_DEGREES - 0.5)
    la_k = np.round(np.asarray(latitude, dtype=float) / GRID_STEP_DEGREES - 0.5)
    lo = np.round((lo_k + 0.5) * GRID_STEP_DEGREES, 6)
    la = np.round((la_k + 0.5) * GRID_STEP_DEGREES, 6)
    return lo, la


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_aweil_counties(boundaries_path: str | Path = ADMIN2_PATH) -> gpd.GeoDataFrame:
    """Admin-2 polygons of the five Aweil counties (standardised columns)."""
    gdf = gpd.read_file(boundaries_path).to_crs("EPSG:4326")
    if "adm2_name" in gdf.columns:
        gdf = gdf.rename(columns={"adm2_name": "county"})
    selected = gdf[gdf["adm1_name"] == STATE][["county", "geometry"]].copy()
    selected = selected.drop_duplicates(subset="county")
    return selected.reset_index(drop=True)


def _annual_path(flood_root: Path, flood_type: str, year: int) -> Path:
    return flood_root / f"compact_{flood_type}" / f"flood_events_{TILE}_{year}.parquet"


def load_pixel_detections(
    flood_root: str | Path = FLOOD_ROOT,
    counties: gpd.GeoDataFrame | None = None,
    years: range = YEARS,
) -> pd.DataFrame:
    """Load all 2000-2025 flood detections inside the Aweil county polygons.

    Returns rows with columns [week, lat, lon, county] where week is the Monday
    of the detection week. Recurring and unusual detections are combined; a pixel
    found by either class counts once per week.
    """
    root = Path(flood_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Flood-mask data not found at {root}. Download the SURFdrive dataset "
            "and extract it into raw_data/ (the course zip contains this folder)."
        )
    if counties is None:
        counties = load_aweil_counties()
    min_lon, min_lat, max_lon, max_lat = counties.total_bounds
    # A 0.05 deg margin so boundary pixels are not cut off by the bbox filter.
    margin = 0.05
    bounds = (min_lon - margin, min_lat - margin, max_lon + margin, max_lat + margin)

    frames = []
    for year in years:
        for ftype in FLOOD_TYPES:
            path = _annual_path(root, ftype, year)
            if not path.is_file():
                raise FileNotFoundError(f"Missing flood file: {path}")
            ds = pds.dataset(path, format="parquet")
            keep = (
                (pds.field("lon") >= bounds[0]) & (pds.field("lon") <= bounds[2])
                & (pds.field("lat") >= bounds[1]) & (pds.field("lat") <= bounds[3])
            )
            table = ds.to_table(columns=["date", "lat", "lon"], filter=keep)
            frames.append(table.to_pandas())
    if not frames:
        raise ValueError("No flood detection rows loaded from the Aweil tile")

    obs = pd.concat(frames, ignore_index=True)
    obs["date"] = pd.to_datetime(obs["date"], errors="raise")
    obs["week"] = obs["date"].dt.to_period("W-SUN").dt.start_time
    obs["lon"], obs["lat"] = snap_to_grid(obs["lon"], obs["lat"])

    # Assign every distinct grid cell once to its county (point-in-polygon).
    cells = obs[["lat", "lon"]].drop_duplicates()
    points = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326"
    )
    located = gpd.sjoin(points, counties, how="inner", predicate="within")
    located = located[["lat", "lon", "county"]]
    obs = obs.merge(located, on=["lat", "lon"], validate="many_to_one")
    obs = obs.drop_duplicates(["week", "lat", "lon"])  # recurring + unusual once
    obs["month"] = obs["week"].dt.month
    return obs.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Climatology aggregation
# --------------------------------------------------------------------------- #
def calendar_weeks_per_month() -> pd.DataFrame:
    """Monday-start weeks 2000-01-03..2025-12-29: total weeks per calendar month."""
    weeks = pd.date_range("2000-01-03", "2025-12-29", freq="W-MON")
    return (
        pd.DataFrame({"week": weeks})
        .assign(month=lambda d: d["week"].dt.month)
        .groupby("month", as_index=False)
        .agg(total_weeks=("week", "size"))
    )


def build_pixel_climatology(
    flood_root: str | Path = FLOOD_ROOT,
    boundaries_path: str | Path = ADMIN2_PATH,
    years: range = YEARS,
) -> dict[str, pd.DataFrame]:
    """Compute the per-pixel, per-month climatology tables.

    Returns a dict with keys:
      grid          : every grid cell inside the 5 counties (county, lat, lon)
      pixel_month   : per (county, lat, lon, month) hit statistics (hits only)
      county_month  : per (county, month) totals + climatological reference
    """
    counties = load_aweil_counties(boundaries_path)
    obs = load_pixel_detections(flood_root, counties, years)

    area = pixel_area_km2(obs["lat"].to_numpy())
    obs["pixel_area_km2"] = area

    pixel_stats = (
        obs.groupby(["county", "lat", "lon", "month"], observed=True)
        .agg(hit_weeks=("week", "size"), pixel_area_km2=("pixel_area_km2", "first"))
        .reset_index()
    )
    county_weeks = (
        obs.drop_duplicates(["county", "week"])
        .assign(month=lambda d: d["week"].dt.month)
        .groupby(["county", "month"], observed=True)
        .agg(flood_weeks=("week", "size"))
        .reset_index()
    )

    total_weeks = calendar_weeks_per_month()
    # Full county x calendar-month product so every (county, month) slot has a
    # reference (flood_weeks == 0 for months with no historical floods).
    counties_list = counties["county"].drop_duplicates().sort_values()
    county_month = (
        pd.DataFrame({"county": counties_list})
        .merge(total_weeks, how="cross")
        .merge(county_weeks, on=["county", "month"], how="left")
    )
    county_month["flood_weeks"] = county_month["flood_weeks"].fillna(0).astype(int)
    # Smoothed county-month climatological detection probability (k+0.5)/(n+1),
    # identical to LightGBM_v1.baselines.climatology_detection_baseline.
    county_month["clim_det_prob"] = (county_month["flood_weeks"] + 0.5) / (
        county_month["total_weeks"] + 1.0
    )

    pixel_month = pixel_stats.merge(
        county_month[["county", "month", "flood_weeks", "total_weeks"]],
        on=["county", "month"],
        validate="many_to_one",
    )
    pixel_month["freq"] = pixel_month["hit_weeks"] / pixel_month["total_weeks"]
    # Fraction of the county's flood weeks in that month that also wet this pixel.
    # Rows only exist where hit_weeks > 0, which implies flood_weeks >= 1.
    pixel_month["freq_given"] = pixel_month["hit_weeks"] / pixel_month["flood_weeks"]

    # Fallback placement for county-months without any historical flood: the
    # county's *annual* pattern, normalised to a distribution that sums to 1
    # over the county (used only as a spatial prior, never as a magnitude).
    annual = (
        obs.groupby(["county", "lat", "lon"], observed=True)["week"]
        .nunique()
        .reset_index()
        .rename(columns={"week": "hit_weeks_annual"})
    )
    total_weeks_all = int(total_weeks["total_weeks"].sum())
    annual["freq_annual"] = annual["hit_weeks_annual"] / total_weeks_all
    totals = annual.groupby("county", observed=True)["freq_annual"].transform("sum")
    annual["fallback_weight"] = annual["freq_annual"] / totals.where(totals > 0, np.nan)

    # Full render domain: every grid cell inside the 5 counties (flooded or not).
    grid = build_full_grid(counties)

    return {
        "grid": grid,
        "pixel_month": pixel_month,
        "county_month": county_month,
        "annual_fallback": annual,
    }


def build_full_grid(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """All MODIS grid-cell centres inside the county polygons.

    ~310k-330k cells for the five Aweil counties. This is the fixed spatial
    domain that every weekly map is drawn on (zeros where never flooded).
    """
    min_lon, min_lat, max_lon, max_lat = counties.total_bounds
    # Product cell centres are at (k + 0.5) * step (confirmed against the raw
    # observed lat/lon values, whose (v / step) mod 1 == 0.5).
    n_lon = int(np.ceil(max_lon / GRID_STEP_DEGREES)) + 1
    n_lat = int(np.ceil(max_lat / GRID_STEP_DEGREES)) + 1
    lons = np.round((np.arange(n_lon) + 0.5) * GRID_STEP_DEGREES, 6)
    lats = np.round((np.arange(n_lat) + 0.5) * GRID_STEP_DEGREES, 6)
    lons = lons[(lons >= min_lon) & (lons <= max_lon)]
    lats = lats[(lats >= min_lat) & (lats <= max_lat)]
    lon_m, lat_m = np.meshgrid(lons, lats)
    cells = pd.DataFrame({"lon": lon_m.ravel(), "lat": lat_m.ravel()})
    points = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326"
    )
    located = gpd.sjoin(points, counties, how="inner", predicate="within")
    located["pixel_area_km2"] = pixel_area_km2(located["lat"].to_numpy())
    grid = located[["county", "lat", "lon", "pixel_area_km2"]].reset_index(drop=True)
    return grid.sort_values(["county", "lat", "lon"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def save_cache(tables: dict[str, pd.DataFrame], out_dir: str | Path = OUT_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "grid": "pixel_grid.csv",
        "pixel_month": "pixel_month_climatology.csv",
        "county_month": "county_month_climatology.csv",
        "annual_fallback": "annual_fallback.csv",
    }
    for key, fname in names.items():
        tables[key].to_csv(out_dir / fname, index=False)
        print(f"  [cache] {out_dir / fname} ({len(tables[key])} rows)")
    return out_dir


def load_cache(out_dir: str | Path = OUT_DIR) -> dict[str, pd.DataFrame] | None:
    out_dir = Path(out_dir)
    names = {
        "grid": "pixel_grid.csv",
        "pixel_month": "pixel_month_climatology.csv",
        "county_month": "county_month_climatology.csv",
        "annual_fallback": "annual_fallback.csv",
    }
    tables = {}
    for key, fname in names.items():
        path = out_dir / fname
        if not path.is_file():
            return None
        tables[key] = pd.read_csv(path)
    return tables


def get_climatology(
    flood_root: str | Path = FLOOD_ROOT,
    boundaries_path: str | Path = ADMIN2_PATH,
    cache_dir: str | Path = OUT_DIR,
    rebuild: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return cached tables or build (and cache) them from the raw Parquets."""
    if not rebuild:
        cached = load_cache(cache_dir)
        if cached is not None:
            print(f"  [climatology] loaded from cache in {cache_dir}")
            return cached
    print("  [climatology] building per-pixel climatology from raw flood masks ...")
    tables = build_pixel_climatology(flood_root, boundaries_path)
    save_cache(tables, cache_dir)
    return tables


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flood-root", default=str(FLOOD_ROOT))
    parser.add_argument("--boundaries", default=str(ADMIN2_PATH))
    parser.add_argument("--outdir", default=str(OUT_DIR))
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache and rebuild")
    args = parser.parse_args()

    tables = get_climatology(
        flood_root=args.flood_root,
        boundaries_path=args.boundaries,
        cache_dir=args.outdir,
        rebuild=args.rebuild,
    )

    pm = tables["pixel_month"]
    cm = tables["county_month"]
    print(f"\nPixels with history: {pm['lat'].nunique()}; grid cells in maps: {len(tables['grid'])}")
    print("\nCounty-month climatology (wettest rows):")
    top = (
        pm.groupby(["county", "month"], observed=True)["hit_weeks"]
        .sum()
        .reset_index()
        .merge(cm, on=["county", "month"], how="left")
        .sort_values("hit_weeks", ascending=False)
        .head(10)
    )
    print(top.to_string(index=False))
    print("\nPeak flood month per county:")
    for county, sub in pm.groupby("county", observed=True):
        peak = sub.groupby("month")["hit_weeks"].sum().idxmax()
        print(f"  {county:<14} peak month {peak:>2} ({sub.groupby('month')['hit_weeks'].sum().max()} pixel-weeks)")


if __name__ == "__main__":
    main()