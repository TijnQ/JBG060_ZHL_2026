"""Data-quality EDA helpers for the satellite flood-mask records.

The raw flood-mask Parquet files used by ``EDA_flood_masks`` store every
location and three-day period where flooding was detected. They contain about
95 million rows split over two MODIS/VIIRS tiles (``h20v08`` and ``h21v08``)
that together span South Sudan, for both the ``recurring`` and ``unusual``
flood classes.

This module computes the same kind of summary statistics at two spatial levels:

- **Country (South Sudan)**: every record across both tiles. The dataset is far
  too large to load into pandas at once, so we aggregate column statistics with
  PyArrow and only ever materialise one annual Parquet file at a time.
- **Northern Bahr el Ghazal state**: only records whose pixel centre falls
  inside the Northern Bahr el Ghazal admin-1 polygon. We first narrow the files
  to the state's bounding box in PyArrow, then do the point-in-polygon filter
  in GeoPandas because the remaining rows are small enough for pandas.

The goal is basic exploratory statistics (missing values, ranges, means) that
tell us what to expect before modelling, not a finished analysis.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# The same time window and tiles used by EDA_flood_masks and processing_data.
YEARS = range(2000, 2026)                            # 2000-2025 inclusive
TILES = ("h20v08", "h21v08")                         # Together span South Sudan
FLOOD_TYPES = ("recurring", "unusual")
STUDY_STATE = "Northern Bahr el Ghazal"
COUNTRY_LABEL = "Country (South Sudan)"
STATE_LABEL = "Northern Bahr el Ghazal state"

# Numeric columns where min / max / mean / std are meaningful.
_NUMERIC = ("lat", "lon", "cloud_frac")
# Profile-table column order shared by the country and state results.
_PROFILE_COLUMNS = [
    "level", "column", "dtype", "non_null_count", "nan_count", "nan_pct",
    "n_unique", "min", "max", "mean", "std", "top_value", "top_count",
]


def expected_file_count(years=YEARS) -> int:
    """Number of annual Parquet files: 2 tiles x 2 classes x len(years)."""
    return len(TILES) * len(FLOOD_TYPES) * len(years)


def flood_files(flood_root: Path, years=YEARS):
    """Yield ``(path, tile, flood_type, year)`` for every expected file."""
    paths = []
    for flood_type in FLOOD_TYPES:
        for tile in TILES:
            for year in years:
                path = flood_root / f"compact_{flood_type}" / f"flood_events_{tile}_{year}.parquet"
                paths.append((path, tile, flood_type, year))
    return paths


def _read_timestamp_dates(tbl: pa.Table) -> pa.Array:
    """Return the date column normalised to a millisecond timestamp array.

    The annual files store dates in different Arrow encodings (string
    dictionaries in some years, timestamps in others), so we push them all to
    one representation before computing ranges or counting distinct values.
    """
    column = tbl["date"]
    return pc.cast(column, pa.timestamp("ms"))


def country_profile(flood_root: Path, years=YEARS) -> pd.DataFrame:
    """Aggregate column statistics for every flood record in both tiles.

    Returns a tidy one-row-per-column summary for the whole South Sudan
    dataset (the ``Country (South Sudan)`` level). Memory stays bounded: only
    one annual file is read into memory at a time.
    """
    columns = ["date", "lat", "lon", "tile", "cloud_frac"]
    # Per-numeric-column running aggregates.
    numeric = {
        column: {"valid": 0, "null": 0, "min": math.inf, "max": -math.inf,
                 "sum": 0.0, "sum_sq": 0.0}
        for column in _NUMERIC
    }
    # Running aggregates for the remaining columns.
    date_agg = {"valid": 0, "null": 0, "min": None, "max": None, "unique": []}
    tile_agg = {"valid": 0, "null": 0, "unique": set()}
    flood_agg = {"valid": 0, "null": 0, "unique": set()}
    # Lists that receive the per-file unique arrays per numeric column.
    uniques = {column: [] for column in _NUMERIC}

    files_loaded = 0
    for path, tile, flood_type, year in flood_files(flood_root, years):
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected annual flood file: {path}")
        tbl = pq.read_table(path, columns=columns)
        files_loaded += 1

        # tile and flood_type are constant per file, but keep flags aligned.
        tile_arr = tbl["tile"]
        tile_agg["valid"] += pc.count(tile_arr).as_py()
        tile_agg["null"] += pc.count(tile_arr, mode="only_null").as_py()
        tile_agg["unique"].add(tile)
        flood_agg["valid"] += len(tile_arr)
        flood_agg["unique"].add(flood_type)

        # Numeric columns: running count, null count, min, max, sum, sum of sq.
        for column in _NUMERIC:
            arr = tbl[column]
            acc = numeric[column]
            valid = pc.count(arr).as_py()
            acc["valid"] += valid
            acc["null"] += pc.count(arr, mode="only_null").as_py()
            if valid:
                acc["min"] = min(acc["min"], pc.min(arr).as_py())
                acc["max"] = max(acc["max"], pc.max(arr).as_py())
                acc["sum"] += pc.sum(arr, min_count=0).as_py()
                squared = pc.multiply(arr, arr)
                acc["sum_sq"] += pc.sum(squared, min_count=0).as_py()
            uniques[column].append(pc.unique(arr))

        # Date column: normalise to timestamps and accumulate range + uniques.
        date_arr = _read_timestamp_dates(tbl)
        date_agg["valid"] += pc.count(date_arr).as_py()
        date_agg["null"] += pc.count(date_arr, mode="only_null").as_py()
        if pc.count(date_arr).as_py():
            day_min = pc.min(date_arr).as_py()
            day_max = pc.max(date_arr).as_py()
            date_agg["min"] = day_min if date_agg["min"] is None else min(date_agg["min"], day_min)
            date_agg["max"] = day_max if date_agg["max"] is None else max(date_agg["max"], day_max)
        date_agg["unique"].append(pc.unique(date_arr))

    if not files_loaded:
        raise ValueError("No flood-mask files were loaded; nothing to profile.")

    rows = []

    def emit(column, dtype, valid, null, n_unique, **extra):
        total = valid + null
        nan_pct = (100.0 * null / total) if total else float("nan")
        row = {
            "level": COUNTRY_LABEL, "column": column, "dtype": dtype,
            "non_null_count": valid, "nan_count": null, "nan_pct": nan_pct,
            "n_unique": n_unique, "min": None, "max": None, "mean": None,
            "std": None, "top_value": None, "top_count": None,
        }
        row.update(extra)
        rows.append(row)

    # Numeric columns.
    for column in _NUMERIC:
        acc = numeric[column]
        distinct = _combined_n_unique(uniques[column])
        extra = {}
        if acc["valid"]:
            mean = acc["sum"] / acc["valid"]
            # Sample standard deviation via sum of squares.
            variance = (acc["sum_sq"] - acc["sum"] ** 2 / acc["valid"]) / (acc["valid"] - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
            extra = {"min": acc["min"], "max": acc["max"], "mean": mean, "std": std}
        emit(column, "double", acc["valid"], acc["null"], distinct, **extra)

    # Date (timestamps) column: min / max formatted as date strings.
    emit(
        "date", "timestamp[ms]", date_agg["valid"], date_agg["null"],
        _combined_n_unique(date_agg["unique"]),
        min=_fmt_date(date_agg["min"]), max=_fmt_date(date_agg["max"]),
    )

    # Tile column (constant small set).
    tile_values = ", ".join(sorted(tile_agg["unique"]))
    emit(
        "tile", "string", tile_agg["valid"], tile_agg["null"], len(tile_agg["unique"]),
        top_value=tile_values, top_count=tile_agg["valid"],
    )

    # flood_type column (constant small set).
    flood_values = ", ".join(sorted(flood_agg["unique"]))
    emit(
        "flood_type", "string", flood_agg["valid"], flood_agg["null"],
        len(flood_agg["unique"]), top_value=flood_values, top_count=flood_agg["valid"],
    )

    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)


def _combined_n_unique(arrays: list[pa.Array]) -> int:
    """Number of distinct values across a list of per-file Arrow arrays."""
    valid = [a for a in arrays if a.type != pa.null()]
    if not valid:
        return 0
    base = valid[0].type
    combined = pc.unique(pa.concat_arrays([pc.cast(a, base) for a in valid]))
    return pc.count(combined, mode="only_valid").as_py() or 0


def _fmt_date(value) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_state_boundary(boundaries_path: Path, state_name: str = STUDY_STATE) -> gpd.GeoDataFrame:
    """Return the admin-1 polygon for the requested state in EPSG:4326."""
    admin1 = gpd.read_file(boundaries_path).to_crs("EPSG:4326")
    state = admin1.loc[admin1["adm1_name"].eq(state_name)].copy()
    if state.empty:
        raise ValueError(f"No admin-1 feature found for {state_name!r}")
    return state[["adm1_name", "area_sqkm", "geometry"]].reset_index(drop=True)


def load_state_records(
    flood_root: Path,
    boundaries_path: Path,
    years=YEARS,
    state_name: str = STUDY_STATE,
) -> pd.DataFrame:
    """Load flood records whose pixel centre lies inside the state polygon.

    The files are first narrowed to the state's bounding box in PyArrow, then
    the pixel centres are matched with a point-in-polygon test in GeoPandas.
    Returns a pandas DataFrame with ``date``, ``lat``, ``lon``, ``tile``,
    ``cloud_frac`` and ``flood_type``.
    """
    state = load_state_boundary(boundaries_path, state_name)
    min_lon, min_lat, max_lon, max_lat = state.total_bounds
    bbox_filter = (
        (ds.field("lon") >= min_lon) & (ds.field("lon") <= max_lon)
        & (ds.field("lat") >= min_lat) & (ds.field("lat") <= max_lat)
    )
    frames = []
    for path, tile, flood_type, year in flood_files(flood_root, years):
        frame = ds.dataset(path, format="parquet").to_table(
            columns=["date", "lat", "lon", "tile", "cloud_frac"], filter=bbox_filter
        ).to_pandas()
        frame["flood_type"] = flood_type
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    records["date"] = pd.to_datetime(records["date"], errors="raise")

    cells = records[["lat", "lon"]].drop_duplicates()
    points = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326"
    )
    inside = gpd.sjoin(points, state[["geometry"]], how="inner", predicate="within")
    inside = inside.drop_duplicates(["lat", "lon"])
    return records.merge(inside[["lat", "lon"]], on=["lat", "lon"], validate="many_to_one")


def profile_dataframe(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Build a tidy one-row-per-column summary for a (small) pandas DataFrame.

    Used for the state subset, where the number of rows is small enough to hold
    in memory. Returns the same columns as :func:`country_profile`, so the two
    can be concatenated into one table.
    """
    rows = []
    for column in df.columns:
        series = df[column]
        total = len(series)
        valid = int(series.notna().sum())
        null = int(total - valid)
        nan_pct = (100.0 * null / total) if total else float("nan")
        row = {
            "level": level, "column": column, "dtype": str(series.dtype),
            "non_null_count": valid, "nan_count": null, "nan_pct": nan_pct,
            "n_unique": int(series.nunique(dropna=True)),
            "min": None, "max": None, "mean": None, "std": None,
            "top_value": None, "top_count": None,
        }
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            if values.notna().any():
                row["min"] = float(values.min())
                row["max"] = float(values.max())
                row["mean"] = float(values.mean())
                row["std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        elif pd.api.types.is_datetime64_any_dtype(series):
            times = pd.to_datetime(series, errors="coerce")
            if times.notna().any():
                row["min"] = times.min().strftime("%Y-%m-%d")
                row["max"] = times.max().strftime("%Y-%m-%d")
        else:
            counts = series.value_counts(dropna=True)
            if not counts.empty:
                row["top_value"] = str(counts.index[0])
                row["top_count"] = int(counts.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)


def build_profiles(flood_root: Path, boundaries_path: Path, years=YEARS) -> pd.DataFrame:
    """Combine the country and state column profiles into one tidy DataFrame."""
    country = country_profile(flood_root, years=years)
    state_records = load_state_records(flood_root, boundaries_path, years=years)
    state = profile_dataframe(state_records, level=STATE_LABEL)
    return pd.concat([country, state], ignore_index=True)


def area_overview(flood_root: Path, boundaries_path: Path, years=YEARS) -> pd.DataFrame:
    """Summarise how many rows, pixels, dates and files each level covers."""
    file_rows = 0
    for path, tile, flood_type, year in flood_files(flood_root, years):
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected annual flood file: {path}")
        file_rows += int(pq.read_metadata(path).num_rows)

    # Country-derived values come from the arrow aggregation.
    country_prof = country_profile(flood_root, years=years)
    country_pixels = country_prof.loc[country_prof["column"].eq("lat"), "n_unique"].iloc[0]
    country_dates = country_prof.loc[country_prof["column"].eq("date"), "n_unique"].iloc[0]
    country_min = country_prof.loc[country_prof["column"].eq("date"), "min"].iloc[0]
    country_max = country_prof.loc[country_prof["column"].eq("date"), "max"].iloc[0]

    state_records = load_state_records(flood_root, boundaries_path, years=years)
    rows = [
        {
            "level": COUNTRY_LABEL, "rows": file_rows,
            "unique_pixels": country_pixels, "unique_dates": country_dates,
            "first_date": country_min, "last_date": country_max,
            "files": expected_file_count(years),
        },
        {
            "level": STATE_LABEL, "rows": len(state_records),
            "unique_pixels": state_records[["lat", "lon"]].drop_duplicates().shape[0],
            "unique_dates": int(state_records["date"].nunique()),
            "first_date": _fmt_date(state_records["date"].min()) if len(state_records) else None,
            "last_date": _fmt_date(state_records["date"].max()) if len(state_records) else None,
            "files": expected_file_count(years),
        },
    ]
    return pd.DataFrame(rows)


def count_state_records(flood_root: Path, boundaries_path: Path, years=YEARS) -> int:
    """Number of flood records with pixel centres inside the state polygon."""
    return len(load_state_records(flood_root, boundaries_path, years=years))