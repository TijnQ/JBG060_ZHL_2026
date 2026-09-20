"""Data-quality EDA helpers for the hydrometeorological forecasting inputs.

This module profiles the inputs used by ``hydrometeorology_eda.ipynb`` in the
same spirit as the flood-mask EDA profiles the flood masks: missing values, value
ranges, cardinality and distributions, but for each hydrometeorological
*source* that will feed a flood-prediction model:

- ERA5 precipitation and runoff, spatially averaged over the whole Nile-basin
  grid (``ERA5 - Nile basin``) and over the Aweil study counties
  (``ERA5 - Aweil counties``).
- AgERA5 reference evapotranspiration at the processed grid cell.
- Dartmouth river discharge for every loaded station.
- Satellite altimetry lake levels (Victoria, Kyoga, Albert).

Besides the flood-style ``column profile``, ``coverage`` and ``quality`` tables
it returns two views written with the eventual machine-learning step in mind:
per-column distribution extras (zeros %, negatives %, skew) and a
``supervised-set readiness`` table that counts how many calendar days provide
all features and a flood label together. A high share of *complete* days means
almost any model (gradient boosting, linear, neural) can be trained on a clean
daily matrix; heavy gaps push us toward gap-tolerant tree models or explicit
imputation before linear / neural models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from data_quality.eda_quality_flood.flood_eda_data import profile_dataframe
from EDA_hydrometeorology.hydrological_analysis import (
    era5_daily_table,
    et_daily_table,
    load_aweil_flood_observations,
    quality_report,
)

# The AgERA5 reference-evapotranspiration grid cell used across the project.
ET_LON = 30.725
ET_LAT = 9.475

_DISTRIBUTION_COLUMNS = ("zero_pct", "negative_pct", "skew")


def _numeric_value_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that hold numeric data (everything except the date column)."""
    return [
        column for column in frame.columns
        if column != "date" and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _date_series(frame: pd.DataFrame) -> pd.Series:
    """Normalised, sorted, de-duplicated date series for a source frame."""
    return pd.to_datetime(frame["date"]).dt.normalize().dropna().drop_duplicates().sort_values()


def _study_window(frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start/end of the ERA5 record, which defines the forecast study period.

    Coverage, quality and readiness all compare against this window rather than
    the union of every source's dates, so the numbers stay meaningful whenever
    the notebook runs a subset of years (and are not distorted by lake/station
    series that span beyond the ERA5 record).
    """
    era5_dates = pd.DatetimeIndex(sorted({
        date for level, frame in frames.items()
        if level.startswith("ERA5") for date in _date_series(frame)
    }))
    if len(era5_dates):
        return era5_dates.min(), era5_dates.max()
    all_dates = pd.DatetimeIndex(sorted({
        date for frame in frames.values() for date in _date_series(frame)
    }))
    return all_dates.min(), all_dates.max()


def era5_basin_frames(rainfall: xr.Dataset) -> dict[str, pd.DataFrame]:
    """Daily basin-mean ERA5 precipitation and runoff over the whole grid."""
    table = era5_daily_table(rainfall)  # date, precipitation_mm, runoff_mm
    return {
        "ERA5 - Nile basin (precipitation)": table[["date", "precipitation_mm"]],
        "ERA5 - Nile basin (runoff)": table[["date", "runoff_mm"]],
    }


def era5_aweil_frames(rainfall: xr.Dataset, bounds) -> dict[str, pd.DataFrame]:
    """Daily ERA5 precipitation and runoff averaged over an area's bounding box."""
    bbox = {
        "lat_min": bounds[1], "lat_max": bounds[3],
        "lon_min": bounds[0], "lon_max": bounds[2],
    }
    table = era5_daily_table(rainfall, bbox)
    return {
        "ERA5 - Aweil counties (precipitation)": table[["date", "precipitation_mm"]],
        "ERA5 - Aweil counties (runoff)": table[["date", "runoff_mm"]],
    }


def build_source_frames(
    rainfall: xr.Dataset,
    et_data,
    dartmouth: dict[int, pd.DataFrame],
    lakes: dict[str, pd.DataFrame],
    county_bounds=None,
) -> dict[str, pd.DataFrame]:
    """Assemble one tidy ``(date, value)`` frame per hydrometeorological source.

    Parameters
    ----------
    rainfall : xr.Dataset
        The concatenated daily ERA5 dataset from ``processing_data.loading``.
    et_data : dict[int, pd.DataFrame] | pd.DataFrame
        Processed AgERA5 evapotranspiration (``load_processed_ET`` output or a
        single frame).
    dartmouth : dict[int, pd.DataFrame]
        Loaded discharge stations (``load_dartmouth_data`` output).
    lakes : dict[str, pd.DataFrame]
        Loaded lake altimetry (``load_lake_stations`` output).
    county_bounds : iterable, optional
        ``(min_lon, min_lat, max_lon, max_lat)``. When provided, adds the Aweil
        study-area ERA5 levels.

    Returns a ``{level: pd.DataFrame}`` mapping used by every profile helper.
    """
    frames: dict[str, pd.DataFrame] = era5_basin_frames(rainfall)
    if county_bounds is not None:
        frames.update(era5_aweil_frames(rainfall, county_bounds))

    frames["Reference ET (AgERA5 gridcell)"] = et_daily_table(et_data)[["date", "et_mm"]]

    for area_id, table in dartmouth.items():
        discharge = pd.DataFrame({
            "date": pd.to_datetime(table.index).normalize(),
            "discharge_m3s": pd.to_numeric(table["Discharge (m3/s)"], errors="coerce").to_numpy(),
        })
        frames[f"Dartmouth discharge - station {area_id}"] = discharge

    for name, table in lakes.items():
        column = "water_level" if "water_level" in table.columns else "height_wrt_ref"
        frame = pd.DataFrame({
            "date": pd.to_datetime(table.index).normalize(),
            column: pd.to_numeric(table[column], errors="coerce").to_numpy(),
        })
        frames[f"Lake {name} (altimetry)"] = frame

    return frames


def profile_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flood-EDA-style column profile: one row per (level, column).

    Columns match the flood EDA (``non_null_count``, ``nan_count``, ``nan_pct``,
    ``n_unique``, ``min``, ``max``, ``mean``, ``std``). ``date`` rows report
    the observed range; value rows report numeric statistics.
    """
    parts = []
    for level, frame in frames.items():
        work = frame.copy()
        work["date"] = pd.to_datetime(work["date"])
        parts.append(profile_dataframe(work, level))
    return pd.concat(parts, ignore_index=True)


def distribution_extras(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Zero %, negative % and skew for every numeric value column.

    These are the distributional facts that drive feature preprocessing and,
    in turn, model choice: zero-heavy, right-skewed columns (precipitation,
    runoff) motivate log / indicator transforms and favour scale-invariant tree
    models over linear or neural models that assume a nicely scaled input.
    """
    rows = []
    for level, frame in frames.items():
        for column in _numeric_value_columns(frame):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                row = {"level": level, "column": column, "zero_pct": np.nan,
                       "negative_pct": np.nan, "skew": np.nan}
            else:
                row = {"level": level, "column": column,
                       "zero_pct": 100.0 * float((values == 0).mean()),
                       "negative_pct": 100.0 * float((values < 0).mean()),
                       "skew": float(values.skew())}
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Temporal coverage of every source over the ERA5 study window.

    Reports records, distinct dates present and the gap structure per source.
    The window is the ERA5 record (see :func:`_study_window`), so the share of
    window days covered is directly comparable to the modelling period. The
    irregular sources (Dartmouth discharge and lake altimetry) show the long
    gaps where interpolation or date-feature engineering is needed before
    supervised modelling.
    """
    normalized = {level: _date_series(frame) for level, frame in frames.items()}
    study_start, study_end = _study_window(frames)
    window_len = (study_end - study_start).days + 1

    rows = []
    for level, frame in frames.items():
        dates = normalized[level]
        gaps = dates.diff().dt.days.dropna()
        value_cols = _numeric_value_columns(frame)
        value_rows = 0
        if value_cols:
            present = pd.concat(
                [pd.to_numeric(frame[c], errors="coerce") for c in value_cols], axis=1
            ).notna().any(axis=1)
            value_rows = int(present.sum())
        rows.append({
            "level": level,
            "records": len(frame),
            "unique_dates_present": len(dates),
            "first_date": dates.min().strftime("%Y-%m-%d"),
            "last_date": dates.max().strftime("%Y-%m-%d"),
            "window_days": window_len,
            "days_missing_in_window": int(window_len - len(dates)),
            "coverage_pct": 100.0 * len(dates) / window_len,
            "median_gap_days": float(gaps.median()) if len(gaps) else 0.0,
            "longest_gap_days": int(gaps.max()) if len(gaps) else 0,
            "value_rows_non_null": value_rows,
        })
    return pd.DataFrame(rows)


def quality_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-source ``quality_report``: duplicates, missing, negatives, coverage."""
    start, end = _study_window(frames)
    reports = []
    for level, frame in frames.items():
        value_cols = _numeric_value_columns(frame)
        if not value_cols:
            continue
        series = pd.Series(
            pd.to_numeric(frame[value_cols[0]], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["date"]),
        )
        reports.append(quality_report(series, level, start, end))
    return pd.concat(reports, ignore_index=True).rename(columns={"name": "level"})


def all_features_daily(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide daily table of every value column, one column per ``level: column``."""
    parts = {}
    for level, frame in frames.items():
        dates = pd.to_datetime(frame["date"]).dt.normalize()
        for column in _numeric_value_columns(frame):
            # Last value wins for irregular per-pass observations on the same day.
            parts[f"{level}: {column}"] = (
                pd.to_numeric(frame[column], errors="coerce").groupby(dates).last()
            )
    return pd.DataFrame(parts).sort_index()


def supervised_set_readiness(
    frames: dict[str, pd.DataFrame], label_dates=None
) -> pd.DataFrame:
    """Count, per year and in total, usable supervised days.

    A daily row is *usable* when every feature column has a value and (when a
    flood label is supplied) a label day exists. ``label_dates`` is an iterable
    of the days on which a flood was detected.
    """
    wide = all_features_daily(frames)
    study_start, study_end = _study_window(frames)
    window = pd.date_range(study_start, study_end, freq="D")
    feature_values = wide.reindex(window)

    all_features = feature_values.notna().all(axis=1)
    any_feature = feature_values.notna().any(axis=1)
    has_label = pd.Series(False, index=window)
    if label_dates is not None:
        labels = pd.to_datetime(np.asarray(label_dates)).normalize()
        has_label = window.isin(labels)

    table = pd.DataFrame({
        "all_features": all_features, "any_feature": any_feature,
        "has_label": has_label,
        "all_and_label": all_features & has_label,
    })
    table["year"] = table.index.year

    by_year = table.groupby("year").agg(
        days=("all_features", "size"),
        days_all_features=("all_features", "sum"),
        days_any_feature=("any_feature", "sum"),
        days_all_features_with_label=("all_and_label", "sum"),
    ).reset_index()

    total = pd.DataFrame([{
        "year": "all",
        "days": int(table["all_features"].size),
        "days_all_features": int(table["all_features"].sum()),
        "days_any_feature": int(table["any_feature"].sum()),
        "days_all_features_with_label": int((table["all_features"] & table["has_label"]).sum()),
    }])
    return pd.concat([by_year, total], ignore_index=True)


def flood_label_dates(flood_root, counties, years) -> np.ndarray:
    """Dates with an active Aweil flood detection, from the flood-mask files."""
    observations = load_aweil_flood_observations(flood_root, counties, years=years)
    return np.sort(observations["date"].dt.normalize().unique())