"""Weekly tabular feature matrix for Task A (county flood extent).

One row per (county, Monday week) for the full record 2000-2025, with
**leakage-safe** features: every feature used for a target week starting
Monday ``m`` is dated at most ``m - EMBARGO_DAYS`` (Friday of the previous
week). A 3-day composite dated ``d`` covers ``d-2..d`` and the first
composite of the target week is dated ``m``, so a feature dated inside
``m-2..m`` would peek into the label window.

Signal sources (all real, from the repo loaders — see MODEL_RESEARCH.md
section 6):

- ERA5 ``tp`` / ``ro`` [mm/day] averaged over a target box and a 3 deg west
  "upstream" band, as rolling sums of 1/3/7/14/30 days.
  * aweiL scope: box = union of the 5 Aweil counties (same box as the
    county EDA);
  * national scope: box = bounding box of all 79 counties (one shared,
    coarse signal — per-county ERA5 boxes are a documented follow-up).
- Dartmouth gauge 100205 discharge [m3/s] (the only gauge in South Sudan):
  level, 3/7/14-day means, 7-day change. Gaps are forward-filled up to 7
  days, then left as NaN (LightGBM handles NaN natively).
- Lake Albert altimetry level [m]: level, 7/14-day means, 7-day change.
- AgERA5 reference evapotranspiration (Aweil point): 7/14/30-day means and
  the 7/14-day net moisture (rainfall minus ET).
- The county's own observed history: areas and detection flags of the
  previous 1-4 weeks plus a coverage proxy (detection days last week / 7).
- Calendar: month, month sin/cos, week of year.

Labels: the weekly *union* detected area (km2) per county from the committed
real-data EDA tables (``EDA_flood_masks/outputs/...``), reindexed onto the
full Monday calendar (missing week = 0 km2). If the EDA tables are ever
regenerated differently, this module simply picks up the new file.

This module never invents data: missing raw inputs raise with a pointer to
``modeling.data_check``.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from modeling import config
from modeling.splits import week_split_map, weekly_index

_era5_cache: dict[tuple, xr.Dataset] = {}

__all__ = [
    "FEATURE_COLUMNS",
    "box_union",
    "build_features",
    "build_labels",
    "county_boxes",
    "daily_signals",
    "era5_box_daily",
    "load_albert_level",
    "load_boundaries",
    "load_era5_dataset",
    "load_et0_daily",
    "load_gauge_daily",
    "upstream_box",
]


def _silence(fn, *args, **kwargs):
    """Run a chatty loader (it prints headers) without polluting our output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


# --- Spatial helpers -------------------------------------------------------------


def load_boundaries():
    """All Admin-2 counties (``state, county, county_code, county_area_km2``)."""
    from EDA_flood_masks import national_flood_eda

    if not config.BOUNDARY_ADMIN2.exists():
        raise FileNotFoundError(
            f"missing {config.BOUNDARY_ADMIN2} — run `python -m modeling.data_check`"
        )
    return national_flood_eda.load_south_sudan_counties(config.BOUNDARY_ADMIN2)


def county_boxes(counties) -> dict[str, dict[str, float]]:
    """One {lat_min, lat_max, lon_min, lon_max} box per county (bbox)."""
    boxes = {}
    for name, geom in zip(counties["county"], counties.geometry):
        (lon_min, lat_min, lon_max, lat_max) = geom.bounds
        boxes[name] = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    return boxes


def box_union(boxes: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "lat_min": min(b["lat_min"] for b in boxes.values()),
        "lat_max": max(b["lat_max"] for b in boxes.values()),
        "lon_min": min(b["lon_min"] for b in boxes.values()),
        "lon_max": max(b["lon_max"] for b in boxes.values()),
    }


def upstream_box(box: dict[str, float]) -> dict[str, float]:
    """A 3-deg band directly west of the target box (upstream in Bahr el Ghazal)."""
    return {
        "lat_min": box["lat_min"],
        "lat_max": box["lat_max"],
        "lon_min": box["lon_min"] - config.UPSTREAM_OFFSET_DEG,
        "lon_max": box["lon_min"],
    }


def target_boxes(scope: str) -> tuple[dict[str, float], dict[str, float]]:
    """(local box, upstream box) for a scope."""
    counties = load_boundaries()
    if scope == config.AWEIL_SCOPE:
        boxes = {c: b for c, b in county_boxes(counties).items() if c in config.AWEIL_COUNTIES}
    else:
        boxes = county_boxes(counties)
    local = box_union(boxes)
    return local, upstream_box(local)


# --- Raw signal loaders (real data only) -------------------------------------------


def load_era5_dataset(years: list[int] | None = None) -> xr.Dataset:
    """ERA5 daily tp/ro over the Nile basin, cached per year-set."""
    from processing_data import loading

    years = years or config.YEARS
    key = tuple(years)
    if key not in _era5_cache:
        _era5_cache[key] = loading.load_rainfall_runoff(np.array(years))
    return _era5_cache[key]


def era5_box_daily(dataset: xr.Dataset, box: dict[str, float]) -> pd.DataFrame:
    """Daily box-mean rainfall/runoff in mm — reuses the EDA's own function."""
    from EDA_hydrometeorology import hydrological_analysis

    table = hydrological_analysis.era5_daily_table(dataset, box)
    table["date"] = pd.to_datetime(table["date"])
    return table.set_index("date").sort_index()


def load_gauge_daily() -> pd.Series:
    """Daily discharge [m3/s] of the only South Sudan gauge (Dartmouth 100205)."""
    from processing_data import loading

    if not config.GAUGE_ROOT.exists():
        raise FileNotFoundError(f"missing {config.GAUGE_ROOT} — run `python -m modeling.data_check`")
    data = _silence(loading.load_dartmouth_data)
    if config.GAUGE_AREA_ID not in data:
        raise KeyError(
            f"gauge {config.GAUGE_AREA_ID} not found; available: {sorted(data)[:10]}..."
        )
    s = data[config.GAUGE_AREA_ID].iloc[:, 0].rename("gauge")
    s = s[~s.index.duplicated()].sort_index()
    return s


def load_albert_level() -> pd.Series:
    """Daily water level [m] of Lake Albert (DAHITI altimetry)."""
    from processing_data import loading

    if not config.LAKE_ROOT.exists():
        raise FileNotFoundError(f"missing {config.LAKE_ROOT} — run `python -m modeling.data_check`")
    lakes = _silence(loading.load_lake_stations)
    df = lakes["Albert"]
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        raise ValueError("Albert lake frame has no numeric column")
    s = df[numeric[0]].rename("albert")
    s = s[~s.index.duplicated()].sort_index()
    return s


def load_et0_daily(years: list[int]) -> pd.Series:
    """Daily reference evapotranspiration [mm/day] at the Aweil grid point.

    Reads the CSVs produced by ``process_ET``; years without a processed file
    simply come back as NaN (the EDA pipeline produces 2000-2025).
    """
    from processing_data import loading

    if not config.ET0_ROOT.exists():
        raise FileNotFoundError(
            f"missing {config.ET0_ROOT} — run `process_ET` (see processing_data/loading.py)"
        )
    frames = _silence(
        loading.load_processed_ET, np.array(years), **config.ET0_TARGET
    )
    if not frames:
        raise FileNotFoundError(
            "no processed ET0 files found — run process_ET for the required years"
        )
    parts = []
    for df in frames.values():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        parts.append(d.set_index("date")["gridcell"].rename("et0"))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


# --- Daily signal table with the embargo ------------------------------------------


def daily_signals(
    scope: str, years: list[int] | None = None
) -> pd.DataFrame:
    """Daily raw signals on a common index, from the warm-up day through the
    last feature-cutoff day.

    Columns: local_tp, local_ro, up_tp, up_ro, gauge, albert, et0 (all in
    mm/day except gauge [m3/s] and albert [m]).
    """
    years = years or config.YEARS
    local, upstream = target_boxes(scope)
    ds = load_era5_dataset(years)
    local_daily = era5_box_daily(ds, local)
    up_daily = era5_box_daily(ds, upstream)

    last_cutoff = weekly_index()[-1] - pd.Timedelta(days=config.EMBARGO_DAYS)
    idx = pd.date_range(pd.Timestamp(years[0], month=1, day=1) - pd.Timedelta(days=31), last_cutoff)

    out = local_daily.reindex(idx)[["tp", "ro"]].rename(
        columns={"tp": "local_tp", "ro": "local_ro"}
    )
    out = out.join(up_daily.reindex(idx)[["tp", "ro"]].rename(
        columns={"tp": "up_tp", "ro": "up_ro"}
    ))

    gauge = load_gauge_daily().reindex(idx).ffill(limit=7)
    out = out.join(gauge)
    albert = load_albert_level().reindex(idx)
    out = out.join(albert)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = out.join(load_et0_daily(years).reindex(idx))
    return out


def _rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Rolling sums/means of the daily signals, one row per calendar day."""
    out = pd.DataFrame(index=daily.index)
    for col, kind in (
        ("local_tp", "sum"),
        ("local_ro", "sum"),
        ("up_tp", "sum"),
        ("up_ro", "sum"),
    ):
        for w in config.ROLL_WINDOWS:
            out[f"{col}_w{w}"] = (
                daily[col].rolling(w, min_periods=w).sum()
                if kind == "sum"
                else daily[col].rolling(w, min_periods=3).mean()
            )
    for col in ("gauge", "albert"):
        for w in (1, 3, 7, 14):
            out[f"{col}_w{w}"] = daily[col].rolling(w, min_periods=min(3, w)).mean()
        out[f"{col}_chg7"] = daily[col] - daily[col].shift(7)
    for w in (7, 14, 30):
        out[f"et0_w{w}"] = daily["et0"].rolling(w, min_periods=min(3, w)).mean()
    out["net_w7"] = out["local_tp_w7"] - out["et0_w7"]
    out["net_w14"] = out["local_tp_w14"] - out["et0_w14"]
    return out


# --- Labels ---------------------------------------------------------------------


def _full_weeks() -> pd.DatetimeIndex:
    return weekly_index()


def build_labels(scope: str) -> pd.DataFrame:
    """Weekly union detected area per county on the full Monday calendar.

    Returns columns: county, week, cutoff, y_true (km2, 0 = no detection),
    y_det (0/1), detected_dates (composite days with a detection that week).
    """
    path = (
        config.AWAIL_WEEKLY_FLOODS_CSV
        if scope == config.AWEIL_SCOPE
        else config.NATIONAL_WEEKLY_FLOODS_CSV
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing committed EDA table {path} — run the EDA scripts first"
        )
    raw = pd.read_csv(path)
    raw["week"] = pd.to_datetime(raw["week"])

    weeks = _full_weeks()
    if scope == config.AWEIL_SCOPE:
        counties = config.AWEIL_COUNTIES
    else:
        counties = sorted(raw["county"].unique())

    label = raw.rename(columns={"detected_area_km2": "y_true", "detected_pixel_days": "detected_dates"})
    label = label[["county", "week", "y_true", "detected_dates"]].drop_duplicates()
    label = label.set_index(["county", "week"])

    grid = pd.MultiIndex.from_product(
        [counties, weeks], names=["county", "week"]
    )
    out = label.reindex(grid, fill_value=0.0).reset_index()
    out["y_true"] = out["y_true"].clip(lower=0.0)
    out["y_det"] = (out["y_true"] > 0).astype(int)
    out["cutoff"] = out["week"] - pd.Timedelta(days=config.EMBARGO_DAYS)
    return out.sort_values(["county", "week"]).reset_index(drop=True)


# --- The feature matrix -------------------------------------------------------------


def build_features(scope: str = config.AWEIL_SCOPE) -> pd.DataFrame:
    """The full Task A matrix: one row per (county, week), features embargoed.

    Sorted by (county, week). The last column group is the target and split.
    """
    labels = build_labels(scope)
    daily = daily_signals(scope)
    rolling = _rolling_features(daily)

    # Feature values as of each week's cutoff date.
    cutoffs = labels["cutoff"].unique()
    feats = rolling.reindex(cutoffs)
    feats = feats.reset_index().rename(columns={"index": "cutoff"})

    frame = labels.merge(feats, on="cutoff", validate="many_to_one")

    # Own-history features (safe: strictly earlier weeks).
    for lag in config.OWN_LAGS:
        frame[f"y_true_lag{lag}"] = (
            frame.groupby("county", observed=True)["y_true"].shift(lag).fillna(0.0)
        )
        frame[f"y_det_lag{lag}"] = (
            frame.groupby("county", observed=True)["y_det"].shift(lag).fillna(0)
        )
    frame["det_dates_prev"] = (
        frame.groupby("county", observed=True)["detected_dates"].shift(1).fillna(0)
    )
    frame["coverage_prev"] = frame["det_dates_prev"] / 7.0

    # Calendar features.
    frame["year"] = frame["week"].dt.year
    frame["month"] = frame["week"].dt.month
    frame["week_of_year"] = frame.groupby("year", observed=True).cumcount() + 1
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    # Split + boundary purge: drop the first week of val/test (its first
    # composite straddles the previous split's label window).
    uniq_weeks = pd.DatetimeIndex(frame["week"].unique())
    split_map = week_split_map(uniq_weeks)
    frame = frame[frame["week"].map(split_map).notna()].copy()
    frame["split"] = frame["week"].map(split_map)
    return frame


def add_spike_threshold(features: pd.DataFrame) -> pd.DataFrame:
    """Per-county 95th percentile of the TRAIN-period area (spike flag)."""
    out = features.copy()

    def _thr(s: pd.Series) -> float:
        tr = s[s["split"] == "train"]
        return float(tr["y_true"].quantile(0.95)) if len(tr) else 0.0

    thresholds = out.groupby("county", observed=True).apply(
        _thr, include_groups=False
    )
    out["spike_threshold"] = out["county"].map(thresholds).fillna(0.0)
    return out


FEATURE_COLUMNS = [
    "local_tp_w1", "local_tp_w3", "local_tp_w7", "local_tp_w14", "local_tp_w30",
    "local_ro_w1", "local_ro_w3", "local_ro_w7", "local_ro_w14", "local_ro_w30",
    "up_tp_w1", "up_tp_w3", "up_tp_w7", "up_tp_w14", "up_tp_w30",
    "up_ro_w1", "up_ro_w3", "up_ro_w7", "up_ro_w14", "up_ro_w30",
    "gauge_w1", "gauge_w3", "gauge_w7", "gauge_w14", "gauge_chg7",
    "albert_w1", "albert_w3", "albert_w7", "albert_w14", "albert_chg7",
    "et0_w7", "et0_w14", "et0_w30", "net_w7", "net_w14",
    "y_true_lag1", "y_true_lag2", "y_true_lag3", "y_true_lag4",
    "y_det_lag1", "y_det_lag2", "y_det_lag3", "y_det_lag4",
    "det_dates_prev", "coverage_prev",
    "year", "month", "week_of_year", "month_sin", "month_cos",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=config.SCOPES, default=config.AWEIL_SCOPE)
    args = parser.parse_args()

    print(f"Building features for scope={args.scope} ...")
    feats = build_features(args.scope)
    print(f"rows={len(feats)}, cols={len(feats.columns)}")
    print(feats[["county", "week", "split", "y_true"]].head(10).to_string())
    print("NaN share per feature (train rows):")
    train = feats[feats["split"] == "train"]
    print(train[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False).head(12).round(3).to_string())
