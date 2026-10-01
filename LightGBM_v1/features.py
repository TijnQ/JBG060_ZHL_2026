"""Feature engineering module for LightGBM_v1 pipeline.

Constructs weekly tabular feature matrices for the 5 Aweil counties.
Enforces strict label embargo: every feature dated for a target week starting Monday m
is dated at most m - 3 days (Friday of previous week).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from LightGBM_v1 import config, data_loader
from LightGBM_v1.splits import assign_split, purge_boundary_weeks

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


def county_boxes(counties_gdf: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Bounding box {lat_min, lat_max, lon_min, lon_max} per county."""
    boxes = {}
    for name, geom in zip(counties_gdf["county"], counties_gdf.geometry):
        lon_min, lat_min, lon_max, lat_max = geom.bounds
        boxes[name] = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    return boxes


def box_union(boxes: dict[str, dict[str, float]]) -> dict[str, float]:
    """Union bounding box covering all counties."""
    return {
        "lat_min": min(b["lat_min"] for b in boxes.values()),
        "lat_max": max(b["lat_max"] for b in boxes.values()),
        "lon_min": min(b["lon_min"] for b in boxes.values()),
        "lon_max": max(b["lon_max"] for b in boxes.values()),
    }


def upstream_box(box: dict[str, float]) -> dict[str, float]:
    """Upstream 3-degree west bounding box in Bahr el Ghazal basin."""
    return {
        "lat_min": box["lat_min"],
        "lat_max": box["lat_max"],
        "lon_min": box["lon_min"] - config.UPSTREAM_OFFSET_DEG,
        "lon_max": box["lon_min"],
    }


def build_daily_signals(years: list[int] | None = None) -> pd.DataFrame:
    """Build continuous daily hydro-meteorological signals table."""
    years = years or config.YEARS
    counties_gdf = data_loader.load_boundaries()
    boxes = county_boxes(counties_gdf)
    local_b = box_union(boxes)
    upstream_b = upstream_box(local_b)

    ds = data_loader.load_era5_dataset(years)
    local_daily = data_loader.era5_box_daily(ds, local_b)
    up_daily = data_loader.era5_box_daily(ds, upstream_b)

    last_cutoff = pd.date_range("2000-01-03", "2025-12-29", freq="W-MON")[-1] - pd.Timedelta(days=config.EMBARGO_DAYS)
    idx = pd.date_range(pd.Timestamp(years[0], 1, 1) - pd.Timedelta(days=31), last_cutoff)

    out = local_daily.reindex(idx)[["tp", "ro"]].rename(columns={"tp": "local_tp", "ro": "local_ro"})
    out = out.join(up_daily.reindex(idx)[["tp", "ro"]].rename(columns={"tp": "up_tp", "ro": "up_ro"}))

    gauge = data_loader.load_gauge_daily().reindex(idx).ffill(limit=7)
    out = out.join(gauge)
    albert = data_loader.load_albert_level().reindex(idx)
    out = out.join(albert)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = out.join(data_loader.load_et0_daily(years).reindex(idx))

    return out


def compute_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling window aggregations on daily signal features."""
    out = pd.DataFrame(index=daily.index)

    for col in ("local_tp", "local_ro", "up_tp", "up_ro"):
        for w in config.ROLL_WINDOWS:
            out[f"{col}_w{w}"] = daily[col].rolling(w, min_periods=max(1, min(w, 3))).sum()

    for col in ("gauge", "albert"):
        for w in (1, 3, 7, 14):
            out[f"{col}_w{w}"] = daily[col].rolling(w, min_periods=min(3, w)).mean()
        out[f"{col}_chg7"] = daily[col] - daily[col].shift(7)

    for w in (7, 14, 30):
        out[f"et0_w{w}"] = daily["et0"].rolling(w, min_periods=min(3, w)).mean()

    out["net_w7"] = out["local_tp_w7"] - out["et0_w7"]
    out["net_w14"] = out["local_tp_w14"] - out["et0_w14"]

    return out


def build_weekly_features() -> pd.DataFrame:
    """Build the embargoed weekly feature matrix for LightGBM_v1."""
    labels = data_loader.load_flood_labels()
    daily = build_daily_signals()
    rolling = compute_rolling_features(daily)

    cutoffs = labels["cutoff"].unique()
    feats = rolling.reindex(cutoffs).reset_index().rename(columns={"index": "cutoff"})

    frame = labels.merge(feats, on="cutoff", validate="many_to_one")

    # Lag features from county's own history
    for lag in config.OWN_LAGS:
        frame[f"y_true_lag{lag}"] = frame.groupby("county", observed=True)["y_true"].shift(lag).fillna(0.0)
        frame[f"y_det_lag{lag}"] = frame.groupby("county", observed=True)["y_det"].shift(lag).fillna(0)

    frame["det_dates_prev"] = frame.groupby("county", observed=True)["detected_dates"].shift(1).fillna(0)
    frame["coverage_prev"] = frame["det_dates_prev"] / 7.0

    # Calendar features
    frame["year"] = frame["week"].dt.year
    frame["month"] = frame["week"].dt.month
    frame["week_of_year"] = frame.groupby("year", observed=True).cumcount() + 1
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    # Split assignment and boundary purging
    uniq_weeks = pd.DatetimeIndex(frame["week"].unique())
    split_series = assign_split(uniq_weeks)
    keep_mask = purge_boundary_weeks(uniq_weeks, split_series)

    valid_weeks = uniq_weeks[keep_mask.to_numpy()]
    frame = frame[frame["week"].isin(valid_weeks)].copy()
    frame["split"] = frame["week"].map(dict(zip(uniq_weeks, split_series)))

    return frame.sort_values(["county", "week"]).reset_index(drop=True)
