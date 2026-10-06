"""Engineered feature set for the LightGBM_v1 pipeline (plan.md: `lightgbm_fe`).

Implements the feature decisions locked in plan.md §3.1 without touching the
frozen v3 feature matrix:

- `county`      — the row's own county as a pandas `category` column (5 levels)
                  → LightGBM native categorical splits. Restores the county
                  dimension the union-box design erased.
- `cnty_tp_w7`  — 7-day sum of ERA5/AgERA5 total precipitation (mm) ending at
                  the Friday cutoff, spatially averaged over the row's OWN
                  county bounding box (reuses features.county_boxes() +
                  data_loader.era5_box_daily(); 5 internal daily series, 1
                  table column — each row carries its own county's value).
- `cnty_ro_w7`  — same as `cnty_tp_w7` for ERA5 `ro` (runoff, mm): the
                  land-surface response, sensitive to antecedent wetness.
- `year`        — REMOVED from the model's input (plan.md §3.1). The column
                  stays in the frame for bookkeeping but is not part of
                  FEATURE_COLUMNS_ENGINEERED, so LightGBM never sees it.

The v3 path stays frozen by construction: `features.FEATURE_COLUMNS` and
`features.build_weekly_features()` are imported, never reassigned or mutated,
so `--model lightgbm`, `--model lightgbm_baseline` and `--model tabpfn` keep
building exactly the frozen v3 matrix. Only `run.py --model lightgbm_fe` and
the `"fe"` entry in `tuning.FEATURE_SETS` consume this module.

Embargo: the engineered features reuse the identical Friday cutoff rule — a
7-day window ending at the cutoff (Friday), i.e. at most m - 3 days before a
target week starting Monday m — so no new leakage surface is introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from LightGBM_v1 import config, data_loader, features

__all__ = [
    "ENGINEERED_ADDITIONS",
    "FEATURE_COLUMNS_ENGINEERED",
    "REMOVED_V3_COLUMNS",
    "build_county_daily_signals",
    "build_weekly_features_fe",
    "compute_county_rolling_features",
    "daily_feature_index",
]

# plan.md §3.1: `year` is dropped from the model input — seasonality is carried
# by month/week_of_year/month_sin/month_cos, interannual forcing by the w30
# rainfall ladders, gauge and lake-level features. Under expanding-window CV a
# calendar index can only project the recent training years forward as a step
# function, which early stopping actively rewards.
REMOVED_V3_COLUMNS = ("year",)

# plan.md §3.1: county identity + county-resolved 7-day precipitation/runoff.
ENGINEERED_ADDITIONS = ("county", "cnty_tp_w7", "cnty_ro_w7")

# Two named column lists, no runtime swapping (plan.md §6): features.FEATURE_COLUMNS
# stays byte-identical and untouched, and this list is DERIVED from it so the
# shared columns can never drift apart (plan.md §4.2 integrity gate). Order:
# the v3 columns minus `year`, then the engineered additions.
FEATURE_COLUMNS_ENGINEERED: list[str] = [
    column for column in features.FEATURE_COLUMNS if column not in REMOVED_V3_COLUMNS
] + list(ENGINEERED_ADDITIONS)


def daily_feature_index(years: list[int] | None = None) -> pd.DatetimeIndex:
    """Continuous daily index the rolling windows are computed on.

    Mirrors the index built inside features.build_daily_signals exactly (31-day
    warm-up before the first year, ending at the last Friday cutoff) so the
    engineered rolling sums see the same daily history — and therefore have the
    same NaN warm-up pattern — as the frozen v3 columns.
    """
    years = years or config.YEARS
    last_cutoff = pd.date_range("2000-01-03", "2025-12-29", freq="W-MON")[-1] - pd.Timedelta(days=config.EMBARGO_DAYS)
    return pd.date_range(pd.Timestamp(years[0], 1, 1) - pd.Timedelta(days=31), last_cutoff)


def build_county_daily_signals(years: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Daily ERA5 tp/ro spatial means over each county's OWN bounding box.

    Reuses features.county_boxes() + data_loader.era5_box_daily(); the ERA5
    dataset comes from the shared in-process cache, so this adds no extra
    NetCDF reads. Returns {county: DataFrame(index=daily dates, columns=[tp, ro])}
    — the +5 internal daily series (tp and ro per county) behind the two new
    table columns.
    """
    years = years or config.YEARS
    boxes = features.county_boxes(data_loader.load_boundaries())
    ds = data_loader.load_era5_dataset(years)
    idx = daily_feature_index(years)

    out: dict[str, pd.DataFrame] = {}
    for name in config.AWEIL_COUNTIES:
        if name not in boxes:
            raise KeyError(f"County '{name}' missing from the Admin-2 boundaries")
        daily = data_loader.era5_box_daily(ds, boxes[name])[["tp", "ro"]]
        out[name] = daily.reindex(idx)
    return out


def compute_county_rolling_features(county_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """7-day rolling sums of the per-county daily series (internal wide frame).

    One column per (variable, county) pair — `cnty_tp_w7::<county>` /
    `cnty_ro_w7::<county>` — collapsed into the single table columns
    `cnty_tp_w7` / `cnty_ro_w7` by build_weekly_features_fe(). Same rolling
    recipe as features.compute_rolling_features (w=7, min_periods=3).
    """
    out: dict[str, pd.Series] = {}
    window = 7
    min_periods = max(1, min(window, 3))
    for name, daily in county_daily.items():
        for var in ("tp", "ro"):
            out[f"cnty_{var}_w7::{name}"] = daily[var].rolling(window, min_periods=min_periods).sum()
    return pd.DataFrame(out)


def _attach_county_columns(frame: pd.DataFrame, county_rolling: pd.DataFrame) -> pd.DataFrame:
    """Give each row its own county's rolling value (wide frame → 2 columns)."""
    county_codes = pd.Categorical(frame["county"], categories=config.AWEIL_COUNTIES).codes.astype(np.int64)
    if (county_codes < 0).any():
        unknown = sorted({str(name) for name in frame.loc[county_codes < 0, "county"].unique()})
        raise ValueError(f"Counties outside config.AWEIL_COUNTIES: {unknown}")

    cutoff_pos = {cutoff: position for position, cutoff in enumerate(county_rolling.index)}
    rows = frame["cutoff"].map(cutoff_pos)
    if rows.isna().any():
        missing = sorted({str(cutoff) for cutoff in frame.loc[rows.isna(), "cutoff"].unique()})
        raise ValueError(f"Cutoffs missing from the county rolling table: {missing}")
    rows = rows.astype(np.int64).to_numpy()

    for var in ("tp", "ro"):
        columns = [f"cnty_{var}_w7::{name}" for name in config.AWEIL_COUNTIES]
        table = county_rolling[columns].to_numpy()
        frame[f"cnty_{var}_w7"] = table[rows, county_codes]
    return frame


def build_weekly_features_fe() -> pd.DataFrame:
    """Build the embargoed weekly feature matrix with the engineered fe columns.

    Starts from the frozen v3 matrix (features.build_weekly_features()) so
    every v3 column stays byte-identical, then:
    1. adds `cnty_tp_w7` / `cnty_ro_w7` (each row carries its own county's
       7-day ERA5 box sum ending at the Friday cutoff),
    2. converts `county` to pandas `category` (LightGBM native categorical
       splits),
    3. recomputes the `complete` flag against FEATURE_COLUMNS_ENGINEERED
       (plan.md §6: completeness is judged against the selected feature list).
    """
    frame = features.build_weekly_features()
    county_rolling = compute_county_rolling_features(build_county_daily_signals())
    frame = _attach_county_columns(frame, county_rolling)

    frame["county"] = pd.Categorical(frame["county"], categories=config.AWEIL_COUNTIES)
    frame["complete"] = ~frame[FEATURE_COLUMNS_ENGINEERED].isna().any(axis=1)

    return frame