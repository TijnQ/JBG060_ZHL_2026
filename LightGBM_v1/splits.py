"""Temporal splits and 5-fold 5-year block definitions for LightGBM_v1 pipeline.

Implements 5 folds of 5 years each (3 years train, 1 year val, 1 year test),
3-day feature embargo cutoff, and boundary week purging.
"""

from __future__ import annotations

import pandas as pd
from LightGBM_v1 import config

__all__ = [
    "assign_split",
    "feature_cutoff",
    "get_5fold_5year_splits",
    "get_temporal_splits",
    "purge_boundary_weeks",
    "weekly_index",
]


def feature_cutoff(week_start: pd.Timestamp) -> pd.Timestamp:
    """Return latest legal date for features used to forecast week_start (Friday)."""
    if week_start.weekday() != 0:
        raise ValueError(f"week_start must be a Monday timestamp, got: {week_start}")
    return week_start - pd.Timedelta(days=config.EMBARGO_DAYS)


def weekly_index(start: str = "2000-01-03", end: str = "2024-12-30") -> pd.DatetimeIndex:
    """Return sequence of Monday week-start dates for full record."""
    return pd.date_range(start=start, end=end, freq="W-MON")


def assign_split(weeks: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Map week timestamps to temporal split names: 'train', 'val', or 'test'."""
    weeks_idx = pd.DatetimeIndex(weeks)
    years = weeks_idx.year
    out = pd.Series(index=weeks_idx, dtype=object)
    for name, (lo, hi) in config.SPLITS.items():
        out[years.isin(range(lo, hi + 1))] = name
    return out


def purge_boundary_weeks(weeks: pd.DatetimeIndex, split_series: pd.Series) -> pd.Series:
    """Drop the first week of non-train splits to prevent 3-day label overlap."""
    keep = pd.Series(True, index=weeks)
    for name in split_series.unique():
        if name == "train":
            continue
        split_weeks = weeks[split_series.to_numpy() == name]
        if len(split_weeks) > 0:
            first_week = split_weeks[0]
            keep[first_week] = False
    return keep


def get_temporal_splits(df: pd.DataFrame, purge: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train, val, and test subsets with optional boundary purging."""
    weeks = pd.DatetimeIndex(df["week"])
    split_series = assign_split(weeks)
    if purge:
        keep_mask = purge_boundary_weeks(weeks, split_series)
        df_clean = df[keep_mask.to_numpy()].copy()
        split_series = assign_split(pd.DatetimeIndex(df_clean["week"]))
    else:
        df_clean = df.copy()

    train_df = df_clean[split_series.values == "train"].copy()
    val_df = df_clean[split_series.values == "val"].copy()
    test_df = df_clean[split_series.values == "test"].copy()

    return train_df, val_df, test_df


def get_5fold_5year_splits(df: pd.DataFrame) -> list[dict[str, pd.DataFrame | int]]:
    """Return 5 folds of 5 years each: 3 years train, 1 year val, 1 year test.

    Fold 1: Train 2000-2002 (3y), Val 2003 (1y), Test 2004 (1y)
    Fold 2: Train 2005-2007 (3y), Val 2008 (1y), Test 2009 (1y)
    Fold 3: Train 2010-2012 (3y), Val 2013 (1y), Test 2014 (1y)
    Fold 4: Train 2015-2017 (3y), Val 2018 (1y), Test 2019 (1y)
    Fold 5: Train 2020-2022 (3y), Val 2023 (1y), Test 2024 (1y)
    """
    df_years = pd.to_datetime(df["week"]).dt.year
    folds = []

    for block in config.FOLD_5YEAR_BLOCKS:
        fold_num = block["fold"]
        tr_lo, tr_hi = block["train"]
        val_lo, val_hi = block["val"]
        tst_lo, tst_hi = block["test"]

        tr_mask = (df_years >= tr_lo) & (df_years <= tr_hi)
        val_mask = (df_years >= val_lo) & (df_years <= val_hi)
        tst_mask = (df_years >= tst_lo) & (df_years <= tst_hi)

        tr_df = df[tr_mask].copy()
        val_df = df[val_mask].copy()
        tst_df = df[tst_mask].copy()

        # Purge boundary week (first week of val and test)
        if len(val_df) > 0:
            val_first = val_df["week"].min()
            val_df = val_df[val_df["week"] > val_first].copy()

        if len(tst_df) > 0:
            tst_first = tst_df["week"].min()
            tst_df = tst_df[tst_df["week"] > tst_first].copy()

        folds.append({
            "fold": fold_num,
            "train": tr_df,
            "val": val_df,
            "test": tst_df,
            "train_years": f"{tr_lo}-{tr_hi}",
            "val_year": f"{val_lo}",
            "test_year": f"{tst_lo}",
        })

    return folds
