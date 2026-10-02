"""Temporal splits for LightGBM_v1 pipeline (v3: 3 expanding-window CV folds).

Walk-forward cross-validation with an expanding training window: each fold
trains on everything before its validation year, validates on one
flood-season year, and scores 2-3 held-out test years (evaluation_metrics.md
sec. 2). The first week of each val/test window is purged so no lag feature or
label window crosses a fold boundary.
"""

from __future__ import annotations

import pandas as pd

from LightGBM_v1 import config

__all__ = [
    "feature_cutoff",
    "get_cv_folds",
]


def feature_cutoff(week_start: pd.Timestamp) -> pd.Timestamp:
    """Return latest legal date for features used to forecast week_start (Friday)."""
    if week_start.weekday() != 0:
        raise ValueError(f"week_start must be a Monday timestamp, got: {week_start}")
    return week_start - pd.Timedelta(days=config.EMBARGO_DAYS)


def _purge_first_week(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the first week of a window (all counties) to prevent boundary overlap."""
    if df.empty:
        return df
    first = df["week"].min()
    return df[df["week"] > first].copy()


def get_cv_folds(df: pd.DataFrame) -> list[dict[str, pd.DataFrame | int | str]]:
    """Build the 3 expanding-window folds defined in config.CV_FOLDS.

    Returns a list of dicts with keys: fold, train, val, test,
    train_years, val_year, test_years.
    """
    years = pd.to_datetime(df["week"]).dt.year
    folds = []
    for spec in config.CV_FOLDS:
        tr_lo, tr_hi = spec["train_years"]
        tr_df = df[(years >= tr_lo) & (years <= tr_hi)].copy()
        val_df = _purge_first_week(df[years == spec["val_year"]].copy())
        te_lo, te_hi = spec["test_years"]
        te_df = _purge_first_week(df[(years >= te_lo) & (years <= te_hi)].copy())

        folds.append({
            "fold": spec["fold"],
            "train": tr_df,
            "val": val_df,
            "test": te_df,
            "train_years": f"{tr_lo}-{tr_hi}",
            "val_year": f"{spec['val_year']}",
            "test_years": f"{te_lo}-{te_hi}",
        })
    return folds
