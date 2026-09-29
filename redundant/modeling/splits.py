"""Weekly time grid, label embargo and strict temporal splits.

Conventions (see MODEL_RESEARCH.md section 5):

- Weeks are Monday-starting (``W-SUN`` periods, as in the EDA).
- For a target week starting Monday ``m``, every feature must be dated at
  most ``m - EMBARGO_DAYS`` (Friday of the previous week), because a 3-day
  composite dated ``d`` covers ``d-2..d`` and the first composite of the
  target week is dated ``m`` itself.
- Splits are strict by year: train 2000-2014, validation 2015-2019,
  test 2020-2025. In addition each non-first split drops its *first* week:
  the first composite of that week (dated Monday) covers the two days before
  it, which belong to the previous split's label window. One week per
  boundary is the price of a clean embargo; ~2 weeks out of ~1350.
"""

from __future__ import annotations

import pandas as pd

from modeling import config

__all__ = [
    "assign_split",
    "feature_cutoff",
    "purge_boundary_weeks",
    "split_masks",
    "week_split_map",
    "weekly_index",
]


def feature_cutoff(week_start: pd.Timestamp) -> pd.Timestamp:
    """Latest date on which a feature may be dated for this target week."""
    if week_start.weekday() != 0:
        raise ValueError(f"week_start must be a Monday, got {week_start}")
    return week_start - pd.Timedelta(days=config.EMBARGO_DAYS)


def weekly_index(
    start: str = "2000-01-03", end: str = "2025-12-29"
) -> pd.DatetimeIndex:
    """All Monday-starting weeks in the record (2000-01-03 .. 2025-12-29)."""
    return pd.date_range(start=start, end=end, freq="W-MON")


def assign_split(week: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Map weeks to 'train' / 'val' / 'test' by their calendar year."""
    weeks = pd.DatetimeIndex(week)
    years = weeks.year
    out = pd.Series(index=weeks, dtype=object)
    for name, (lo, hi) in config.SPLITS.items():
        out[years.isin(range(lo, hi + 1))] = name
    return out


def purge_boundary_weeks(
    weeks: pd.DatetimeIndex, split: pd.Series
) -> pd.Series:
    """Boolean mask: weeks to drop because their label overlaps the previous split.

    The first composite of the first week of a split is dated Monday and covers
    the two days before it (still inside the previous split's label window), so
    that first week is dropped from every non-first split.
    """
    keep = pd.Series(True, index=weeks)
    for name in split.unique():
        if name == "train":
            continue
        first = weeks[split.to_numpy() == name][0]
        keep[first] = False
    return keep


def split_masks(
    weeks: pd.DatetimeIndex, purge: bool = True
) -> dict[str, pd.Series]:
    """Boolean mask per split, with boundary weeks purged when ``purge=True``."""
    split = assign_split(weeks)
    masks = {
        name: (split.to_numpy() == name) for name in ("train", "val", "test")
    }
    if purge:
        keep = purge_boundary_weeks(weeks, split)
        masks = {name: (mask & keep.to_numpy()) for name, mask in masks.items()}
    return masks


def week_split_map(weeks, purge: bool = True) -> pd.Series:
    """Map each week to its split label; purged boundary weeks map to None."""
    weeks = pd.DatetimeIndex(weeks)
    split = assign_split(weeks)
    if purge:
        keep = purge_boundary_weeks(weeks, split)
        split = split.mask(~keep, other=None)
    return split
