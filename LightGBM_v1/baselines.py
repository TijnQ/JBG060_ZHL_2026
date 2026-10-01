"""Reference baselines module for LightGBM_v1 pipeline.

Implements fair, calibrated reference forecasts:
1. Climatology: (k + 0.5) / (n + 1) smoothed probability & median area.
2. First-order Markov Persistence: p_on/p_off continuation probabilities & prev-week area.
3. Last Detection: Most recent non-zero area within 4 weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from LightGBM_v1 import config

__all__ = [
    "add_baselines_to_dataframe",
    "climatology_area_baseline",
    "climatology_detection_baseline",
    "last_detection_baseline",
    "persistence_area_baseline",
    "persistence_detection_baseline",
]


def climatology_detection_baseline(df: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> pd.Series:
    """Smoothed climatological detection probability (k + 0.5) / (n + 1) per (county, month)."""
    train = df[train_mask]
    grouped = train.groupby(["county", "month"], observed=True)["y_det"]

    k = grouped.sum()
    n = grouped.count()
    probs = (k + 0.5) / (n + 1.0)

    county_mean = train.groupby("county", observed=True)["y_det"].mean()
    overall_mean = train["y_det"].mean() if len(train) else 0.2

    prob_map = probs.to_dict()
    county_map = county_mean.to_dict()

    out = []
    for co, mo in zip(df["county"], df["month"]):
        if (co, mo) in prob_map:
            out.append(prob_map[(co, mo)])
        elif co in county_map:
            out.append(county_map[co])
        else:
            out.append(overall_mean)

    return pd.Series(out, index=df.index, name="det_prob_clim")


def climatology_area_baseline(df: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> pd.Series:
    """Median train-period flood area per (county, month) as area baseline."""
    train = df[train_mask]
    cm_median = train.groupby(["county", "month"], observed=True)["y_true"].median()
    c_median = train.groupby("county", observed=True)["y_true"].median()
    overall_median = train["y_true"].median() if len(train) else 0.0

    cm_map = {f"{co}|{mo}": v for (co, mo), v in cm_median.items()}
    c_map = {co: v for co, v in c_median.items()}

    keys = df["county"].astype(str) + "|" + df["month"].astype(int).astype(str)
    values = keys.map(cm_map).fillna(df["county"].astype(str).map(c_map)).fillna(overall_median)

    return pd.Series(values.to_numpy(), index=df.index, name="area_clim_q50")


def persistence_detection_baseline(df: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> pd.Series:
    """First-order Markov persistence detection probability based on lag-1 state."""
    train = df[train_mask]

    det_prev_1 = train[train["y_det_lag1"] == 1]
    det_prev_0 = train[train["y_det_lag1"] == 0]

    p_on = det_prev_1["y_det"].mean() if len(det_prev_1) > 0 else 0.6
    p_off = det_prev_0["y_det"].mean() if len(det_prev_0) > 0 else 0.1

    out = np.where(df["y_det_lag1"] == 1, p_on, p_off)
    return pd.Series(out, index=df.index, name="det_prob_persist")


def persistence_area_baseline(df: pd.DataFrame) -> pd.Series:
    """Area observed in previous week (y_true_lag1)."""
    return df["y_true_lag1"].copy().rename("area_persist")


def last_detection_baseline(df: pd.DataFrame) -> pd.Series:
    """Most recent non-zero observed area within last four weeks."""
    records = []
    for county, sub in df.groupby("county", observed=True):
        area = sub["y_true"].to_numpy(dtype=float)
        last = np.zeros(len(area))
        for i in range(len(area)):
            window = area[max(0, i - config.OWN_LAGS[-1]) : i]
            nonzero = window[window > 0]
            last[i] = nonzero[-1] if nonzero.size else 0.0
        records.append(pd.Series(last, index=sub.index, name="area_lastdet"))
    return pd.concat(records).reindex(df.index)


def add_baselines_to_dataframe(df: pd.DataFrame, train_mask: pd.Series | np.ndarray) -> pd.DataFrame:
    """Add all reference baseline forecast columns to the DataFrame."""
    out = df.copy()
    out["det_prob_clim"] = climatology_detection_baseline(out, train_mask)
    out["area_clim_q50"] = climatology_area_baseline(out, train_mask)

    out["det_prob_persist"] = persistence_detection_baseline(out, train_mask)
    out["area_persist"] = persistence_area_baseline(out)

    out["area_lastdet"] = last_detection_baseline(out)

    out["det_prob_baseline"] = out["det_prob_persist"]
    out["area_baseline"] = out["area_persist"]

    return out
