"""Simple, hard-to-beat baselines (MODEL_RESEARCH.md experiment E2).

Every learned model must beat these before it earns a place in the report:

- **Climatology**: the mean observed area for (county, month) computed on
  the TRAIN split only. This encodes the wet-season cycle without any input.
- **Persistence**: the observed area of the previous week (a flood that is
  happening this week is very likely still there next week).
- **Last detection**: the most recent non-zero area within the last four
  weeks; a stand-in for "the last satellite composite" nowcast.

All three are deterministic functions of the feature frame produced by
``features.build_features`` (they only use columns that are legal at
prediction time — see that module's embargo rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling import config

__all__ = [
    "add_baselines",
    "climatology_baseline",
    "last_detection_baseline",
    "persistence_baseline",
]


def climatology_baseline(
    features: pd.DataFrame, train_mask: np.ndarray | pd.Series
) -> pd.Series:
    """Mean train-period area per (county, month); county mean as fallback.

    Dictionary-based lookup on purpose: a ``merge`` silently drops rows
    whose (county, month) never appeared in the training period (e.g.
    counties without a single flood detection in 2000-2014 at national
    scope), which would vanish whole counties from the frame.
    """
    train = features[train_mask]
    cm = train.groupby(["county", "month"], observed=True)["y_true"].mean()
    c = train.groupby("county", observed=True)["y_true"].mean()
    cm_map = {f"{co}|{mo}": v for (co, mo), v in cm.items()}
    c_map = {co: v for co, v in c.items()}
    keys = features["county"].astype(str) + "|" + features["month"].astype(int).astype(str)
    values = keys.map(cm_map).fillna(features["county"].astype(str).map(c_map)).fillna(0.0)
    return pd.Series(values.to_numpy(), index=features.index, name="climatology")


def persistence_baseline(features: pd.DataFrame) -> pd.Series:
    """Area observed in the previous week (0 when that week was dry).

    Requires the frame sorted by (county, week) — ``build_features`` does that.
    """
    shifted = features.groupby("county", observed=True)["y_true"].shift(1)
    return shifted.fillna(0.0)


def last_detection_baseline(features: pd.DataFrame) -> pd.Series:
    """Most recent non-zero observed area within the last four weeks."""
    records = []
    for county, sub in features.groupby("county", observed=True):
        area = sub["y_true"].to_numpy(dtype=float)
        last = np.zeros(len(area))
        for i in range(len(area)):
            window = area[max(0, i - config.OWN_LAGS[-1]) : i]
            nonzero = window[window > 0]
            last[i] = nonzero[-1] if nonzero.size else 0.0
        records.append(pd.Series(last, index=sub.index, name="last_det"))
    return pd.concat(records)


def add_baselines(
    features: pd.DataFrame, train_mask: np.ndarray | pd.Series
) -> pd.DataFrame:
    """Return a copy of the feature frame with baseline forecast columns."""
    out = features.copy()
    out["y_pred_clim"] = climatology_baseline(out, train_mask)
    out["y_pred_persist"] = persistence_baseline(out)
    out["y_pred_lastdet"] = last_detection_baseline(out)
    # Baseline used for the headline skill score: persistence is usually the
    # strongest trivial baseline for weekly extent, so it sets the bar.
    out["y_pred_baseline"] = out["y_pred_persist"]
    return out
