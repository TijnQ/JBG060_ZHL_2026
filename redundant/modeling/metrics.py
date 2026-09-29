"""Evaluation metrics for flood extent forecasting.

Two families:

- **Detection metrics** for spike weeks (the ones that matter operationally):
  POD (hit rate), FAR (false alarm rate), CSI (critical success index).
- **Probabilistic metrics** for the area forecast: MAE / RMSE / R2 on the
  point forecast plus CRPS, evaluated exactly from the predicted 10/50/90
  quantiles (piecewise-linear CDF, no Monte Carlo).

``summarise_forecast`` turns a prediction frame into a tidy metrics table
with per-split, per-slice rows and a skill score against a baseline
(1 - MSE_model / MSE_baseline; >0 means better than the baseline).

Known-answer expectations (used by ``check_modeling``):

- A step CDF at ``q`` (one quantile at level 0.5) has CRPS ``|y - q|``.
- The symmetric 3-quantile case ``q=(1, 2, 3)``, ``y=2`` has CRPS exactly
  31/150 ~= 0.20667 (two equal integrals of a linear CDF, see below).
- POD/FAR/CSI for 1 hit, 1 false alarm, 1 miss: 0.5, 0.5, 1/3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "crps_mean",
    "crps_quantiles",
    "detection_scores",
    "mae",
    "r2",
    "rmse",
    "summarise_forecast",
]


# --- Point / detection metrics -------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d**2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def detection_scores(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0
) -> tuple[float, float, float]:
    """(POD, FAR, CSI) for binary detections.

    y_true: 1 when a flood was detected, 0 otherwise.
    y_pred: predicted probability (or area); >= threshold counts as a hit.
    Undefined metrics return nan (e.g. no predicted events -> FAR = nan).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    hit = int(np.sum((y_pred >= threshold) & (y_true == 1)))
    far = int(np.sum((y_pred >= threshold) & (y_true == 0)))
    miss = int(np.sum((y_pred < threshold) & (y_true == 1)))
    pod = hit / (hit + miss) if (hit + miss) > 0 else float("nan")
    far_rate = far / (far + hit) if (far + hit) > 0 else float("nan")
    csi = hit / (hit + far + miss) if (hit + far + miss) > 0 else float("nan")
    return pod, far_rate, csi


# --- CRPS from 3 quantiles (exact, piecewise-linear CDF) --------------------------


def _int_f2(a: float, b: float, fa: float, fb: float) -> float:
    """Integral of F^2 over [a, b] where F is linear from fa to fb."""
    if b <= a:
        return 0.0
    if fb == fa:
        return (b - a) * fa * fa
    m = (fb - fa) / (b - a)
    c = fa - m * a
    return ((m * b + c) ** 3 - (m * a + c) ** 3) / (3.0 * m)


def _int_1mf2(a: float, b: float, fa: float, fb: float) -> float:
    """Integral of (1-F)^2 over [a, b] where F is linear from fa to fb."""
    if b <= a:
        return 0.0
    ga, gb = 1.0 - fa, 1.0 - fb
    if gb == ga:
        return (b - a) * ga * ga
    m = (gb - ga) / (b - a)
    c = ga - m * a
    return ((m * b + c) ** 3 - (m * a + c) ** 3) / (3.0 * m)


def crps_quantiles(
    y: float, q10: float, q50: float, q90: float
) -> float:
    """CRPS of the 3-quantile forecast at outcome ``y``.

    The forecast CDF is 0 below ``q10`` (point mass of 0.1 at ``q10``),
    linear from 0.1 to 0.5 on [q10, q50], linear from 0.5 to 0.9 on
    [q50, q90], and 1 above ``q90``. CRPS = ∫(F(t) - 1{y <= t})^2 dt,
    split at t = y and evaluated exactly per segment (the infinite tails
    are constant-CDF segments, so all integrals are finite and closed-form).

    Known answers for q = (1, 2, 3) (checked in ``check_modeling``):
        y = 0 -> 1.6066667,  y = 1 -> 0.6066667,  y = 2 -> 31/150,
        y = 3 -> 0.6066667,  y = 5 -> 2.6066667
    """
    q10, q50, q90 = sorted((q10, q50, q90))

    def ramp_f(a: float, b: float, fa: float, fb: float, x: float) -> float:
        if x <= a:
            return fa
        if x >= b:
            return fb
        return fa + (x - a) * (fb - fa) / (b - a)

    # (lo, hi, kind, params): kind 'c' constant value, 'r' linear ramp
    segments = [
        (-np.inf, q10, "c", 0.0),
        (q10, q50, "r", (0.1, 0.5)),
        (q50, q90, "r", (0.5, 0.9)),
        (q90, np.inf, "c", 1.0),
    ]

    total = 0.0
    for region_lo, region_hi, one_minus in ((-np.inf, y, False), (y, np.inf, True)):
        for lo, hi, kind, params in segments:
            a, b = max(lo, region_lo), min(hi, region_hi)
            if b <= a:
                continue
            if kind == "c":
                # integrand: F^2 for I1, (1-F)^2 for I2
                f = params if not one_minus else 1.0 - params
                if f == 0.0:  # integrand zero (also covers infinite tails)
                    continue
                # f == 1: the overlap [a, b) is always finite in that case
                total += max(0.0, b - a)
            else:
                fa0, fb0 = params
                fa = ramp_f(lo, hi, fa0, fb0, a)
                fb = ramp_f(lo, hi, fa0, fb0, b)
                if one_minus:
                    total += _int_1mf2(a, b, fa, fb)
                else:
                    total += _int_f2(a, b, fa, fb)
    return float(total)


def crps_mean(
    y_true: pd.Series, q10: pd.Series, q50: pd.Series, q90: pd.Series
) -> float:
    """Mean CRPS over rows (NaN rows skipped)."""
    vals = [
        crps_quantiles(y, a, b, c)
        for y, a, b, c in zip(
            y_true.to_numpy(), q10.to_numpy(), q50.to_numpy(), q90.to_numpy()
        )
        if not (np.isnan(y) or np.isnan(a) or np.isnan(b) or np.isnan(c))
    ]
    return float(np.mean(vals)) if vals else float("nan")


# --- Tidy summary -----------------------------------------------------------------


def _slice_rows(
    df: pd.DataFrame,
    y_true: pd.Series,
    q10: pd.Series,
    q50: pd.Series,
    q90: pd.Series,
    baseline: pd.Series,
) -> list[dict]:
    rows = []
    for split in df["split"].unique():
        sub = df[df["split"] == split]
        y = y_true.loc[sub.index]
        b = baseline.loc[sub.index]
        for slice_name, mask in [
            ("all", pd.Series(True, index=sub.index)),
            ("positive", (y > 0).to_numpy()),
            ("spike", (y >= sub["spike_threshold"]).to_numpy()),
        ]:
            n = int(mask.sum())
            if n == 0:
                continue
            row: dict = {"split": split, "slice": slice_name, "n": n}
            row["y_pred_q50"] = float(sub.loc[mask, "q50"].mean())
            row["y_true_mean"] = float(y[mask].mean())
            row["mae_q50"] = mae(y[mask], sub.loc[mask, "q50"])
            row["mae_baseline"] = mae(y[mask], b[mask])
            row["rmse_q50"] = rmse(y[mask], sub.loc[mask, "q50"])
            row["rmse_baseline"] = rmse(y[mask], b[mask])
            row["skill_q50"] = 1.0 - (
                rmse(y[mask], sub.loc[mask, "q50"])
                / rmse(y[mask], b[mask])
            ) if rmse(y[mask], b[mask]) > 0 else float("nan")
            row["crps"] = crps_mean(
                y[mask],
                q10.loc[sub.index][mask],
                q50.loc[sub.index][mask],
                q90.loc[sub.index][mask],
            )
            row["crps_baseline"] = float("nan")  # baseline is a point forecast
            row["pod"] = detection_scores(
                (y[mask] > 0).astype(int).to_numpy(),
                sub.loc[mask, "det_prob"].to_numpy(),
                threshold=0.5,
            )[0]
            row["far"] = detection_scores(
                (y[mask] > 0).astype(int).to_numpy(),
                sub.loc[mask, "det_prob"].to_numpy(),
                threshold=0.5,
            )[1]
            row["csi"] = detection_scores(
                (y[mask] > 0).astype(int).to_numpy(),
                sub.loc[mask, "det_prob"].to_numpy(),
                threshold=0.5,
            )[2]
            rows.append(row)
    return rows


def summarise_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy metrics table from a prediction frame.

    Required columns: ``split``, ``q50``, ``det_prob``, ``spike_threshold``,
    ``y_pred_baseline``; optionally ``q10``, ``q90`` (CRPS needs all three).
    ``spike_threshold`` must be a per-row column (set from the TRAIN period).
    """
    rows = _slice_rows(
        df,
        df["y_true"],
        df.get("q10", pd.Series(np.nan, index=df.index)),
        df["q50"],
        df.get("q90", pd.Series(np.nan, index=df.index)),
        df["y_pred_baseline"],
    )
    return pd.DataFrame(rows)
