"""Evaluation metrics module for LightGBM_v1 pipeline (v3: protocol-aligned).

The metric set is exactly the protocol frozen in evaluation_metrics.md (Task A):

- det_prob (weekly flood-detection probability):
    * Brier Score — headline probabilistic accuracy metric
    * Brier Skill Score vs climatology AND vs persistence (both references
      estimated per fold inside the fold's training window)
    * Reliability diagram (binned predicted-vs-observed frequency table)
- q10 / q50 / q90 (flood area in km²):
    * mean Pinball Loss across the three quantiles
    * [q10, q90] interval coverage — flood-weeks headline, all-weeks beside it
    * MAE and Bias on flood weeks (y_true > 0) only
- Duration (per horizon h1–h4, "flood detected h weeks ahead"):
    * Brier per horizon vs county-month climatology and Markov persistence
      references at the target week (per-fold estimated)
    * Reliability diagram per horizon

The cutoff-dependent classification family (accuracy, precision, recall, F1,
CSI, FAR) is intentionally NOT part of Task A: it throws the forecast
probability away, and CSI / POD / FAR are reserved for Task B event metrics
in evaluation_metrics.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from LightGBM_v1 import config

__all__ = [
    "MONTH_NAMES",
    "brier_score",
    "brier_skill_score",
    "duration_climatology_reference",
    "duration_persistence_reference",
    "flood_week_mae_and_bias",
    "interval_coverage",
    "mean_pinball_loss",
    "pinball_loss",
    "reliability_diagram_data",
    "evaluate_county_metrics",
    "evaluate_duration_metrics",
    "evaluate_monthly_metrics",
    "evaluate_summary_metrics",
]

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

_SUMMARY_MODELS = ("lightgbm", "climatology", "persistence")


# --- Detection probability metrics (protocol: Brier / BSS / reliability) -----

def brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Brier Score of predicted probabilities against 0/1 outcomes."""
    y_t = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(y_prob, dtype=float)
    return float(np.mean((p_pred - y_t) ** 2))


def brier_skill_score(brier_model: float, brier_ref: float) -> float:
    """Brier Skill Score: 1 - Brier_model / Brier_ref (>0 means added skill)."""
    if brier_ref == 0 or np.isnan(brier_ref):
        return float("nan")
    return float(1.0 - (brier_model / brier_ref))


def reliability_diagram_data(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Binned reliability table: mean predicted probability vs observed frequency."""
    y_t = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(p_pred, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            mean_prob = float(np.mean(p_pred[mask]))
            obs_freq = float(np.mean(y_t[mask]))
        else:
            mean_prob = float((bins[b] + bins[b + 1]) / 2.0)
            obs_freq = float("nan")

        rows.append({
            "bin": b,
            "bin_lower": float(bins[b]),
            "bin_upper": float(bins[b + 1]),
            "bin_midpoint": float((bins[b] + bins[b + 1]) / 2.0),
            "count": count,
            "mean_pred_prob": mean_prob,
            "observed_freq": obs_freq,
        })

    return pd.DataFrame(rows)

# --- Flood area quantile & point metrics (pinball / coverage / flood-week MAE) --

def pinball_loss(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, q: float) -> float:
    """Pinball (quantile) loss for a single quantile q."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    errors = y_t - y_p
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    return float(np.mean(loss))


def mean_pinball_loss(
    y_true: np.ndarray | pd.Series,
    q10: np.ndarray | pd.Series,
    q50: np.ndarray | pd.Series,
    q90: np.ndarray | pd.Series,
) -> float:
    """Mean pinball loss across q10, q50, q90."""
    l10 = pinball_loss(y_true, q10, 0.1)
    l50 = pinball_loss(y_true, q50, 0.5)
    l90 = pinball_loss(y_true, q90, 0.9)
    return float((l10 + l50 + l90) / 3.0)


def interval_coverage(
    y_true: np.ndarray | pd.Series,
    q10: np.ndarray | pd.Series,
    q90: np.ndarray | pd.Series,
    flood_only: bool = False,
) -> float:
    """Fraction of y_true inside [q10, q90] (optionally flood weeks only)."""
    y_t = np.asarray(y_true, dtype=float)
    lo = np.asarray(q10, dtype=float)
    hi = np.asarray(q90, dtype=float)

    if flood_only:
        mask = y_t > 0
        if not np.any(mask):
            return float("nan")
        y_t, lo, hi = y_t[mask], lo[mask], hi[mask]

    inside = (y_t >= lo) & (y_t <= hi)
    return float(np.mean(inside))


def flood_week_mae_and_bias(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> tuple[float, float]:
    """MAE and Bias computed strictly on flood weeks (y_true > 0)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    flood_mask = y_t > 0
    if not np.any(flood_mask):
        return float("nan"), float("nan")

    y_t_flood = y_t[flood_mask]
    y_p_flood = y_p[flood_mask]

    mae_val = float(np.mean(np.abs(y_p_flood - y_t_flood)))
    bias_val = float(np.mean(y_p_flood - y_t_flood))
    return mae_val, bias_val


# --- Duration reference forecasts (per fold, at the target week) --------------

def duration_climatology_reference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: tuple[int, ...] = config.DURATION_HORIZONS,
) -> pd.DataFrame:
    """Smoothed county-month climatology probability of a flood at the target week.

    Estimated strictly on the fold's training window. The target week for
    horizon h is `week + h`, so its (county, month) determines the reference.
    Returns columns det_prob_clim_h1..h4 aligned to test_df.index.
    """
    train = train_df
    county_mean = train.groupby("county", observed=True)["y_det"].mean()
    overall = float(train["y_det"].mean()) if len(train) else 0.2

    out = pd.DataFrame(index=test_df.index)
    for h in horizons:
        tgt_month = (train["week"] + pd.Timedelta(weeks=h)).dt.month
        grp = train.groupby([train["county"], tgt_month], observed=True)["y_det"]
        k = grp.sum()
        n = grp.count()
        prob = ((k + 0.5) / (n + 1.0)).to_dict()

        test_month = (test_df["week"] + pd.Timedelta(weeks=h)).dt.month
        vals = []
        for co, mo in zip(test_df["county"], test_month):
            if (co, int(mo)) in prob:
                vals.append(prob[(co, int(mo))])
            elif co in county_mean.index:
                vals.append(county_mean[co])
            else:
                vals.append(overall)
        out[f"det_prob_clim_h{h}"] = vals
    return out


def duration_persistence_reference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_frame: pd.DataFrame,
    horizons: tuple[int, ...] = config.DURATION_HORIZONS,
) -> pd.DataFrame:
    """Markov persistence probability of a flood at the target week (week + h).

    p_on / p_off are estimated on the fold's training window from the
    week-before-target state, exactly like det_prob_persist in baselines.py.
    `full_frame` supplies the observed y_det at the intermediate state weeks
    (the target is up to 4 weeks after the prediction week).
    Returns columns det_prob_persist_h1..h4 aligned to test_df.index.
    """
    train = train_df
    det_prev_1 = train[train["y_det_lag1"] == 1]["y_det"]
    det_prev_0 = train[train["y_det_lag1"] == 0]["y_det"]
    p_on = float(det_prev_1.mean()) if len(det_prev_1) > 0 else 0.6
    p_off = float(det_prev_0.mean()) if len(det_prev_0) > 0 else 0.1

    state_lookup = (
        full_frame.set_index(["county", "week"])["y_det"]
        .to_dict()
    )

    out = pd.DataFrame(index=test_df.index)
    for h in horizons:
        state_week = test_df["week"] + pd.Timedelta(weeks=h - 1)
        states = np.array(
            [state_lookup.get((co, w), np.nan) for co, w in zip(test_df["county"], state_week)],
            dtype=float,
        )
        out[f"det_prob_persist_h{h}"] = np.where(
            states == 1, p_on, np.where(states == 0, p_off, np.nan)
        )
    return out

# --- Protocol summary & breakdowns --------------------------------------------

def _model_columns(sub: pd.DataFrame, model_name: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (det_prob, q10, q50, q90) columns for a named model/reference."""
    if model_name == "lightgbm":
        return sub["det_prob"], sub["q10"], sub["q50"], sub["q90"]
    if model_name == "climatology":
        prob = sub["det_prob_clim"]
        return prob, sub["area_clim_q50"], sub["area_clim_q50"], sub["area_clim_q50"]
    if model_name == "persistence":
        prob = sub["det_prob_persist"]
        return prob, sub["area_persist"], sub["area_persist"], sub["area_persist"]
    raise ValueError(f"Unknown model name: {model_name}")


def _protocol_row(scope: str, model_name: str, sub: pd.DataFrame) -> dict:
    """One protocol-metric row for a (scope, model) block of test rows."""
    y_true = sub["y_true"].to_numpy(dtype=float)
    y_det = (sub["y_true"] > 0).astype(int).to_numpy()
    p_det, q10, q50, q90 = _model_columns(sub, model_name)

    brier = brier_score(y_det, p_det)
    bss_clim = brier_skill_score(brier, brier_score(y_det, sub["det_prob_clim"]))
    bss_pers = brier_skill_score(brier, brier_score(y_det, sub["det_prob_persist"]))

    cov_flood = interval_coverage(y_true, q10, q90, flood_only=True)
    cov_all = interval_coverage(y_true, q10, q90, flood_only=False)
    mae_flood, bias_flood = flood_week_mae_and_bias(y_true, q50)

    return {
        "scope": scope,
        "model": model_name,
        "n_samples": len(sub),
        "flood_weeks": int(np.sum(y_det == 1)),
        "flood_rate_pct": round(100.0 * np.mean(y_det), 1),
        "brier_score": round(brier, 4),
        "bss_vs_climatology": round(bss_clim, 4),
        "bss_vs_persistence": round(bss_pers, 4),
        "mean_pinball_loss": round(mean_pinball_loss(y_true, q10, q50, q90), 3),
        "coverage_flood_weeks": round(cov_flood, 3),
        "coverage_all_weeks": round(cov_all, 3),
        "mae_flood_weeks": round(mae_flood, 2),
        "bias_flood_weeks": round(bias_flood, 2),
    }


def evaluate_summary_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Per-fold + pooled protocol rows for LightGBM vs Climatology vs Persistence.

    Returns one row per (fold, model) plus three pooled rows (scope='pooled').
    """
    rows = []
    for fold in sorted(df_pred["fold"].unique()):
        sub = df_pred[df_pred["fold"] == fold]
        for model in _SUMMARY_MODELS:
            rows.append(_protocol_row(f"fold_{fold}", model, sub))
    for model in _SUMMARY_MODELS:
        rows.append(_protocol_row("pooled", model, df_pred))
    return pd.DataFrame(rows)


def _breakdown_row(sub: pd.DataFrame) -> dict:
    """Protocol rows for the LightGBM model over an arbitrary subset."""
    y_true = sub["y_true"].to_numpy(dtype=float)
    y_det = (sub["y_true"] > 0).astype(int).to_numpy()
    p_det, q10, q50, q90 = _model_columns(sub, "lightgbm")

    brier = brier_score(y_det, p_det)
    bss_clim = brier_skill_score(brier, brier_score(y_det, sub["det_prob_clim"]))
    bss_pers = brier_skill_score(brier, brier_score(y_det, sub["det_prob_persist"]))

    cov_flood = interval_coverage(y_true, q10, q90, flood_only=True)
    cov_all = interval_coverage(y_true, q10, q90, flood_only=False)
    mae_flood, bias_flood = flood_week_mae_and_bias(y_true, q50)

    return {
        "n_samples": len(sub),
        "flood_weeks": int(np.sum(y_det == 1)),
        "flood_rate_pct": round(100.0 * np.mean(y_det), 1),
        "brier_score": round(brier, 4),
        "bss_vs_climatology": round(bss_clim, 4),
        "bss_vs_persistence": round(bss_pers, 4),
        "mean_pinball_loss": round(mean_pinball_loss(y_true, q10, q50, q90), 3),
        "coverage_flood_weeks": round(cov_flood, 3) if not np.isnan(cov_flood) else np.nan,
        "coverage_all_weeks": round(cov_all, 3),
        "mae_flood_weeks": round(mae_flood, 2) if not np.isnan(mae_flood) else np.nan,
        "bias_flood_weeks": round(bias_flood, 2) if not np.isnan(bias_flood) else np.nan,
    }


def evaluate_county_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Per-county protocol metrics for the LightGBM model (OOF test weeks)."""
    rows = []
    for county_name, c_df in df_pred.groupby("county", observed=True):
        row = _breakdown_row(c_df)
        row["county"] = county_name
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("county").reset_index(drop=True)
    return out


def evaluate_monthly_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-month protocol metrics for the LightGBM model."""
    rows = []
    for mo in range(1, 13):
        m_df = df_pred[pd.to_datetime(df_pred["week"]).dt.month == mo]
        if m_df.empty:
            continue
        row = _breakdown_row(m_df)
        row["month_num"] = mo
        row["month_name"] = MONTH_NAMES[mo]
        row["season"] = "Rainy" if mo in (6, 7, 8, 9, 10, 11) else "Dry"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month_num").reset_index(drop=True)

# --- Duration protocol (Brier per horizon + reliability) -----------------------

def evaluate_duration_metrics(
    df_pred: pd.DataFrame,
    horizons: tuple[int, ...] = config.DURATION_HORIZONS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Per-horizon duration metrics for the LightGBM duration models.

    Expects, per test row: y_det_h{h} (target), det_prob_h{h} (model),
    det_prob_clim_h{h} and det_prob_persist_h{h} (per-fold references at the
    target week, see duration_*_reference).

    Returns
    -------
    (summary, reliability) where summary has one row per (fold, horizon) plus
    pooled rows, and reliability maps horizon -> binned reliability table
    pooled over all OOF test weeks.
    """
    horizon_cols = [f"y_det_h{h}" for h in horizons]  # noqa: F841  # documented contract
    rows = []

    def _score(sub: pd.DataFrame, scope: str, h: int) -> None:
        valid = sub[sub[f"y_det_h{h}"].notna()]
        y = valid[f"y_det_h{h}"].astype(int).to_numpy()
        p = valid[f"det_prob_h{h}"].to_numpy(dtype=float)
        brier = brier_score(y, p)
        bss_clim = brier_skill_score(brier, brier_score(y, valid[f"det_prob_clim_h{h}"].to_numpy(dtype=float)))
        bss_pers = brier_skill_score(brier, brier_score(y, valid[f"det_prob_persist_h{h}"].to_numpy(dtype=float)))
        rows.append({
            "scope": scope,
            "horizon_weeks": h,
            "n_samples": len(valid),
            "flood_weeks": int(np.sum(y == 1)),
            "flood_rate_pct": round(100.0 * np.mean(y), 1),
            "brier_score": round(brier, 4),
            "bss_vs_climatology": round(bss_clim, 4),
            "bss_vs_persistence": round(bss_pers, 4),
        })

    for fold in sorted(df_pred["fold"].unique()):
        sub = df_pred[df_pred["fold"] == fold]
        for h in horizons:
            _score(sub, f"fold_{fold}", h)
    for h in horizons:
        _score(df_pred, "pooled", h)

    summary = pd.DataFrame(rows)

    reliability: dict[str, pd.DataFrame] = {}
    for h in horizons:
        valid = df_pred[df_pred[f"y_det_h{h}"].notna()]
        reliability[f"h{h}"] = reliability_diagram_data(
            valid[f"y_det_h{h}"], valid[f"det_prob_h{h}"]
        )
    return summary, reliability