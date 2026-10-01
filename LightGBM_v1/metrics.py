"""Evaluation metrics module for LightGBM_v1 pipeline.

Includes complete technical evaluation metrics:
1. Detection probability & classification metrics:
   - Accuracy, Precision, Recall (POD), F1 Score, CSI, FAR
   - Brier Score & Brier Skill Score (BSS vs Climatology & Persistence)
   - Reliability diagram binned data
2. Flood area & quantile metrics:
   - Pinball Loss (q10, q50, q90)
   - Interval Coverage (flood weeks & all weeks)
   - MAE and Bias on flood weeks (y_true > 0)
3. Per-month & per-county breakdown statistics:
   - Monthly breakdown for all 12 months (Jun-Nov rainy season)
   - County-level breakdown for the 5 Aweil counties
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "accuracy_score",
    "brier_score",
    "brier_skill_score",
    "classification_metrics",
    "detection_scores",
    "evaluate_all_outputs",
    "evaluate_county_metrics",
    "evaluate_monthly_metrics",
    "f1_score",
    "flood_week_mae_and_bias",
    "interval_coverage",
    "mae",
    "mean_pinball_loss",
    "pinball_loss",
    "precision_score",
    "recall_score",
    "reliability_diagram_data",
    "rmse",
]

MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]


# --- Classification & Detection Metrics --------------------------------------

def accuracy_score(y_true: np.ndarray | pd.Series, y_pred_binary: np.ndarray | pd.Series) -> float:
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred_binary, dtype=int)
    return float(np.mean(y_t == y_p))


def precision_score(y_true: np.ndarray | pd.Series, y_pred_binary: np.ndarray | pd.Series) -> float:
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred_binary, dtype=int)
    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else float("nan")


def recall_score(y_true: np.ndarray | pd.Series, y_pred_binary: np.ndarray | pd.Series) -> float:
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred_binary, dtype=int)
    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")


def f1_score(y_true: np.ndarray | pd.Series, y_pred_binary: np.ndarray | pd.Series) -> float:
    prec = precision_score(y_true, y_pred_binary)
    rec = recall_score(y_true, y_pred_binary)
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        return float("nan")
    return float(2.0 * prec * rec / (prec + rec))


def detection_scores(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, threshold: float = 0.5
) -> tuple[float, float, float]:
    y_t = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(y_prob, dtype=float)

    hits = int(np.sum((p_pred >= threshold) & (y_t == 1)))
    false_alarms = int(np.sum((p_pred >= threshold) & (y_t == 0)))
    misses = int(np.sum((p_pred < threshold) & (y_t == 1)))

    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
    csi = hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else float("nan")

    return pod, far, csi


def classification_metrics(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    y_t = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(y_prob, dtype=float)
    y_bin = (p_pred >= threshold).astype(int)

    acc = accuracy_score(y_t, y_bin)
    prec = precision_score(y_t, y_bin)
    rec = recall_score(y_t, y_bin)
    f1 = f1_score(y_t, y_bin)
    pod, far, csi = detection_scores(y_t, p_pred, threshold)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "csi": csi,
        "far": far,
    }


def brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    y_t = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(y_prob, dtype=float)
    return float(np.mean((p_pred - y_t) ** 2))


def brier_skill_score(brier_model: float, brier_ref: float) -> float:
    if brier_ref == 0 or np.isnan(brier_ref):
        return float("nan")
    return float(1.0 - (brier_model / brier_ref))


def reliability_diagram_data(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    y_t = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

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


# --- Flood Area Quantile & Point Metrics --------------------------------------

def pinball_loss(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, q: float) -> float:
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


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def flood_week_mae_and_bias(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> tuple[float, float]:
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


# --- Comprehensive Evaluation Summary ---------------------------------------

def evaluate_all_outputs(df_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name in df_pred["split"].unique():
        sub = df_pred[df_pred["split"] == split_name].copy()
        y_true = sub["y_true"]
        y_det = (y_true > 0).astype(int)

        for model_name in ["lightgbm", "climatology", "persistence"]:
            if model_name == "lightgbm":
                p_det = sub["det_prob"]
                q10 = sub["q10"]
                q50 = sub["q50"]
                q90 = sub["q90"]
            elif model_name == "climatology":
                p_det = sub["det_prob_clim"]
                q10 = sub["area_clim_q50"]
                q50 = sub["area_clim_q50"]
                q90 = sub["area_clim_q50"]
            elif model_name == "persistence":
                p_det = sub["det_prob_persist"]
                q10 = sub["area_persist"]
                q50 = sub["area_persist"]
                q90 = sub["area_persist"]

            clf_m = classification_metrics(y_det, p_det, threshold=0.5)

            brier = brier_score(y_det, p_det)
            brier_clim = brier_score(y_det, sub["det_prob_clim"])
            brier_pers = brier_score(y_det, sub["det_prob_persist"])

            bss_vs_clim = brier_skill_score(brier, brier_clim)
            bss_vs_pers = brier_skill_score(brier, brier_pers)

            mean_pinball = mean_pinball_loss(y_true, q10, q50, q90)
            cov_all = interval_coverage(y_true, q10, q90, flood_only=False)
            cov_flood = interval_coverage(y_true, q10, q90, flood_only=True)

            mae_all = mae(y_true, q50)
            mae_flood, bias_flood = flood_week_mae_and_bias(y_true, q50)

            rows.append({
                "split": split_name,
                "model": model_name,
                "n_samples": len(sub),
                "accuracy": round(clf_m["accuracy"], 4),
                "precision": round(clf_m["precision"], 4),
                "recall": round(clf_m["recall"], 4),
                "f1_score": round(clf_m["f1_score"], 4),
                "csi": round(clf_m["csi"], 4),
                "far": round(clf_m["far"], 4),
                "brier_score": round(brier, 4),
                "bss_vs_climatology": round(bss_vs_clim, 4),
                "bss_vs_persistence": round(bss_vs_pers, 4),
                "mean_pinball_loss": round(mean_pinball, 3),
                "coverage_all_weeks": round(cov_all, 3),
                "coverage_flood_weeks": round(cov_flood, 3),
                "mae_all_weeks": round(mae_all, 2),
                "mae_flood_weeks": round(mae_flood, 2),
                "bias_flood_weeks": round(bias_flood, 2),
            })

    return pd.DataFrame(rows)


def evaluate_monthly_metrics(df_pred: pd.DataFrame, model_name: str = "lightgbm") -> pd.DataFrame:
    """Compute detailed evaluation statistics broken down per calendar month (1-12)."""
    sub = df_pred[df_pred.get("model", model_name) == model_name].copy() if "model" in df_pred.columns else df_pred.copy()
    sub["month"] = pd.to_datetime(sub["week"]).dt.month

    rows = []
    for mo in range(1, 13):
        m_df = sub[sub["month"] == mo].copy()
        n_samples = len(m_df)
        if n_samples == 0:
            continue

        y_true = m_df["y_true"]
        y_det = (y_true > 0).astype(int)
        p_det = m_df["det_prob"]
        q10 = m_df["q10"]
        q50 = m_df["q50"]
        q90 = m_df["q90"]

        n_flood = int(np.sum(y_det == 1))
        flood_pct = round(100.0 * n_flood / n_samples, 1)

        clf_m = classification_metrics(y_det, p_det, threshold=0.5)

        brier = brier_score(y_det, p_det)
        brier_clim = brier_score(y_det, m_df["det_prob_clim"])
        brier_pers = brier_score(y_det, m_df["det_prob_persist"])

        bss_vs_clim = brier_skill_score(brier, brier_clim)
        bss_vs_pers = brier_skill_score(brier, brier_pers)

        mean_pinball = mean_pinball_loss(y_true, q10, q50, q90)
        cov_all = interval_coverage(y_true, q10, q90, flood_only=False)
        cov_flood = interval_coverage(y_true, q10, q90, flood_only=True)

        mae_all = mae(y_true, q50)
        mae_flood, bias_flood = flood_week_mae_and_bias(y_true, q50)

        is_rainy = "Yes (Peak)" if mo in (6, 7, 8, 9, 10, 11) else "No (Dry)"

        rows.append({
            "month_num": mo,
            "month_name": MONTH_NAMES[mo],
            "season": is_rainy,
            "n_samples": n_samples,
            "flood_weeks": n_flood,
            "flood_rate_pct": flood_pct,
            "accuracy": round(clf_m["accuracy"], 4),
            "precision": round(clf_m["precision"], 4),
            "recall": round(clf_m["recall"], 4),
            "f1_score": round(clf_m["f1_score"], 4),
            "csi": round(clf_m["csi"], 4),
            "far": round(clf_m["far"], 4),
            "brier_score": round(brier, 4),
            "bss_vs_climatology": round(bss_vs_clim, 4),
            "bss_vs_persistence": round(bss_vs_pers, 4),
            "mean_pinball_loss": round(mean_pinball, 3),
            "coverage_flood_weeks": round(cov_flood, 3) if not np.isnan(cov_flood) else 0.0,
            "coverage_all_weeks": round(cov_all, 3),
            "mae_flood_weeks": round(mae_flood, 2) if not np.isnan(mae_flood) else 0.0,
            "bias_flood_weeks": round(bias_flood, 2) if not np.isnan(bias_flood) else 0.0,
        })

    return pd.DataFrame(rows).sort_values("month_num").reset_index(drop=True)


def evaluate_county_metrics(df_pred: pd.DataFrame, model_name: str = "lightgbm") -> pd.DataFrame:
    """Compute detailed evaluation statistics broken down per county."""
    sub = df_pred[df_pred.get("model", model_name) == model_name].copy() if "model" in df_pred.columns else df_pred.copy()

    rows = []
    for county_name, c_df in sub.groupby("county", observed=True):
        n_samples = len(c_df)
        if n_samples == 0:
            continue

        y_true = c_df["y_true"]
        y_det = (y_true > 0).astype(int)
        p_det = c_df["det_prob"]
        q10 = c_df["q10"]
        q50 = c_df["q50"]
        q90 = c_df["q90"]

        n_flood = int(np.sum(y_det == 1))
        flood_pct = round(100.0 * n_flood / n_samples, 1)

        clf_m = classification_metrics(y_det, p_det, threshold=0.5)

        brier = brier_score(y_det, p_det)
        brier_clim = brier_score(y_det, c_df["det_prob_clim"])
        brier_pers = brier_score(y_det, c_df["det_prob_persist"])

        bss_vs_clim = brier_skill_score(brier, brier_clim)
        bss_vs_pers = brier_skill_score(brier, brier_pers)

        mean_pinball = mean_pinball_loss(y_true, q10, q50, q90)
        cov_all = interval_coverage(y_true, q10, q90, flood_only=False)
        cov_flood = interval_coverage(y_true, q10, q90, flood_only=True)

        mae_all = mae(y_true, q50)
        mae_flood, bias_flood = flood_week_mae_and_bias(y_true, q50)

        rows.append({
            "county": county_name,
            "n_samples": n_samples,
            "flood_weeks": n_flood,
            "flood_rate_pct": flood_pct,
            "accuracy": round(clf_m["accuracy"], 4),
            "precision": round(clf_m["precision"], 4),
            "recall": round(clf_m["recall"], 4),
            "f1_score": round(clf_m["f1_score"], 4),
            "csi": round(clf_m["csi"], 4),
            "far": round(clf_m["far"], 4),
            "brier_score": round(brier, 4),
            "bss_vs_climatology": round(bss_vs_clim, 4),
            "bss_vs_persistence": round(bss_vs_pers, 4),
            "mean_pinball_loss": round(mean_pinball, 3),
            "coverage_flood_weeks": round(cov_flood, 3) if not np.isnan(cov_flood) else 0.0,
            "coverage_all_weeks": round(cov_all, 3),
            "mae_flood_weeks": round(mae_flood, 2) if not np.isnan(mae_flood) else 0.0,
            "bias_flood_weeks": round(bias_flood, 2) if not np.isnan(bias_flood) else 0.0,
        })

    return pd.DataFrame(rows).sort_values("county").reset_index(drop=True)
