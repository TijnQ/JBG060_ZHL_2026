"""LightGBM model training and prediction module for LightGBM_v1 pipeline.

Trains:
1. LGBMClassifier for flood detection probability (`det_prob`).
2. LGBMRegressor (alpha=0.1, 0.5, 0.9) for quantile flood area in km² (`q10`, `q50`, `q90`).

Every train function falls back to the shared default hyperparameter dict
(`_BASE_PARAMS`) when called with `params=None`, so run.py behavior is exactly
unchanged; LightGBM_v1.tuning passes explicit per-head overrides.
"""

from __future__ import annotations

import warnings

import lightgbm as lgbm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*")

from LightGBM_v1 import config
from LightGBM_v1.features import FEATURE_COLUMNS

__all__ = [
    "build_and_train_lightgbm",
    "predict_lightgbm",
    "train_detection_classifier",
    "train_quantile_regressors",
]

# Shared default hyperparameters (v3 baseline). Every train function uses these
# when called with params=None — exactly the hardcoded values run.py has always
# trained with (plan.md §2).
_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": config.SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def _merged_params(params: dict[str, object] | None) -> dict[str, object]:
    """Default hyperparameters with optional per-key overrides (None = defaults)."""
    return dict(_BASE_PARAMS) if params is None else {**_BASE_PARAMS, **params}


def train_detection_classifier(
    X_train: pd.DataFrame,
    y_train_det: pd.Series,
    X_val: pd.DataFrame,
    y_val_det: pd.Series,
    params: dict[str, object] | None = None,
) -> lgbm.LGBMClassifier:
    """Train LightGBM binary classifier for flood detection probability.

    `params` overrides individual default hyperparameters (None = the exact
    default dict used by run.py).
    """
    clf = lgbm.LGBMClassifier(**_merged_params(params))
    clf.fit(
        X_train,
        y_train_det,
        eval_set=[(X_val, y_val_det)],
        callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return clf


def train_quantile_regressors(
    X_train: pd.DataFrame,
    y_train_area: pd.Series,
    X_val: pd.DataFrame,
    y_val_area: pd.Series,
    quantiles: tuple[float, ...] = config.QUANTILES,
    params: dict[str, object] | None = None,
) -> dict[str, lgbm.LGBMRegressor]:
    """Train LightGBM quantile regressors for specified quantiles (q10, q50, q90).

    `params` overrides individual default hyperparameters (None = the exact
    default dict used by run.py).
    """
    model_params = _merged_params(params)

    models = {}
    for alpha in quantiles:
        reg = lgbm.LGBMRegressor(objective="quantile", alpha=alpha, **model_params)
        reg.fit(
            X_train,
            y_train_area,
            eval_set=[(X_val, y_val_area)],
            callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=False)],
        )
        q_key = f"q{int(alpha * 100)}"
        models[q_key] = reg

    return models


def build_and_train_lightgbm(
    X_train: pd.DataFrame,
    y_train_area: pd.Series,
    y_train_det: pd.Series,
    X_val: pd.DataFrame,
    y_val_area: pd.Series,
    y_val_det: pd.Series,
    det_params: dict[str, object] | None = None,
    quant_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train all 4 LightGBM models (det_prob classifier + q10, q50, q90 quantile regressors).

    `det_params` / `quant_params` override individual default hyperparameters of
    the detection classifier / the 3 quantile regressors (None = the exact
    default dicts used by run.py).
    """
    models = train_quantile_regressors(X_train, y_train_area, X_val, y_val_area, params=quant_params)
    det_clf = train_detection_classifier(X_train, y_train_det, X_val, y_val_det, params=det_params)
    models["det"] = det_clf
    return models


def predict_lightgbm(models: dict[str, object], X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Generate LightGBM predictions for detection probability and quantile flood area."""
    X_feat = X[FEATURE_COLUMNS]

    det_prob = models["det"].predict_proba(X_feat)[:, 1]

    q10 = np.clip(models["q10"].predict(X_feat), 0.0, None)
    q50 = np.clip(models["q50"].predict(X_feat), 0.0, None)
    q90 = np.clip(models["q90"].predict(X_feat), 0.0, None)

    q10_clean = np.minimum(q10, q50)
    q90_clean = np.maximum(q90, q50)

    return {
        "det_prob": det_prob,
        "q10": q10_clean,
        "q50": q50,
        "q90": q90_clean,
    }
