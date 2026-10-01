"""LightGBM model training and prediction module for LightGBM_v1 pipeline.

Trains:
1. LGBMClassifier for flood detection probability (`det_prob`).
2. LGBMRegressor (alpha=0.1, 0.5, 0.9) for quantile flood area in km² (`q10`, `q50`, `q90`).
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


def train_detection_classifier(
    X_train: pd.DataFrame,
    y_train_det: pd.Series,
    X_val: pd.DataFrame,
    y_val_det: pd.Series,
) -> lgbm.LGBMClassifier:
    """Train LightGBM binary classifier for flood detection probability."""
    clf = lgbm.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=-1,
    )
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
) -> dict[str, lgbm.LGBMRegressor]:
    """Train LightGBM quantile regressors for specified quantiles (q10, q50, q90)."""
    base_params = {
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

    models = {}
    for alpha in quantiles:
        reg = lgbm.LGBMRegressor(objective="quantile", alpha=alpha, **base_params)
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
) -> dict[str, object]:
    """Train all 4 LightGBM models (det_prob classifier + q10, q50, q90 quantile regressors)."""
    models = train_quantile_regressors(X_train, y_train_area, X_val, y_val_area)
    det_clf = train_detection_classifier(X_train, y_train_det, X_val, y_val_det)
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
