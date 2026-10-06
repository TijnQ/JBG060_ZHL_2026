"""Duration forecasting module for LightGBM_v1 pipeline.

Predicts multi-horizon flood duration probabilities (1, 2, 3, 4 weeks ahead).
"""

from __future__ import annotations

import warnings
# pyrefly: ignore [missing-import]
import lightgbm as lgbm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*")

from LightGBM_v1 import config
from LightGBM_v1.features import FEATURE_COLUMNS

__all__ = [
    "build_duration_targets",
    "predict_duration",
    "train_duration_models",
]


def build_duration_targets(df: pd.DataFrame, horizons: tuple[int, ...] = config.DURATION_HORIZONS) -> pd.DataFrame:
    """Construct multi-horizon duration targets (y_det at t + h for h in 1..4)."""
    out = df.copy()
    for h in horizons:
        out[f"y_det_h{h}"] = out.groupby("county", observed=True)["y_det"].shift(-h)
    return out


def train_duration_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    horizons: tuple[int, ...] = config.DURATION_HORIZONS,
    feature_columns: list[str] | None = None,
) -> dict[int, lgbm.LGBMClassifier]:
    """Train binary LightGBM classifiers for each duration horizon (1..4 weeks ahead).

    `feature_columns` overrides the model input columns (None = the frozen v3
    FEATURE_COLUMNS; run.py passes the engineered list for --model lightgbm_fe).
    """
    columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    train_with_targets = build_duration_targets(df_train, horizons)
    val_with_targets = build_duration_targets(df_val, horizons)

    models = {}
    for h in horizons:
        target_col = f"y_det_h{h}"

        tr_valid = train_with_targets.dropna(subset=[target_col])
        val_valid = val_with_targets.dropna(subset=[target_col])

        Xtr = tr_valid[columns]
        ytr = tr_valid[target_col].astype(int)

        Xval = val_valid[columns]
        yval = val_valid[target_col].astype(int)

        clf = lgbm.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=config.SEED + h,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            Xtr,
            ytr,
            eval_set=[(Xval, yval)],
            callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=False)],
        )
        models[h] = clf

    return models


def predict_duration(
    duration_models: dict[int, lgbm.LGBMClassifier],
    X: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Predict duration probabilities for horizons 1..4 weeks ahead.

    `feature_columns` selects the prediction input (None = the frozen v3
    FEATURE_COLUMNS; run.py passes the engineered list for --model lightgbm_fe).
    """
    columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    X_feat = X[columns]
    out = pd.DataFrame(index=X.index)

    for h, clf in duration_models.items():
        probs = clf.predict_proba(X_feat)[:, 1]
        out[f"det_prob_h{h}"] = probs

    return out
