"""LightGBM model training and prediction module for LightGBM_v1 pipeline.

Trains:
1. LGBMClassifier for flood detection probability (`det_prob`).
2. LGBMRegressor (alpha=0.1, 0.5, 0.9) for quantile flood area in km² (`q10`, `q50`, `q90`).

Every train function falls back to the shared default hyperparameter dict
(`_BASE_PARAMS`) when called with `params=None`, so run.py behavior is exactly
unchanged; LightGBM_v1.tuning passes explicit per-head overrides.

`PARAM_SETS` is the named 2x2 hyperparameter registry (feature set x tuning
status) that run.py exposes via --params: "baseline" / "fe_baseline" resolve to
params=None (the frozen v3 defaults), while "baseline_optuna" / "fe_optuna"
carry the per-head Optuna winners from outputs/tuning/.
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
    "PARAM_SETS",
    "build_and_train_lightgbm",
    "extract_feature_importance",
    "get_param_set",
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


# --- Named hyperparameter sets (run.py --params) ------------------------------
#
# The 2x2 grid of (feature set x hyperparameters) the project compares:
#
#   |                    | v3 defaults   | Optuna winners     |
#   |--------------------|---------------|--------------------|
#   | baseline features  | "baseline"    | "baseline_optuna"  |
#   | fe features        | "fe_baseline" | "fe_optuna"        |
#
# - "baseline" / "fe_baseline": params=None -> the frozen v3 defaults
#   (_BASE_PARAMS / _DURATION_BASE_PARAMS), exactly what run.py has always
#   trained. The values are identical; "fe_baseline" exists so the grid stays
#   explicit (config.py: the fe model shares the v3 default hyperparameters).
# - "baseline_optuna" / "fe_optuna": per-head winners from
#   outputs/tuning/best_params_lgbm_aweil_{baseline,fe}_v3_all_heads.json
#   (3 sub-studies x 50 trials each; created 2026-10-09 / 2026-10-08), passed
#   through the same override merge, so they reproduce the tuning evaluation
#   (outputs/tuning/cv_metrics_tuned_lgbm_aweil_*_all_heads.csv) exactly.
#
# Cross combos (e.g. fe features + baseline_optuna) are allowed — that is the
# point of exposing the grid as a run.py argument.
_TUNED_FRAME: dict[str, object] = {
    # Fixed per-head frame from tuning.py _full_head_params: n_estimators is a
    # cap (early stopping, patience 100, picks the effective tree count);
    # max_depth/min_split_gain sit at LightGBM's defaults. random_state is
    # intentionally NOT pinned here: train_detection_classifier /
    # train_quantile_regressors fill config.SEED and train_duration_models
    # fills config.SEED + horizon — identical to the tuning runs (seed 2026).
    "n_estimators": 1200,
    "max_depth": -1,
    "subsample_freq": 1,
    "min_split_gain": 0.0,
    "n_jobs": -1,
    "verbose": -1,
}

PARAM_SETS: dict[str, dict[str, dict[str, object] | None]] = {
    "baseline": {"det": None, "quantiles": None, "duration": None},
    "fe_baseline": {"det": None, "quantiles": None, "duration": None},
    "baseline_optuna": {
        "det": {
            **_TUNED_FRAME,
            "learning_rate": 0.03374301288709528,
            "num_leaves": 53,
            "min_child_samples": 64,
            "colsample_bytree": 0.8993876625797363,
            "subsample": 0.9886512030503938,
            "reg_lambda": 0.510583075898472,
            "reg_alpha": 8.298099464514976,
        },
        "quantiles": {
            **_TUNED_FRAME,
            "learning_rate": 0.026942206158197972,
            "num_leaves": 27,
            "min_child_samples": 43,
            "colsample_bytree": 0.7649486780786229,
            "subsample": 0.8277284857284459,
            "reg_lambda": 0.001135390334129899,
            "reg_alpha": 9.877305340862936,
        },
        "duration": {
            **_TUNED_FRAME,
            "learning_rate": 0.022392518725672314,
            "num_leaves": 12,
            "min_child_samples": 24,
            "colsample_bytree": 0.525534269901265,
            "subsample": 0.7900295130662434,
            "reg_lambda": 0.007715079128989936,
            "reg_alpha": 5.889468580783452,
        },
    },
    "fe_optuna": {
        "det": {
            **_TUNED_FRAME,
            "learning_rate": 0.07996982194675185,
            "num_leaves": 121,
            "min_child_samples": 55,
            "colsample_bytree": 0.7304549344692116,
            "subsample": 0.6423607389514663,
            "reg_lambda": 0.0211950325646498,
            "reg_alpha": 8.11172392180777,
        },
        "quantiles": {
            **_TUNED_FRAME,
            "learning_rate": 0.05782526811115778,
            "num_leaves": 53,
            "min_child_samples": 39,
            "colsample_bytree": 0.958157827274865,
            "subsample": 0.9305883837704264,
            "reg_lambda": 0.0435619394614929,
            "reg_alpha": 4.669366259624795,
        },
        "duration": {
            **_TUNED_FRAME,
            "learning_rate": 0.018519653007639282,
            "num_leaves": 15,
            "min_child_samples": 40,
            "colsample_bytree": 0.5041866990950903,
            "subsample": 0.9980197319153823,
            "reg_lambda": 0.02135985663554151,
            "reg_alpha": 0.019594141859569845,
        },
    },
}


def get_param_set(name: str) -> dict[str, dict[str, object] | None]:
    """Resolve a named hyperparameter set (run.py --params); None = v3 defaults."""
    if name not in PARAM_SETS:
        raise ValueError(f"Unknown param set '{name}'. Available: {sorted(PARAM_SETS)}")
    return PARAM_SETS[name]


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


def predict_lightgbm(
    models: dict[str, object],
    X: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Generate LightGBM predictions for detection probability and quantile flood area.

    `feature_columns` selects/reorders the prediction input (None = the frozen
    v3 FEATURE_COLUMNS); run.py passes the engineered list for --model
    lightgbm_fe so the models are queried with exactly the columns they were
    fitted on.
    """
    columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    X_feat = X[columns]

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


def extract_feature_importance(
    models: dict[str, object],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Split & gain feature importance of fitted LightGBM models (long format).

    One row per (model head, importance type, feature):
    - `head`: key of the trained model dict (det, q10, q50, q90)
    - `importance_type`: "split" (split count) or "gain" (total split gain)
    - `importance`: raw importance value
    - `share_pct`: the feature's share of that head's total importance (0-100)

    Entries without a LightGBM booster (e.g. non-tree fallback models) are
    skipped, so the result is empty for the TabPFN backend.
    """
    rows: list[dict[str, object]] = []
    for head, model in models.items():
        booster = getattr(model, "booster_", None)
        if booster is None:
            continue
        names = list(booster.feature_name())
        if not names and feature_names is not None:
            names = list(feature_names)
        for importance_type in ("split", "gain"):
            values = booster.feature_importance(importance_type=importance_type)
            total = float(np.sum(values))
            for name, value in zip(names, values):
                rows.append({
                    "head": head,
                    "importance_type": importance_type,
                    "feature": name,
                    "importance": int(value) if importance_type == "split" else float(value),
                    "share_pct": (100.0 * float(value) / total) if total > 0 else 0.0,
                })
    return pd.DataFrame(rows)
