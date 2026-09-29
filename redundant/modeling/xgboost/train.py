"""XGBoost — Task A cross-check #1.

Same 3-quantile + detection design as the LightGBM primary
(``lightgbm/train.py``), but with ``reg:quantileresponse`` and
``binary:logistic``. Its job is narrow: does a second tree family, with a
different optimiser and regulariser, reach the same answer? If it does,
the LightGBM result is robust; if it doesn't, the LightGBM result is
suspect (overfit to its own inductive biases) — either way, that is the
information this folder exists to provide.

Run it only *after* the primary (the model card compares against the
primary's saved metrics):

    python -m modeling.xgboost.train --scope aweil
"""

from __future__ import annotations

import argparse

from modeling import config
from modeling.methods_common import (
    fit_with_early_stop,
    prepare_features,
    run_task_a_method,
    split_frames,
)

__all__ = ["build_models", "main"]


def build_models(Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det) -> dict:
    import xgboost as xgb

    base = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 30,
        "reg_lambda": 1.0,
        "random_state": config.SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    }
    models = {}
    for alpha in (0.1, 0.5, 0.9):
        m = xgb.XGBRegressor(objective="reg:quantileresponse", quantile_alpha=alpha, **base)
        models[f"q{int(alpha * 100)}"] = fit_with_early_stop(
            m, Xtr, ytr_area, Xval, yval_area
        )
    det = xgb.XGBClassifier(objective="binary:logistic", **base)
    models["det"] = fit_with_early_stop(det, Xtr, ytr_det, Xval, yval_det)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=config.SCOPES, default=config.AWEIL_SCOPE
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="top-feature SHAP attribution (best-effort)",
    )
    args = parser.parse_args()

    print("XGBoost — Task A cross-check (primary: lightgbm)")
    features = prepare_features(args.scope)
    Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det, _ = split_frames(features)
    models = build_models(Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det)
    run_task_a_method(
        args.scope,
        "xgboost",
        role="cross-check",
        features=features,
        models=models,
        shap=args.shap,
    )


if __name__ == "__main__":
    main()
