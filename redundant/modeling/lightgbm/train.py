"""LightGBM — the PRIMARY Task A backbone.

Native quantile regression for the 3-quantile forecast (10/50/90) plus a
binary classifier for "will there be a detection this week". This is the
model the research document (MODEL_RESEARCH.md §4.2) designates as the
starting backbone: fast to train, native quantiles, strong default
regularisation, and — if the signal exists at all in these 50 embargoed
features — the one most likely to find it first.

Usage (from the repo root, in an env with requirements-ml.txt installed):

    python -m modeling.lightgbm.train --scope aweil
    python -m modeling.lightgbm.train --scope aweil --shap
    python -m modeling.lightgbm.train --scope national

Read `guide.md` in this folder for what to test and the keep/kill rules.
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
    """Fit the primary model: 3 quantile regressors + 1 detection classifier.

    Conservative, nearly-default hyperparameters on purpose: the first
    question is whether *any* model beats persistence, not hyperparameter
    archaeology. (If this method survives its gates, tuning is a follow-up
    experiment, not part of the test.)
    """
    import lightgbm as lgbm

    base = {
        "objective": "regression",
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
    for alpha in (0.1, 0.5, 0.9):
        m = lgbm.LGBMRegressor(objective=f"quantile:alpha={alpha}", **base)
        models[f"q{int(alpha * 100)}"] = fit_with_early_stop(
            m, Xtr, ytr_area, Xval, yval_area
        )
    det = lgbm.LGBMClassifier(
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
    models["det"] = fit_with_early_stop(det, Xtr, ytr_det, Xval, yval_det)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=config.SCOPES,
        default=config.AWEIL_SCOPE,
        help="aweil = 5 Aweil counties (default); national = 79 counties",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="compute top-feature SHAP attribution (requires shap; best-effort)",
    )
    args = parser.parse_args()

    print("LightGBM — Task A primary backbone")
    features = prepare_features(args.scope)
    Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det, _ = split_frames(features)
    models = build_models(Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det)
    run_task_a_method(
        args.scope,
        "lightgbm",
        role="primary",
        features=features,
        models=models,
        shap=args.shap,
    )


if __name__ == "__main__":
    main()
