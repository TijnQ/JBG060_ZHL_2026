"""CatBoost — Task A cross-check #2.

The third tree family (ordered boosting, its own training procedure).
Same 3-quantile + detection design as the primary; native
``Quantile:alpha`` objective. Third opinion in the cross-check panel —
the stronger the agreement across all three families, the more any
positive result can be trusted.

Run only after the primary:

    python -m modeling.catboost.train --scope aweil
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
    from catboost import CatBoostClassifier, CatBoostRegressor

    base = {
        "iterations": 600,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 10,
        "random_seed": config.SEED,
        "task_type": "CPU",
        "verbose": False,
        "early_stopping_rounds": 100,
    }
    models = {}
    for alpha in (0.1, 0.5, 0.9):
        m = CatBoostRegressor(loss="Quantile:alpha", quantile=alpha, **base)
        models[f"q{int(alpha * 100)}"] = fit_with_early_stop(
            m, Xtr, ytr_area, Xval, yval_area
        )
    det = CatBoostClassifier(loss="Logloss", **base)
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

    print("CatBoost — Task A cross-check (primary: lightgbm)")
    features = prepare_features(args.scope)
    Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det, _ = split_frames(features)
    models = build_models(Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det)
    run_task_a_method(
        args.scope,
        "catboost",
        role="cross-check",
        features=features,
        models=models,
        shap=args.shap,
    )


if __name__ == "__main__":
    main()
