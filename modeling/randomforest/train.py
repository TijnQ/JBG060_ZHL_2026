"""Random Forest — Task A cross-check #3 (sklearn).

A bagged-trees model with *no* native quantile regression, so this method
forecasts a **degenerate** 3-quantile forecast: q10 = q50 = q90 = the mean
prediction. ``metrics.crps_quantiles`` on such a forecast reduces exactly
to ``|y - yhat|`` (100% point mass at the prediction), so its CRPS is an
honest, directly comparable number: a proper quantile model (the LightGBM
primary) can only do *better* than a degenerate one, so the CRPS gap
between this folder's table and the primary's table measures exactly how
much the primary gains from calibrated quantiles.

No early stopping (a forest is fit once on the full train split); the
extra outputs include the model's own feature importances, which are a
cheap second opinion on the SHAP ranking.

    python -m modeling.randomforest.train --scope aweil
"""

from __future__ import annotations

import argparse

import pandas as pd

from modeling import config
from modeling.features import FEATURE_COLUMNS
from modeling.methods_common import prepare_features, run_task_a_method

__all__ = ["build_models", "main"]


def build_models(features: pd.DataFrame) -> dict:
    """Fit one RF regressor (point) + one RF classifier (detection).

    The same fitted regressor is used for all three quantile slots —
    that *is* the degenerate forecast (see module docstring).
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    train = features[features["split"] == "train"]
    Xtr = train[FEATURE_COLUMNS]
    base = {
        "n_estimators": 400,
        "min_samples_leaf": 10,
        "max_features": 0.3,
        "n_jobs": -1,
        "random_state": config.SEED,
    }
    reg = RandomForestRegressor(**base)
    reg.fit(Xtr, train["y_true"])
    clf = RandomForestClassifier(**base)
    clf.fit(Xtr, train["y_det"])
    return {"q10": reg, "q50": reg, "q90": reg, "det": clf}


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

    print("Random Forest — Task A cross-check (primary: lightgbm)")
    features = prepare_features(args.scope)
    models = build_models(features)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": models["q50"].feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    run_task_a_method(
        args.scope,
        "randomforest",
        role="cross-check",
        features=features,
        models=models,
        shap=args.shap,
        extras=[("importances", importance)],
    )


if __name__ == "__main__":
    main()
