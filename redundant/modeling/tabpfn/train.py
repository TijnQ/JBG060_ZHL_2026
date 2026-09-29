"""Optional zero-shot tabular cross-check with TabPFN (experiment E4).

TabPFN runs an in-context learning transformer over the training set — no
gradient training. On our small, non-i.i.d. weekly series it is a useful
*independent* reference: if a zero-shot model already matches tuned
gradient boosting, the data is not carrying much more signal.

The public API changed across TabPFN major versions (2.x -> 3.x -> 9.x);
this module probes for quantile support and degrades to point predictions
when it is absent, so a version mismatch never breaks the experiment ladder.
The weight download is large (~1-3 GB) — run only when you want the check:

    python -m modeling.tabpfn.train --scope aweil

Read ``guide.md`` in this folder for what to test and the keep/kill rules.
"""

from __future__ import annotations

import argparse
import inspect

import numpy as np
import pandas as pd

from modeling import config, metrics
from modeling.features import FEATURE_COLUMNS
from modeling.methods_common import (
    headline,
    metrics_for,
    model_card,
    prepare_features,
)

__all__ = ["main", "run_tabpfn"]


def _normalise_quantiles(values, n_rows: int) -> np.ndarray:
    """Return TabPFN quantiles as rows x quantiles across API versions."""
    q = np.asarray(values)
    if q.ndim != 2:
        raise ValueError(f"expected 2-D quantile output, got shape {q.shape}")
    if q.shape[1] == n_rows and q.shape[0] != n_rows:
        q = q.T
    if q.shape[0] != n_rows or q.shape[1] < 3:
        raise ValueError(
            f"quantile output shape {q.shape} does not match {n_rows} rows"
        )
    return q[:, [0, q.shape[1] // 2, q.shape[1] - 1]]


def _predict_area_quantiles(regressor, features: pd.DataFrame) -> np.ndarray:
    """Predict q10, q50 and q90, with point predictions as a fallback."""
    parameters = inspect.signature(regressor.predict).parameters
    if "output_type" in parameters:
        try:
            options = {"output_type": "quantiles"}
            if "quantiles" in parameters:
                options["quantiles"] = [0.1, 0.5, 0.9]
            quantiles = regressor.predict(features, **options)
            print("  quantile output available - using q10/q50/q90")
            return _normalise_quantiles(quantiles, len(features))
        except (TypeError, ValueError) as exc:
            print(f"  quantile output unavailable ({exc}) - using point predictions")

    point = np.asarray(regressor.predict(features))
    return np.column_stack((point, point, point))


def _prepare_features_cached(scope: str) -> pd.DataFrame:
    """Build once and reuse the exact TabPFN feature frame after restarts."""
    tables = config.method_dirs("tabpfn")["tables"]
    path = tables / f"task_a_features_{scope}.parquet"
    if path.exists():
        print(f"loading cached features from {path}")
        return pd.read_parquet(path)
    features = prepare_features(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    print(f"cached features at {path}")
    return features


def run_tabpfn(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit TabPFN on the train split, predict val+test.

    ``features`` must already carry baselines and a spike threshold
    (see the module ``main`` below).
    """
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except ImportError as exc:
        raise SystemExit(
            "tabpfn is not installed (optional). "
            "Install it from requirements-ml.txt to run this cross-check."
        ) from exc

    train = features[features["split"] == "train"]
    test = features[features["split"] != "train"]
    Xtr, Xtest = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]

    print(f"TabPFN regressor: {len(Xtr)} train rows x {Xtr.shape[1]} features")
    reg = TabPFNRegressor()
    reg.fit(Xtr, train["y_true"])

    area_quantiles = _predict_area_quantiles(reg, Xtest)

    clf = TabPFNClassifier()
    clf.fit(Xtr, train["y_det"])
    det_prob = pd.Series(clf.predict_proba(Xtest)[:, 1], index=test.index)

    pred = pd.DataFrame(
        {
            "county": test["county"],
            "week": test["week"],
            "split": test["split"],
            "y_true": test["y_true"],
            "q10": np.clip(area_quantiles[:, 0], 0, None),
            "q50": np.clip(area_quantiles[:, 1], 0, None),
            "q90": np.clip(area_quantiles[:, 2], 0, None),
            "det_prob": det_prob,
            "y_pred_baseline": test["y_pred_baseline"],
            "spike_threshold": test["spike_threshold"],
        }
    )
    return pred, metrics.summarise_forecast(pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=config.SCOPES, default=config.AWEIL_SCOPE)
    args = parser.parse_args()

    print("TabPFN — Task A zero-shot cross-check (primary: lightgbm)")
    features = _prepare_features_cached(args.scope)

    pred, _ = run_tabpfn(features)
    pred["model"] = "tabpfn"
    all_metrics = metrics_for(pred)

    dirs = config.method_dirs("tabpfn")
    dirs["tables"].mkdir(parents=True, exist_ok=True)
    for suffix, frame in (("predictions", pred), ("metrics", all_metrics)):
        out = dirs["tables"] / f"task_a_tabpfn_{suffix}_{args.scope}.csv"
        frame.to_csv(out, index=False)
        print(f"wrote {out}")
    card_path = dirs["tables"] / f"task_a_modelcard_{args.scope}.md"
    card_path.write_text(
        model_card("tabpfn", "cross-check", args.scope, features, all_metrics),
        encoding="utf-8",
    )
    print(f"wrote {card_path}")
    headline("tabpfn", "cross-check", args.scope, all_metrics)


if __name__ == "__main__":
    main()
