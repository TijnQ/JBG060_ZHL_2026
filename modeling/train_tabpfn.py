"""Optional zero-shot tabular cross-check with TabPFN (experiment E4).

TabPFN runs an in-context learning transformer over the training set — no
gradient training. On our small, non-i.i.d. weekly series it is a useful
*independent* reference: if a zero-shot model already matches tuned
gradient boosting, the data is not carrying much more signal.

The public API changed across TabPFN major versions (2.x -> 3.x -> 9.x);
this module probes for quantile support and degrades to point predictions
when it is absent, so a version mismatch never breaks the experiment ladder.
The weight download is large (~1-3 GB) — run only when you want the check:

    python -m modeling.train_tabpfn --scope aweiL
"""

from __future__ import annotations

import argparse
import inspect

import numpy as np
import pandas as pd

from modeling import config, metrics
from modeling.baselines import add_baselines
from modeling.features import FEATURE_COLUMNS, add_spike_threshold, build_features

__all__ = ["main", "run_tabpfn"]


def _supports_quantiles(regressor) -> bool:
    return "output_type" in inspect.signature(regressor.predict).parameters


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

    point = reg.predict(Xtest)
    q10 = q50 = q90 = pd.Series(point, index=test.index)
    if _supports_quantiles(reg):
        try:
            q = np.asarray(reg.predict(Xtest, output_type="quantiles"))
            if q.shape[1] >= 3:
                idx = np.argsort(q[0])
                q10 = pd.Series(q[:, idx[0]], index=test.index)
                q50 = pd.Series(q[:, idx[len(idx) // 2]], index=test.index)
                q90 = pd.Series(q[:, idx[-1]], index=test.index)
                print("  quantile output available — using 3 quantiles")
        except TypeError:
            print("  quantile output unavailable — using point predictions")

    clf = TabPFNClassifier()
    clf.fit(Xtr, train["y_det"])
    det_prob = pd.Series(clf.predict_proba(Xtest)[:, 1], index=test.index)

    pred = pd.DataFrame(
        {
            "county": test["county"],
            "week": test["week"],
            "split": test["split"],
            "y_true": test["y_true"],
            "q10": q10.clip(lower=0),
            "q50": q50.clip(lower=0),
            "q90": q90.clip(lower=0),
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

    features = build_features(args.scope)
    features = add_baselines(features, features["split"] == "train")
    features = add_spike_threshold(features)

    pred, score = run_tabpfn(features)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    for suffix, frame in (("predictions", pred), ("metrics", score)):
        out = config.TABLES / f"task_a_tabpfn_{suffix}_{args.scope}.csv"
        frame.to_csv(out, index=False)
        print(f"wrote {out}")
    print(score.to_string(index=False))


if __name__ == "__main__":
    main()
