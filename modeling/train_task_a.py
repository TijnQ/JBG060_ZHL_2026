"""Task A driver: probabilistic weekly county flood extent (experiments E2-E4).

On a strict temporal split (train 2000-2014 / validation 2015-2019 /
test 2020-2025, boundary weeks purged, features embargoed to Friday of the
week before the target week) this trains:

- LightGBM quantile regression (alpha 0.1 / 0.5 / 0.9) + binary detection;
- XGBoost ``reg:quantileresponse`` + ``binary:logistic`` (cross-check);
- CatBoost ``Quantile`` + classifier (cross-check);

and scores all of them — plus the three trivial baselines (climatology,
persistence, last detection from ``baselines.py``) — with the protocol in
``metrics.py``: MAE/RMSE/skill-vs-baseline, CRPS from the 3 quantiles, and
POD/FAR/CSI on the positive and spike slices. The project's headline
question is answered per split: **does the model beat persistence on spike
weeks?**

Usage:
    python -m modeling.train_task_a --scope aweiL           # primary scope
    python -m modeling.train_task_a --scope national
    python -m modeling.train_task_a --scope aweiL --shap    # + SHAP (E4)

Outputs (under ``modeling/outputs/``):
    tables/task_a_predictions_{scope}.csv   val+test rows, all models
    tables/task_a_metrics_{scope}.csv       tidy metrics (model x split x slice)
    tables/task_a_modelcard_{scope}.md      one-page model card for the report
    figures/task_a_top_shap_{scope}.png     (with --shap)
    models/task_a_lgbm_q50_{scope}.txt      fitted LightGBM booster
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from modeling import config, metrics
from modeling.baselines import add_baselines
from modeling.features import FEATURE_COLUMNS, add_spike_threshold, build_features
from modeling.splits import split_masks

__all__ = ["build_model", "main", "train_scope"]

# Conservative defaults for a few-thousand-row, spiky, non-i.i.d. problem:
# shallow trees, small learning rate, many rounds + early stopping.
_LGBM_BASE = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": config.SEED,
    "n_jobs": -1,
    "verbose": -1,
}
_ALPHAS = (0.1, 0.5, 0.9)
FAMILIES = ("lgbm", "xgb", "cat")


def build_model(family: str, kind: str, alpha: float | None = None):
    """One unfitted model per (family, kind). kind: 'area' | 'det'."""
    if family == "lgbm":
        import lightgbm as lgbm

        if kind == "area":
            return lgbm.LGBMRegressor(objective="quantile", alpha=alpha or 0.5, **_LGBM_BASE)
        return lgbm.LGBMClassifier(objective="binary", **_LGBM_BASE)
    if family == "xgb":
        import xgboost as xgb

        base = {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": config.SEED,
            "n_jobs": -1,
            "tree_method": "hist",
            "early_stopping_rounds": 100,
        }
        if kind == "area":
            return xgb.XGBRegressor(
                objective="reg:quantileresponse",
                quantile_alpha=alpha or 0.5,
                eval_metric="mae",
                **base,
            )
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **base)
    if family == "cat":
        import catboost as cb

        base = {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 10,
            "random_seed": config.SEED,
            "verbose": False,
            "early_stopping_rounds": 100,
        }
        if kind == "area":
            return cb.CatBoostRegressor(loss_function=f"Quantile:alpha={alpha or 0.5}", **base)
        return cb.CatBoostClassifier(**base)
    raise ValueError(f"unknown family {family!r}")


def _fit_with_early_stop(model, Xtr, ytr, Xval, yval):
    module = model.__class__.__module__.split(".")[0]
    if module == "lightgbm":
        import lightgbm as lgbm

        model.fit(
            Xtr, ytr, eval_set=[(Xval, yval)],
            callbacks=[lgbm.early_stopping(100, verbose=False)],
        )
    elif module == "xgboost":
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    elif module == "catboost":
        model.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
    else:  # pragma: no cover
        model.fit(Xtr, ytr)
    return model


def _booster(model):
    """Underlying booster (LightGBM sklearn wrapper) for SHAP."""
    return getattr(model, "booster_", model)


def _model_frame(
    name: str,
    sub: pd.DataFrame,
    q: dict | None,
    det_prob: pd.Series | None,
    baseline_col: str = "y_pred_baseline",
) -> pd.DataFrame:
    rows = {
        "model": name,
        "county": sub["county"],
        "week": sub["week"],
        "split": sub["split"],
        "y_true": sub["y_true"],
        "y_pred_baseline": sub[baseline_col],
        "spike_threshold": sub["spike_threshold"],
    }
    if q is not None:
        rows.update(
            {f"q{int(a * 100)}": np.clip(q[a].predict(sub[FEATURE_COLUMNS]), 0, None)
             for a in _ALPHAS}
        )
        rows["det_prob"] = det_prob
    else:  # trivial baseline: degenerate quantiles, detection = baseline > 0
        for a in _ALPHAS:
            rows[f"q{int(a * 100)}"] = sub[baseline_col]
        rows["det_prob"] = (sub[baseline_col] > 0).astype(float)
    return pd.DataFrame(rows)


def train_scope(scope: str, shap: bool = False) -> dict[str, str]:
    """Run the full Task A pipeline for one scope; returns a path map."""
    features = build_features(scope)
    features = add_baselines(features, features["split"] == "train")
    features = add_spike_threshold(features)

    weeks = pd.DatetimeIndex(features["week"].unique())
    masks = split_masks(weeks)
    train_mask = features["split"] == "train"
    val_mask = features["week"].map(masks["val"])
    test_mask = features["week"].map(masks["test"])

    Xtr, Xval = features.loc[train_mask, FEATURE_COLUMNS], features.loc[val_mask, FEATURE_COLUMNS]
    ytr_area, ytr_det = features.loc[train_mask, "y_true"], features.loc[train_mask, "y_det"]
    yval_area, yval_det = features.loc[val_mask, "y_true"], features.loc[val_mask, "y_det"]
    sub = features[val_mask | test_mask]
    print(
        f"scope={scope}: train={int(train_mask.sum())} rows, "
        f"val={int(val_mask.sum())} rows, test={int(test_mask.sum())} rows, "
        f"features={len(FEATURE_COLUMNS)}"
    )

    predictions: list[pd.DataFrame] = []
    fitted: dict[str, tuple[dict, object]] = {}
    for family in FAMILIES:
        try:
            area_models = {
                a: _fit_with_early_stop(
                    build_model(family, "area", a), Xtr, ytr_area, Xval, yval_area
                )
                for a in _ALPHAS
            }
            det_model = _fit_with_early_stop(
                build_model(family, "det"), Xtr, ytr_det, Xval, yval_det
            )
        except ImportError as exc:
            print(f"  {family}: not installed ({exc.name}) — skipping (optional)")
            continue
        fitted[family] = (area_models, det_model)
        predictions.append(
            _model_frame(
                family,
                sub,
                area_models,
                det_prob=pd.Series(
                    det_model.predict_proba(sub[FEATURE_COLUMNS])[:, 1], index=sub.index
                ),
            )
        )
        best = area_models[0.5]
        n_best = getattr(best, "best_iteration_", None)
        print(f"  {family}: fitted (q50 best iteration: {n_best})")

    # Trivial baselines expressed as degenerate quantile forecasts, so the
    # metric table stays comparable to the learned models.
    for base_name, base_col in (
        ("persistence", "y_pred_persist"),
        ("climatology", "y_pred_clim"),
        ("lastdet", "y_pred_lastdet"),
    ):
        predictions.append(_model_frame(base_name, sub, None, None, baseline_col=base_col))

    all_pred = pd.concat(predictions, ignore_index=True)

    # ---- metrics -----------------------------------------------------------
    metric_frames = [
        metrics.summarise_forecast(group).assign(model=model)
        for model, group in all_pred.groupby("model", observed=True)
    ]
    metric_frames = [f for f in metric_frames if len(f)]
    all_metrics = (
        pd.concat(metric_frames, ignore_index=True)
        .sort_values(["model", "split", "slice"])
        .reset_index(drop=True)
    )

    # ---- outputs ------------------------------------------------------------
    config.TABLES.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    pred_path = config.TABLES / f"task_a_predictions_{scope}.csv"
    all_pred.to_csv(pred_path, index=False)
    paths["predictions"] = str(pred_path)

    met_path = config.TABLES / f"task_a_metrics_{scope}.csv"
    all_metrics.to_csv(met_path, index=False)
    paths["metrics"] = str(met_path)

    card = _model_card(scope, features, all_metrics, fitted, shap)
    card_path = config.TABLES / f"task_a_modelcard_{scope}.md"
    card_path.write_text(card, encoding="utf-8")
    paths["modelcard"] = str(card_path)
    print(f"wrote {pred_path}\nwrote {met_path}\nwrote {card_path}")

    if "lgbm" in fitted:
        booster_path = config.MODELS / f"task_a_lgbm_q50_{scope}.txt"
        fitted["lgbm"][0][0.5].booster_.save_model(str(booster_path))
        paths["lgbm_booster"] = str(booster_path)
        print(f"wrote {booster_path}")

    if shap and "lgbm" in fitted:
        from modeling.shap_report import shap_plot, shap_summary

        summary = shap_summary(
            _booster(fitted["lgbm"][0][0.5]),
            Xtr,
            list(FEATURE_COLUMNS),
        )
        shap_path = config.FIGURES / f"task_a_top_shap_{scope}.png"
        shap_plot(summary, shap_path, title=f"Task A {scope}: mean |SHAP| (q50)")
        shap_csv = config.TABLES / f"task_a_shap_{scope}.csv"
        summary.to_csv(shap_csv, index=False)
        paths["shap"] = str(shap_path)
        print(f"wrote {shap_path}")

    # Headline: test-split spike slice, q50 vs persistence.
    head = all_metrics[
        (all_metrics["model"] == "lgbm")
        & (all_metrics["split"] == "test")
        & (all_metrics["slice"] == "spike")
    ]
    if len(head):
        row = head.iloc[0]
        print(
            f"\nHEADLINE ({scope}, test, spike weeks): MAE q50={row['mae_q50']:.1f} km2 "
            f"vs persistence {row['mae_baseline']:.1f} km2 "
            f"(skill {row['skill_q50']:+.2f}); CSI {row['csi']:.2f}"
        )
    return paths


def _metrics_table(metrics_df: pd.DataFrame) -> str:
    try:
        return metrics_df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:  # tabulate not installed
        nl = chr(10)
        return nl + "```" + nl + metrics_df.to_string(index=False) + nl + "```"


def _model_card(
    scope: str,
    features: pd.DataFrame,
    metrics_df: pd.DataFrame,
    fitted: dict,
    shap: bool,
) -> str:
    counts = features.groupby("split").size()
    lines = [
        f"# Task A model card — {scope} ({datetime.now(timezone.utc).date().isoformat()})",
        "",
        f"- Scope: **{scope}** ({len(features['county'].unique())} counties)",
        (
            f"- Data: weekly detected flood area (km2), 2000-2025; "
            f"train/val/test rows: "
            f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)}"
        ),
        (
            f"- Features: {len(FEATURE_COLUMNS)} "
            f"(embargoed to Friday before the target week), see `modeling/features.py`"
        ),
        f"- Models fitted: {', '.join(fitted) or 'none installed'}",
        (
            "- Split: strict temporal (2000-2014 / 2015-2019 / 2020-2025), "
            "boundary weeks purged, 3-day label embargo on all features."
        ),
        "",
        "## Headline metrics (model x split x slice)",
        "",
        _metrics_table(metrics_df),
        "",
        "## Reading the card",
        "",
        "- `skill_q50` > 0 means the model beats the persistence baseline on that slice.",
        (
            "- The spike slice is the operationally important one: a model that only wins on "
            "dry weeks is useless."
        ),
        "- `det_prob` drives POD/FAR/CSI at threshold 0.5.",
        "",
    ]
    if shap:
        lines.append(f"(SHAP attribution: see `task_a_top_shap_{scope}.png`)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=config.SCOPES, default=config.AWEIL_SCOPE)
    parser.add_argument("--shap", action="store_true", help="add SHAP attribution (needs shap)")
    args = parser.parse_args()
    train_scope(args.scope, shap=args.shap)


if __name__ == "__main__":
    main()
