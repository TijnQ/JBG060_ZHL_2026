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
    python -m modeling.train_task_a --scope aweiL                 # LightGBM primary only (default)
    python -m modeling.train_task_a --scope aweiL --shap          # + SHAP on the primary
    python -m modeling.train_task_a --scope aweiL --families lgbm,xgb,cat  # full ladder
    python -m modeling.train_task_a --scope national

Family policy (MODEL_RESEARCH.md §4.2): LightGBM is the PRIMARY backbone. The
default run trains only LightGBM ("try LightGBM first"); XGBoost and CatBoost
are cross-checks added via ``--families`` and are only promoted if they
consistently beat the primary on the spike slice.

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

# MODEL_RESEARCH.md §4.2: LightGBM is the PRIMARY backbone for Task A.
# XGBoost and CatBoost are cross-checks only — trained on request, scored in
# the same table, promoted only if they consistently beat the primary on the
# spike slice.
PRIMARY_FAMILY = "lgbm"
FAMILY_PRIORITY = ("lgbm", "xgb", "cat")
# Backwards-compatible alias (priority order).
FAMILIES = FAMILY_PRIORITY


def resolve_families(spec: str | None) -> list[str]:
    """Families to train, in priority order.

    ``None`` (the default) trains only the primary (LightGBM) — that is the
    ladder in MODEL_RESEARCH.md: try LightGBM first. A comma-separated spec
    selects any subset (order is normalised to priority); unknown names raise.
    """
    if spec is None or spec.strip() == "":
        return [PRIMARY_FAMILY]
    wanted = [p.strip() for p in spec.split(",") if p.strip()]
    unknown = [w for w in wanted if w not in FAMILY_PRIORITY]
    if unknown:
        raise ValueError(f"unknown families {unknown}; valid: {list(FAMILY_PRIORITY)}")
    return [f for f in FAMILY_PRIORITY if f in wanted]


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


def train_scope(scope: str, shap: bool = False, families: str | None = None) -> dict[str, str]:
    """Run the full Task A pipeline for one scope; returns a path map.

    ``families``: comma-separated subset of ``FAMILY_PRIORITY`` to train;
    ``None`` = primary only (LightGBM first, per MODEL_RESEARCH.md §4.2).
    """
    fam_order = resolve_families(families)
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
    for family in fam_order:
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

    # Role column + ordering: primary backbone first, then cross-checks in
    # priority order, then the trivial baselines (persistence first — it is
    # the headline bar every model must beat).
    role = {f: ("primary" if f == PRIMARY_FAMILY else "cross-check") for f in fam_order}
    base_rank = {"persistence": len(fam_order), "climatology": len(fam_order) + 1,
                 "lastdet": len(fam_order) + 2}
    fam_rank = {f: i for i, f in enumerate(fam_order)}
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics["role"] = all_metrics["model"].map(lambda m: role.get(m, "baseline"))
    all_metrics["_rank"] = all_metrics["model"].map(lambda m: fam_rank.get(m, base_rank.get(m, 99)))
    all_metrics = (
        all_metrics.sort_values(["_rank", "split", "slice"])
        .drop(columns="_rank")
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

    card = _model_card(scope, features, all_metrics, fitted, shap, fam_order)
    card_path = config.TABLES / f"task_a_modelcard_{scope}.md"
    card_path.write_text(card, encoding="utf-8")
    paths["modelcard"] = str(card_path)
    print(f"wrote {pred_path}\nwrote {met_path}\nwrote {card_path}")

    if "lgbm" in fitted:
        booster_path = config.MODELS / f"task_a_lgbm_q50_{scope}.txt"
        fitted["lgbm"][0][0.5].booster_.save_model(str(booster_path))
        paths["lgbm_booster"] = str(booster_path)
        print(f"wrote {booster_path}")

    if shap:
        if "lgbm" in fitted:
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
        else:
            print("  shap: skipped (SHAP attribution is defined for the LightGBM primary; "
                  "this run did not fit it)")

    # Headline: first fitted family in priority order (LightGBM when fitted),
    # test-split spike slice, q50 vs the persistence baseline.
    head_fam = next((f for f in fam_order if f in fitted), None)
    if head_fam is None:
        print("\nHEADLINE: none of the requested families is installed — no model fitted")
    else:
        head = all_metrics[
            (all_metrics["model"] == head_fam)
            & (all_metrics["split"] == "test")
            & (all_metrics["slice"] == "spike")
        ]
        if len(head):
            row = head.iloc[0]
            tag = " (primary backbone)" if head_fam == PRIMARY_FAMILY else \
                " (cross-check — NOT the primary backbone; see --families)"
            print(
                f"\nHEADLINE ({scope}, test, spike weeks, {head_fam}{tag}): "
                f"MAE q50={row['mae_q50']:.1f} km2 "
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
    fam_order: list[str],
) -> str:
    counts = features.groupby("split").size()
    if PRIMARY_FAMILY in fitted:
        others = [f for f in fam_order if f != PRIMARY_FAMILY]
        models_line = f"- Models: **{PRIMARY_FAMILY} (primary backbone, per MODEL_RESEARCH.md §4.2)**"
        if others:
            models_line += (f"; cross-checks: {', '.join(others)} (promoted only if they "
                            "consistently beat the primary on the spike slice)")
    else:
        models_line = (
            f"- Models: {', '.join(fitted) or 'none installed'} "
            f"— note: {PRIMARY_FAMILY} is the designated primary backbone and was not fitted in this run"
        )
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
        models_line,
        (
            "- Split: strict temporal (2000-2014 / 2015-2019 / 2020-2025), "
            "boundary weeks purged, 3-day label embargo on all features."
        ),
        "",
        "## Headline metrics (model x split x slice)",
        "",
        "Ordered primary → cross-checks → baselines; the `role` column marks which is which.",
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
    lines.extend(_card_verdict(metrics_df, fitted, fam_order))
    if shap and "lgbm" in fitted:
        lines.append(f"(SHAP attribution: see `task_a_top_shap_{scope}.png`)")
    return "\n".join(lines)


def _card_verdict(metrics_df: pd.DataFrame, fitted: dict, fam_order: list[str]) -> list[str]:
    """Primary-vs-cross-check verdict for the card (test split, `all` slice)."""
    lines: list[str] = ["## Primary / cross-check verdict (test, all weeks)", ""]
    prim = fitted.get(PRIMARY_FAMILY)
    if prim is None:
        lines.append(f"- {PRIMARY_FAMILY} was not fitted in this run; no primary verdict.")
        return lines

    def _skill(model: str, split: str, slc: str) -> float | None:
        row = metrics_df[
            (metrics_df["model"] == model) & (metrics_df["split"] == split) & (metrics_df["slice"] == slc)
        ]
        return float(row.iloc[0]["skill_q50"]) if len(row) else None

    p_all, p_spike = _skill(PRIMARY_FAMILY, "test", "all"), _skill(PRIMARY_FAMILY, "test", "spike")
    if p_all is not None:
        lines.append(
            f"- **{PRIMARY_FAMILY} (primary)**: skill vs persistence = "
            f"{p_all:+.2f} (all weeks), {p_spike if p_spike is not None else float('nan'):+.2f} (spike weeks)."
        )
    crossed = [f for f in fam_order if f != PRIMARY_FAMILY and f in fitted]
    if not crossed:
        lines.append("- No cross-checks fitted in this run (default = primary only; "
                     "`--families lgbm,xgb,cat` adds them).")
    for f in crossed:
        c_all = _skill(f, "test", "all")
        if c_all is None:
            continue
        if p_all is not None and c_all > p_all + 0.05:
            lines.append(
                f"- **{f} (cross-check) beats the primary** on test-all skill "
                f"({c_all:+.2f} vs {p_all:+.2f}) — per §4.2, investigate before finalising "
                f"the backbone choice."
            )
        else:
            lines.append(
                f"- {f} (cross-check): {c_all:+.2f} vs primary {p_all if p_all is not None else float('nan'):+.2f} "
                "— does not beat the primary by a meaningful margin; primary stands."
            )
    if not crossed or all(_skill(f, "test", "all") is None for f in crossed):
        lines.append("- Verdict: **LightGBM remains the Task A backbone**.")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=config.SCOPES, default=config.AWEIL_SCOPE)
    parser.add_argument("--shap", action="store_true", help="add SHAP attribution (needs shap)")
    parser.add_argument(
        "--families",
        default=None,
        help=(
            "comma-separated families to train (subset of lgbm,xgb,cat); "
            "default = primary only (lgbm)"
        ),
    )
    args = parser.parse_args()
    train_scope(args.scope, shap=args.shap, families=args.families)


if __name__ == "__main__":
    main()
