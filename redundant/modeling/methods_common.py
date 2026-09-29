"""Shared pipeline core for the Task A tabular model methods.

The method folders (``lightgbm/``, ``xgboost/``, ``catboost/``,
``randomforest/``, ``tabpfn/``) each own *how their model is built and
fitted*; this module owns everything around that which is identical for
all of them:

- feature preparation (build + baselines + spike threshold),
- the split/embargo layout (via ``splits.py``),
- prediction-frame construction (quantiles clipped at 0 + detection prob),
- the three trivial baselines as degenerate quantile forecasts,
- metrics assembly (``metrics.summarise_forecast`` + role column + ordering),
- per-method output writing, model cards and the headline print.

Conventions
-----------
- A method passes ``models`` as a dict with keys ``"q10" | "q50" | "q90"``
  (fitted objects with ``.predict(X)``) and ``"det"`` (fitted classifier with
  ``.predict_proba(X)``). If all three quantile keys are the *same* object,
  the method is a point-forecaster: the 3-quantile forecast is degenerate
  (100% mass at the point prediction) and ``metrics.crps_quantiles`` reduces
  to ``|y - yhat|`` — an honest, comparable number (a proper quantile model
  can only do better than its own degenerate version).
- The primary backbone is ``config.TASK_A_METHODS[0]`` (LightGBM). The model
  card of a cross-check automatically compares against the primary's saved
  metrics file when it exists (ladder: run the primary first).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from modeling import config, metrics
from modeling.baselines import add_baselines
from modeling.features import FEATURE_COLUMNS, add_spike_threshold, build_features
from modeling.splits import split_masks

__all__ = [
    "ALPHAS",
    "baseline_frames",
    "fit_with_early_stop",
    "headline",
    "metrics_for",
    "model_card",
    "model_frame",
    "prepare_features",
    "primary_skill",
    "run_task_a_method",
    "split_frames",
]

ALPHAS = (0.1, 0.5, 0.9)

_BASELINE_COLUMNS = {
    "persistence": "y_pred_persist",
    "climatology": "y_pred_clim",
    "lastdet": "y_pred_lastdet",
}


def prepare_features(scope: str) -> pd.DataFrame:
    """Feature frame + the three baselines + spike threshold for one scope."""
    features = build_features(scope)
    features = add_baselines(features, features["split"] == "train")
    return add_spike_threshold(features)


def split_frames(features: pd.DataFrame):
    """(Xtr, ytr_area, ytr_det, Xval, yval_area, yval_det, sub) on the
    purged temporal split; ``sub`` = validation + test rows to score."""
    weeks = pd.DatetimeIndex(features["week"].unique())
    masks = split_masks(weeks)
    train_mask = features["split"] == "train"
    val_mask = features["week"].map(masks["val"])
    test_mask = features["week"].map(masks["test"])
    Xtr = features.loc[train_mask, FEATURE_COLUMNS]
    Xval = features.loc[val_mask, FEATURE_COLUMNS]
    return (
        Xtr,
        features.loc[train_mask, "y_true"],
        features.loc[train_mask, "y_det"],
        Xval,
        features.loc[val_mask, "y_true"],
        features.loc[val_mask, "y_det"],
        features[val_mask | test_mask],
    )


def fit_with_early_stop(model, Xtr, ytr, Xval, yval):
    """Fit with family-appropriate early stopping (100 rounds)."""
    module = model.__class__.__module__.split(".")[0]
    if module == "lightgbm":
        import lightgbm as lgbm

        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            callbacks=[lgbm.early_stopping(100, verbose=False)],
        )
    elif module == "xgboost":
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    elif module == "catboost":
        model.fit(Xtr, ytr, eval_set=(Xval, yval), verbose=False)
    else:
        model.fit(Xtr, ytr)
    return model


def model_frame(
    name: str,
    sub: pd.DataFrame,
    q: dict | None,
    det_prob: pd.Series | None,
    baseline_col: str = "y_pred_baseline",
) -> pd.DataFrame:
    """One prediction frame per model (or degenerate baseline)."""
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
        rows.update({f"q{int(a * 100)}": np.clip(q[a].to_numpy(), 0, None)
                     for a in ALPHAS})
        rows["det_prob"] = det_prob
    else:  # trivial baseline: degenerate quantiles, detection = baseline > 0
        for a in ALPHAS:
            rows[f"q{int(a * 100)}"] = sub[baseline_col].to_numpy()
        rows["det_prob"] = (sub[baseline_col] > 0).astype(float).to_numpy()
    return pd.DataFrame(rows, index=sub.index)


def baseline_frames(sub: pd.DataFrame) -> list[pd.DataFrame]:
    """The three trivial baselines, in headline order (persistence first)."""
    return [
        model_frame(name, sub, None, None, baseline_col=col)
        for name, col in _BASELINE_COLUMNS.items()
    ]


def metrics_for(all_pred: pd.DataFrame) -> pd.DataFrame:
    """Tidy metrics per model with a ``role`` column and priority ordering
    (primary first, then the other methods in ladder order, then baselines)."""
    frames = [
        metrics.summarise_forecast(group).assign(model=model)
        for model, group in all_pred.groupby("model", observed=True)
    ]
    frames = [f for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True)

    def _role(m: str) -> str:
        if m == config.TASK_A_METHODS[0]:
            return "primary"
        if m in config.TASK_A_METHODS:
            return "cross-check"
        return "baseline"

    order = {m: i for i, m in enumerate(config.TASK_A_METHODS)}
    base_rank = {n: len(config.TASK_A_METHODS) + i
                 for i, n in enumerate(_BASELINE_COLUMNS)}
    out["role"] = out["model"].map(_role)
    out["_rank"] = out["model"].map(lambda m: order.get(m, base_rank.get(m, 99)))
    return (
        out.sort_values(["_rank", "split", "slice"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def primary_skill(scope: str, split: str = "test", slc: str = "all") -> float | None:
    """The primary backbone's saved skill for (split, slice), if it has run."""
    path = config.method_dirs(config.TASK_A_METHODS[0])["tables"] / f"task_a_metrics_{scope}.csv"
    if not path.exists():
        return None
    tab = pd.read_csv(path)
    row = tab[(tab["model"] == config.TASK_A_METHODS[0])
              & (tab["split"] == split) & (tab["slice"] == slc)]
    if not len(row):
        return None
    val = row.iloc[0]["skill_q50"]
    return None if pd.isna(val) else float(val)


def headline(method: str, role: str, scope: str, all_metrics: pd.DataFrame) -> None:
    """Test-split spike-slice headline for the method (vs persistence)."""
    head = all_metrics[
        (all_metrics["model"] == method)
        & (all_metrics["split"] == "test")
        & (all_metrics["slice"] == "spike")
    ]
    if not len(head):
        return
    row = head.iloc[0]
    tag = "primary backbone" if role == "primary" else "cross-check"
    print(
        f"\nHEADLINE ({scope}, {method}, {tag} | test, spike weeks): "
        f"MAE q50={row['mae_q50']:.1f} km2 "
        f"vs persistence {row['mae_baseline']:.1f} km2 "
        f"(skill {row['skill_q50']:+.2f}); CSI {row['csi']:.2f}"
    )


def _metrics_table(metrics_df: pd.DataFrame) -> str:
    try:
        return metrics_df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:  # tabulate not installed
        nl = chr(10)
        return nl + "```" + nl + metrics_df.to_string(index=False) + nl + "```"


def model_card(
    method: str,
    role: str,
    scope: str,
    features: pd.DataFrame,
    metrics_df: pd.DataFrame,
    shap: bool = False,
) -> str:
    """One-page model card for the report, written per method folder."""
    counts = features.groupby("split").size()
    prim = config.TASK_A_METHODS[0]
    if role == "primary":
        models_line = (
            f"- Models: **{method}** — the designated primary backbone "
            f"(MODEL_RESEARCH.md §4.2). A cross-check is only promoted if it "
            f"beats this model's test-all skill by more than 0.05."
        )
    else:
        models_line = (
            f"- Models: **{method}** — cross-check only. Verdict below compares "
            f"it against the saved primary (`{prim}`) run."
        )
    lines = [
        (
            f"# Task A model card — {method} — {scope} "
            f"({datetime.now(timezone.utc).date().isoformat()})"
        ),
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
        "## Headline metrics (split x slice)",
        "",
        _metrics_table(
            metrics_df[metrics_df["model"] == method]
            .drop(columns=["model", "role"], errors="ignore")
        ),
        "",
        "## Reading the card",
        "",
        "- `skill_q50` > 0 means this model beats the persistence baseline on that slice.",
        (
            "- The spike slice is the operationally important one: a model that only wins on "
            "dry weeks is useless."
        ),
        "- `det_prob` drives POD/FAR/CSI at threshold 0.5.",
        "",
        "## Keep / kill verdict",
        "",
    ]
    lines.extend(_verdict_lines(method, role, scope, metrics_df))
    if shap:
        lines.append(f"(SHAP attribution: see `figures/task_a_top_shap_{scope}.png`)")
    lines.append("")
    return "\n".join(lines)


def _verdict_lines(
    method: str, role: str, scope: str, metrics_df: pd.DataFrame
) -> list[str]:
    """Automatic keep/kill comparison against the baselines (and, for a
    cross-check, against the saved primary run)."""

    def _skill(split: str, slc: str) -> float | None:
        row = metrics_df[
            (metrics_df["model"] == method)
            & (metrics_df["split"] == split)
            & (metrics_df["slice"] == slc)
        ]
        if not len(row):
            return None
        val = row.iloc[0]["skill_q50"]
        return None if pd.isna(val) else float(val)

    lines: list[str] = []
    prim = config.TASK_A_METHODS[0]
    if role == "cross-check":
        p = primary_skill(scope, "test", "all")
        own = _skill("test", "all")
        if p is not None and own is not None:
            if own > p + 0.05:
                lines.append(
                    f"- **{method} beats the primary backbone** on test-all skill "
                    f"({own:+.2f} vs {p:+.2f} for `{prim}`) — investigate promotion "
                    f"(guide, §Keep/kill) before the report is written."
                )
            else:
                lines.append(
                    f"- {method} {own:+.2f} vs primary `{prim}` {p:+.2f} — no meaningful "
                    "uplift over the backbone; default verdict: **kill** (delete the "
                    "folder) unless the spike slice tells a different story."
                )
        else:
            lines.append(
                f"- Primary `{prim}` has no saved metrics for this scope yet — run "
                f"`python -m modeling.{prim}.train --scope {scope}` first, then "
                "compare test/skill_q50 rows manually (guide, §Keep/kill)."
            )
    spike = _skill("test", "spike")
    if spike is not None:
        if spike > 0:
            lines.append(
                f"- test spike skill {spike:+.2f} > 0 — beats persistence on the "
                "operationally important weeks."
            )
        else:
            lines.append(
                f"- test spike skill {spike:+.2f} <= 0 — **does not beat persistence on "
                "spike weeks: this method earns its keep only if the cross-checks "
                "do; otherwise kill.**"
            )
    return lines


def run_task_a_method(
    scope: str,
    method: str,
    role: str,
    features: pd.DataFrame,
    models: dict,
    shap: bool = False,
    shap_model=None,
    extras: list | None = None,
) -> dict[str, str]:
    """Score a fitted method on val+test and write its per-method outputs.

    ``models``: keys ``"q10" | "q50" | "q90"`` (fitted, ``.predict``; the same
    object in all three = degenerate point forecast) and ``"det"`` (``.predict_proba``).
    ``shap_model``: the fitted tree model to attribute (defaults to the q50 one).
    ``extras``: optional ``(name, DataFrame)`` pairs written as extra CSVs
    (e.g. random-forest feature importances).

    Returns a path map; prints the headline.
    """
    Xtr, _ytr_area, _ytr_det, Xval, _yval_area, _yval_det, sub = split_frames(features)
    print(
        f"{method} | scope={scope} role={role}: "
        f"train={len(Xtr)} rows, val={len(Xval)} rows, "
        f"scored={len(sub)} rows, features={len(FEATURE_COLUMNS)}"
    )

    _cache: dict[int, pd.Series] = {}

    def _q(a: float) -> pd.Series:
        m = models[f"q{int(a * 100)}"]
        key = id(m)
        if key not in _cache:
            _cache[key] = pd.Series(
                np.clip(np.asarray(m.predict(sub[FEATURE_COLUMNS]), dtype=float), 0, None),
                index=sub.index,
            )
        return _cache[key]

    q = {a: _q(a) for a in ALPHAS}
    det_prob = pd.Series(
        np.asarray(models["det"].predict_proba(sub[FEATURE_COLUMNS]))[:, 1],
        index=sub.index,
    )

    all_pred = pd.concat([model_frame(method, sub, q, det_prob)] + baseline_frames(sub),
                         ignore_index=True)
    all_metrics = metrics_for(all_pred)

    dirs = config.method_dirs(method)
    dirs["tables"].mkdir(parents=True, exist_ok=True)
    dirs["figures"].mkdir(parents=True, exist_ok=True)
    dirs["models"].mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    pred_path = dirs["tables"] / f"task_a_predictions_{scope}.csv"
    all_pred.to_csv(pred_path, index=False)
    paths["predictions"] = str(pred_path)

    met_path = dirs["tables"] / f"task_a_metrics_{scope}.csv"
    all_metrics.to_csv(met_path, index=False)
    paths["metrics"] = str(met_path)

    card_path = dirs["tables"] / f"task_a_modelcard_{scope}.md"
    card_path.write_text(
        model_card(method, role, scope, features, all_metrics, shap), encoding="utf-8"
    )
    paths["modelcard"] = str(card_path)

    if extras:
        for name, frame in extras:
            p = dirs["tables"] / f"task_a_{name}_{scope}.csv"
            frame.to_csv(p, index=False)
            paths[name] = str(p)

    if shap:
        from modeling.shap_report import shap_plot, shap_summary

        target = shap_model if shap_model is not None else models["q50"]
        try:
            summary = shap_summary(target, Xtr, list(FEATURE_COLUMNS))
            shap_path = dirs["figures"] / f"task_a_top_shap_{scope}.png"
            shap_plot(summary, shap_path, title=f"Task A {method} ({scope}): mean |SHAP|")
            shap_csv = dirs["tables"] / f"task_a_shap_{scope}.csv"
            summary.to_csv(shap_csv, index=False)
            paths["shap"] = str(shap_path)
            print(f"wrote {shap_path}")
        except Exception as exc:  # noqa: BLE001 — attribution is best-effort
            print(f"  shap: skipped ({type(exc).__name__}: {exc})")

    print(f"wrote {pred_path}\nwrote {met_path}\nwrote {card_path}")
    headline(method, role, scope, all_metrics)
    return paths
