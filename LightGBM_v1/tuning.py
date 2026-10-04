"""Optuna hyperparameter tuning module for LightGBM_v1 pipeline.

Tunes the existing LightGBM pipeline (detection classifier + 3 quantile
regressors) on the frozen 3 expanding-window CV folds — one study per feature
set — and (optionally) evaluates the winning parameters on the frozen protocol
test years with a run.py-format metrics table (plan.md; evaluation_metrics.md).

Design summary (plan.md v3):
- Two prefixed hyperparameter groups per trial: `det_*` for the LGBMClassifier
  and `quant_*` for the three LGBMRegressor quantile models. 7 searched
  parameters per head, 14 sampled values per trial (§6).
- Composite, climatology-normalized objective (§7):
      r_det(f) = Brier(model) / Brier(county-month climatology)
      r_q(f)   = mean pinball(model) / mean pinball(degenerate climatology)
      score(f) = 0.5 * r_det(f) + 0.5 * r_q(f)   (folds f = 1..3, minimized)
  1.0 = climatology parity, < 1.0 = added skill. The groups are separable:
  each head is optimized against its own frozen protocol metric.
- Test years are never touched during the search (D4); `--evaluate` refits the
  winner and scores them exactly once through the run.py fold loop (§10).
- Single-metric objectives `brier` / `pinball` sample and train only one head
  (~4x cheaper trials); the other head keeps today's hardcoded defaults.

Feature sets (§4.3): register engineered sets with ONE entry in FEATURE_SETS
(baseline is built-in), or call tune_lightgbm(feats_df=..., feature_columns=...)
directly. Feature authors keep the Friday embargo invariant (§4.2).

Usage:
    python -m LightGBM_v1.tuning --list-feature-sets
    python -m LightGBM_v1.tuning --feature-set baseline --n-trials 50 --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from LightGBM_v1 import (
    baselines,
    config,
    features,
    lgbm_model,
    metrics,
    splits,
)

__all__ = [
    "FEATURE_SETS",
    "evaluate_best_params",
    "tune_lightgbm",
]

# --- Tuning configuration (plan.md §6, §7, §9, §12) ----------------------------

TUNING_DIR = config.OUT_ROOT / "tuning"
DEFAULT_STORAGE = f"sqlite:///{TUNING_DIR / 'studies.db'}"
DEFAULT_N_TRIALS = 50
DEFAULT_TIMEOUT = 3600

OBJECTIVES = ("composite", "brier", "pinball")

# Fixed n_estimators cap (§6.2): early stopping (patience 100) picks the
# effective tree count; the smoke test audits that no fit pins at this cap.
N_ESTIMATORS_CAP = 1200

# Numerical floor for climatology-normalized denominators (§7).
_EPS = 1e-6

# Contract columns every feature set must provide (§4.1).
CONTRACT_COLUMNS = ("county", "week", "y_true", "y_det", "month")

# Columns emitted by metrics.evaluate_summary_metrics — identical to
# outputs/outputs_v3/tables/cv_metrics_summary_aweil.csv so the tuned table
# can be diffed/joined against the untuned one (§9).
_SUMMARY_COLS = [
    "scope", "model", "n_samples", "flood_weeks", "flood_rate_pct",
    "brier_score", "bss_vs_climatology", "bss_vs_persistence",
    "mean_pinball_loss", "coverage_flood_weeks", "coverage_all_weeks",
    "mae_flood_weeks", "bias_flood_weeks",
]

# The 7 searched hyperparameters per head (sklearn-API names, §6.1), sampled
# twice per trial with `det_` / `quant_` prefixes. Everything else stays at
# today's values (see _full_head_params).
_SEARCH_SPACE: tuple[tuple[str, str, dict], ...] = (
    ("learning_rate", "suggest_float", {"low": 0.01, "high": 0.1, "log": True}),
    ("num_leaves", "suggest_int", {"low": 8, "high": 128, "log": True}),
    ("min_child_samples", "suggest_int", {"low": 10, "high": 100}),
    ("colsample_bytree", "suggest_float", {"low": 0.5, "high": 1.0}),
    ("subsample", "suggest_float", {"low": 0.5, "high": 1.0}),
    ("reg_lambda", "suggest_float", {"low": 1e-3, "high": 10.0, "log": True}),
    ("reg_alpha", "suggest_float", {"low": 1e-3, "high": 10.0, "log": True}),
)

# Feature-set registry (§4.3): each entry returns (feats_df, feature_columns).
# Add ONE line per engineered feature set — no other code changes needed.
FEATURE_SETS: dict[str, Callable[[], tuple[pd.DataFrame, list[str]]]] = {
    "baseline": lambda: (features.build_weekly_features(), features.FEATURE_COLUMNS),
}


def _default_study_name(feature_set: str) -> str:
    """Deterministic default study name, e.g. 'lgbm_aweil_baseline_v3' (plan.md §9)."""
    return f"lgbm_aweil_{feature_set}_{config.MODEL_VERSION}"


def _resolve_feature_set(feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Build one registered feature set from the FEATURE_SETS registry."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Available: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]()


def _sample_head_params(trial: optuna.Trial, prefix: str) -> dict[str, float]:
    """Sample one head's 7 hyperparameters with `<prefix>_` param names (plan.md §6.1)."""
    params: dict[str, float] = {}
    for name, suggest_name, kwargs in _SEARCH_SPACE:
        params[name] = getattr(trial, suggest_name)(f"{prefix}_{name}", **kwargs)
    return params


def _full_head_params(searched: dict[str, float] | None, seed: int) -> dict[str, object] | None:
    """Complete per-head LightGBM params: searched values + fixed §6.2 values.

    Returns None when `searched` is empty (the head is not tuned by the study's
    objective and keeps today's hardcoded defaults via params=None).
    """
    if not searched:
        return None
    return {
        "n_estimators": N_ESTIMATORS_CAP,
        "max_depth": -1,
        "subsample_freq": 1,
        "min_split_gain": 0.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
        **searched,
    }


def _prepare_features(feats_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Enforce the plan.md §4.1 input contract; return cleaned (feats_df, feature_columns).

    - required contract columns must exist;
    - missing `y_true_lag1` / `y_det_lag1` are auto-derived (per-county shift(1)
      + fillna(0), the features.py recipe) so persistence references keep working;
    - rows with complete == False (when a `complete` column exists) or with any
      NaN feature / target are dropped.
    """
    missing_contract = [c for c in CONTRACT_COLUMNS if c not in feats_df.columns]
    if missing_contract:
        raise ValueError(f"feats_df is missing required contract columns: {missing_contract}")
    missing_features = [c for c in feature_columns if c not in feats_df.columns]
    if missing_features:
        raise ValueError(f"feats_df is missing feature columns: {missing_features}")

    df = feats_df.copy()
    # Lag-1 columns for persistence references (the baseline set already has them).
    if "y_true_lag1" not in df.columns or "y_det_lag1" not in df.columns:
        df = df.sort_values(["county", "week"]).reset_index(drop=True)
    if "y_true_lag1" not in df.columns:
        df["y_true_lag1"] = df.groupby("county", observed=True)["y_true"].shift(1).fillna(0.0)
    if "y_det_lag1" not in df.columns:
        df["y_det_lag1"] = df.groupby("county", observed=True)["y_det"].shift(1).fillna(0.0)

    if "complete" in df.columns:
        df = df[df["complete"].astype(bool)].drop(columns=["complete"])
    df = df.dropna(subset=list(feature_columns) + ["y_true", "y_det"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows left after cleaning the feature set.")
    if not df[list(feature_columns)].notna().all().all():
        raise ValueError("NaN values remain in feature columns after cleaning.")
    if not pd.api.types.is_datetime64_any_dtype(df["week"]):
        df["week"] = pd.to_datetime(df["week"])
    return df, list(feature_columns)


def _best_iteration(model: object) -> int:
    """Effective tree count of a fitted sklearn-API LightGBM model.

    LightGBM 4.x semantics (verified against 4.7.0): `best_iteration_` is > 0
    only when early stopping fired; otherwise it stays 0 and the booster simply
    holds all `n_estimators` trees. Either way the value below is the number of
    trees a predict() call actually uses, which is what the cap audit checks.
    """
    best = int(getattr(model, "best_iteration_", 0) or 0)
    if best <= 0:
        booster = getattr(model, "booster_", None)
        best = int(booster.current_iteration()) if booster is not None else -1
    return best


def _fit_and_score_fold(
    fold: dict,
    feature_columns: list[str],
    det_params: dict[str, object] | None,
    quant_params: dict[str, object] | None,
) -> dict[str, float]:
    """Fit both heads on one fold and score the validation year (plan.md §5, §7).

    Climatology references are estimated strictly inside the fold's training
    window. A head with params=None is skipped (single-metric objectives).
    Returns climatology-normalized ratios, raw metrics, and per-fit
    best_iteration values (for the early-stopping audit).
    """
    tr_df, val_df = fold["train"], fold["val"]
    fold_df = pd.concat([tr_df, val_df])
    train_mask = fold_df.index.isin(tr_df.index)

    det_clim = baselines.climatology_detection_baseline(fold_df, train_mask).loc[val_df.index].to_numpy(dtype=float)
    area_clim = baselines.climatology_area_baseline(fold_df, train_mask).loc[val_df.index].to_numpy(dtype=float)

    X_tr, X_val = tr_df[feature_columns], val_df[feature_columns]
    y_det_val = val_df["y_det"].to_numpy(dtype=float)
    y_area_val = val_df["y_true"].to_numpy(dtype=float)

    out: dict[str, float] = {}
    if det_params is not None:
        clf = lgbm_model.train_detection_classifier(X_tr, tr_df["y_det"], X_val, val_df["y_det"], params=det_params)
        det_prob = clf.predict_proba(X_val)[:, 1]
        brier_model = metrics.brier_score(y_det_val, det_prob)
        brier_clim = metrics.brier_score(y_det_val, det_clim)
        out["r_det"] = brier_model / max(brier_clim, _EPS)
        out["brier_model"] = brier_model
        out["brier_clim"] = brier_clim
        out["best_iter_det"] = _best_iteration(clf)

    if quant_params is not None:
        regs = lgbm_model.train_quantile_regressors(X_tr, tr_df["y_true"], X_val, val_df["y_true"], params=quant_params)
        # Same monotonicity cleanup as lgbm_model.predict_lightgbm.
        q10 = np.clip(regs["q10"].predict(X_val), 0.0, None)
        q50 = np.clip(regs["q50"].predict(X_val), 0.0, None)
        q90 = np.clip(regs["q90"].predict(X_val), 0.0, None)
        pinball_model = metrics.mean_pinball_loss(y_area_val, np.minimum(q10, q50), q50, np.maximum(q90, q50))
        pinball_clim = metrics.mean_pinball_loss(y_area_val, area_clim, area_clim, area_clim)
        out["r_q"] = pinball_model / max(pinball_clim, _EPS)
        out["pinball_model"] = pinball_model
        out["pinball_clim"] = pinball_clim
        for key in ("q10", "q50", "q90"):
            out[f"best_iter_{key}"] = _best_iteration(regs[key])
    return out


def _make_objective(
    folds: list[dict],
    feature_columns: list[str],
    objective_name: str,
    seed: int,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective closure (plan.md §5, §7)."""

    def objective(trial: optuna.Trial) -> float:
        det_searched: dict[str, float] | None = None
        quant_searched: dict[str, float] | None = None
        if objective_name in ("composite", "brier"):
            det_searched = _sample_head_params(trial, "det")
        if objective_name in ("composite", "pinball"):
            quant_searched = _sample_head_params(trial, "quant")

        det_params = _full_head_params(det_searched, seed)
        quant_params = _full_head_params(quant_searched, seed)

        scores: list[float] = []
        for step, fold in enumerate(folds):
            res = _fit_and_score_fold(fold, feature_columns, det_params, quant_params)

            if objective_name == "brier":
                score = float(res["r_det"])
            elif objective_name == "pinball":
                score = float(res["r_q"])
            else:
                score = 0.5 * float(res["r_det"]) + 0.5 * float(res["r_q"])
            scores.append(score)

            # Per-fold user-attrs: r_det_f<f>, r_q_f<f>, brier_*_f<f>,
            # pinball_*_f<f>, best_iter_*_f<f> (plan.md §7, §9).
            for key, value in res.items():
                trial.set_user_attr(f"{key}_f{fold['fold']}", value)
            trial.set_user_attr(f"score_f{fold['fold']}", score)

            trial.report(score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned by MedianPruner after fold {fold['fold']}")
        return float(np.mean(scores))

    return objective


def _get_or_create_study(
    study_name: str,
    storage: str,
    objective_name: str,
    feature_set: str,
    seed: int,
) -> optuna.Study:
    """Create or resume the RDB study; the objective is part of a study's identity."""
    # SQLite cannot create a database file inside a missing directory — make sure
    # the parent exists before optuna touches the storage (default: TUNING_DIR).
    if storage.startswith("sqlite:///"):
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_min_trials=1),
        storage=storage,
        load_if_exists=True,
    )
    existing = study.user_attrs.get("objective")
    if existing is None:
        study.set_user_attr("objective", objective_name)
        study.set_user_attr("feature_set", feature_set)
        study.set_user_attr("seed", seed)
        study.set_user_attr("optuna_version", optuna.__version__)
    elif existing != objective_name:
        raise ValueError(
            f"Study '{study_name}' was created with objective='{existing}' (the objective is part "
            f"of a study's identity); use a different --study-name or delete the old study."
        )
    return study


def _write_trials_csv(study: optuna.Study, study_name: str) -> Path:
    """Persist every trial (params + user-attrs) via optuna.trials_dataframe (plan.md §9)."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    out = TUNING_DIR / f"optuna_trials_{study_name}.csv"
    study.trials_dataframe().to_csv(out, index=False)
    return out


def _write_best_params_json(
    study: optuna.Study,
    study_name: str,
    feature_set: str,
    objective_name: str,
    seed: int,
    best_trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
) -> Path:
    """Dump the winning parameters (searched + fixed) and per-fold scores (plan.md §9)."""
    det_searched = {k.removeprefix("det_"): v for k, v in best_trial.params.items() if k.startswith("det_")}
    quant_searched = {k.removeprefix("quant_"): v for k, v in best_trial.params.items() if k.startswith("quant_")}
    per_fold: dict[str, dict[str, object]] = {}
    for f_num in (1, 2, 3):
        fold_attrs = {k: v for k, v in best_trial.user_attrs.items() if k.endswith(f"_f{f_num}")}
        if fold_attrs:
            per_fold[f"fold_{f_num}"] = {k.removesuffix(f"_f{f_num}"): v for k, v in fold_attrs.items()}

    payload = {
        "study_name": study_name,
        "feature_set": feature_set,
        "objective": objective_name,
        "n_trials": len(study.trials),
        "best_value": best_trial.value,
        "best_params": {"det": det_searched or None, "quantiles": quant_searched or None},
        "lightgbm_params": {
            "det": _full_head_params(det_searched, seed),
            "quantiles": _full_head_params(quant_searched, seed),
        },
        "per_fold": per_fold,
        "seed": seed,
        "optuna_version": optuna.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    out = TUNING_DIR / f"best_params_{study_name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def tune_lightgbm(
    feats_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    feature_set: str = "baseline",
    study_name: str | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    objective: str = "composite",
    storage: str | None = None,
    timeout: int | None = DEFAULT_TIMEOUT,
    n_jobs: int = 1,
    seed: int = config.SEED,
    evaluate: bool = False,
) -> optuna.Study:
    """Run one Optuna study for one feature set (plan.md §5).

    Either pass `feats_df` + `feature_columns` directly, or pick a registered
    `feature_set` (the registry is used only when feats_df is None). The study
    is stored in an SQLite RDB and can be resumed by re-running with the same
    study name; `--evaluate` then scores the winner on the frozen test years.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got: {objective}")

    t0 = time.perf_counter()
    if feats_df is None:
        feats_df, feature_columns = _resolve_feature_set(feature_set)
    elif feature_columns is None:
        raise ValueError("feature_columns is required when feats_df is given directly.")
    feats_df, feature_columns = _prepare_features(feats_df, feature_columns)
    print(
        f"  Feature matrix ready: {len(feats_df)} rows x {len(feature_columns)} features "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    folds = splits.get_cv_folds(feats_df)
    study_name = study_name or _default_study_name(feature_set)
    storage = storage or DEFAULT_STORAGE
    study = _get_or_create_study(study_name, storage, objective, feature_set, seed)

    print("=" * 100)
    print(f"OPTUNA TUNING — study '{study_name}' | objective={objective} | feature set '{feature_set}'")
    print(
        f"  {len(folds)} folds (val years 2009/2015/2021), {len(feats_df)} usable rows, "
        f"{len(feature_columns)} features, 14 sampled params/trial"
    )
    print(f"  storage: {storage}")
    print("=" * 100)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(
        _make_objective(folds, feature_columns, objective, seed),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        catch=(Exception,),
        gc_after_trial=True,
    )

    trials_csv = _write_trials_csv(study, study_name)
    print(f"\n  All trials CSV -> {trials_csv}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    print(f"  Trials: {len(completed)} complete, {pruned} pruned, {failed} failed")

    if not completed:
        print("  WARNING: no completed trials (all pruned/failed); skipping best-params JSON.")
        return study

    best_trial = study.best_trial
    best_json = _write_best_params_json(study, study_name, feature_set, objective, seed, best_trial)

    print("=" * 100)
    print(f"BEST TRIAL #{best_trial.number} — {objective} score {best_trial.value:.4f} "
          "(1.0 = climatology parity, lower is better)")
    for f_num in (1, 2, 3):
        parts = [f"fold {f_num}:"]
        if f"r_det_f{f_num}" in best_trial.user_attrs:
            parts.append(f"r_det={best_trial.user_attrs[f'r_det_f{f_num}']:.3f}")
        if f"r_q_f{f_num}" in best_trial.user_attrs:
            parts.append(f"r_q={best_trial.user_attrs[f'r_q_f{f_num}']:.3f}")
        if f"score_f{f_num}" in best_trial.user_attrs:
            parts.append(f"score={best_trial.user_attrs[f'score_f{f_num}']:.3f}")
        print("   ", "  ".join(parts))
    print(f"  Best params JSON -> {best_json}")
    print("=" * 100)

    if evaluate:
        eval_table = evaluate_best_params(
            study, feats_df=feats_df, feature_columns=feature_columns, feature_set=feature_set, seed=seed
        )
        pooled = eval_table[(eval_table["scope"] == "pooled") & (eval_table["model"] == "lightgbm")].iloc[0]
        print(
            f"  Pooled tuned: Brier {pooled['brier_score']:.4f}, "
            f"BSS clim {pooled['bss_vs_climatology']:.3f}, BSS pers {pooled['bss_vs_persistence']:.3f}, "
            f"pinball {pooled['mean_pinball_loss']:.3f}, "
            f"coverage(flood) {pooled['coverage_flood_weeks']:.3f}, "
            f"MAE(flood) {pooled['mae_flood_weeks']:.2f} km2"
        )
    return study


def evaluate_best_params(
    study: optuna.Study,
    feats_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    feature_set: str | None = None,
    seed: int = config.SEED,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Refit the winning parameter groups and score the frozen TEST years (plan.md §10).

    Reproduces the run.py fold loop (minus the duration/advisory steps): per
    fold, fit on train, early-stop on val, predict the test year once, then
    score with metrics.evaluate_summary_metrics. The untuned head of a
    single-metric study keeps today's hardcoded defaults (params=None).
    """
    det_searched = {k.removeprefix("det_"): v for k, v in study.best_params.items() if k.startswith("det_")}
    quant_searched = {k.removeprefix("quant_"): v for k, v in study.best_params.items() if k.startswith("quant_")}
    det_params = _full_head_params(det_searched, seed)
    quant_params = _full_head_params(quant_searched, seed)

    t0 = time.perf_counter()
    if feats_df is None:
        feature_set = feature_set or "baseline"
        feats_df, feature_columns = _resolve_feature_set(feature_set)
    elif feature_columns is None:
        raise ValueError("feature_columns is required when feats_df is given directly.")
    feats_df, feature_columns = _prepare_features(feats_df, feature_columns)

    folds = splits.get_cv_folds(feats_df)
    oof_rows = []
    for fold in folds:
        tr_df, val_df, tst_df = fold["train"], fold["val"], fold["test"]
        fold_df = pd.concat([tr_df, val_df, tst_df])
        train_mask = fold_df.index.isin(tr_df.index)
        fold_df = baselines.add_baselines_to_dataframe(fold_df, train_mask)
        tst_df = fold_df.loc[tst_df.index]

        X_tr, X_val, X_tst = tr_df[feature_columns], val_df[feature_columns], tst_df[feature_columns]
        clf = lgbm_model.train_detection_classifier(X_tr, tr_df["y_det"], X_val, val_df["y_det"], params=det_params)
        regs = lgbm_model.train_quantile_regressors(X_tr, tr_df["y_true"], X_val, val_df["y_true"], params=quant_params)

        q50 = np.clip(regs["q50"].predict(X_tst), 0.0, None)
        res_df = tst_df[["county", "week", "y_true", "y_det", "det_prob_clim", "area_clim_q50",
                         "det_prob_persist", "area_persist"]].copy()
        res_df["test_year"] = res_df["week"].dt.year
        res_df["fold"] = fold["fold"]
        res_df["det_prob"] = clf.predict_proba(X_tst)[:, 1]
        res_df["q10"] = np.minimum(np.clip(regs["q10"].predict(X_tst), 0.0, None), q50)
        res_df["q50"] = q50
        res_df["q90"] = np.maximum(np.clip(regs["q90"].predict(X_tst), 0.0, None), q50)
        oof_rows.append(res_df)

    all_oof_df = pd.concat(oof_rows, ignore_index=True)
    eval_table = metrics.evaluate_summary_metrics(all_oof_df)

    if out_csv is None:
        out_csv = TUNING_DIR / f"cv_metrics_tuned_{study.study_name}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_table.to_csv(out_csv, index=False)

    print("=" * 100)
    print(f"TEST-YEAR EVALUATION of best params from study '{study.study_name}' "
          f"(untuned head falls back to defaults)")
    print(f"  Fold loop + scoring took {time.perf_counter() - t0:.1f}s | metrics table -> {out_csv}")
    print(eval_table[_SUMMARY_COLS].to_string(index=False))
    print("=" * 100)
    return eval_table


def main() -> None:
    """CLI entry point (plan.md §12)."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for the LightGBM_v1 pipeline (plan.md).",
    )
    parser.add_argument("--feature-set", default="baseline", choices=sorted(FEATURE_SETS),
                        help="Registered feature set to tune (default: baseline).")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                        help=f"Max trials (default: {DEFAULT_N_TRIALS}).")
    parser.add_argument("--evaluate", action="store_true",
                        help="After tuning, refit the winner and score the frozen test years (§10).")
    parser.add_argument("--objective", choices=OBJECTIVES, default="composite",
                        help="Objective (default: composite; part of a study's identity).")
    parser.add_argument("--study-name", default=None,
                        help="Study name (default: lgbm_aweil_<feature-set>_<version>).")
    parser.add_argument("--storage", default=None,
                        help=f"Optuna storage URL (default: {DEFAULT_STORAGE}).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Study timeout in seconds (default: {DEFAULT_TIMEOUT}).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel trials (default: 1; LightGBM already uses all cores).")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"Random seed (default: {config.SEED}).")
    parser.add_argument("--list-feature-sets", action="store_true",
                        help="List registered feature sets and exit.")
    args = parser.parse_args()

    if args.list_feature_sets:
        print("Registered feature sets (LightGBM_v1.tuning.FEATURE_SETS):")
        for name in FEATURE_SETS:
            print(f"  {name}")
        return

    tune_lightgbm(
        feature_set=args.feature_set,
        study_name=args.study_name,
        n_trials=args.n_trials,
        objective=args.objective,
        storage=args.storage,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        seed=args.seed,
        evaluate=args.evaluate,
    )


if __name__ == "__main__":
    main()
