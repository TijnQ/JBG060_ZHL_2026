"""Shared LightGBM/TabPFN runner (baseline: 3 expanding-window CV folds).

Model choice (--model):
- lightgbm           — the baseline model, saved to outputs/outputs_v3/tables/
- lightgbm_baseline  — the same frozen model, flagged as the project baseline
                       and saved to outputs/outputs_baseline/tables/ so the later
                       improved LightGBM model stays clearly separated
- lightgbm_fe        — feature-engineered counterpart of the baseline
                       (features_fe.py: + county, + cnty_tp_w7, + cnty_ro_w7,
                       - year from the model input), same baseline hyperparameters,
                       saved to outputs/outputs_fe/tables/
- tabpfn             — zero-shot TabPFN reference, saved to
                       TabPFN_v1/outputs/outputs_v3/tables/

Hyperparameter set (--params, LightGBM models only; default "baseline"):
- baseline        — the frozen baseline defaults (run.py behavior unchanged)
- baseline_optuna — Optuna winners tuned on the baseline features
- fe_baseline     — the same baseline defaults, named for the fe grid cell
- fe_optuna       — Optuna winners tuned on the engineered features
Optuna sets route the run to their own output folder
(e.g. lightgbm_fe --params fe_optuna -> outputs/outputs_fe_optuna/tables/)
so the frozen outputs_v3/outputs_baseline/outputs_fe tables stay untouched.

Run mode (--mode, default "cv"):
- cv     — the 3-fold expanding-window CV protocol described above (main)
- final  — retrain the chosen configuration once on all data (train
           2000-2024, early-stop on 2025) and forecast the latest
           feature-complete week: det_prob + q10/q50/q90 plus duration
           horizons h1/h2 -> outputs/outputs_final_fit/tables/
           final_forecast_aweil.csv (final_fit)

Performs:
1.  Builds the embargoed weekly feature matrix (2000-2025; 2025 stays outside CV)
2.  3-Fold Expanding-Window Cross-Validation (config.CV_FOLDS; 2025 excluded)
3.  Per-fold honest references: climatology & persistence baselines estimated
    strictly inside each fold's training window
4.  Protocol-aligned metrics (evaluation_metrics.md): Brier / BSS, mean pinball,
    interval coverage, flood-week MAE/bias, per-horizon duration Brier,
    reliability diagrams (no cutoff-dependent classification metrics)
5.  Data provenance report (one row per real source, console only)
6.  Outputs saved to the selected model's output folder; LightGBM runs also
    report pooled (mean-over-folds) split & gain feature importance
    (feature_importance_pooled.csv)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LightGBM_v1 import (
    baselines,
    config,
    data_loader,
    duration_model,
    features,
    features_fe,
    lgbm_model,
    metrics,
    splits,
)

_SUMMARY_COLS = [
    "scope", "model", "n_samples", "flood_weeks", "flood_rate_pct",
    "brier_score", "bss_vs_climatology", "bss_vs_persistence",
    "mean_pinball_loss", "coverage_flood_weeks", "coverage_all_weeks",
    "mae_flood_weeks", "bias_flood_weeks",
]
_BREAKDOWN_COLS = [
    "n_samples", "flood_weeks", "flood_rate_pct",
    "brier_score", "bss_vs_climatology", "bss_vs_persistence",
    "mean_pinball_loss", "coverage_flood_weeks", "coverage_all_weeks",
    "mae_flood_weeks", "bias_flood_weeks",
]


def _resolve_backend(model_name: str, params_name: str):
    """Resolve model functions, feature build, output folder and params.

    Single source of truth shared by the CV run (main) and the final-fit run
    (final_fit), so both always select the identical backend wiring.
    """
    if model_name not in {"lightgbm", "lightgbm_baseline", "lightgbm_fe", "tabpfn"}:
        raise ValueError(f"Unknown model: {model_name}")
    # Named hyperparameter set (lgbm_model.PARAM_SETS). LightGBM models thread
    # its per-head entries into training; TabPFN has no tunable params and
    # ignores it (validated here anyway so a typo fails fast).
    param_set = lgbm_model.get_param_set(params_name)
    # Use LightGBM by default. The rest of the steps stay the same for all models.
    train_forecast = lgbm_model.build_and_train_lightgbm
    predict_forecast = lgbm_model.predict_lightgbm
    duration_backend = duration_model
    tables_dir = config.TABLES_DIR
    # Feature matrix + column list: the baseline choices share the frozen build;
    # the engineered choice swaps both (plan.md §6) without touching the baseline path.
    build_features = features.build_weekly_features
    feature_columns = features.FEATURE_COLUMNS
    # Feature importance needs the fitted LightGBM boosters; TabPFN has none.
    collect_importance = True
    if model_name == "tabpfn":
        from LightGBM_v1 import tabpfn_model
        # Switch the model functions and save TabPFN results in its own folder.
        train_forecast = tabpfn_model.train_forecast
        predict_forecast = tabpfn_model.predict_forecast
        duration_backend = tabpfn_model
        tables_dir = PROJECT_ROOT / "TabPFN_v1" / "outputs" / f"outputs_{config.MODEL_VERSION}" / "tables"
        collect_importance = False
    elif model_name == "lightgbm_baseline":
        # The frozen model is the project baseline: identical model code and
        # default hyperparameters, but the outputs land in outputs_baseline/ so
        # the baseline stays clearly separated from the improved LightGBM model.
        tables_dir = config.BASELINE_TABLES_DIR
    elif model_name == "lightgbm_fe":
        # Feature-engineered counterpart of the frozen baseline (plan.md §3.1):
        # same LightGBM backend and baseline default hyperparameters, but the
        # engineered feature set (features_fe.py) and its own output folder so
        # the run stays table-by-table diffable against outputs_baseline/.
        tables_dir = config.FE_TABLES_DIR
        build_features = features_fe.build_weekly_features_fe
        feature_columns = features_fe.FEATURE_COLUMNS_ENGINEERED

    # Optuna-tuned param sets never overwrite the frozen default-set folders:
    # route the run to its own tables dir, deduplicating the feature-set word
    # (lightgbm_baseline --params baseline_optuna -> outputs_baseline_optuna;
    # cross combos stay unambiguous, e.g. -> outputs_fe_baseline_optuna).
    if params_name.endswith("_optuna") and model_name != "tabpfn":
        stem = tables_dir.parent.name.removeprefix("outputs_")
        suffix = params_name if params_name.startswith(f"{stem}_") else f"{stem}_{params_name}"
        tables_dir = config.OUT_ROOT / f"outputs_{suffix}" / "tables"

    # Per-head LightGBM kwargs from the named set (None = the frozen baseline
    # defaults); TabPFN has no tunable hyperparameters, so all three stay None
    # there and the historic kwarg-free call shape is preserved.
    det_params = param_set["det"] if model_name != "tabpfn" else None
    quant_params = param_set["quantiles"] if model_name != "tabpfn" else None
    dur_params = param_set["duration"] if model_name != "tabpfn" else None

    return (train_forecast, predict_forecast, duration_backend, build_features,
            feature_columns, tables_dir, collect_importance,
            det_params, quant_params, dur_params)


def _complete_feature_frame(build_features) -> pd.DataFrame:
    """Build the weekly feature matrix and drop incomplete county-weeks.

    Shared by the CV run and the final-fit run. Incomplete rows only occur at
    the data start (rolling/lag warm-up), so the gate doubles as an honesty
    check: no row — the forecast row included — ever carries partial windows.
    """
    feats_df = build_features()
    n_incomplete = int((~feats_df["complete"]).sum())
    if n_incomplete:
        bad_years = sorted(pd.to_datetime(feats_df.loc[~feats_df["complete"], "week"]).dt.year.unique())
        print(f"      Dropped {n_incomplete} incomplete county-weeks (years: {bad_years}) — see provenance report")
    feats_df = feats_df[feats_df["complete"]].reset_index(drop=True)
    feats_df = feats_df.drop(columns=["complete"])
    return feats_df


def main(model_name: str = "lightgbm", params_name: str = "baseline") -> None:
    (train_forecast, predict_forecast, duration_backend, build_features,
     feature_columns, tables_dir, collect_importance,
     det_params, quant_params, dur_params) = _resolve_backend(model_name, params_name)

    model_label = model_name.upper()
    print("=" * 95)
    print(f"{model_label} PIPELINE RUN — VERSION {config.MODEL_VERSION.upper()} — "
          f"{len(config.CV_FOLDS)} EXPANDING-WINDOW FOLDS, PROTOCOL METRICS")
    if params_name.endswith("_optuna") and model_name != "tabpfn":
        print(f"  Hyperparameters: {params_name} (tables -> {tables_dir})")
    else:
        print(f"  Hyperparameters: {params_name}")
    if model_name == "tabpfn" and params_name != "baseline":
        print("  note: --params has no effect on the tabpfn backend")
    print("=" * 95)

    # 1. Build the input features using data from Friday or earlier.
    print("\n[1/7] Building weekly feature matrix with Friday 3-day embargo...")
    feats_df = _complete_feature_frame(build_features)
    print(f"      Total rows: {len(feats_df)}, Total features: {len(feature_columns)}")
    if model_name == "lightgbm_fe":
        added = [c for c in feature_columns if c not in features.FEATURE_COLUMNS]
        removed = [c for c in features.FEATURE_COLUMNS if c not in feature_columns]
        print(f"      Engineered set: + {added} | - {removed} (all other baseline columns identical)")

    # Full-frame duration targets (computed once; labels exist through end-2025)
    full_duration_targets = duration_model.build_duration_targets(feats_df)

    # 2. Create three time splits. Each split uses a longer training period.
    print("\n[2/7] Generating 3 expanding-window folds (2025 excluded from all folds)...")
    folds = splits.get_cv_folds(feats_df)

    oof_rows = []
    importance_rows: list[pd.DataFrame] = []

    for fold_info in folds:
        f_num = fold_info["fold"]
        tr_df = fold_info["train"]
        val_df = fold_info["val"]
        tst_df = fold_info["test"]

        print(f"      Fold {f_num}: Train {fold_info['train_years']} ({len(tr_df)} rows) | "
              f"Val {fold_info['val_year']} ({len(val_df)} rows) | "
              f"Test {fold_info['test_years']} ({len(tst_df)} rows)")

        # 3. Calculate the baselines using only this fold's training rows.
        # Keep the original row numbers so we can select the validation and
        # test rows again with .loc after adding the baseline columns.
        fold_df = pd.concat([tr_df, val_df, tst_df])
        train_mask = fold_df.index.isin(tr_df.index)
        fold_df = baselines.add_baselines_to_dataframe(fold_df, train_mask)
        val_df = fold_df.loc[val_df.index]
        tst_df = fold_df.loc[tst_df.index]

        # Fit the selected model, then predict the test rows. The per-head
        # kwargs are only forwarded when a non-default param set is active so
        # the historic params=None call (and mocks of train_forecast) keep
        # working unchanged.
        if det_params is None and quant_params is None:
            models_fold = train_forecast(
                tr_df[feature_columns], tr_df["y_true"], tr_df["y_det"],
                val_df[feature_columns], val_df["y_true"], val_df["y_det"],
            )
        else:
            models_fold = train_forecast(
                tr_df[feature_columns], tr_df["y_true"], tr_df["y_det"],
                val_df[feature_columns], val_df["y_true"], val_df["y_det"],
                det_params=det_params, quant_params=quant_params,
            )
        preds_tst = predict_forecast(models_fold, tst_df[feature_columns], feature_columns=feature_columns)
        if collect_importance:
            # Read the booster importances before this fold's models are freed.
            imp_fold = lgbm_model.extract_feature_importance(models_fold, feature_columns)
            imp_fold.insert(0, "fold", f_num)
            importance_rows.append(imp_fold)
        del models_fold

        res_df = tst_df[["county", "week", "y_true", "y_det",
                         "det_prob_clim", "area_clim_q50", "det_prob_persist", "area_persist"]].copy()
        res_df["test_year"] = res_df["week"].dt.year
        res_df["det_prob"] = preds_tst["det_prob"]
        res_df["q10"] = preds_tst["q10"]
        res_df["q50"] = preds_tst["q50"]
        res_df["q90"] = preds_tst["q90"]
        res_df["fold"] = f_num

        # Per-fold duration models (train/val inside this fold only); the
        # duration head's hyperparameters come from the same named set
        # (None = the hardcoded _DURATION_BASE_PARAMS).
        if dur_params is None:
            dur_models = duration_backend.train_duration_models(tr_df, val_df, feature_columns=feature_columns)
        else:
            dur_models = duration_backend.train_duration_models(
                tr_df, val_df, feature_columns=feature_columns, params=dur_params)
        dur_pred = duration_backend.predict_duration(dur_models, tst_df, feature_columns=feature_columns)
        del dur_models
        res_df = res_df.join(dur_pred)

        # Duration targets (from full frame) + per-fold references at target week
        target_cols = [f"y_det_h{h}" for h in config.DURATION_HORIZONS]
        res_df = res_df.merge(
            full_duration_targets.set_index(["county", "week"])[target_cols],
            left_on=["county", "week"], right_index=True, how="left",
        )
        res_df = res_df.join(metrics.duration_climatology_reference(tr_df, tst_df))
        res_df = res_df.join(metrics.duration_persistence_reference(tr_df, tst_df, feats_df))

        oof_rows.append(res_df)

    all_oof_df = pd.concat(oof_rows, ignore_index=True)

    # Per-fold feature importance pooled to a mean across folds (LightGBM only).
    importance_detail = pd.concat(importance_rows, ignore_index=True) if importance_rows else None
    importance_pooled = None
    if importance_detail is not None:
        importance_pooled = (
            importance_detail
            .groupby(["head", "importance_type", "feature"], as_index=False)
            .agg(mean_importance=("importance", "mean"), mean_share_pct=("share_pct", "mean"))
            .sort_values(["head", "importance_type", "mean_share_pct"], ascending=[True, True, False])
            .reset_index(drop=True)
        )

    # 4. Protocol metrics: per-fold + pooled summary, duration, breakdowns
    print("\n[3/7] Computing protocol metrics (Brier/BSS, pinball, coverage, flood-week MAE/bias)...")
    eval_table = metrics.evaluate_summary_metrics(all_oof_df)
    eval_table["model"] = eval_table["model"].replace({"lightgbm": model_name})
    duration_table, duration_reliability = metrics.evaluate_duration_metrics(all_oof_df)
    reliability_det = metrics.reliability_diagram_data(all_oof_df["y_det"], all_oof_df["det_prob"])
    county_table = metrics.evaluate_county_metrics(all_oof_df)
    monthly_table = metrics.evaluate_monthly_metrics(all_oof_df)

    # 5. Data provenance report (one row per real source)
    provenance_df = data_loader.get_provenance_report()

    print("\n" + "=" * 95)
    print("DATA PROVENANCE (real sources only — missing inputs raise)")
    print("=" * 95)
    print(provenance_df.to_string(index=False))
    print("-" * 95)

    print("\n" + "=" * 115)
    print("OUT-OF-FOLD PROTOCOL METRICS — 3 EXPANDING-WINDOW FOLDS + POOLED (test years: 2010-11, 2016-17, 2022-24)")
    print("=" * 115)
    print(eval_table[_SUMMARY_COLS].to_string(index=False))
    print("=" * 115)

    print("\n" + "=" * 115)
    print("DURATION METRICS PER HORIZON (Brier vs per-fold climatology & persistence at target week)")
    print("=" * 115)
    print(duration_table.to_string(index=False))
    print("=" * 115)

    print("\n" + "=" * 115)
    print(f"COUNTY BREAKDOWN ({model_label}, OOF TEST WEEKS)")
    print("=" * 115)
    print(county_table[["county"] + _BREAKDOWN_COLS].to_string(index=False))
    print("=" * 115)

    print("\n" + "=" * 115)
    print(f"MONTHLY BREAKDOWN ({model_label}, OOF TEST WEEKS)")
    print("=" * 115)
    print(monthly_table[["month_num", "month_name", "season"] + _BREAKDOWN_COLS].to_string(index=False))
    print("=" * 115)

    if importance_pooled is not None:
        gain_rows = importance_pooled[importance_pooled["importance_type"] == "gain"]
        head_labels = {
            "det": "detection classifier",
            "q10": "q10 quantile regressor",
            "q50": "q50 quantile regressor",
            "q90": "q90 quantile regressor",
        }
        print("\n" + "=" * 115)
        print(f"FEATURE IMPORTANCE ({model_label}, mean over {len(config.CV_FOLDS)} folds — "
              f"top 10 per model head by gain share)")
        print("=" * 115)
        for head in ("det", "q10", "q50", "q90"):
            head_rows = gain_rows[gain_rows["head"] == head]
            if head_rows.empty:
                continue
            print(f"  {head} ({head_labels[head]}):")
            for _, row in head_rows.nlargest(10, "mean_share_pct").iterrows():
                print(f"      {row['feature']:<16} mean gain {row['mean_importance']:>12.1f}   "
                      f"share {row['mean_share_pct']:>5.1f}%")
        print("=" * 115)

    # 6. Save Output Tables to the selected model's output folder
    print("\n[4/7] Saving output tables...")
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "cv_test_predictions_aweil.csv": all_oof_df,
        "cv_metrics_summary_aweil.csv": eval_table,
        "duration_metrics_aweil.csv": duration_table,
        "reliability_det_prob.csv": reliability_det,
        "county_metrics_aweil.csv": county_table,
        "monthly_metrics_aweil.csv": monthly_table,
    }
    if importance_pooled is not None:
        out_files["feature_importance_pooled.csv"] = importance_pooled
    for name, frame in out_files.items():
        frame.to_csv(tables_dir / name, index=False)
    for h_name, rel_df in duration_reliability.items():
        rel_df.to_csv(tables_dir / f"reliability_duration_{h_name}.csv", index=False)

    print(f"      -> {tables_dir}")
    for name in list(out_files) + [f"reliability_duration_{k}.csv" for k in duration_reliability]:
        print(f"         {name}")

    # 7. Final summary
    pooled_lgbm = eval_table[(eval_table["scope"] == "pooled") & (eval_table["model"] == model_name)].iloc[0]
    print("\n" + "=" * 95)
    print("PIPELINE COMPLETE — BASELINE SUMMARY")
    print("=" * 95)
    print(f"  OOF test weeks scored: {len(all_oof_df)} "
          f"({all_oof_df['test_year'].min()}-{all_oof_df['test_year'].max()}, 2025 excluded)")
    print(f"  Pooled {model_label}: Brier {pooled_lgbm['brier_score']:.4f}, "
          f"BSS vs clim {pooled_lgbm['bss_vs_climatology']:.3f}, "
          f"BSS vs pers {pooled_lgbm['bss_vs_persistence']:.3f}, "
          f"mean pinball {pooled_lgbm['mean_pinball_loss']:.3f}, "
          f"coverage(flood) {pooled_lgbm['coverage_flood_weeks']:.3f}, "
          f"MAE(flood) {pooled_lgbm['mae_flood_weeks']:.2f} km2")
    n_real = int((provenance_df['provenance'] == 'real').sum())
    print(f"  Provenance: {n_real} real sources (missing inputs raise; no synthetic fallback)")
    print("=" * 95)


def final_fit(model_name: str = "lightgbm_fe", params_name: str = "fe_baseline") -> None:
    """Final-fit mode: retrain the chosen configuration on all data and issue
    the operational forecast (run.py --mode final).

    Instead of the 3-fold CV the model is fit once — the expanding-window
    protocol taken one fold further: train on everything before the final year
    (2000-2024), early-stopping on the final year (2025). It then forecasts
    the latest feature-complete week (2025-12-29 with inputs through Dec 2025):
    det_prob + q10/q50/q90 for that week plus the duration heads
    det_prob_h1/h2 (the honest forecast of the following two weeks; h3/h4 lose
    to climatology and persistence in duration_metrics_aweil.csv). Later weeks
    are not forecast: they would need weather beyond the inputs that exist,
    and partial rolling windows would silently change what the features mean.
    The output feeds climatology_maps.forecast_maps --predictions.
    """
    if model_name == "tabpfn":
        raise ValueError("final_fit supports the LightGBM models only (tabpfn was a one-shot reference)")
    (train_forecast, predict_forecast, duration_backend, build_features,
     feature_columns, _tables_dir, _collect_importance,
     det_params, quant_params, dur_params) = _resolve_backend(model_name, params_name)

    print("=" * 95)
    print(f"{model_name.upper()} FINAL FIT — HYPERPARAMETERS: {params_name}")
    print("=" * 95)

    # 1. Feature matrix; the completeness gate also guarantees the forecast
    # row carries no partial rolling windows.
    print("\n[1/4] Building weekly feature matrix with Friday 3-day embargo...")
    feats_df = _complete_feature_frame(build_features)
    print(f"      Total rows: {len(feats_df)}, Total features: {len(feature_columns)}")

    # 2. Train on everything before the final year, early-stop on the final
    # year (first week purged — same boundary rule as the CV folds).
    train_df, val_df, final_year = splits.final_fit_split(feats_df)
    first_year = feats_df["week"].min().year
    print(f"\n[2/4] Final fit: train {first_year}-{final_year - 1} ({len(train_df)} rows) | "
          f"early-stop year {final_year} ({len(val_df)} rows)")

    # 3. Fit the four heads + duration classifiers (same params-None call
    # shapes as main, so the historic call signatures stay unchanged).
    print("\n[3/4] Training detection/quantile heads + duration classifiers...")
    if det_params is None and quant_params is None:
        models = train_forecast(
            train_df[feature_columns], train_df["y_true"], train_df["y_det"],
            val_df[feature_columns], val_df["y_true"], val_df["y_det"],
        )
    else:
        models = train_forecast(
            train_df[feature_columns], train_df["y_true"], train_df["y_det"],
            val_df[feature_columns], val_df["y_true"], val_df["y_det"],
            det_params=det_params, quant_params=quant_params,
        )
    if dur_params is None:
        dur_models = duration_backend.train_duration_models(train_df, val_df, feature_columns=feature_columns)
    else:
        dur_models = duration_backend.train_duration_models(
            train_df, val_df, feature_columns=feature_columns, params=dur_params)

    # 4. Forecast the latest feature-complete week (last Monday of the grid).
    forecast_week = feats_df["week"].max()
    forecast_df = feats_df[feats_df["week"] == forecast_week]
    preds = predict_forecast(models, forecast_df[feature_columns], feature_columns=feature_columns)
    dur_pred = duration_backend.predict_duration(dur_models, forecast_df, feature_columns=feature_columns)

    out = forecast_df[["county", "week"]].copy()
    out["det_prob"] = preds["det_prob"]
    out["q10"] = preds["q10"]
    out["q50"] = preds["q50"]
    out["q90"] = preds["q90"]
    # Duration horizons h1/h2 only (see docstring).
    out["det_prob_h1"] = dur_pred["det_prob_h1"]
    out["det_prob_h2"] = dur_pred["det_prob_h2"]

    out_path = config.FINAL_FIT_TABLES_DIR / "final_forecast_aweil.csv"
    config.FINAL_FIT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # 5. Provenance report + compact summary
    provenance_df = data_loader.get_provenance_report()
    print("\n" + "=" * 95)
    print("DATA PROVENANCE (real sources only — missing inputs raise)")
    print("=" * 95)
    print(provenance_df.to_string(index=False))
    print("-" * 95)

    print("\n" + "=" * 95)
    print("FINAL FIT COMPLETE")
    print("=" * 95)
    print(f"  Forecast week: {forecast_week.date()} (latest feature-complete week —")
    print("  later weeks would need weather data beyond Dec 2025; see main.ipynb section 9)")
    print(f"  Trained {first_year}-{final_year - 1}, early stopping on {final_year} (no post-2025 data exists)")
    print(out.to_string(index=False))
    print(f"  -> {out_path}")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("lightgbm", "lightgbm_baseline", "lightgbm_fe", "tabpfn"),
        default="lightgbm",
        help="lightgbm: baseline model -> outputs_v3 | "
             "lightgbm_baseline: frozen baseline -> outputs_baseline | "
             "lightgbm_fe: engineered features (+county, +cnty_tp_w7, "
             "+cnty_ro_w7, -year) -> outputs_fe | "
             "tabpfn: TabPFN reference",
    )
    parser.add_argument(
        "--params",
        choices=tuple(lgbm_model.PARAM_SETS),
        default="baseline",
        help="hyperparameter set (LightGBM models only): "
             "baseline: frozen baseline defaults (unchanged behavior) | "
             "baseline_optuna: Optuna winners tuned on the baseline features | "
             "fe_baseline: the same baseline defaults (fe grid cell) | "
             "fe_optuna: Optuna winners tuned on the engineered features. "
             "Optuna sets save to their own outputs_<model>_<params>/tables "
             "folder instead of the frozen per-model folders.",
    )
    parser.add_argument(
        "--mode",
        choices=("cv", "final"),
        default="cv",
        help="cv: 3-fold expanding-window CV protocol (default) | "
             "final: retrain once (train to the year before the last, "
             "early-stop on the final year) and forecast the latest "
             "feature-complete week -> outputs_final_fit/tables/"
             "final_forecast_aweil.csv",
    )
    args = parser.parse_args()
    if args.mode == "final":
        final_fit(args.model, args.params)
    else:
        main(args.model, args.params)
