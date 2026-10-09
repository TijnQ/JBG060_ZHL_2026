"""Shared LightGBM/TabPFN runner (v3: 3 expanding-window CV folds).

Model choice (--model):
- lightgbm           — the current v3 model, saved to outputs/outputs_v3/tables/
- lightgbm_baseline  — the same frozen v3 model, flagged as the project baseline
                       and saved to outputs/outputs_baseline/tables/ so the later
                       improved LightGBM model stays clearly separated
- lightgbm_fe        — feature-engineered counterpart of the baseline
                       (features_fe.py: + county, + cnty_tp_w7, + cnty_ro_w7,
                       - year from the model input), same v3 hyperparameters,
                       saved to outputs/outputs_fe/tables/ with a write-only
                       feature-matrix audit snapshot in outputs/outputs_fe/features/
- tabpfn             — zero-shot TabPFN reference, saved to
                       TabPFN_v1/outputs/outputs_v3/tables/

Hyperparameter set (--params, LightGBM models only; default "baseline"):
- baseline        — the frozen v3 defaults (run.py behavior unchanged)
- baseline_optuna — Optuna winners tuned on the baseline features
- fe_baseline     — the same v3 defaults, named for the fe grid cell
- fe_optuna       — Optuna winners tuned on the engineered features
Optuna sets route the run to their own output folder
(e.g. lightgbm_fe --params fe_optuna -> outputs/outputs_fe_optuna/tables/)
so the frozen outputs_v3/outputs_baseline/outputs_fe tables stay untouched.

Performs:
1.  Builds the embargoed weekly feature matrix (2000-2025; 2025 stays outside CV)
2.  3-Fold Expanding-Window Cross-Validation (config.CV_FOLDS; 2025 excluded)
3.  Per-fold honest references: climatology & persistence baselines and advisory
    tier thresholds estimated strictly inside each fold's training window
4.  Protocol-aligned metrics (evaluation_metrics.md): Brier / BSS, mean pinball,
    interval coverage, flood-week MAE/bias, per-horizon duration Brier,
    reliability diagrams (no cutoff-dependent classification metrics)
5.  Stakeholder advisories & cattle movement advice for every fold test week
6.  Data provenance report (real vs synthetic_fallback per source)
7.  Outputs saved to the selected model's output folder; LightGBM runs also
    report split & gain feature importance per fold
    (feature_importance.csv + feature_importance_pooled.csv)
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
    advisory,
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


def main(model_name: str = "lightgbm", params_name: str = "baseline") -> None:
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
    # Feature matrix + column list: the v3 choices share the frozen v3 build;
    # the engineered choice swaps both (plan.md §6) without touching the v3 path.
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
        # The frozen v3 model is the project baseline: identical model code and
        # default hyperparameters, but the outputs land in outputs_baseline/ so
        # the baseline stays clearly separated from the improved LightGBM model.
        tables_dir = config.BASELINE_TABLES_DIR
    elif model_name == "lightgbm_fe":
        # Feature-engineered counterpart of the frozen baseline (plan.md §3.1):
        # same LightGBM backend and v3 default hyperparameters, but the
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

    # Per-head LightGBM kwargs from the named set (None = the frozen v3
    # defaults); TabPFN has no tunable hyperparameters, so all three stay None
    # there and the historic kwarg-free call shape is preserved.
    det_params = param_set["det"] if model_name != "tabpfn" else None
    quant_params = param_set["quantiles"] if model_name != "tabpfn" else None
    dur_params = param_set["duration"] if model_name != "tabpfn" else None

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
    feats_df = build_features()
    n_incomplete = int((~feats_df["complete"]).sum())
    if n_incomplete:
        bad_years = sorted(pd.to_datetime(feats_df.loc[~feats_df["complete"], "week"]).dt.year.unique())
        print(f"      Dropped {n_incomplete} incomplete county-weeks (years: {bad_years}) — see provenance report")
    feats_df = feats_df[feats_df["complete"]].reset_index(drop=True)
    feats_df = feats_df.drop(columns=["complete"])
    print(f"      Total rows: {len(feats_df)}, Total features: {len(feature_columns)}")
    if model_name == "lightgbm_fe":
        added = [c for c in feature_columns if c not in features.FEATURE_COLUMNS]
        removed = [c for c in features.FEATURE_COLUMNS if c not in feature_columns]
        print(f"      Engineered set: + {added} | - {removed} (all other v3 columns identical)")
        # Write-only audit snapshot (plan.md §6): dump the exact matrix the fe
        # model sees — engineered columns, dtypes and completeness included —
        # so runs can be compared without re-deriving features. Never read back.
        config.FE_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = config.FE_FEATURES_DIR / "weekly_features_fe.csv"
        feats_df.to_csv(snapshot_path, index=False)
        print(f"      fe feature-matrix snapshot -> {snapshot_path}")

    # Full-frame duration targets (computed once; labels exist through end-2025)
    full_duration_targets = duration_model.build_duration_targets(feats_df)

    # 2. Create three time splits. Each split uses a longer training period.
    print("\n[2/7] Generating 3 expanding-window folds (2025 excluded from all folds)...")
    folds = splits.get_cv_folds(feats_df)

    oof_rows = []
    advisory_rows = []
    movement_rows = []
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

        # Per-fold advisory tier thresholds (estimated on this fold's train window)
        thresh_stats = advisory.climatology_threshold_stats(tr_df)
        advisories_f = advisory.generate_advisories(res_df, thresh_stats)
        movement_f = advisory.generate_movement_advice(
            advisories_f,
            dur_pred.assign(week=tst_df["week"], county=tst_df["county"]),
        )
        advisory_rows.append(advisories_f)
        movement_rows.append(movement_f)

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

    advisories_df = pd.concat(advisory_rows, ignore_index=True)
    movement_df = pd.concat(movement_rows, ignore_index=True)

    # 5. Data provenance report (real vs synthetic_fallback per source)
    provenance_df = data_loader.get_provenance_report()

    print("\n" + "=" * 95)
    print(f"DATA PROVENANCE (allow synthetic fallback: {config.ALLOW_SYNTHETIC_FALLBACK})")
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
        "advisories.csv": advisories_df,
        "movement_advice.csv": movement_df,
        "data_provenance.csv": provenance_df,
    }
    if importance_detail is not None:
        out_files["feature_importance.csv"] = importance_detail
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
    n_advisory_weeks = advisories_df["week"].nunique() if len(advisories_df) else 0
    print("\n" + "=" * 95)
    print("PIPELINE COMPLETE — V3 SUMMARY")
    print("=" * 95)
    print(f"  OOF test weeks scored: {len(all_oof_df)} "
          f"({all_oof_df['test_year'].min()}-{all_oof_df['test_year'].max()}, 2025 excluded)")
    print(f"  Pooled {model_label}: Brier {pooled_lgbm['brier_score']:.4f}, "
          f"BSS vs clim {pooled_lgbm['bss_vs_climatology']:.3f}, "
          f"BSS vs pers {pooled_lgbm['bss_vs_persistence']:.3f}, "
          f"mean pinball {pooled_lgbm['mean_pinball_loss']:.3f}, "
          f"coverage(flood) {pooled_lgbm['coverage_flood_weeks']:.3f}, "
          f"MAE(flood) {pooled_lgbm['mae_flood_weeks']:.2f} km2")
    print(f"  Advisories for {n_advisory_weeks} test weeks; "
          f"{len(movement_df)} movement recommendations")
    n_real = int((provenance_df['provenance'] == 'real').sum())
    n_synth = int((provenance_df['provenance'] == 'synthetic_fallback').sum())
    print(f"  Provenance: {n_real} real sources, {n_synth} synthetic fallbacks")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("lightgbm", "lightgbm_baseline", "lightgbm_fe", "tabpfn"),
        default="lightgbm",
        help="lightgbm: current v3 model -> outputs_v3 | "
             "lightgbm_baseline: frozen v3 baseline -> outputs_baseline | "
             "lightgbm_fe: engineered features (+county, +cnty_tp_w7, "
             "+cnty_ro_w7, -year) -> outputs_fe | "
             "tabpfn: TabPFN reference",
    )
    parser.add_argument(
        "--params",
        choices=tuple(lgbm_model.PARAM_SETS),
        default="baseline",
        help="hyperparameter set (LightGBM models only): "
             "baseline: frozen v3 defaults (unchanged behavior) | "
             "baseline_optuna: Optuna winners tuned on the baseline features | "
             "fe_baseline: the same v3 defaults (fe grid cell) | "
             "fe_optuna: Optuna winners tuned on the engineered features. "
             "Optuna sets save to their own outputs_<model>_<params>/tables "
             "folder instead of the frozen per-model folders.",
    )
    args = parser.parse_args()
    main(args.model, args.params)
