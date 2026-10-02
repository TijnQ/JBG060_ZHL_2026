"""Main execution script for LightGBM_v1 pipeline (v3: 3 expanding-window CV folds).

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
7.  Outputs saved to LightGBM_v1/outputs/outputs_v3/tables/
"""

from __future__ import annotations

import sys
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


def main() -> None:
    print("=" * 95)
    print(f"LIGHTGBM PIPELINE RUN — VERSION {config.MODEL_VERSION.upper()} — "
          f"{len(config.CV_FOLDS)} EXPANDING-WINDOW FOLDS, PROTOCOL METRICS")
    print("=" * 95)

    # 1. Feature Engineering & Embargo
    print("\n[1/7] Building weekly feature matrix with Friday 3-day embargo...")
    feats_df = features.build_weekly_features()
    n_incomplete = int((~feats_df["complete"]).sum())
    if n_incomplete:
        bad_years = sorted(pd.to_datetime(feats_df.loc[~feats_df["complete"], "week"]).dt.year.unique())
        print(f"      Dropped {n_incomplete} incomplete county-weeks (years: {bad_years}) — see provenance report")
    feats_df = feats_df[feats_df["complete"]].reset_index(drop=True)
    feats_df = feats_df.drop(columns=["complete"])
    print(f"      Total rows: {len(feats_df)}, Total features: {len(features.FEATURE_COLUMNS)}")

    # Full-frame duration targets (computed once; labels exist through end-2025)
    full_duration_targets = duration_model.build_duration_targets(feats_df)

    # 2. 3 Expanding-Window Folds
    print("\n[2/7] Generating 3 expanding-window folds (2025 excluded from all folds)...")
    folds = splits.get_cv_folds(feats_df)

    oof_rows = []
    advisory_rows = []
    movement_rows = []

    for fold_info in folds:
        f_num = fold_info["fold"]
        tr_df = fold_info["train"]
        val_df = fold_info["val"]
        tst_df = fold_info["test"]

        print(f"      Fold {f_num}: Train {fold_info['train_years']} ({len(tr_df)} rows) | "
              f"Val {fold_info['val_year']} ({len(val_df)} rows) | "
              f"Test {fold_info['test_years']} ({len(tst_df)} rows)")

        # 3. Per-fold baselines: estimated ONLY on this fold's training window.
        # Keep original index labels (NO ignore_index): tr/val/test are disjoint
        # year windows of the same frame, so labels are unique, and the
        # .loc[label] round-trip below requires them.
        fold_df = pd.concat([tr_df, val_df, tst_df])
        train_mask = fold_df.index.isin(tr_df.index)
        fold_df = baselines.add_baselines_to_dataframe(fold_df, train_mask)
        val_df = fold_df.loc[val_df.index]
        tst_df = fold_df.loc[tst_df.index]

        # Train & predict this fold's LightGBM models
        models_fold = lgbm_model.build_and_train_lightgbm(
            tr_df[features.FEATURE_COLUMNS], tr_df["y_true"], tr_df["y_det"],
            val_df[features.FEATURE_COLUMNS], val_df["y_true"], val_df["y_det"],
        )
        preds_tst = lgbm_model.predict_lightgbm(models_fold, tst_df[features.FEATURE_COLUMNS])

        res_df = tst_df[["county", "week", "y_true", "y_det",
                         "det_prob_clim", "area_clim_q50", "det_prob_persist", "area_persist"]].copy()
        res_df["test_year"] = res_df["week"].dt.year
        res_df["det_prob"] = preds_tst["det_prob"]
        res_df["q10"] = preds_tst["q10"]
        res_df["q50"] = preds_tst["q50"]
        res_df["q90"] = preds_tst["q90"]
        res_df["fold"] = f_num

        # Per-fold duration models (train/val inside this fold only)
        dur_models = duration_model.train_duration_models(tr_df, val_df)
        dur_pred = duration_model.predict_duration(dur_models, tst_df)
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

    # 4. Protocol metrics: per-fold + pooled summary, duration, breakdowns
    print("\n[3/7] Computing protocol metrics (Brier/BSS, pinball, coverage, flood-week MAE/bias)...")
    eval_table = metrics.evaluate_summary_metrics(all_oof_df)
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
    print("COUNTY BREAKDOWN (LIGHTGBM, OOF TEST WEEKS)")
    print("=" * 115)
    print(county_table[["county"] + _BREAKDOWN_COLS].to_string(index=False))
    print("=" * 115)

    print("\n" + "=" * 115)
    print("MONTHLY BREAKDOWN (LIGHTGBM, OOF TEST WEEKS)")
    print("=" * 115)
    print(monthly_table[["month_num", "month_name", "season"] + _BREAKDOWN_COLS].to_string(index=False))
    print("=" * 115)

    # 6. Save Output Tables to versioned outputs_v3 folder
    print("\n[4/7] Saving output tables...")
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

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
    for name, frame in out_files.items():
        frame.to_csv(config.TABLES_DIR / name, index=False)
    for h_name, rel_df in duration_reliability.items():
        rel_df.to_csv(config.TABLES_DIR / f"reliability_duration_{h_name}.csv", index=False)

    print(f"      -> {config.TABLES_DIR}")
    for name in list(out_files) + [f"reliability_duration_{k}.csv" for k in duration_reliability]:
        print(f"         {name}")

    # 7. Final summary
    pooled_lgbm = eval_table[(eval_table["scope"] == "pooled") & (eval_table["model"] == "lightgbm")].iloc[0]
    n_advisory_weeks = advisories_df["week"].nunique() if len(advisories_df) else 0
    print("\n" + "=" * 95)
    print("PIPELINE COMPLETE — V3 SUMMARY")
    print("=" * 95)
    print(f"  OOF test weeks scored: {len(all_oof_df)} "
          f"({all_oof_df['test_year'].min()}-{all_oof_df['test_year'].max()}, 2025 excluded)")
    print(f"  Pooled LightGBM: Brier {pooled_lgbm['brier_score']:.4f}, "
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
    main()
