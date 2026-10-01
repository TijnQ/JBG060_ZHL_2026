"""Main execution script for LightGBM_v1 pipeline (Version v2 with 5-fold 5-year splits, monthly & county breakdowns).

Performs:
1. 5-Fold Cross Validation (each fold: 3 years train, 1 year val, 1 year test)
2. Detailed monthly performance breakdown (Jan - Dec) focusing on rainy season months (Jun - Nov)
3. Detailed county-level performance breakdown across all 5 Aweil counties
4. Outputs saved to LightGBM_v1/outputs/outputs_v2/tables/
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


def main() -> None:
    print("=" * 95)
    print(f"LIGHTGBM PIPELINE RUN — VERSION {config.MODEL_VERSION.upper()} — 5-FOLD (5-YR BLOCKS), MONTHLY & COUNTY BREAKDOWNS")
    print("=" * 95)

    # 1. Feature Engineering & Embargo
    print("\n[1/8] Building weekly feature matrix with Friday 3-day embargo...")
    feats_df = features.build_weekly_features()
    print(f"      Total rows: {len(feats_df)}, Total features: {len(features.FEATURE_COLUMNS)}")

    # 2. Add Baselines
    print("\n[2/8] Estimating Climatology & Markov Persistence baselines...")
    train_mask = pd.to_datetime(feats_df["week"]).dt.year <= 2014
    feats_df = baselines.add_baselines_to_dataframe(feats_df, train_mask)

    # 3. 5-Fold Cross Validation (5 blocks of 5 years: 3 train, 1 val, 1 test)
    print("\n[3/8] Generating 5 folds (each fold: 3 yrs Train, 1 yr Val, 1 yr Test)...")
    folds = splits.get_5fold_5year_splits(feats_df)

    oof_test_preds = []

    for fold_info in folds:
        f_num = fold_info["fold"]
        tr_df = fold_info["train"]
        val_df = fold_info["val"]
        tst_df = fold_info["test"]

        print(f"      Fold {f_num}: Train {fold_info['train_years']} ({len(tr_df)} rows) | "
              f"Val {fold_info['val_year']} ({len(val_df)} rows) | "
              f"Test {fold_info['test_year']} ({len(tst_df)} rows)")

        X_train = tr_df[features.FEATURE_COLUMNS]
        y_train_area = tr_df["y_true"]
        y_train_det = tr_df["y_det"]

        X_val = val_df[features.FEATURE_COLUMNS]
        y_val_area = val_df["y_true"]
        y_val_det = val_df["y_det"]

        X_test = tst_df[features.FEATURE_COLUMNS]

        # Train models for this fold
        models_fold = lgbm_model.build_and_train_lightgbm(
            X_train, y_train_area, y_train_det, X_val, y_val_area, y_val_det
        )

        # Predict test year
        preds_tst = lgbm_model.predict_lightgbm(models_fold, X_test)

        res_df = tst_df[["county", "week", "split", "y_true", "y_det", "det_prob_clim", "area_clim_q50", "det_prob_persist", "area_persist"]].copy()
        res_df["det_prob"] = preds_tst["det_prob"]
        res_df["q10"] = preds_tst["q10"]
        res_df["q50"] = preds_tst["q50"]
        res_df["q90"] = preds_tst["q90"]
        res_df["fold"] = f_num

        oof_test_preds.append(res_df)

    # Combine out-of-fold test predictions across all 5 folds
    all_oof_test_df = pd.concat(oof_test_preds, ignore_index=True)

    # 4. Overall 5-Fold Test Metrics
    print("\n[4/8] Computing overall out-of-fold 5-Fold Test performance...")
    eval_table = metrics.evaluate_all_outputs(all_oof_test_df)

    print("\n" + "-" * 105)
    print(f"OVERALL 5-FOLD TEST PERFORMANCE SUMMARY (Out-of-Fold Test Years: 2004, 2009, 2014, 2019, 2024)")
    print("-" * 105)
    display_cols = [
        "model", "accuracy", "precision", "recall", "f1_score", "csi",
        "brier_score", "bss_vs_climatology", "bss_vs_persistence",
        "mean_pinball_loss", "coverage_flood_weeks", "mae_flood_weeks"
    ]
    print(eval_table[display_cols].to_string(index=False))
    print("-" * 105)

    # 5. Per-County Detailed Statistics
    print("\n[5/8] Computing detailed county-level performance breakdown...")
    county_table = metrics.evaluate_county_metrics(all_oof_test_df, model_name="lightgbm")

    print("\n" + "=" * 115)
    print("DETAILED COUNTY-LEVEL PERFORMANCE BREAKDOWN (LIGHTGBM MODEL — 5-FOLD TEST YEARS)")
    print("=" * 115)
    c_cols = [
        "county", "n_samples", "flood_weeks", "flood_rate_pct", "accuracy", "precision",
        "recall", "f1_score", "csi", "brier_score", "bss_vs_climatology",
        "bss_vs_persistence", "coverage_flood_weeks", "mae_flood_weeks", "bias_flood_weeks"
    ]
    print(county_table[c_cols].to_string(index=False))
    print("=" * 115)

    # 6. Per-Month Detailed Statistics
    print("\n[6/8] Computing detailed monthly breakdown (Rainy Season vs Dry Season)...")
    monthly_table = metrics.evaluate_monthly_metrics(all_oof_test_df, model_name="lightgbm")

    print("\n" + "=" * 115)
    print("DETAILED MONTHLY PERFORMANCE BREAKDOWN (LIGHTGBM MODEL — 5-FOLD TEST YEARS)")
    print("=" * 115)
    m_cols = [
        "month_num", "month_name", "season", "n_samples", "flood_weeks", "flood_rate_pct",
        "accuracy", "precision", "recall", "f1_score", "csi", "brier_score",
        "bss_vs_climatology", "bss_vs_persistence", "coverage_flood_weeks", "mae_flood_weeks"
    ]
    print(monthly_table[m_cols].to_string(index=False))
    print("=" * 115)

    # 7. Train Duration & Advisory Models
    print("\n[7/8] Training LightGBM duration models and generating advisories...")
    tr_full, val_full, tst_full = splits.get_temporal_splits(feats_df, purge=True)
    dur_models = duration_model.train_duration_models(tr_full, val_full)
    duration_preds_df = duration_model.predict_duration(dur_models, feats_df)
    duration_preds_df[["week", "county"]] = feats_df[["week", "county"]]

    advisories_df = advisory.generate_advisories(all_oof_test_df, feats_df)
    movement_df = advisory.generate_movement_advice(advisories_df, duration_preds_df)

    # 8. Save Output Tables to versioned outputs_v2 folder
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = config.TABLES_DIR / f"5fold_test_predictions_{config.SCOPE}.csv"
    metrics_path = config.TABLES_DIR / f"5fold_metrics_summary_{config.SCOPE}.csv"
    county_path = config.TABLES_DIR / f"county_metrics_{config.SCOPE}.csv"
    monthly_path = config.TABLES_DIR / f"monthly_metrics_{config.SCOPE}.csv"
    advisories_path = config.TABLES_DIR / "advisories.csv"
    movement_path = config.TABLES_DIR / "movement_advice.csv"

    all_oof_test_df.to_csv(preds_path, index=False)
    eval_table.to_csv(metrics_path, index=False)
    county_table.to_csv(county_path, index=False)
    monthly_table.to_csv(monthly_path, index=False)
    advisories_df.to_csv(advisories_path, index=False)
    movement_df.to_csv(movement_path, index=False)

    print(f"\n[8/8] Pipeline execution complete! Output files saved to folder:")
    print(f" -> {config.TABLES_DIR}")
    print(f"    1. Out-of-fold test predictions: {preds_path.name}")
    print(f"    2. 5-Fold metrics summary:      {metrics_path.name}")
    print(f"    3. County breakdown metrics:    {county_path.name}")
    print(f"    4. Monthly breakdown metrics:   {monthly_path.name}")
    print(f"    5. Advisories:                 {advisories_path.name}")
    print(f"    6. Movement advice:             {movement_path.name}")
    print("=" * 95)


if __name__ == "__main__":
    main()
