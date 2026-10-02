# LightGBM Flood Prediction & Advisory Pipeline (`LightGBM_v1`)

> **Project**: JBG060 / ZHL — Northern Bahr el Ghazal Flood Modeling (South Sudan)
> **Scope**: 5 Aweil Counties (*Aweil Centre, Aweil East, Aweil North, Aweil South, Aweil West*)
> **Version**: `v3` (3-Fold Expanding-Window Cross-Validation, Protocol-Aligned Metrics, Provenance Tracking)

---

## 1. Executive Summary

This package implements an end-to-end **LightGBM Machine Learning Pipeline** that predicts flood risk at county level in Northern Bahr el Ghazal, South Sudan.

The pipeline produces three core predictions per county and per week:
1. **`det_prob`**: Probability of a detected flood event ($P(\text{flood}) \in [0, 1]$).
2. **`q10 / q50 / q90`**: Quantile flood area in km² (Pessimistic $q10$, Median $q50$, Worst-case $q90$).
3. **Multi-Horizon Duration**: Probability that flooding is detected 1, 2, 3, and 4 weeks ahead.

These predictions feed into rule-based stakeholder layers to generate:
- **`advisories.csv`**: Risk Tiers (0 to 3), agricultural crop phase, and exposed hectares.
- **`movement_advice.csv`**: Multi-week cattle relocation recommendations pointing herders to safe destination counties.

---

## 2. What Changed in v3

v3 fixes three evaluation-honesty problems found in the v2 design, following `evaluation_metrics.md`:

1. **3 expanding-window CV folds** (replacing 5 folds of 5-year blocks). Walk-forward design with an expanding training window, 1 validation year, and 2–3 held-out test years per fold. Every flood season 2009–2024 is used for validation/test exactly once.
2. **Per-fold references**: climatology & persistence baselines and advisory tier thresholds are estimated *inside each fold's training window* and never see validation/test years.
3. **Protocol-aligned metrics**: the cutoff-dependent classification family (accuracy, precision, recall, F1, CSI, FAR) is removed from the headline summary — it throws the forecast probability away. Headline detection accuracy is the **Brier Score** with skill scores vs climatology and persistence, plus reliability diagrams. Area metrics are pinball loss, interval coverage, and flood-week MAE/bias. Duration gets its own Brier metric per horizon.

### 2.1 Cross-Validation Folds (`config.CV_FOLDS`)

| Fold | Train | Val | Test |
|------|-------|-----|------|
| 1 | 2000–2008 | 2009 | 2010–2011 |
| 2 | 2000–2014 | 2015 | 2016–2017 |
| 3 | 2000–2020 | 2021 | 2022–2024 |

**2025 is loaded** (so feature completeness is real, not padded) **but stays outside the CV**: no fold trains, validates, or tests on 2025, and no CV output contains 2025. The first week of each validation/test window is purged so no lag feature or label window crosses a fold boundary.

### 2.2 Data Provenance

Every data loader records its source in a provenance registry (`data_loader.PROVENANCE`): `real` or `synthetic_fallback`, plus first/last valid date. Synthetic fallbacks are gated by `config.ALLOW_SYNTHETIC_FALLBACK` (currently `True`); set it to `False` and any missing real source raises `RuntimeError` instead of silently substituting generated data. Each run prints the provenance report and writes it to `tables/data_provenance.csv` — check it before interpreting results.

---

## 3. Pipeline File Structure

```
LightGBM_v1/
├── __init__.py            # Package initialization
├── config.py              # Paths, MODEL_VERSION, CV_FOLDS, embargo, provenance gate, scope
├── data_loader.py         # Signal & label loaders with provenance registry (real vs synthetic_fallback)
├── splits.py              # 3 expanding-window CV folds & boundary purging
├── features.py            # 50 rolling hydro features enforcing the Friday 3-day embargo cutoff
├── baselines.py           # Climatology (smoothed & median) and Markov Persistence reference baselines
├── lgbm_model.py          # LightGBM detection classifier & q10/q50/q90 quantile area regressors
├── duration_model.py      # Multi-horizon binary LightGBM classifiers (1-4 weeks ahead)
├── metrics.py             # Protocol metrics (Brier, BSS, pinball, coverage, MAE/bias, reliability)
├── advisory.py            # Stakeholder advisories.csv & movement_advice.csv generator
├── run.py                 # Master execution script orchestrating the end-to-end pipeline
└── outputs/
    ├── outputs_v1/        # Execution outputs for initial baseline model
    ├── outputs_v2/        # Execution outputs for 5-fold CV & monthly/county breakdown
    └── outputs_v3/        # Execution outputs for 3-fold expanding-window CV (current)
        └── tables/
            ├── cv_test_predictions_aweil.csv
            ├── cv_metrics_summary_aweil.csv
            ├── duration_metrics_aweil.csv
            ├── reliability_det_prob.csv
            ├── reliability_duration_h1..h4.csv
            ├── county_metrics_aweil.csv
            ├── monthly_metrics_aweil.csv
            ├── advisories.csv
            ├── movement_advice.csv
            └── data_provenance.csv
```

---

## 4. Key Modules & Functions

### **`config.py`**
- Centralizes project paths, seed (`SEED=2026`), versioning (`MODEL_VERSION="v3"`), geographic scope, and embargo settings (`EMBARGO_DAYS=3`).
- Defines `CV_FOLDS`: 3 expanding-window folds (see table above).
- `ALLOW_SYNTHETIC_FALLBACK`: provenance gate for synthetic data substitution.

### **`data_loader.py`**
- `load_boundaries()`, `load_flood_labels()`: Admin-2 GeoJSON boundaries and weekly county detected flood area (km²) from committed satellite EDA datasets.
- `load_era5_dataset()`, `load_gauge_daily()`, `load_albert_level()`, `load_et0_daily()`: AgERA5 precipitation/runoff, Dartmouth gauge 100205 discharge, Lake Albert levels, reference evapotranspiration.
- `PROVENANCE` + `get_provenance_report()`: real-vs-synthetic registry per source, written to `data_provenance.csv` every run.

### **`splits.py`**
- `get_cv_folds()`: builds the 3 expanding-window folds from `config.CV_FOLDS`; purges the first week of each val/test window.

### **`features.py`**
- `build_weekly_features()`: embargoed weekly feature matrix — every feature dated for a target week starting Monday *m* is dated at most *m* − 3 days. County-weeks with any missing hydro feature are flagged `complete=False`; `run.py` drops and reports them. Lake Albert altimetry is sparse (~1 obs per 10-day satellite revisit, first obs 2002-07-10) and is forward-filled to daily (past data only); its timestamps carry the satellite pass time-of-day and are normalized to the pass day so they align with the daily grid; weeks before the first observation (~2000-01–2002-07) are incomplete and excluded.

### **`baselines.py`**
- Climatology (smoothed `(k+0.5)/(n+1)` probability, median area) and first-order Markov Persistence (`p_on`/`p_off`) plus last-detection area baseline. Called **per fold** by `run.py` with a training-window mask, so references never leak.

### **`lgbm_model.py` / `duration_model.py`**
- `build_and_train_lightgbm()`: LGBMClassifier for `det_prob` + LGBMRegressor quantiles (α=0.1/0.5/0.9) for area. Early stopping on the fold's validation year.
- `train_duration_models()` / `predict_duration()`: one binary classifier per horizon h1–h4, "flood detected h weeks ahead".

### **`metrics.py`**
- Detection: `brier_score`, `brier_skill_score`, `reliability_diagram_data`.
- Area: `pinball_loss`, `mean_pinball_loss`, `interval_coverage`, `flood_week_mae_and_bias`.
- Duration: `duration_climatology_reference`, `duration_persistence_reference`, `evaluate_duration_metrics`.
- Summaries: `evaluate_summary_metrics` (per-fold + pooled rows), `evaluate_county_metrics`, `evaluate_monthly_metrics` — protocol columns only.

---

## 5. Evaluation Protocol (aligned with `evaluation_metrics.md`)

The headline summary (`cv_metrics_summary_aweil.csv`) contains **protocol metrics only**, one row per (fold, model) plus pooled rows for LightGBM, Climatology, and Persistence:

### **A. Detection Probability**
- **Brier Score**: $\frac{1}{N} \sum (\hat{p}_i - y_i)^2$ — headline probabilistic accuracy of `det_prob` (lower is better).
- **Brier Skill Score (BSS)**: $1 - \frac{\text{Brier}_{\text{model}}}{\text{Brier}_{\text{ref}}}$ — relative skill vs **Climatology** and vs **Persistence**, both estimated per fold inside the fold's training window. > 0 means added skill.
- **Reliability Diagram**: binned predicted-vs-observed frequency table (`reliability_det_prob.csv`), pooled over all OOF test weeks.

### **B. Flood Area Quantiles**
- **Pinball Loss**: $L_q(y, \hat{y}) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$ averaged over $q \in \{0.1, 0.5, 0.9\}$.
- **Interval Coverage**: percentage of actual flood area falling inside $[q10, q90]$ (nominal ~80%), on **flood weeks ($y_{\text{true}} > 0$)** as the headline metric and on all weeks.
- **Flood-Week MAE & Bias**: computed strictly on flood weeks.

### **C. Duration (per horizon h1–h4)**
- **Brier Score** of `det_prob_h{h}` against "flood detected h weeks ahead", with BSS vs a **county-month climatology** reference and a **Markov persistence** reference, both evaluated **at the target week** and estimated per fold (`duration_metrics_aweil.csv`).
- **Reliability diagrams** per horizon (`reliability_duration_h1..h4.csv`).

> The classification family (accuracy, precision, recall, F1, CSI, FAR) is intentionally absent from Task-A outputs: it is cutoff-dependent and reserved for Task B event metrics in `evaluation_metrics.md`.

### **Honesty note — label censoring**
The flood labels come from satellite flood masks with irregular observation coverage. Weeks without a valid detection are filled with zero area, so some "dry" labels are censored (unobserved) rather than truly dry. This biases detection metrics conservatively but is inherited from the committed EDA datasets; the provenance report and the EDA provenance files document the observation coverage.

---

## 6. Explanation of Output Files

All outputs are saved to `LightGBM_v1/outputs/outputs_v3/tables/`:

1. **`cv_test_predictions_aweil.csv`** — Out-of-fold predictions for every fold test week (2010–2011, 2016–2017, 2022–2024). Columns: `county`, `week`, `test_year`, `y_true`, `y_det`, `det_prob`, `q10/q50/q90`, per-fold baselines, `fold`, duration predictions `det_prob_h1..h4`, duration targets & per-fold references.
2. **`cv_metrics_summary_aweil.csv`** — 3 fold rows × 3 models + 3 pooled rows, protocol columns only.
3. **`duration_metrics_aweil.csv`** — Per (fold, horizon) + pooled: Brier and BSS vs the two per-fold references.
4. **`reliability_det_prob.csv` / `reliability_duration_h1..h4.csv`** — Binned reliability tables (pooled OOF).
5. **`county_metrics_aweil.csv` / `monthly_metrics_aweil.csv`** — Per-county and per-month (Jan–Dec, rainy vs dry) protocol breakdowns.
6. **`advisories.csv`** — Weekly stakeholder advisories for all fold test weeks: `tier` (0–3), agricultural `phase`, exposed hectares, cattle count, plain-language text.
7. **`movement_advice.csv`** — Duration-based cattle relocation guidance recommending destination counties for high-risk weeks.
8. **`data_provenance.csv`** — Per-source provenance (`real` / `synthetic_fallback`) with coverage ranges.

---

## 7. How to Run & Versioning

### **Running the Pipeline**
Execute the master script from the repository root:
```bash
python LightGBM_v1/run.py
```

### **Changing Model Version**
To run an experiment and output to a new version folder (e.g., `outputs_v4`):
1. Open `LightGBM_v1/config.py`.
2. Change `MODEL_VERSION = "v4"`.
3. Run `python LightGBM_v1/run.py`.
4. Outputs are automatically written to `LightGBM_v1/outputs/outputs_v4/tables/`.