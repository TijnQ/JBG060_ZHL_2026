# LightGBM Flood Prediction & Advisory Pipeline (`LightGBM_v1`)

> **Project**: JBG060 / ZHL — Northern Bahr el Ghazal Flood Modeling (South Sudan)  
> **Scope**: 5 Aweil Counties (*Aweil Centre, Aweil East, Aweil North, Aweil South, Aweil West*)  
> **Version**: `v2` (5-Fold Cross-Validation, Monthly & County Breakdown)

---

## 1. Executive Summary

This package implements an end-to-end **LightGBM Machine Learning Pipeline** that predicts flood risk at county level in Northern Bahr el Ghazal, South Sudan. 

The pipeline produces three core predictions per county and per week:
1. **`det_prob`**: Probability of a detected flood event ($P(\text{flood}) \in [0, 1]$).
2. **`q10 / q50 / q90`**: Quantile flood area in km² (Pessimistic $q10$, Median $q50$, Worst-case $q90$).
3. **Multi-Horizon Duration**: Probability that flooding will persist 1, 2, 3, and 4 weeks ahead.

These predictions feed into rule-based stakeholder layers to generate:
- **`advisories.csv`**: Risk Tiers (0 to 3), agricultural crop phase, and exposed hectares.
- **`movement_advice.csv`**: Multi-week cattle relocation recommendations pointing herders to safe destination counties.

---

## 2. Pipeline File Structure

The pipeline is organized into modular Python files inside [`LightGBM_v1/`](file:///c:/Users/20244086/OneDrive%20-%20TU%20Eindhoven/Universiteit/Year%203/Q1/JBG060/JBG060_ZHL_2026/LightGBM_v1):

```
LightGBM_v1/
├── __init__.py            # Package initialization
├── config.py              # Configuration settings, paths, versioning (MODEL_VERSION), and embargo rules
├── data_loader.py         # Hydro-meteorology signals & committed flood mask label loaders
├── splits.py              # 5-fold 5-year block splits & boundary purging
├── features.py            # 50 rolling hydro features enforcing the Friday 3-day embargo cutoff
├── baselines.py           # Climatology (smoothed & median) and Markov Persistence reference baselines
├── lgbm_model.py          # LightGBM detection classifier & q10/q50/q90 quantile area regressors
├── duration_model.py      # Multi-horizon binary LightGBM classifiers (1-4 weeks ahead)
├── metrics.py             # Evaluation metric calculations (Accuracy, Precision, Recall, F1, Brier, BSS, Coverage)
├── advisory.py            # Stakeholder advisories.csv & movement_advice.csv generator
├── run.py                 # Master execution script orchestrating the end-to-end pipeline
└── outputs/
    ├── outputs_v1/        # Execution outputs for initial baseline model
    └── outputs_v2/        # Execution outputs for 5-fold CV & monthly/county breakdown
        └── tables/
            ├── 5fold_test_predictions_aweil.csv
            ├── 5fold_metrics_summary_aweil.csv
            ├── monthly_metrics_aweil.csv
            ├── county_metrics_aweil.csv
            ├── advisories.csv
            └── movement_advice.csv
```

---

## 3. Key Modules & Functions

### **`config.py`**
- Centralizes project paths, seed (`SEED=2026`), versioning (`MODEL_VERSION="v2"`), geographic scope, and embargo settings.
- Defines `FOLD_5YEAR_BLOCKS`: 5 folds of 5 years each (3 years Train, 1 year Val, 1 year Test).

### **`data_loader.py`**
- `load_boundaries()`: Loads Admin-2 GeoJSON county boundaries.
- `load_flood_labels()`: Loads weekly county detected flood area (km²) and detection flags from committed satellite EDA datasets.
- `load_era5_dataset()`: Loads AgERA5 daily precipitation (`tp`) and runoff (`ro`).
- `load_gauge_daily()`, `load_albert_level()`, `load_et0_daily()`: Loads Dartmouth gauge 100205 discharge, Lake Albert water levels, and evapotranspiration data.

### **`features.py`**
- `build_weekly_features()`: Constructs 50 tabular features.
- **Strict 3-Day Embargo Enforcement**: For a target week starting Monday $m$, all features are dated at most **Friday of the previous week ($m - 3\text{ days}$)** to prevent target-week data leakage.
- Features include: 1, 3, 7, 14, 30-day rolling sums of rainfall/runoff, gauge/lake means & 7-day changes, net moisture, county flood lags (`y_true_lag1..4`), and calendar seasonality.

### **`splits.py`**
- `get_5fold_5year_splits()`: Generates 5 non-overlapping 5-year folds:
  - **Fold 1**: Train 2000–2002, Val 2003, Test 2004
  - **Fold 2**: Train 2005–2007, Val 2008, Test 2009
  - **Fold 3**: Train 2010–2012, Val 2013, Test 2014
  - **Fold 4**: Train 2015–2017, Val 2018, Test 2019
  - **Fold 5**: Train 2020–2022, Val 2023, Test 2024
- `purge_boundary_weeks()`: Drops the first week of non-train splits so composite label windows do not overlap across split boundaries.

### **`baselines.py`**
- `climatology_detection_baseline()`: $(k + 0.5) / (n + 1)$ Laplace-smoothed monthly flood probability estimated strictly on training window.
- `climatology_area_baseline()`: Historical median flood area per county-month on training window.
- `persistence_detection_baseline()`: First-order Markov continuation probabilities ($p_{\text{on}} = P(\text{det}_t \mid \text{det}_{t-1}=1)$ vs $p_{\text{off}}$).
- `persistence_area_baseline()`: Prev-week observed flood area ($\text{lag-1}$).

### **`lgbm_model.py`**
- `train_detection_classifier()`: Fits `LGBMClassifier` for detection probability `det_prob`.
- `train_quantile_regressors()`: Fits `LGBMRegressor` for quantiles $\alpha \in \{0.1, 0.5, 0.9\}$. Enforces monotonicity ($q10 \le q50 \le q90$) and non-negativity.

### **`duration_model.py`**
- `train_duration_models()`: Fits binary `LGBMClassifier` models for horizons $h \in \{1, 2, 3, 4\}$ weeks ahead.

### **`metrics.py`**
- `evaluate_all_outputs()`: Computes global evaluation summary across splits.
- `evaluate_monthly_metrics()`: Computes monthly breakdown (Jan–Dec), highlighting rainy season peak months (Jun–Nov).
- `evaluate_county_metrics()`: Computes county-level breakdown for all 5 Aweil counties.

### **`advisory.py`**
- `magnitude_tier()`: Compares predicted area $q50$ to county-month historical thresholds (mean, p80, p95) to set Risk Tiers 0–3.
- `generate_advisories()`: Generates `advisories.csv` combining agricultural crop phase and exposed hectares.
- `generate_movement_advice()`: Generates `movement_advice.csv` recommending safe destination counties for cattle relocation.

---

## 4. Evaluation Metrics Defined

### **A. Detection Classification Metrics**
- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ — Percentage of all county-weeks correctly classified.
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Percentage of predicted flood events that actually flooded.
- **Recall (POD / Hit Rate)**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Percentage of actual flood events successfully detected.
- **F1 Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean of Precision and Recall.
- **CSI (Critical Success Index)**: $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ — Threat score ignoring true negatives.
- **FAR (False Alarm Rate)**: $\frac{\text{FP}}{\text{TP} + \text{FP}}$ — Proportion of false flood alarms.
- **Brier Score**: $\frac{1}{N} \sum (\hat{p}_i - y_i)^2$ — Mean squared error of probabilities (lower is better, 0 is perfect).
- **Brier Skill Score (BSS)**: $1 - \frac{\text{Brier}_{\text{model}}}{\text{Brier}_{\text{ref}}}$ — Relative skill improvement over Climatology and Persistence (>0 means added skill).

### **B. Flood Area Quantile & Interval Metrics**
- **Pinball Loss (Quantile Loss)**: $L_q(y, \hat{y}) = \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$ averaged over $q \in \{0.1, 0.5, 0.9\}$.
- **Interval Coverage**: Percentage of actual flood area $y_{\text{true}}$ falling inside $[q10, q90]$ (nominal target ~80%). Reported on **flood weeks ($y_{\text{true}} > 0$)** as the headline metric, and on all weeks.
- **Flood-Week MAE & Bias**: Mean Absolute Error and Mean Error $(\hat{y} - y)$ calculated strictly on flood weeks ($y_{\text{true}} > 0$).

---

## 5. Explanation of Output Files

All outputs are saved to [`LightGBM_v1/outputs/outputs_v2/tables/`](file:///c:/Users/20244086/OneDrive%20-%20TU%20Eindhoven/Universiteit/Year%203/Q1/JBG060/JBG060_ZHL_2026/LightGBM_v1/outputs/outputs_v2/tables):

1. **`5fold_test_predictions_aweil.csv`**
   - Out-of-fold predictions on test years (2004, 2009, 2014, 2019, 2024).
   - Columns: `county`, `week`, `split`, `y_true`, `y_det`, `det_prob`, `q10`, `q50`, `q90`, `fold`, baseline predictions.

2. **`5fold_metrics_summary_aweil.csv`**
   - Overall cross-validation performance comparison between LightGBM, Climatology, and Persistence.

3. **`monthly_metrics_aweil.csv`**
   - Monthly breakdown (Jan–Dec) showing performance during rainy season months (June–November) vs dry season months.

4. **`county_metrics_aweil.csv`**
   - County breakdown across the 5 Aweil counties (*Aweil Centre, Aweil East, Aweil North, Aweil South, Aweil West*), highlighting where the model makes mistakes (e.g. low recall in dry Aweil North vs area under-prediction in large Aweil East).

5. **`advisories.csv`**
   - Weekly stakeholder advisories containing `tier` (0–3), `phase` (agricultural phase), `crop_exposed_ha`, `rangeland_exposed_ha`, and plain-language advisory text.

6. **`movement_advice.csv`**
   - Multi-week duration relocation guidance recommending destination counties for cattle during high-risk weeks.

---

## 6. How to Run & Versioning

### **Running the Pipeline**
Execute the master script from the repository root:
```bash
python run.py
```
or directly inside `LightGBM_v1`:
```bash
python LightGBM_v1/run.py
```

### **Changing Model Version**
To run an experiment and output to a new version folder (e.g., `outputs_v3`):
1. Open [`LightGBM_v1/config.py`](file:///c:/Users/20244086/OneDrive%20-%20TU%20Eindhoven/Universiteit/Year%203/Q1/JBG060/JBG060_ZHL_2026/LightGBM_v1/config.py).
2. Change `MODEL_VERSION = "v3"`.
3. Run `python run.py`.
4. Outputs will automatically be written to `LightGBM_v1/outputs/outputs_v3/tables/`.
