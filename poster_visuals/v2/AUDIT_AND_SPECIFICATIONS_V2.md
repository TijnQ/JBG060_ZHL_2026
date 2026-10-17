# Poster Visuals Suite V2: Audit Checklist & Specifications

This directory (`poster_visuals/v2`) contains the complete set of 5 tailored, publication-ready poster figures for the **JBG060 Capstone Project (Northern Bahr el Ghazal Flood Forecasting)**.

---

## 1. Suite Overview & File Structure

```
poster_visuals/
├── v1/                                        [Original 6 Visualizations Suite]
└── v2/                                        [New Tailored 5 Visualizations Suite]
    ├── 1_End_to_End_Pipeline_Architecture/
    │   ├── Visual1_Hero_Pipeline_Architecture.png (300 DPI Hero Banner)
    │   ├── Visual1_Hero_Pipeline_Architecture.pdf (Vector Print PDF)
    │   └── Visual1_Hero_Pipeline_Architecture.svg (Lossless Standalone SVG)
    ├── 2_Upstream_Feature_Lag_and_Schematic/
    │   ├── Visual2_Upstream_Lag_and_Catchment_Schematic.png (300 DPI)
    │   └── Visual2_Upstream_Lag_and_Catchment_Schematic.pdf (Vector Print PDF)
    ├── 3_Model_Benchmark_and_Lead_Time_Skill/
    │   ├── Visual3_Model_Benchmark_and_Lead_Time_Skill.png (300 DPI)
    │   └── Visual3_Model_Benchmark_and_Lead_Time_Skill.pdf (Vector Print PDF)
    ├── 4_County_to_Grid_Downscaling_Flow/
    │   ├── Visual4_County_to_Grid_Downscaling_Flow.png (300 DPI)
    │   └── Visual4_County_to_Grid_Downscaling_Flow.pdf (Vector Print PDF)
    ├── 5_Multi_Layer_Spatial_Exposure_Map/
    │   ├── Visual5_Multi_Layer_Spatial_Exposure_Map.png (300 DPI)
    │   └── Visual5_Multi_Layer_Spatial_Exposure_Map.pdf (Vector Print PDF)
    └── AUDIT_AND_SPECIFICATIONS_V2.md
```

---

## 2. Scientific Specifications & Verification

### Visual 1: End-to-End Pipeline Architecture Diagram (Hero Visual)
- **Role:** Placed horizontally across the top or center of the poster to anchor the methodology.
- **Left (Data Ingestion & Embargo Setting):**
  - Features upstream signals: Catchment Runoff ($Q_{\text{upstream}}$), Net Water Surplus ($P - ET_0$), and Remote Water Levels / Gauges (`gauge_w1` ... `gauge_w14`).
  - Strict Operational Embargo Badge: Highlights Friday 00:00 UTC cutoff preventing lookahead data leakage and aligning with weekly Red Cross / humanitarian action protocols.
- **Center (Model Engine):**
  - Multi-Head LightGBM GBDT architecture:
    - Head 1: Flood Detection Classifier ($P(\text{flood} > 0)$)
    - Head 2: Tabular Flood Extent Quantiles ($q_{10}, q_{50}, q_{90}\text{ km}^2$)
    - Head 3: Forward Flood Duration & Persistence ($h_1, h_2, h_3, h_4$ weeks)
  - Optuna Hyperparameter Exploration Node: Explains convergence to the irreducible noise floor (~0.366 loss).
- **Right (Downscaling & Exposure Split):**
  - Area-preserving disaggregation onto ~232 m grid cells via conditional pixel climatology $P(p \mid m, c)$.
  - Direct split into two operational exposure branches: Stationary Cropland vs. Mobile Pastoral Cattle Herds.

### Visual 2: Upstream Feature Lag & Lead-Time Schematic (Sub-Objective 1)
- **Panel A (Predictive Feature Attribution):**
  - Grouped horizontal bar chart across 7-, 14-, 21-, and 28-day lead windows.
  - Derived directly from trained LightGBM pooled gain importance: upstream runoff share increases from 1.16% at W1 to 3.11% at W4 (+168%), proving that large-catchment runoff accumulation provides multi-week predictive signal while local rainfall decays within 7 days.
- **Panel B (Hydrological Cross-Section):**
  - Topographic profile from the Congo-Nile divide headwaters (~1,000 m asl) across ~300 km of river channels into the low-gradient Northern Bahr el Ghazal floodplain (~400 m asl, slope < 0.05%).
  - Explains the physical transit delay that gives humanitarians 2–4 weeks of advance notice before upstream floodwaters pond downstream.

### Visual 3: Model Benchmark & Lead-Time Skill Decay (Sub-Objective 2)
- **Main Chart (Lead-Time Skill Curve):**
  - Evaluates Brier Skill Score (BSS vs Climatology) across Weeks 1 to 4:
    - Week 1: BSS = +24.6% (Brier = 0.0766)
    - Week 2: BSS = +22.0% (Brier = 0.0826)
    - Week 3: BSS = +20.9% (Brier = 0.0884)
    - Week 4: BSS = +26.0% (Brier = 0.0894)
  - Comparison against baselines: GBDT consistently maintains >+20% skill gain over Climatology, while First-order Markov Persistence collapses and drops below the model by Week 3 (-6.8%) and Week 4 (-7.5%).
- **Mini-Inset (Optuna Convergence Flatline):**
  - Real 54-trial convergence trajectory from `optuna_trials_lgbm_aweil_baseline_v3.csv`.
  - Visual proof that tuning saturated after trial ~10 at objective loss ~0.366–0.380, demonstrating that performance is bound by data noise rather than tree hyperparameters.

### Visual 4: County-to-Grid Downscaling Flow (Sub-Objective 3)
- **Step 1 (Tabular County Forecast):** Choropleth of Aweil counties displaying tabular median targets ($q_{50} = 105.80\text{ km}^2$ state total).
- **Step 2 (Conditional Pixel Climatology Mask):** 26-year empirical MODIS flood frequency prior revealing natural drainage channels and depression bowls.
- **Step 3 (Calibrated Inundation Grid):** Native ~232 m grid matching the exact tabular area constraint.
- **Exact Area Conservation Audit:**
  $$\sum_{p \in c} \text{sev}_p \times A_p \equiv q_{50}^{(c)} \quad (\text{Area Error } \Delta = 0.000\text{ km}^2)$$

### Visual 5: Multi-Layer Spatial Exposure Map (Impact & Anticipatory Action)
- **GIS Composite Map:** Integrates predicted inundation footprint (~232 m), ASAP cropland mask (`asap_mask_crop_v04.tif`), and GeoNode pastoral cattle density (`geonode__cattle_gha.tif`) alongside River Lol watercourses and settlement hubs.
- **SLE (Societal-Legal-Ethical) Audit Inset:**
  - Physical exposure quantification: $142\text{ km}^2$ cropland vs. $185,000$ cattle heads exposed during peak flood surge.
  - Stationary vs. Mobile Vulnerability Analysis: Stationary crops require 2–3 week warnings for early harvest; mobile herds require 7–14 day high-confidence warnings to relocate along dry corridors without triggering intercommunal grazing clashes.

---

## 3. Poster Print & Legibility Verification Checklist

- [x] **A0 Print Legibility:** All body text >= 8 pt, callouts >= 9.5 pt, titles >= 12 pt.
- [x] **Resolution:** All PNGs exported at 300 DPI native resolution.
- [x] **Vector Quality:** All figures exported as vector PDFs and Hero visual exported as SVG.
- [x] **Contrast:** Tested with dual-stroke text outlines and high-contrast WCAG-compliant color palettes.
- [x] **Area Conservation:** Mathematical area conservation confirmed with $\Delta = 0.000\text{ km}^2$.
