# Part 2: Visualizations Needed & Audit Checklist (JBG060 Capstone)

> **Status:** All 6 required poster visualization assets and technical audits have been fully produced, verified, and exported into this `poster_visuals/` directory.

---

## Visual Assets Directory Structure

```
poster_visuals/
├── 1_Study_Area_Locator_Map/
│   ├── Map1_Study_Area_Locator_Map.png            [300 DPI high-resolution for A0 poster]
│   └── Map1_Study_Area_Locator_Map.pdf            [Vector PDF format]
├── 2_End_to_End_Pipeline_Diagram/
│   ├── Diagram1_Pipeline_Architecture.svg         [Pure scalable vector format]
│   ├── Diagram1_Pipeline_Architecture.png         [300 DPI high-resolution]
│   └── Diagram1_Pipeline_Architecture.pdf         [Vector PDF format]
├── 3_Hydrological_Lag_and_Decoupling/
│   ├── Chart1_OptionA_2024_Hydrological_Lag.png   [Option A: 2024 Weekly Rain vs Net Surplus vs Floods]
│   ├── Chart1_OptionA_2024_Hydrological_Lag.pdf   [Option A Vector PDF]
│   ├── Chart1_OptionB_Basin_Decoupling_2023_2024.png [Option B: White Nile vs River Lol Decoupling]
│   ├── Chart1_OptionB_Basin_Decoupling_2023_2024.pdf [Option B Vector PDF]
│   └── Option_Comparison_and_Recommendation.md    [Selection guide for poster team]
├── 4_Agricultural_Exposure_Phenology_Plot/
│   ├── Chart2_Crop_Phenology_Flood_Collision.png  [Gantt stages vs monthly flood risk curve]
│   └── Chart2_Crop_Phenology_Flood_Collision.pdf  [Vector PDF format]
├── 5_Sample_Downscaled_Forecast_Raster/
│   ├── Map2_Downscaled_Forecast_Peak_Flood_2024-11-04.png [Primary 300 DPI raster map]
│   ├── Map2_Downscaled_Forecast_Peak_Flood_2024-11-04.pdf [Vector PDF format]
│   └── Map2_MultiPanel_Forecast_Comparison.png    [Comparative prior vs prob vs severity]
├── 6_Benchmark_Performance_Callouts/
│   ├── Benchmark_Summary_Table.png                [Compact 3-row benchmark table]
│   ├── Benchmark_Summary_Table.pdf                [Vector PDF table]
│   ├── Benchmark_Statistical_Badges.png           [Large 300 DPI stat badges]
│   └── Benchmark_Statistical_Badges.svg           [Pure vector SVG stat badges]
└── AUDIT_CHECKLIST_AND_METADATA.md                [This document]
```

---

## Item-by-Item Audit Checklist

### 1. Study Area Locator Map (Map 1)
- [x] **Status:** COMPLETED & EXPORTED.
- **What is provided:**
  - `Map1_Study_Area_Locator_Map.png` (300 DPI) and `Map1_Study_Area_Locator_Map.pdf`.
  - Dual-panel layout:
    - **Panel A (National Context):** South Sudan national administrative boundary, all 10 states, neighboring countries (Sudan, CAR, DRC, Uganda, Ethiopia), Sudd wetland / White Nile corridor, and red bounding box framing the River Lol study area.
    - **Panel B (Study Area Detail):** Northern Bahr el Ghazal with all five Aweil counties clearly outlined and labelled (**Aweil Centre, Aweil East, Aweil North, Aweil South, Aweil West**), River Lol main channel and primary tributaries (Pongo / Chel) flowing through the catchment into the downstream Sudd system.
- **Audit Verification:**
  - Font sizes: County names set in uppercase bold (`11.5 pt` with rounded white contrast badges). Subtitle `10.5 pt`, panel titles `13.0 pt`, supertitle `15.5 pt`.
  - Scale bar: Explicitly calibrated in kilometres (`0 - 25 - 50 km` based on latitude degree scaling at 9°N).
  - Legibility: Scale bar and all labels remain 100% legible at A0 print dimensions (841 × 1189 mm).

---

### 2. End-to-End Pipeline Diagram (Diagram 1)
- [x] **Status:** COMPLETED & REDESIGNED.
- **What is provided:**
  - `Diagram1_Pipeline_Architecture.svg` (Pure vector SVG), `Diagram1_Pipeline_Architecture.pdf`, and `Diagram1_Pipeline_Architecture.png` (300 DPI).
  - Clear visual flowchart representing the complete 4-stage architecture:
    $$	ext{Data Ingestion (Embargoed)} \longrightarrow 	ext{Task A: County LightGBM} \longrightarrow 	ext{Task B: Pixel Downscaler} \longrightarrow 	ext{Asset Exposure Overlay}$$
- **Audit Verification:**
  - Vector format: Provided as both SVG and PDF with pure text tags, rounded cards, and scalable bezier connectors.
  - Visual contrast: Color-coded stages (Navy Blue → Teal → Amber → Red) against light neutral canvas (`#F8FAFC`).
  - Technical specifics included:
    - Friday embargo protocol ($t-3$ days, zero lookahead leak).
    - Multi-sensor ingestion (ERA5, GLEAM, Hydroweb, DAHITI, Dartmouth, MODIS).
    - Dual LightGBM heads (`det_prob` logloss, quantile pinball area $q_{10}/q_{50}/q_{90}$, multi-week duration $h_1..h_4$).
    - Downscaler coordinate-snapping bugfix ($k = 	ext{round}(v/	ext{step}-0.5)$) and exact area conservation ($\sum_p 	ext{sev} 	imes A = q_{50}$).
    - Asset exposure (ASAP crop phenology, FAO cattle density, WorldPop 100m).

---

### 3. Hydrological Lag & Decoupling Visual (Chart 1)
- [x] **Status:** COMPLETED — BOTH OPTIONS AUDITED & EXPORTED.
- **Option A (Recommended for Lag):** `Chart1_OptionA_2024_Hydrological_Lag.png` & `.pdf`
  - 2024 Weekly Local Rainfall vs. 7-Day Net Moisture Surplus vs. Detected Flood Pixels in Aweil.
  - **Dual Y-Axes Audit:**
    - Left Y-axis: Rainfall & Net Water Balance in **mm/week** (`-70` to `+125 mm`).
    - Right Y-axis: Detected Active Flood Pixels in **Pixel Counts** (`0` to `4,200` pixels).
  - **Key Finding:** Prominently highlights the **~9-week hydrological transit and soil saturation lag**: rainfall peaks in late August (Aug 26, 2024: $106.7	ext{ mm}$, net surplus $+78.3	ext{ mm}$, 0 active flood pixels), while flood inundation peaks catastrophically in early November (Nov 4, 2024: $3,522$ flood pixels, rainfall dropped to $<1	ext{ mm}$).
- **Option B (Recommended for Basin Decoupling):** `Chart1_OptionB_Basin_Decoupling_2023_2024.png` & `.pdf`
  - Hydrological Basin Decoupling: White Nile vs. River Lol in 2023 vs. 2024.
  - **Dual Y-Axes / Subplots Audit:**
    - Left Subplot: White Nile & Sudd Wetlands in **Satellite Detections (Millions)**. 2023: $12.3	ext{ M}$ vs. 2024: $7.1	ext{ M}$ ($-42\%$ drop).
    - Right Subplot: River Lol Catchment (Aweil East) in **Cumulative Inundation Area ($	ext{km}^2	ext{-weeks}$)**. 2023: $13.09	ext{ km}^2	ext{-weeks}$ (drought) vs. 2024: $850.97	ext{ km}^2	ext{-weeks}$ ($+6,400\%$ surge).
  - **Key Finding:** Proves that national flood headlines (driven by Lake Victoria / White Nile outflow) are decoupled from Aweil catchment dynamics.
- **Recommendation:**
  - Use **Option A** if your poster section focuses on the meteorological lag and why weather forecasts alone cannot predict flood timing without hydrological models.
  - Use **Option B** if your poster section focuses on the national policy narrative and why national early warning systems failed in Northern Bahr el Ghazal.

---

### 4. Agricultural Exposure / Phenology Collision Plot (Chart 2)
- [x] **Status:** COMPLETED & EXPORTED.
- **What is provided:**
  - `Chart2_Crop_Phenology_Flood_Collision.png` (300 DPI) and `Chart2_Crop_Phenology_Flood_Collision.pdf`.
  - 2-panel layout:
    - **Top Panel:** Gantt timeline for Sorghum & Pearl Millet phenological stages (Stage 1 Planting: May–Jun; Stage 2 Vegetative: Jul–Aug; Stage 3 Grain Ripening: Sep–Oct; Stage 4 Harvest & Drying: Oct–Nov).
    - **Bottom Panel:** Dual y-axis 26-year MODIS climatology curve showing Flood Detection Probability (%) [Navy curve] vs. Mean Inundation Area ($	ext{km}^2$) [Sky blue bars].
- **Audit Verification:**
  - **Collision Window Callout:** Prominently marked with shaded red vertical zone (`Sep–Nov`), dashed boundary lines, and header badge `"CRITICAL HARVEST WINDOW (Flood Risk > 80%)"`.
  - **Collision Milestones:**
    - October Area Peak: $80.6	ext{ km}^2$ (directly inundates ripening crops).
    - November Probability Peak: $91.8\%$ (coincides with harvest and outdoor drying).
  - Typography: Clean academic styling with zero en-dash/em-dash syntax issues.

---

### 5. Sample Downscaled Forecast Raster Map (Map 2)
- [x] **Status:** COMPLETED & EXPORTED.
- **What is provided:**
  - `Map2_Downscaled_Forecast_Peak_Flood_2024-11-04.png` (300 DPI), `Map2_Downscaled_Forecast_Peak_Flood_2024-11-04.pdf`, and `Map2_MultiPanel_Forecast_Comparison.png`.
  - Exported for catastrophic peak flood week **Nov 4, 2024** across the five Aweil counties.
- **Audit Verification:**
  - **Coordinate-Snapping Bugfix:** 100% reflected. Canonical cell centres snapped via $k = 	ext{round}(v/	ext{step}-0.5)$ ensuring perfect 100% cell merge between Parquet records and render grid with zero floating artifacts.
  - **Area Conservation Matching $q_{50}$:**
    - Model county forecast for Nov 4, 2024: $q_{50} = 166.32	ext{ km}^2$ (Aweil East: $83.71	ext{ km}^2$, Aweil South: $58.18	ext{ km}^2$, Aweil West: $17.44	ext{ km}^2$, Aweil North: $5.60	ext{ km}^2$, Aweil Centre: $1.39	ext{ km}^2$).
    - Map-implied unclipped area: $\sum_p 	ext{sev}_u(p) 	imes A_p = 166.32	ext{ km}^2$.
    - Area conservation error = $\mathbf{0.0000\%}$ (exact machine precision match).
  - **Color Contrast:** Deep `YlOrRd` heatmap overlay against bold dual-stroked county administrative boundaries (white outline with crisp slate-black boundary lines and clear pill badges for county names).

---

### 6. Benchmark Performance Callout Box & Summary Table
- [x] **Status:** COMPLETED & EXPORTED.
- **What is provided:**
  - `Benchmark_Summary_Table.png` & `.pdf`: Compact 3-row benchmark table strictly comparing LightGBM vs. Climatology vs. Persistence.
  - `Benchmark_Statistical_Badges.png` & `.svg`: Three large statistical badge callouts for high-impact poster presentation.
- **Audit Verification:**
  - **Strict Probabilistic Scoring Alignment:**
    - Zero arbitrary classification thresholds (Accuracy and F1 are excluded as uncalibrated and threshold-dependent).
    - Reported metrics strictly follow probabilistic meteorology standards (Brier Score, Brier Skill Score BSS, Quantile Pinball Loss, Interval Coverage, and Flood-Week MAE).
  - **Key Metrics Summary (Pooled Out-of-Fold Test Set, 1,810 County-Weeks):**

| Model / Protocol Reference | Brier Score ↓ | BSS vs. Climatology ↑ | BSS vs. Persistence ↑ | Pinball Loss ↓ | Flood-Week MAE ↓ | Flood-Week Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task A: LightGBM (Ours)** | **0.0650** | **+34.8% (+0.348)** | **+21.7% (+0.217)** | **0.685 km²** | **8.68 km²** | **73.8% (Target: ~80%)** |
| **Climatology Baseline** | 0.0996 | 0.0% (Ref) | -20.1% (-0.201) | 1.727 km² | 15.52 km² | 0.0% |
| **Persistence Baseline** | 0.0830 | +16.7% (+0.167) | 0.0% (Ref) | 0.949 km² | 8.82 km² | 0.5% |

  - **Key Badge Figures:**
    - **+34.8% BSS vs. Climatology:** LightGBM delivers 34.8% probabilistic skill over the 26-year seasonal baseline.
    - **+21.7% BSS vs. Persistence:** Delivers 21.7% skill over the previous week's satellite detection.
    - **8.68 km² Flood-Week MAE:** Achieves a **44.1% reduction in area prediction error** compared to Climatology ($15.52	ext{ km}^2$).

---

## Instructions to Reproduce or Regenerate

To regenerate all visual assets in this folder, run:
```bash
.venv/Scripts/python.exe poster_visuals/v1/generate_poster_visuals.py
```
