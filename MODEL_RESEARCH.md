# Model Research Notes — Flood Forecasting & Impact Advisory

**Status:** exploratory research only — **no models implemented yet**. This is a living document:
update it as experiments proceed (keep dated entries so the reasoning trail stays visible).

- 2026-09-21: initial research sweep (web literature + project EDA outputs); added TL;DR at top. See "Research log" at the bottom.
- 2026-09-21 (evening): **verification pass** — every reference re-checked against the arXiv API, Crossref and the four `literature/` PDFs; several citation errors corrected (§4.2–§4.4, References); regional South Sudan evidence added (§4.5); concrete **RTX 3090 implementation plan** added (§9, gated phases, pinned versions, runtime estimates).

## TL;DR — the recommended approach (2026-09-21)

**Build it as three stages; start with the cheap one.**

| Stage | What to build | Model | Framework / hardware |
|---|---|---|---|
| **A. County flood extent** (7-day ahead, per Aweil county, weekly) | tabular features (rolling ERA5 rain/runoff, gauge, lakes, ET0, calendar, lagged own extent) → next week's detected km² + detection probability | **LightGBM quantile regression** (primary), XGBoost/CatBoost as cross-checks, **TabPFN** as zero-shot baseline, climatology + persistence as the bar to beat; GRU/Mamba only if a temporal residual remains | scikit-learn + LightGBM, **CPU**, minutes per run (add to `requirements.txt` when we start) |
| **B. Sub-county "where"** (250 m risk map) | ERA5 tile stack (40×40 at 0.25°) + terrain/ag channels → 250 m flood mask, 7-day ahead | **small 2D U-Net** (probabilistic output; physics-informed U-Net+FNO / PIML variants only if it plateaus) | **PyTorch 2.x + CUDA on the RTX 3090** (24 GB fits it easily; 10–30 min/fold, ~1–3 h total — §9.5) |
| **C. Impact → advisory** (crops / cattle / what to do) | predicted extent × EDA exposure math × agricultural phase → exposed ha / cattle / escalating advisory text | **deterministic post-processing first — no learning** (IPC 2022–2025 is too short to train on; use it for case studies only) | pandas + the EDA modules we already have |

**Why this ladder:** our Aweil table is small (~1.4k non-zero county-weeks out of ~10.5k) and zero-inflated — the literature (NeurIPS 2022, TabArena 2025, Kratzert HESS 2018) says gradient boosting is the strongest first move on exactly this kind of data, while the GPU only earns its keep on the spatial task where there is genuine structure to exploit. Deep learning on *tabular* A is not justified yet; it gets promoted only if stage A shows a measurable temporal residual.

**Non-negotiables before any of it:** (1) recompute all lag correlations over **2015–2025** — the 2024-only Aweil results are negative and weak and must not leak into feature selection (§2, §5.3); (2) strict temporal splits (2000–2014 / 2015–2019 / 2020–2025) with a **3-day embargo** around the composite labels; (3) every model is scored **against climatology + persistence**, including spike-week metrics (POD/FAR/CSI), not just overall R².

**Bottom line in one sentence:** LightGBM quantile regression on weekly county features (CPU) as the backbone forecast, a small PyTorch U-Net on the 3090 for the spatial "where", and a rule-based exposure→advisory layer on top — with a multi-year feature audit as gate 0.

**Code status (2026-09-21):** fully implemented and **merged into `main`** (commit `46798d0`):
the `modeling/` package (14 modules) covers the data check, the gate-0 multi-year audit (E1),
baselines (E2), LightGBM / XGBoost / CatBoost / TabPFN training (E3–E4), SHAP attribution, the
U-Net (E6) and the deterministic advisory layer (E7). All logic is validated by a 68-check
known-answer suite (`python -m modeling.check_modeling`, no data or GPU needed) which caught and
fixed four real bugs during development. **Nothing has been run on the raw data yet and no model
has produced a single number.** Full maturity status — what is verified, what is not, and the
gates that turn "written" into "tested" — is in the **"Status of the modeling code"** section
below.

## 1. The question we are answering

1. **Where will the next flood be?** — forecast flood extent ~7 days ahead (the project's target
   lead time from the hydrometeorology EDA) for the five Aweil counties and, where feasible,
   sub-county "where" on the 250 m MCDWD grid.
2. **What do crops and cattle need to protect livelihoods?** — turn a flood forecast into impact
   estimates (hectares of crop / rangeland at risk, cattle exposed, phase of the agricultural
   cycle) and into plain-language advisories (e.g., "harvest early", "move herds to higher
   ground this month").

## 2. What the project EDA tells us (data reality check)

Facts gathered from `EDA_flood_masks/` and `EDA_hydrometeorology/` outputs (Aweil county CSVs verified
genuine on 2026-09-21; the `outputs_country` CSVs turned out to be synthetic — see the table below):

| Fact | Value / consequence |
|---|---|
| Target label | MCDWD 3-day composite detections, 2000–2025, ~250 m points. **Label = a 3-day window, not an instant.** Consequences: (a) a model predicting "week t" must not use any feature dated later than week t's window minus the 3-day composite overlap, (b) consecutive weeks' labels overlap in time → validation needs an **embargo/gap** (see §5). |
| Aweil volume | 972,588 pixel records, 1,457 county-weeks with detections, 357 detection runs, max run 33 weeks, peak month Oct (Nov for Aweil South), max weekly area 375 km² (Aweil East). |
| National volume | 58,053,688 records, 40,904 county-weeks, 8,476 runs (longest 279 weeks) — from the **real** nationwide flood-exposure EDA (`EDA_flood_masks/NATIONAL_README.md`: 2 tiles h20v08/h21v08, 2000–2025, 79 counties). |
| **Signal quality, Aweil 2024** (real data) | Lag correlations vs. daily flood pixels: local rainfall −0.13, local runoff −0.08, upstream rainfall −0.13, upstream runoff −0.04, **river discharge −0.45** (all at lag 0, decaying slowly to ~−0.30 at lag 14). The county notebook has **no synthetic fallback**, so these are genuine 2024 values — but a single year. |
| **Nationwide hydro signal — NOT yet established** | The committed `outputs_country/tables/south_sudan_lag_correlations.csv` (rainfall +0.56 … discharge **+0.81**) was produced by the **synthetic fallback** in `EDA_hydrometeorology/hydrometeorology_eda_country.py`: when the raw data was unavailable, the script silently generated sine-wave/gamma rainfall and Poisson flood pixels *from the same synthetic signals* — a circular result, not evidence. Its real-data path additionally reads flood observations from tile `h20v08` only, so it is not even nationwide. The genuine nationwide evidence is the flood-exposure EDA (flood extent + cattle/rangeland exposure only, no hydro signals). **The real nationwide lag audit (2015–2025 ERA5/gauge vs detections) is open work** — `modeling/audit.py`. |
| Interpretation | The only real measured correlations we have are the 2024 Aweil ones: **negative and weak** — possibly a 2024-specific / gauge-location artifact: the only gauge (area ID 100205, 9.6°N 31.6°E) sits downstream on the Sudd/Nile side, so "discharge rising" there plausibly tracks water *draining away from* Bahr el Ghazal after the flood plain recedes. **Action: recompute all lag correlations over 2015–2025 (multi-year) before trusting any feature** (`modeling/audit.py`, gate 0). Do not build on the 2024-only sign. |
| Sparsity | Weekly Aweil flood pixel counts are often 0–14 in the dry season; the label is **zero-inflated and spiky** (a 375 km² week follows many near-zero weeks). Any model must be evaluated on the *spike* component, not just overall R². |
| Observation quality | All `cloud_frac` values are zero → cannot filter bad observations; weeks without detections are "no detection", not "dry" (coverage gaps). Missingness is part of the label. |
| Exposure data | Crop & rangeland masks (ASAP v04) are **land fractions** (no crop type, no yield); cattle map is a **static coarse raster** (no goats/sheep, no herd mobility); IPC phase 3+ population only exists for **2022–2025** (5 three-month assessment rounds) — a short outcome window for any learned "impact" model. |
| Hydro inputs | ERA5 `tp`/`ro` daily at 0.25°, processed reference ET0 at one grid cell, one Dartmouth gauge, 3 lake level series (Victoria/Kyoga incomplete after 2002). |

**Bottom line:** this is a **data-scarce, single-region, zero-inflated forecasting problem** with
a rich *seasonal* structure and a few *weak* continuous predictors. The literature says the
right first moves are cheap tabular models, not heavy deep learning (below).

## 3. Task decomposition

| Task | Inputs → outputs | Scale | Best-fit model class (per §4) |
|---|---|---|---|
| **A. County flood extent** | tabular features per county-week (ERA5 rain/runoff rolling means, gauge, lakes, ET0, calendar, lagged own extent) → next week's detected area (km²), plus 0/>0 detection class | ~7k rows (Aweil), ~100k (national) | **Gradient boosting** (LightGBM/XGBoost/CatBoost); TabPFN as zero-shot check; climatology & persistence as mandatory baselines |
| **B. Sub-county "where"** | ERA5 tiles (40×40 at 0.25° per 10° tile) + terrain/ag masks → 250 m flood mask for the tile, 7-day ahead | per tile per week: 26 yrs × ~52 wks ≈ 1,300 samples/tile | **Convolutional U-Net family** on GPU (fits the 3090 easily); rainfall+terrain attention prior art exists |
| **C. Impact → advisory** | predicted extent (A/B) × crop/rangeland/cattle exposure (EDA already has the overlap machinery) → exposed ha / cattle / ag-phase → rule-based advisory text; optionally a small learned model linking exposure → IPC 3+ for the 2022–2025 window | small | **post-processing first** (deterministic), learned impact model only if the IPC window proves it |

## 4. Model options — evidence and recommendation

### 4.1 Baselines (must be beaten, always)

1. **Climatology**: mean weekly detected area per (county, month) from 2000–2014 (or leave-one-year-out). The EDA `monthly_seasonality` tables are essentially this already.
2. **Persistence**: last observed week's extent (detection runs make the process sticky).
3. **Nowcast from the 3-day composite**: last composite date's area.

These are cheap, defensible, and probably already capture most of the explainable weekly
variance given how strongly seasonal the signal is. Every model below is judged *against* them.

### 4.2 Task A — county-level tabular forecasting

**Recommended: gradient-boosted trees (LightGBM or XGBoost, CPU, minutes per run).**

Evidence:
- Grinsztay & Bengio, *Why do tree-based models still outperform deep learning on tabular
  data?* (NeurIPS 2022; arXiv:2207.08815; title per the arXiv record): on typical tabular
  data, GBMs remain ahead of deep nets, especially with limited data. Our Aweil target has only
  ~1,400 non-zero county-weeks out of ~10,500 possible — that is "limited data".
- TabArena, *A Living Benchmark for ML on Tabular Data* (2025; arXiv:2506.16791): tree-based
  models still anchor the state of the art for tabular prediction, with deep models catching up
  on large-data regimes we do not have.
- Nevo S. et al. (2021). *Flood forecasting with machine learning models in an operational
  framework*. arXiv:2111.02780 — Google's operational flood-forecasting pipeline (live since
  2018, expanded geographically): ML stage-forecasting (LSTM + linear models) feeding two ML
  inundation models (Thresholding, and the 'Manifold' model — an ML alternative to hydraulic
  inundation modelling). Relevant three ways: (a) ML forecasts are proven in near-real-time
  operations; (b) it is the same "learn the where" bet our Task B makes (an ML 'where' layer
  replacing/informing a hydraulic model); (c) its feature philosophy (gauged + recent
  observations + static context, minimal assumptions) matches our data-scarce setting.
- Hydrology-side precedent: Kratzert et al. 2018 (HESS 22:6005, DOI 10.5194/hess-22-6005-2018,
  ~2,000 citations) showed LSTMs beat XGBoost on **daily streamflow** mainly via temporal memory;
  the follow-up large-sample work (arXiv:1907.08456) found **GRU/LSTM and GBM within a small
  margin of each other, with the edge going to the recurrent model at longer leads**. Our target
  is *weekly* aggregates (memory already baked into rolling features) and our predictor set is
  small — so the GBM is the rational first stop, with a GRU/Mamba promoted only if Task A shows
  a residual temporal-dependence the rolling features miss.

**Also worth running (cheap, no tuning): TabPFN.**
TabPFN-3.5 technical report (2026; arXiv:2609.17895) documents a tabular foundation model that
is explicitly evaluated on **non-i.i.d. data with temporal or grouped splits** and strong on
small datasets — a match for a ~7k-row Aweil table. It needs no hyperparameter tuning, runs in
minutes, and its probabilistic output (full predictive distribution) comes free. Caveat: check
current row/feature limits at experiment time; the code is **Apache-2.0 on GitHub
(PriorLabs/TabPFN LICENSE, verified 2026-09-21)** — still confirm the licence note attached to
the exact checkpoint (v1 vs v2/v3 vs TabPFN-3.5) before shipping anything.

**Later candidates (only if A is good but not great):**
- Small **GRU / Mamba-style SSM**: Kratzert's *NeuralHydrology* (arXiv:1903.07903) for LSTM
  interpretability; *RiverMamba: A State Space Model for Global River Discharge and Flood
  Forecasting* (2025; arXiv:2505.22535) as a modern reference for the SSM approach on exactly
  our signal type (discharge).
- **Extreme-event-aware forecasting**: our target is zero-inflated with rare large spikes.
  *Extreme Adaptive Transformer for Time Series Forecasting* (2026; arXiv:2607.02437) tackles
  exactly this (rare extreme events in hydrologic forecasting) — a template for re-weighting /
  re-sampling spikes rather than a model we must adopt.

**Probabilistic outputs:** recommend quantile regression (LightGBM has native quantile loss) so
the advisory layer can say "80% chance of >X km²" instead of a point. Klotz et al. 2022
(HESS 26:1673, DOI 10.5194/hess-26-1673-2022; the first author is **Klotz**, not 'Bech' —
corrected after the Crossref re-verification) is a solid reference for calibrating DL
uncertainty in rainfall–runoff if we go that route.

### 4.3 Task B — sub-county spatial "where" (the GPU task)

**Recommended: a small 2D U-Net (or U-Net + light attention) on the 3090.**

Inputs: stacked ERA5 channels (tp, ro, optionally ET0-derived moisture) at 0.25° over the
10°×10° tile h20v08 (40×40 grid) for a rolling window (e.g., last 7–14 days, or weekly
snapshots), plus static channels (elevation/terrain, crop fraction, rangeland fraction,
cattle density). Output: binary flood mask at 250 m (1920×1920 per tile) for the *next* week.

Prior art (all verified 2026-09-21):
- *Hierarchical Terrain Attention and Multi-Scale Rainfall Guidance For Flood Image Prediction*
  (2022; arXiv:2212.01819): CNN predicting flood maps **from rainfall + terrain** — the closest
  published analog to our "predict where from reanalysis" setting.
- *Mapping Global Floods with 10 Years of Satellite Radar Data* (2024; arXiv:2411.01411):
  DL-based global flood extent from Sentinel-1; useful as a comparison of what a "reference"
  flood mask looks like (and as a potential cross-check/label source).
- *Flood Detection with SAR: A Review of Techniques and Datasets* (Remote Sensing 2024;
  DOI 10.3390/rs16040656): survey of the mapping side, incl. dataset landscape (Sen1Floods11
  line of work; see Bonafilia et al. 2020, IEEE/CVF CVPR Workshops, DOI
  10.1109/CVPRW50498.2020.00113 — the previously cited 'Remote Sensing 13:2220' DOI belongs to
  a different SAR-fusion paper, Bai et al. 2021; corrected 2026-09-21).
- Bentivoglio R., Isufi E., Jonkman S.N., Taormina R. (2022). *Deep learning methods for flood
  mapping: a review of existing applications and future research directions*. *HESS* 26, 4345.
  DOI 10.5194/hess-26-4345-2022 — the definitive survey of DL flood-mapping architectures,
  data and pitfalls (SAR vs multispectral, label scarcity, evaluation) — read before designing
  the Task-B net and before writing the report's methods section.
- *A Comprehensive Survey on Deep Learning Solutions for 3D Flood Mapping* (2025;
  arXiv:2506.13201): if we add temporal depth (3D conv / video-style), this is the map.

**Honest caveat:** most DL flood *mapping* work is **event-triggered, SAR-supervised** (it sees
SAR imagery of the flood itself). Predicting a 7-day-ahead flood mask from **reanalysis only**,
in a data-scarce African flood plain, is **not well trodden** — a search for "flood extent
forecasting from reanalysis + ML" returns operational hybrid hydrologic-hydraulic systems
(arXiv:2405.00567, arXiv:2306.10059) rather than end-to-end DL. That is (a) a risk (no
off-the-shelf recipe) and (b) the project's novelty claim. Expect the U-Net to be *bounded*
by the physical information content of ERA5 (0.25° cells vs 250 m flooding in a 300 km²
floodplain); the realistic goal is **probabilistic risk maps** (where flooding is more likely),
not crisp extent. Two near-term variants have direct verified precedent: a CNN–LSTM that
fuses Sentinel-1-derived inundation with a MODIS-line signal across time (Giezendanner et al.
2023, arXiv:2305.00640 — Bangladesh; beats a CNN-only baseline) and FNO-based inundation
surrogates (Sun et al. 2023, arXiv:2307.16090 — FNO beat a U-Net baseline on simulated urban
inundation). If we ever want a modern curated flood-extent dataset for transfer/cross-checks,
STURM-Flood (Notarangelo et al. 2025, *Big Earth Data*, DOI 10.1080/20964471.2025.2458714) is
up to date — but our labels stay MCDWD-based (§2).

**Physics-informed variants (later, if plain U-Net plateaus):**
- *Advanced Flood Prediction with Physics-Guided Deep Learning: Combining UNet, FNO, and SAR/
  Optical Imagery* (2026; arXiv:2606.06524): U-Net + Fourier Neural Operator with shallow-water
  equation constraints.
- *Physics-Informed Machine Learning for Short-Term Flood Prediction* (2026; arXiv:2606.04143):
  hydrological constraints inside the LSTM loss, motivated explicitly by **data-scarce**
  environments (our case).
- *Process-Aware AI for Rainfall-Runoff Modeling* (2026; arXiv:2603.25093): mass-conserving
  networks.

**Geo-foundation-model stretch option:** *Prithvi-Complimentary Adaptive Fusion Encoder (CAFE):
unlocking full-potential for flood inundation mapping* (2026; arXiv:2601.02315) fine-tunes the
Prithvi geo-foundation model for flood mapping on 24 GB-class hardware. Only worth trying after
the small U-Net, and only if we can download the weights.

### 4.4 Task C — impact & advisory layer

**Do this as deterministic post-processing first; learn nothing here yet.**

1. Pipeline already half-built in the EDA: `seasonal_agricultural_exposure` (county, month,
   flood type → crop/rangeland exposed hectares) and the national cattle/rangeland exposure
   tables. The model's job is to feed a *predicted* extent where the EDA used *observed* extent.
2. Advisory rules (v1, transparent and defensible):
   - predicted extent × season phase (Aug vegetative vs Oct–Nov maturation/harvest, from EDA) →
     "crops at harvest: prioritize early harvest / relocate stored grain";
   - predicted rangeland exposure × cattle map → "herds at risk: move to higher ground /
     reserve forage now";
   - magnitude tiers: below climatology / above 80th percentile / above 95th percentile →
     escalating urgency wording.
3. **Learned impact (only later):** the only observed "real-world impact" series is IPC phase 3+
   (2022–2025, 5 rounds) — too short to learn a reliable causal model; use it for **sanity
   checks and case-study narrative**, not training. References framing this exact
   forecast→impact→decision step: *Impact-based flood forecasting in the Greater Horn of Africa*
   (Alfieri et al. 2024, NHESS 24, 199–224, DOI 10.5194/nhess-24-199-2024 — the paper already
   in `literature/`); *Floods and food security: a method to estimate the effect of inundation
   on crop availability* (Pacetti, Caporali & Rulli 2017, *Advances in Water Resources* 110,
   494–504, DOI 10.1016/j.advwatres.2017.06.019 — this IS the PDF in `literature/`; the earlier
   draft's 'Molinari 2019' attribution was wrong — verified against the PDF and Crossref); the
   companion crop-damage estimator *AGRIDE-c* (Molinari et al. 2019, NHESS 19, 2565, DOI
   10.5194/nhess-19-2565-2019); and the anticipatory-action framing for South Sudan
   (Easton-Calabria et al. 2024, *Disasters*, DOI 10.1111/disa.12654 — §4.5). For the
   mobility/accessibility angle:
   *Assessing road criticality and loss of healthcare accessibility during floods* (Int J Health
   Geographics 2022, DOI 10.1186/s12942-022-00315-2 — also in `literature/`), which motivates
   using the OSM road network loader we already have.
4. Operational context: GloFAS (*GloFAS – global ensemble streamflow forecasting and flood early
   warning*, Alfieri et al. 2013, HESS 17, 1161, DOI 10.5194/hess-17-1161-2013 — the author
   list was corrected 2026-09-21 after Crossref re-verification) is the reference
   **operational** system; ours is a regional, extent-based, impact-focused complement — worth
   citing in the report as the benchmark of "what operational looks like" (ensemble
   hydrological models, 2–14 day lead, river discharge focus — *not* extent). The
   ML-operations precedent is Nevo et al. 2021 (§4.2, arXiv:2111.02780).

### 4.5 Regional & operational context (South Sudan / East Africa) — all verified 2026-09-21

These do not change the model ladder — they sharpen the impact/advisory layer (Task C) and the
report's framing, and give the group region-specific references:

- *Geospatial Analysis of Population Exposure to Flooding in the Sudd Region, South Sudan*
  (Chol et al. 2026, *Journal of Flood Risk Management*, DOI 10.1111/jfr3.70168): directly on
  our study region — an independent, peer-reviewed population-exposure estimate to
  sanity-check our Task C exposure numbers and the wet-season peak narrative.
- *Possibilities and limitations of anticipatory action in complex crises: acting in advance of
  flooding in South Sudan* (Easton-Calabria et al. 2024, *Disasters*, DOI
  10.1111/disa.12654): real-world evidence on lead times, trust and what a forecast-based
  advisory can/cannot achieve in this crisis context — informs Task C wording (advisories must
  be believable and actionable, not just accurate).
- *Drivers and impacts of Eastern African rainfall variability* (Palmer et al. 2023, *Nature
  Reviews Earth & Environment*, DOI 10.1038/s43017-023-00397-x): authoritative review of the
  climate drivers (ENSO, Indian Ocean Dipole, long/short rains) behind our seasonal signal —
  use for the report's climate-dynamics section and to justify seasonal/lag features.
- *Key Considerations for Responding to Floods in South Sudan Through the
  Humanitarian-Peace-Development Nexus* (Moro et al. 2024, SSHAP briefing, DOI
  10.19088/sshap.2024.005): grey literature on how flood response is organised (Northern Bahr
  el Ghazal is a priority area) — grounds the advisory layer in how agencies actually operate.

## 5. Validation protocol (non-negotiable, agreed up front)

1. **Strict temporal splits, no shuffling, ever.** Proposed: train 2000–2014, validation
   2015–2019, test 2020–2025 (Aweil track); same years for national.
2. **Embargo the 3-day composites:** keep a ≥3-day gap between the last training feature date
   and the first test label window (label windows overlap in time — "purged" CV, the concept from
   de Prado's *Advances in Financial Machine Learning*, adapted here to satellite composites).
   Without this, "validation" silently uses near-future information.
3. **Multi-year feature audit first:** recompute all lag correlations (0–14 d) over 2015–2025
   *before* any model is trained. The 2024-only Aweil results (negative, weak) must not leak
   into feature selection.
4. **Metrics:**
   - extent: RMSE / MAE / R² **against baselines**, reported separately for (a) all weeks,
     (b) weeks where truth > 0, (c) spike weeks (truth > 95th percentile);
   - detection: POD (probability of detection), FAR (false alarm rate), CSI (critical success
     index) — standard in flood/nowcasting literature;
   - probabilistic: calibration curve + CRPS for the quantile outputs.
   - A model that beats persistence on R² but misses spikes is a failed model for our purpose.
5. **Sensitivity:** rerun with flood_type split (recurring vs unusual), and with/without the
   single gauge (it is one noisy station on the wrong side of the basin — the model must not
   depend on it).

## 6. Framework & hardware plan

- **Python 3.12+** (project standard; this session runs 3.13.5). All stack versions below are
  **pinned to what PyPI / pytorch.org actually serve (verified 2026-09-21)**; the full pinned
  list and install commands live in §9.1.
- **Tabular stack (Task A):** scikit-learn 1.9.1 (preprocessing, time-series split utilities)
  + **LightGBM 4.7.0** (primary; native `quantile` and `tweedie` objectives, GPU-capable) with
  XGBoost 3.4.1 / CatBoost 1.2.10 as cross-checks and SHAP 0.52.0 for attribution. These are
  **CPU weapons** — a full Aweil run is minutes; the GPU is not needed here (keep it for
  Task B). `requirements.txt` currently pins none of these — add the `requirements-ml.txt`
  block in §9.1 when implementation starts.
- **Spatial/GPU stack (Task B):** PyTorch **2.14.0** + torchvision 0.29.0 with the CUDA 12.x
  wheel (install line from pytorch.org/get-started; `cu126` at the time of writing). The
  **RTX 3090** (specs verified 2026-09-21 on NVIDIA's official product page + CUDA-GPUs page):
  GA102 Ampere, **10,496 CUDA cores, 1.70 GHz boost, 24 GB GDDR6X (384-bit, 936 GB/s), compute
  capability 8.6 (sm_86)**, PCIe 4.0 ×16; published FP32 ≈ 35.6 TFLOPS (GA102 class). Small/mid
  CNNs (U-Net-class, 10⁶–10⁷ params) fit trivially at batch sizes 16–64; with fp16 autocast +
  GradScaler our 40×40 U-Net trains in **10–30 min per fold**, ≈ **1–3 h for the full
  hardening suite** (§9.5). Plain PyTorch or Lightning 2.6.6 — pick one, not both.
- **Reuse, don't rewrite:** `processing_data/loading.py` and `loading_impact_data.py` already
  cover all raw inputs; `EDA_flood_masks/flood_eda.py` already computes the county-week labels
  (`weekly_county_floods.csv`), the event table and the climatology tables; the EDA modules
  contain the exposure-overlap math.
- **Interpretability:** SHAP for GBM feature attributions (report which of rain/runoff/gauge/
  season actually drove the forecast) — cheap, and required for a defensible course report.
- **No GPU in this session** (no `nvidia-smi` here) — all development of Task B should be
  written but only *smoke-tested* locally; real training happens on the 3090 machine (60-second
  sanity script in §9.1).

## 7. Risks & open questions (to revisit each milestone)

| # | Risk / question | Mitigation |
|---|---|---|
| 1 | 2024-only Aweil correlations are negative & weak → features may be near-noise at county scale | Multi-year audit (§5.3); if confirmed weak, lean on seasonality + persistence and treat ML uplift as modest |
| 2 | One gauge, downstream of the basin | Sensitivity run without it; consider removing permanently |
| 3 | Labels are 3-day composites over a weekly grid; missing observations look like "dry" | Embargoed splits; report coverage (detections/week) as a side feature; never claim "dry ground" from a gap |
| 4 | Exposure maps are static 2000s-era rasters; cattle map misses goats/sheep & mobility | State clearly in report; advisory wording uses "mapped rangeland/cattle", not actual herds |
| 5 | IPC outcome window = 5 rounds (2022–2025) | Use for validation narrative only, never for training an impact model |
| 6 | U-Net may underperform climatology-derived spatial prior | Ship the climatology-spatial-prior baseline (e.g., per-pixel historical flood frequency map) as the spatial null model |
| 7 | Era5 0.25° may be too coarse for 250 m floodplain detail | Predict at 0.25° first, then a nearest-neighbor "where" map at 250 m as a derived output; don't over-claim resolution |
| 8 | `EDA_hydrometeorology/hydrometeorology_eda_country.py` **silently falls back to synthetic data** (broad `except` in `main()` → sine/gamma/Poisson generator) when raw files are missing; its committed `outputs_country/` tables are synthetic and its real path reads only tile h20v08 for flood observations | `modeling/` never consumes that fallback (real loaders only; `modeling/data_check.py` fails loudly on missing inputs). The EDA script itself should be fixed to raise or print a loud warning instead of the silent fallback — suggested for the group, not done here |

## 8. Experiment ladder (all steps implemented in `modeling/` — see the status section for what remains unverified)

1. **E1 data check & audit** — `raw_data` completeness report; multi-year (2015–2025) lag
   correlations, county + national; coverage statistics. *(cheap, CPU, day-scale)*
2. **E2 baselines** — climatology, persistence, last-composite nowcast; score on the E5
   protocol. *(CPU, hours)*
3. **E3 GBM Task A** — LightGBM quantile regression on county-week features (Aweil first,
   then national); SHAP report; compare vs E2. In the code the **default run trains LightGBM
   only** (the primary backbone); XGBoost/CatBoost are cross-checks added via
   `--families lgbm,xgb,cat`, with a primary-vs-cross-check verdict on the model card.
   *(CPU, hours)*
4. **E4 TabPFN zero-shot** — same table, no tuning; check whether the foundation model beats
   E3 on this small, non-i.i.d. table. *(GPU/CPU, hours)*
5. **E5 protocol hardening** — embargoed walk-forward, spike-weighted metrics, gauge
   sensitivity. *(CPU, days)*
6. **E6 U-Net Task B** — tile-level 7-day-ahead mask; CSI/POD/FAR vs spatial-prior null;
   train on the 3090. *(GPU, hours — budget and runtime estimate in §9.5)*
7. **E7 advisory layer** — deterministic forecast→exposure→advisory pipeline + one worked
   2024 case study (e.g., an October week in Aweil East). *(CPU, days)*

## 9. Implementation plan — RTX 3090 platform (added 2026-09-21)

Goal: a **cheap, defensible forecast + impact-to-advisory stack** that runs end-to-end on the
3090 machine, built in **gated phases** so a broken phase never blocks the next one. Every
phase below maps to a standing experiment (E1–E7) and reuses the repo's existing loaders/EDA
(§6). Version pins were checked against PyPI/pytorch.org on 2026-09-21.

**Status (2026-09-21):** Phase 1–2 code (data check, audit, features, baselines, LightGBM +
XGBoost/CatBoost/TabPFN, SHAP, advisory layer) is written in `modeling/` and passes the synthetic
check suite; Phase 4 (U-Net) code is written for the 3090. **No phase has been run on the raw
data yet** — the dataset is not on the dev machine, and Phase 0's 60-second GPU smoke test has
not been executed (no GPU here). Everything below marked *(written)* is code-first, data-pending.

### 9.1 Phase 0 — Environment & hardware sanity (≤ 30 min)
- Pin the ML stack (create `requirements-ml.txt` — keep it separate from the pinned geo stack):
  ```
  scikit-learn==1.9.1
  lightgbm==4.7.0
  xgboost==3.4.1
  catboost==1.2.10
  shap==0.52.0
  torch==2.14.0
  torchvision==0.29.0
  pytorch-lightning==2.6.6   # only if the group chooses Lightning
  tabpfn==9.0.0              # Task-A-only, optional
  ```
  CPU-only dev machine: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  3090 machine (GPU, CUDA 12.x):
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` (pin to the
  CUDA build the local driver supports: run `nvidia-smi` first).
- 60-second GPU sanity on the 3090 (run before touching any model):
  ```python
  import torch
  assert torch.cuda.is_available(), "no CUDA"
  g = torch.cuda.get_device_properties(0)
  print(g.name, g.major, g.minor, g.total_memory/1e9, "GB")  # expect 'GeForce RTX 3090', 8, 6, 24
  x = torch.randn(256, 64, 40, 40, device="cuda")
  y = torch.nn.Conv2d(64, 64, 3, padding=1)(x).double().sum()
  y.backward(); torch.cuda.synchronize()
  print("smoke OK")
  ```
- **Exit criteria:** smoke test passes in < 60 s; expected 24 GB, compute capability `8, 6`
  (verified against NVIDIA's CUDA-GPUs list).

### 9.2 Phase 1 — E1 data audit + multi-year feature audit (CPU, ~1 day)
Feature engineering lives in one module `features/` so it is shared by every model. Outputs:
- **County-week feature table (Task A).** For each Aweil county-week (2000–2025): rolling
  ERA5 `tp`/`ro` means over 1/3/7/14 days; Dartmouth gauge level (rolling); lake-level series;
  ET0; calendar (month, day-of-year, wet/dry season); lagged own extent (t−1 … t−4);
  lagged upstream rain/runoff. **All covariates must be dated fully before the first day of the
  target 3-day window (label embargo applies to features too) and transformed only on the
  training split (no leakage, no global scaling).**
- **Multi-year lag-correlation audit (gate 0):** recompute the §2 correlations over
  **2015–2025** for county + national. If Aweil stays negative/weak after 10 years, the §7
  mitigation stands: lean on seasonality + persistence and report ML uplift honestly.
- **Deliverable:** `features/county_week.csv` + `plots/lag_corr_multiyr.png` + a short
  `NOTES.md` on which features survive.
- **Exit criteria:** every feature is pre-embargo-dated; the audit table is committed.

### 9.3 Phase 2 — E2 baselines + E3 GBM Task A (CPU, ~hours)
- **Baselines first (mandatory):** climatology (mean weekly area per county-month from
  2000–2014), persistence (last week's area), last-composite nowcast.
- **LightGBM quantile** (targets 0.1/0.5/0.9; `objective="quantile"`) + a `binary` detection
  head for the spike-vs-not decision. Sanity-check against XGBoost and CatBoost on the same
  features.
- **Evaluation fixed at the agreed temporal split** (train 2000–2014 → val 2015–2019 → test
  2020–2025), **no shuffling**, with the ≥3-day embargo; metrics per §5.4 (RMSE/MAE/R²/POD/
  FAR/CSI/calendar/CRPS) split into all / >0 / spike weeks. Report **relative to baselines**
  (skill score), not raw.
- **SHAP** on the test fold → one figure listing feature importance.
- **Deliverables:** `results/task_a_*.csv`, the metric table, and the baseline-comparison figure.
- **Exit criteria:** we can state, in one line, whether and by how much LightGBM beats
  climatology/persistence on spikes. If it does not, **stop and interrogate the features
  before spending GPU hours.**

### 9.4 Phase 3 — E4 TabPFN zero-shot (optional, ~hours)
Run TabPFN on the same table with no tuning; record whether the foundation model beats E3.
Decision gate only — if it does not, drop it. License caveat from §4.2 applies.

### 9.5 Phase 4 — E6 U-Net Task B on the 3090 (GPU; the only heavy-GPU phase)
- **Tile design (from §3/§4.3):** for tile h20v08 (40×40 at 0.25°), stack ERA5 `tp`/`ro` over
  a 7–14-day window + static channels (elevation, crop/rangeland fraction, cattle density).
  Downsample labels to the map that matches the EDA CSVs; predict a **probabilistic** flood mask
  (sigmoid on grid) and score with CSI/POD/FAR against a per-pixel historical flood-frequency
  (spatial-climatology) null.
- **Model:** small U-Net (~1–5 M params), channel depth modest (16/32/64), MixedPrecision
  (fp16 autocast + GradScaler) on the 3090; AdamW, cosine schedule, early stopping.
- **Budget & runtime (fits the 3090 comfortably):** 40×40 input at batch 32–64 fits in 24 GB
  even at fp32 and is *tiny* at fp16 — expect **10–30 min per fold** on full tile history, i.e.
  **≈ 1–3 h for the whole suite** including the physics/climatology null, gauge-sensitivity and
  reanalysis-window ablations. Keep a per-run budget (e.g. ≤ 30 epochs / ≤ 1 h) so an experiment
  can never burn a day.
- **Fallbacks in order:** if U-Net plateaus, try (1) FNO (Sun et al. 2023), (2) CNN–LSTM
  temporal fusion (Giezendanner et al. 2023), (3) physics-informed/FNO hybrids (§4.3).
- **Deliverables:** `results/task_b_*` metrics vs the spatial null + example forecast maps.
- **Exit criteria:** CSI/POD/FAR on the hold-out years reported against the spatial-climatology
  null; decision on whether to keep the small net as-is.

### 9.6 Phase 5 — E7 advisory / impact layer (CPU, ~days)
Deterministic forecast→exposure→advisory pipeline (no learned impact model — §4.4). Reuse the
EDA exposure tables; write the tiered advisory rules; one worked **October 2024, Aweil East**
case study linking predicted extent → exposed ha / cattle → advisory text. Sanity-check the
exposure numbers against the independent Sudd-exposure paper (Chol et al. 2026, §4.5).

### 9.7 Phase 6 — Write-up & ablation report
Consolidate: model cards, metric tables vs baselines, the multi-year audit, and the case study
into the course report; cite the verified references in §References.

---

## Status of the modeling code — what is verified, what is still untested (2026-09-21)

This section is the single source of truth for how far the implementation has actually come.
Short version: **the code is complete and self-checked, but it has never touched the real data
and no model has been trained.** Anything below "Verified" is a hypothesis until the gates in
"What makes it tested" have run.

### Verified (done on the CPU dev machine, 2026-09-21, no raw data present)

| Claim | Evidence |
|---|---|
| All 14 `modeling/` modules compile, import (without torch/shap/tabpfn installed) and pass lint | `compileall` over the whole repo; ruff + trailing-whitespace pre-commit hooks green; clean fast-forward merge into `main` (`46798d0`) |
| The 68-check known-answer suite passes | `python -m modeling.check_modeling` → `0 failure(s)`; pins exact CRPS values (incl. y outside [q10, q90]), split boundaries + boundary purge, baseline values, advisory tiers & exposure scaling, rolling-window arithmetic, audit coverage counting, U-Net shapes & null baseline |
| The suite is not decoration — it caught **4 real bugs** before merge | (1) `crps_quantiles` returned NaN for y outside the quantile range (would silently corrupt the headline metric on dry weeks); (2) the climatology baseline's `merge` silently dropped counties absent from the training period; (3) the audit counted *detection-days* instead of *weeks* with detections; (4) `rolling(1, min_periods=3)` invalid on pandas 3.0 |
| `requirements-ml.txt` pins are valid | all 10 versions re-checked against the PyPI JSON API on 2026-09-21 (all equal then-current latest) |

### Not verified (what "untested" actually means here)

1. **Zero real-data runs.** No loader in `modeling/` has ever seen a real NetCDF / CSV / raster
   (the dataset is not on the dev machine). Unknowns only real data can answer: NetCDF variable
   names & chunking, the gauge `information.xlsx` parse, the county-name join between the
   boundary file and the flood CSVs, missing ET0 weeks, the runtime of the national ERA5 box
   load (the heaviest step), and whether CatBoost/XGBoost/TabPFN behave as expected on this
   table (incl. whether TabPFN 9.0.0 supports quantile outputs — `train_tabpfn.py` degrades to
   point predictions if not).
2. **The U-Net has never executed.** torch is not installed on the dev machine, so the forward
   pass, the training loop, AMP and the CUDA path are code-reviewed only; the check suite
   covers shapes, grid mapping and the null baseline with synthetic tensors but not a trained
   network. First execution will be the §9.5 smoke test on the 3090 (also the first test of the
   CUDA 12.8 wheel against that machine's driver — §9.1).
3. **Zero trained models, zero metrics.** No model cards, SHAP plots or prediction CSVs exist.
   Every performance statement in this document (GBM beats persistence, U-Net beats the spatial
   null, advisory tiers are useful) is a hypothesis pending E1–E7.
4. **Environment gap:** `requirements-ml.txt` is not yet installed in any venv (the data-room
   venv only has the base `requirements.txt` stack).

**Most likely first failure points** (in expected order): gauge xlsx parsing → county-name join
→ missing ET0 weeks → TabPFN API drift → rasterio transform handling → runtime/memory of the
national ERA5 load.

### What makes it "tested" (gate order — each gate must run on real data and be inspected)

| Step | Command | Passes when… |
|---|---|---|
| 1. data check | `python -m modeling.data_check` | every inventory item is present & readable (no silent workarounds) |
| 2. **gate 0 audit** | `python -m modeling.audit --scope aweil` | multi-year (2000–2024) lag structure inspected: signal present → continue; weak → fall back to national scope (§7 risk 1) |
| 3. Task A training | `python -m modeling.train_task_a --scope aweil --shap` (then `--scope national`) | metric tables exist for both scopes; models beat persistence on the **spike slice** — an honest "does not beat" is a valid, recordable result |
| 4. Task B U-Net (3090) | `python -m modeling.unet --epochs 20` | it trains, the val-CSI curve is inspected, and the result (CSI vs null baseline) is recorded either way |
| 5. Advisory | `python -m modeling.advisory --scope aweil` | readable, correctly tiered text for a real 2024 week (case study) |

**Definition used here:** the code is *written* today, *validated* at the logic level (synthetic
known answers), and becomes *tested* only after steps 1–5 have each run once on real data with
their outputs inspected. Until then, no model result in any report may be cited from this
package.

---

## References (re-verified 2026-09-21 via arXiv API + Crossref + `literature/` PDFs)

- Kratzert F. et al. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM)
  networks. *HESS* 22, 6005. DOI 10.5194/hess-22-6005-2018
- Kratzert F. et al. (2019). Towards Learning Universal, Regional, and Local Hydrological
  Behaviors via Machine-Learning Applied to Large-Sample Datasets. arXiv:1907.08456
- Kratzert F. et al. (2019). NeuralHydrology — Interpreting LSTMs in Hydrology. arXiv:1903.07903
- Klotz D. et al. (2022). Uncertainty estimation with deep learning for rainfall–runoff
  modeling. *HESS* 26, 1673. DOI 10.5194/hess-26-1673-2022
- (2023). Caravan — A global community dataset for large-sample hydrology. *Scientific Data*.
  DOI 10.1038/s41597-023-01975-w
- Grinsztay L., Bengio S. (2022). Why do tree-based models still outperform deep learning on
  tabular data? *NeurIPS 2022*. arXiv:2207.08815
- (2025). TabArena: A Living Benchmark for Machine Learning on Tabular Data. arXiv:2506.16791
- Prior Labs (2026). TabPFN-3.5: Technical Report. arXiv:2609.17895
- Hollmann N., Müller S., Eggensperger K., Hutter F. (2022). TabPFN: A Transformer That Solves
  Small Tabular Classification Problems in a Second. arXiv:2207.01848
- (2025). RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting.
  arXiv:2505.22535
- Nevo S. et al. (2021). Flood forecasting with machine learning models in an operational
  framework. arXiv:2111.02780
- Ke G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  *NeurIPS 2017*. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Chen T., Guestrin C. (2016). XGBoost: A Scalable Tree Boosting System. arXiv:1603.02754
- (2026). Extreme Adaptive Transformer for Time Series Forecasting. arXiv:2607.02437
- (2026). Advanced Flood Prediction with Physics-Guided Deep Learning: Combining UNet, FNO, and
  SAR/Optical Imagery. arXiv:2606.06524
- (2026). Physics-Informed Machine Learning for Short-Term Flood Prediction. arXiv:2606.04143
- (2026). Process-Aware AI for Rainfall-Runoff Modeling. arXiv:2603.25093
- (2022). Hierarchical Terrain Attention and Multi-Scale Rainfall Guidance For Flood Image
  Prediction. arXiv:2212.01819
- (2024). Mapping Global Floods with 10 Years of Satellite Radar Data. arXiv:2411.01411
- (2024). Flood Detection with SAR: A Review of Techniques and Datasets. *Remote Sensing* 16(4).
  DOI 10.3390/rs16040656
- Bai Y. et al. (2021). Enhancement of Detecting Permanent Water and Temporary Water in Flood
  Disasters by Fusing Sentinel-1 and Sentinel-2 Imagery. *Remote Sensing* 13(11):2220.
  DOI 10.3390/rs13112220 (note: first author is Bai — this SAR-fusion paper is the actual
  holder of the DOI previously mislabelled 'Sen1Floods11'; the Sen1Floods11 dataset paper is
  Bonafilia 2020 below)
- Bonafilia D., Tellman B., Anderson T., Aiken E. (2020). Sen1Floods11: a georeferenced dataset
  to train and test deep learning flood algorithms for Sentinel-1. *IEEE/CVF CVPR Workshops*.
  DOI 10.1109/CVPRW50498.2020.00113
- Giezendanner J. et al. (2023). Inferring the past: a combined CNN-LSTM deep learning framework
  to fuse satellites for historical inundation mapping. arXiv:2305.00640
- Sun A.Y. et al. (2023). Rapid Flood Inundation Forecast Using Fourier Neural Operator.
  arXiv:2307.16090
- Notarangelo N.M. et al. (2025). STURM-Flood: a curated dataset for DL-based flood extent
  mapping. *Big Earth Data*. DOI 10.1080/20964471.2025.2458714
- Bentivoglio R., Isufi E., Jonkman S.N., Taormina R. (2022). Deep learning methods for flood
  mapping: a review. *HESS* 26, 4345. DOI 10.5194/hess-26-4345-2022
- (2025). A Comprehensive Survey on Deep Learning Solutions for 3D Flood Mapping. arXiv:2506.13201
- (2026). Prithvi-Complimentary Adaptive Fusion Encoder (CAFE) for flood inundation mapping.
  arXiv:2601.02315
- Alfieri L. et al. (2024). Impact-based flood forecasting in the Greater Horn of Africa.
  *NHESS* 24, 199–224. DOI 10.5194/nhess-24-199-2024 *(in `literature/`)*
- Pacetti T., Caporali E., Rulli M.C. (2017). Floods and food security: a method to estimate
  the effect of inundation on crop availability. *Advances in Water Resources* 110, 494–504.
  DOI 10.1016/j.advwatres.2017.06.019 *(in `literature/` — this is the actual PDF; corrected
  from the earlier 'Molinari 2019' attribution)*
- Molinari D. et al. (2019). AGRIDE-c, a conceptual model for the estimation of flood damage
  to crops: development and implementation. *NHESS* 19, 2565. DOI 10.5194/nhess-19-2565-2019
- Petricola S., Reinmuth M., Lautenbach S., Hatfield C., Zipf A. (2022). Assessing road
  criticality and loss of healthcare accessibility during floods: the case of Cyclone Idai,
  Mozambique 2019. *Int J Health Geographics* 21:14. DOI 10.1186/s12942-022-00315-2
  *(in `literature/`)*
- Alfieri L. et al. (2013). GloFAS – global ensemble streamflow forecasting and flood early
  warning. *HESS* 17, 1161. DOI 10.5194/hess-17-1161-2013
- Chol O. et al. (2026). Geospatial Analysis of Population Exposure to Flooding in the Sudd
  Region, South Sudan. *Journal of Flood Risk Management*. DOI 10.1111/jfr3.70168
- Easton-Calabria E. et al. (2024). Possibilities and limitations of anticipatory action in
  complex crises: acting in advance of flooding in South Sudan. *Disasters*.
  DOI 10.1111/disa.12654
- Palmer P.I. et al. (2023). Drivers and impacts of Eastern African rainfall variability.
  *Nature Reviews Earth & Environment*. DOI 10.1038/s43017-023-00397-x
- Moro L.N. et al. (2024). Key Considerations for Responding to Floods in South Sudan Through
  the Humanitarian-Peace-Development Nexus. *SSHAP*. DOI 10.19088/sshap.2024.005
- NASA (2025). MCDWD/VCDWD User Guide Rev F. *(in `literature/`)*
- de Prado M. (2018). *Advances in Financial Machine Learning* — purged/embargoed cross-
  validation (adapted here to overlapping 3-day composite labels).

## Research log

- **2026-09-21 13:56 CEST (doc update)** — Added the **"Status of the modeling code"** section
(verified vs untested, gate order that turns "written" into "tested", likely first-failure
points); §8 retitled (no longer "nothing built yet"); TL;DR code status now points to the merge
into `main` (`46798d0`) instead of the `model-exploration` branch, which was deleted after the
fast-forward merge. No research-content changes.

- **2026-09-21 (re-evaluation pass, after the evening pass)** — Read all repo changes made since the
document was created (verification pass, `data_quality/` package, national flood-exposure EDA with
real 2000–2025 data for 79 counties, and the nationwide hydrometeorology script). **Key correction:**
`EDA_hydrometeorology/hydrometeorology_eda_country.py` contains a **silent synthetic fallback**
(`build_country_hydrometeorological_dataset` — sine/gamma/Poisson generators); its committed
`outputs_country/tables/*` (incl. `south_sudan_lag_correlations.csv`) are therefore synthetic, and its
real-data path reads flood observations from tile h20v08 only. §2 corrected: the Aweil 2024 row is
genuine (county notebook has no fallback), the "nationwide 2024 positive correlations" row was removed
as evidence and replaced with the audit gap; new risk #8 added. Verified all §9.1 PyPI pins against the
PyPI JSON API (all equal the current latest releases). **Started implementation** on `model-exploration`:
new `modeling/` package (`data_check`, `audit`, `features`, `splits`, `metrics`, `baselines`,
`train_task_a`, `train_tabpfn`, `shap_report`, `advisory`, `unet`, `check_modeling`) +
`requirements-ml.txt` (verified pins) + module README; root README structure updated. Synthetic
check suite runs green on the CPU dev machine; real-data runs and the 3090 smoke test are pending.

- **2026-09-21 (evening)** — Verification & implementation-planning pass. (a) **Reference
  audit:** every cited arXiv ID re-checked against the arXiv Atom API (all exist); every DOI
  re-checked against Crossref; the four `literature/` PDFs parsed (pypdf) to confirm their true
  title/author/DOI. **Corrections made:** "Bech 2022" → Klotz et al. 2022 (HESS 26:1673); the
  PDF claimed as 'Molinari 2019, NHESS 19:2565' is actually **Pacetti, Caporali & Rulli 2017
  (Adv. Water Resour. 110, 494–504)** while the NHESS DOI belongs to the separate Molinari
  2019 AGRIDE-c paper; DOI 10.3390/rs13112220 is Bai et al. (SAR fusion), Sen1Floods11 is
  **Bonafilia et al. 2020 (CVPRW)**; GloFAS = **Alfieri et al. 2013**; Grinsztay & Bengio
  title aligned to the arXiv record; Bentivoglio survey (HESS 26:4345) added.
  (b) **New, region-specific sources verified (arXiv/Crossref/OpenAlex):** Chol et al. 2026
  (Sudd exposure, J. Flood Risk Mgmt), Easton-Calabria et al. 2024 (anticipatory action,
  Disasters), Palmer et al. 2023 (E. African rainfall drivers, Nat. Rev. Earth Environ.),
  Moro et al. 2024 (SSHAP), Nevo et al. 2021 (operational ML flood forecasting,
  arXiv:2111.02780), Giezendanner et al. 2023 (CNN–LSTM inundation, arXiv:2305.00640),
  Sun et al. 2023 (FNO inundation, arXiv:2307.16090), STURM-Flood (Notarangelo et al. 2025),
  Ke et al. 2017 (LightGBM, NeurIPS proceedings), Chen & Guestrin 2016 (XGBoost).
  (c) **Hardware:** RTX 3090 verified against NVIDIA's official product page + CUDA-GPUs list
  (10,496 cores, 1.70 GHz, 24 GB GDDR6X / 936 GB/s, compute capability 8.6); PyPI/pytorch.org
  versions pinned (torch 2.14.0, torchvision 0.29.0, sklearn 1.9.1, lightgbm 4.7.0, xgboost
  3.4.1, catboost 1.2.10, shap 0.52.0, tabpfn 9.0.0, pytorch-lightning 2.6.6).
  (d) **Written:** corrected §4.2–§4.4, new §4.5 (regional context), rewritten §6 (hardware),
  and §9 Implementation plan (gated CPU→3090 phases, 60-s smoke test, runtime budgets,
  exit criteria).
- **2026-09-21** — Created this document. Swept arXiv + OpenAlex for: flood extent
  forecasting/mapping (DL), hydrology ML benchmarks (LSTM/GRU/SSM/transformer), tabular-ML
  state of the art (GBM, TabPFN, TabArena), impact-based forecasting, and the three `literature/`
  papers (all identified with DOIs). Checked the Aweil + national 2024 lag-correlation tables
  (negative vs positive sign split — §2) and the weekly forecast-signal series (zero-inflated
  spikes — §2). Environment: venv created with project requirements (Python 3.13.5, pandas
  3.0.3, numpy 2.4.6, xarray 2026.4.0); no GPU on this machine; scikit-learn/LightGBM not yet
  in `requirements.txt`.
