# Model Research Notes — Flood Forecasting & Impact Advisory

**Status:** exploratory research only — **no models implemented yet**. This is a living document:
update it as experiments proceed (keep dated entries so the reasoning trail stays visible).

- 2026-09-21: initial research sweep (web literature + project EDA outputs); added TL;DR at top. See "Research log" at the bottom.

## TL;DR — the recommended approach (2026-09-21)

**Build it as three stages; start with the cheap one.**

| Stage | What to build | Model | Framework / hardware |
|---|---|---|---|
| **A. County flood extent** (7-day ahead, per Aweil county, weekly) | tabular features (rolling ERA5 rain/runoff, gauge, lakes, ET0, calendar, lagged own extent) → next week's detected km² + detection probability | **LightGBM quantile regression** (primary), XGBoost/CatBoost as cross-checks, **TabPFN** as zero-shot baseline, climatology + persistence as the bar to beat; GRU/Mamba only if a temporal residual remains | scikit-learn + LightGBM, **CPU**, minutes per run (add to `requirements.txt` when we start) |
| **B. Sub-county "where"** (250 m risk map) | ERA5 tile stack (40×40 at 0.25°) + terrain/ag channels → 250 m flood mask, 7-day ahead | **small 2D U-Net** (probabilistic output; physics-informed U-Net+FNO / PIML variants only if it plateaus) | **PyTorch 2.x + CUDA on the RTX 3090** (24 GB fits it easily; expect hours of training) |
| **C. Impact → advisory** (crops / cattle / what to do) | predicted extent × EDA exposure math × agricultural phase → exposed ha / cattle / escalating advisory text | **deterministic post-processing first — no learning** (IPC 2022–2025 is too short to train on; use it for case studies only) | pandas + the EDA modules we already have |

**Why this ladder:** our Aweil table is small (~1.4k non-zero county-weeks out of ~10.5k) and zero-inflated — the literature (NeurIPS 2022, TabArena 2025, Kratzert HESS 2018) says gradient boosting is the strongest first move on exactly this kind of data, while the GPU only earns its keep on the spatial task where there is genuine structure to exploit. Deep learning on *tabular* A is not justified yet; it gets promoted only if stage A shows a measurable temporal residual.

**Non-negotiables before any of it:** (1) recompute all lag correlations over **2015–2025** — the 2024-only Aweil results are negative and weak and must not leak into feature selection (§2, §5.3); (2) strict temporal splits (2000–2014 / 2015–2019 / 2020–2025) with a **3-day embargo** around the composite labels; (3) every model is scored **against climatology + persistence**, including spike-week metrics (POD/FAR/CSI), not just overall R².

**Bottom line in one sentence:** LightGBM quantile regression on weekly county features (CPU) as the backbone forecast, a small PyTorch U-Net on the 3090 for the spatial "where", and a rule-based exposure→advisory layer on top — with a multi-year feature audit as gate 0.

## 1. The question we are answering

1. **Where will the next flood be?** — forecast flood extent ~7 days ahead (the project's target
   lead time from the hydrometeorology EDA) for the five Aweil counties and, where feasible,
   sub-county "where" on the 250 m MCDWD grid.
2. **What do crops and cattle need to protect livelihoods?** — turn a flood forecast into impact
   estimates (hectares of crop / rangeland at risk, cattle exposed, phase of the agricultural
   cycle) and into plain-language advisories (e.g., "harvest early", "move herds to higher
   ground this month").

## 2. What the project EDA tells us (data reality check)

Facts gathered from `EDA_flood_masks/` and `EDA_hydrometeorology/` outputs (all verified
against the committed CSVs on 2026-09-21):

| Fact | Value / consequence |
|---|---|
| Target label | MCDWD 3-day composite detections, 2000–2025, ~250 m points. **Label = a 3-day window, not an instant.** Consequences: (a) a model predicting "week t" must not use any feature dated later than week t's window minus the 3-day composite overlap, (b) consecutive weeks' labels overlap in time → validation needs an **embargo/gap** (see §5). |
| Aweil volume | 972,588 pixel records, 1,457 county-weeks with detections, 357 detection runs, max run 33 weeks, peak month Oct (Nov for Aweil South), max weekly area 375 km² (Aweil East). |
| National volume | 58,053,688 records, 40,904 county-weeks, 8,476 runs (longest 279 weeks). |
| **Signal quality, Aweil 2024** | Lag correlations vs. daily flood pixels: local rainfall −0.13, local runoff −0.08, upstream rainfall −0.13, upstream runoff −0.04, **river discharge −0.45** (all at lag 0, decaying slowly to ~−0.30 at lag 14). |
| **Signal quality, nationwide 2024** | Same signals, **all positive and much stronger**: rainfall +0.56, runoff +0.51, upstream +0.64/+0.63, discharge **+0.81**. |
| Interpretation | The Aweil county-level result is a **single year** of data with a **negative** correlation — almost certainly a 2024-specific / gauge-location artifact: the only gauge (area ID 100205, 9.6°N 31.6°E) sits downstream on the Sudd/Nile side, so "discharge rising" there plausibly tracks water *draining away from* Bahr el Ghazal after the flood plain recedes. **Action: recompute all lag correlations over 2015–2025 (multi-year) before trusting any feature.** Do not build on the 2024-only sign. |
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
- Grinsztay & Bengio, *Why do tree-based models still outperform deep learning on typical
  tabular data?* (NeurIPS 2022; arXiv:2207.08815): on typical tabular data, GBMs remain ahead
  of deep nets, especially with limited data. Our Aweil target has only ~1,400 non-zero
  county-weeks out of ~10,500 possible — that is "limited data".
- TabArena, *A Living Benchmark for ML on Tabular Data* (2025; arXiv:2506.16791): tree-based
  models still anchor the state of the art for tabular prediction, with deep models catching up
  on large-data regimes we do not have.
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
current row/feature limits at experiment time and license (Prior Labs, commercial; v2+ is the
open-weights line).

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
the advisory layer can say "80% chance of >X km²" instead of a point. Bech et al. 2022
(HESS 26:1673, DOI 10.5194/hess-26-1673-2022) is a solid reference for calibrating DL
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
  line of work; see Voss et al. 2021, Remote Sensing 13:2220, DOI 10.3390/rs13112220).
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
not crisp extent.

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
   (NHESS 2024, DOI 10.5194/nhess-24-199-2024 — the paper already in `literature/`) and the
   repo's *Floods and food insecurity* (Molinari et al. 2019, NHESS 19:2565,
  https://nhess.copernicus.org/articles/19/2565/2019/). For the mobility/accessibility angle:
   *Assessing road criticality and loss of healthcare accessibility during floods* (Int J Health
   Geographics 2022, DOI 10.1186/s12942-022-00315-2 — also in `literature/`), which motivates
   using the OSM road network loader we already have.
4. Operational context: GloFAS (*GloFAS – global ensemble streamflow forecasting and flood early
   warning*, HESS 2013, DOI 10.5194/hess-17-1161-2013) is the reference **operational** system;
   ours is a regional, extent-based, impact-focused complement — worth citing in the report as
   the benchmark of "what operational looks like" (ensemble hydrological models, 2–14 day lead,
   river discharge focus — *not* extent).

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

- **Python 3.12+** (project standard; this session runs 3.13.5).
- **Tabular stack:** scikit-learn (preprocessing, time-series split utilities) + **LightGBM**
  (primary; native quantile loss, fast) with XGBoost/CatBoost as cross-checks. **Note:**
  `requirements.txt` currently pins none of these — add `scikit-learn`, `lightgbm` (and
  optionally `xgboost`, `catboost`, `shap`) when implementation starts.
- **Spatial/GPU stack:** PyTorch 2.x (CUDA 12 build) — the RTX 3090 (24 GB GDDR6X, ~35.6
  TFLOPS FP32, sm_86) handles small/mid CNNs (U-Net-class, ~10⁶–10⁷ params) comfortably at
  batch sizes 16–64; expect full training of the Task-B U-Net to be a **few hours**, not days.
  Plain PyTorch or Lightning, our call — prefer whichever the group is more comfortable with.
- **Reuse, don't rewrite:** `processing_data/loading.py` and `loading_impact_data.py` already
  cover all raw inputs; the EDA modules already contain the exposure-overlap math.
- **Interpretability:** SHAP for GBM feature attributions (report which of rain/runoff/gauge/
  season actually drove the forecast) — cheap, and required for a defensible course report.
- **No GPU in this session** (no `nvidia-smi` here) — all development of Task B should be
  written but only *smoke-tested* locally; real training happens on the 3090 machine.

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

## 8. Proposed experiment ladder (for when data lands — nothing built yet)

1. **E1 data check & audit** — `raw_data` completeness report; multi-year (2015–2025) lag
   correlations, county + national; coverage statistics. *(cheap, CPU, day-scale)*
2. **E2 baselines** — climatology, persistence, last-composite nowcast; score on the E5
   protocol. *(CPU, hours)*
3. **E3 GBM Task A** — LightGBM quantile regression on county-week features (Aweil first,
   then national); SHAP report; compare vs E2. *(CPU, hours)*
4. **E4 TabPFN zero-shot** — same table, no tuning; check whether the foundation model beats
   E3 on this small, non-i.i.d. table. *(GPU/CPU, hours)*
5. **E5 protocol hardening** — embargoed walk-forward, spike-weighted metrics, gauge
   sensitivity. *(CPU, days)*
6. **E6 U-Net Task B** — tile-level 7-day-ahead mask; CSI/POD/FAR vs spatial-prior null;
   train on the 3090. *(GPU, days)*
7. **E7 advisory layer** — deterministic forecast→exposure→advisory pipeline + one worked
   2024 case study (e.g., an October week in Aweil East). *(CPU, days)*

## References (all verified 2026-09-21 via arXiv API / OpenAlex)

- Kratzert F. et al. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM)
  networks. *HESS* 22, 6005. DOI 10.5194/hess-22-6005-2018
- Kratzert F. et al. (2019). Towards Learning Universal, Regional, and Local Hydrological
  Behaviors via Machine-Learning Applied to Large-Sample Datasets. arXiv:1907.08456
- Kratzert F. et al. (2019). NeuralHydrology — Interpreting LSTMs in Hydrology. arXiv:1903.07903
- Bech B. et al. (2022). Uncertainty estimation with deep learning for rainfall–runoff
  modeling. *HESS* 26, 1673. DOI 10.5194/hess-26-1673-2022
- (2023). Caravan — A global community dataset for large-sample hydrology. *Scientific Data*.
  DOI 10.1038/s41597-023-01975-w
- Grinsztay L., Bengio S. (2022). Why do tree-based models still outperform deep learning on
  typical tabular data? *NeurIPS 2022*. arXiv:2207.08815
- (2025). TabArena: A Living Benchmark for Machine Learning on Tabular Data. arXiv:2506.16791
- Prior Labs (2026). TabPFN-3.5: Technical Report. arXiv:2609.17895
- (2025). RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting.
  arXiv:2505.22535
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
- Voss M. et al. (2021). Enhancement of Detecting Permanent Water and Temporary Water in Flood
  Disasters ... Sen1Floods11. *Remote Sensing* 13(11):2220. DOI 10.3390/rs13112220
- (2025). A Comprehensive Survey on Deep Learning Solutions for 3D Flood Mapping. arXiv:2506.13201
- (2026). Prithvi-Complimentary Adaptive Fusion Encoder (CAFE) for flood inundation mapping.
  arXiv:2601.02315
- (2024). Impact-based flood forecasting in the Greater Horn of Africa. *NHESS* 24, 199.
  DOI 10.5194/nhess-24-199-2024 *(in `literature/`)*
- Molinari D. et al. (2019). Floods and food insecurity — a method to estimate the effect of
  inundation on crop availability. *NHESS* 19, 2565. *(in `literature/`)*
- (2022). Assessing road criticality and loss of healthcare accessibility during floods:
  Cyclone Idai, Mozambique 2019. *Int J Health Geographics*. DOI 10.1186/s12942-022-00315-2
  *(in `literature/`)*
- Van Enk L. et al. (2013). GloFAS – global ensemble streamflow forecasting and flood early
  warning. *HESS* 17, 1161. DOI 10.5194/hess-17-1161-2013
- NASA (2025). MCDWD/VCDWD User Guide Rev F. *(in `literature/`)*
- de Prado M. (2018). *Advances in Financial Machine Learning* — purged/embargoed cross-
  validation (adapted here to overlapping 3-day composite labels).

## Research log

- **2026-09-21** — Created this document. Swept arXiv + OpenAlex for: flood extent
  forecasting/mapping (DL), hydrology ML benchmarks (LSTM/GRU/SSM/transformer), tabular-ML
  state of the art (GBM, TabPFN, TabArena), impact-based forecasting, and the three `literature/`
  papers (all identified with DOIs). Checked the Aweil + national 2024 lag-correlation tables
  (negative vs positive sign split — §2) and the weekly forecast-signal series (zero-inflated
  spikes — §2). Environment: venv created with project requirements (Python 3.13.5, pandas
  3.0.3, numpy 2.4.6, xarray 2026.4.0); no GPU on this machine; scikit-learn/LightGBM not yet
  in `requirements.txt`.
