# research-tijs.md — CatBoost & U-Net for the Aweil flood-forecasting data

**Author:** Tijs · **2026-09-22**
**Question investigated:** which type of model should we train to get the best results for the data we have? I was assigned the two prescribed candidates, one per task in the modeling ladder (`modeling/README.md`):

| Model | Task | Data type |
|---|---|---|
| **CatBoost** | A — weekly county flood area ("how much") | tabular: 5 Aweil counties × 2000–2025 |
| **U-Net** | B — weekly per-pixel flood presence ("where") | spatial: 40×40 grid of 0.25° (≈25 km) cells |

The two models are **not competitors** — they answer different questions. The real question is whether each is the right inductive bias for its own data.

**Scope note.** I was assigned **CatBoost and U-Net only**. LightGBM (Task A primary), XGBoost, RandomForest and TabPFN are other members' methods; they are mentioned here only where needed to state CatBoost's role in the repo's three-family vote (per `modeling/catboost/guide.md`), not as part of my investigation. All external sources below were verified on 2026-09-22 via the arXiv API, the Crossref API, and the CatBoost GitHub repo (see §6).

---

## 1. What the EDA says the models must cope with

From the committed EDA outputs (paths relative to the repo root):

- **Zero-inflated and spiky target (Task A).** ~1,356 Monday-start weeks in 2000–2025 × 5 counties ≈ 6.8k county-weeks, but only **1,457 non-zero county-weeks (~21%)**; detections cluster in Oct–Nov (peak month in 4 of 5 counties); 357 detection runs, longest 33 consecutive weeks. (`EDA_flood_masks/README.md`, `EDA_flood_masks/outputs/tables/`)
- **Weak-to-moderate daily signal.** Committed Aweil lag correlations (single year 2024, n≈366 days): river discharge r ≈ −0.45 at lag 0, decaying slowly; local rainfall r ≈ −0.13; runoff r ≈ −0.08. The committed nationwide tables are a **synthetic fallback**, and the county tables are single-year — the multi-year version is the gate-0 audit (`modeling/audit.py`), which has not yet been run on a machine with `raw_data/`. (`EDA_hydrometeorology/README.md`, `outputs_county/tables/aweil_lag_correlations.csv`)
- **Censored labels.** The flood product contains detection pixels from 3-day composites only; `cloud_frac` is zero in every record, so **a week with zero detections is not proof of dry land** (e.g. 2024 Jul–Sep: 26 of 92 days had any detected pixel, ~28%). 26-year record with partial 2000/2025. (`EDA_flood_masks/README.md`, `data_quality/eda_quality_flood/outputs/`)
- **Small tabular sample (Task A).** ~6.8k rows × **50 embargoed features** (rolling ERA5 precipitation/runoff local + upstream band, one Dartmouth gauge 100205, Lake Albert altimetry, AgERA5 ET/net moisture, the county's own 1–4 week lags, calendar). (`modeling/features.py`, `modeling/splits.py`)
- **Small spatial sample (Task B).** ≤1,236 weekly tensors of 40×40×17 (14 daily channels: 7 days × {tp, ro} + 3 static: crop, rangeland, log(cattle+1)) → binary per-cell label. The same flood event repeats over consecutive weeks, so the **effective independent sample is far smaller than the number of weeks**. (`modeling/unet/train.py`, `modeling/unet/guide.md`)
- **Hardware.** CatBoost: CPU, minutes (`modeling/catboost/guide.md`). U-Net: RTX 3090; torch is not installed on the dev machine and the code has **never been executed** (`modeling/unet/guide.md`).

---

## 2. CatBoost (Task A)

### 2.1 What it is

CatBoost is an implementation of gradient-boosted decision trees (GBDT; Friedman 2001 [S4]) whose two signature ideas are **ordered boosting** (a permutation-driven alternative to the classic update rule) and **native handling of categorical features**. Both were designed to fight *prediction shift* — a special kind of target leakage that the NeurIPS paper shows is present in all earlier GBDT implementations and which hurts test-time quality [S1]. The open-source library adds GPU training and fast CPU scoring [S2]; the developers claim superior quality over other GBDT libraries on many datasets and publish quality/speed benchmarks against XGBoost, LightGBM and H2O [S6, S7].

In our repo it is the **third tree family** in the Task A panel (`modeling/catboost/`): same 50 embargoed features, same purged temporal split (train 2000–2014 / val 2015–2019 / test 2020–2025), a 3-quantile regression head for CRPS evaluation plus a detection-probability head, evaluated against persistence/climatology/last-detection baselines.

### 2.2 Advantages (for our data)

1. **Explicit anti-overfitting design.** Ordered boosting directly targets the prediction shift that makes standard boosting overfit small samples [S1]. Our training set is small (~6.8k rows, ~1.4k non-zero) — exactly the regime where naive GBDT overfits. This is the main argument for CatBoost over a generic GBDT here.
2. **Tabular-native.** No spatial or sequential machinery is needed; numeric features with NaN (gauge gaps are forward-filled then left NaN) are handled natively; the county-week table maps 1:1 onto the model.
3. **Native categorical handling.** County IDs and flood-class flags can be used without hand-rolled target encodings, which are a classic leakage source in small samples [S1, S6].
4. **A third independent family for the three-family vote.** CatBoost's defined role in the repo is the third vote alongside the (other members') LightGBM primary and XGBoost cross-check: 3/3 beats persistence on the spike slice = strong evidence, 1/3 = one model overfit, distrust all three, 0/3 = no signal in this feature set (`modeling/catboost/guide.md`). An independent implementation catches implementation-specific artifacts.
5. **Documented quality claims vs other GBDT libraries** on standard benchmark sets [S1, S6, S7].
6. **Probabilistic path exists.** CatBoostLSS extends CatBoost to full conditional distributions (quantiles, prediction intervals) [S3] — a route if the Task A quantile head matures into a real distributional forecast. (Our current implementation uses per-quantile losses instead.)
7. **Practical.** Trains in minutes on CPU at our size; fast CPU scoring and GPU training available if ever needed [S2, S6].

### 2.3 Disadvantages (for our data)

1. **It cannot manufacture missing signal.** If the gate-0 audit or the test split shows the embargoed ERA5/gauge features carry little power, CatBoost collapses to a persistence-like zero. The model is not the bottleneck; the honest comparison is against persistence/climatology, which is exactly what `modeling/baselines.py` and the keep/kill rules implement.
2. **Still a high-capacity ensemble.** Ordered boosting mitigates but does not eliminate overfitting; with ~21% non-zero weeks a boosting model can degenerate to "always 0". The spike-slice skill test is the guard against this (`modeling/catboost/guide.md`).
3. **Not the fastest GBDT on CPU.** The developers' own benchmarks repo compares CPU training/applier speed against LightGBM and XGBoost [S7], and the guide notes CatBoost is "a bit slower than LightGBM at these sizes". At our scale this is minutes either way; it is a reason LightGBM was chosen as primary by the other members, not a reason against CatBoost as cross-check.
4. **Not spatial.** A tree ensemble cannot exploit the 2D structure of the floodplain — it is useless for Task B. (A fit boundary, not a defect.)
5. **Black box.** Interpretation needs post-hoc attribution (we use TreeSHAP via `modeling/shap_report.py`); the advisory layer (`modeling/advisory.py`) must translate the numbers into plain text for the 2024 case study.
6. **API/version drift.** The `Quantile:alpha` objective + `quantile=` constructor spelling is version-specific (1.2.x); a future major can break `build_models` (`modeling/catboost/guide.md`, known failure mode).
7. **Flagship features underused.** GPU training is wasted on our tiny CPU job, and the famous categorical machinery is barely exercised because nearly all 50 features are numeric.

---

## 3. U-Net (Task B)

### 3.1 What it is

U-Net is a symmetric encoder–decoder convolutional network with skip connections: the contracting path captures context, the mirroring expanding path restores resolution, and skip links give precise localization [S8]. It was designed to be trained **end-to-end from very few images** (170 annotated electron-microscopy stacks in the original paper) by heavy data augmentation, and it won the ISBI segmentation challenges of 2015/2016 [S8]. It became the workhorse of modern image segmentation (UNet++ redesigned its skip pathways [S9]) and is used specifically for **flood mapping/detection**: bi-temporal Sentinel-1 U-Net variants outperform prior state of the art on the Sen1Flood11 benchmark [S10], U-Net-based Sentinel-1/2 flood mapping [S11], and elevation-guided flood-extent mapping [S12]. For *prediction* of gridded hydrological fields (rather than segmentation of an observation), related deep spatial forecasters such as Fourier-Neural-Operator surrogates for hydrodynamic inundation [S13] show that gridded flood fields are learnable by spatial networks.

Our implementation (`modeling/unet/train.py`): a small **4-block U-Net** (16/32/64/128 channels, bottleneck 5×5, ~1 M parameters) takes 17 channels (7 days × {tp, ro} + 3 static land channels) on the 40×40 grid of HEC/USGS tile h20v08 and outputs a binary flood probability per cell for the target week. Anti-leakage is enforced the same way as Task A (features dated ≤ m−3). Evaluation: CSI/POD/FAR against a **threshold-calibrated climatology "null" map** (each pixel's historical flood frequency), with the same train 2000–2014 / val 2015–2019 / test 2020–2025 split.

### 3.2 Advantages (for our data)

1. **Spatial inductive bias.** Convolution shares weights across space, so the network exploits the fact that flooding is spatially structured (Bahr el Ghazal channels, floodplain, dry ridges) — something a per-pixel GBDT cannot do at all [S8, S10, S11].
2. **Proven at small-sample scale.** Unlike modern large CNNs, U-Net was explicitly designed to work from very few labelled images [S8]. Our effective sample (≤1,236 weekly maps, temporally autocorrelated) is small, which is the strongest argument for a small U-Net over deeper/wider alternatives here.
3. **Full-resolution output with precise localization** (skip connections) [S8] — exactly what Task B needs: *where* in the floodplain, at the target week.
4. **Per-pixel probability output** → thresholdable, directly comparable to the calibrated climatology map (the null), and usable as a risk statement in the advisory layer.
5. **Static covariates come free.** Crop/rangeland fractions and cattle density are constant input channels — direct support for the food-security angle of the project.
6. **Empirically competitive for flood mapping.** U-Net-family methods outperform prior state of the art on the Sen1Flood11 flood-detection benchmark (+6% IoU for the attentive siamese variant) [S10]; U-Net is the standard architecture in recent flood-mapping work [S11, S12].
7. **Cheap to run at our size.** 40×40×17 → 40×40 with ~1 M parameters fits the RTX 3090 with wide margin; minutes per run (`modeling/unet/guide.md`).

### 3.3 Disadvantages (for our data)

1. **Overfitting is the #1 threat.** Deep CNNs generally need large labelled data sets; we have ≤1,236 weekly "images" and consecutive weeks share the same flood events, so the effective independent sample is much smaller. Mitigations: small architecture, early stopping (patience 5), and the decisive test being *beating the climatology null*, not the training loss (`modeling/unet/guide.md`). Note that the original paper's small-data recipe relied on **spatial augmentation**, which we cannot safely apply to weekly time-series weeks (it would fabricate data) — our mitigations are weaker.
2. **Censored labels.** A zero-detection week can simply be a missed observation (`cloud_frac` = 0 in every record; 2024 Jul–Sep had detections on only 26 of 92 days). The network sees "dry" most of the time and will learn it; it cannot distinguish unobserved from dry. The per-pixel formulation **amplifies** Task A's censoring problem.
3. **Extreme per-pixel class imbalance.** Most cells in most weeks are dry → BCE needs `pos_weight`, which we cap near ~100 because unbounded weights make the loss unstable, and CSI comparisons only make sense with threshold calibration (`modeling/unet/guide.md`, known failure mode).
4. **Coarse output.** 250 m flood detections are aggregated into 25 km (0.25°) cells; the result is a "which part of the floodplain" map, not a 250 m hazard map. A resolution limit of our setup, not of U-Net itself.
5. **No interpretability.** No SHAP for CNN weights; the practical checks are visual (the mean probability map must put high probability on the Bahr el Ghazal / Aweil East–West channels, not on dry ridges) plus agreement with climatology (`modeling/unet/guide.md`).
6. **GPU/CUDA dependency and first-run risk.** torch (CUDA 12.8 wheel) must be installed; the code has never been executed on any machine, so OOM/NaN/API-break at the smoke test is a realistic gate failure, not science (`modeling/unet/guide.md`).
7. **No distributional output.** BCE yields one probability, not quantiles; the CRPS-style evaluation of Task A does not apply (we use CSI/POD/FAR vs the null).
8. **Does not answer "how much".** Task B complements Task A; a U-Net cannot replace the county-area model.

---

## 4. Fit verdict

| | **CatBoost (Task A)** | **U-Net (Task B)** |
|---|---|---|
| Data type | tabular county-week table | 2D weekly pixel map |
| Right inductive bias? | Yes — trees are the standard for small tabular sets; ordered boosting targets our small-sample risk [S1] | Yes — spatial autocorrelation is exactly what convolution exploits [S8, S10] |
| Main risk | the features carry too little signal (not a model flaw) | overfitting + censored labels (a model×data interaction) |
| Decisive test | beat **persistence** on the spike slice, test 2020–2025 | beat the **calibrated climatology null** on CSI (test) *and* a visually sane map |
| Hardware | CPU, minutes | RTX 3090, minutes |

**Bottom line.** For *the data we have*, the "which model is optimal" question splits in two and both prescribed models are the right family for their task: **CatBoost** (a GBDT with ordered boosting, chosen specifically for its small-sample robustness and as an independent third tree family) for the small, zero-inflated, tabular Task A; **a small U-Net** for the spatial Task B, where the win/loss is defined strictly against the climatology null. Neither model beats its null on paper — that is exactly what the keep/kill rules in `modeling/catboost/guide.md` and `modeling/unet/guide.md` exist to decide once the real data run happens.

---

## 5. How the sources were verified (2026-09-22)

- **arXiv API** (`export.arxiv.org/api/query`, `id_list=...`): titles, abstracts, authors, dates for [S1], [S2], [S3], [S8], [S9], [S10], [S12], [S13].
- **Crossref API** (`api.crossref.org`): DOIs/metadata for [S4] (10.1214/aos/1013203451), [S5] (10.1145/2939672.2939785), [S8] MICCAI version (10.1007/978-3-319-24574-4_28), [S11] (10.13053/cys-29-1-5539).
- **GitHub raw** (`raw.githubusercontent.com`): official CatBoost README (claimed advantages, doc links) and the CatBoost benchmarks README (comparison scope) for [S6], [S7].
- **Repo-internal facts** (EDA numbers, split/embargo, architecture details, failure modes): `EDA_flood_masks/README.md`, `EDA_hydrometeorology/README.md`, `data_quality/`, `modeling/README.md`, `modeling/features.py`, `modeling/splits.py`, `modeling/unet/train.py`, `modeling/unet/guide.md`, `modeling/catboost/guide.md` — read directly, no external source.

---

## 6. Sources

- **[S1]** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* NeurIPS 2018. arXiv:1706.09516. — ordered boosting; prediction shift (target-leakage-style overfitting) analysis; categorical-feature algorithm; quality claims vs other boosting implementations.
- **[S2]** Dorogush, A. V., Ershov, V., Gulin, A., et al. (2018). *CatBoost: gradient boosting with categorical features support.* arXiv:1810.11363. — open-source library; GPU training; CPU scoring faster than other GBDT libraries at equal ensemble size.
- **[S3]** März, A. (2020). *CatBoostLSS — An extension of CatBoost to probabilistic forecasting.* arXiv:2001.02121. — full conditional distributions (mean/location/scale/shape); quantiles and prediction intervals from a parametric family.
- **[S4]** Friedman, J. (2001). *Greedy function approximation: a gradient boosting machine.* The Annals of Statistics, 29(5), 1189–1232. DOI: 10.1214/aos/1013203451. — the GBDT foundation all three tree families descend from.
- **[S5]** Chen, T., Guestrin, C. (2016). *XGBoost: a scalable tree boosting system.* KDD '16. DOI: 10.1145/2939672.2939785. — sibling GBDT family; context for the three-family vote.
- **[S6]** CatBoost project (2026, accessed 2026-09-22). *README, github.com/catboost/catboost.* — stated advantages: quality vs other GBDT libraries on many datasets; best-in-class prediction speed; numerical + categorical features; fast GPU/multi-GPU training; visualization; Spark/CLI distributed training.
- **[S7]** CatBoost project (2026, accessed 2026-09-22). *Benchmarks repo, github.com/catboost/benchmarks.* — CatBoost vs XGBoost vs LightGBM vs H2O: quality, training speed (CPU/GPU), model-evaluation speed, SHAP speed.
- **[S8]** Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: convolutional networks for biomedical image segmentation.* MICCAI 2015, pp. 234–241. arXiv:1505.04597; DOI: 10.1007/978-3-319-24574-4_28. — contracting/expanding path with skip connections; trainable end-to-end from very few images via data augmentation; ISBI challenge wins.
- **[S9]** Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., Liang, J. (2018). *UNet++: a nested U-Net architecture for medical image segmentation.* arXiv:1807.10165. — deeply supervised encoder–decoder with nested dense skip pathways; reduces the encoder/decoder semantic gap.
- **[S10]** Yadav, R., Nascetti, A., Ban, Y. (2022). *Attentive Dual Stream Siamese U-net for flood detection on multi-temporal Sentinel-1 data.* arXiv:2204.09387. — U-Net-family flood detection on the Sen1Flood11 benchmark; +6% IoU over prior (uni-temporal) state of the art.
- **[S11]** Pech May, F., Álvarez Cárdenas, O., Ríos Toledo, G. (2025). *Flood mapping through Sentinel-1, Sentinel-2 imagery and U-NET deep learning model.* Computación y Sistemas, 29(1). DOI: 10.13053/cys-29-1-5539.
- **[S12]** Sami, M. T., Yan, D., Adhikari, S., et al. (2024). *EvaNet: elevation-guided flood extent mapping on Earth imagery (extended version).* arXiv:2404.17917.
- **[S13]** Sun, A. Y., Li, Z., Lee, W., Huang, Q., Scanlon, B. R., Dawson, C. (2023). *Rapid flood inundation forecast using Fourier neural operator.* arXiv:2307.16090. — deep spatial surrogate for hydrodynamic inundation forecasting (Houston case study).
