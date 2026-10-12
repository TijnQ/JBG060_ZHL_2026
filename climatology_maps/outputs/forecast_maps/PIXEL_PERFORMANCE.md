# Pixel-level performance of the climatology-downscaled forecast maps

**Model under test:** LightGBM county forecast x per-pixel climatology (`forecast_maps.prob` = `det_prob x freq_given`; severity layer `sev` = climatology pattern scaled to the model's q50 area).

**Honest setup:** out-of-fold test weeks only (2010-11, 2016-17, 2022-24); the pixel climatology prior is rebuilt per fold on that fold's **training years only** (2000-2008 / 2000-2014 / 2000-2020), so neither the model probability nor the spatial prior ever saw a test year. Weeks without a satellite observation score as dry (censoring, conservative).

Scored: **211,518,048 pixel-weeks**, wet prevalence **0.05%** (112,500 wet).

## 1. Probabilistic pixel scores

| score | value |
|---|---|
| Brier, model map `prob` | 0.0005 |
| Brier, pure climatology `clim` | 0.0005 |
| Brier, persistence (last week's mask) | 0.0004 |
| **BSS vs climatology** | **+0.058** |
| BSS vs persistence | -0.086 |
| AUROC (model map) | 0.9214 |
| PR-AUC (model map) | 0.2118 |

Brier = mean((p - y)^2), lower is better. BSS = 1 - Brier_model / Brier_reference (>0 = skill over the reference).

## 2. Threshold classification (model map, pooled)

| threshold | accuracy | precision | recall | F1 | CSI | FAR | TNR |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.9976 | 0.1395 | 0.6767 | 0.2313 | 0.1307 | 0.8605 | 0.9978 |
| 0.10 | 0.9987 | 0.2070 | 0.4934 | 0.2916 | 0.1707 | 0.7930 | 0.9990 |
| 0.20 | 0.9993 | 0.2979 | 0.2278 | 0.2582 | 0.1482 | 0.7021 | 0.9997 |
| 0.30 | 0.9994 | 0.3554 | 0.1248 | 0.1848 | 0.1018 | 0.6446 | 0.9999 |
| 0.50 | 0.9995 | 0.4822 | 0.0354 | 0.0659 | 0.0341 | 0.5178 | 1.0000 |
| 0.70 | 0.9995 | 0.9037 | 0.0015 | 0.0030 | 0.0015 | 0.0963 | 1.0000 |

## 3. Weekly flooded area (km2, region-total per week)

| metric | value |
|---|---|
| MAE map-implied area (Σ p x km2) vs observed, all 362 weeks | 15.31 km2 |
| MAE (probability map) on flood weeks only (obs > 0, n=153) | 34.14 km2 |
| MAE severity layer (Σ sev x km2 ~ q50) on flood weeks | 18.51 km2 |
| bias (map-implied − observed), all weeks | -3.06 km2 |
| Pearson r, weekly implied vs observed area | 0.684 |
| R2 severity-layer weekly area | 0.647 |

## 4. Per-county breakdown (pooled over test weeks)

| county | Brier model | Brier climatology | BSS vs clim | wet pixel-weeks |
|---|---|---|---|---|
| Aweil Centre | 0.0001 | 0.0001 | +0.042 | 5,079 |
| Aweil East | 0.0010 | 0.0010 | +0.052 | 48,232 |
| Aweil North | 0.0001 | 0.0001 | +0.039 | 4,089 |
| Aweil South | 0.0025 | 0.0027 | +0.078 | 41,758 |
| Aweil West | 0.0004 | 0.0004 | +0.037 | 13,342 |

## 5. Monthly breakdown (pooled over test years)

| month | Brier model | Brier climatology | BSS vs clim |
|---|---|---|---|
| Jan | 0.0000 | 0.0000 | -0.046 |
| Feb | 0.0000 | 0.0000 | -0.039 |
| Mar | 0.0000 | 0.0000 | -0.128 |
| Apr | 0.0000 | 0.0000 | -0.142 |
| May | 0.0000 | 0.0000 | -0.023 |
| Jun | 0.0000 | 0.0000 | +0.303 |
| Jul | 0.0000 | 0.0000 | -0.004 |
| Aug | 0.0001 | 0.0001 | -0.019 |
| Sep | 0.0010 | 0.0011 | +0.036 |
| Oct | 0.0020 | 0.0021 | +0.042 |
| Nov | 0.0019 | 0.0021 | +0.096 |
| Dec | 0.0004 | 0.0004 | +0.046 |

## 6. Reliability (calibration of the model `prob`, pooled)

| predicted p bin | mean predicted | observed wet freq | n |
|---|---|---|---|
| 0.0-0.1 | 0.001 | 0.000 | 211,249,885 |
| 0.1-0.2 | 0.135 | 0.164 | 182,147 |
| 0.2-0.3 | 0.242 | 0.249 | 46,501 |
| 0.3-0.4 | 0.344 | 0.309 | 20,617 |
| 0.4-0.5 | 0.446 | 0.347 | 10,650 |
| 0.5-0.6 | 0.544 | 0.427 | 5,899 |
| 0.6-0.7 | 0.635 | 0.598 | 2,162 |
| 0.7-0.8 | 0.723 | 0.904 | 187 |

## Notes & limitations

* The labels are satellite detections; a week without a valid observation
  is zero-filled (censored), which biases scores conservatively — some
  'dry' pixels were unobserved, not truly dry.
* The `clim` reference uses the same (k+0.5)/(n+1) formula as the LightGBM
  protocol, rebuilt per fold on training years. The shipped gallery PNGs
  use the full 2000-2025 prior (explanatory maps), so this table is the
  honest out-of-sample evaluation.
* Threshold metrics depend on the cutoff; the headline ranking is the
  Brier family. Accuracy is dominated by the ~98% dry pixels.
* Regenerate with: `python -m climatology_maps.evaluate_pixel_maps` -> /home/boulshou/School/JBG060_ZHL_2026/climatology_maps/outputs/forecast_maps/PIXEL_PERFORMANCE.md
