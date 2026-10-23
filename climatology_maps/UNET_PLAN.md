# U-Net Implementation Plan — per-pixel weekly flood maps (Aweil)

> **Status: plan only — nothing implemented yet.**
> This document is the agreed design for the third mapping stage. The folder
> currently ships two working stages:
> `climatology_maps.py` (historical county baseline) and
> `forecast_maps.py` + `pixel_climatology.py` (LightGBM county forecasts
> downscaled with the pixel climatology). The U-Net below is the next stage,
> compared against those two as references.

---

## 0. Is a U-Net even worth it? (hypothesis, with numbers)

**Short answer:** worth a *bounded* experiment, not a foregone win. Expect a
small/noisy pooled gain over the LightGBM×climatology map, with a real chance
of winning where climatology is weakest — heavy anomaly years and 1-week
(spatial) persistence.

Dataset-size facts that drive the hypothesis:

| Fact | Number | Implication |
|---|---|---|
| Calendar weeks 2000–2025 | 1,357 | training material is weekly samples |
| Flood weeks in region | 567 (42 %) | rare event → class imbalance (~5 % of pixels ever wet) |
| **Distinct flood events** | **357** (~2.7 / county / yr; mean 2.5–6.4 wks) | adjacent weeks are near-copies → effective sample ≈ events, not weeks |
| Distinct wet pixels | 28,925 of 584,304 (5 %) | small positive class; most pixels never flood |
| County-level LightGBM (fe) | Brier 0.065, **BSS vs clim 0.35**, vs persist 0.22, flood-week MAE 8.7 km² | learnable signal exists at county level but is modest and currently thrown away spatially |
| Inter-annual variability (Oct–Nov area, CV) | 1.4–1.9 per county | heavy vs light years differ ~2× — the regime where static priors fail most |

Why a U-Net may NOT beat climatology disagg:
* **Small effective sample size** (≈357 events, autocorrelated, dominated by
  2022–24) vs millions of parameters → the easy local optimum is "memorize the
  monthly wetland map", i.e. rediscover what `freq_given` already encodes,
  plus overfitting the OOF test years.
* **Placement is terrain-driven** and the climatology pattern already captures
  it; the learnable spatial *residual* (deviation given this week's forcing)
  is small and noisy.
* **No elevation / distance-to-river in the dataset** — the single most
  informative static feature is missing, so the net would rediscover the
  wetland map with one hand tied.

Why it is still worth trying:
* The county model's **BSS 0.35 proves learnable signal exists**, and a U-Net
  can express something climatology structurally cannot: "upstream rain this
  week → wet area expands corridor-wise".
* Heavy years (CV 1.4–1.9 (std/mean) of annual flooded area) grow far beyond
  the historical core — **the tails are where a learned expansion model beats
  any static prior**, and those are the years that matter most.
* **Duration/persistence is the county pipeline's weakest spot** (h1–h4 heads
  barely beat references); a U-Net fed last week's mask + this week's forcing
  has a clear, cheap win there.
* Cost of finding out is small: coarse-grid U-Net ≈ 1–2 GPU-hours on the
  3090; the OOF protocol and map renderers already exist.

Prior on the final verdict: **pooled BSS ≈ +0.0–0.1, heavy-year BSS ≈
+0.1–0.25; keep/kill should be decided on the anomaly split, not the pooled
number.** Two ablations that maximize information per GPU-hour: (1) report
BSS pooled *and* split into heavy vs normal years; (2) train with and without
last-week's mask as input, to isolate short-lead spatial persistence value.
Optional but high-value: add a free SRTM DEM (~90 m) as an input channel.

---

## 1. Goal & position in the pipeline

LightGBM predicts one number per county per week (`det_prob`, `q50/q90` area);
the climatology disagg (`forecast_maps.py`) places that county forecast on the
~232 m grid using the historical pattern. The U-Net instead learns a **spatial
model**: it takes raster inputs and outputs a **per-pixel flood probability
field** for a target week directly.

| Stage | Granularity | Where |
|---|---|---|
| LightGBM (county) | county/week | `LightGBM_v1/` |
| Climatology disagg | pixel/week | this folder, working |
| **U-Net** | pixel/week | this folder, **planned** |

**Keep/kill rule (same as the rest of the repo):** the U-Net is kept only if it
beats both references — the pure climatology map and the
LightGBM×climatology map — on the out-of-fold protocol metrics (§5). Otherwise
the climatology disagg stays and the U-Net is dropped.

---

## 2. Data & preprocessing

### 2.1 Labels (target)
- Raw masks: `raw_data/flood_masks/compact_{recurring,unusual}/flood_events_h20v08_YYYY.parquet`
  (2000–2025), deduped per `(week, lat, lon)`, grouped into Monday weeks
  (`W-SUN` periods — identical to `pixel_climatology.load_pixel_detections`).
- Binary per-pixel wet/dry mask per week; pixel area from latitude
  (`pixel_area_km2`, ~0.053 km²/pixel).
- Censoring caveat: weeks without a valid satellite observation are zero-filled
  (same convention as the EDA and LightGBM labels) — a conservative bias, not
  proof of dry land.

### 2.2 Input channels (all embargoed: dated ≤ Friday of the week before target Monday)
| Channel | Source | Form |
|---|---|---|
| Past flood masks (lag 1–4 wks) | flood parquets | raster (the "memory", mirrors `y_true_lag*`) |
| ERA5 precipitation/runoff rolling sums (1/3/7/14/30 d) | `raw_data/rainfall and runoff/ERA5_*.nc` | spatial fields resampled to the model grid |
| Gauge discharge 100205, Lake Albert level, ET0 | Dartmouth / DAHITI / processed ET | scalar inputs (broadcast) or FiLM-style conditioning |
| Climatology prior | `pixel_climatology` cache | raster channel(s): `freq`, `freq_given` for the target month |
| Calendar | — | month & week-of-year sin/cos (scalar) |
| Optional statics | `raw_data/farmland/*.tif`, worldpop, county ID | raster or one-hot |

Design choice: feeding the **climatology as an input channel** makes the net
learn *deviations from history* instead of rediscovering it — and guarantees a
sensble map when training data is thin.

### 2.3 Grid & resolution
- v1: **coarse grid** over the whole Aweil bbox (~2–5 km cells → ~40×40 to
  100×100 px per week). Fast, cheap, good for validating the protocol.
- v2: **native 232 m** with sliding crops (e.g. 128×128 px patches) to fit GPU
  memory; loss computed only inside county polygons.

### 2.4 Train/val/test
Identical double to the LightGBM protocol so results are directly comparable:
- 3 expanding-window folds (`config.CV_FOLDS`): train 2000–2008/2014/2020,
  val 2009/2015/2021, test 2010–11 / 2016–17 / 2022–24.
- 3-day Friday embargo, first week of each val/test window purged, 2025 loaded
  but never scored.

---

## 3. Architecture & training

- **U-Net**: depth 3–4, base channels 32–64, BN + ReLU, max-pool or strided
  conv encoder, concatenated skip connections, decoder up to full res, final
  1×1 conv → single logit per pixel (sigmoid = P(wet)).
- **Loss**: weighted BCE + Dice (weights from the pixel class balance; floods
  are ~1–3 % of all pixels). Focal loss as drop-in alternative.
- **Optimizer**: AdamW, lr 1e-3 with cosine decay, batch 16–32, mixed precision
  (AMP), early stopping on validation Brier, ~50–100 epochs.
- **Data**: pyarrow filtered reads → cached `torch.Tensor` dataset; augment
  with H/V flips and 90° rotations (flood geometry is not rotation-invariant
  w.r.t. river flow, so keep rotations optional).
- **Area head**: v1 has no second head — expected flooded km² per county =
  Σ per-pixel `p · pixel_area`. Quantile scenarios (q10/q90) can come later
  via thresholds on p.

---

## 4. Implementation layout (future files)

```
climatology_maps/unet/
├── dataset.py            # parquet -> tensor dataset (embargoed, cached)
├── model.py              # U-Net definition
├── train.py              # folds, losses, AMP, early stopping, checkpointing
├── evaluate.py           # protocol metrics + references + keep/kill verdict
└── outputs/              # tables/figures (gitignored, like LightGBM outputs)
```

Maps reuse the existing rendering: feed OOF test-week probability masks into
`climatology_maps.forecast_maps.render_week_png` /
`render_interactive_html` and the `week_summary.csv` audit — identical output
format to the climatology disagg, so the two are visually diffable.

---

## 5. Evaluation protocol

- **Per-pixel**: Brier score, reliability diagram, and **Brier skill score vs
  three references** — (a) pure climatology map (`clim_det_prob × freq_given`),
  (b) LightGBM×climatology disagg map (current `forecast_maps` `prob` layer),
  (c) persistence (last observed mask).
- **Regional**: weekly flooded-km² MAE/bias per county and month; CSI/F1 at
  decision thresholds (0.2/0.5/0.8) — the cutoff family is secondary here, as
  in the LightGBM protocol.
- **Outputs**: per-fold + pooled comparison table
  `model × (brier, bss_clim, bss_lgbm, area_mae, csi)`, reliability tables,
  maps + summary CSV.

---

## 6. Milestones

| # | Task | Machine | Effort |
|---|---|---|---|
| M1 | Dataset loader + tensor cache (embargoed, 3 folds) | any | 1 session |
| M2 | Coarse-grid U-Net baseline train/val | RTX 3090 | 1–2 sessions, hours on GPU |
| M3 | Evaluation harness vs climatology & LightGBM maps + map outputs | any | 1 session |
| M4 | Native 232 m patched training + hyperparameter sweep (Optuna) | RTX 3090 | 2 sessions |
| M5 | Report: tables, figures, keep/kill verdict | any | 1 session |

Total ≈ 6 sessions. The RTX 3090 (24 GB) is overkill for v1 (coarse grid could
even train on the laptop) and comfortably fits v2.

---

## 7. Risks & mitigations

- **Class imbalance** (few wet pixels) → weighted BCE/Dice or focal loss.
- **Censored labels** (cloud gaps as zero) → same conservative bias as all
  prior stages; document in provenance.
- **Few historical flood events** → climatology input channel, augmentation,
  early stopping, honest OOF scoring.
- **Boundary effects** at the bbox edges → mask loss to counties only.
- **Overfitting on a small study area** → rely on the expanding-window test
  years (2010–11/16–17/22–24) and the pooled protocol numbers.
- **Missing static drivers**: the zip has farmland/rangeland/cattle/worldpop
  but no elevation or distance-to-river raster (`Data_overview.xlsx` check).
  Elevation would likely add skill — optional future channel.

---

## 8. Deliverables

1. `climatology_maps/unet/` code + gitignored outputs.
2. Comparison CSV: U-Net vs climatology vs LightGBM×climatology, pooled + per fold.
3. Weekly probability PNGs + interactive HTML + `week_summary.csv` via the
   existing renderers.
4. A one-page verdict: keep or kill, with the numbers that justify it.