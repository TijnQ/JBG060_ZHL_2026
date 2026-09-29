# Evaluation Metrics — Task 1 (County-level model)

> Companion to [`model_plan.md`](model_plan.md). Covers **Task 1 / Task A only**: the county-level
> LightGBM that predicts, per Aweil county and per week, (1) the probability of a *detected* flood,
> (2) the flood area in km² as q10/q50/q90, and (3) the duration (still flooded 1–4 weeks ahead).
> Task B (the spatial map) has its own metrics (CSI/POD/FAR vs the climatology null) — not covered here.
> Status: **protocol fixed on paper before any model run** (to be confirmed in the teacher feedback session).

---

## 1. What we evaluate

| Model output | Grain | Feeds into |
|---|---|---|
| `det_prob` — P(flood detected in county this week) | county-week | headline of the bulletin, exposure weighting |
| `q10 / q50 / q90` — flood area in km² | county-week | advisory **tiers** and exposed-hectare calculation |
| Duration — P(still flooded in 1/2/3/4 weeks) | county-week × horizon | `movement_advice.csv` (where to move to) |

We evaluate at **two levels**: the model outputs (1–3, technical metrics) and the final advice
(4, stakeholder metrics). Both must be reported.

## 2. Evaluation protocol (fixed before any run)

- **Walk-forward time-series cross-validation.** Train on everything up to a cutoff, validate on the
  *next complete flood season*, roll the cutoff forward — every season serves as validation exactly once.
- **Expanding vs sliding training window**: open decision (for the teacher session). Expanding = more
  data per fold, assumes a roughly stationary process; sliding = adapts to regime change, less data per fold.
- **Embargo**: features must be dated at most *Friday of the previous week* (`splits.feature_cutoff`);
  one boundary week is purged per fold so label windows cannot overlap (`splits.purge_boundary_weeks`).
- **Baselines are re-estimated on every training fold** — never on validation/test data.
- Optional (teacher decision): a **final untouched holdout** (e.g. 2020–2025) scored exactly once after
  all tuning is frozen.

---

## 3. Metrics per output

### 3.1 Detection probability — `det_prob`

**Brier score** — the mean squared error of the probabilities:

```
Brier = (1/N) · Σ (p_i − y_i)²        y_i = 1 if a flood was detected that week, else 0
```

Lower is better, 0 is perfect. It grades *honest confidence*: correctly saying 90% scores better than
correctly saying 60%, and wrongly saying 90% is punished hard. We do **not** use raw accuracy as the
headline: accuracy needs a yes/no cutoff and throws the confidence away.

**Reliability diagram** — "can the number be trusted?" Group all predictions by stated probability
(all weeks said 60–70%, all said 20–30%, …) and plot how often flooding *actually* happened per group.
Points on the diagonal = honest; below the diagonal = overconfident (tiers would trigger too often).
This is where the censoring problem becomes visible: under-prediction clustered in peak-cloud weeks
is the missed-observation signal (see §7).

**Brier Skill Score (BSS)** — the same Brier score expressed against a reference forecast:

```
BSS = (Brier_ref − Brier_model) / Brier_ref      (= 1 − Brier_model / Brier_ref)
```

Positive = real added value, 0 = no better than the reference, negative = worse. We require skill
against **both** reference forecasts (§4): *climatology* (the calendar) and *persistence* (last week's
state). Beating both is our evidence that the weather/gauge features carry information beyond the
calendar and the latest observation.

### 3.2 Flood area — `q10 / q50 / q90`

The area numbers are load-bearing: the tiers compare predicted area to county-month climatology
thresholds, and the exposure estimate multiplies area by cattle density. An unvalidated area number
would silently drive the tier. Three metrics, all compared against the **county-month climatological
median area** as reference:

1. **Pinball loss** (quantile loss), averaged over q ∈ {0.1, 0.5, 0.9} — the standard scoring rule
   for quantile forecasts: `L_q(y, ŷ) = max(q·(y − ŷ), (q−1)·(y − ŷ))`
2. **Interval coverage** — how often the observed area falls inside the predicted q10–q90 band.
   Nominal target ≈ 80%. Reported on **flood weeks** as the headline (across all weeks the trivial
   zeros make coverage near-automatic), with the all-weeks number next to it.
3. **MAE and bias on flood weeks only** (`y_true > 0`). Across all weeks the error looks beautifully
   tiny because ~79% of county-weeks are ≈ 0 km²; the honest error figure is conditional on flooding.
   Bias matters specifically: systematically under-predicting area = systematically under-tiering.

### 3.3 Duration (1–4 week horizons)

Same probabilistic metrics as §3.1 — Brier score + reliability diagram — computed **per horizon**
(1, 2, 3, 4 weeks ahead), one short line each in the report. No new machinery.


---

## 4. The two reference forecasts (baselines)

Principle: **a baseline must predict the same kind of thing as the model** — probabilities to be
scored with the Brier score, km² to be scored with pinball/MAE. Every learned model must beat these
before it earns a place in the report (same keep/kill rule as the `baselines.py` docstring).

### 4.1 Climatology — "what usually happens here, at this time of year"

A forecast that never looks at weather, only at the calendar. It exists because our floods are
strongly seasonal; a model could score decently just by learning "October = flood season".
Climatology is the bar that proves the weather matters. (In `model_plan.md` the word "climatology"
is used in three senses: the per-cell flood-frequency map for Task B, the county-month *area
thresholds* behind the tiers, and this baseline forecast. Same idea — historical frequencies — but
different objects; we name them separately in the report.)

**Probability version (for det_prob):** for each (county, month), the fraction of *training-window*
weeks with a detection:

```
P(det | Aweil East, October) = (# October weeks with a detection) / (# October weeks)
```

≈ 5 counties × 12 months = 60 numbers. Dry-season months land near 0.00–0.05, peak months (Oct–Nov)
well above the ~21% overall base rate. A tiny smoothing `(k + 0.5) / (n + 1)` avoids literal 0.0
predictions, which would be overconfident given censoring. Fallback chain: (county, month) → county
mean → overall mean.

**Week-of-year variant (open, cheap to test):** since we forecast weekly, a weekly climatology is
the finer reference — but 52 bins × 5 counties = 260 cells estimated from only ~26 observations each
is noisy, and a *noisy* baseline is a *weak* floor (beating it proves little). Options, in increasing
sophistication: (a) month-of-year (default), (b) week-of-year smoothed over ±2 neighboring weeks or
shrunk toward the month mean, (c) a per-county harmonic curve (logistic regression on sin/cos of
week-of-year — smooth annual cycle, still zero weather input). We build (a) and one smoothed weekly
variant, compare **on validation folds only**, and adopt the stronger as the official baseline.
The tier thresholds stay at month resolution — different object, different job.

**Area version (for q50 / pinball / MAE):** the median observed area per (county, month) on the
training window — median, not mean, because the target is spiky. (Note: `baselines.py` currently
uses the *mean*; we switch the skill baseline to the median or state the choice explicitly.
The mean/p80/p95 remain in use as *tier thresholds* — that is intentional and separate.)

### 4.2 Persistence — "last week's state continues"

**Probability version (for det_prob):** the naive version — predict 1 if last week had a detection,
else 0 — is a trap: a hard 0/1 forecast has Brier = its own error rate, so the comparison would be
rigged in our favor. The fair version estimates how likely continuation actually is, on the training
window:

```
p_on  = P(det this week | det last week)      # continuation rate, e.g. ~0.6
p_off = P(det this week | no det last week)   # e.g. ~0.1–0.15
```

Forecast for week *t*: `p_on` if `y_det_lag1 = 1`, else `p_off`. Still pure persistence — it uses
*only* the previous week's observation, no weather — but it outputs calibrated probabilities.
(In the report: *first-order Markov persistence*.) Inputs already exist: `features.py` builds
`y_det_lag1..4`. Optional refinement: condition on run length (k consecutive detected weeks so far)
instead of the binary state — only if the simple version looks too weak.

**Area version:** the observed area of the previous week (`y_true.shift(1)` per county) — already
implemented in `persistence_baseline()`. A third implemented baseline, **last-detection** (most
recent non-zero area within 4 weeks), stands in for a "last satellite composite" nowcast.

### 4.3 Why the comparison is fair

1. **Climatology is estimated on the training window only and recomputed per fold.** Letting it peek
   at validation weeks would leak the answer into the baseline.
2. **Persistence may use the last training week at a fold boundary** — in operations that observation
   is genuinely available, so it is legal, not leakage. The embargo guarantees the lag-1 label is
   dated before the target week's feature cutoff.
3. **First-week fallback:** a county's first week has no lag-1; fall back to that county's
   climatological probability (never to 0).
4. **Identical scoring:** model and baselines go through the same metrics functions, and both face
   the same imperfect (censored) labels — label noise cannot systematically favour LightGBM.

---

## 5. What counts as success

- **det_prob:** BSS > 0 against *both* baselines, stable across folds, reliability diagram roughly
  on the diagonal (checked also on the clear-weeks slice).
- **Area:** pinball loss below the climatological-median reference; interval coverage ≈ 80%;
  flood-week MAE/bias materially better than the reference.
- **Realistic pattern to expect:** performance ≈ persistence during long flood runs (those weeks are
  inherently easy), with the model's added value concentrated at **onset** — upstream rain and the
  river gauge rise before the floodplain fills. That is a good result, not a disappointing one.
- **Weekly grain is the right grain:** it is the native aggregation of the 3-day composites (a weekly
  union only needs one clear composite day), it matches the decision cadence of moving cattle, and it
  does not promise village-level timing.

