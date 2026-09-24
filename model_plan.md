# Model Plan - Predict Floods, Move the Cattle (Northern Bahr el Ghazal)

> Plain-language summary of the modeling discussion. Status: **plan only - no code changed yet.**
> Project: JBG060 / ZHL, working dir `JBG060_ZHL_2026`.

---

## 1. The goal

**Predict where floods will happen in Northern Bahr el Ghazal (NBeG), and use that to tell stakeholder ZHL (a) where cattle are in danger and (b) where the cattle should be moved to.**

That is really two questions in one:

1. **"How bad will it be, and when?"** - a risk number per area per week.
2. **"Where exactly on the map, and where should the cattle go?"** - a map plus a movement recommendation.

The plan answers question 1 at **county level**, question 2 at **sub-county level**, and turns both into advice with two **rule-based** layers.

---

## 2. The proposed plan: two model levels + two advice layers

### Level 1 - County level, "Task A" (LightGBM) -> "How bad?"

- **What it predicts** for each of the 5 Aweil counties and each week:
  - `det_prob`: probability that ANY flood is detected in the county that week;
  - `q10 / q50 / q90`: flood area in km2 (pessimistic / typical / bad case);
  - (planned) **duration** via multi-horizon classifiers: will the flood still be there in 1, 2, 3, 4 weeks? Duration labels already exist (`detected_flood_events.csv`, `national_detection_runs.csv`).
- **What it reads:** rain and river runoff (ERA5, over the county plus a 3-degree upstream band), the river gauge, lake level, evapotranspiration, the county's own flood history, calendar effects. Weekly data 2000-2025 (~6.8k county-week rows, ~21% of them with flood).
- **Output:** `advisories.csv` with risk tier 0-3 per county-week, exposed hectares, cattle counts.

### Level 2 - Sub-county, "Task B" -> "Where exactly?"

- **What it predicts:** a weekly flood map on a ~25 km grid (0.25-degree cells, ~700 km2 each): per cell, the probability that the cell floods that week. A county spans roughly 3-16 of these cells (Aweil Centre ~16, East ~9, North ~9, West ~7, South ~3).
- **Two candidates, tested against each other:**
  1. **U-Net** - a neural network that reads a picture-like stack of maps (last 7 days of local rain/runoff + static crop/rangeland/cattle layers). Written in `modeling/unet/`,
  2. **Per-cell LightGBM** - the same table model as Task A, but one row per cell per week. Cheaper, runs on a laptop, and unlike the U-Net it CAN read the gauge/lake signals.
- **The baseline both must beat:** **climatology** = a static map of "where does it USUALLY flood" (per-cell historical flood frequency, 2000-2025).

### How the two levels connect (the fusion recipe)

```
Task A says:  Aweil East - 83% flood chance, ~12 km2 expected.
Task B says:  inside that county, THESE cells are most likely flooded.
Step 1:       spread the 12 km2 across the county's cells using Task B's
              pattern (normalize cell probabilities so they sum to Task A's area)
Step 2:       multiply by the cattle-density map
Result:       "this many cattle likely standing in water, in these spots"
```

**GOLDEN RULE: Task A decides WHETHER and HOW MUCH; Task B only decides WHERE.**
Task B never contradicts Task A. Its map is always presented as: "IF the county floods (83%), the water is most likely HERE." Cell probabilities are used as relative weights only - never cut into a binary wet/dry verdict.

### Advice layers (rule-based, deliberately not ML)

- **`advisories.csv`** - tier per county-week: compare the predicted area to the **county-month climatology thresholds** (above the month's mean -> tier 1, above p80 -> tier 2, above p95 -> tier 3, else tier 0). Rule-based on purpose: IPC food-security data covers only 2022-2025 - too short and too tangled with other crises to train ML on.
- **`movement_advice.csv` (identified gap - still to build)** - *where to move to:* rank destination areas by their flood risk over the next 1-4 weeks (from Task A's duration/horizon predictions) x available rangeland x accessibility. Nothing in the repo does this yet.
- **ZHL bulletin** - a one-pager: the weekly map + `advisories.csv` + `movement_advice.csv`.

---

## 3. Why do we need both levels? (the dummy-proof version)

**The weather-service analogy.** The national forecast says "heavy rain over the whole province tomorrow" - one big, trustworthy headline (= Task A). The street map says "the underpass floods, the hill stays dry" (= Task B). You need both: a street map with no headline cannot say how bad it gets; a headline with no map cannot tell you where to walk the cows.

1. **Different data grains.** The strong signals (river gauge, lake level, upstream rain) are few and coarse - they explain a county total, not a 25-km map. The local signals (last week's local rain + land shape) exist per cell but are weak.
2. **Different sample sizes.** County-weeks: ~6.8k rows, ~21% flooded - learnable. Single cells: mostly dry, far sparser - much harder to learn from.
3. **Risk isolation (the safety net).** If the fancy map model fails its test, the county numbers still stand and the bulletin still ships - with the climatology map doing the "where".
4. **The two models would disagree anyway - so we design for it.** Example: the river rises because of rain that fell hundreds of km upstream. Task A sees the gauge -> "flood coming". The U-Net only sees local rain -> "looks dry". Both are right given the information they have. The fix is the golden rule (Task A owns the verdict), not a better model.


---

## 4. Alternatives we discussed - what works and what doesn't

| Alternative | Verdict | Why (plain language) |
|---|---|---|
| Only county LightGBM, no map | PARTLY WORKS | You know "Aweil East at risk" but not which rangeland; cattle advice stays vague. Still a complete fallback product on its own. |
| Only U-Net, no county model | DOES NOT WORK | The U-Net cannot see the river gauge or lake (its inputs are local rain + static land only) and cannot give a calibrated "how many km2". It answers "where", never "how much". |
| One per-cell LightGBM for everything (drop the county model) | DOES NOT WORK | Adding up 10 cell predictions does not give a reliable county total (the median of a sum is NOT the sum of the medians). You lose the trustworthy headline number, and cell-level data is the sparsest we have. |
| U-Net map shown as an independent second opinion (thresholded wet/dry) | DOES NOT WORK | Two maps that can contradict each other (see 3.4) confuse ZHL. Use it only as the conditional "where" layer. |
| One single model that does everything | NOT NOW | Different inputs, different grain, different sample sizes; no consistency guarantee. Two specialists plus strict rules beat one generalist we cannot check. |
| ML for the advice tier (learn tiers from IPC data) | DOES NOT WORK | IPC covers only 2022-2025; too short and confounded. Rule-based thresholds instead. |
| Forecast at 250 m (field level) | IMPOSSIBLE | Labels exist at 250 m, but weather predictors only at ~25 km. Below that, neighboring places share identical weather info - you would be predicting from geography alone (that is climatology, not forecasting). |
| Climatology-only product | WORKS AS THE FLOOR | For a slow, repeating seasonal floodplain, "where it usually floods" is already a decent map. It is the guaranteed fallback every model must beat. |
| TabPFN as a zero-shot baseline (Task A) | USE AS BASELINE | A pre-trained transformer that never trains on our data: it reads the training table like a prompt and predicts with zero tuning. Fully independent of LightGBM, so any agreement is meaningful: TabPFN ~ LightGBM means our features are saturated (no more signal to squeeze out); LightGBM clearly winning means the tuning earns its keep. Run once, record one comparison paragraph, then delete the folder per its guide. |

### The decision ladder (what to actually run, in order)

1. **Gate-0 audit** (`modeling/audit.py`): does county-level flood signal exist across years at all? Everything depends on this.
2. **County LightGBM + climatology map**: already a complete, honest product; no GPU needed.
3. **Add per-cell LightGBM**: cheap CPU experiment; doubles as backup if the U-Net fails.
4. **Run the U-Net on the GPU**: keep it only if it allocates better than both climatology AND the per-cell model, measured on flood weeks only (conditional skill), not on all weeks.
5. **Ship whichever passes**: the output format is identical; only a "spatial model used" column changes (unet / per-cell-lgbm / climatology).

---

## 5. What the weekly product for ZHL looks like

- **Headline per county:** tier 0-3, det_prob, q50 km2, expected duration (1-4 weeks).
- **Map:** 25-km cells colored by flood likelihood GIVEN the county flood, with expected cattle per cell overlaid.
- **Tables:** `advisories.csv` (where cattle is in danger) + `movement_advice.csv` (where to go).
- **Every map states its fallback honestly:** "spatial allocation: U-Net / per-cell model / climatology".

## 6. Known limits (honesty box)

- Zero satellite detections does NOT mean dry land: `cloud_frac` is always 0 in this data, so zero-detection weeks are censored (we cannot distinguish "no flood" from "no observation").
- Cattle and rangeland maps are static: wording stays "potential exposure", not exact animal counts.
- 25 km grid = "which part of the floodplain", never village-level.
- Heavy class imbalance: ~21% of county-weeks have any flood; per-cell far sparser still.

## 7. Notes and housekeeping from the discussion

- Add a small consistency monitor: log how often Task A and Task B disagree, and which allocator was used each week.
- Two independent models, one product: they train separately and only meet in the fusion step; never let Task B output override Task A.
- "Climatology", defined: (1) the per-cell historical flood-frequency map (Task B null baseline + fallback allocator); (2) the county-month area thresholds (mean / p80 / p95) used for the tiers; (3) the guaranteed floor: "what normally happens here, at this time of year".
- TabPFN as the zero-shot baseline for Task A (already coded in `modeling/tabpfn/`, unrun): pre-trained on millions of synthetic tables, it never trains on ours - no tuning, no SHAP, fully independent of LightGBM, which makes it an honest judge of how much signal our features carry. Run it AFTER the primary; its model card auto-compares against the saved LightGBM run. Watch-outs: ~1-3 GB weight download on first install, API version drift across TabPFN majors (the script auto-probes whether quantile output is available - record which mode ran), and it gets slow on big tables (national scope ~11k rows may need the CPU machine). Per its guide: run once, record the comparison in the research doc, then delete the folder.
