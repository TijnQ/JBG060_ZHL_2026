# Random Forest — Task A cross-check (new, added 2026-09-21)

**Role:** the "no tuning, no gradient" reference in the Task A panel.
Random Forests have no native quantile regression, so this method
forecasts a **degenerate** 3-quantile forecast (q10 = q50 = q90 = the
point prediction) plus a detection classifier. CRPS of that degenerate
forecast is exactly `|y - ŷ|`, so every number in its table is directly
comparable with the primary's.

**Status: UNTESTED on real data.**

## What to test

1. **The tuning premium.** Compare this method's test CRPS/skill against
   the LightGBM primary's: the gap between a degenerate RF forecast and a
   proper quantile LGBM forecast measures what calibration is worth on
   this data. A tiny gap suggests the primary's quantiles aren't buying
   much (the signal is mostly in the mean, not the spread) — a useful
   finding in itself.
2. **The tuning gap (overfitting canary).** RF with default-ish settings
   and no early stopping is a weak but stable model. If a *tuned,
   early-stopped* primary beats persistence but this *untuned* forest
   doesn't, the primary's win may be tuning artefact — flag it and
   double-check val/test agreement in the primary's card before trusting
   the headline.
3. **Feature-importance second opinion.**
   `tables/task_a_importances_<scope>.csv` (the model's own
   importances) vs the SHAP plot: do they rank the same features?
   Agreement = confidence in the attribution story for the report.

## How to run

```bash
# AFTER the primary (card compares against the saved primary run):
.venv/bin/python -m modeling.randomforest.train --scope aweil
.venv/bin/python -m modeling.randomforest.train --scope national
```

`--shap` works (SHAP's TreeExplainer handles sklearn forests).

## Keep / kill verdict

| result | verdict |
|---|---|
| beats persistence on the spike slice (unlikely for an untuned forest, but possible on a strong signal) | **KEEP** as evidence; consider in the three-family vote |
| doesn't beat persistence, but the primary does | expected and fine — it is doing its job as the overfitting canary and the tuning-premium gauge; **delete the folder** once the comparison is recorded in `MODEL_RESEARCH.md` (outputs stay in `outputs/methods/randomforest/`, gitignored) |
| beats the primary by > 0.05 test-all skill | **surprise — investigate before believing** (an untuned forest beating a tuned quantile model means the quantile machinery is fighting the signal); treat as a promotion candidate only after re-running with a larger forest and checking stability |

## Expected runtime / hardware

Seconds to a couple of minutes on CPU (400 trees, ≤ 1.4k train rows per
county-week scope — trivial).

## Known failure modes

- sklearn API: `feature_importances_`, `predict_proba` are stable APIs —
  low risk. `max_features=0.3` (fraction) is valid for regressors and
  classifiers in current sklearn.
- The degenerate forecast makes its CRPS a *lower bound on what a
  point-forecaster can achieve* — never read its CRPS as "this is what
  the mean of the primary would give"; it's a separate model.

## References

`MODEL_RESEARCH.md` §4.2, §5; `modeling/metrics.py` (CRPS of a degenerate
distribution = absolute error, covered by the known-answer suite).
