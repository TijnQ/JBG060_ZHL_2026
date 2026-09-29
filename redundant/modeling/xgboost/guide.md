# XGBoost — Task A cross-check

**Role:** cross-check of the LightGBM primary (see `lightgbm/guide.md` for
the question the whole Task A ladder answers). XGBoost is the second tree
family; if it reaches the same verdict as LightGBM, the primary's result
is robust across optimisers.

**Status: UNTESTED on real data.**

## What to test

1. **Agreement with the primary.** Compare `tables/task_a_metrics_*.csv`
   here against `outputs/methods/lightgbm/tables/task_a_metrics_*.csv`:
   - same verdict (both beat persistence / both don't) → the primary is
     robust; record that in `MODEL_RESEARCH.md` and **delete this folder**
     (its job is done — the research doc keeps the one-line result).
   - different verdict on the spike slice → investigate which is
     overfitting before trusting either (check the val-vs-test gap in
     both cards).
2. **Promotion rule.** If this method beats the primary's test-all
   `skill_q50` by **more than 0.05**, flag it as the new backbone
   candidate (the model card does this comparison automatically when the
   primary's metrics file exists).

## How to run

```bash
# AFTER the primary (the card compares against the saved primary run):
.venv/bin/python -m modeling.xgboost.train --scope aweil
.venv/bin/python -m modeling.xgboost.train --scope national
```

`--shap` works too (TreeExplainer handles XGBoost boosters).

## Keep / kill verdict

| result | verdict |
|---|---|
| beats primary by > 0.05 test-all skill | **PROMOTION CANDIDATE** — discuss in the research doc; if confirmed on national scope as well, it becomes the new primary (swap the two folders' roles in `config.TASK_A_METHODS`) |
| same verdict as primary, no meaningful uplift | **KILL** — job done (robustness confirmed); delete the folder, note in `MODEL_RESEARCH.md` |
| worse than primary everywhere | **KILL** — delete the folder; nothing to learn beyond "primary stands" |

This folder is by design disposable: cross-checks exist to be deleted.

## Expected runtime / hardware

Minutes on CPU; same data volume as the primary.

## Known failure modes

- XGBoost 3.x `quantile_alpha` kwarg placement (it is a constructor
  parameter in current versions; if a future version moves it, fix
  `build_models` here).
- A quantile forecast with q10 > q90 (miscalibrated alpha) would make
  CRPS meaningless — the known-answer suite in `check_modeling.py`
  guards the metric side; the model side would show up as a negative
  "CSI-like" pattern and is a kill signal in itself.

## References

`MODEL_RESEARCH.md` §4.2 (cross-check logic), §5 (validation protocol).
