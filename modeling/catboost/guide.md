# CatBoost — Task A cross-check

**Role:** third tree family (ordered boosting) in the Task A cross-check
panel. Same contract as `xgboost/`: agreement with the LightGBM primary =
robustness; a > 0.05 test-all skill advantage = promotion candidate.

**Status: UNTESTED on real data.**

## What to test

1. **Agreement with the primary** (same comparison as the XGBoost guide,
   against `outputs/methods/lightgbm/tables/task_a_metrics_*.csv`).
2. **The three-family vote.** Once lightgbm + xgboost + catboost have all
   run, the verdict table is: how many of the three beat persistence on
   the spike slice? 3/3 = strong evidence; 1/3 = one model overfit,
   distrust all three; 0/3 = no signal in this feature set (the negative
   result to report).
3. **Promotion rule** (same 0.05 margin vs the primary, automatic in the
   model card).

## How to run

```bash
# AFTER the primary:
.venv/bin/python -m modeling.catboost.train --scope aweil
.venv/bin/python -m modeling.catboost.train --scope national
```

## Keep / kill verdict

| result | verdict |
|---|---|
| part of a 3/3 (or 2/3 agreeing with the primary) positive spike result | **KEEP** only as evidence in the research doc; the folder itself can be deleted once the doc records the three-family table (the outputs live in `outputs/methods/catboost/`, which is gitignored) |
| promotion candidate (> 0.05 over primary) | discuss as new backbone (see `xgboost/guide.md`) |
| isolated outlier vs the other two families | **KILL** — delete the folder; its only value was as a vote |

## Expected runtime / hardware

Minutes on CPU (`task_type="CPU"` is set; CatBoost is a bit slower than
LightGBM at these sizes).

## Known failure modes

- CatBoost API drift: `Quantile:alpha` + `quantile=` constructor kwarg is
  the current (1.2.x) spelling; a future major may rename it — fix
  `build_models` here.
- `predict_proba` returns columns in class order; the pipeline takes
  `[:, 1]` — with only 0/1 labels that's safe.

## References

`MODEL_RESEARCH.md` §4.2, §5.
