# TabPFN — Task A zero-shot cross-check

**Role:** the *independent* reference. TabPFN is a pre-trained transformer
that does in-context learning over your training set — no gradient
training, no hyperparameters, nothing tuned to this problem. Its place in
the ladder (per `MODEL_RESEARCH.md` §4.2): if a zero-shot model already
matches a tuned gradient-boosting model, the data is not carrying much
more signal; if the primary matches TabPFN's ceiling, the primary has
saturated what the features offer.

**Status: TESTED on real Aweil county data on 29 September 2026.**
TabPFN detected flood events well, but it was worse than persistence at
predicting the exact flooded area. The national run has not been done.

## What to test

1. **The zero-shot ceiling.** Compare test `skill_q50` / CRPS against the
   LightGBM primary:
   - TabPFN ≈ primary → the feature set is near-saturated; further tuning
     of the primary is unlikely to help; the honest next step is better
     features (or accept the negative result).
   - primary clearly > TabPFN → tuning/inductive bias is earning its keep
     on this data; the primary's quantile machinery is working.
   - TabPFN clearly > primary → the primary is mis-tuned (or the data is
     easy enough that defaults win); re-examine the primary before
     reporting it.
2. **Version-drift behaviour.** The API changed across TabPFN majors
   (2.x → 9.x); this folder probes for quantile support and degrades to
   point predictions (degenerate forecast, CRPS = |y − ŷ|) when absent.
   The *first line printed* on a run tells you which mode it used —
   record that in the research doc, because the comparability of the CRPS
   depends on it.
3. **Agreement on the spike slice** — same rule as the other cross-checks.

## How to run

```bash
# First time only. The model files are downloaded during the first run.
python -m pip install -r requirements-ml.txt

# Run the county-level baseline.
python -m modeling.tabpfn.train --scope aweil
python -m modeling.tabpfn.train --scope national
```

Each user needs their own Prior Labs account, accepted licence and API
key. Do not save an API key in the repository.

After the first run, the prepared features are loaded from a local
Parquet cache. This avoids repeating the slow data preparation.

No `--shap` (TabPFN is not a tree model; attribution is not available for
it — that's part of what makes it an independent check).

## Outputs (in `outputs/methods/tabpfn/`)

| file | what it is |
|---|---|
| `tables/task_a_tabpfn_predictions_<scope>.csv` | val+test predictions |
| `tables/task_a_tabpfn_metrics_<scope>.csv` | metric table incl. baselines, with role column |
| `tables/task_a_modelcard_<scope>.md` | card with automatic comparison vs the saved primary |

## Keep / kill verdict

| result | verdict |
|---|---|
| ran successfully and produced its numbers | **KILL after use** — this folder's value is the one comparison paragraph in `MODEL_RESEARCH.md` (zero-shot ceiling vs tuned primary). Record it, delete the folder. Its outputs live in the gitignored `outputs/methods/tabpfn/`. |
| version drift made quantiles unavailable | still usable (point-forecast mode) — note the mode in the research doc; same kill-after-use verdict |
| import/API broken beyond the probe | park it (leave the folder), note in the research log; do not let a version fight delay the ladder |

## Aweil result

TabPFN was useful as a flood detection baseline. Its ROC-AUC was 0.945
and it detected about 79 percent of the flood events. It was weaker at
predicting flooded area. Its test MAE was 3.895 square kilometres,
compared with 2.883 square kilometres for persistence. It also predicted
an average of 8.558 square kilometres during spike events, while the
real average was 40.217 square kilometres. This means that it strongly
underestimated the largest floods.

## Known failure modes

- TabPFN 9.0.0 may not expose the `output_type="quantiles"` API — the
  probe handles it, but record which mode ran.
- In-context learning scales with training-set size; at the national
  scope (~11k rows) the run is heavier (minutes) and may need the CPU
  machine rather than the GPU one (TabPFN doesn't use CUDA for the
  classifier by default in current versions).

## References

`MODEL_RESEARCH.md` §4.2, §5; the TabPFN paper (Hollmann et al. 2023,
arXiv:2209.05549) is in the References section there.
