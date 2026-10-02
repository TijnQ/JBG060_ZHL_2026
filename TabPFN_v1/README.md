# TabPFN using the LightGBM v3 protocol

Run from the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe -u LightGBM_v1/run.py --model tabpfn
```

This executes the same runner as LightGBM: corrected real-data loaders,
50 feature columns, incomplete-row filtering, three expanding-window folds,
boundary purging, training-only baselines, duration targets, metrics and
advisories. 2025 remains outside CV. Synthetic fallback stays disabled.
Outputs go to `TabPFN_v1/outputs/outputs_v3/tables/`.

Only the model backend differs. A pretrained TabPFN regressor supplies genuine
q10/q50/q90 area quantiles, and TabPFN classifiers supply detection and four
duration probabilities. Seed 2026 is shared (duration uses seed + horizon).
All training rows are used, with no subsampling. Validation rows are held out;
TabPFN does not use LightGBM's boosting early stopping. TabPFN defaults are
otherwise retained. Quantile failures stop the run rather than substituting
point forecasts. The previous `redundant/modeling/tabpfn` outputs are not reused.

LightGBM remains the default: `python LightGBM_v1/run.py`.
Compare runs from the same commit and real-data files. Existing committed
LightGBM outputs may predate the latest loader fixes and should be regenerated
before reporting a comparison.

## Completed run: 2 October 2026

Run using repository base commit `3f49df9` plus the shared-runner/model-backend
changes, TabPFN 9.0.0 with the v3.5 checkpoint, and CUDA on an RTX 3060 Laptop
GPU. The fixed loader produced 6,120 complete rows; 665 incomplete rows from
2000–2002 were dropped by the shared pipeline. All six sources were real.

All three folds completed: 515, 515 and 780 test rows, respectively (1,810
total; 374 flood weeks). All 13 output tables were saved. Output checks
confirmed unique county/week/fold keys, exact test years, finite probabilities
in [0,1], nonnegative ordered quantiles, and recomputed summary metrics.
Four automated checks passed, including a shared-runner comparison with mock
predictions and checks of training boundaries and duration targets. A real
TabPFN inference smoke check also passed.

| Pooled metric | TabPFN |
|---|---:|
| Detection Brier score | 0.0624 |
| Brier skill vs climatology | 0.3737 |
| Brier skill vs persistence | 0.2478 |
| Mean pinball loss | 0.771 |
| Flood-week interval coverage | 0.666 |
| Flood-week MAE (km²) | 12.28 |
| Flood-week bias (km²) | -11.46 |

Detection improves on both shared references. Flood area remains
underestimated: the median forecast's flood-week MAE exceeds persistence's
8.82 km². These are results under the new protocol and replace the old
TabPFN evaluation; a comparison with LightGBM needs its corresponding
fixed-loader run.
