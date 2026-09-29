# modeling/ — the model-development testing ground

This package exists to **test which models actually work** on the Aweil
flood-forecasting problem — and the code is disposable by design. Each
candidate model lives in its **own folder with a `guide.md`**; a method
that fails its keep/kill test gets **deleted wholesale** (one folder, one
`git rm`), while its evidence (a few numbers, one sentence) is recorded
in `MODEL_RESEARCH.md` at the repo root. Nothing here may fall back to
synthetic data.

The research context, hypotheses and the definition of what counts as
"tested" live in [`MODEL_RESEARCH.md`](../MODEL_RESEARCH.md) (sections 4,
5 and the "Status of the modeling code" section). This README is the map.

## The methods (the unit of testing)

| folder | task | role | what it tests | status (2026-09-21) |
|---|---|---|---|---|
| [`lightgbm/`](lightgbm/guide.md) | A — county weekly flood area | **primary backbone** | does *any* model beat persistence on spike weeks | code done, **untested** on real data |
| [`xgboost/`](xgboost/guide.md) | A | cross-check | does a 2nd tree family reach the same verdict | code done, **untested** |
| [`catboost/`](catboost/guide.md) | A | cross-check | 3rd tree family; the 3-family vote | code done, **untested** |
| [`randomforest/`](randomforest/guide.md) | A | cross-check (untuned) | tuning premium + overfitting canary; degenerate point forecast (CRPS = \|y−ŷ\|) | code done, **untested** |
| [`tabpfn/`](tabpfn/guide.md) | A | zero-shot reference | the no-training ceiling for this feature set | code done, **untested** (package never installed here) |
| [`unet/`](unet/guide.md) | B — pixel map of the floodplain | only Task B method | does a spatial model beat the climatology ("null") map | code done, **never executed** (no torch on this machine) |

Task C (advisory) is **not** a competing model — it is the deterministic
post-processing layer that turns Task A predictions into guidance text.
It stays a module: [`advisory.py`](advisory.py).

## Shared core (not per-method — do not duplicate into folders)

| module | what it owns |
|---|---|
| [`config.py`](config.py) | all paths/constants, the method registry (`TASK_A_METHODS`, `TASK_B_METHODS`), per-method output dirs |
| [`features.py`](features.py) | the 50 embargoed weekly features + county/week labels (the one feature set all Task A methods share) |
| [`splits.py`](splits.py) | weekly calendar, 3-day label embargo, purged temporal 3-way split |
| [`metrics.py`](metrics.py) | CRPS (linear 3-quantile CDF), POD/FAR/CSI, spike/dry slicing |
| [`baselines.py`](baselines.py) | persistence / climatology / last-detection — the bar every method must beat |
| [`methods_common.py`](methods_common.py) | the Task A pipeline: fit contract, prediction frames, metrics table, model cards, per-method output writing |
| [`audit.py`](audit.py) | **gate 0**: multi-year (2015–2025) feature/lag audit — run this *before* any method |
| [`data_check.py`](data_check.py) | raw-data inventory; fails loudly on anything missing |
| [`shap_report.py`](shap_report.py) | TreeExplainer attribution (works for lgbm/xgb/cat/RF) |
| [`check_modeling.py`](check_modeling.py) | known-answer suite (~80 checks, no data needed) — run after *any* change here |

## How a method gets tested (the ladder)

```bash
# 0. once, on the machine with the raw data:
.venv/bin/pip install -r requirements-ml.txt
python -m modeling.data_check          # all inputs present & readable?
python -m modeling.audit --scope aweil # GATE 0 — any multi-year signal at all?

# 1. primary first:
python -m modeling.lightgbm.train --scope aweil
python -m modeling.lightgbm.train --scope national

# 2. cross-checks, in order (cards compare against the saved primary run):
python -m modeling.xgboost.train     --scope aweil
python -m modeling.catboost.train    --scope aweil
python -m modeling.randomforest.train --scope aweil
python -m modeling.tabpfn.train      --scope aweil

# 3. spatial:
python -m modeling.unet.train --epochs 2   # smoke test (seconds)
python -m modeling.unet.train --epochs 20  # the test (3090)

# 4. advisory layer (reads the primary's predictions by default):
python -m modeling.advisory --case-week 2024-10-07 --case-county "Aweil East"
```

Each run writes into `outputs/methods/<name>/` (gitignored): a
predictions CSV, a metrics CSV with a **role** column, and a **model
card** (`task_a_modelcard_*.md`) containing an automatic keep/kill
verdict. The keep/kill rules themselves are in each folder's `guide.md`
— that file is the contract; when a method dies, the guide's "record it
in MODEL_RESEARCH.md" step is what survives.

## Status

- **Code complete and logic-validated** (known-answer suite, 2026-09-21):
  CRPS arithmetic, embargo, split boundaries, baselines, advisory tiers,
  U-Net shapes/grid/null-baseline, method-policy checks.
- **Never run on real data** — there is no `raw_data/` on this machine,
  so no method has produced a single real number, and the U-Net has
  never been executed at all (torch not installed).
- The full maturity story (verified vs untested, most-likely first
  failure points, what "tested" means) is the "Status of the modeling
  code" section in `MODEL_RESEARCH.md`.
