# `modeling/` — flood forecasting models (Tasks A, B, C)

Implementation of the experiment plan in [`MODEL_RESEARCH.md`](../MODEL_RESEARCH.md):

| Task | Question | Entry point |
|------|----------|-------------|
| A | Forecast weekly flood extent per county (2000-2025, ERA5 + gauge inputs) | `python -m modeling.train_task_a --scope aweil` (LightGBM primary) / `--families lgbm,xgb,cat` for the cross-check ladder |
| B | Forecast *where* floods happen: weekly pixel grids from ERA5 tiles (U-Net) | `python -m modeling.unet --epochs 20` |
| C | Turn a forecast into an impact advisory (exposed farmland / rangeland / cattle) | `python -m modeling.advisory --scope aweil` |

Supporting modules:

| Module | Role |
|--------|------|
| `config.py` | All paths, split boundaries, hyperparameters. Change numbers here, not in code. |
| `data_check.py` | Pre-flight inventory of the raw data (no processing): `python -m modeling.data_check`. **Run this first.** |
| `audit.py` | **Gate 0**: multi-year (2000-2024) lagged hydro-meteorology audit vs flood extent before training: `python -m modeling.audit --scope aweil`. Writes `outputs/audit/{scope}_audit.csv` and a heatmap. |
| `splits.py` | Split assignment + 1-week boundary purge + the embargo rule (`feature_cutoff`). |
| `metrics.py` | MAE/RMSE/R², CRPS from the 3-quantile forecast, detection (POD/FAR/CSI), skill vs persistence. |
| `baselines.py` | Climatology, persistence, last-detection baselines (every model must beat these). |
| `features.py` | Feature frame builder: ERA5 boxes, gauge, Albert level, ET0, rolling windows, spike threshold. |
| `train_task_a.py` | Task A: **LightGBM is the primary backbone** (default run trains only it); XGBoost / CatBoost are cross-checks added via `--families`, with an explicit primary-vs-cross-check verdict on the model card. Quantile + detection models, metrics, SHAP on the primary. |
| `train_tabpfn.py` | Optional zero-shot TabPFN cross-check (degrades gracefully to point predictions). |
| `unet.py` | Task B: 40x40 tile U-Net, 17 channels (14 daily ERA5 + 3 static), BCE with pos_weight, early stopping, null baseline (historical flood frequency), 2-panel figure. |
| `shap_report.py` | SHAP summary for a trained LightGBM model (bar plot of the top-15 |mean SHAP|). |
| `advisory.py` | Task C: magnitude tier (county-month climatology), flood-phase guidance, scaled exposure (crop + rangeland ha, cattle head-count), text advisory. |
| `check_modeling.py` | **Known-answer checks — no raw data required.** `python -m modeling.check_modeling`. Pins CRPS values, split boundaries, purge behaviour, baselines, advisory tiers, U-Net shapes. Run after any change to this package. |

## Validation protocol (all three tasks)

- **Weeks**: Monday-aligned (`W-SUN`); target week `M` covers composites `M .. M+2` (Task A) — Task B unions all seven composites of the week, matching the EDA convention.
- **Splits** (`config.py`): train 2000-01 .. 2014-12, validation 2015 .. 2019, test 2020 .. 2025.
- **Purge**: the first week of the validation and test splits is dropped (its 3-day
  composite window overlaps the previous split) — `splits.week_split_map` returns
  `None` for those weeks and both tasks drop them.
- **Embargo**: every feature for a target week is dated **<= Monday - 3 days**
  (`splits.feature_cutoff`). No later data is read, so no leakage.
- **Labels**: Task A from the committed weekly flood CSVs (sparse -> full calendar,
  0 = no detection). Task B from the MCDWD composite masks directly.
- **No synthetic data, ever**: every loader in this package raises with a pointer
  to `data_check.py` when a file is missing. (The repository was already burned
  once by a silent synthetic fallback in `EDA_hydrometeorology` — do not repeat it.)

## Running it

```bash
cd <repo root>
.venv/bin/pip install -r requirements-ml.txt     # heavy ML stack (separate from requirements.txt)

# 1. data present?
.venv/bin/python -m modeling.data_check

# 2. gate 0: multi-year signal audit (do NOT skip)
.venv/bin/python -m modeling.audit --scope aweil

# 3. Task A (CPU is fine). Default = LightGBM primary only ("try LightGBM first");
#    add --families lgbm,xgb,cat for the full ladder incl. cross-checks.
.venv/bin/python -m modeling.train_task_a --scope aweil --shap

# 4. Task B (GPU strongly recommended; CPU works for a smoke test with --epochs 2)
.venv/bin/python -m modeling.unet --epochs 20 --device auto

# 5. Task C (deterministic; uses Task A predictions when present)
.venv/bin/python -m modeling.advisory --scope aweil

# known-answer checks (always, after changes)
.venv/bin/python -m modeling.check_modeling
```

Outputs land in `modeling/outputs/` (gitignored):
`audit/`, `features/`, `task_a/` (predictions, metrics, model cards, SHAP plot),
`task_b/` (metrics + map figure), `advisory/`.

## Status

- **Code complete and self-checked** as of 2026-09-21 (this machine has no raw
  data and no GPU; everything here was validated with `check_modeling.py`).
- **Next on the GPU machine**: `data_check` -> `audit` -> `train_task_a` ->
  `unet` smoke test -> `advisory`. If the gate-0 audit shows no meaningful
  multi-year signal in the Aweil scope, the Task A national run is the fallback
  (more counties = more signal per feature).
- Hardware target: RTX 3090 (24 GB, sm_86). The U-Net (40x40x17 input, ~1 M
  parameters) fits with a large margin; no gradient checkpointing needed.
