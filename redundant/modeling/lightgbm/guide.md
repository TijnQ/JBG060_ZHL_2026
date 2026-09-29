# LightGBM — Task A primary backbone

**Role:** the backbone model for Task A ("how much flood area next week").
Per `MODEL_RESEARCH.md` §4.2 the ladder is *LightGBM first, other models as
cross-checks* — this folder is that first rung.

**Status: UNTESTED on real data** (code complete 2026-09-21; first real run
on the machine with `raw_data/` + the ML env).

## The question this method tests

1. **Does any model beat persistence on spike weeks?** The weekly Aweil
   series is zero-inflated and spiky; a model must beat "last week's area"
   on the non-zero weeks to be operationally useful. This is the single
   most important number in the project:
   `tables/task_a_metrics_aweil.csv`, row `split=test, slice=spike,
   model=lightgbm, skill_q50`.
2. **Do the embargoed ERA5 features carry signal at all?** If the primary
   can't beat persistence and the gate-0 audit showed weak correlations,
   the honest conclusion is "the weekly signal is too weak for this
   feature set" — which is itself the answer to the research question, and
   it *ends* this method's story.

## How to run

```bash
# first time only:
.venv/bin/pip install -r requirements-ml.txt

# from the repo root:
.venv/bin/python -m modeling.lightgbm.train --scope aweil
.venv/bin/python -m modeling.lightgbm.train --scope aweil --shap
.venv/bin/python -m modeling.lightgbm.train --scope national
```

- `--scope aweil` (default): 5 Aweil counties, ~1.4k non-zero weeks.
- `--scope national`: 79 counties; more rows, more signal per feature —
  run it after the Aweil run (it's the fallback if Aweil is too thin,
  §7 risk 1).
- `--shap`: top-feature attribution (best-effort; prints a skip note if
  `shap` misbehaves with this version of LightGBM).

## Outputs (in this folder's `outputs/methods/lightgbm/`)

| file | what it is |
|---|---|
| `tables/task_a_predictions_aweil.csv` | val+test predictions (3 quantiles + detection prob + y_true + baselines) — the advisory layer reads this file |
| `tables/task_a_metrics_aweil.csv` | metric table: this model + 3 baselines, per split x slice, with `skill_q50` vs persistence |
| `tables/task_a_modelcard_aweil.md` | one-page card incl. an automatic keep/kill verdict |
| `figures/task_a_top_shap_aweil.png` | (with `--shap`) mean \|SHAP\| by feature |

## Keep / kill verdict

| result | verdict |
|---|---|
| test spike `skill_q50` > 0 **and** test-all CRPS ≤ best baseline | **KEEP** — promote to the report; a cross-check that beats it by > 0.05 test-all skill is a promotion candidate, otherwise this stays the backbone |
| test spike skill ≤ 0 but test-all skill > 0 | **MARGINAL** — useful on dry weeks only; not operationally useful. Run the cross-checks; if none beat persistence on spikes, record "no method beats persistence at county-week resolution" and demote the whole Task A ladder to the national scope |
| test spike skill ≤ 0 and test-all skill ≤ 0 | **KILL** — delete this folder (git keeps the history); record the negative result in `MODEL_RESEARCH.md` |

## Expected runtime / hardware

Minutes on CPU (hundreds of thousands of rows max, 50 features, early
stopping at 100 rounds). No GPU needed.

## Known failure modes

- LightGBM API drift (callback/param names) — the version is pinned in
  `requirements-ml.txt`; if a new major breaks `fit_with_early_stop`,
  that's the function to fix.
- `NaN` in `skill_q50` when a slice has zero non-zero weeks (dry test
  period) — read it as "no spike weeks in that slice", not a bug.
- If the gate-0 audit (shared `modeling/audit.py`) already showed no
  signal, skip running this method and go straight to national scope —
  don't chase noise on the small Aweil panel.

## References

`MODEL_RESEARCH.md` §4.1 (baselines), §4.2 (model options), §5 (validation
protocol); the original ladder is the experiment E3 there.
