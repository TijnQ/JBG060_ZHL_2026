# U-Net — Task B ("where will it flood?")

**Role:** the *only* Task B method in the current ladder. A small
4-block U-Net predicts, per 0.25° grid cell of HEC/USGS tile h20v08
(40×40 pixels covering the Aweil floodplain), the probability that the
pixel is flooded during the target week. Task A says *how much* (county
area); this says *where* — the 2024-report case study needs both.

**Status: UNTESTED on real data — the code has never been executed**
(torch is not installed on the dev machine; the known-answer suite
covers only shapes/grid/null-baseline with synthetic arrays). The first
run — including the first test of the CUDA 12.8 wheel against the 3090's
driver — happens on the GPU machine.

## What to test

1. **Does the network beat the null baseline on the TEST split?** The
   null baseline is each pixel's historical flood frequency,
   threshold-calibrated to the model's own detection rate — a fair
   "climatological where-map". If the U-Net's test CSI does not beat the
   null's, the spatial signal is not beyond climatology and Task B's
   honest answer is "climatology is the map".
   → `tables/task_b_unet_metrics.csv`: compare `baseline=model` vs
   `baseline=null` rows for `split=test`.
2. **Is the map *right*, not just *more accurate*?** Open
   `figures/task_b_unet_maps.png`: does the mean-val-probability map put
   high probability on the Bahr el Ghazal / Aweil East-West channels and
   low probability on the dry ridges? A model that beats the null by
   0.01 CSI while painting nonsense is not a win.
3. **First-execution checks (gate, not science):**
   - it trains without OOM on the 3090 (40×40×17 input, ~1 M params —
     should fit with wide margin),
   - AMP on/off both work,
   - val-CSI history is not monotone garbage (plateau or dip-then-rise
     are both fine; pure noise is a kill signal for the training loop,
     not the idea).

## How to run

```bash
# first time only (torch 2.14.0, CUDA 12.8 wheel):
.venv/bin/pip install -r requirements-ml.txt

# from the repo root, on the machine with raw_data/ and the 3090:
.venv/bin/python -m modeling.unet.train --epochs 2      # smoke test: it fits, it predicts
.venv/bin/python -m modeling.unet.train --epochs 20     # the actual test
.venv/bin/python -m modeling.unet.train --epochs 20 --device cpu  # fallback if CUDA fails
```

## Outputs (in `outputs/methods/unet/`)

| file | what it is |
|---|---|
| `tables/task_b_unet_metrics.csv` | CSI/POD/FAR, model vs null, per split |
| `figures/task_b_unet_maps.png` | mean val probability (model) vs calibrated frequency (null) |

## Keep / kill verdict

| result | verdict |
|---|---|
| test CSI(model) > test CSI(null) **and** the map is visually sane | **KEEP** — promote to the report; this is the deliverable spatial model (a 2026 paper can describe it as a 24h-lead weekly map, per the research doc's scope honesty) |
| test CSI(model) ≈ null within noise, map sane | **MARGINAL** — report climatology as the honest Task B answer; keep the folder only if a follow-up (more channels, longer history) is planned |
| training loop fails at the smoke test (OOM / NaN loss / API break) | fix the loop first (this is gate-level, not science) — the guide's first-execution checks are the debug order |
| test CSI(model) < null and map is nonsense | **KILL** — delete the folder; record the negative result |

## Expected runtime / hardware

`--epochs 2` smoke: seconds on CPU, ~1 min on GPU. `--epochs 20` with
early stopping (patience 5): minutes on the 3090; batch 32 over ≤ 1236
train weeks of 40×40×17 tensors is small.

## Known failure modes

- **Never executed** — the first run may reveal typos that no test can
  catch (tensor layout, AMP edge cases). Read the training-loop section
  of the module docstring before debugging.
- rasterio transform handling in `raster_to_grid` (assumes a standard
  north-up affine transform; a rotated/UTM farmland raster would need
  reprojection — check the static channels look right in the map figure).
- `pos_weight` can explode when the training period is extremely dry
  (it is sqrt(neg/pos)); a weight > ~100 makes BCE unstable — cap it if
  the loss history shows it.

## References

`MODEL_RESEARCH.md` §4.3, §5, §9.5; U-Net (Ronneberger et al. 2015,
arXiv:1505.04597) in the References section there.
