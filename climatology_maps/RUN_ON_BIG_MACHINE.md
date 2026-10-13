# Running the heavy pipeline on the big machine (RTX 3090)

The small laptop/PC (4 cores, 8 GB RAM) is fine for the **map module** (the
streaming `forecast_maps` peaks around 0.7 GB), but the **model training**
(LightGBM CV, Optuna tuning, TabPFN, and later the U-Net) belongs on the beefy
machine. This page is the copy-paste runbook. Everything is executed from the
repository root.

## 0. Facts

* Machine: any PC with a recent GPU (RTX 3090 listed) and 32+ GB RAM.
* Repo: `JBG060_ZHL_2026` (clone from GitHub; code-only, `raw_data/` is gitignored).
* Data: the SURFdrive zip (`data-JBG060-2026.zip`), ~4 GB, extracted into `raw_data/`.

## 1. Environment (once)

```bash
git clone https://github.com/TijnQ/JBG060_ZHL_2026.git   # or your fork
cd JBG060_ZHL_2026

conda create -n jbg060 "python>=3.12" -y
conda activate jbg060
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA-accelerated TabPFN / future U-Net (optional but recommended):

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
# and check tabpfn picks up the GPU:  python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Data (once)

Either copy `raw_data/` over from a machine that already has it, or download:

```bash
# SURFdrive public share (the password is the student share password)
curl -u "x:<SHARE_PASSWORD>" -o data-JBG060-2026.zip \
  "https://surfdrive.surf.nl/public.php/dav/files/<TOKEN>/?accept=zip"
```

Extract with any unzip tool into `raw_data/` so that:
`raw_data/flood_masks/compact_recurring/flood_events_h20v08_YYYY.parquet` etc.
Warning: `/tmp` on a small box is often a RAM disk — extract onto the real disk.

## 3. Model training (the heavy part — run here, not on the laptop)

```bash
# (1) prepare the evapotranspiration CSVs the LightGBM pipeline reads
python process_et_NBeG.py                # ~2000-2025, Aweil grid cell

# (2) LightGBM 3-fold expanding-window CV
python LightGBM_v1/run.py --model lightgbm_fe            # engineered features
python LightGBM_v1/run.py --model lightgbm_baseline     # frozen baseline

# (3) optional: Optuna tuning (long; 50 trials x 3 folds)
python -m LightGBM_v1.tuning --feature-set fe --n-trials 50 --evaluate

# (4) optional: TabPFN reference (GPU via torch)
python LightGBM_v1/run.py --model tabpfn
```

## 4. Forecast maps (fast even on the laptop)

```bash
# build the per-pixel climatology once from the raw flood Parquets (~30 s)
python -m climatology_maps.pixel_climatology

# maps from the committed LightGBM predictions (no retraining needed)
python -m climatology_maps.forecast_maps                       # 12 peak weeks + HTML
python -m climatology_maps.forecast_maps --year 2024 --all-weeks   # every week of 2024
python -m climatology_maps.forecast_maps --horizon 2 --metric prob # persistence h2
python -m climatology_maps.forecast_maps --agg-step 0.01           # finer slider grid
```

The map output is written to `climatology_maps/outputs/`:

* `forecast_maps/*.png`                 — one image per week per layer
  (`prob`/`sev`/`worst`/`clim`)
* `forecast_maps/forecast_maps_interactive.html` — week slider + play (gitignored)
* `forecast_maps/week_summary.csv`      — per county-week: inputs + implied totals
* `climatology_data/`                   — pixel-climatology cache (gitignored)

## 5. Copying results back

The repo stays small: code is pushed via git; data and generated outputs stay
local. To bring results to the laptop/other people, copy **only artifacts**:

```bash
# from the big machine
scp -r climatology_maps/outputs/forecast_maps/  you@laptop:/path/JBG060_ZHL_2026/climatology_maps/outputs/
scp climatology_maps/outputs/climatology_data/  you@laptop:/path/JBG060_ZHL_2026/climatology_maps/outputs/   # optional cache
# LightGBM outputs (tables + figures) if you retrained
scp -r LightGBM_v1/outputs/  you@laptop:/path/JBG060_ZHL_2026/LightGBM_v1/
```

Anything in `.gitignore` (`raw_data/`, `climatology_maps/outputs/*`,
`LightGBM_v1/outputs/tuning/`) will not be pushed — which is what we want.

## 6. Memory notes (why this works)

* `pixel_climatology.py` reads the 52 flood Parquets with pyarrow filters.
* `forecast_maps.py` is **streaming**: one week is composed and rendered at a
  time; only tiny max-aggregated mats are kept for the HTML. Peak RSS ~0.7 GB.
  (A previous version held all weekly frames in RAM and froze an 8 GB laptop.)
* If the big machine is missing a `raw_data` source, check provenance:
  `LightGBM_v1` fails loudly instead of silently using fake data
  (the synthetic-fallback option was removed entirely).