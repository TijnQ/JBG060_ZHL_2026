# Flood-climatology & weekly forecast maps of the Aweil region (Tijs)

Two stages live in this folder, both drawing maps for the five Aweil counties
(Northern Bahr el Ghazal):

1. `climatology_maps.py` — **purely historical** county-level choropleths
   (26 years of satellite flood detections -> per-county, per-calendar-month
   mean weekly flood area / frequency / km3). No model, nothing trained.
2. `forecast_maps.py` + `pixel_climatology.py` — **weekly pixel-level risk maps**:
   the LightGBM county forecast (`det_prob`, `q50/q90` area, duration heads)
   is downscaled to the ~232 m MODIS grid using the historical per-pixel
   spatial pattern as the placement prior. One image per week + interactive
   week slider.

This is the "climatology first" step of the group's plan: the historical
baseline is the reference that the LightGBM/U-Net models must beat.

## What the map shows, and what it does NOT do

The satellite flood product tells us, per ~0.25° cell and per week, whether water was
detected. From that we derive, for every county and every calendar month:

| Column | Meaning |
|---|---|
| `mean_weekly_flood_km2` | historical average weekly flood area in the county that month |
| `mean_weekly_flood_percent` | that area as % of the county's own area |
| `flood_frequency` | fraction of weeks in that month (across the baseline years) that had *any* flood |
| `mean_weekly_water_km3` | implied water volume = flood area × assumed depth (default 1 m) |

What we do **not** claim: a week with zero detections is not proof the land was dry, and
all cloud values in the source data are zero, so missed floods are possible. The maps are
a *harmonised historical average*, not a hazard forecast.

### The km³ of water — and the one assumption

The data only gives *area*. To turn area into **km³ of water** we multiply by an assumed
mean depth of standing flood water (`--depth-m`, default 1.0 m). This is the **only
non-observed number in the whole pipeline**, and it is explicitly user-controlled: change
it and only the km³ column/maps change.

## "Only changes based on input"

The output is a **function of its inputs and nothing else**:

```
input (month, baseline years, depth assumption, metric)
        |
        v
  [ pure mean over historical weeks ]   -->   map
```

The interactive output (`climatology_flood_km2_interactive.html`) is a Plotly choropleth
with a **month slider** — move the input, the map moves; there is no model and no learned
parameter anywhere.

## Forecast maps: LightGBM x pixel climatology

`pixel_climatology.py` reads the raw flood-mask Parquets
(`raw_data/flood_masks/compact_{recurring,unusual}/flood_events_h20v08_*.parquet`,
2000-2025) and records, for every ~232 m grid cell inside the five counties and
every calendar month, how often the satellite detected it as flooded
(`freq`), how often *given that the county flooded* that month
(`freq_given`, the spatial signature), plus the county-month climatological
probability `(k+0.5)/(n+1)`. The rebuilt county-week areas reproduce the
committed EDA table exactly (median ratio 1.000, r = 1.0). Output is cached in
`outputs/climatology_data/` (gitignored).

`forecast_maps.py` composes, for each prediction week:

| Layer  | Pixel value | Meaning |
|--------|-------------|---------|
| `prob` | `det_prob x freq_given` | P(pixel is wet this week) — *how likely* |
| `sev`  | `min(1, freq x q50 / hist_area)` | expected flood share of the pixel — *how heavy*, median forecast |
| `worst`| `min(1, freq x q90 / hist_area)` | worst-case (q90) severity |
| `clim` | `clim_det_prob x freq_given` | pure-climatology reference (no model) |

With no clipping, `sum_p sev x pixel_area == q50`: the map's lit area reproduces
the LightGBM area forecast (`week_summary.csv` keeps the clipped and unclipped
implied areas for audit). County-months with no historical floods fall back to
the county's annual pattern for placement (severity layers = 0).

## Usage

All the required historical data is **already committed** to the repo
(`EDA_flood_masks/outputs/national/tables/national_weekly_floods.csv`), so the
county-level climatology maps run without the big raw flood Parquet files. Only
the county boundary GeoJSON is needed; if
`raw_data/Administrative boundaries/ssd_admin2.geojson` is missing it is
downloaded automatically from HDX (`processing_data/loading_impact_data.py` uses
the same HDX source).

```bash
# every month -> 12 static PNGs + interactive HTML + summary CSV (km2 metric)
.venv/bin/python -m climatology_maps.climatology_maps

# only October, as water volume in km3 with a 0.5 m depth assumption
.venv/bin/python -m climatology_maps.climatology_maps --month 10 --metric km3 --depth-m 0.5

# flood frequency map for August
.venv/bin/python -m climatology_maps.climatology_maps --month 8 --metric freq
# ----------------------------------------------------------------
# weekly pixel-level forecast maps (LightGBM x pixel climatology)
# ----------------------------------------------------------------
# build the per-pixel climatology once (needs raw_data/flood_masks Parquets, ~30 s)
.venv/bin/python -m climatology_maps.pixel_climatology

# default: 12 peak-risk weeks as PNGs (4 layers each) + interactive HTML (362 weeks)
.venv/bin/python -m climatology_maps.forecast_maps

# every week of 2024
.venv/bin/python -m climatology_maps.forecast_maps --year 2024 --all-weeks

# flood-persistence map: chance the water is still there 2 weeks ahead
.venv/bin/python -m climatology_maps.forecast_maps --horizon 2 --metric prob

# custom prediction file (any csv with county/week/det_prob/q10/q50/q90)
.venv/bin/python -m climatology_maps.forecast_maps --predictions PATH

# OPERATIONAL PRODUCT (--mode final): downscale the Level-1 final-fit forecast
# (LightGBM_v1/outputs/outputs_final_fit/tables/final_forecast_aweil.csv, written
# by `python -m LightGBM_v1.run --mode final`) -> outputs/outputs_final_fit/
.venv/bin/python -m climatology_maps.forecast_maps --mode final

# the same product 2 weeks ahead (only h1/h2 ship in the final table)
.venv/bin/python -m climatology_maps.forecast_maps --mode final --horizon 2 --metric prob
```

Run the small self-checks (no raw data needed):

```bash
.venv/bin/python -m climatology_maps.test_climatology_maps
.venv/bin/python -m climatology_maps.test_forecast_maps
```

Outputs land in `climatology_maps/outputs/`: `*.png`, `*_interactive.html`,
`climatology_<metric>.csv` (county climatology), `forecast_maps/` (weekly maps,
`week_summary.csv`), `outputs_final_fit/` (the operational `--mode final`
product) and `climatology_data/` (pixel-climatology cache). The
interactive `*.html` files are large and are **gitignored** — regenerate them on
demand. See `RUN_ON_BIG_MACHINE.md` for running the heavy model training and map
generation on the RTX 3090 machine.

## Results (climatology over 2000-2025)

Peak flood season is **October-November**, concentrated in the Sudd floodplain and the
western states:

| Month | Top county | Mean weekly flood area |
|---|---|---|
| Oct | Rubkona | 137.0 km² |
| Nov | Rubkona | 168.5 km² |

Within the five Aweil counties the climatology reproduces the EDA: **Aweil East** is the
largest, and **Aweil South** is the highest in November. These maps are the historical
baseline that spelling out a flood risk for the Aweil region and for later comparison
against the LightGBM/U-Net predictions.

## Design note: ready for model output

`render_static_map`/`render_interactive_map` draw whatever **county -> value** column they
are given, and `forecast_maps.py` already consumes the LightGBM county-week
forecasts (`det_prob`, `q50/q90`, `det_prob_h1..h4`) from the committed
`LightGBM_v1/outputs/*/tables/cv_test_predictions_aweil.csv`, using the
per-pixel climatology as the spatial prior. When the U-Net (per-pixel)
predictions are ready, feed `compose_week`-equivalent pixel forecasts into the
same rendering functions. The climatology stays the baseline that any model
must beat.