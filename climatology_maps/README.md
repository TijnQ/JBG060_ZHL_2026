# Flood-climatology maps of South Sudan (Tijs)

A **purely historical** ("climatology") mapping module. Nothing is trained and no
forecast is made: we summarise 26 years (2000-2025) of satellite flood detections into a
per-county, per-calendar-month **climatology** and draw choropleth maps of where floods
occur.

This is the "climatology first" step of the group's plan: before a LightGBM/U-Net model
says anything useful, we need the historical baseline map to compare against and to show
non-technical readers *where* flooding happens.

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

## Usage

All the required historical data is **already committed** to the repo
(`EDA_flood_masks/outputs/national/tables/national_weekly_floods.csv`), so this runs
without the big raw flood Parquet files. Only the county boundary GeoJSON is needed; if
`raw_data/Administrative boundaries/ssd_admin2.geojson` is missing it is downloaded
automatically from HDX (`processing_data/loading_impact_data.py` uses the same HDX source).

```bash
# every month -> 12 static PNGs + interactive HTML + summary CSV (km2 metric)
.venv/bin/python -m climatology_maps.climatology_maps

# only October, as water volume in km3 with a 0.5 m depth assumption
.venv/bin/python -m climatology_maps.climatology_maps --month 10 --metric km3 --depth-m 0.5

# flood frequency map for August
.venv/bin/python -m climatology_maps.climatology_maps --month 8 --metric freq

# restrict the climatology baseline window
.venv/bin/python -m climatology_maps.climatology_maps --baseline-start 2015 --baseline-end 2025
```

Run the small self-checks (no raw data needed):

```bash
.venv/bin/python -m climatology_maps.test_climatology_maps
```

Outputs land in `climatology_maps/outputs/` (`*.png`, `*_interactive.html`,
`climatology_<metric>.csv`). The interactive `*.html` files are large and are
**gitignored** — regenerate them on demand.

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
are given. When the LightGBM (county-week area) or U-Net (per-pixel) predictions are ready,
feed the same mapping functions the predicted values instead of `monthly_climatology(...)`
and the identical maps render the forecasts. The climatology stays as the baseline that any
model must beat (same "keep/kill" idea as in `modeling/`).