# Flood-mask EDA: Northern Bahr el Ghazal

This is Imad's EDA of the satellite flood masks for Aweil Centre, East, North, South and West.
We look at where floods are detected, which months have the most detected flooding, and how the
detected area changes over time. We also count consecutive weeks with detections in each county.

These results are a starting point for the group's work on flood forecasting and food security.
The forecasting model and the link with food-security data still need to be developed.

## Running the notebook

1. Install the packages in [requirements.txt](../requirements.txt) in a Python 3.12+ environment.
2. Put the supplied data in the repository's `raw_data/` folder. The notebook uses:
   - `Administrative boundaries/ssd_admin2.geojson`
   - `flood_masks/compact_recurring/flood_events_h20v08_YYYY.parquet`
   - `flood_masks/compact_unusual/flood_events_h20v08_YYYY.parquet`
   We need both flood classes for each year from 2000 to 2025, so 52 Parquet files in total.
   The code gives an error if a file is missing.
3. Open [flood_masks_eda.ipynb](flood_masks_eda.ipynb) in VS Code with the Python and Jupyter extensions.
   Select the project environment as the kernel and click **Run All**.
   You can start from Capstone Data Challenge, the main project folder or the notebook folder.
4. The notebook saves CSVs in [outputs/tables/](outputs/tables/) and PNGs in
   [outputs/figures/](outputs/figures/). Save the notebook after running it to keep the results visible.

The notebook was tested with Python 3.12.10, pandas 3.0.3, NumPy 2.4.6, GeoPandas 1.1.4,
PyArrow 24.0.0, Matplotlib 3.10.9, Seaborn 0.13.2 and ipykernel 7.3.0.

If the notebook was already open when its files changed, close it and open the saved version again.
Then restart the kernel and run all cells. The setup cell also reloads the Python helper files.
Use the PNGs in `outputs/figures/`. PNGs directly in `outputs/` come from the old notebook version.

You can also run the small checks below from the main project folder. They use made-up data,
so you do not need the raw files for this step.

```bash
python -m EDA_flood_masks.check_flood_eda
```

## Files

| File or folder | What it contains |
|---|---|
| `flood_masks_eda.ipynb` | The analysis, plots and explanations of the results. |
| `flood_eda.py` | Functions for loading the data and calculating the weekly areas, monthly averages and consecutive detection weeks. |
| `check_flood_eda.py` | Small examples with known answers to check the calculations. |
| `README.md` | Instructions, methods, results and an explanation of the output columns. |
| `outputs/tables/` | Six CSV files with the results. |
| `outputs/figures/` | Three PNG figures made by the notebook. |

## How we process the data

### Selecting the study area

The Parquet files only contain records where flooding was detected. Each date comes from a
three-day composite, which combines observations from that day and the two previous days.
These periods overlap. A detection on several dates therefore does not tell us the exact number
of days that a location was under water.

The five counties are between roughly 26.14-28.04 degrees E and 7.89-9.90 degrees N.
They all fall inside tile `h20v08` (20-30 degrees E, 0-10 degrees N), so we only load that tile.

We first filter the data using a bounding box. Arrow applies this filter before loading the
selected rows into pandas. We read the annual files separately because their Arrow encodings differ.

Next, we use the Admin-2 boundaries to assign each pixel to a county. We only do this once per
pixel location and then add the county name to all its dates. This avoids repeating the same
spatial join for every observation.

The coordinates use EPSG:4326. A pixel belongs to a county if its centre is inside the polygon,
using `within`. We count the full pixel area in that county. We do not split pixels at county
borders, and centres exactly on a boundary are excluded.

### Calculating pixel area

The data is described as having a 250 m resolution, but the grid cells are not exactly 250 m
squares. According to the [NASA user guide](../literature/MCDWD_VCDWD_UserGuide_RevF.pdf),
Table 5 on page 28, the grid step is 10/4800 degrees. NASA uses a sphere with a radius of
6,371,007.181 m for this product.

We use that grid definition to calculate area from each pixel's latitude:

```text
area_km2 = R^2 * delta_lon * (sin(phi + delta_lat/2) - sin(phi - delta_lat/2)) / 1,000,000
```

Here, `phi` is the latitude of the pixel centre. The angles are in radians.
At 9 degrees N, one pixel covers about 0.053004 km2. This is smaller than the
0.0625 km2 we used in the first version.

The earlier review gave about 0.0528 km2 using WGS84. The code now follows the sphere
specified in the NASA guide, which explains the small difference between these estimates.

### Weekly areas

We use weeks from Monday to Sunday. For each county and week, we count every detected pixel once,
even if it appears on several dates or in both flood classes. We then add up those pixels' areas.
This is the weekly union: all locations where flooding was detected at some point during the week.
It does not mean that all those locations were flooded at the same time.

A pixel can be labelled `recurring` on one date and `unusual` on another date in the same week.
Adding the two class areas would then count that pixel twice. We therefore save the total area
and the areas per class in separate tables. Use the total table when you need a county's weekly area.

### Monthly averages

For each county, we include every week starting on a Monday in 2000-2025.
We fill weeks with no records with zero detected area. This means that the dataset has no
detections for that week. It does not prove that the land was dry.

We first calculate the average weekly area within each month of each year.
Then we average those values across the 26 years, giving each year the same weight.
A week belongs to the month and year of its Monday. Taking an average keeps the result in km2
and avoids giving a month a larger value just because it has five Mondays.

We do not know how many valid satellite observations there were in every week.
Having a file for a year does not mean that the satellite observed the area throughout that year.
This also applies to 2000 and 2025. The monthly pattern can therefore be affected by missing observations.
The CSV includes the number of years and weeks used in the calculation.

### Consecutive weeks with detections

We group consecutive weeks with detections somewhere in the same county. We call each group a
county detection run. A week without detections ends the run, although missing observations could
also explain that gap.

The detected pixels can change from week to week. A run of five weeks therefore does not mean
that the same field was flooded for five weeks. We use the name `county_detected_weeks` to make
that distinction clear.

For each run, we also save the largest weekly area and the sum of all weekly areas.
That sum has units km2-weeks and counts a pixel again if it is detected in another week.

## Results

The dataset contains **972,588 records** in the five counties. There are **1,457 combinations of a
county and a week with detections**, grouped into **357 detection runs**.
The first detection is on **2000-10-21** and the last is on **2025-12-21**.
These are the first and last dates with records, not the start and end of satellite coverage.

| County | Detection runs | Median weeks per run | Longest run in weeks | Largest weekly area (km2) | Peak month | Mean weekly area in that month (km2) |
|---|---:|---:|---:|---:|---|---:|
| Aweil Centre | 82 | 2 | 20 | 93.65 | October | 4.37 |
| Aweil East | 98 | 2 | 33 | 375.26 | October | 43.84 |
| Aweil North | 56 | 1 | 13 | 42.85 | October | 2.42 |
| Aweil South | 50 | 4 | 25 | 169.81 | November | 22.09 |
| Aweil West | 71 | 2 | 14 | 91.06 | October | 9.92 |

Most of the detections are concentrated towards the east and southeast of the study area.
Aweil East has the largest weekly detected area. The monthly average peaks in October in four
counties and in November in Aweil South.

We compare areas in km2 here. Larger counties can also have larger detected areas, so this table
alone does not show which county has the highest relative flood risk or the most affected farmland.
The longest detection run is 33 weeks. As explained above, this is a county-level measure.

We found **3,004 pixel-week combinations** that appeared in both flood classes, spread across
**83 county-weeks**. Removing this overlap reduced the largest weekly area from 460.06 to
442.38 km2 with the old pixel size. Using the correct pixel areas brings it to **375.26 km2**.
The number and length of the runs stayed the same.

These numbers come from the current data. The notebook calculates its results text again when
it runs. Update this README too if the data or the method changes.

## Output dictionary

### `coverage_summary.csv`

One row per county and flood class. Despite the filename, this table summarises detections.
It does not tell us how often the satellite could observe the area.

| Column | Meaning |
|---|---|
| `county`, `flood_type` | County name and class, either `recurring` or `unusual`. |
| `detections` | Number of records. |
| `first_detection`, `last_detection` | First and last dates with a detection. |
| `mean_cloud_fraction` | Average of the supplied cloud values. All values are zero here, so they cannot tell us whether the observations were cloud-free. |

### `weekly_county_floods.csv` and `weekly_county_floods_by_type.csv`

The first table has one row per county and week with detections. The second also splits these rows
by flood class. Weeks without detections are left out of both tables.

| Column | Meaning |
|---|---|
| `county`, `week` | County name and Monday date. |
| `flood_type` | Only included in the table that splits the results by class. |
| `detected_pixel_days` | Number of unique pixel-and-date combinations in the group. |
| `mean_cloud_fraction` | Mean cloud value after removing repeated pixel-and-date records within the group. This does not correct for missing observations. |
| `unique_grid_cells` | Number of different pixels detected during the week. |
| `detected_area_km2` | Total area of those pixels, counting each pixel once within the group. |
| `year`, `month` | Calendar year and month of the Monday date. The year is not the ISO week-year. |

### `monthly_seasonality.csv`

One row per county and month, including months without detections.

| Column | Meaning |
|---|---|
| `county`, `month` | County name and month number from 1 to 12. |
| `mean_weekly_detected_area_km2` | Average weekly area within each year-month, then averaged across years. |
| `years_included` | Number of yearly averages used, which is 26 here. This does not mean that observations were complete in all those years. |
| `calendar_weeks` | Total number of weeks assigned to this month across the selected years. |
| `weeks_with_detection` | Number of those weeks that have at least one detection. |

### `detected_flood_events.csv`

Each row is a county detection run. We kept the original filename, but these rows do not describe
the duration of flooding at one location.

| Column | Meaning |
|---|---|
| `county`, `run_id` | County name and run number within that county. The numbers can change if the data changes. |
| `start_week`, `end_week` | Monday dates of the first and last weeks in the run. These are not exact flood start and end dates. |
| `county_detected_weeks` | Number of consecutive weeks with detections somewhere in the county. |
| `peak_weekly_union_area_km2` | Largest weekly detected area in the run. |
| `cumulative_detected_area_km2_weeks` | Sum of the weekly areas in km2-weeks. Pixels can count again in different weeks. |

### `event_summary_by_county.csv`

One row per county with the number of runs (`detection_run_count`), median length
(`median_county_detected_weeks`), longest run (`longest_county_detected_weeks`) and largest weekly
area (`largest_weekly_union_area_km2`). Use `county` to join this table to the other results.

### Figures

- `flood_detection_frequency.png` shows where detections occur. Both maps use the same logarithmic
  colour scale, so the same colour means the same number of dates with a detection.
- `monthly_seasonality.png` shows the average weekly area per month in km2. Values between zero
  and 0.1 km2 are shown as `<0.1` to keep them separate from zero detections.
- `event_size_duration.png` compares the length of each county detection run with its largest weekly
  area. Colours show the counties. Bubble sizes show the sum of weekly areas.
  The area axis is logarithmic so the smaller runs are easier to see.

## Limitations and next steps

- `recurring` and `unusual` tell us about recurrence. They do not directly tell us how severe or
  predictable a flood is.
- All `cloud_frac` values are zero. We cannot use them to separate dry land from missed flooding.
- The three-day composites overlap, and grouping them into weeks makes exact flood timing harder to measure.
- County totals can hide local differences. We still need to combine the floods with crop, rangeland
  and population data, including the possible effects on herders who move between areas.
- To study crop damage, we need to follow the same pixels or fields over time and account for missing observations.
- Before modelling, we need to align the dates of the flood, weather, river, lake, crop and IPC data.
  We also need to check how the recurrence labels were made. If they use information from later years,
  using them directly could cause data leakage. A forecast model should be tested on later time periods.

## Changes and checks (2026-09-09)

- Fixed pixels being counted twice when they appeared in both flood classes during one week.
- Changed the pixel area from a fixed 0.0625 km2 to an area based on the actual grid and latitude.
- Changed the monthly calculation to use average weekly areas.
- Regenerated the figures because the old seasonality figure did not match the notebook.
- Renamed the duration columns to make clear that they describe detections across a county.
- Put CSVs and PNGs in separate folders and added the results and column explanations to this README.
- Added county names, matching map colours, clearer labels and a log scale for the event plot.
- Added checks with small example datasets, a check for missing annual files and `ipykernel` to the requirements.

Some CSV columns changed. `detected_pixels` is now `detected_pixel_days`, and the main weekly
CSV now contains totals. In the run CSV, `event_id`, `duration_weeks`, `peak_area_km2` and
`area_week_km2` became `run_id`, `county_detected_weeks`, `peak_weekly_union_area_km2` and
`cumulative_detected_area_km2_weeks`. Use these new names if you load the tables in other code.

The checks pass on both the example data and the real weekly totals. All eight code cells ran in a
fresh Jupyter kernel. Starting from Capstone Data Challenge, the main project folder and the notebook folder worked.
The saved notebook includes the tables, three figures and calculated results.
The full notebook takes about 12 seconds on the local machine.
