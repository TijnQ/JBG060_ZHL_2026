# Floods, cattle and grazing land in South Sudan

We look at where floods were detected in South Sudan and what was in those areas according to the cattle and land maps. We want to see which counties stand out and how the results change through the year.

The analysis covers 2000 to 2025 and all 79 counties in the supplied boundary file. We also compare the five Aweil counties with the rest of the country.

## Running the notebook

The main notebook for this analysis is [south_sudan_flood_exposure_eda.ipynb](south_sudan_flood_exposure_eda.ipynb). Open it, select the project Python environment and choose **Run All**. It reads the raw data, runs the calculations and saves the tables and figures. You do not need a separate script to create the charts.

Use Python 3.12 or newer. The package versions are listed in [requirements.txt](../requirements.txt). Install them from the main project folder with:

```bash
python -m pip install -r requirements.txt
```

Keep these supplied files in `raw_data/` with their original names:

- `Administrative boundaries/ssd_admin2.geojson`
- All 104 Parquet files in `flood_masks/compact_recurring/` and `flood_masks/compact_unusual/`
- `farmland/geonode__cattle_gha.tif`
- `farmland/asap_mask_rangeland_v04.tif`

The notebook handles one year at a time to keep memory use down. A full run took about three minutes on our computer.

The Python files contain functions that the notebook uses:

- `national_flood_eda.py` loads the flood data and groups it by county and week.
- `cattle_rangeland_exposure.py` compares the flood locations with the cattle and land maps.
- `flood_eda.py` provides the flood-pixel area calculation and finds consecutive weeks with detections.
- `check_national_exposure.py` checks the calculations with a small example.

The calculation check also runs when you start the notebook. To run it separately, use this command from the main project folder:

```bash
python -m EDA_flood_masks.check_national_exposure
```

## What do we mean by overlap?

We compare maps of the same places. First, we find the places where the satellite detected flooding. Then we check how much grazing land and how many cattle the other maps show in those places. This is what we call overlap.

For grazing land, we use the grazing percentage on the land map. For example, if a detected flood area covers 1 km² and the land map says that half of it is grazing land, we count 0.5 km² of grazing land.

For cattle, the map uses much larger squares than the flood data. Suppose a cattle-map square contains an estimated 100 cattle and flood detections cover 20% of that square in a week. We estimate 20 cattle in the flood area. This assumes that cattle are spread evenly across the square. We do not know where individual animals actually were.

We call these estimates exposure in the notebook and tables. They do not tell us how many animals died or how much land was damaged.

## How we count the data

We assign each flood pixel to a county using its centre. We count it once per week, even if the satellite detected it on several days or it appears in both flood categories. We calculate its area using its latitude.

A place can be counted again in another week. This helps show flooding that keeps being detected. Adding the weekly values therefore does not give a total of different animals or different pieces of land.

For the county bar charts, we divide the weekly sums by all 1,357 weeks in the study period. Each county uses the same number of weeks. A week without a detection counts as zero, although the satellite may have missed a flood.

## Reading the monthly figure

**This figure is for all 79 counties in South Sudan added together. The numbers do not refer to one county.**

We first add up the values across the country for each week. We then take the average of all January weeks, all February weeks and so on, using 2000 to 2025. We assign a week to the month of its Monday. Each month is divided by its own number of weeks, so longer months do not automatically get higher bars.

For January, the country-wide weekly averages are about:

- **1,623 km² of land with flood detections**
- **698 km² of grazing land within those flood areas**

The 698 km² is part of the 1,623 km². Do not add them together. These are averages across January weeks in the study period, not the area of one particular flood.

The cattle estimate is highest in December, at about **38,067 cattle per week across the country**. This comes from the cattle map and the detected flood area. It is not a count of animals seen by the satellite.

The three charts use different vertical scales. They help us compare the months, while the county charts show where the highest averages occur. The older percentage chart has been replaced by this figure.

## Why the county and monthly numbers are different

The county chart shows one county per bar and averages all months together. The monthly chart adds all 79 counties together and shows each month separately.

For example, **5,900 cattle for Rubkona** is that county's average week across all months. **38,067 cattle in December** is the whole country's average week in December. These values describe different groups of weeks and different areas. Also, the county chart only shows the top ten counties, so its bars do not add up to the whole country.

Orange means cattle in both figures. Green means grazing land. Aweil county names are shown in bold in the county chart.

## Main results

Rubkona has the highest average for both cattle and grazing land in detected flood areas. Its weekly averages are about 5,900 cattle and 44.4 km² of grazing land.

Aweil East ranks 9th for the average number of cattle and 16th for the average grazing area. Within the five Aweil counties, Aweil South stands out more when we compare the results with how much cattle and grazing land each county has on its maps.

The country-wide averages are highest in January for detected flood area and grazing land. The cattle estimate is highest in December. We still need to check this pattern against local knowledge and how often the satellite could observe the area.

The data contains 58,053,688 detection records within the country boundaries. There are 40,904 combinations of a county and a week with at least one detection. We also find 8,476 runs of consecutive weeks with detections. The longest is 279 weeks. The detections can come from different places within that county, so this does not mean that the same field stayed flooded for 279 weeks.

## Saved files

The notebook saves eight tables in `outputs/national/tables/`:

| File | What it contains |
| --- | --- |
| `national_weekly_floods.csv` | Detected flood area for each county and week |
| `national_weekly_floods_by_type.csv` | The two flood categories shown separately. Do not add them because the same place can appear in both |
| `national_weekly_exposure.csv` | Estimated cattle and grazing land in detected flood areas each week |
| `national_monthly_exposure.csv` | Country-wide weekly averages for each month and the number of weeks used |
| `national_annual_detection_coverage.csv` | Dates and counts of detections. This does not show every date when the satellite could observe the area |
| `national_county_exposure_baseline.csv` | Total cattle and grazing land on the supplied maps for each county |
| `national_county_exposure_summary.csv` | County averages, rankings and other summary values |
| `national_detection_runs.csv` | Consecutive weeks with flood detections somewhere in a county |

The four figures are saved in `outputs/national/figures/`. They show the county maps, the top-ten county averages, the monthly pattern, and the largest detected areas and longest runs of detections.

## Things to keep in mind

The cattle and land maps are fixed snapshots. They do not show how herds moved or how land use changed between 2000 and 2025. The cattle map also leaves out goats and sheep.

A missing flood detection does not always mean dry land. Satellites can miss floods. The supplied cloud values are all zero, so we cannot use them to correct for missed observations.

Flooding can damage grazing land or make it harder to reach. It can also bring water and help grass grow later. Our calculations only show where the maps and flood detections meet. They do not measure these outcomes or explain changes in food insecurity.

County averages can hide differences between villages and groups of people. Herders may move their animals before a flood, while conflict can make moving harder. These results need local context before they can help with decisions about support.

The two flood categories describe how often flooding occurs. They are not labels for how damaging a flood is. Before using them in a prediction model, we need to check that the labels do not use information from after the prediction date.

## Reading the code

Start with the notebook and run its cells from top to bottom. It loads the maps, processes one year at a time, checks the results, compares counties and months, and saves the outputs. The helper files keep the map-reading and calculation steps separate from the charts.

We use ordinary loops for file loading and chart setup. Pandas groups records by county and week, while NumPy and the map libraries handle whole arrays of pixels. These array operations keep the analysis practical with millions of records; comments explain the less familiar map calculations.

## Scope

This analysis focuses on cattle and grazing land. It does not load or calculate cropland exposure. Interpreting possible crop losses would also require information such as crop type, growth stage, flood timing, depth and duration ([Molinari et al., 2019](https://nhess.copernicus.org/articles/19/2565/2019/)). This is a scope choice: cattle and grazing-land exposure also do not measure actual losses.
