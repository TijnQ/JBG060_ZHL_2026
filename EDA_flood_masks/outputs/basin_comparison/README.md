# White Nile/Sudd and River Lol: 2018–2019 comparison

The approved figure is `consecutive_years_annual_contrast.png` (and editable `.svg`). Only its grey subtitle and footer were removed. All bar positions, colours, year labels and numerical labels are unchanged.

## Meaning of the bars

Each bar is the mean weekly detected area in km² over the indicated year's Monday-based weeks, rounded to a whole km². Recurring and unusual flood detections are combined and each grid cell is counted once within a week. The same cell can count again in another week. Zero-detection weeks are included as zero detections, not as verified dry observations.

| Proposed region | 2018 | 2019 |
| --- | --- | --- |
| White Nile/Sudd corridor | 16 km² per average week | 9 km² per average week |
| Lol catchment | 7 km² per average week | 21 km² per average week |

2018 uses 53 weeks starting on Mondays in 2018, including the final week extending into January 2019. 2019 uses 52 such weeks, with the same rule. Exact values, denominators and cumulative area-week totals are in `consecutive_years_annual_contrast_values.csv`. These are absolute detected areas, not basin-area percentages, total annual unique area, flood peaks or water volumes. Both panels have the same linear scale. Some flooding was detected in both regions.

## Reproduce

Using the project's existing Python environment, from the repository root:

```powershell
.venv/Scripts/python.exe -m EDA_flood_masks.basin_contrast
```

This reads the included `all_years_annual_comparison.csv` and `all_years_weekly_floods.csv` and regenerates the final figure and its value table. The script also generates an optional weekly diagnostic plot. A small runnable check ensures that missing years cannot be treated as consecutive years.

To rebuild the intermediate tables from the supplied raw flood files:

```powershell
.venv/Scripts/python.exe -m EDA_flood_masks.basin_boundaries --all-years
```

This downloads the official HydroBASINS archive into ignored `raw_data/hydrobasins/` if needed. Raw MODIS files for 2000–2025 and the project's administrative boundaries must be present. The script checks region validity and routing, non-overlap, weekly pixel deduplication and valid area totals.

## Transparent year selection

The scan covers 2000–2024. A complete last Monday-based week in 2025 requires January 2026 inputs, which are not supplied. The selected pair is the latest adjacent pair whose mean weekly detected-area shares changed in opposite directions. Five of 24 pairs satisfy this criterion, and the full scan is included in `consecutive_year_pair_scan.csv`. Because each region's analysis area is fixed, the direction of change is the same for area shares and absolute km². This is an illustrative pair and does not establish hydrological independence.

## Regions, sources and limitations

The regional outlines are **candidates**, not validated officially named basin boundaries. HydroBASINS provides geometry and routing identifiers, not the regional names used in the plot. The White Nile/Sudd candidate uses PFAF prefix 1727 (a corridor/incremental region, not the full upstream White Nile catchment or an official wetland boundary). The Lol candidate uses prefix 17286. The definitions and exact member units are included in `region_definitions.csv`, `proposed_regions.geojson` and `source_subbasins.geojson`.

The analysis polygons are clipped to the nominal supplied h20v08/h21v08 footprint, 20–40°E and 0–10°N. The same clipped areas are used in both years. Clipped geometry is in `candidate_analysis_footprints.geojson`. Pixel and polygon area calculations use the same sphere radius, 6,371,007.181 m. Optical cloud/observation differences remain a limitation. No detection does not establish dry conditions, and county boundaries are not river-basin boundaries.

Source: [HydroBASINS v1.c Africa standard level 6](https://www.hydrosheds.org/products/hydrobasins), Lehner & Grill (2013), [DOI 10.1002/hyp.9740](https://doi.org/10.1002/hyp.9740). Download URL and SHA-256 are recorded in `provenance.json`. All underlying flood records come from the project's supplied compact recurring/unusual MODIS Parquet files. The older mixed-unit 2023/2024 poster is not replaced by this figure.
