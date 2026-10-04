"""Small checks for the flood-climatology map module.

Run from repo root:
    .venv/bin/python -m climatology_maps.test_climatology_maps

These checks do not need the raw per-pixel flood files; they use the committed
national weekly table and a synthetic geometry-free head-count.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from climatology_maps.climatology_maps import (
    full_county_week_grid,
    load_boundaries,
    load_historical_floods,
    monthly_climatology,
)


def test_grid_counts_zero_detections() -> None:
    weeks = pd.Series(pd.to_datetime(["2020-10-05", "2020-10-12"]))
    grid = full_county_week_grid(pd.Index(["SS0101"]), weeks)
    assert len(grid) == 2
    assert (grid["month"] == 10).all()
    assert grid["year"].iloc[0] == 2020
    print("  ok: county x week grid expands weeks including zero-detection weeks")


def test_climatology_known_small() -> None:
    # Synthetic: one county, two Octobers. Week1 floods (10 km2), week2 dry.
    floods = pd.DataFrame(
        {
            "county_code": ["SS0101", "SS0101"],
            "week": pd.to_datetime(["2020-10-05", "2021-10-11"]),
            "detected_area_km2": [10.0, 0.0],
            "county_area_km2": [100.0, 100.0],
        }
    )
    boundaries = gpd.GeoDataFrame(
        {
            "county_code": ["SS0101"],
            "state": ["S"], "county": ["Juba"],
            "geometry": [box(0, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )
    floods["year"] = floods["week"].dt.year
    floods["month"] = floods["week"].dt.month
    stats = monthly_climatology(floods, boundaries, baseline_years=(2020, 2021))
    row = stats[(stats["month"] == 10) & (stats["county_code"] == "SS0101")].iloc[0]
    assert abs(row["mean_weekly_flood_km2"] - 5.0) < 1e-9, row["mean_weekly_flood_km2"]
    assert abs(row["flood_frequency"] - 0.5) < 1e-9
    assert abs(row["mean_weekly_water_km3"] - 5.0 * 1.0 / 1000.0) < 1e-12
    print("  ok: climatology mean/frequency/km3 over a flat 50%-flooded county")


def test_real_data_shapes() -> None:
    floods = load_historical_floods()
    boundaries = load_boundaries()
    stats = monthly_climatology(floods, boundaries, baseline_years=(2000, 2025))
    assert "county" in stats.columns
    assert stats["county_code"].nunique() >= 77
    assert set(stats["month"]) == set(range(1, 13))
    oct = stats[stats["month"] == 10]
    top = oct.nlargest(1, "mean_weekly_flood_km2").iloc[0]
    print(f"  ok: real data climatology computed; "
          f"{oct['county_code'].nunique()} counties x 12 months; "
          f"top October mean: {top['mean_weekly_flood_km2']:.2f} km2 "
          f"({top['county']}, {top['mean_weekly_water_km3']:.4f} km3)")


if __name__ == "__main__":
    print("Running climatology map checks ...")
    test_grid_counts_zero_detections()
    test_climatology_known_small()
    test_real_data_shapes()
    print("All checks passed.")