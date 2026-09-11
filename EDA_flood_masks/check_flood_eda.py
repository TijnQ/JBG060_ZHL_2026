"""Run with: python -m EDA_flood_masks.check_flood_eda (no raw data required)."""

import numpy as np
import pandas as pd
from pyproj import Geod

from EDA_flood_masks.flood_eda import (
    GRID_STEP_DEGREES,
    PRODUCT_RADIUS_M,
    find_detected_events,
    pixel_area_km2,
    summarise_seasonality,
    summarise_weekly,
    validate_outputs,
)


def run_checks():
    """Use a small dataset to check pixel counts, areas and consecutive weeks."""
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03",
                "2024-01-03", "2024-01-08", "2024-01-22", "2024-01-01",
            ]),
            "lat": [9.0] * 8,
            "lon": [27.0, 27.0, 27.0, 27.1, 27.1, 27.0, 27.1, 27.0],
            "county": ["A"] * 7 + ["B"],
            "flood_type": [
                "recurring", "unusual", "unusual", "unusual",
                "unusual", "recurring", "unusual", "recurring",
            ],
            "cloud_frac": [0.0] * 8,
        }
    )
    weekly = summarise_weekly(observations)
    classified = summarise_weekly(observations, by_type=True)
    area = float(pixel_area_km2(9.0))
    first = weekly.loc[weekly["county"].eq("A")].iloc[0]
    assert first["unique_grid_cells"] == 2
    assert first["detected_pixel_days"] == 3
    assert np.isclose(first["detected_area_km2"], 2 * area)
    assert classified.loc[
        classified["county"].eq("A") & classified["week"].eq(first["week"]),
        "unique_grid_cells",
    ].sum() == 3
    validate_outputs(observations, weekly)
    _check_area(area)
    _check_runs_and_seasons(weekly, classified, area)
    summarise_weekly(observations.loc[observations["county"].eq("B")])
    print("Passed: class overlap, repeated days, cell area, gaps, monthly means and single-class data.")


def _check_area(area):
    """Check our pixel area against the area calculated by pyproj."""
    geod = Geod(a=PRODUCT_RADIUS_M, b=PRODUCT_RADIUS_M)
    half = GRID_STEP_DEGREES / 2
    reference, _ = geod.polygon_area_perimeter(
        [27 - half, 27 + half, 27 + half, 27 - half],
        [9 - half, 9 - half, 9 + half, 9 + half],
    )
    assert np.isclose(area, abs(reference) / 1e6, rtol=1e-7, atol=0)
    assert 0.0525 < area < 0.0535
    assert pixel_area_km2(60) < pixel_area_km2(9) < pixel_area_km2(0)


def _check_runs_and_seasons(weekly, classified, area):
    runs = find_detected_events(weekly)
    runs_a = runs.loc[runs["county"].eq("A")]
    assert runs_a["county_detected_weeks"].tolist() == [2, 1]
    assert np.allclose(runs_a["peak_weekly_union_area_km2"], [2 * area, area])
    assert np.allclose(runs_a["cumulative_detected_area_km2_weeks"], [3 * area, area])
    zero = weekly.iloc[[0]].assign(week=pd.Timestamp("2024-01-15"), detected_area_km2=0)
    pd.testing.assert_frame_equal(runs, find_detected_events(pd.concat([weekly, zero])))
    monthly = summarise_seasonality(weekly, ["A", "B", "C"], years=[2023, 2024])
    january = monthly.loc[monthly["county"].eq("A") & monthly["month"].eq(1)].iloc[0]
    assert np.isclose(january["mean_weekly_detected_area_km2"], (4 * area / 5) / 2)
    assert january["weeks_with_detection"] == 3
    assert january["calendar_weeks"] == 10
    assert january["years_included"] == 2
    assert monthly.loc[monthly["county"].eq("C"), "mean_weekly_detected_area_km2"].eq(0).all()
    for downstream in (
        find_detected_events,
        lambda data: summarise_seasonality(data, ["A", "B"], years=[2024]),
    ):
        try:
            downstream(classified)
        except ValueError as error:
            assert "county-week totals" in str(error)
        else:
            raise AssertionError("Class-specific areas must never be silently summed")


if __name__ == "__main__":
    run_checks()
