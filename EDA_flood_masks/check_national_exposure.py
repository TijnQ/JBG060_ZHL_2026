"""Run with: python -m EDA_flood_masks.check_national_exposure."""

import numpy as np
import pandas as pd
from affine import Affine
from EDA_flood_masks.flood_eda import pixel_area_km2

from EDA_flood_masks.cattle_rangeland_exposure import (
    _grid_area_km2,
    calculate_weekly_exposure,
)


def run_checks():
    """Check deduplication, land fractions and proportional cattle exposure."""
    week = pd.Timestamp("2024-08-05")
    observations = pd.DataFrame({
        "state": ["State A"] * 4,
        "county": ["County A"] * 4,
        "county_code": ["AA01"] * 4,
        "week": [week] * 4,
        "lat": [9.95, 9.95, 9.95, 9.85],
        "lon": [26.05, 26.05, 26.15, 26.05],
        "flood_type": ["recurring", "unusual", "recurring", "recurring"],
    })
    fine_transform = Affine.translation(26, 10) * Affine.scale(0.1, -0.1)
    cattle_transform = Affine.translation(26, 10) * Affine.scale(0.2, -0.2)
    layers = {
        "rangeland": {"data": np.ma.array([[50, 100], [0, 25]]), "transform": fine_transform},
        "cattle": {"data": np.ma.array([[100.0]]), "transform": cattle_transform},
    }
    result = calculate_weekly_exposure(observations, layers).iloc[0]

    areas = pixel_area_km2(np.array([9.95, 9.95, 9.85]))
    assert np.isclose(result["flooded_rangeland_km2"], 0.5 * areas[0] + areas[1])
    cattle_area = _grid_area_km2(9.9, 0.2, 0.2)
    assert np.isclose(result["potential_cattle_exposed"], 100 * areas.sum() / cattle_area)
    print("Passed: weekly union, land fractions and proportional cattle exposure.")


if __name__ == "__main__":
    run_checks()
