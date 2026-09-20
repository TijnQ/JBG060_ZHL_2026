"""Run with: python -m EDA_hydrometeorology.check_hydrological_analysis."""

import numpy as np
import pandas as pd
import xarray as xr

from EDA_hydrometeorology.hydrological_analysis import (
    era5_daily_table,
    era5_spatial_summary,
    et_daily_table,
    flood_event_metrics,
    lake_altimetry_quality,
    lagged_correlation,
    monthly_climatology,
    percentile_extremes,
    quality_report,
    water_budget,
)


def run_checks():
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    hydro = pd.DataFrame({
        "date": dates,
        "precipitation_mm": [10, 0, 4, 8, 2, 1, 5, 6, 7, 9],
        "runoff_mm": [2, 0, 1, 4, 0, 0, 1, 1, 2, 3],
    })
    et = pd.DataFrame({"date": dates, "gridcell": 3.0})

    budget = water_budget(hydro, et)
    assert np.isclose(budget.loc[0, "net_water_balance_mm"], 22)
    assert np.isclose(budget.loc[0, "runoff_coefficient"], 14 / 52)
    assert pd.isna(water_budget(hydro).loc[0, "net_water_balance_mm"])
    assert len(monthly_climatology(hydro, ["precipitation_mm"])) == 1

    thresholds, extremes = percentile_extremes(hydro, "precipitation_mm")
    assert len(thresholds) == 2 and len(extremes) == 1
    correlation = lagged_correlation(
        pd.Series([1, 2, 3], index=dates[:3]),
        pd.Series([1, 2, 3], index=dates[:3]),
        max_lag=1,
    )
    assert correlation["correlation"].notna().any()

    flood = pd.DataFrame({
        "date": dates[:4], "lat": [1, 1, 1, 1], "lon": [2, 2, 2, 2],
        "flood_type": [1, 1, 0, 0],
    })
    assert flood_event_metrics(flood)["max_duration_days"].max() == 2

    quality = quality_report(
        pd.Series([1, np.nan], index=dates[:2]), "test", dates[0], dates[2]
    )
    assert quality.loc[0, "missing_values"] == 1

    lakes = {"victoria": pd.DataFrame({
        "height_wrt_ref": [1000.0, 1001.0],
        "mode1": [4, 2], "mode2": [0, 0], "ice_flag": [0, 0],
        "data_source_flag": [0, 0],
    }, index=dates[:2])}
    lake_quality = lake_altimetry_quality(lakes)
    assert lake_quality["valid"].tolist() == [True, False]

    raw = xr.Dataset({
        "tp": (("valid_time", "latitude", "longitude"), np.ones((2, 1, 1)) / 1000),
        "ro": (("valid_time", "latitude", "longitude"), np.ones((2, 1, 1)) / 2000),
    }, coords={"valid_time": dates[:2], "latitude": [2], "longitude": [30]})
    assert era5_daily_table(raw).loc[0, "precipitation_mm"] == 1
    assert era5_spatial_summary(raw).loc[0, "mean_precipitation_mm_per_day"] == 1
    assert et_daily_table(et)["et_mm"].notna().all()
    print("Passed: water budget, units, climatology, extremes, lag, floods, lakes and quality checks.")


if __name__ == "__main__":
    run_checks()
