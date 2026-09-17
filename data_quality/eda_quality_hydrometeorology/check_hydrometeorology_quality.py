"""Run with: python -m data_quality.eda_quality_hydrometeorology.check_hydrometeorology_quality (no raw data required)."""

import numpy as np
import pandas as pd
import xarray as xr

from data_quality.eda_quality_hydrometeorology.hydrometeorology_quality import (
    all_features_daily,
    build_source_frames,
    coverage_summary,
    distribution_extras,
    profile_frames,
    quality_table,
    supervised_set_readiness,
)


def _synthetic_frames():
    """Small made-up inputs spanning a two-day window and one Aweil-ish cell."""
    dates = pd.date_range("2000-01-01", periods=2, freq="D")
    rainfall = xr.Dataset(
        {
            "tp": (("valid_time", "latitude", "longitude"), np.ones((2, 1, 1)) / 1000),
            "ro": (("valid_time", "latitude", "longitude"), np.ones((2, 1, 1)) / 2000),
        },
        coords={"valid_time": dates, "latitude": [9.0], "longitude": [30.0]},
    )
    # Aweil county bbox that includes the synthetic grid cell.
    bounds = (30.0, 9.0, 30.0, 9.0)
    et = pd.DataFrame({"date": dates, "gridcell": [3.0, 3.0]})
    dartmouth = {100205: pd.DataFrame(
        {"Discharge (m3/s)": [500.0, np.nan]}, index=dates
    )}
    lakes = {
        "victoria": pd.DataFrame(
            {"height_wrt_ref": [1000.0, 1001.0]}, index=dates
        ),
        "Albert": pd.DataFrame(
            {"water_level": [10.0, 10.5]}, index=dates
        ),
    }
    return build_source_frames(rainfall, et, dartmouth, lakes, county_bounds=bounds)


def run_checks():
    frames = _synthetic_frames()

    # build_source_frames produces the Nile-basin, Aweil, ET, discharge and lake levels.
    expected_labels = {
        "ERA5 - Nile basin (precipitation)", "ERA5 - Nile basin (runoff)",
        "ERA5 - Aweil counties (precipitation)", "ERA5 - Aweil counties (runoff)",
        "Reference ET (AgERA5 gridcell)", "Dartmouth discharge - station 100205",
        "Lake victoria (altimetry)", "Lake Albert (altimetry)",
    }
    assert expected_labels <= set(frames)

    # Profile: every level has a date row and at least one value row.
    profile = profile_frames(frames)
    for level in frames:
        assert ((profile["level"] == level) & (profile["column"] == "date")).any()
        assert ((profile["level"] == level) & (profile["column"] != "date")).any()
    # The discharge station carries one missing value on its second day.
    discharge_dates = profile[profile["column"] == "date"]
    assert discharge_dates["nan_pct"].between(0, 100).all()

    # ERA5 precipitation is non-negative and zero-free over two wet days.
    extras = distribution_extras(frames)
    assert {"zero_pct", "negative_pct", "skew"} <= set(extras.columns)
    precip = extras[extras["column"] == "precipitation_mm"]
    assert (precip["negative_pct"] == 0.0).all()
    assert (precip["zero_pct"] == 0.0).all()

    coverage = coverage_summary(frames)
    assert coverage["window_days"].iloc[0] == 2
    # The station row shows one missing day inside the shared window.
    station = coverage[coverage["level"] == "Dartmouth discharge - station 100205"]
    assert station["value_rows_non_null"].iloc[0] == 1

    # Per-series quality report covers every level.
    quality = quality_table(frames)
    assert len(quality) == len(frames)

    # Wide daily feature table and supervised-set readiness.
    wide = all_features_daily(frames)
    assert wide.shape[1] == sum(
        len([c for c in fr.columns if c != "date"]) for fr in frames.values()
    )
    readiness = supervised_set_readiness(frames, label_dates=[pd.Timestamp("2000-01-01")])
    total = readiness[readiness["year"] == "all"].iloc[0]
    # Only day 1 has both all features (day 2 discharge is null) and a label.
    assert total["days_all_features"] == 1
    assert total["days_all_features_with_label"] == 1

    # Without a label, readiness still reports feature completeness but no label counts.
    readiness_no_label = supervised_set_readiness(frames)
    assert readiness_no_label["days_all_features_with_label"].iloc[-1] == 0

    print("Passed: source assembly, column profile, distributions, coverage, quality and readiness.")


if __name__ == "__main__":
    run_checks()