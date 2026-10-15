"""Small checks for the pixel-climatology forecast maps.

Run from repo root:
    .venv/bin/python -m climatology_maps.test_forecast_maps

These checks do not need the raw flood Parquets or the LightGBM output; they
feed synthetic pixel tables and predictions through the compositing math and
verify the conservation/limit properties that the real maps rely on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from climatology_maps.pixel_climatology import pixel_area_km2
from climatology_maps.forecast_maps import (
    _validate_final_horizon,
    compose_week,
    county_month_assets,
    load_final_predictions,
    pivot_raster,
)


def _synthetic_climatology(n_pixels_per_county: int = 6) -> dict[str, pd.DataFrame]:
    """One county, 2x3 pixel block, Oct with 3 flood weeks out of 130 total."""
    county = "Aweil Test"
    lats = np.array([9.0, 9.0, 9.0, 9.2, 9.2, 9.2])
    lons = np.array([27.0, 27.2, 27.4, 27.0, 27.2, 27.4])
    n = len(lats)
    # Pixel 0 floods in all 3 flood weeks, others in 1 each -> spatial pattern.
    hit_weeks = np.array([3, 1, 1, 1, 1, 1])
    pm = pd.DataFrame({
        "county": county, "lat": lats, "lon": lons, "month": 10,
        "hit_weeks": hit_weeks, "total_weeks": 130, "flood_weeks": 4,
        "freq": hit_weeks / 130,
        "freq_given": hit_weeks / 4,
        "pixel_area_km2": pixel_area_km2(lats),
    })
    cm = pd.DataFrame({
        "county": county,
        "month": list(range(1, 13)),
        "total_weeks": 130,
        "flood_weeks": [0] * 9 + [4, 0, 0],  # October is the flood month
    })
    cm["clim_det_prob"] = (cm["flood_weeks"] + 0.5) / (cm["total_weeks"] + 1.0)
    grid = pd.DataFrame({
        "county": county,
        "lat": np.repeat(lats, 1), "lon": np.repeat(lons, 1),
        "pixel_area_km2": pixel_area_km2(lats),
    })
    annual = pm[["county", "lat", "lon", "freq"]].rename(columns={"freq": "freq_annual"})
    t = annual.groupby("county")["freq_annual"].transform("sum")
    annual["fallback_weight"] = annual["freq_annual"] / t
    return {"grid": grid, "pixel_month": pm, "county_month": cm, "annual_fallback": annual}


def _synthetic_predictions(county: str = "Aweil Test") -> pd.DataFrame:
    return pd.DataFrame([
        {"county": county, "week": pd.Timestamp("2024-10-07"),
         "det_prob": 0.8, "q10": 0.02, "q50": 0.05, "q90": 0.12,
         "det_prob_h1": 0.6, "det_prob_h2": 0.4, "det_prob_h3": 0.2, "det_prob_h4": 0.1},
        {"county": county, "week": pd.Timestamp("2024-01-08"),
         "det_prob": 0.9, "q10": 1.0, "q50": 3.0, "q90": 20.0,
         "det_prob_h1": 0.5, "det_prob_h2": 0.3, "det_prob_h3": 0.2, "det_prob_h4": 0.1},
    ])


def test_prob_is_det_prob_times_spatial_signature() -> None:
    cl = _synthetic_climatology()
    preds = _synthetic_predictions()
    assets = county_month_assets(cl)
    frame = compose_week(pd.Timestamp("2024-10-07"), preds, cl, assets)
    p0 = frame.set_index(["lat", "lon"]).loc[(9.0, 27.0), "prob"]
    assert abs(p0 - 0.8 * (3 / 4)) < 1e-12, f"expected 0.6 = 0.8*3/4, got {p0}"
    # pixel 1 had 1/4 of flood weeks
    p1 = frame.set_index(["lat", "lon"]).loc[(9.0, 27.2), "prob"]
    assert abs(p1 - 0.8 * (1 / 4)) < 1e-12
    print("  ok: prob = det_prob x freq_given (per-pixel signature)")


def test_severity_sum_conserves_predicted_area() -> None:
    cl = _synthetic_climatology()
    preds = _synthetic_predictions()
    assets = county_month_assets(cl)
    frame = compose_week(pd.Timestamp("2024-10-07"), preds, cl, assets)
    hist_area = float((cl["pixel_month"]["freq"] * cl["pixel_month"]["pixel_area_km2"]).sum())
    implied = float((frame["sev"] * frame["pixel_area_km2"]).sum())
    # scale = q50 / hist_area, no clipping here -> implied == q50 exactly
    assert abs(implied - 0.05) < 1e-9, f"implied {implied} != q50 0.05"
    assert hist_area > 0
    print(f"  ok: sum(sev x area) = {implied:.6f} reproduces q50 = 0.05 "
          f"(hist monthly mean {hist_area:.6f} km2)")


def test_worst_case_uses_q90() -> None:
    cl = _synthetic_climatology()
    preds = _synthetic_predictions()
    assets = county_month_assets(cl)
    frame = compose_week(pd.Timestamp("2024-10-07"), preds, cl, assets)
    assert (frame["worst"] >= frame["sev"]).all()
    hi = float((frame["worst"] * frame["pixel_area_km2"]).sum())
    assert abs(hi - 0.12) < 1e-9, f"implied worst {hi} != q90 0.12"
    print(f"  ok: worst-case sum = {hi:.6f} reproduces q90 = 0.12")


def test_no_history_month_uses_annual_pattern_without_magnitude() -> None:
    cl = _synthetic_climatology()
    preds = _synthetic_predictions()
    assets = county_month_assets(cl)
    # January: county never flooded in January historically.
    frame = compose_week(pd.Timestamp("2024-01-08"), preds, cl, assets)
    # Placement still sums to the model probability (annual normalised pattern).
    assert abs(frame["prob"].sum() - 0.9) < 1e-9, frame["prob"].sum()
    # No historical magnitude -> severity layers are zero.
    assert (frame["sev"] == 0).all() and (frame["worst"] == 0).all()
    print("  ok: no-history month falls back to annual pattern (prob sums to det_prob), "
          "severity = 0")


def test_duration_horizon_swaps_event_probability() -> None:
    cl = _synthetic_climatology()
    preds = _synthetic_predictions()
    assets = county_month_assets(cl)
    f0 = compose_week(pd.Timestamp("2024-10-07"), preds, cl, assets, horizon=0)
    f2 = compose_week(pd.Timestamp("2024-10-07"), preds, cl, assets, horizon=2)
    # same spatial signature, different event probability (h2 = 0.4 vs 0.8)
    ratio = f2["prob"].to_numpy() / f0["prob"].to_numpy()
    assert np.allclose(ratio, 0.4 / 0.8), ratio
    print("  ok: horizon h swaps det_prob (prob scaled by 0.4/0.8 as expected)")


def test_pivot_raster_has_nan_outside_domain() -> None:
    cl = _synthetic_climatology()
    preds = _synthetic_predictions()
    assets = county_month_assets(cl)
    frame = compose_week(pd.Timestamp("2024-10-07"), preds, cl, assets)
    # drop one specific (lat, lon) cell while its row/column remain via others
    frame = frame[~((frame["lat"] == 9.0) & (frame["lon"] == 27.2))]
    z, lons, lats = pivot_raster(frame, "prob")
    assert np.isnan(z).any(), "expected a NaN hole where a cell is outside the domain"
    assert 27.2 in lons and 9.0 in lats  # row/column still present
    print("  ok: raster has NaN outside the domain (rendered as no data)")


def test_load_final_predictions_sorts_and_reports_horizons() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "final_forecast_aweil.csv"
        pd.DataFrame([
            {"county": "B", "week": "2025-12-29", "det_prob": 0.1, "q10": 0.0,
             "q50": 0.01, "q90": 0.05, "det_prob_h1": 0.08, "det_prob_h2": 0.06},
            {"county": "A", "week": "2025-12-29", "det_prob": 0.2, "q10": 0.0,
             "q50": 0.02, "q90": 0.09, "det_prob_h1": 0.15, "det_prob_h2": 0.12},
        ]).to_csv(path, index=False)
        df, horizons = load_final_predictions(path)
    assert horizons == [1, 2], horizons
    assert list(df["county"]) == ["A", "B"]  # sorted by county, then week
    assert (df["week"] == pd.Timestamp("2025-12-29")).all()
    print("  ok: final table loads, sorts by county/week, reports duration heads h1/h2")


def test_load_final_predictions_missing_columns_raise() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.csv"
        pd.DataFrame([{"county": "A", "week": "2025-12-29", "det_prob": 0.2}]).to_csv(
            path, index=False)
        try:
            load_final_predictions(path)
        except ValueError as exc:
            assert "q50" in str(exc) and "q90" in str(exc), exc
        else:
            raise AssertionError("expected ValueError for missing columns")
    print("  ok: final table with missing columns raises a clear ValueError")


def test_final_horizon_guard_rejects_heads_the_table_lacks() -> None:
    # h0 (same week) always exists; the final table ships h1/h2 only.
    _validate_final_horizon([1, 2], 0)
    _validate_final_horizon([1, 2], 1)
    _validate_final_horizon([1, 2], 2)
    for bad in (3, 4):
        try:
            _validate_final_horizon([1, 2], bad)
        except SystemExit as exc:
            assert "h3/h4" in str(exc), exc
        else:
            raise AssertionError(f"expected SystemExit for h{bad}")
    try:
        _validate_final_horizon([], 1)
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit when the table ships no heads")
    print("  ok: final-mode horizon guard accepts h0/h1/h2, rejects h3/h4 and empty")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\nAll {len(tests)} checks passed.")