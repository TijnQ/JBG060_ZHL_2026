"""Run with: python -m risk_exposure.check_forecast_risk_map.

The checks use a tiny synthetic pixel grid, two synthetic exposure GeoTIFFs
and a synthetic climatology, so no raw data is needed. They verify the
forecast composition (prob = det_prob x freq_given, duration heads on the
same footprint), the cattle expectation (density x pixel area, with the
prob=1 ceiling reproducing the mapped herd exactly), the crop mask-cell rule
(max pixel probability, count-once, prob x area allocation back to pixels),
the summary table (target weeks, prob heads, percentages, behaviour without
duration heads) and that the renderer writes PNGs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from shapely.geometry import box

matplotlib.use("Agg")

from climatology_maps.forecast_maps import county_month_assets
from risk_exposure.forecast_risk_map import (
    build_forecast_risk_maps,
    cattle_ceiling,
    load_asset_layers,
    load_forecast,
    select_weeks,
    summarise_week,
    week_frames,
)

WEEK = pd.Timestamp("2024-10-07")
COUNTIES = ("Alpha", "Beta")
LON_STEPS = (0.005, 0.015, 0.025, 0.035, 0.045)   # pixel centres inside a county
LAT_STEPS = (0.005, 0.015, 0.025, 0.035, 0.045)
CATTLE_STEP = 0.05       # coarse cattle cells: exactly 5x5 grid pixels each
CROP_STEP = 0.02         # crop mask cells: exactly 2x2 grid pixels each
CROP_FRACTION = 40.0


def _grid_area_km2_local(latitude, lon_step: float, lat_step: float) -> np.ndarray:
    """Mirror of the module's cell-area formula for expected-value math."""
    latitude = np.asarray(latitude, dtype=float)
    half = lat_step / 2
    south = np.deg2rad(latitude - half)
    north = np.deg2rad(latitude + half)
    return 6371007.181**2 * np.deg2rad(lon_step) * (np.sin(north) - np.sin(south)) / 1e6


def _synthetic_grid() -> pd.DataFrame:
    """50 grid pixels (25 per county) whose areas fill their cattle cell."""
    cell_area = float(_grid_area_km2_local(9.525, CATTLE_STEP, CATTLE_STEP))
    rows = []
    for county, lon_base in zip(COUNTIES, (27.0, 27.05)):
        for lat in (9.5 + step for step in LAT_STEPS):
            for lon in (lon_base + step for step in LON_STEPS):
                rows.append({
                    "county": county, "lat": lat, "lon": lon,
                    "pixel_area_km2": cell_area / 25.0,
                })
    return pd.DataFrame(rows)


def _synthetic_climatology(grid: pd.DataFrame, mode: str) -> dict[str, pd.DataFrame]:
    """County x month=10 climatology with a uniform or concentrated signature.

    ``uniform`` spreads freq_given evenly (1/25 per pixel); ``concentrated``
    puts it on the four Alpha pixels of one crop mask cell (lons 27.005/27.015
    x lats 9.505/9.515). Both signatures carry the same total area weight,
    so the cattle expectation must come out identical for the two modes.
    """
    if mode == "uniform":
        freq_given = np.full(len(grid), 1.0 / 25.0)
        freq = np.full(len(grid), 0.04)
    else:
        concentrated = (
            (grid["county"] == "Alpha")
            & grid["lon"].isin((27.005, 27.015))
            & grid["lat"].isin((9.505, 9.515))
        ).to_numpy()
        freq_given = np.where(concentrated, 0.25, 0.0)
        freq = np.full(len(grid), 0.04)
        freq[concentrated] = 0.04
    pixel_month = pd.DataFrame({
        "county": grid["county"], "month": 10,
        "lat": grid["lat"], "lon": grid["lon"],
        "freq": freq, "freq_given": freq_given,
        "pixel_area_km2": grid["pixel_area_km2"],
    })
    county_month = pd.DataFrame({
        "county": list(COUNTIES), "month": 10, "total_weeks": 52,
        "flood_weeks": 5, "clim_det_prob": 0.1,
    })
    annual_fallback = grid.assign(fallback_weight=0.0)[
        ["county", "lat", "lon", "fallback_weight"]
    ]
    return {
        "grid": grid.reset_index(drop=True),
        "pixel_month": pixel_month.reset_index(drop=True),
        "county_month": county_month,
        "annual_fallback": annual_fallback,
    }


def _synthetic_predictions(with_heads: bool = True) -> pd.DataFrame:
    rows = [
        {"county": "Alpha", "week": WEEK, "det_prob": 0.4, "q10": 0.0,
         "q50": 5.0, "q90": 10.0},
        {"county": "Beta", "week": WEEK, "det_prob": 0.0, "q10": 0.0,
         "q50": 0.0, "q90": 0.0},
    ]
    if with_heads:
        rows[0].update(det_prob_h1=0.2, det_prob_h2=0.1)
        rows[1].update(det_prob_h1=0.0, det_prob_h2=0.0)
    return pd.DataFrame(rows)


def _synthetic_totals() -> pd.DataFrame:
    return pd.DataFrame({
        "county": list(COUNTIES),
        "mapped_cattle": [300.0, 200.0],
        "crop_hectares": [10_000.0, 8_000.0],
    })


def _synthetic_boundaries() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"county": list(COUNTIES)},
        geometry=[
            box(26.99, 9.49, 27.06, 9.56),
            box(27.04, 9.49, 27.11, 9.56),
        ],
        crs="EPSG:4326",
    )


def _write_raster(path: Path, data: np.ndarray, transform: Affine) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as target:
        target.write(data, 1)


def _write_rasters(directory: Path) -> tuple[Path, Path]:
    """Cattle counts on 0.05 deg cells; a 40% crop fraction on 0.02 deg cells.

    Grid alignment: all 25 Alpha pixels fall in cattle cell (row 1, col 2)
    and all 25 Beta pixels in (row 1, col 3); Alpha's lons land in crop cols
    1-3 and Beta's in cols 3-5, both within crop rows 3-5.
    """
    cattle_path = directory / "cattle.tif"
    cattle = np.zeros((4, 8), dtype=np.float32)
    cattle[1, 2] = 300.0
    cattle[1, 3] = 200.0
    _write_raster(
        cattle_path, cattle, Affine(CATTLE_STEP, 0, 26.9, 0, -CATTLE_STEP, 9.6)
    )
    crop_path = directory / "crop.tif"
    crop = np.zeros((10, 12), dtype=np.float32)
    crop[3:6, 1:6] = CROP_FRACTION
    _write_raster(
        crop_path, crop, Affine(CROP_STEP, 0, 26.98, 0, -CROP_STEP, 9.62)
    )
    return cattle_path, crop_path


def _cell_ha() -> float:
    """Area of one synthetic crop cell in hectares (module formula)."""
    return CROP_STEP**2 * 111_320.0**2 / 10_000


def check_load_forecast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pred.csv"
        pd.DataFrame({"county": ["A"], "week": ["2024-10-07"]}).to_csv(path, index=False)
        try:
            load_forecast(path)
        except ValueError as error:
            assert "det_prob" in str(error)
        else:
            raise AssertionError("missing forecast columns must raise")
        path = Path(tmp) / "pred_heads.csv"
        _synthetic_predictions().to_csv(path, index=False)
        table, horizons = load_forecast(path)
        assert horizons == [1, 2]
        assert table["week"].iloc[0] == WEEK
        path = Path(tmp) / "pred_no_heads.csv"
        _synthetic_predictions(with_heads=False).to_csv(path, index=False)
        _, horizons = load_forecast(path)
        assert horizons == []


def check_week_frames_uniform_signature() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cattle_path, crop_path = _write_rasters(Path(tmp))
        grid = _synthetic_grid()
        layers = load_asset_layers(grid, cattle_path=cattle_path, crop_path=crop_path)
        climatology = _synthetic_climatology(grid, "uniform")
        assets = county_month_assets(climatology)
        predictions = _synthetic_predictions()
        frames = week_frames(WEEK, predictions, climatology, assets, layers, [1, 2])
        assert sorted(frames) == [0, 1, 2]
        alpha = frames[0]
        alpha = alpha[alpha["county"] == "Alpha"]
        assert np.allclose(alpha["prob"], 0.4 / 25.0)
        beta = frames[0]
        beta = beta[beta["county"] == "Beta"]
        assert np.allclose(beta["prob"], 0.0)
        # Cattle: density x pixel area x prob summed = det_prob x (count / 25).
        cattle = frames[0].groupby("county", observed=True)["exp_cattle"].sum()
        assert np.isclose(cattle["Alpha"], 0.4 * 300.0 / 25.0)
        assert np.isclose(cattle["Beta"], 0.0)
        # Crop: one count-once mask cell per (col, row) combo -> 9 cells.
        cell_ha = _cell_ha()
        crop = frames[0].groupby("county", observed=True)["exp_crop"].sum()
        expected_crop = 9 * (0.4 / 25.0) * cell_ha * CROP_FRACTION / 100.0
        assert np.isclose(crop["Alpha"], expected_crop)
        assert np.isclose(crop["Beta"], 0.0)
        # Duration heads reuse the same spatial signature.
        alpha_h1 = frames[1]
        alpha_h1 = alpha_h1[alpha_h1["county"] == "Alpha"]
        assert np.allclose(alpha_h1["prob"], 0.2 / 25.0)
        cattle_h1 = frames[1].groupby("county", observed=True)["exp_cattle"].sum()
        assert np.isclose(cattle_h1["Alpha"], 0.2 * 300.0 / 25.0)
        # Expected wet area = det_prob x signature area (sum freq_given x area).
        signature_area = float((alpha["freq_given"] * alpha["pixel_area_km2"]).sum())
        summary = summarise_week(WEEK, frames, predictions, _synthetic_totals())
        wet = summary[
            (summary["county"] == "Alpha") & (summary["horizon"] == 0)
        ]["expected_wet_area_km2"].iloc[0]
        assert np.isclose(wet, 0.4 * signature_area)
        # A week without predictions yields no frames.
        assert week_frames(
            WEEK + pd.Timedelta(weeks=4), predictions, climatology, assets,
            layers, [1, 2],
        ) == {}


def check_concentrated_signature_and_ceiling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cattle_path, crop_path = _write_rasters(Path(tmp))
        grid = _synthetic_grid()
        layers = load_asset_layers(grid, cattle_path=cattle_path, crop_path=crop_path)
        climatology = _synthetic_climatology(grid, "concentrated")
        assets = county_month_assets(climatology)
        frames = week_frames(
            WEEK, _synthetic_predictions(), climatology, assets, layers, []
        )
        alpha = frames[0]
        alpha = alpha[alpha["county"] == "Alpha"]
        wet = alpha["prob"] > 0
        assert int(wet.sum()) == 4
        assert np.allclose(alpha.loc[wet, "prob"], 0.4 * 0.25)
        assert np.allclose(alpha.loc[~wet, "prob"], 0.0)
        # Cattle invariance: same signature area -> same expectation as uniform.
        cattle = frames[0].groupby("county", observed=True)["exp_cattle"].sum()
        assert np.isclose(cattle["Alpha"], 0.4 * 300.0 / 25.0)
        # Crop: only the one concentrated mask cell is at risk, counted once.
        crop = frames[0].groupby("county", observed=True)["exp_crop"].sum()
        expected_crop = (0.4 * 0.25) * _cell_ha() * CROP_FRACTION / 100.0
        assert np.isclose(crop["Alpha"], expected_crop)
        per_pixel = alpha.loc[wet, "exp_crop"].to_numpy()
        assert np.allclose(per_pixel, expected_crop / 4.0)
        assert np.allclose(alpha.loc[~wet, "exp_crop"], 0.0)
        # Ceiling: prob = 1 everywhere reproduces the mapped herd exactly.
        ceiling = cattle_ceiling(grid, layers["cattle"]).set_index("county")
        assert np.isclose(ceiling.loc["Alpha", "cattle_ceiling_prob1"], 300.0)
        assert np.isclose(ceiling.loc["Beta", "cattle_ceiling_prob1"], 200.0)


def check_summary_table() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cattle_path, crop_path = _write_rasters(Path(tmp))
        grid = _synthetic_grid()
        layers = load_asset_layers(grid, cattle_path=cattle_path, crop_path=crop_path)
        climatology = _synthetic_climatology(grid, "uniform")
        assets = county_month_assets(climatology)
        predictions = _synthetic_predictions()
        frames = week_frames(WEEK, predictions, climatology, assets, layers, [1, 2])
        summary = summarise_week(WEEK, frames, predictions, _synthetic_totals())
        assert len(summary) == 6  # 2 counties x 3 horizons
        columns = [
            "week", "target_week", "horizon", "county", "prob_head", "prob_value",
            "q50_km2", "expected_wet_area_km2", "expected_cattle_exposed",
            "cattle_percent_of_mapped", "expected_crop_exposed_ha",
            "crop_percent_of_mapped", "mapped_cattle", "crop_hectares",
        ]
        assert list(summary.columns) == columns
        h1 = summary[(summary["county"] == "Alpha") & (summary["horizon"] == 1)].iloc[0]
        assert h1["prob_head"] == "det_prob_h1"
        assert np.isclose(h1["prob_value"], 0.2)
        assert h1["target_week"] == WEEK + pd.Timedelta(weeks=1)
        expected_cattle = 0.2 * 300.0 / 25.0
        assert np.isclose(h1["expected_cattle_exposed"], expected_cattle)
        assert np.isclose(
            h1["cattle_percent_of_mapped"], 100 * expected_cattle / 300.0
        )
        h0 = summary[(summary["county"] == "Alpha") & (summary["horizon"] == 0)].iloc[0]
        assert h0["prob_head"] == "det_prob"
        assert np.isclose(h0["q50_km2"], 5.0)
        assert np.isclose(
            h0["cattle_percent_of_mapped"], 100 * 0.4 * 300.0 / 25.0 / 300.0
        )
        beta = summary[(summary["county"] == "Beta") & (summary["horizon"] == 2)].iloc[0]
        assert beta["prob_head"] == "det_prob_h2"
        assert np.isclose(beta["expected_cattle_exposed"], 0.0)
        # Without duration heads only the forecast week is mapped.
        predictions = _synthetic_predictions(with_heads=False)
        frames = week_frames(WEEK, predictions, climatology, assets, layers, [])
        summary = summarise_week(WEEK, frames, predictions, _synthetic_totals())
        assert len(summary) == 2
        assert (summary["horizon"] == 0).all()


def check_select_weeks() -> None:
    weeks = [WEEK, WEEK + pd.Timedelta(weeks=1), WEEK + pd.Timedelta(weeks=2)]
    frames = []
    for week, det_prob in zip(weeks, (0.1, 0.9, 0.5)):
        frame = _synthetic_predictions()
        frame["week"] = week
        frame["det_prob"] = det_prob
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    assert select_weeks(predictions) == sorted(weeks)
    # Cap keeps the highest-det_prob weeks (0.9 and 0.5), sorted ascending.
    assert select_weeks(predictions, max_weeks=2) == [weeks[1], weeks[2]]
    assert select_weeks(predictions, max_weeks=5) == sorted(weeks)


def check_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cattle_path, crop_path = _write_rasters(tmp_path)
        grid = _synthetic_grid()
        climatology = _synthetic_climatology(grid, "uniform")
        predictions_csv = tmp_path / "pred.csv"
        _synthetic_predictions().to_csv(predictions_csv, index=False)
        _summary, figures = build_forecast_risk_maps(
            predictions_path=predictions_csv,
            climatology=climatology,
            mapped_totals=_synthetic_totals(),
            boundaries=_synthetic_boundaries(),
            cattle_path=cattle_path,
            crop_path=crop_path,
            out_table=tmp_path / "summary.csv",
            figure_dir=tmp_path / "figures",
        )
        assert (tmp_path / "summary.csv").is_file()
        assert {path.name for path in figures} == {
            "forecast_risk_cattle_2024-10-07.png",
            "forecast_risk_crop_2024-10-07.png",
        }
        for path in figures:
            assert path.stat().st_size > 0
        table = pd.read_csv(tmp_path / "summary.csv")
        assert len(table) == 6
        row = table[(table["county"] == "Alpha") & (table["horizon"] == 0)].iloc[0]
        assert np.isclose(row["expected_cattle_exposed"], 0.4 * 300.0 / 25.0)
        assert np.isclose(
            row["cattle_percent_of_mapped"], 100 * 0.4 * 300.0 / 25.0 / 300.0
        )
        _summary, no_figures = build_forecast_risk_maps(
            predictions_path=predictions_csv,
            climatology=climatology,
            mapped_totals=_synthetic_totals(),
            boundaries=_synthetic_boundaries(),
            cattle_path=cattle_path,
            crop_path=crop_path,
            out_table=tmp_path / "summary2.csv",
            figure_dir=tmp_path / "figures2",
            make_figures=False,
        )
        assert no_figures == []


def main() -> None:
    checks = [
        check_load_forecast,
        check_week_frames_uniform_signature,
        check_concentrated_signature_and_ceiling,
        check_summary_table,
        check_select_weeks,
        check_end_to_end,
    ]
    for check in checks:
        check()
        print(f"Passed: {check.__name__}")
    print("All checks passed!")


if __name__ == "__main__":
    main()
