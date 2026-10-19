"""Forecast-week risk maps: model forecast x cattle/crop exposure overlap.

After each Level-1 model run this module maps the forecast for *the week in
question* — not a typical year — onto the cattle and crop exposure assets.

The Level-1 run (``python -m LightGBM_v1.run --mode final``) forecasts one
week: ``det_prob`` (P(any flood detected in the county that week)) plus
duration heads ``det_prob_h1`` / ``det_prob_h2`` (P(the flood is still there
one / two weeks later)). The Level-2 downscaling in
``climatology_maps.forecast_maps`` turns a county forecast into a per-pixel
probability,

    prob(p) = det_prob(c, W) x freq_given(p, c, month(W))

and this module reuses exactly that compositing code, then overlaps the
probabilistic flood map with the two asset rasters:

* cattle — ICPAC cattle counts on a ~9 km grid. Expected exposed head per
  MODIS pixel = prob x (cell count / cell area) x pixel area. Summed over a
  cattle cell this reproduces the national EDA rule (count x flooded share of
  the cell) in expectation and can never exceed the mapped herd.
* crop — ASAP crop mask (~0.5 km cells, land fraction 0-100). One mask cell
  counts once when any of its MODIS pixels is detected (the section-1 chain),
  so the cell's wet probability is the max pixel probability and its expected
  hectares = P(wet) x cell area x fraction / 100. Those hectares are
  allocated back to pixels proportional to prob x pixel area, so the map
  integrates to the county total.

Duration heads compose on the *same* footprint: the persistence question is
whether this week's water is still there, not where new water goes, so
prob_h(p) = det_prob_h(c, W) x freq_given(p, c, month(W)).

Outputs (per forecast week): one PNG per asset showing the three ingredients
side by side — Level-2 flood probability | asset map | expected exposure for
the forecast week (duration heads stay in the table) — plus
``forecast_risk_summary.csv`` (county x horizon expected exposures and
shares of the mapped totals). The PNG background outside the Northern Bahr
el Ghazal study region is transparent so the figures overlay cleanly.

Run the synthetic-data checks with: python -m risk_exposure.check_forecast_risk_map
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from matplotlib.colors import LogNorm
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from climatology_maps.forecast_maps import (
    compose_week,
    county_month_assets,
    load_final_predictions,
    pivot_raster,
)
from climatology_maps.pixel_climatology import PRODUCT_RADIUS_M, get_climatology
from EDA_hydrometeorology.hydrological_analysis import (
    agricultural_baseline,
    load_aweil_counties,
)

__all__ = [
    "build_forecast_risk_maps",
    "build_mapped_totals",
    "load_asset_layers",
    "load_forecast",
    "select_weeks",
    "summarise_week",
    "week_frames",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATTLE_RASTER = PROJECT_ROOT / "raw_data" / "farmland" / "geonode__cattle_gha.tif"
CROP_RASTER = PROJECT_ROOT / "raw_data" / "farmland" / "asap_mask_crop_v04.tif"
ADMIN2_PATH = PROJECT_ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"
NATIONAL_BASELINE_CSV = (
    PROJECT_ROOT / "EDA_flood_masks" / "outputs" / "national" / "tables"
    / "national_county_exposure_baseline.csv"
)
CLIMATOLOGY_CACHE = PROJECT_ROOT / "climatology_maps" / "outputs" / "climatology_data"
FIGURE_DIR = PROJECT_ROOT / "risk_exposure" / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "risk_exposure" / "outputs" / "tables"

ASSETS = {
    "cattle": {
        "column": "exp_cattle",
        "unit": "head",
        "figure": "Expected cattle standing in forecast flood water (head)",
        "colorbar": "Expected cattle exposed (head per pixel)",
        "asset_cmap": "Oranges",
        "asset_title": "Where the cattle are",
        "asset_subtitle": "ICPAC herd density (head per km², log colour)",
        "asset_colorbar": "Herd density (head per km²)",
    },
    "crop": {
        "column": "exp_crop",
        "unit": "ha",
        "figure": "Expected crop land under forecast flood water (hectares)",
        "colorbar": "Expected crop exposure (ha per pixel share)",
        "asset_cmap": "Greens",
        "asset_title": "Where the crop land is",
        "asset_subtitle": "ASAP crop-land fraction (%)",
        "asset_colorbar": "Crop-land fraction (%)",
    },
}

# Flat-earth metres per degree used by the section-1 exposure chain; kept so
# the forecast-week crop hectares stay comparable with the section-1 numbers.
METRES_PER_DEGREE = 111_320.0


def load_forecast(path: str | Path | None = None) -> tuple[pd.DataFrame, list[int]]:
    """Standardise a forecast table and list its duration heads.

    Accepts any CSV with county/week/det_prob/q10/q50/q90; ``det_prob_hN``
    duration-head columns are optional. When ``path`` is None the Level-1
    final-fit forecast written by the last model run is used. Missing head
    values count as no persistence forecast (probability 0).
    """
    if path is None:
        return load_final_predictions()
    df = pd.read_csv(path, parse_dates=["week"])
    required = {"county", "week", "det_prob", "q10", "q50", "q90"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction table {path} misses columns {sorted(missing)}")
    horizons = sorted(
        int(col.rsplit("h", 1)[1])
        for col in df.columns
        if col.startswith("det_prob_h") and col[len("det_prob_h"):].isdigit()
    )
    head_columns = [f"det_prob_h{h}" for h in horizons]
    df[head_columns] = df[head_columns].fillna(0.0)
    return df.sort_values(["county", "week"]).reset_index(drop=True), horizons


def _grid_area_km2(latitude, lon_step: float, lat_step: float) -> np.ndarray:
    """Spherical area (km2) of a regular lat/lon cell (EDA formula)."""
    latitude = np.asarray(latitude, dtype=float)
    half = lat_step / 2
    south = np.deg2rad(latitude - half)
    north = np.deg2rad(latitude + half)
    return (
        PRODUCT_RADIUS_M**2 * np.deg2rad(lon_step)
        * (np.sin(north) - np.sin(south)) / 1e6
    )


def load_asset_layers(
    grid: pd.DataFrame,
    cattle_path: str | Path = CATTLE_RASTER,
    crop_path: str | Path = CROP_RASTER,
) -> dict[str, dict]:
    """Read both exposure rasters once, windowed to the pixel grid."""
    pad = 0.05
    bounds = (
        float(grid["lon"].min()) - pad,
        float(grid["lat"].min()) - pad,
        float(grid["lon"].max()) + pad,
        float(grid["lat"].max()) + pad,
    )
    layers: dict[str, dict] = {}
    for name, path in (("cattle", cattle_path), ("crop", crop_path)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required exposure raster not found: {path}")
        with rasterio.open(path) as source:
            if source.crs is None or source.crs.to_epsg() != 4326:
                raise ValueError(f"Expected EPSG:4326 for {Path(path).name}")
            window = from_bounds(*bounds, transform=source.transform)
            window = window.round_offsets().round_lengths()
            data = source.read(1, window=window, masked=True)
            transform = source.window_transform(window)
        layers[name] = {
            "data": np.ma.filled(data.astype(float), np.nan),
            "transform": transform,
        }
    return layers


def _cell_indices(
    layer: dict, lons: np.ndarray, lats: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row/column of each point inside a loaded raster window (floor rule).

    The floor rule assigns every point to the cell that contains it, so all
    MODIS pixels inside one coarse cell share that cell — the grouping the
    exposure math relies on.
    """
    data = layer["data"]
    inverse = ~layer["transform"]
    cols_float, rows_float = inverse * (np.asarray(lons), np.asarray(lats))
    rows = np.floor(np.asarray(rows_float)).astype(int)
    cols = np.floor(np.asarray(cols_float)).astype(int)
    valid = (
        (rows >= 0) & (rows < data.shape[0])
        & (cols >= 0) & (cols < data.shape[1])
    )
    return rows, cols, valid


def add_cattle_exposure(frame: pd.DataFrame, cattle_layer: dict) -> pd.DataFrame:
    """Expected cattle head per pixel = prob x density x pixel area.

    Density is the mapped count of the coarse cattle cell divided by that
    cell's area. A fully wet county (prob = 1 everywhere) then reproduces the
    mapped herd, and any fractional forecast scales it linearly — the
    national EDA's "count x flooded share of cell" rule evaluated in
    expectation, with no cap needed because prob <= 1.
    """
    transform = cattle_layer["transform"]
    rows, cols, valid = _cell_indices(
        cattle_layer, frame["lon"].to_numpy(), frame["lat"].to_numpy()
    )
    data = cattle_layer["data"]
    count = np.full(len(frame), np.nan)
    count[valid] = data[rows[valid], cols[valid]]
    count = np.clip(count, 0.0, None)  # negative/nodata counts are not cattle
    centre_latitude = transform.f + (rows + 0.5) * transform.e
    cell_area = _grid_area_km2(centre_latitude, abs(transform.a), abs(transform.e))
    density = np.where(np.isfinite(count), count, 0.0) / cell_area
    out = frame.copy()
    out["exp_cattle"] = out["prob"] * density * out["pixel_area_km2"]
    return out


def add_crop_exposure(frame: pd.DataFrame, crop_layer: dict) -> pd.DataFrame:
    """Expected crop hectares per pixel via the mask-cell rule.

    One ~0.5 km mask cell is exposed when any of its MODIS pixels is wet (the
    section-1 chain counts the cell once), so the cell's wet probability is
    the max pixel probability and its expected hectares =
    P(wet) x cell area x fraction / 100. Those hectares are shared between
    the cell's pixels proportional to prob x pixel area, so per-pixel values
    sum back to the county total exactly.
    """
    transform = crop_layer["transform"]
    rows, cols, valid = _cell_indices(
        crop_layer, frame["lon"].to_numpy(), frame["lat"].to_numpy()
    )
    data = crop_layer["data"]
    fraction = np.zeros(len(frame))
    inside = data[rows[valid], cols[valid]]
    fraction[valid] = np.nan_to_num(np.clip(inside, 0.0, 100.0), nan=0.0)
    pixel_ha = (
        abs(transform.a) * abs(transform.e) * METRES_PER_DEGREE**2 / 10_000
    )
    out = frame.copy()
    out["_row"], out["_col"] = rows, cols
    out["_pa"] = out["prob"] * out["pixel_area_km2"]
    grouped = out.groupby(["county", "_row", "_col"], observed=True, sort=False)
    out["_p_wet"] = grouped["prob"].transform("max")
    out["_cell_pa"] = grouped["_pa"].transform("sum")
    out["_cell_ha"] = out["_p_wet"] * pixel_ha * fraction / 100.0
    out["exp_crop"] = np.where(
        out["_cell_pa"] > 0,
        out["_cell_ha"] * out["_pa"] / out["_cell_pa"],
        0.0,
    )
    return out.drop(columns=["_row", "_col", "_pa", "_p_wet", "_cell_pa", "_cell_ha"])


def cattle_ceiling(grid: pd.DataFrame, cattle_layer: dict) -> pd.DataFrame:
    """Head per county if every grid pixel were wet (prob = 1 audit limit)."""
    ceiling = add_cattle_exposure(
        grid.assign(prob=1.0), cattle_layer
    ).groupby("county", observed=True)["exp_cattle"].sum()
    return ceiling.rename("cattle_ceiling_prob1").reset_index()


def week_frames(
    week: pd.Timestamp,
    predictions: pd.DataFrame,
    climatology: dict[str, pd.DataFrame],
    assets: pd.DataFrame,
    layers: dict[str, dict],
    horizons: list[int],
) -> dict[int, pd.DataFrame]:
    """Compose exposure frames for one forecast week.

    Horizon 0 is the forecast week itself; each duration head h composes the
    same spatial signature with ``det_prob_h{h}`` (persistence of this week's
    water). Counties without a forecast for the week are absent.
    """
    frames: dict[int, pd.DataFrame] = {}
    for horizon in [0, *horizons]:
        frame = compose_week(week, predictions, climatology, assets, horizon=horizon)
        if frame.empty:
            continue
        frame = add_cattle_exposure(frame, layers["cattle"])
        frame = add_crop_exposure(frame, layers["crop"])
        frames[horizon] = frame
    return frames


def summarise_week(
    week: pd.Timestamp,
    frames: dict[int, pd.DataFrame],
    predictions: pd.DataFrame,
    mapped_totals: pd.DataFrame,
) -> pd.DataFrame:
    """One summary row per county x horizon for one forecast week."""
    rows = []
    for horizon, frame in sorted(frames.items()):
        stats = frame.assign(_pa=frame["prob"] * frame["pixel_area_km2"]).groupby(
            "county", observed=True
        ).agg(
            expected_wet_area_km2=("_pa", "sum"),
            expected_cattle_exposed=("exp_cattle", "sum"),
            expected_crop_exposed_ha=("exp_crop", "sum"),
        ).reset_index()
        stats["horizon"] = horizon
        stats["week"] = week
        stats["target_week"] = week + pd.Timedelta(weeks=horizon)
        pred = predictions[predictions["week"] == week].set_index("county")
        prob_head = "det_prob" if horizon == 0 else f"det_prob_h{horizon}"
        stats["prob_head"] = prob_head
        stats["prob_value"] = pred[prob_head].reindex(stats["county"]).to_numpy()
        stats["q50_km2"] = pred["q50"].reindex(stats["county"]).to_numpy()
        rows.append(stats)
    summary = pd.concat(rows, ignore_index=True)
    summary = summary.merge(mapped_totals, on="county", how="inner", validate="many_to_one")
    summary["cattle_percent_of_mapped"] = (
        100 * summary["expected_cattle_exposed"] / summary["mapped_cattle"]
    )
    summary["crop_percent_of_mapped"] = (
        100 * summary["expected_crop_exposed_ha"] / summary["crop_hectares"]
    )
    columns = [
        "week", "target_week", "horizon", "county", "prob_head", "prob_value",
        "q50_km2", "expected_wet_area_km2", "expected_cattle_exposed",
        "cattle_percent_of_mapped", "expected_crop_exposed_ha",
        "crop_percent_of_mapped", "mapped_cattle", "crop_hectares",
    ]
    return summary[columns].sort_values(["horizon", "county"]).reset_index(drop=True)


def _native_extent(layer: dict) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) extent of a loaded raster window."""
    transform = layer["transform"]
    nrows, ncols = layer["data"].shape
    left, top = transform.c, transform.f
    return left, left + transform.a * ncols, top + transform.e * nrows, top


def _region_alpha(
    boundaries: gpd.GeoDataFrame, transform: Affine, shape: tuple[int, int]
) -> np.ndarray:
    """1.0 inside the mapped county polygons, 0.0 outside.

    A cell counts as inside when its centre falls in one of the polygons —
    the same centre rule the exposure rasters use — so the alpha mask hugs
    the drawn county outlines and everything outside the Northern Bahr el
    Ghazal study region exports fully transparent.
    """
    inside = rasterize(
        ((geometry, 1) for geometry in boundaries.geometry),
        out_shape=shape, transform=transform, fill=0, dtype="uint8",
    )
    return np.where(inside == 1, 1.0, 0.0)


def render_risk_figure(
    frame: pd.DataFrame,
    week: pd.Timestamp,
    asset: str,
    layer: dict,
    out_path: str | Path,
    boundaries: gpd.GeoDataFrame | None,
) -> Path:
    """One PNG: flood probability | asset map | expected exposure.

    The left panel is the Level-2 probabilistic flood map (same YlOrRd 0-1
    style as the section-11 ``prob_<week>.png``), the middle panel the asset
    raster as loaded, and the right panel their product — the figure shows
    the rule ``expected = chance of water x asset`` with all three
    ingredients. County labels on the right panel carry the county's
    expected total; where no forecast exists for a county its pixels stay
    grey. Everything outside the mapped counties — the Northern Bahr el
    Ghazal study region — is exported transparent (alpha 0, including the
    figure canvas), so the PNG overlays on any background.
    """
    spec = ASSETS[asset]
    value_column = spec["column"]
    figure, axes = plt.subplots(1, 3, figsize=(19.2, 7.8), constrained_layout=True)

    matrix, lons, lats = pivot_raster(frame, "prob")
    span_lon = lons.max() - lons.min()
    span_lat = lats.max() - lats.min()
    extent = (
        lons.min() - 0.5 * span_lon / len(lons),
        lons.max() + 0.5 * span_lon / len(lons),
        lats.min() - 0.5 * span_lat / len(lats),
        lats.max() + 0.5 * span_lat / len(lats),
    )
    limits = (
        lons.min() - 0.02, lons.max() + 0.02, lats.min() - 0.02, lats.max() + 0.02,
    )
    if boundaries is None:
        grid_alpha = np.ones(matrix.shape)
    else:
        step_lon = (lons.max() - lons.min()) / max(len(lons) - 1, 1)
        step_lat = (lats.max() - lats.min()) / max(len(lats) - 1, 1)
        grid_transform = Affine(
            step_lon, 0, float(lons.min()) - step_lon / 2,
            0, step_lat, float(lats.min()) - step_lat / 2,
        )
        grid_alpha = _region_alpha(boundaries, grid_transform, matrix.shape)

    hazard_cmap = plt.get_cmap("YlOrRd").copy()
    hazard_cmap.set_bad("lightgrey")
    hazard_image = axes[0].imshow(
        matrix, extent=extent, origin="lower", cmap=hazard_cmap,
        vmin=0.0, vmax=1.0, aspect="equal", alpha=grid_alpha,
    )
    axes[0].set_title(
        f"1. Where the model puts water\nforecast week of {week.date()}\n"
        "flood probability = det_prob × flood pattern",
        fontsize=10, pad=8,
    )
    figure.colorbar(hazard_image, ax=axes[0], shrink=0.72, pad=0.02).set_label(
        "Flood probability", rotation=270, labelpad=14
    )

    asset_cmap = plt.get_cmap(spec["asset_cmap"]).copy()
    asset_cmap.set_bad("lightgrey")
    asset_data = np.ma.masked_invalid(layer["data"])
    left, right, bottom, top = _native_extent(layer)
    if boundaries is None:
        asset_alpha = np.ones(asset_data.shape)
    else:
        asset_alpha = _region_alpha(
            boundaries, layer["transform"], layer["data"].shape
        )
    if asset == "cattle":
        transform = layer["transform"]
        centre_lat = transform.f + (np.arange(asset_data.shape[0]) + 0.5) * transform.e
        cell_area = _grid_area_km2(centre_lat, abs(transform.a), abs(transform.e))
        asset_data = np.ma.masked_less_equal(asset_data / cell_area[:, None], 0.0)
        values = asset_data.compressed()
        vmax = float(values.max()) if values.size else 1.0
        vmin = float(values.min()) if values.size else vmax / 100.0
        if vmin >= vmax:
            vmin = vmax / 100.0
        asset_image = axes[1].imshow(
            asset_data, extent=(left, right, bottom, top), origin="upper",
            cmap=asset_cmap, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="equal",
            alpha=asset_alpha,
        )
    else:
        asset_image = axes[1].imshow(
            asset_data, extent=(left, right, bottom, top), origin="upper",
            cmap=asset_cmap, vmin=0.0, vmax=100.0, aspect="equal",
            alpha=asset_alpha,
        )
    axes[1].set_title(
        f"2. {spec['asset_title']}\n{spec['asset_subtitle']}", fontsize=10, pad=8
    )
    figure.colorbar(asset_image, ax=axes[1], shrink=0.72, pad=0.02).set_label(
        spec["asset_colorbar"], rotation=270, labelpad=14
    )

    value_cmap = plt.get_cmap("Reds").copy()
    value_cmap.set_bad("lightgrey")
    value_matrix, _, _ = pivot_raster(frame, value_column)
    vmax = float(frame[value_column].max()) or 1.0
    value_image = axes[2].imshow(
        value_matrix, extent=extent, origin="lower", cmap=value_cmap,
        vmin=0.0, vmax=vmax, aspect="equal", alpha=grid_alpha,
    )
    axes[2].set_title(
        f"3. {spec['figure']}\nchance of water × asset", fontsize=10, pad=8
    )
    figure.colorbar(value_image, ax=axes[2], shrink=0.72, pad=0.02).set_label(
        spec["colorbar"], rotation=270, labelpad=14
    )

    for axis in axes:
        if boundaries is not None:
            boundaries.boundary.plot(ax=axis, color="black", linewidth=0.9, alpha=0.95)
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        axis.set_axis_off()

    if boundaries is not None:
        totals = frame.groupby("county", observed=True)[value_column].sum()
        for _, row in boundaries.iterrows():
            if row["county"] not in totals.index:
                continue
            value = float(totals[row["county"]])
            text = f"{row['county']}\n~{value:,.0f} {spec['unit']}"
            axes[2].annotate(
                text,
                xy=row.geometry.representative_point().coords[0],
                ha="center", va="center", fontsize=7, color="white",
                bbox={"boxstyle": "round,pad=0.15", "fc": "black",
                      "ec": "none", "alpha": 0.6},
            )

    figure.suptitle(
        f"Aweil {asset} risk — chance of water × asset = expected exposure"
        f" — week of {week.date()}",
        fontsize=12,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, transparent=boundaries is not None)
    plt.close(figure)
    return out_path


def select_weeks(predictions: pd.DataFrame, max_weeks: int | None = None) -> list:
    """Weeks to map: every week, or the highest-det_prob ones up to the cap."""
    weeks = sorted(pd.Timestamp(w) for w in predictions["week"].unique())
    if max_weeks is None or len(weeks) <= max_weeks:
        return weeks
    ranked = predictions.groupby("week")["det_prob"].max().sort_values(ascending=False)
    return sorted(pd.Timestamp(w) for w in ranked.head(max_weeks).index)


def county_totals_cattle(csv_path, counties: pd.DataFrame) -> pd.DataFrame:
    """Mapped cattle totals ("cattle in region") per study county.

    Reads ``state``, ``county`` and ``mapped_cattle`` from the national
    exposure-baseline table and raises if a study county is missing from it.
    """
    baseline = pd.read_csv(csv_path)
    selected = baseline[baseline["county"].isin(set(counties["county"]))]
    expected = counties["county"].nunique()
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} study counties in {csv_path}, found {len(selected)}"
        )
    return (
        selected[["state", "county", "mapped_cattle"]]
        .sort_values("county")
        .reset_index(drop=True)
    )


def county_totals_crop(counties, crop_mask_path) -> pd.DataFrame:
    """Mapped crop hectares per study county from the ASAP crop mask."""
    return agricultural_baseline(counties, {"crop": str(crop_mask_path)})


def build_mapped_totals(
    counties: gpd.GeoDataFrame,
    baseline_csv: str | Path = NATIONAL_BASELINE_CSV,
    crop_path: str | Path = CROP_RASTER,
) -> pd.DataFrame:
    """Mapped cattle and crop hectares per county (the denominators)."""
    cattle = county_totals_cattle(baseline_csv, counties)
    crop = county_totals_crop(counties, str(crop_path))
    return cattle.merge(
        crop[["county", "crop_hectares"]], on="county", validate="one_to_one"
    )


def build_forecast_risk_maps(
    predictions_path: str | Path | None = None,
    climatology: dict[str, pd.DataFrame] | None = None,
    mapped_totals: pd.DataFrame | None = None,
    boundaries: gpd.GeoDataFrame | None = None,
    cattle_path: str | Path = CATTLE_RASTER,
    crop_path: str | Path = CROP_RASTER,
    out_table: str | Path = TABLE_DIR / "forecast_risk_summary.csv",
    figure_dir: str | Path = FIGURE_DIR,
    max_weeks: int | None = None,
    make_figures: bool = True,
) -> tuple[pd.DataFrame, list[Path]]:
    """Risk maps + summary for every week in a model-run forecast table.

    Defaults map the Level-1 final-fit forecast (the operational single-week
    product each model run writes); pass ``--predictions`` to map any other
    run's table. Returns the summary frame (also written to ``out_table``)
    and the list of figure paths.

    ``climatology`` accepts Level 2's ``get_climatology()`` output so callers
    (the notebook sections) can make the Level-2 hand-off explicit; the
    default loads the same prior from the shared cache.
    """
    predictions, horizons = load_forecast(predictions_path)
    if climatology is None:
        climatology = get_climatology(cache_dir=CLIMATOLOGY_CACHE)
    assets = county_month_assets(climatology)
    grid = climatology["grid"]
    layers = load_asset_layers(grid, cattle_path=cattle_path, crop_path=crop_path)
    if mapped_totals is None:
        counties = load_aweil_counties(ADMIN2_PATH)
        counties = counties.sort_values("county").reset_index(drop=True)
        mapped_totals = build_mapped_totals(counties)
        boundaries = counties if boundaries is None else boundaries
    weeks = select_weeks(predictions, max_weeks)
    summary_parts: list[pd.DataFrame] = []
    figures: list[Path] = []
    for week in weeks:
        frames = week_frames(week, predictions, climatology, assets, layers, horizons)
        if not frames:
            continue
        summary_parts.append(summarise_week(week, frames, predictions, mapped_totals))
        if make_figures and boundaries is not None:
            for asset in ASSETS:
                out_path = (
                    Path(figure_dir) / f"forecast_risk_{asset}_{week.date()}.png"
                )
                render_risk_figure(
                    frames[0], week, asset, layers[asset], out_path, boundaries
                )
                figures.append(out_path)
    if not summary_parts:
        raise ValueError("No forecast weeks could be mapped for this table")
    summary = pd.concat(summary_parts, ignore_index=True)
    rounded = summary.copy()
    for column, digits in (
        ("prob_value", 4), ("q50_km2", 4), ("expected_wet_area_km2", 2),
        ("expected_cattle_exposed", 1), ("cattle_percent_of_mapped", 2),
        ("expected_crop_exposed_ha", 1), ("crop_percent_of_mapped", 2),
        ("mapped_cattle", 0), ("crop_hectares", 0),
    ):
        rounded[column] = rounded[column].astype(float).round(digits)
    out_table = Path(out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    rounded.to_csv(out_table, index=False)
    print(f"[risk-map] weeks mapped: {len(weeks)}; horizons: {[0, *horizons]}")
    audit = cattle_ceiling(grid, layers["cattle"]).merge(
        mapped_totals[["county", "mapped_cattle"]], on="county", how="left"
    )
    print("[risk-map] audit — cattle head if every pixel were wet vs mapped herd:")
    for _, row in audit.iterrows():
        print(
            f"  {row['county']:<14} ceiling {row['cattle_ceiling_prob1']:>12,.0f}"
            f"  mapped {float(row['mapped_cattle']):>12,.0f}"
        )
    print(f"[table] {out_table}")
    for path in figures:
        print(f"[figure] {path}")
    return summary, figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exposure-overlap risk map for a model-run forecast week",
    )
    parser.add_argument(
        "--predictions", default=None,
        help="county/week forecast CSV (default: the Level-1 final-fit forecast)",
    )
    parser.add_argument(
        "--max-weeks", type=int, default=None,
        help="cap the number of mapped weeks (highest det_prob first)",
    )
    parser.add_argument(
        "--outdir", default=None, help="output directory for the CSV and PNGs",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="summary CSV only",
    )
    args = parser.parse_args()
    out_dir = Path(args.outdir) if args.outdir else None
    summary, _ = build_forecast_risk_maps(
        predictions_path=args.predictions,
        max_weeks=args.max_weeks,
        make_figures=not args.no_figures,
        out_table=(
            out_dir / "forecast_risk_summary.csv" if out_dir
            else TABLE_DIR / "forecast_risk_summary.csv"
        ),
        figure_dir=out_dir if out_dir else FIGURE_DIR,
    )
    print()
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
