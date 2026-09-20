"""Potential cattle and rangeland exposure to satellite flood detections."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from EDA_flood_masks.flood_eda import PRODUCT_RADIUS_M, pixel_area_km2


def load_raster_window(path: Path, bounds: tuple) -> dict:
    """Load a geographic raster window covering South Sudan."""
    if not path.is_file():
        raise FileNotFoundError(f"Required exposure raster not found: {path}")
    with rasterio.open(path) as source:
        if source.crs is None or source.crs.to_epsg() != 4326:
            raise ValueError(f"Expected EPSG:4326 for {path.name}, found {source.crs}")
        window = from_bounds(*bounds, transform=source.transform)
        window = window.round_offsets().round_lengths()
        data = source.read(1, window=window, masked=True)
        transform = source.window_transform(window)
    return {"name": path.stem, "data": data, "transform": transform}


def load_exposure_layers(farmland_root: Path, bounds: tuple) -> dict:
    """Load the cattle and grazing-land maps for the country bounds."""
    paths = {
        "cattle": farmland_root / "geonode__cattle_gha.tif",
        "rangeland": farmland_root / "asap_mask_rangeland_v04.tif",
    }
    layers = {}
    for name, path in paths.items():
        layers[name] = load_raster_window(path, bounds)
    return layers


def _sample(layer: dict, longitude, latitude) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a loaded raster and return values plus row/column indices."""
    # Find the row and column on the map for each flood location.
    inverse = ~layer["transform"]
    columns_float, rows_float = inverse * (np.asarray(longitude), np.asarray(latitude))
    rows = np.floor(rows_float).astype(int)
    columns = np.floor(columns_float).astype(int)
    data = layer["data"]
    valid = (
        (rows >= 0) & (rows < data.shape[0])
        & (columns >= 0) & (columns < data.shape[1])
    )
    values = np.full(len(rows), np.nan, dtype=float)
    values[valid] = np.ma.filled(data[rows[valid], columns[valid]], np.nan)
    return values, rows, columns


def _grid_area_km2(latitude, longitude_step: float, latitude_step: float):
    """Calculate spherical cell area for a regular longitude/latitude grid."""
    latitude = np.asarray(latitude, dtype=float)
    half_latitude = latitude_step / 2
    south = np.deg2rad(latitude - half_latitude)
    north = np.deg2rad(latitude + half_latitude)
    return (
        PRODUCT_RADIUS_M**2 * np.deg2rad(longitude_step)
        * (np.sin(north) - np.sin(south)) / 1e6
    )


def calculate_weekly_exposure(located: pd.DataFrame, layers: dict) -> pd.DataFrame:
    """Estimate weekly exposed grazing land and cattle per county."""
    keys = ["state", "county", "county_code", "week"]
    pixel_keys = keys + ["lat", "lon"]
    # Count each location once per week, even if it appears in both flood types.
    cells = located[pixel_keys].drop_duplicates().copy()
    cells["flood_cell_area_km2"] = pixel_area_km2(cells["lat"])
    grazing_percent, _, _ = _sample(layers["rangeland"], cells["lon"], cells["lat"])
    cells["flooded_rangeland_km2"] = cells["flood_cell_area_km2"] * grazing_percent / 100

    cattle, rows, columns = _sample(layers["cattle"], cells["lon"], cells["lat"])
    cells["cattle_count"] = np.clip(cattle, 0, None)
    cells["cattle_row"] = rows
    cells["cattle_column"] = columns
    # The cattle map has bigger cells, so several flood pixels can share one cell.
    cattle_groups = keys + ["cattle_row", "cattle_column"]
    cattle_cells = cells.groupby(cattle_groups, observed=True).agg(
        flooded_area_km2=("flood_cell_area_km2", "sum"),
        cattle_count=("cattle_count", "first"),
    ).reset_index()
    cattle_cells = _add_cattle_cell_area(cattle_cells, layers["cattle"])
    # If 20% of a cell is flooded, count 20% of its cattle. Never count more than 100%.
    flooded_fraction = np.minimum(
        cattle_cells["flooded_area_km2"] / cattle_cells["cattle_cell_area_km2"], 1
    )
    cattle_cells["potential_cattle_exposed"] = cattle_cells["cattle_count"] * flooded_fraction

    land = cells.groupby(keys, observed=True).agg(
        flooded_rangeland_km2=("flooded_rangeland_km2", "sum"),
    )
    cattle_weekly = cattle_cells.groupby(keys, observed=True).agg(
        potential_cattle_exposed=("potential_cattle_exposed", "sum")
    )
    return land.join(cattle_weekly, how="outer").fillna(0).reset_index()


def _add_cattle_cell_area(cells: pd.DataFrame, layer: dict) -> pd.DataFrame:
    """Attach the full area of each coarse cattle raster cell."""
    transform = layer["transform"]
    centre_latitude = transform.f + (cells["cattle_row"] + 0.5) * transform.e
    result = cells.copy()
    result["cattle_cell_area_km2"] = _grid_area_km2(
        centre_latitude, abs(transform.a), abs(transform.e)
    )
    return result


def summarise_county_exposure_baseline(counties, layers: dict) -> pd.DataFrame:
    """Calculate mapped cattle and grazing-land totals for each county."""
    result = counties[["state", "county", "county_code"]].copy()
    for name in ("cattle", "rangeland"):
        totals = _zonal_total(counties, layers[name], name)
        result = result.merge(totals, on="county_code", how="left", validate="one_to_one")
    return result.fillna(0)


def _zonal_total(counties, layer: dict, name: str) -> pd.DataFrame:
    """Aggregate one static raster using pixel-centre county membership."""
    # Use 0 for places outside the counties and 1, 2, ... for the counties.
    shapes = []
    for county_number, geometry in enumerate(counties.geometry, start=1):
        shapes.append((geometry, county_number))
    labels = rasterize(
        shapes, out_shape=layer["data"].shape, transform=layer["transform"],
        fill=0, all_touched=False, dtype="int16",
    )
    values = np.ma.filled(layer["data"], np.nan).astype(float)
    valid = (labels > 0) & np.isfinite(values) & (values >= 0)
    if name == "cattle":
        weights = values
        output_name = "mapped_cattle"
    else:
        rows = np.arange(values.shape[0])
        transform = layer["transform"]
        latitude = transform.f + (rows + 0.5) * transform.e
        row_area = _grid_area_km2(latitude, abs(transform.a), abs(transform.e))
        # Cell size changes with latitude, so each row gets its own area.
        weights = values / 100 * row_area[:, None]
        output_name = "mapped_rangeland_km2"
    # Add up the values for each county. This is faster than looping over every pixel.
    totals = np.bincount(labels[valid], weights=weights[valid], minlength=len(counties) + 1)
    return pd.DataFrame({
        "county_code": counties["county_code"].to_numpy(),
        output_name: totals[1:len(counties) + 1],
    })
