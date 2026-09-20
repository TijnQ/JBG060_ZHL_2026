"""Memory-conscious flood-mask summaries for all South Sudan counties."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds

from EDA_flood_masks.flood_eda import YEARS, pixel_area_km2

TILES = ("h20v08", "h21v08")
FLOOD_TYPES = ("recurring", "unusual")


def load_south_sudan_counties(boundaries_path: Path) -> gpd.GeoDataFrame:
    """Load the supplied Admin-2 boundaries for the nationwide analysis."""
    counties = gpd.read_file(boundaries_path)
    required = ["adm1_name", "adm2_name", "adm2_pcode", "area_sqkm", "geometry"]
    missing = set(required).difference(counties.columns)
    if missing:
        raise ValueError(f"Boundary file misses columns: {sorted(missing)}")
    if counties.crs is None:
        raise ValueError("Boundary file must specify a coordinate reference system")
    if counties["adm2_pcode"].duplicated().any():
        raise ValueError("Admin-2 codes must be unique")
    return counties[required].rename(columns={
        "adm1_name": "state",
        "adm2_name": "county",
        "adm2_pcode": "county_code",
        "area_sqkm": "county_area_km2",
    }).to_crs("EPSG:4326")


def _annual_path(flood_root: Path, flood_type: str, tile: str, year: int) -> Path:
    return flood_root / f"compact_{flood_type}" / f"flood_events_{tile}_{year}.parquet"


def validate_flood_files(flood_root: Path, years=YEARS) -> None:
    """Fail early when one of the expected annual files is unavailable."""
    missing = []
    for year in years:
        for flood_type in FLOOD_TYPES:
            for tile in TILES:
                path = _annual_path(flood_root, flood_type, tile, year)
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} flood files. First missing:\n{preview}")


def _read_file(path: Path, bounds: tuple, start: str, end: str) -> pd.DataFrame:
    """Read only relevant columns, dates and the South Sudan bounding box."""
    min_lon, min_lat, max_lon, max_lat = bounds
    keep = (
        (ds.field("lon") >= min_lon) & (ds.field("lon") <= max_lon)
        & (ds.field("lat") >= min_lat) & (ds.field("lat") <= max_lat)
        & (ds.field("date") >= start) & (ds.field("date") < end)
    )
    columns = ["date", "lat", "lon", "cloud_frac"]
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=columns, filter=keep)
    return table.to_pandas()


def load_analysis_year(
    flood_root: Path, year: int, bounds: tuple, final_year: int
) -> pd.DataFrame:
    """Load observations whose Monday-based week starts in the requested year."""
    frames = []
    end = f"{year + 1}-01-07"
    for flood_type in FLOOD_TYPES:
        for tile in TILES:
            path = _annual_path(flood_root, flood_type, tile, year)
            frame = _read_file(path, bounds, f"{year}-01-01", end)
            frame["flood_type"] = flood_type
            frames.append(frame)
            # A week starting in December can include days from next January.
            if year < final_year:
                next_path = _annual_path(flood_root, flood_type, tile, year + 1)
                extra = _read_file(next_path, bounds, f"{year + 1}-01-01", end)
                extra["flood_type"] = flood_type
                frames.append(extra)
    observations = pd.concat(frames, ignore_index=True)
    observations["date"] = pd.to_datetime(observations["date"], errors="raise")
    # W-SUN means a week ending on Sunday, so its start is Monday.
    observations["week"] = observations["date"].dt.to_period("W-SUN").dt.start_time
    observations["flood_type"] = observations["flood_type"].astype("category")
    return observations.loc[observations["week"].dt.year.eq(year)].copy()


def assign_counties(
    observations: pd.DataFrame, counties: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Assign each distinct grid cell to one Admin-2 county."""
    cells = observations[["lat", "lon"]].drop_duplicates()
    points = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    )
    columns = ["state", "county", "county_code", "geometry"]
    located = gpd.sjoin(points, counties[columns], how="inner", predicate="within")
    result = observations.merge(
        located[["lat", "lon", "state", "county", "county_code"]],
        on=["lat", "lon"],
        validate="many_to_one",
    )
    if result.empty:
        raise ValueError("No flood detections remained inside South Sudan")
    for column in ("state", "county", "county_code"):
        result[column] = result[column].astype("category")
    return result


def summarise_located(located: pd.DataFrame, by_type: bool) -> pd.DataFrame:
    """Calculate union areas per county-week, optionally split by flood class."""
    keys = ["state", "county", "county_code", "week"]
    if by_type:
        keys.append("flood_type")
    daily = located.drop_duplicates(keys + ["date", "lat", "lon"])
    pixel_days = daily.groupby(keys, observed=True).size().rename("detected_pixel_days")
    cells = daily.drop_duplicates(keys + ["lat", "lon"]).copy()
    cells["cell_area_km2"] = pixel_area_km2(cells["lat"])
    areas = cells.groupby(keys, observed=True).agg(
        unique_grid_cells=("lon", "size"),
        detected_area_km2=("cell_area_km2", "sum"),
    )
    return pixel_days.to_frame().join(areas).reset_index()


def summarise_coverage(located: pd.DataFrame, year: int) -> pd.DataFrame:
    """Summarise positive detections; this is not satellite availability."""
    keys = ["state", "county", "county_code", "flood_type"]
    result = located.groupby(keys, observed=True).agg(
        detections=("date", "size"),
        first_detection=("date", "min"),
        last_detection=("date", "max"),
        mean_cloud_fraction=("cloud_frac", "mean"),
    ).reset_index()
    result.insert(3, "year", year)
    return result


def add_relative_area(
    weekly: pd.DataFrame, counties: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Add detected area as a percentage of each county's official area."""
    areas = counties[["county_code", "county_area_km2"]].drop_duplicates()
    result = weekly.merge(areas, on="county_code", validate="many_to_one")
    result["detected_county_percent"] = (
        100 * result["detected_area_km2"] / result["county_area_km2"]
    )
    return result
