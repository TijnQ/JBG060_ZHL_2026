"""Functions for the flood-mask EDA in the five Aweil counties."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

STUDY_AREA = "Northern Bahr el Ghazal"
STUDY_TILE = "h20v08"
YEARS = range(2000, 2026)
# NASA MCDWD/VCDWD User Guide Rev. F, Table 5 (p. 28).
GRID_STEP_DEGREES = 10 / 4800
PRODUCT_RADIUS_M = 6371007.181


def pixel_area_km2(latitude):
    """Calculate pixel area from latitude using the grid defined by NASA."""
    latitude = np.asarray(latitude, dtype=float)
    half_step = GRID_STEP_DEGREES / 2
    if not np.isfinite(latitude).all() or (np.abs(latitude) + half_step > 90).any():
        raise ValueError("Grid-cell centres must be finite and inside the poles")
    south = np.deg2rad(latitude - half_step)
    north = np.deg2rad(latitude + half_step)
    return (
        PRODUCT_RADIUS_M**2 * np.deg2rad(GRID_STEP_DEGREES)
        * (np.sin(north) - np.sin(south)) / 1e6
    )


def load_study_counties(boundaries_path: Path) -> gpd.GeoDataFrame:
    """Load the county boundaries and convert them to longitude and latitude."""
    counties = gpd.read_file(boundaries_path)
    required = {"adm1_name", "adm2_name", "area_sqkm", "geometry"}
    missing = required.difference(counties.columns)
    if missing:
        raise ValueError(f"Boundary file misses columns: {sorted(missing)}")
    if counties.crs is None:
        raise ValueError("Boundary file must specify a coordinate reference system")
    selected = counties[counties["adm1_name"] == STUDY_AREA].copy()
    if selected.empty:
        raise ValueError(f"No counties found for {STUDY_AREA!r}")
    selected = selected[["adm2_name", "area_sqkm", "geometry"]]
    selected = selected.rename(columns={"adm2_name": "county"})
    return selected.to_crs("EPSG:4326")


def _read_bbox(files: list[Path], bounds: tuple) -> pd.DataFrame:
    """Load only the columns and locations needed for the study area."""
    missing = [str(path) for path in files if not path.is_file()]
    if not files or missing:
        raise FileNotFoundError(f"Missing expected annual flood files: {missing}")
    min_lon, min_lat, max_lon, max_lat = bounds
    bbox_filter = (
        (ds.field("lon") >= min_lon) & (ds.field("lon") <= max_lon)
        & (ds.field("lat") >= min_lat) & (ds.field("lat") <= max_lat)
    )
    # Some files store their columns differently, so read each file separately.
    frames = []
    columns = ["date", "lat", "lon", "tile", "cloud_frac"]
    for path in files:
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(columns=columns, filter=bbox_filter)
        frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True)


def load_flood_observations(flood_root: Path, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """Load the flood records and find the county for each pixel."""
    frames = []
    for flood_type in ("recurring", "unusual"):
        files = []
        for year in YEARS:
            filename = f"flood_events_{STUDY_TILE}_{year}.parquet"
            files.append(flood_root / f"compact_{flood_type}" / filename)
        frame = _read_bbox(files, tuple(counties.total_bounds))
        frame["flood_type"] = flood_type
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    observations["date"] = pd.to_datetime(observations["date"], errors="raise")
    # A pixel stays in the same county, so we only look it up once.
    cells = observations[["lat", "lon"]].drop_duplicates()
    points = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326"
    )
    located = gpd.sjoin(points, counties, how="inner", predicate="within")
    result = observations.merge(
        located[["lat", "lon", "county"]], on=["lat", "lon"], validate="many_to_one"
    )
    if result.empty:
        raise ValueError("No flood detections remained in the selected counties")
    return result


def summarise_weekly(observations: pd.DataFrame, *, by_type: bool = False) -> pd.DataFrame:
    """Calculate weekly areas, counting each pixel once in the total.

    Set by_type=True to get separate results for recurring and unusual floods.
    A pixel can appear in both classes during one week, so those areas cannot be
    added together. The total includes all pixels detected at any point that week.
    """
    required = {"date", "lat", "lon", "county", "flood_type", "cloud_frac"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Observations miss columns: {sorted(missing)}")
    if observations.empty or observations[list(required)].isna().any().any():
        raise ValueError("Observations must be non-empty and required values non-null")
    if not set(observations["flood_type"]).issubset({"recurring", "unusual"}):
        raise ValueError("Flood classes must be recurring or unusual")
    keys = ["county", "week"]
    daily_keys = ["county", "date", "lat", "lon"]
    if by_type:
        keys.append("flood_type")
        daily_keys.append("flood_type")
    data = observations.drop_duplicates(daily_keys).copy()
    # W-SUN means the week ends on Sunday, so start_time gives us Monday.
    data["week"] = data["date"].dt.to_period("W-SUN").dt.start_time
    detections = data.groupby(keys, observed=True).agg(
        detected_pixel_days=("lon", "size"), mean_cloud_fraction=("cloud_frac", "mean")
    )
    cells = data.drop_duplicates(keys + ["lat", "lon"]).copy()
    cells["cell_area_km2"] = pixel_area_km2(cells["lat"])
    areas = cells.groupby(keys, observed=True).agg(
        unique_grid_cells=("lon", "size"), detected_area_km2=("cell_area_km2", "sum")
    )
    weekly = detections.join(areas).reset_index()
    weekly["year"] = weekly["week"].dt.year
    weekly["month"] = weekly["week"].dt.month
    return weekly


def _validate_weekly_totals(weekly: pd.DataFrame) -> None:
    """Check that there is only one total for each county and week."""
    if "flood_type" in weekly or weekly.duplicated(["county", "week"]).any():
        raise ValueError("Use unique county-week totals from summarise_weekly(by_type=False)")
    if weekly.empty or weekly[["county", "week", "detected_area_km2"]].isna().any().any():
        raise ValueError("Weekly totals must be non-empty and non-null")
    if not np.isfinite(weekly["detected_area_km2"]).all() or (weekly["detected_area_km2"] < 0).any():
        raise ValueError("Detected area must be finite and non-negative")
    if (weekly["week"].dt.dayofweek != 0).any():
        raise ValueError("Weeks must start on Monday")


def summarise_seasonality(weekly: pd.DataFrame, county_names, years=YEARS) -> pd.DataFrame:
    """Find the average weekly area for each month across the selected years.

    First average the weeks within each year-month, then average across years.
    Weeks with no records count as zero detections, which does not prove dry land.
    The Monday date determines the month. We also return the number of weeks used.
    """
    _validate_weekly_totals(weekly)
    years = sorted(set(years))
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Seasonality requires a non-empty consecutive year range")
    names = sorted(set(county_names))
    if not set(weekly["county"]).issubset(names):
        raise ValueError("County list must include every county in the weekly totals")
    calendar = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="W-MON")
    # Give each county every week, including weeks with no flood records.
    county_calendars = []
    for county in names:
        county_calendar = pd.DataFrame({"county": county, "week": calendar})
        county_calendars.append(county_calendar)
    complete = pd.concat(county_calendars, ignore_index=True)
    areas = weekly[["county", "week", "detected_area_km2"]]
    complete = complete.merge(areas, on=["county", "week"], how="left")
    complete["area"] = complete["detected_area_km2"].fillna(0)
    complete["year"] = complete["week"].dt.year
    complete["month"] = complete["week"].dt.month
    complete["positive"] = complete["area"] > 0
    monthly = complete.groupby(["county", "year", "month"], observed=True).agg(
        weekly_mean=("area", "mean"), calendar_weeks=("area", "size"),
        weeks_with_detection=("positive", "sum"),
    ).reset_index()
    return monthly.groupby(["county", "month"], observed=True).agg(
        mean_weekly_detected_area_km2=("weekly_mean", "mean"),
        years_included=("year", "size"), calendar_weeks=("calendar_weeks", "sum"),
        weeks_with_detection=("weeks_with_detection", "sum"),
    ).reset_index()


def find_detected_events(weekly: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive weeks with detections in the same county.

    Different pixels can be detected each week, so the length of a group does not
    tell us how long one field was flooded. A week with no detections ends the group,
    although missing observations could also explain that gap.
    """
    _validate_weekly_totals(weekly)
    totals = weekly[weekly["detected_area_km2"] > 0].copy()
    totals = totals.sort_values(["county", "week"])
    days_since_previous_week = totals.groupby("county")["week"].diff().dt.days
    # Start a new run at the first detection or when the previous week is missing.
    totals["new_run"] = days_since_previous_week != 7
    totals["run_id"] = totals.groupby("county")["new_run"].cumsum()
    return totals.groupby(["county", "run_id"], observed=True).agg(
        start_week=("week", "min"), end_week=("week", "max"),
        county_detected_weeks=("week", "size"),
        peak_weekly_union_area_km2=("detected_area_km2", "max"),
        cumulative_detected_area_km2_weeks=("detected_area_km2", "sum"),
    ).reset_index()


def validate_outputs(observations: pd.DataFrame, weekly: pd.DataFrame) -> None:
    """Check the weekly pixel counts and make sure the areas are within range."""
    _validate_weekly_totals(weekly)
    keys = observations[["county", "lat", "lon", "date"]].copy()
    keys["week"] = keys["date"].dt.to_period("W-SUN").dt.start_time
    expected = keys.drop_duplicates(["county", "week", "lat", "lon"])
    expected = expected.groupby(["county", "week"]).size().sort_index()
    actual = weekly.set_index(["county", "week"])["unique_grid_cells"].sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert weekly["detected_pixel_days"].ge(weekly["unique_grid_cells"]).all()
    area_per_cell = weekly["detected_area_km2"] / weekly["unique_grid_cells"]
    bounds = pixel_area_km2(observations["lat"])
    assert area_per_cell.between(bounds.min() - 1e-10, bounds.max() + 1e-10).all()
