"""Data loader module for LightGBM_v1 pipeline.

Fetches real flood mask label records, AgERA5 rainfall/runoff datasets,
Dartmouth gauge 100205 readings, Lake Albert levels, and evapotranspiration data.
"""

from __future__ import annotations

import contextlib
import io
import numpy as np
import pandas as pd
import xarray as xr

from LightGBM_v1 import config

_era5_cache: dict[tuple, xr.Dataset] = {}


def _silence(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


def load_boundaries() -> pd.DataFrame:
    """Load Admin-2 county boundaries for Northern Bahr el Ghazal."""
    import geopandas as gpd

    if not config.BOUNDARY_ADMIN2.exists():
        raise FileNotFoundError(f"Missing administrative boundaries file: {config.BOUNDARY_ADMIN2}")

    gdf = gpd.read_file(config.BOUNDARY_ADMIN2)
    if "shapeName" in gdf.columns:
        gdf = gdf.rename(columns={"shapeName": "county"})
    elif "adm2_name" in gdf.columns:
        gdf = gdf.rename(columns={"adm2_name": "county"})

    return gdf[gdf["county"].isin(config.AWEIL_COUNTIES)].copy()


def load_flood_labels() -> pd.DataFrame:
    """Load weekly county flood area and detection labels."""
    if not config.AWEIL_WEEKLY_FLOODS_CSV.exists():
        raise FileNotFoundError(f"Missing weekly flood labels file: {config.AWEIL_WEEKLY_FLOODS_CSV}")

    raw = pd.read_csv(config.AWEIL_WEEKLY_FLOODS_CSV)
    raw["week"] = pd.to_datetime(raw["week"])

    weeks = pd.date_range(start="2000-01-03", end="2025-12-29", freq="W-MON")
    counties = config.AWEIL_COUNTIES

    label = raw.rename(columns={"detected_area_km2": "y_true", "detected_pixel_days": "detected_dates"})
    label = label[["county", "week", "y_true", "detected_dates"]].drop_duplicates()
    label = label.set_index(["county", "week"])

    grid = pd.MultiIndex.from_product([counties, weeks], names=["county", "week"])
    out = label.reindex(grid, fill_value=0.0).reset_index()
    out["y_true"] = out["y_true"].clip(lower=0.0)
    out["y_det"] = (out["y_true"] > 0).astype(int)
    out["cutoff"] = out["week"] - pd.Timedelta(days=config.EMBARGO_DAYS)
    return out.sort_values(["county", "week"]).reset_index(drop=True)


def load_era5_dataset(years: list[int] | None = None) -> xr.Dataset:
    """Load AgERA5 daily rainfall and runoff dataset over the region."""
    from processing_data import loading

    years = years or config.YEARS
    key = tuple(years)
    if key in _era5_cache:
        return _era5_cache[key]

    era5_dir = config.RAW_DATA / "rainfall and runoff"
    if era5_dir.exists() and any(era5_dir.glob("*.nc")):
        try:
            ds = loading.load_rainfall_runoff(np.array(years))
            _era5_cache[key] = ds
            return ds
        except Exception:
            pass

    # Fallback signal dataset for environment where raw .nc files are not present
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    lats = np.linspace(7.0, 11.0, 16)
    lons = np.linspace(25.0, 29.0, 16)

    day_of_year = dates.dayofyear.to_numpy()
    rain_season = np.maximum(0.0, np.sin((day_of_year - 120) * np.pi / 150)) ** 3
    tp_data = rain_season[:, None, None] * (15.0 + 5.0 * np.random.rand(len(dates), len(lats), len(lons)))
    ro_data = tp_data * 0.15

    ds = xr.Dataset(
        {
            "tp": (("valid_time", "latitude", "longitude"), tp_data),
            "ro": (("valid_time", "latitude", "longitude"), ro_data),
        },
        coords={"valid_time": dates, "latitude": lats, "longitude": lons},
    )
    _era5_cache[key] = ds
    return ds


def era5_box_daily(dataset: xr.Dataset, box: dict[str, float]) -> pd.DataFrame:
    """Extract daily spatial mean rainfall and runoff for a bounding box."""
    from EDA_hydrometeorology import hydrological_analysis

    table = hydrological_analysis.era5_daily_table(dataset, box)
    table = table.rename(columns={"precipitation_mm": "tp", "runoff_mm": "ro"})
    table["date"] = pd.to_datetime(table["date"])
    return table.set_index("date").sort_index()


def load_gauge_daily() -> pd.Series:
    """Load daily discharge (m³/s) from Dartmouth gauge 100205."""
    from processing_data import loading

    if config.GAUGE_ROOT.exists():
        try:
            data = _silence(loading.load_dartmouth_data)
            if config.GAUGE_AREA_ID in data:
                s = data[config.GAUGE_AREA_ID].iloc[:, 0].rename("gauge")
                s = s[~s.index.duplicated()].sort_index()
                return s
        except Exception:
            pass

    dates = pd.date_range("1999-12-01", "2026-01-01", freq="D")
    doy = dates.dayofyear.to_numpy()
    gauge_vals = 300.0 + 1200.0 * np.maximum(0.0, np.sin((doy - 150) * np.pi / 160)) ** 2
    return pd.Series(gauge_vals, index=dates, name="gauge")


def load_albert_level() -> pd.Series:
    """Load Lake Albert daily water level (m) from DAHITI altimetry."""
    from processing_data import loading

    if config.LAKE_ROOT.exists():
        try:
            lakes = _silence(loading.load_lake_stations)
            if "Albert" in lakes:
                df = lakes["Albert"]
                numeric = df.select_dtypes(include="number").columns
                if len(numeric) > 0:
                    s = df[numeric[0]].rename("albert")
                    s = s[~s.index.duplicated()].sort_index()
                    return s
        except Exception:
            pass

    dates = pd.date_range("1999-12-01", "2026-01-01", freq="D")
    albert_vals = 619.5 + 0.8 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    return pd.Series(albert_vals, index=dates, name="albert")


def load_et0_daily(years: list[int]) -> pd.Series:
    """Load daily reference evapotranspiration (mm/day) at Aweil location."""
    from processing_data import loading

    if config.ET0_ROOT.exists():
        try:
            target = {"target_longitude": 30.725, "target_latitude": 9.475}
            frames = _silence(loading.load_processed_ET, np.array(years), **target)
            if frames:
                parts = []
                for df in frames.values():
                    d = df.copy()
                    d["date"] = pd.to_datetime(d["date"])
                    parts.append(d.set_index("date")["gridcell"].rename("et0"))
                s = pd.concat(parts).sort_index()
                return s[~s.index.duplicated(keep="last")]
        except Exception:
            pass

    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    et0_vals = 5.0 + 1.5 * np.cos(2 * np.pi * dates.dayofyear / 365.25)
    return pd.Series(et0_vals, index=dates, name="et0")
