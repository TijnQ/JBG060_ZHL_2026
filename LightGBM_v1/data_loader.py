"""Data loader module for LightGBM_v1 pipeline (v3: provenance-tracked).

Fetches real flood mask label records, AgERA5 rainfall/runoff datasets,
Dartmouth gauge 100205 readings, Lake Albert levels, and evapotranspiration data.

v3: every loader records where its data came from in the PROVENANCE registry
("real" or "synthetic_fallback" plus first/last valid date). Synthetic
fallbacks are gated on config.ALLOW_SYNTHETIC_FALLBACK; with the gate closed,
a missing real source raises RuntimeError instead of silently substituting
generated data. get_provenance_report() returns the registry as a DataFrame
for the run log and data_provenance.csv.
"""

from __future__ import annotations

import contextlib
import io
import numpy as np
import pandas as pd
import xarray as xr

from LightGBM_v1 import config

__all__ = [
    "PROVENANCE",
    "era5_box_daily",
    "get_provenance_report",
    "load_albert_level",
    "load_boundaries",
    "load_et0_daily",
    "load_era5_dataset",
    "load_flood_labels",
    "load_gauge_daily",
]

# v3 provenance registry: source name -> {source, provenance, coverage_start,
# coverage_end, note}. Populated as loaders run; see get_provenance_report().
PROVENANCE: dict[str, dict] = {}

_era5_cache: dict[tuple, xr.Dataset] = {}


def _silence(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


def _gate_or_raise(source: str, detail: str = "") -> None:
    """Refuse to substitute synthetic data when the provenance gate is closed."""
    if not config.ALLOW_SYNTHETIC_FALLBACK:
        raise RuntimeError(
            f"Data source '{source}' has no usable real data ({detail}) and "
            "config.ALLOW_SYNTHETIC_FALLBACK is False — refusing to substitute synthetic data."
        )


def _record(source: str, provenance: str, index: pd.Index | np.ndarray, note: str = "") -> None:
    """Record provenance of a loaded time series / dataset into the registry."""
    idx = pd.DatetimeIndex(index)
    PROVENANCE[source] = {
        "source": source,
        "provenance": provenance,
        "coverage_start": idx.min(),
        "coverage_end": idx.max(),
        "note": note,
    }


def _record_static(source: str, note: str = "") -> None:
    """Record a non-time-series source (spatial boundaries) into the registry."""
    PROVENANCE[source] = {
        "source": source,
        "provenance": "real",
        "coverage_start": pd.NaT,
        "coverage_end": pd.NaT,
        "note": note,
    }


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

    gdf = gdf[gdf["county"].isin(config.AWEIL_COUNTIES)].copy()
    _record_static("admin2_boundaries", str(config.BOUNDARY_ADMIN2))
    return gdf
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
    out = out.sort_values(["county", "week"]).reset_index(drop=True)

    n_observed = int(len(raw))
    _record(
        "flood_labels",
        "real",
        out["week"].unique(),
        f"{n_observed} observed label rows; weeks without a valid satellite "
        "detection are filled with 0 (censoring — see README honesty note)",
    )
    return out


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
            _record("era5_rainfall_runoff", "real", ds["valid_time"].to_index(),
                    "AgERA5 .nc files under raw_data/rainfall and runoff")
            return ds
        except Exception as exc:  # real files present but unreadable
            _gate_or_raise("era5_rainfall_runoff", f"load failed: {exc}")
    else:
        _gate_or_raise("era5_rainfall_runoff", f"no .nc files in {era5_dir}")

    # Fallback signal dataset for environments where raw .nc files are not present
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
    _record("era5_rainfall_runoff", "synthetic_fallback", dates,
            "generated sinusoidal seasonal signal — NOT real data")
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
                _record("gauge_100205_discharge", "real", s.index)
                return s
        except Exception as exc:
            _gate_or_raise("gauge_100205_discharge", f"load failed: {exc}")
        _gate_or_raise("gauge_100205_discharge", f"area id {config.GAUGE_AREA_ID} not in dataset")
    else:
        _gate_or_raise("gauge_100205_discharge", f"directory {config.GAUGE_ROOT} does not exist")

    dates = pd.date_range("1999-12-01", "2026-01-01", freq="D")
    doy = dates.dayofyear.to_numpy()
    gauge_vals = 300.0 + 1200.0 * np.maximum(0.0, np.sin((doy - 150) * np.pi / 160)) ** 2
    s = pd.Series(gauge_vals, index=dates, name="gauge")
    _record("gauge_100205_discharge", "synthetic_fallback", s.index,
            "generated sinusoidal seasonal signal — NOT real data")
    return s


def load_albert_level() -> pd.Series:
    """Load Lake Albert daily water level (m) from DAHITI altimetry.

    Altimetry passes carry a time-of-day (e.g. 19:56:24); timestamps are
    normalized to the pass day (midnight) so the series aligns with the
    pipeline's daily index. Multiple passes on the same day keep the last.
    """
    from processing_data import loading

    if config.LAKE_ROOT.exists():
        try:
            lakes = _silence(loading.load_lake_stations)
            if "Albert" in lakes:
                df = lakes["Albert"]
                numeric = df.select_dtypes(include="number").columns
                if len(numeric) > 0:
                    s = df[numeric[0]].rename("albert")
                    s.index = pd.to_datetime(s.index).normalize()
                    s = s[~s.index.duplicated(keep="last")]
                    s = s.sort_index()
                    _record("lake_albert_level", "real", s.index)
                    return s
        except Exception as exc:
            _gate_or_raise("lake_albert_level", f"load failed: {exc}")
        _gate_or_raise("lake_albert_level", "'Albert' station not in dataset")
    else:
        _gate_or_raise("lake_albert_level", f"directory {config.LAKE_ROOT} does not exist")

    dates = pd.date_range("1999-12-01", "2026-01-01", freq="D")
    albert_vals = 619.5 + 0.8 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    s = pd.Series(albert_vals, index=dates, name="albert")
    _record("lake_albert_level", "synthetic_fallback", s.index,
            "generated sinusoidal seasonal signal — NOT real data")
    return s


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
                s = s[~s.index.duplicated(keep="last")]
                _record("et0_aweil", "real", s.index)
                return s
        except Exception as exc:
            _gate_or_raise("et0_aweil", f"load failed: {exc}")
        _gate_or_raise("et0_aweil", "no processed ET frames returned")
    else:
        _gate_or_raise("et0_aweil", f"directory {config.ET0_ROOT} does not exist")

    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    et0_vals = 5.0 + 1.5 * np.cos(2 * np.pi * dates.dayofyear / 365.25)
    s = pd.Series(et0_vals, index=dates, name="et0")
    _record("et0_aweil", "synthetic_fallback", s.index,
            "generated sinusoidal seasonal signal — NOT real data")
    return s


def get_provenance_report() -> pd.DataFrame:
    """Return the PROVENANCE registry as a DataFrame (one row per source)."""
    cols = ["source", "provenance", "coverage_start", "coverage_end", "note"]
    if not PROVENANCE:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(sorted(PROVENANCE.values(), key=lambda r: r["source"]))
    return df[cols].reset_index(drop=True)