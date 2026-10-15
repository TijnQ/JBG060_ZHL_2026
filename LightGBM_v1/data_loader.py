"""Data loader module for LightGBM_v1 pipeline (baseline: provenance-tracked).

Fetches real flood mask label records, AgERA5 rainfall/runoff datasets,
Dartmouth gauge 100205 readings, Lake Albert levels, and evapotranspiration data.

Baseline: every loader records where its data came from in the PROVENANCE registry
("real" plus first/last valid date). There is no synthetic fallback: any
missing real source raises immediately instead of substituting generated
data. get_provenance_report() returns the registry as a DataFrame for the
run log (printed to the console at the end of every run).
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

# Baseline provenance registry: source name -> {source, provenance, coverage_start,
# coverage_end, note}. Populated as loaders run; see get_provenance_report().
PROVENANCE: dict[str, dict] = {}

_era5_cache: dict[tuple, xr.Dataset] = {}


def _silence(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


def _raise_missing(source: str, detail: str = "") -> None:
    """Fail loudly: synthetic data is never substituted for a missing source."""
    raise RuntimeError(
        f"Data source '{source}' has no usable real data ({detail}) — "
        "missing inputs raise instead of being substituted with generated data."
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
            _raise_missing("era5_rainfall_runoff", f"load failed: {exc}")
    else:
        _raise_missing("era5_rainfall_runoff", f"no .nc files in {era5_dir}")


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
            _raise_missing("gauge_100205_discharge", f"load failed: {exc}")
        _raise_missing("gauge_100205_discharge", f"area id {config.GAUGE_AREA_ID} not in dataset")
    else:
        _raise_missing("gauge_100205_discharge", f"directory {config.GAUGE_ROOT} does not exist")


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
            _raise_missing("lake_albert_level", f"load failed: {exc}")
        _raise_missing("lake_albert_level", "'Albert' station not in dataset")
    else:
        _raise_missing("lake_albert_level", f"directory {config.LAKE_ROOT} does not exist")


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
            _raise_missing("et0_aweil", f"load failed: {exc}")
        _raise_missing("et0_aweil", "no processed ET frames returned")
    else:
        _raise_missing("et0_aweil", f"directory {config.ET0_ROOT} does not exist")


def get_provenance_report() -> pd.DataFrame:
    """Return the PROVENANCE registry as a DataFrame (one row per source)."""
    cols = ["source", "provenance", "coverage_start", "coverage_end", "note"]
    if not PROVENANCE:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(sorted(PROVENANCE.values(), key=lambda r: r["source"]))
    return df[cols].reset_index(drop=True)