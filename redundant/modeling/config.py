"""Central settings for the ``modeling`` package.

Paths, scopes, time ranges, split definitions and the label embargo live here
so that every module agrees on the same conventions. See ``MODEL_RESEARCH.md``
(section 5, validation protocol) for the reasoning behind the embargo and the
temporal splits.

Nothing in this package may fall back to synthetic data: if a required input
is missing, ``data_check`` and the loaders raise. That is deliberate — the
repo has already been bitten once by a silent synthetic fallback
(``EDA_hydrometeorology/hydrometeorology_eda_country.py``).
"""

from pathlib import Path

# --- Paths ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "raw_data"

MODELING_DIR = Path(__file__).resolve().parent
OUT = MODELING_DIR / "outputs"
# Shared outputs: data inventory, gate-0 audit, advisory layer, cross-method artifacts.
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
# Per-method outputs: outputs/methods/<name>/{tables,figures,models}.
# The method folders are the units of the testing ground: kill a method and
# you delete one folder plus its output folder — nothing shared is touched.
METHODS_OUT = OUT / "methods"


def method_dirs(name: str) -> dict[str, Path]:
    """Output directories for one model method (created on demand)."""
    root = METHODS_OUT / name
    return {"tables": root / "tables", "figures": root / "figures", "models": root / "models"}


# --- Model methods (the testing ground) ---------------------------------------
# Task A methods in testing order; index 0 is the PRIMARY backbone
# (MODEL_RESEARCH.md §4.2 — LightGBM first), the rest are cross-checks that
# must justify themselves against it. Each name is a folder in this package
# with a guide.md (what to test, how to run it, keep/kill criteria).
# Task B is a separate competition (spatial "where", different output).
TASK_A_METHODS = ("lightgbm", "xgboost", "catboost", "randomforest", "tabpfn")
TASK_B_METHODS = ("unet",)

BOUNDARIES_DIR = RAW_DATA / "Administrative boundaries"
BOUNDARY_ADMIN1 = BOUNDARIES_DIR / "ssd_admin1.geojson"
BOUNDARY_ADMIN2 = BOUNDARIES_DIR / "ssd_admin2.geojson"

FLOOD_ROOT = RAW_DATA / "flood_masks"
ERA5_ROOT = RAW_DATA / "rainfall and runoff"
# Folder name typo is in the raw dataset layout itself; keep it.
GAUGE_ROOT = RAW_DATA / "Darthmouth Flood Observatory"
LAKE_ROOT = RAW_DATA / "Water levels lakes"
ET0_ROOT = PROJECT_ROOT / "processing_data" / "evapotranspiration"
FARMLAND_ROOT = RAW_DATA / "farmland"
# Static raster channels for the U-Net (file names as in the raw layout;
# data_check reports their presence/readability).
FARMLAND_CATTLE = FARMLAND_ROOT / "geonode__cattle_gha.tif"
FARMLAND_CROPS = FARMLAND_ROOT / "asap_mask_crops_v04.tif"
FARMLAND_RANGELAND = FARMLAND_ROOT / "asap_mask_rangeland_v04.tif"

# Committed real-data EDA outputs that this package reuses as label tables
# (regenerate them with the EDA scripts if the raw data changes).
AWAIL_WEEKLY_FLOODS_CSV = (
    PROJECT_ROOT / "EDA_flood_masks" / "outputs" / "tables" / "weekly_county_floods.csv"
)
NATIONAL_WEEKLY_FLOODS_CSV = (
    PROJECT_ROOT / "EDA_flood_masks" / "outputs" / "national" / "tables" / "national_weekly_floods.csv"
)
AWAIL_MONTHLY_EXPOSURE_CSV = (
    PROJECT_ROOT / "EDA_hydrometeorology" / "outputs_county" / "tables" / "aweil_monthly_agricultural_exposure.csv"
)
NATIONAL_EXPOSURE_BASELINE_CSV = (
    PROJECT_ROOT / "EDA_flood_masks" / "outputs" / "national" / "tables" / "national_county_exposure_baseline.csv"
)

# --- Scopes ------------------------------------------------------------------

AWEIL_COUNTIES = [
    "Aweil Centre",
    "Aweil East",
    "Aweil North",
    "Aweil South",
    "Aweil West",
]
NATIONAL_SCOPE = "national"
AWEIL_SCOPE = "aweil"
SCOPES = (AWEIL_SCOPE, NATIONAL_SCOPE)

# --- Time ---------------------------------------------------------------------

# Full record available in the flood parquets and ERA5.
YEARS = list(range(2000, 2026))
# Gate-0 audit range: long enough for 3 wet seasons in each split, short enough
# that one run finishes in minutes.
AUDIT_YEARS = list(range(2015, 2026))

# Strict temporal 3-way split (never shuffled, never crossed):
# train 2000-2014 (15 y), validation 2015-2019 (5 y), test 2020-2025 (6 y).
SPLITS = {
    "train": (2000, 2014),
    "val": (2015, 2019),
    "test": (2020, 2025),
}

# Label embargo in days: a 3-day composite dated ``d`` covers d-2..d, so a
# feature may be used for a target week starting Monday ``m`` only if the
# feature is dated at most ``m - EMBARGO_DAYS`` (Friday of the previous week).
EMBARGO_DAYS = 3

# --- Hydro-meteorology ----------------------------------------------------------

# The single gauge of the Dartmouth Flood Observatory covering South Sudan.
GAUGE_AREA_ID = 100205
# AgERA5 reference evapotranspiration point used by the EDA (near Aweil).
ET0_TARGET = {"target_longitude": 30.725, "target_latitude": 9.475}
# The upstream rainfall band sits this many degrees west of the target area
# (same convention as the EDA notebooks).
UPSTREAM_OFFSET_DEG = 3.0

# Rolling windows (days) for rainfall / runoff / moisture features.
ROLL_WINDOWS = (1, 3, 7, 14, 30)
# Lags (weeks) of the county's own observed flood area used as features.
OWN_LAGS = (1, 2, 3, 4)

# --- Task B (sub-county "where") -------------------------------------------------

# HEC/USGS tile covering the Aweil floodplain; 20-30 E x 0-10 N at 0.25 deg is
# exactly 40 x 40 ERA5 grid cells.
TILE = "h20v08"
TILE_BBOX = {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 20.0, "lon_max": 30.0}
ERA5_CELL_DEG = 0.25
TILE_GRID = round((TILE_BBOX["lat_max"] - TILE_BBOX["lat_min"]) / ERA5_CELL_DEG)

# Daily channels per side of the U-Net input: last 7 days of tp and ro.
UNET_DAILY_CHANNELS = 2  # tp, ro
UNET_HISTORY_DAYS = 7
UNET_STATIC_CHANNELS = 3  # crop fraction, rangeland fraction, cattle density
UNET_IN_CHANNELS = (
    UNET_DAILY_CHANNELS * UNET_HISTORY_DAYS + UNET_STATIC_CHANNELS
)

# --- Seeds / misc ---------------------------------------------------------------

SEED = 2026
