"""Configuration module for LightGBM_v1 pipeline.

Manages paths, model versioning (outputs_v1, outputs_v2), temporal splits,
embargo parameters, and geographic scope for Northern Bahr el Ghazal.
"""

from __future__ import annotations

from pathlib import Path

# --- Versioning & Output Paths ----------------------------------------------
MODEL_VERSION = "v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "LightGBM_v1"

OUT_ROOT = PIPELINE_DIR / "outputs"
OUT_DIR = OUT_ROOT / f"outputs_{MODEL_VERSION}"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"
MODELS_DIR = OUT_DIR / "models"

RAW_DATA = PROJECT_ROOT / "raw_data"

# Data file locations
AWEIL_WEEKLY_FLOODS_CSV = (
    PROJECT_ROOT / "EDA_flood_masks" / "outputs" / "tables" / "weekly_county_floods.csv"
)
AWEIL_MONTHLY_EXPOSURE_CSV = (
    PROJECT_ROOT / "EDA_hydrometeorology" / "outputs_county" / "tables" / "aweil_monthly_agricultural_exposure.csv"
)
NATIONAL_EXPOSURE_BASELINE_CSV = (
    PROJECT_ROOT / "EDA_flood_masks" / "outputs" / "national" / "tables" / "national_county_exposure_baseline.csv"
)

BOUNDARY_ADMIN2 = RAW_DATA / "Administrative boundaries" / "ssd_admin2.geojson"
GAUGE_ROOT = RAW_DATA / "Darthmouth Flood Observatory"
LAKE_ROOT = RAW_DATA / "Water levels lakes"
ET0_ROOT = PROJECT_ROOT / "processing_data" / "evapotranspiration"

# --- Geographic Scope: Northern Bahr el Ghazal (5 Aweil Counties) -----------
AWEIL_COUNTIES = [
    "Aweil Centre",
    "Aweil East",
    "Aweil North",
    "Aweil South",
    "Aweil West",
]
SCOPE = "aweil"

# --- Time & 5-Fold Specifications (5 blocks of 5 years: 3 train, 1 val, 1 test)
YEARS = list(range(2000, 2025))

FOLD_5YEAR_BLOCKS = [
    {"fold": 1, "train": (2000, 2002), "val": (2003, 2003), "test": (2004, 2004)},
    {"fold": 2, "train": (2005, 2007), "val": (2008, 2008), "test": (2009, 2009)},
    {"fold": 3, "train": (2010, 2012), "val": (2013, 2013), "test": (2014, 2014)},
    {"fold": 4, "train": (2015, 2017), "val": (2018, 2018), "test": (2019, 2019)},
    {"fold": 5, "train": (2020, 2022), "val": (2023, 2023), "test": (2024, 2024)},
]

# Legacy split fallback
SPLITS = {
    "train": (2000, 2014),
    "val": (2015, 2019),
    "test": (2020, 2024),
}

# Embargo rule: For target week starting Monday m, features are dated at most m - 3 days (Friday)
EMBARGO_DAYS = 3

# Target Quantiles & Duration Horizons
QUANTILES = (0.1, 0.5, 0.9)
DURATION_HORIZONS = (1, 2, 3, 4)

# Feature Windows & Lags
ROLL_WINDOWS = (1, 3, 7, 14, 30)
OWN_LAGS = (1, 2, 3, 4)
UPSTREAM_OFFSET_DEG = 3.0
GAUGE_AREA_ID = 100205

# Reproducibility
SEED = 2026
