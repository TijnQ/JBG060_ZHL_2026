"""Configuration module for LightGBM_v1 pipeline.

Manages paths, model versioning (outputs_v1..v3), the 3 expanding-window
CV folds, embargo parameters, and geographic scope for Northern Bahr el Ghazal.
"""

from __future__ import annotations

from pathlib import Path

# --- Versioning & Output Paths ----------------------------------------------
MODEL_VERSION = "v3"

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

# --- Time & Cross-Validation Specification (v3: 3 expanding-window folds) ----
# Real data coverage: ERA5 2000-2025, gauge & Lake Albert 2000-2025.
# 2025 is loaded so feature completeness is real, but it stays OUTSIDE the CV:
# no fold trains, validates, or tests on 2025, and no CV output contains 2025.
YEARS = list(range(2000, 2026))

# 3 expanding-window folds (evaluation_metrics.md sec. 2: walk-forward CV,
# expanding train window, 1-year validation, 2-3-year holdout test).
#   Fold 1: train 2000-2008, val 2009, test 2010-2011
#   Fold 2: train 2000-2014, val 2015, test 2016-2017
#   Fold 3: train 2000-2020, val 2021, test 2022-2024
# Each season 2009-2024 is used for validation/test exactly once.
CV_FOLDS = [
    {"fold": 1, "train_years": (2000, 2008), "val_year": 2009, "test_years": (2010, 2011)},
    {"fold": 2, "train_years": (2000, 2014), "val_year": 2015, "test_years": (2016, 2017)},
    {"fold": 3, "train_years": (2000, 2020), "val_year": 2021, "test_years": (2022, 2024)},
]

# Provenance gate (v3): loaders must record real vs synthetic_fallback.
# When False, any synthetic fallback raises RuntimeError instead of silently
# substituting generated data.
ALLOW_SYNTHETIC_FALLBACK = True

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
