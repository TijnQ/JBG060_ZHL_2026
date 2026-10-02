"""Process raw AgERA5 evapotranspiration into the CSVs the LightGBM pipeline reads.

Runs processing_data.loading.process_ET() for every year in LightGBM_v1.config.YEARS
(2000-2025), extracting daily reference evapotranspiration (Penman-Monteith FAO56)
at the grid cell nearest to Aweil, Northern Bahr el Ghazal (9.475N, 30.725E).

Input : raw_data/evapotranspiration/ET_{year}/   one daily .nc file per day
Output: processing_data/evapotranspiration/ET_{year}_9.475N_30.725E_processed.csv

LightGBM_v1/data_loader.load_et0_daily() reads exactly these CSVs and, with
config.ALLOW_SYNTHETIC_FALLBACK = False, refuses to run without them - so this
script is the reproducible preprocessing step for the et0 feature block.

Usage:
    .venv/bin/python process_et_NBeG.py                # all years from config.YEARS
    .venv/bin/python process_et_NBeG.py --years 2024   # subset only

The run is resumable: years whose processed CSV already exists are skipped.
Exits non-zero if any requested year ends up without a usable CSV, so it can
double as a preflight check before LightGBM_v1/run.py.

Raw data source: https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators
(distributed via the project Surfdrive share; see README.md).
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LightGBM_v1 import config  # noqa: E402  (needs REPO_ROOT on sys.path first)
from processing_data.loading import process_ET  # noqa: E402

# Must match the hardcoded target in LightGBM_v1/data_loader.load_et0_daily();
# otherwise the pipeline will not find the processed CSVs.
TARGET_LATITUDE = 9.475
TARGET_LONGITUDE = 30.725

RAW_ET_ROOT = REPO_ROOT / "raw_data" / "evapotranspiration"
PROCESSED_ET_DIR = REPO_ROOT / "processing_data" / "evapotranspiration"


def expected_csv(year: int) -> Path:
    """Return the processed CSV path process_ET() writes for one year."""
    lat_str = f"{TARGET_LATITUDE:.3f}N"
    lon_str = f"{TARGET_LONGITUDE:.3f}E"
    return PROCESSED_ET_DIR / f"ET_{year}_{lat_str}_{lon_str}_processed.csv"


def preflight(years: list[int]) -> None:
    """Fail fast with a clear message when the raw input data is missing."""
    if not RAW_ET_ROOT.exists():
        sys.exit(
            f"ERROR: raw evapotranspiration data not found at {RAW_ET_ROOT}.\n"
            "Download the AgERA5 'Reference ET (Penman-Monteith FAO56)' daily\n"
            "NetCDF files first (project Surfdrive share or Copernicus CDS, see\n"
            "README.md), keeping the per-year folder layout\n"
            "raw_data/evapotranspiration/ET_{year}/."
        )
    missing = [year for year in years if not (RAW_ET_ROOT / f"ET_{year}").exists()]
    if missing:
        print(f"WARNING: no raw folder for year(s) {missing}; processing will\n"
              "skip them and verification will fail below.")


def verify(years: list[int]) -> bool:
    """Check each expected CSV exists and is usable; print a summary table.

    Returns True when every requested year has non-empty processed output with
    the 'date' and 'gridcell' columns the pipeline loader expects.
    """
    print(f"\nVerifying processed files in {PROCESSED_ET_DIR} ...")
    header = f"{'year':>6}  {'days':>5}  {'expected':>8}  status"
    print(header)
    print("-" * len(header))
    ok = True
    for year in years:
        path = expected_csv(year)
        expected_days = 366 if calendar.isleap(year) else 365
        if not path.exists():
            print(f"{year:>6}  {'-':>5}  {expected_days:>8}  MISSING")
            ok = False
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - report any read problem as failure
            print(f"{year:>6}  {'?':>5}  {expected_days:>8}  UNREADABLE ({exc})")
            ok = False
            continue
        if not {"date", "gridcell"}.issubset(df.columns):
            print(f"{year:>6}  {len(df):>5}  {expected_days:>8}  BAD COLUMNS "
                  "(need 'date' and 'gridcell')")
            ok = False
        elif len(df) == 0:
            print(f"{year:>6}  {len(df):>5}  {expected_days:>8}  EMPTY")
            ok = False
        elif len(df) < expected_days:
            print(f"{year:>6}  {len(df):>5}  {expected_days:>8}  SHORT "
                  "(possible NaN gaps in downstream features)")
        else:
            print(f"{year:>6}  {len(df):>5}  {expected_days:>8}  OK")
    return ok


def main() -> None:
    """Process ET for all requested years and verify the resulting CSVs."""
    parser = argparse.ArgumentParser(
        description="Process raw AgERA5 ET NetCDFs into the annual CSVs read by "
                    "LightGBM_v1/data_loader.load_et0_daily().",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Years to process (default: LightGBM_v1 config.YEARS).",
    )
    args = parser.parse_args()
    years = list(args.years) if args.years else list(config.YEARS)

    # process_ET() resolves './raw_data/...' and './processing_data/...' relative
    # to the CWD, so pin the CWD to the repository root. This makes the script
    # runnable from anywhere (e.g. inside an IDE task or a Makefile).
    os.chdir(REPO_ROOT)

    print("=" * 95)
    print(f"PROCESS ET — NORTHERN BAHR EL GHAZAL — "
          f"{TARGET_LATITUDE:.3f}N {TARGET_LONGITUDE:.3f}E — {years[0]}-{years[-1]}")
    print("=" * 95)

    preflight(years)

    for year in tqdm(years, desc="Processing ET"):
        process_ET(year=int(year),
                   target_longitude=TARGET_LONGITUDE,
                   target_latitude=TARGET_LATITUDE)

    if verify(years):
        print("\nAll requested years processed successfully.")
    else:
        sys.exit("\nERROR: one or more requested years lack usable processed "
                 "output (see table above).")


if __name__ == "__main__":
    main()