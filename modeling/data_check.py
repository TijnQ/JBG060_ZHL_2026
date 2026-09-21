"""Raw-data inventory: one command tells you exactly what is missing.

The modelling modules only run on real data and fail loudly, so before the
first training run do:

    python -m modeling.data_check

It walks the raw-data layout the loaders expect (``raw_data/``, plus the
processed ET0 folder and the committed EDA tables this package reuses) and
prints a table with per-file status and size. Exit code is 1 when any
*critical* input is missing, 0 otherwise (optional inputs only warn).

Critical vs optional:

- critical: boundaries, flood-mask parquets, ERA5, Dartmouth gauge, lake
  levels, farmland rasters, IPC, GDP, WorldPop, health facilities;
- optional: processed ET0 (regenerate with ``process_ET``), WorldPop detail,
  OSM road network.

Deliberately there is **no synthetic fallback anywhere in this package**:
the repo was already burned by one (the nationwide hydrometeorology EDA
silently generates sine/gamma signals when the raw data is absent — see
MODEL_RESEARCH.md risk table, entry 8).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from modeling import config

__all__ = ["build_inventory", "main", "run_check"]


def _file_rows() -> list[dict]:
    rows: list[dict] = []

    def add(path: Path, category: str, critical: bool, note: str = "") -> None:
        rows.append(
            {
                "path": str(path.relative_to(config.PROJECT_ROOT)),
                "category": category,
                "critical": critical,
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1e6, 3)
                if path.exists() and path.is_file()
                else (0.0 if not path.exists() else float("nan")),
                "note": note,
            }
        )

    # Administrative boundaries
    add(config.BOUNDARY_ADMIN1, "boundaries", True)
    add(config.BOUNDARY_ADMIN2, "boundaries", True)

    # Flood-mask detections (parquet, per tile / type / year)
    n_parquet = 0
    for ftype in ("compact_recurring", "compact_unusual"):
        for tile in ("h20v08", "h21v08"):
            for year in config.YEARS:
                p = config.FLOOD_ROOT / ftype / f"flood_events_{tile}_{year}.parquet"
                if p.exists():
                    n_parquet += 1
    add(config.FLOOD_ROOT, "flood_masks", True,
        f"{n_parquet}/{2 * 2 * len(config.YEARS)} expected parquet files present")

    # ERA5
    n_era5 = 0
    for year in config.YEARS:
        p = config.ERA5_ROOT / f"ERA5_{year}.nc"
        if p.exists():
            n_era5 += 1
    add(config.ERA5_ROOT, "era5", True,
        f"{n_era5}/{len(config.YEARS)} expected ERA5 files present")

    # Dartmouth gauge
    info = config.GAUGE_ROOT / "information.xlsx"
    add(config.GAUGE_ROOT, "gauge", True)
    add(info, "gauge", True)
    if info.exists():
        try:
            stations = pd.read_excel(info)
            n_csv = 0
            for area_id in stations["area id"]:
                if (config.GAUGE_ROOT / f"{area_id}_discharge.csv").exists():
                    n_csv += 1
            add(config.GAUGE_ROOT, "gauge", True,
                f"{n_csv}/{len(stations)} station discharge CSVs present")
        except Exception as exc:  # noqa: BLE001  # unreadable xlsx is itself a finding
            add(config.GAUGE_ROOT, "gauge", True, f"information.xlsx unreadable: {exc!r}")

    # Lake levels
    add(config.LAKE_ROOT / "water_level_altimetry_Albert.nc", "lakes", True)
    add(config.LAKE_ROOT / "water_level_victoria.txt", "lakes", True)
    add(config.LAKE_ROOT / "water_level_Kyoga.txt", "lakes", True)

    # Farmland / exposure rasters
    add(config.FARMLAND_ROOT / "geonode__cattle_gha.tif", "farmland", True)
    add(config.FARMLAND_ROOT / "asap_mask_crops_v04.tif", "farmland", True)
    add(config.FARMLAND_ROOT / "asap_mask_rangeland_v04.tif", "farmland", True)

    # Impact context
    worldpop = config.RAW_DATA / "worldpop"
    add(worldpop, "worldpop", True,
        f"{len(list(worldpop.glob('*')))} files" if worldpop.exists() else "")
    ipc = config.RAW_DATA / "IPC"
    add(ipc, "ipc", True, f"{len(list(ipc.glob('*.xlsx')))} files" if ipc.exists() else "")
    add(config.RAW_DATA / "GDP" / "API_SSD_DS2_en_csv_v2_2529.csv", "gdp", True)
    add(
        config.RAW_DATA / "health facilities" / "Sub-Saharan_public_health_facilities.geojson",
        "health",
        True,
    )

    # Processed ET0 (generated, not raw)
    n_et0 = 0
    for year in config.YEARS:
        p = (
            config.ET0_ROOT
            / f"ET_{year}_{config.ET0_TARGET['target_latitude']:.3f}N_"
            f"{config.ET0_TARGET['target_longitude']:.3f}E_processed.csv"
        )
        if p.exists():
            n_et0 += 1
    add(config.ET0_ROOT, "et0_processed", False,
        f"{n_et0}/{len(config.YEARS)} years processed (optional input)")

    # Committed EDA tables this package reuses as label sources
    add(config.AWAIL_WEEKLY_FLOODS_CSV, "eda_labels", True)
    add(config.NATIONAL_WEEKLY_FLOODS_CSV, "eda_labels", True)
    add(config.AWAIL_MONTHLY_EXPOSURE_CSV, "eda_exposure", True)
    add(config.NATIONAL_EXPOSURE_BASELINE_CSV, "eda_exposure", True)

    return rows


def build_inventory() -> pd.DataFrame:
    df = pd.DataFrame(_file_rows())
    df["status"] = [
        "ok" if (e and s is not None and s >= 0) else "missing"
        for e, s in zip(df["exists"], df["size_mb"])
    ]
    df.loc[~df["exists"], "status"] = "missing"
    return df


def run_check(verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """Run the check; returns (table, all_critical_present)."""
    df = build_inventory()
    if verbose:
        show = df.copy()
        show["exists"] = show["exists"].map({True: "yes", False: "NO "})
        print(show.to_string(index=False))
    n_ok = int((df["status"] == "ok").sum())
    n_missing = int((df["status"] == "missing").sum())
    critical_missing = df[(df["status"] == "missing") & df["critical"]]
    print(
        f"\n{n_ok} inputs ok, {n_missing} missing "
        f"({len(critical_missing)} critical)."
    )
    if len(critical_missing):
        print("Critical inputs still missing:")
        for _, r in critical_missing.iterrows():
            print(f"  - {r['path']}  ({r['note']})")
    return df, len(critical_missing) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(config.TABLES / "raw_data_inventory.csv"),
        help="where to write the inventory CSV",
    )
    args = parser.parse_args()

    df, ok = run_check()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"inventory written to {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
