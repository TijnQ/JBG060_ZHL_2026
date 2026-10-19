"""Download and inspect candidate regions for a White Nile/Sudd vs Lol comparison.

Run from the repository root: python -m EDA_flood_masks.basin_boundaries
HydroBASINS supplies geometry and routing IDs, not the regional names used here.
The outputs are proposed analysis regions, not validated named-basin boundaries.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data/hydrobasins"
OUT = ROOT / "EDA_flood_masks/outputs/basin_comparison"
SOURCE_URL = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_af_lev06_v1c.zip"
REGIONS = {
    "1727": "White Nile / Sudd corridor candidate",
    "17286": "Lol catchment candidate",
}


def prepare_regions():
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "hybas_af_lev06_v1c.zip"
    if not archive.exists():
        with urlopen(SOURCE_URL, timeout=60) as response:
            content = response.read()
        if not content.startswith(b"PK"):
            raise ValueError("The HydroBASINS download is not a ZIP archive")
        archive.write_bytes(content)
    basins = gpd.read_file("zip://" + str(archive))
    members, rows, geometries = [], [], []
    for prefix, label in REGIONS.items():
        selected = basins[basins.PFAF_ID.astype(str).str.startswith(prefix)].copy()
        if selected.empty:
            raise ValueError(f"No HydroBASINS units match {prefix}")
        outlets = selected[~selected.NEXT_DOWN.isin(selected.HYBAS_ID)]
        if len(outlets) != 1:
            raise ValueError(f"Region {prefix} must have one downstream exit")
        geometry = selected.geometry.union_all()
        if not geometry.is_valid:
            raise ValueError(f"Invalid region geometry: {prefix}")
        selected["proposed_region"] = label
        members.append(selected)
        outlet = outlets.iloc[0]
        rows.append({"region": label, "pfaf_prefix": prefix,
                     "selection_status": "candidate_not_validated_named_basin",
                     "source": "HydroBASINS v1.c Africa standard level 6",
                     "source_url": SOURCE_URL, "member_count": len(selected),
                     "outlet_hybas_id": int(outlet.HYBAS_ID),
                     "next_down_hybas_id": int(outlet.NEXT_DOWN),
                     "selected_sub_area_km2": selected.SUB_AREA.sum(),
                     "outlet_total_upstream_area_km2": outlet.UP_AREA})
        geometries.append(geometry)
    regions = gpd.GeoDataFrame(rows, geometry=geometries, crs=basins.crs)
    overlap = regions.to_crs("EPSG:6933").geometry.iloc[0].intersection(
        regions.to_crs("EPSG:6933").geometry.iloc[1]).area
    assert overlap < 1, "The proposed comparison regions must not overlap"
    lol = regions.iloc[1]
    assert abs(lol.selected_sub_area_km2 / lol.outlet_total_upstream_area_km2 - 1) < .001
    OUT.mkdir(parents=True, exist_ok=True)
    regions.to_file(OUT / "proposed_regions.geojson", driver="GeoJSON")
    gpd.GeoDataFrame(pd.concat(members), crs=basins.crs).to_file(
        OUT / "source_subbasins.geojson", driver="GeoJSON")
    regions.drop(columns="geometry").to_csv(OUT / "region_definitions.csv", index=False)
    metadata = {"download_url": SOURCE_URL,
                "download_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "source_page": "https://www.hydrosheds.org/products/hydrobasins",
                "selection_status": "Candidate regions. Geographic labels assigned by analyst.",
                "white_nile_scope": "Selected corridor only, not its full upstream catchment or an official Sudd wetland boundary.",
                "routing_note": "Both regions ultimately drain through the Nile network. Non-overlap does not imply hydrological independence.",
                "data_coverage_note": "The supplied h20v08/h21v08 flood tiles do not cover every part of these candidate regions. Check the footprint before calculating area percentages."}
    (OUT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return regions


def plot_regions(regions):
    counties = gpd.read_file(ROOT / "raw_data/Administrative boundaries/ssd_admin2.geojson").to_crs(regions.crs)
    colors = ["#3185BA", "#D96B43"]
    fig, ax = plt.subplots(figsize=(11, 7))
    counties.plot(ax=ax, facecolor="#F5F5F5", edgecolor="#CCCCCC", linewidth=.4)
    for (_, region), color in zip(regions.iterrows(), colors):
        gpd.GeoSeries([region.geometry], crs=regions.crs).plot(
            ax=ax, facecolor=color, alpha=.4, edgecolor=color, linewidth=1.3)
    counties[counties.adm2_name.eq("Aweil East")].boundary.plot(
        ax=ax, color="#333333", linewidth=1.8, linestyle="--")
    ax.annotate("Aweil East county", xy=(27.6, 9.3), xytext=(25.3, 10.0),
                arrowprops={"arrowstyle": "-", "color": "#444444"}, fontsize=11)
    ax.text(26.1, 7.5, "Lol candidate\nPFAF 17286", ha="center", color="#843E27", weight="bold")
    ax.text(31.35, 8.2, "Sudd corridor\ncandidate\nPFAF 1727", ha="center", color="#184866", weight="bold")
    ax.axhline(10, ls=":", color="#777777", lw=1)
    ax.set(xlim=(23.8, 33.2), ylim=(5.3, 10.7), xlabel="Longitude", ylabel="Latitude")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Proposed regions for the basin comparison", fontsize=19, weight="bold", y=.98)
    fig.text(.5, .925, "HydroBASINS v1.c · basin selection requires hydrological review", ha="center", fontsize=11, color="#555555")
    fig.legend(handles=[Patch(facecolor=colors[0], alpha=.5, label="White Nile / Sudd corridor candidate"),
                        Patch(facecolor=colors[1], alpha=.5, label="Lol catchment candidate"),
                        Line2D([], [], color="#333333", ls="--", label="Aweil East county"),
                        Line2D([], [], color="#777777", ls=":", label="10°N flood-tile edge")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5,.015))
    fig.tight_layout(rect=(.02,.12,1,.90))
    fig.savefig(OUT / "proposed_regions_map.png", dpi=180)
    fig.savefig(OUT / "proposed_regions_map.svg")
    plt.close(fig)


def weekly_detected_areas(cells, calendar):
    """Count a detected grid cell once per week across dates and flood classes."""
    from .flood_eda import pixel_area_km2
    unique = cells.drop_duplicates(["week", "lat", "lon"]).copy()
    unique["area_km2"] = pixel_area_km2(unique.lat)
    return unique.groupby("week").area_km2.sum().reindex(calendar, fill_value=0)


def check_weekly_area_union():
    """Repeated detections count once in a week and again in a different week."""
    import numpy as np
    from .flood_eda import pixel_area_km2
    calendar = pd.date_range("2023-01-02", periods=3, freq="W-MON")
    cells = pd.DataFrame({"week": [calendar[0], calendar[0], calendar[1]],
                          "lat": [9., 9., 9.], "lon": [27., 27., 27.]})
    actual = weekly_detected_areas(cells, calendar)
    area = float(pixel_area_km2(9.))
    assert np.allclose(actual, [area, area, 0])


def compare_candidate_regions(regions, years=(2023, 2024), output_prefix="candidate"):
    """Exploratory comparison within the supplied flood-tile footprint only."""
    import numpy as np
    import pyarrow.dataset as ds
    from pyproj import Geod
    from shapely import contains_xy, prepare
    from shapely.geometry import box
    from .flood_eda import PRODUCT_RADIUS_M

    geod = Geod(a=PRODUCT_RADIUS_M, b=PRODUCT_RADIUS_M)
    footprint = box(20, 0, 40, 10)
    analysed = regions.copy()
    analysed.geometry = analysed.geometry.intersection(footprint)
    analysed.to_file(OUT / "candidate_analysis_footprints.geojson", driver="GeoJSON")
    weekly_frames, totals = [], []
    for _, region in analysed.iterrows():
        shape = region.geometry
        prepare(shape)
        area_km2 = abs(geod.geometry_area_perimeter(shape)[0]) / 1e6
        west, south, east, north = shape.bounds
        tiles = [tile for tile, bounds in [("h20v08", box(20, 0, 30, 10)), ("h21v08", box(30, 0, 40, 10))] if shape.intersects(bounds)]
        for year in years:
            parts = []
            for file_year in (year, year + 1):
                for tile in tiles:
                    for kind in ("recurring", "unusual"):
                        path = ROOT / f"raw_data/flood_masks/compact_{kind}/flood_events_{tile}_{file_year}.parquet"
                        keep = ((ds.field("lon") >= west) & (ds.field("lon") <= east)
                                & (ds.field("lat") >= south) & (ds.field("lat") <= north)
                                & (ds.field("date") >= f"{year}-01-01")
                                & (ds.field("date") < f"{year+1}-01-07"))
                        frame = ds.dataset(path, format="parquet").to_table(columns=["date", "lat", "lon"], filter=keep).to_pandas()
                        frame = frame.loc[contains_xy(shape, frame.lon.to_numpy(), frame.lat.to_numpy())].copy()
                        frame["week"] = pd.to_datetime(frame.date).dt.to_period("W-SUN").dt.start_time
                        frame = frame.loc[frame.week.dt.year.eq(year), ["week", "lat", "lon"]]
                        parts.append(frame.drop_duplicates())
            cells = pd.concat(parts, ignore_index=True).drop_duplicates(["week", "lat", "lon"])
            calendar = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-MON")
            areas = weekly_detected_areas(cells, calendar)
            assert np.isfinite(areas).all() and areas.ge(0).all()
            assert areas.le(area_km2 * 1.01).all()
            weekly = pd.DataFrame({"region": region.region, "year": year, "week": calendar,
                                   "detected_area_km2": areas.to_numpy(), "analysis_area_km2": area_km2})
            weekly["detected_area_percent"] = 100 * weekly.detected_area_km2 / area_km2
            weekly_frames.append(weekly)
            totals.append({"region": region.region, "year": year, "n_weeks": len(calendar),
                           "analysis_area_km2": area_km2,
                           "detected_area_km2_weeks": areas.sum(),
                           "mean_weekly_detected_area_percent": weekly.detected_area_percent.mean(),
                           "status": "exploratory_candidate_boundaries"})
            print(f"Calculated {region.region}, {year}", flush=True)
    pd.concat(weekly_frames, ignore_index=True).to_csv(OUT / f"{output_prefix}_weekly_floods.csv", index=False)
    summary = pd.DataFrame(totals)
    summary.to_csv(OUT / f"{output_prefix}_annual_comparison.csv", index=False)
    if output_prefix == "candidate":
        plot_candidate_comparison(summary)
    return summary


def plot_candidate_comparison(summary):
    """Use one unit and one y-axis while making the provisional scope explicit."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(2)
    for year, offset, color in [(2023, -.18, "#1E3F52"), (2024, .18, "#D96B43")]:
        values = summary[summary.year.eq(year)].set_index("region").loc[list(REGIONS.values()), "mean_weekly_detected_area_percent"].to_numpy()
        bars = ax.bar(positions + offset, values, width=.32, color=color, label=str(year))
        for bar, value in zip(bars, values):
            label = f"{value:.3f}%" if value < .1 else f"{value:.2f}%"
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, value), xytext=(0,7), textcoords="offset points", ha="center", fontsize=12, weight="bold")
    ax.set_xticks(positions, ["White Nile / Sudd corridor", "Lol catchment"])
    ax.set_ylabel("Mean weekly area with flood detections (%)")
    ax.set_ylim(0, summary.mean_weekly_detected_area_percent.max() * 1.28)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.suptitle("Weekly flood detections by region", fontsize=18, weight="bold", y=.97)
    fig.text(.5, .90, "EXPLORATORY · candidate boundaries · available flood-tile coverage only", ha="center", fontsize=10, color="#7D472E")
    fig.text(.11, .03, "Same weekly pixel-union method and area normalisation for both regions.\nNo detection does not confirm dry conditions. These regions do not establish independent basins.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(.02,.09,1,.86))
    fig.savefig(OUT / "candidate_comparison.png", dpi=180)
    fig.savefig(OUT / "candidate_comparison.svg")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Also recompute an exploratory 2023/2024 comparison on the candidate regions")
    parser.add_argument("--all-years", action="store_true", help="Search all 25 years with a complete following January file (2000-2024)")
    args = parser.parse_args()
    check_weekly_area_union()
    regions = prepare_regions()
    plot_regions(regions)
    if args.all_years:
        compare_candidate_regions(regions, years=range(2000, 2025), output_prefix="all_years")
    elif args.compare:
        print(compare_candidate_regions(regions).to_string(index=False))
    print(f"Saved candidate boundary data and map to {OUT}")


if __name__ == "__main__":
    main()
