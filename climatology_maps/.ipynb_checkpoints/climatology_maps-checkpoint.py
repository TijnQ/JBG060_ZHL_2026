"""Flood-climatology maps for South Sudan.

Pure *historical* (climatology) approach: no model is trained and no forecast is made.
We summarise 26 years of satellite flood detections (2000-2025) into a per-county,
per-calendar-month climatology and render it as choropleth maps.

Two key ideas behind the design:

* *Only changes based on input*: the map is a function of its inputs and nothing else.
  Give it a calendar month (and a baseline year window) and you get that month's
  climatology. Move the month slider and the map changes; nothing is learned or fit.
* *km3 of water per county*: the raw data only tells us *where* and *when* a pixel was
  detected as flooded, i.e. an area in km2. A volume in km3 needs a depth. We turn the
  detected area into a water volume with an explicitly-configurable assumed mean depth
  (default 1 m), so the user controls the only non-observed ingredient.

The plotting function accepts *any* county -> value column, so the same map
can later be fed *model output* (LightGBM/U-Net county-week predictions) instead of
the historical climatology, without changing the mapping code.

Inputs / files used
-------------------
* `EDA_flood_masks/outputs/national/tables/national_weekly_floods.csv` (committed historical data)
* `raw_data/Administrative boundaries/ssd_admin2.geojson` (IPC/HDX boundaries, gitignored;
  downloaded automatically from HDX if missing)

References to the source convention
-----------------------------------
Counties are joined on `adm2_pcode == county_code` (e.g. SS0101 == Juba).
`area_sqkm` in the boundaries equals `county_area_km2` in the flood table.

Example
-------
    python -m climatology_maps.climatology_maps --month 10 --baseline-start 2000 --baseline-end 2025
    python -m climatology_maps.climatology_maps --month 10 --depth-m 0.5 --metric km3 --title "Aweil area water"
    python -m climatology_maps.climatology_maps                            # all 12 months, interactive HTML
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEEKLY_FLOODS_CSV = (
    PROJECT_ROOT
    / "EDA_flood_masks"
    / "outputs"
    / "national"
    / "tables"
    / "national_weekly_floods.csv"
)
ADMIN2_PATH = PROJECT_ROOT / "raw_data" / "Administrative boundaries" / "ssd_admin2.geojson"

# Fallback download of the same HDX source the supplied boundaries come from.
HDX_URL = (
    "https://data.humdata.org/dataset/cdd62bd9-e442-4eac-9b44-cfee8bf79153/"
    "resource/c6bb2f28-6379-4750-b792-15915e9ab320/download/ssd_admin_boundaries.geojson.zip"
)
DEFAULT_DEPTH_M = 1.0  # assumed mean depth of standing flood water in metres -> km3
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_historical_floods(csv_path: str | Path = WEEKLY_FLOODS_CSV) -> pd.DataFrame:
    """Load the committed historical (2000-2025) county-week flood table."""
    df = pd.read_csv(csv_path, parse_dates=["week"])
    df["year"] = df["week"].dt.year
    df["month"] = df["week"].dt.month
    return df


def load_boundaries(admin2_path: str | Path = ADMIN2_PATH) -> gpd.GeoDataFrame:
    """Load admin-2 boundaries, downloading the HDX source if not present locally."""
    path = Path(admin2_path)
    if not path.exists():
        print(f"  [loader] boundaries not found at {path}; downloading from HDX ...")
        import io
        import urllib.request
        import zipfile

        raw = urllib.request.urlopen(HDX_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read("ssd_admin2.geojson"))
    gdf = gpd.read_file(path)
    # Normalise join columns regardless of HDX naming version.
    gdf = gdf.rename(columns={"adm2_pcode": "county_code"})
    if "state" not in gdf.columns:
        gdf["state"] = gdf["adm1_name"]
    if "county" not in gdf.columns:
        gdf["county"] = gdf["adm2_name"]
    return gdf


def full_county_week_grid(counties: pd.Index, weeks: pd.Series) -> pd.DataFrame:
    """Cartesian product of counties x weeks so that weeks without any detection
    count as zero (absence of a record means 'week with no detections', per the EDA)."""
    grid = pd.DataFrame(
        [[c, w] for c in counties for w in weeks],
        columns=["county_code", "week"],
    )
    grid["year"] = grid["week"].dt.year
    grid["month"] = grid["week"].dt.month
    return grid


# --------------------------------------------------------------------------- #
# Climatology calculation (pure historical average, nothing trained)
# --------------------------------------------------------------------------- #
def monthly_climatology(
    floods: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    baseline_years: tuple[int, int] = (2000, 2025),
    months: list[int] | None = None,
    depth_m: float = DEFAULT_DEPTH_M,
) -> pd.DataFrame:
    """Per county x calendar month climatology over the baseline year window.

    For every county and every Monday-start week in the window we count the detected
    flood area (0 where there was no detection). We then average, for each calendar
    month, the weekly detected area over the whole baseline -> a mean historical
    weekly flood extent (km2), its frequency, and the implied water volume (km3).

    Returns one row per county x month.
    """
    if months is None:
        months = list(range(1, 13))

    year0, year1 = baseline_years
    weeks = floods["week"].drop_duplicates().sort_values()
    weeks = weeks[(weeks.dt.year >= year0) & (weeks.dt.year <= year1)]
    counties = boundaries["county_code"].drop_duplicates().sort_values()

    grid = full_county_week_grid(counties, weeks)
    grid = grid.merge(
        floods[["county_code", "week", "detected_area_km2", "county_area_km2"]],
        on=["county_code", "week"],
        how="left",
    )
    grid["detected_area_km2"] = grid["detected_area_km2"].fillna(0.0)
    grid["county_area_km2"] = grid["county_area_km2"].ffill().bfill()

    g = grid[grid["month"].isin(months)]
    stats = (
        g.groupby(["county_code", "month"])
        .agg(
            n_weeks=("detected_area_km2", "size"),
            flood_weeks=("detected_area_km2", lambda s: int((s > 0).sum())),
            mean_weekly_flood_km2=("detected_area_km2", "mean"),
            county_area_km2=("county_area_km2", "first"),
        )
        .reset_index()
    )
    # Percent of county flooded in an average week: need county area.
    stats["mean_weekly_flood_percent"] = (
        100.0 * stats["mean_weekly_flood_km2"] / stats["county_area_km2"]
    )
    stats["flood_frequency"] = stats["flood_weeks"] / stats["n_weeks"]
    stats["mean_weekly_water_km3"] = stats["mean_weekly_flood_km2"] * (depth_m / 1000.0)
    stats = stats.merge(boundaries[["county_code", "state", "county"]], on="county_code")
    return stats.sort_values(["month", "county_code"]).reset_index(drop=True)


def pivot_for_map(stats: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str, str]:
    """Return a month-pivoted frame + (metric column, colourbar label)."""
    base_cols = ["county_code", "state", "county"]
    metrics = {
        "flood_km2": ("mean_weekly_flood_km2", "Mean weekly flood area [km²]"),
        "flood_pct": ("mean_weekly_flood_percent", "Mean % of county flooded per week"),
        "km3": ("mean_weekly_water_km3", "Mean weekly water volume [km³]"),
        "freq": ("flood_frequency", "Fraction of weeks with flooding (frequency)"),
    }
    col, label = metrics[metric]
    data = stats[base_cols + [col, "month"]].copy()
    data = data.rename(columns={col: "value"})
    return data, f"{metric} (value)", label


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_static_map(
    merged: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    out_path: str | Path,
    vmax: float | None = None,
    cmap: str = "YlOrRd",
) -> None:
    """Static matplotlib choropleth of one county->value series."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    merged.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": colorbar_label, "shrink": 0.7},
        vmin=0,
        vmax=vmax,
        missing_kwds={"color": "lightgrey", "label": "no data"},
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [map] saved {out_path}")


def render_interactive_map(
    data: pd.DataFrame,
    geojson: dict,
    title: str,
    colorbar_label: str,
    out_path: str | Path,
) -> None:
    """Plotly choropleth with a month slider: the map changes only when the user
    changes the input (the selected month). One row per county x month."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = data.assign(
        month_label=data["month"].map(lambda m: MONTH_NAMES[m - 1])
    )
    fig = px.choropleth(
        data_frame=data,
        geojson=geojson,
        locations="county_code",
        featureidkey="properties.adm2_pcode",
        color="value",
        animation_frame="month_label",
        scope="africa",
        labels={"value": colorbar_label},
        color_continuous_scale="YlOrRd",
        hover_name="county",
        hover_data={"state": True, "county_code": True, "value": ":.3f"},
    )
    fig.update_layout(title=title, coloraxis_colorbar={"title": colorbar_label})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.write_html(out_path)
    print(f"  [map] interactive HTML saved {out_path}")


def merge_value_to_geometry(boundaries: gpd.GeoDataFrame, value_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Join a county-level value series onto the geometry frame.

    The `value_df` must have one row per county_code with a `value` column.
    """
    return boundaries.merge(value_df, on="county_code", how="left")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_geojson(boundaries: gpd.GeoDataFrame) -> dict:
    # to_json cannot serialise datetime columns (e.g. valid_on/valid_to) -> drop them.
    drop = [c for c in boundaries.columns if pd.api.types.is_datetime64_any_dtype(boundaries[c])]
    gj = json.loads(boundaries.drop(columns=drop).to_json())
    # property name used by featureidkey is adm2_pcode (already present).
    return gj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", type=int, default=None,
                        help="calendar month 1-12; default: all months")
    parser.add_argument("--baseline-start", type=int, default=2000)
    parser.add_argument("--baseline-end", type=int, default=2025)
    parser.add_argument("--depth-m", type=float, default=DEFAULT_DEPTH_M,
                        help="assumed mean flood depth in metres (for km3)")
    parser.add_argument("--metric", choices=["flood_km2", "flood_pct", "km3", "freq"],
                        default="flood_km2")
    parser.add_argument("--title", default="South Sudan flood climatology",
                        help="plot title; '{metric}' and '{month}' are substituted")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "climatology_maps" / "outputs"))
    args = parser.parse_args()

    print("Loading historical flood detections ...")
    floods = load_historical_floods()
    print("Loading/administering county boundaries ...")
    boundaries = load_boundaries()

    months = [args.month] if args.month is not None else list(range(1, 13))
    baseline = (args.baseline_start, args.baseline_end)

    stats = monthly_climatology(
        floods, boundaries, baseline_years=baseline, months=months, depth_m=args.depth_m
    )
    data, _, cbar_label = pivot_for_map(stats, args.metric)

    outdir = Path(args.outdir)
    gj = build_geojson(boundaries)

    for month in months:
        frame = data[data["month"] == month]
        merged = merge_value_to_geometry(boundaries, frame)
        month_name = MONTH_NAMES[month - 1]
        title = args.title.replace("{month}", f"{month_name} {baseline[0]}-{baseline[1]}")
        vmax = float(data["value"].quantile(0.95))
        render_static_map(
            merged, "value", title, cbar_label,
            outdir / f"climatology_{args.metric}_month{month:02d}.png", vmax=vmax,
        )

    render_interactive_map(
        data, gj,
        f"{args.title} ({baseline[0]}-{baseline[1]})",
        cbar_label,
        outdir / f"climatology_{args.metric}_interactive.html",
    )

    # Summary table for the requested months.
    summary = stats[["county", "county_code", "month",
                     "flood_frequency", "mean_weekly_flood_km2",
                     "mean_weekly_water_km3", "mean_weekly_flood_percent"]]
    summary["month"] = summary["month"].map(lambda m: MONTH_NAMES[m - 1])
    out_csv = outdir / f"climatology_{args.metric}.csv"
    summary.to_csv(out_csv, index=False)
    print(f"  [table] saved {out_csv}")


if __name__ == "__main__":
    main()