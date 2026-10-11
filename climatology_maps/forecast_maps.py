"""Weekly flood-risk maps for the five Aweil counties (LightGBM * climatology).

Second stage of the pipeline: the LightGBM model (county-level, weekly) says
*how likely* a flood is this week and *how big* (q50/q90 area). It cannot say
*where* inside the county the water goes. The historical per-pixel climatology
(`pixel_climatology.py`) supplies that spatial pattern. This module combines the
two into one raster map per prediction week:

    prob(p)   = det_prob(c,w)  x  freq_given(p,c,m)     P(pixel p is wet this week)
    sev(p)    = min(1, freq(p,c,m) x q50(c,w) / hist_area(c,m))
                                                       expected flood share of the
                                                       pixel (heaviest/lightest),
                                                       replaces the historical mean
                                                       area with the model's median
    worst(p)  = same with q90(c,w)                      worst-case severity
    clim(p)   = clim_det_prob(c,m) x freq_given(p,c,m)  pure-climatology reference
                                                       (the map without the model)

where m is the calendar month of the target week, freq is the pixel's historical
weekly flood frequency, freq_given its frequency conditional on the county
flooding that month (the *spatial signature*), and hist_area(c,m) the historical
mean weekly flooded area of the county in that month. With no clipping,
sum_p sev(p) x pixel_area == q50(c,w), i.e. the map's lit area reproduces the
LightGBM area forecast.

Weeks with no historical month data for a county fall back to the county's
annual pattern (normalised), and their severity/worst layers are zero.

Inputs
------
* prediction table: LightGBM CV output (`cv_test_predictions_aweil.csv`,
  columns county, week, det_prob, q10, q50, q90, det_prob_h1..h4). Any of the
  committed output folders works; pass --predictions for a custom file.
* climatology cache: built once by `pixel_climatology.py` (raw Parquets needed
  only for that build; afterwards maps run from the cache alone).

Outputs (climatology_maps/outputs/forecast_maps/)
* <metric>_<week>.png        one image per week per metric (see --all-weeks)
* forecast_maps_interactive.html  all weeks, week slider + play button
* week_summary.csv           per-week county-level numbers used + implied totals

Examples
--------
    # gallery of the 12 highest-risk OOF weeks + interactive HTML for all weeks
    python -m climatology_maps.forecast_maps
    # every week of 2024 as PNGs
    python -m climatology_maps.forecast_maps --year 2024 --all-weeks
    # flood-persistence map: chance the flood is still there 2 weeks ahead
    python -m climatology_maps.forecast_maps --horizon 2 --metric prob
    # custom prediction file, July-December only
    python -m climatology_maps.forecast_maps --predictions PATH --month 7 8 9 10 11 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climatology_maps import pixel_climatology as pc

ADMIN2_PATH = pc.ADMIN2_PATH
OUT_DIR = PROJECT_ROOT / "climatology_maps" / "outputs" / "forecast_maps"

# Committed LightGBM CV prediction tables, most advanced first.
DEFAULT_PREDICTION_CANDIDATES = [
    PROJECT_ROOT / "LightGBM_v1" / "outputs" / "outputs_fe" / "tables" / "cv_test_predictions_aweil.csv",
    PROJECT_ROOT / "LightGBM_v1" / "outputs" / "outputs_fe_optuna" / "tables" / "cv_test_predictions_aweil.csv",
    PROJECT_ROOT / "LightGBM_v1" / "outputs" / "outputs_baseline_optuna" / "tables" / "cv_test_predictions_aweil.csv",
    PROJECT_ROOT / "LightGBM_v1" / "outputs" / "outputs_baseline" / "tables" / "cv_test_predictions_aweil.csv",
    PROJECT_ROOT / "LightGBM_v1" / "outputs" / "outputs_v3" / "tables" / "cv_test_predictions_aweil.csv",
]

METRICS = {
    "prob": "prob",
    "sev": "sev",
    "worst": "worst",
    "clim": "clim",
}
METRIC_TITLES = {
    "prob": "P(pixel flooded this week)",
    "sev": "Expected flood severity (median forecast)",
    "worst": "Expected flood severity (worst case q90)",
    "clim": "Climatology reference (no model)",
}


def load_predictions(path: str | Path | None = None) -> pd.DataFrame:
    """Load the county-week prediction table, standardised columns."""
    if path is None:
        for cand in DEFAULT_PREDICTION_CANDIDATES:
            if Path(cand).is_file():
                path = cand
                break
        else:
            raise FileNotFoundError(
                "No committed prediction table found; pass --predictions PATH "
                "(e.g. LightGBM_v1/outputs/outputs_fe/tables/cv_test_predictions_aweil.csv)"
            )
    df = pd.read_csv(path, parse_dates=["week"])
    required = {"county", "week", "det_prob", "q10", "q50", "q90"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction table {path} misses columns {sorted(missing)}")
    df = df.sort_values(["county", "week"]).reset_index(drop=True)
    print(f"  [predictions] {path}")
    print(f"    {len(df)} county-weeks, {df['week'].nunique()} weeks, "
          f"counties: {', '.join(sorted(df['county'].unique()))}")
    return df


# --------------------------------------------------------------------------- #
# County-month assets (climatology-derived, indexed for fast week compositing)
# --------------------------------------------------------------------------- #
def county_month_assets(climatology: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (county, month): historical mean weekly area + climatological prob."""
    pm, cm = climatology["pixel_month"], climatology["county_month"]
    hist_area = pm.assign(contribution=pm["freq"] * pm["pixel_area_km2"]).groupby(
        ["county", "month"], observed=True
    )["contribution"].sum().reset_index()
    assets = cm.merge(hist_area, on=["county", "month"], how="left")
    assets["hist_area_km2"] = assets["contribution"].fillna(0.0)
    return assets.drop(columns=["contribution"])


def compose_week(
    week_date: pd.Timestamp,
    predictions: pd.DataFrame,
    climatology: dict[str, pd.DataFrame],
    assets: pd.DataFrame,
    horizon: int = 0,
) -> pd.DataFrame:
    """Compose the four layers for one target week across the five counties.

    Returns one row per grid cell: county, lat, lon, pixel_area_km2,
    prob, sev, worst, clim (all in [0, 1] or NaN where no prediction exists).
    """
    grid = climatology["grid"]
    pm = climatology["pixel_month"]
    fallback = climatology["annual_fallback"]

    rows = []
    for county, cgrid in grid.groupby("county", observed=True):
        pred = predictions[(predictions["county"] == county)
                           & (predictions["week"] == week_date)]
        if pred.empty:
            continue  # no forecast for this county/week (NaN everywhere)
        pred = pred.iloc[0]

        month = week_date.month
        month_pm = pm[(pm["county"] == county) & (pm["month"] == month)]
        month_assets = assets[(assets["county"] == county) & (assets["month"] == month)]

        base = cgrid[["lat", "lon", "pixel_area_km2"]].merge(
            month_pm[["lat", "lon", "freq", "freq_given"]], on=["lat", "lon"], how="left"
        )
        base["freq"] = base["freq"].fillna(0.0)
        base["freq_given"] = base["freq_given"].fillna(0.0)

        hist_area = float(month_assets["hist_area_km2"].iloc[0]) if len(month_assets) else 0.0
        clim_dp = float(month_assets["clim_det_prob"].iloc[0]) if len(month_assets) else float("nan")

        if hist_area <= 0 or base["freq_given"].sum() == 0:
            # No historical month data for this county: use the county's annual
            # pattern (normalised weights) for placement only.
            fb = fallback[fallback["county"] == county][["lat", "lon", "fallback_weight"]]
            base = base.merge(fb, on=["lat", "lon"], how="left")
            base["freq_given"] = base["fallback_weight"].fillna(0.0)
            base = base.drop(columns=["fallback_weight"])

        det_prob = float(pred["det_prob"] if horizon == 0 else pred[f"det_prob_h{horizon}"])
        q50, q90 = float(pred["q50"]), float(pred["q90"])

        base["prob"] = det_prob * base["freq_given"]
        if hist_area > 0:
            base["sev"] = np.minimum(1.0, base["freq"] * (q50 / hist_area))
            base["worst"] = np.minimum(1.0, base["freq"] * (q90 / hist_area))
            base["sev_u"] = base["freq"] * (q50 / hist_area)  # unclipped, for audit
        else:
            base["sev"] = 0.0
            base["worst"] = 0.0
            base["sev_u"] = 0.0
        base["clim"] = clim_dp * base["freq_given"]
        base["county"] = county
        rows.append(base)

    if not rows:
        return pd.DataFrame(columns=["county", "lat", "lon", "pixel_area_km2", "prob", "sev", "worst", "clim"])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Raster helpers
# --------------------------------------------------------------------------- #
def pivot_raster(frame: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot (lat, lon, metric) onto a rectilinear raster; NaN outside counties."""
    lats = np.sort(frame["lat"].unique())
    lons = np.sort(frame["lon"].unique())
    idx = pd.MultiIndex.from_product([lats, lons], names=["lat", "lon"])
    values = frame.set_index(["lat", "lon"])[metric].reindex(idx)
    matrix = values.to_numpy().reshape(len(lats), len(lons))
    return matrix, lons, lats


def aggregate_raster(frame: pd.DataFrame, metric: str, agg_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse (max-aggregated) raster for the interactive map."""
    f = frame.copy()
    f["lon_a"] = np.round(f["lon"] / agg_step) * agg_step
    f["lat_a"] = np.round(f["lat"] / agg_step) * agg_step
    g = f.groupby(["lat_a", "lon_a"], observed=True)[metric].max().reset_index()
    return pivot_raster(g.rename(columns={"lat_a": "lat", "lon_a": "lon"}), metric)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_week_png(
    frame: pd.DataFrame,
    week: pd.Timestamp,
    metric: str,
    out_dir: str | Path,
    boundaries: gpd.GeoDataFrame,
    cmap: str = "YlOrRd",
    title_suffix: str = "",
) -> Path:
    """One static PNG for one (week, metric) at native grid resolution."""
    matrix, lons, lats = pivot_raster(frame, metric)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric}_{week.date()}.png"

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("lightgrey")

    min_lon, min_lat = lons.min(), lats.min()
    max_lon, max_lat = lons.max(), lats.max()
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    im = ax.imshow(
        matrix[::-1, :],
        extent=(min_lon - 0.5 * span_lon / len(lons), max_lon + 0.5 * span_lon / len(lons),
                min_lat - 0.5 * span_lat / len(lats), max_lat + 0.5 * span_lat / len(lats)),
        origin="lower",
        cmap=cm,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    boundaries.boundary.plot(ax=ax, color="white", linewidth=0.8, alpha=0.9)
    # County name labels
    for _, row in boundaries.iterrows():
        ax.annotate(row["county"], xy=row.geometry.representative_point().coords[0],
                    ha="center", va="center", fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45))
    ax.set_xlim(min_lon - 0.02, max_lon + 0.02)
    ax.set_ylim(min_lat - 0.02, max_lat + 0.02)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label(METRIC_TITLES[metric], rotation=270, labelpad=16)
    title = f"Aweil flood risk — week of {week.date()}\n{METRIC_TITLES[metric]}"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_interactive_html(
    coarse_frames: dict[pd.Timestamp, np.ndarray],
    lons: np.ndarray,
    lats: np.ndarray,
    metric: str,
    out_path: str | Path,
    title: str = "Aweil flood risk by week",
) -> Path:
    """Plotly animated heatmap: one frame per week, slider + play button.

    `coarse_frames` maps week -> already-aggregated value matrix (same
    lons/lats for every frame, produced by aggregate_raster). The full-resolution
    weekly frames are never held in memory at once (see main()).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = go.Heatmap(
        z=coarse_frames[first := min(coarse_frames)], x=lons, y=lats,
        colorscale="YlOrRd", zmin=0, zmax=1,
        colorbar={"title": METRIC_TITLES[metric]},
        hovertemplate="lon %{x:.3f}<br>lat %{y:.3f}<br>value %{z:.3f}<extra></extra>",
        zsmooth=False,
    )

    ordered = sorted(coarse_frames)
    go_frames = []
    for wk in ordered:
        go_frames.append(go.Frame(
            data=[go.Heatmap(z=coarse_frames[wk], x=lons, y=lats)],
            name=str(wk.date()),
        ))

    steps = [
        dict(
            method="animate", label=str(wk.date())[:10],
            args=[[str(wk.date())],
                  dict(mode="immediate",
                       frame={"duration": 250, "redraw": False},
                       transition={"duration": 0})],
        )
        for wk in ordered
    ]
    slider = dict(steps=steps, x=0, y=0, len=1.0, pad=dict(t=30), currentvalue={"prefix": "week: "})
    buttons = [
        dict(label="Play", method="animate",
             args=[None, dict(frame={"duration": 300, "redraw": False}, fromcurrent=True)]),
        dict(label="Pause", method="animate", args=[[None], dict(frame={"duration": 0})]),
    ]

    fig = go.Figure(data=[base], frames=go_frames)
    fig.update_layout(
        title=title,
        xaxis=dict(title="longitude", showgrid=False, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(title="latitude", showgrid=False, zeroline=False),
        height=720, width=860,
        updatemenus=[dict(type="buttons", buttons=buttons, x=1.0, y=1.12, xanchor="right")],
        sliders=[slider],
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.write_html(out_path)
    print(f"  [html] {out_path} ({len(coarse_frames)} weeks, {len(lats)}x{len(lons)} cells/frame)")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _week_selector(predictions: pd.DataFrame, year: int | None, months: list[int] | None) -> pd.Series:
    weeks = pd.Series(sorted(predictions["week"].unique()))
    if year is not None:
        weeks = weeks[weeks.dt.year == year]
    if months:
        weeks = weeks[weeks.dt.month.isin(months)]
    return weeks


def _peak_weeks(predictions: pd.DataFrame, n: int) -> pd.Series:
    """Highest-risk weeks: rank by expected area (sum of det_prob x q50)."""
    risk = predictions.assign(risk=predictions["det_prob"] * predictions["q50"])
    ranked = risk.groupby("week")["risk"].sum().sort_values(ascending=False).head(n)
    return pd.Series(sorted(ranked.index))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None, help="path to a county-week prediction CSV")
    parser.add_argument("--horizon", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="duration head to use as the event probability "
                             "(1-4 = flood still detected h weeks ahead; 0 = same week)")
    parser.add_argument("--year", type=int, default=None, help="only weeks of this year")
    parser.add_argument("--month", type=int, nargs="*", default=None,
                        help="only weeks whose month is in this list (1-12)")
    parser.add_argument("--all-weeks", action="store_true",
                        help="render a PNG for every selected week (default: top-12 peak weeks)")
    parser.add_argument("--top-weeks", type=int, default=12, help="size of the default PNG gallery")
    parser.add_argument("--metric", choices=list(METRICS), default="sev",
                        help="layer shown in the interactive HTML")
    parser.add_argument("--cmap", default="YlOrRd")
    parser.add_argument("--agg-step", type=float, default=0.025,
                        help="interactive raster cell size in degrees (0.025 ~ 2.5 km)")
    parser.add_argument("--outdir", default=str(OUT_DIR))
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    print("=" * 95)
    print("AWEIL WEEKLY FLOOD-RISK MAPS — LightGBM county forecasts x pixel climatology")
    print("=" * 95)

    climatology = pc.get_climatology()
    assets = county_month_assets(climatology)
    predictions = load_predictions(args.predictions)
    boundaries = pc.load_aweil_counties()

    selected = _week_selector(predictions, args.year, args.month)
    if selected.empty:
        raise SystemExit("No prediction weeks match the --year/--month filters.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.all_weeks:
        png_weeks = list(selected)
    else:
        png_weeks = [w for w in _peak_weeks(predictions, args.top_weeks) if w in set(selected)]
        if args.year is not None or args.month:
            print(f"  [note] PNG gallery limited to the top-{args.top_weeks} peak weeks "
                  f"of the selection; use --all-weeks for every week")

    print(f"\nCompositing {len(selected)} weeks (horizon h={args.horizon}) ...")
    summary_rows = []
    boundary_geo = boundaries
    png_weeks = set(png_weeks)

    # Streaming design: compose and render one week at a time; hold only the
    # tiny max-aggregated coarse matrices for the interactive HTML, never the
    # full-resolution frames for all weeks (that was the v1 OOM: ~25 GB in RAM).
    coarse: dict[pd.Timestamp, np.ndarray] = {}
    coarse_meta: tuple[np.ndarray, np.ndarray] | None = None
    for i, week in enumerate(selected):
        frame = compose_week(week, predictions, climatology, assets, horizon=args.horizon)
        # Audit summary: per county, inputs vs implied totals.
        for county, sub in frame.groupby("county", observed=True):
            pred = predictions[(predictions["county"] == county) & (predictions["week"] == week)].iloc[0]
            summary_rows.append({
                "week": week.date(), "county": county,
                "det_prob": pred["det_prob"], "q50": pred["q50"], "q90": pred["q90"],
                "implied_area_km2": float((sub["sev"] * sub["pixel_area_km2"]).sum()),
                "implied_area_km2_unclipped": float((sub["sev_u"] * sub["pixel_area_km2"]).sum()),
                "pixels_wet_50pct": int((sub["prob"] > 0.5).sum()),
                "pixels_sev_gt_half": int((sub["sev"] > 0.5).sum()),
            })
        if week in png_weeks:
            for metric in ("prob", "sev", "worst", "clim"):
                p = render_week_png(frame, week, metric, outdir, boundary_geo,
                                    cmap=args.cmap,
                                    title_suffix=f"h{args.horizon}" if args.horizon else "")
                print(f"  [png] {p}")
        if not args.no_html:
            z, lons, lats = aggregate_raster(frame, args.metric, args.agg_step)
            coarse[week] = z
            if coarse_meta is None:
                coarse_meta = (lons, lats)
        del frame  # free the full-resolution week before the next one
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(selected)} weeks", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "week_summary.csv", index=False)
    print(f"  [table] {outdir / 'week_summary.csv'}")

    if not args.no_html:
        print("\nRendering interactive HTML (week slider) ...")
        render_interactive_html(
            coarse, coarse_meta[0], coarse_meta[1], args.metric,
            outdir / "forecast_maps_interactive.html",
            title=f"Aweil flood risk by week — {METRIC_TITLES[args.metric]}"
                  + (f" (horizon h{args.horizon})" if args.horizon else ""),
        )

    peak = summary.sort_values("implied_area_km2", ascending=False).head(5)
    print("\nTop-5 county-weeks by implied flooded area:")
    print(peak[["week", "county", "det_prob", "q50", "implied_area_km2"]].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()