"""Reproduce the Aweil weekly comparison for every available year.

Run from the repository root: python -m EDA_hydrometeorology.weekly_climatology
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .hydrological_analysis import (
    daily_flood_pixels, era5_daily_table, et_daily_table,
    load_aweil_counties, load_aweil_flood_observations,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EDA_hydrometeorology" / "outputs_county"
METRICS = {
    "rainfall_mm": ("Weekly rainfall", "mm / 7 days", "#2878AE"),
    "net_moisture_mm": ("Rainfall minus reference evaporation", "mm / 7 days", "#25866A"),
    "flood_pixels": ("Detected flood pixels", "Weekly maximum daily count", "#C84B4B"),
}


def weekly_bins(daily):
    """Align years in 52 complete seven-day bins from Jan 1. Omit final 1–2 days."""
    frame = daily.copy()
    frame["year"] = frame.index.year
    frame["week"] = (frame.index.dayofyear - 1) // 7 + 1
    frame = frame[frame.week <= 52]
    grouped = frame.groupby(["year", "week"])
    result = grouped[["rainfall_mm", "net_moisture_mm"]].agg(lambda s: s.sum(min_count=7))
    result["flood_pixels"] = grouped.flood_pixels.max()
    result.loc[grouped.size().ne(7), :] = np.nan
    return result.reset_index()


def load_weekly():
    counties = load_aweil_counties(ROOT / "raw_data/Administrative boundaries/ssd_admin2.geojson")
    bbox = dict(zip(["lon_min", "lat_min", "lon_max", "lat_max"], counties.total_bounds))
    flood_root = ROOT / "raw_data/flood_masks"
    frames = []
    for path in sorted((ROOT / "raw_data/rainfall and runoff").glob("ERA5_*.nc")):
        year = int(path.stem.split("_")[-1])
        et_path = ROOT / f"processing_data/evapotranspiration/ET_{year}_9.475N_30.725E_processed.csv"
        if not et_path.exists() or not all((flood_root / f"compact_{kind}/flood_events_h20v08_{year}.parquet").exists() for kind in ("recurring", "unusual")):
            raise FileNotFoundError(f"Missing ET or flood input for {year}")
        # Select the small Aweil box before computing hourly-to-daily sums.
        with xr.open_dataset(path) as ds:
            local = ds[["tp", "ro"]].sel(latitude=slice(bbox["lat_max"], bbox["lat_min"]), longitude=slice(bbox["lon_min"], bbox["lon_max"]))
            hydro = era5_daily_table(local.resample(valid_time="1D").sum(min_count=1)).set_index("date")
        daily = hydro.precipitation_mm.rename("rainfall_mm").to_frame()
        et = et_daily_table(pd.read_csv(et_path)).set_index("date").et_mm
        daily["net_moisture_mm"] = daily.rainfall_mm - et
        observations = load_aweil_flood_observations(flood_root, counties, years=[year])
        counts = daily_flood_pixels(observations).groupby("date").active_flood_pixels.sum()
        # Zero means no recorded detections, not confirmed dry satellite coverage.
        daily["flood_pixels"] = counts.reindex(daily.index, fill_value=0)
        frames.append(weekly_bins(daily))
        print(f"Prepared {year}", flush=True)
    return pd.concat(frames, ignore_index=True)


def plot_weekly(weekly):
    """Export the approved two-panel weekly mean figure without extra captions."""
    summary = weekly.groupby("week")[list(METRICS)].agg(["mean", "count", lambda s: s.quantile(.1), lambda s: s.quantile(.9)])
    stat_names = {"mean": "mean", "count": "n_years", "<lambda_0>": "p10", "<lambda_1>": "p90"}
    summary.columns = [f"{metric}_{stat_names[stat]}" for metric, stat in summary.columns]
    means = weekly.groupby("week")[list(METRICS)].mean()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
    surplus = means.net_moisture_mm.clip(lower=0)
    axes[0].bar(means.index, surplus, width=.7, color="#25866A",
                label="Water surplus", zorder=2)
    panels = [
        ("rainfall_mm", "1   Rainfall and water surplus", "mm per week", "#3185BA"),
        ("flood_pixels", "2   Detected flooding", "Flood pixels", "#CB5751"),
    ]
    ticks = (pd.date_range("2023-01-01", periods=12, freq="MS").dayofyear - 1) / 7 + 1
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax, (metric, title, unit, color) in zip(axes, panels):
        ax.plot(means.index, means[metric], color=color, linewidth=2.6,
                marker="o", markersize=3, label="Rainfall" if metric == "rainfall_mm" else "Detected flooding", zorder=3)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=10)
        ax.set_ylabel(unit)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#ECECEC")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0, labelbottom=True)
        ax.axhline(0, color="#888888", lw=.8)
        ax.set_xticks(ticks, months)
        ax.set_xlim(.4, 52.6)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(h_pad=2)
    for extension in ("png", "svg"):
        output_path = OUT / f"figures/aweil_weekly_climatology.{extension}"
        fig.savefig(output_path, dpi=200)
        if extension == "svg":
            lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)
    return summary


def check_weekly_bins():
    """Known answers for leap years, totals, maxima and incomplete observations."""
    dates = pd.date_range("2023-01-01", "2024-12-31")
    daily = pd.DataFrame({"rainfall_mm": 1., "net_moisture_mm": -2., "flood_pixels": 3.}, index=dates)
    weekly = weekly_bins(daily)
    assert len(weekly) == 104
    assert weekly.rainfall_mm.eq(7).all()
    assert weekly.net_moisture_mm.eq(-14).all()
    assert weekly.flood_pixels.eq(3).all()
    daily.loc["2023-01-02", "rainfall_mm"] = np.nan
    assert pd.isna(weekly_bins(daily).iloc[0].rainfall_mm)


def main():
    check_weekly_bins()
    for folder in ("tables", "figures"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    weekly = load_weekly()
    summary = plot_weekly(weekly)
    weekly.to_csv(OUT / "tables/aweil_weekly_all_years.csv", index=False)
    summary.to_csv(OUT / "tables/aweil_weekly_climatology.csv")
    assert weekly.groupby("year").size().eq(52).all()
    assert np.allclose(summary.rainfall_mm_mean, weekly.pivot(index="week", columns="year", values="rainfall_mm").mean(axis=1), equal_nan=True)
    print(f"Saved {weekly.year.nunique()} years and weekly means to {OUT}")


if __name__ == "__main__":
    main()
