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
    summary = weekly.groupby("week")[list(METRICS)].agg(["mean", "count", lambda s: s.quantile(.1), lambda s: s.quantile(.9)])
    stat_names = {"mean": "mean", "count": "n_years", "<lambda_0>": "p10", "<lambda_1>": "p90"}
    summary.columns = [f"{metric}_{stat_names[stat]}" for metric, stat in summary.columns]
    # Use the midpoint of each seven-day bin to group weekly values by month.
    dated = weekly.copy()
    dated["month"] = [
        (pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=(int(w)-1)*7+3)).month
        for y, w in zip(dated.year, dated.week)
    ]
    annual_months = dated.groupby(["year", "month"])[list(METRICS)].mean()
    monthly = annual_months.groupby("month").mean()
    monthly.to_csv(OUT / "tables/aweil_monthly_typical_week.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = [
        ("rainfall_mm", "1   Rainfall", "mm per week", "#3185BA"),
        ("net_moisture_mm", "2   Rainfall minus reference evaporation", "mm per week", "#25866A"),
        ("flood_pixels", "3   Detected flooding", "Flood pixels", "#CB5751"),
    ]
    for ax, (metric, title, unit, color) in zip(axes, labels):
        values = monthly[metric]
        colors = ["#D9A776" if v < 0 else color for v in values] if metric == "net_moisture_mm" else color
        ax.bar(monthly.index, values, width=.65, color=colors)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=10)
        ax.set_ylabel(unit)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#ECECEC")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.axhline(0, color="#888888", lw=.8)
        ax.set_xticks(range(1,13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.tick_params(labelbottom=True)
        ax.set_xlim(.4,12.6)
    axes[1].text(.99, .92, "Above zero: rainfall exceeds evaporation", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=10, color="#555555")
    first, last = weekly.year.min(), weekly.year.max()
    fig.suptitle("Rain comes first. Flooding peaks later.", x=.12, ha="left", fontsize=22, weight="bold", y=.98)
    fig.text(.12,.925, f"Aweil · Average weekly conditions by month · {first}–{last}", color="#555555", fontsize=12)
    fig.text(.12,.025,f"Each bar shows a typical week, averaged across {weekly.year.nunique()} years. Weeks are assigned to their midpoint month.\nFlooding = weekly maximum detected pixel count. No detection does not necessarily mean dry land.", fontsize=9, color="#666666")
    fig.tight_layout(rect=(.02,.075,1,.90), h_pad=2)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"figures/aweil_weekly_climatology.{extension}", dpi=200)
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
