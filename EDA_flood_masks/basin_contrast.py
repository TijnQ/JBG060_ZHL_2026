"""Find the latest consecutive years with opposite regional detection changes.

Reproduce from the repository root after basin_boundaries --all-years:
python -m EDA_flood_masks.basin_contrast
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .basin_boundaries import OUT, REGIONS

NILE, LOL = REGIONS.values()
COLORS = {NILE: "#287FAE", LOL: "#D96B43"}


def select_consecutive_pair(summary):
    annual = summary.pivot(index="year", columns="region", values="mean_weekly_detected_area_percent").sort_index()
    annual = annual[[NILE, LOL]]
    changes = annual.pct_change(fill_method=None) * 100
    adjacent = annual.index.to_series().diff().eq(1)
    opposite = changes[NILE].mul(changes[LOL]).lt(0) & adjacent
    eligible = annual.index[opposite]
    if len(eligible) == 0:
        raise ValueError("No consecutive years with opposite annual changes were found")
    end = int(eligible.max())
    scan = annual.join(changes.add_suffix("_change_percent"))
    scan["opposite_changes_from_previous_year"] = opposite
    return end - 1, end, annual, changes, scan


def check_year_selection():
    rows = []
    for year, values in [(2017, [1., 1.]), (2018, [2., .5]), (2019, [1., 2.]), (2021, [3., 1.])]:
        for region, value in zip([NILE, LOL], values):
            rows.append({"year": year, "region": region, "mean_weekly_detected_area_percent": value})
    start, end, _, _, _ = select_consecutive_pair(pd.DataFrame(rows))
    assert (start, end) == (2018, 2019), "A gap in the year record must not count as a consecutive pair"


def plot_weekly_contrast(weekly, start, end):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)
    selected = weekly[weekly.year.isin([start, end])]
    ymax = selected.detected_area_percent.max() * 1.2
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax, year, focus in zip(axes, [start, end], [NILE, LOL]):
        data = weekly[weekly.year.eq(year)].pivot(index="week", columns="region", values="detected_area_percent")
        for region, label in [(NILE, "White Nile / Sudd"), (LOL, "River Lol")]:
            ax.plot(data.index.dayofyear, data[region], color=COLORS[region], label=label, lw=2.5)
        peak = data[focus].idxmax()
        values = data.loc[peak]
        x = peak.dayofyear
        ax.scatter([x], [values[focus]], color=COLORS[focus], s=34, zorder=5)
        text = (f"Week of {peak:%d %b}\n"
                f"White Nile / Sudd: {values[NILE]:.3f}%\n"
                f"River Lol: {values[LOL]:.4f}%")
        text_x = max(25, x - 125)
        ax.annotate(text, xy=(x, values[focus]), xytext=(text_x, ymax*.70),
                    arrowprops={"arrowstyle": "-", "color": "#666666"}, fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=.4"})
        ticks = pd.date_range(f"{year}-01-01", periods=12, freq="MS").dayofyear
        ax.set_xticks(ticks, months)
        ax.set(xlim=(1, 366), ylim=(0, ymax), ylabel="Detected area (%)")
        ax.set_title(str(year), loc="left", weight="bold", fontsize=15)
        ax.grid(axis="y", color="#ECECEC")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.09,.91), ncol=2, frameon=False)
    fig.suptitle("Different flood patterns in consecutive years", x=.11, ha="left", y=.99, fontsize=19, weight="bold")
    fig.text(.11,.94, "Weekly detected area as % of each analysed region. Identical scales in both panels.", fontsize=11, color="#555555")
    fig.text(.11,.02, "Exploratory results using proposed basin boundaries and the available flood-tile footprint.\nLow detections do not establish dry conditions. The two highlighted weeks are illustrative peaks.", fontsize=9, color="#666666")
    fig.tight_layout(rect=(.02,.075,1,.85), h_pad=1.8)
    for ext in ["png", "svg"]:
        fig.savefig(OUT/f"consecutive_years_weekly_contrast.{ext}", dpi=200)
    plt.close(fig)


def plot_annual_contrast(summary, start, end, output_stem="consecutive_years_annual_contrast"):
    """Present whole-year weekly mean detected areas in ordinary square kilometres."""
    data = summary[summary.year.isin([start, end])].copy()
    data["mean_weekly_area_km2"] = data.detected_area_km2_weeks / data.n_weeks
    data.to_csv(OUT / f"{output_stem}_values.csv", index=False)
    maximum = data.mean_weekly_area_km2.max()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for ax, year in zip(axes, [start, end]):
        values = data[data.year.eq(year)].set_index("region").loc[[NILE, LOL], "mean_weekly_area_km2"]
        ax.barh([1, 0], values, height=.45, color=[COLORS[NILE], COLORS[LOL]])
        for position, value in zip([1, 0], values):
            ax.text(value + maximum*.025, position, f"{value:.0f} km²", va="center", fontsize=17, weight="bold", color="#252525")
        ax.set_yticks([1, 0], ["White Nile / Sudd", "River Lol"], fontsize=13)
        ax.set(xlim=(0, maximum*1.28), ylim=(-.55, 1.55))
        ax.set_title(str(year), loc="left", fontsize=20, weight="bold", pad=10)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.axvline(0, color="#DDDDDD", linewidth=1)
    fig.text(.08,.95,"Where was more flooding detected?", fontsize=23, weight="bold", ha="left")
    fig.subplots_adjust(left=.27, right=.95, top=.75, bottom=.15, hspace=.8)
    for ext in ["png", "svg"]:
        fig.savefig(OUT/f"{output_stem}.{ext}", dpi=200)
    plt.close(fig)


def main():
    check_year_selection()
    summary = pd.read_csv(OUT/"all_years_annual_comparison.csv")
    weekly = pd.read_csv(OUT/"all_years_weekly_floods.csv", parse_dates=["week"])
    start, end, annual, changes, scan = select_consecutive_pair(summary)
    scan.to_csv(OUT/"consecutive_year_pair_scan.csv")
    summary[summary.year.isin([start,end])].to_csv(OUT/"selected_consecutive_years.csv",index=False)
    plot_weekly_contrast(weekly, start, end)
    plot_annual_contrast(summary, start, end)
    print(f"Selected the latest adjacent pair with opposite annual changes: {start}, {end}")
    print(annual.loc[[start,end]].to_string())
    print("Changes (%)", changes.loc[end].to_dict())


if __name__ == "__main__":
    main()
