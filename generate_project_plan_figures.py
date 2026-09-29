"""Generate publication-ready academic figures for JBG060 Capstone Project Plan.

This module creates two clean, academic figures with zero en-dashes or em-dashes:
1. Figure 1: Hydrological Basin Decoupling (2023 vs. 2024 Flood Divergence)
   Saved as 'basin_decoupling_2023_2024.png'
2. Figure 2: Staple Crop Phenology vs. Monthly Flood Risk in Northern Bahr el Ghazal
   Saved as 'crop_phenology_vs_flood_timeline_clean.png'
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns


def set_academic_style() -> None:
    """Configure matplotlib and seaborn with clean academic styling."""
    sns.set_theme(style="white", font="sans-serif")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#94A3B8",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#1E293B",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "figure.titlesize": 13.0,
        "figure.titleweight": "bold",
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.2,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def generate_figure_basin_decoupling(output_path: Path | str = "basin_decoupling_2023_2024.png") -> plt.Figure:
    """Generate Figure 1: River Basin Decoupling (2023 vs. 2024 Flood Divergence).

    Parameters
    ----------
    output_path : Path or str
        Destination path for saving the PNG file.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure object.
    """
    set_academic_style()
    output_path = Path(output_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=300)

    years = ["2023", "2024"]
    x = np.arange(len(years))
    width = 0.44

    # -------------------------------------------------------------
    # Left Subplot: National / White Nile and Sudd Wetlands
    # -------------------------------------------------------------
    white_nile_detections = [12.3, 7.1]
    color_nile = "#1E3A8A"  # Navy / Steel Blue

    bars1 = ax1.bar(
        x, white_nile_detections, width=width, color=color_nile,
        edgecolor="#0F172A", linewidth=1.0, zorder=3, alpha=0.92
    )

    ax1.set_title("White Nile and Sudd Wetlands\n(Lake Victoria Outflow System)", pad=10)
    ax1.set_ylabel("Satellite Detections (Millions)", labelpad=8)
    ax1.set_ylim(0, 15)
    ax1.yaxis.set_major_locator(MultipleLocator(3))
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6, color="#CBD5E1", zorder=0)

    # Numerical labels directly on top of bars
    for bar, val in zip(bars1, white_nile_detections):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.35,
            f"{val:.1f} M",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#0F172A"
        )

    # -------------------------------------------------------------
    # Right Subplot: River Lol Catchment (Aweil East, NBeG)
    # -------------------------------------------------------------
    lol_inundation = [13.09, 850.97]
    color_lol = "#D97706"  # Burnt Orange / Amber

    bars2 = ax2.bar(
        x, lol_inundation, width=width, color=color_lol,
        edgecolor="#78350F", linewidth=1.0, zorder=3, alpha=0.92
    )

    ax2.set_title("River Lol Catchment (Aweil East)\n(Northern Bahr el Ghazal Basin)", pad=10)
    ax2.set_ylabel("Detected Inundation (km²-weeks)", labelpad=8)
    ax2.set_ylim(0, 1000)
    ax2.yaxis.set_major_locator(MultipleLocator(200))
    ax2.set_xticks(x)
    ax2.set_xticklabels(years, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6, color="#CBD5E1", zorder=0)

    # Numerical labels directly on top of bars
    for bar, val in zip(bars2, lol_inundation):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, height + 20,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#78350F"
        )

    # Despine clean academic borders
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=4, width=0.8)

    # -------------------------------------------------------------
    # Overall Title and Subtitle / Footnote
    # -------------------------------------------------------------
    fig.suptitle(
        "Figure: Hydrological Basin Decoupling (2023 vs. 2024 Flood Divergence)",
        fontsize=13.0, fontweight="bold", y=0.98, color="#0F172A"
    )

    subtitle_text = (
        "Empirical hydrological decoupling: In 2023, record White Nile inundation occurred while Aweil East experienced drought.\n"
        "In 2024, River Lol flooded catastrophically while White Nile levels normalized, confirming independent catchment dynamics."
    )
    fig.text(
        0.5, 0.02, subtitle_text, ha="center", va="bottom",
        fontsize=8.8, color="#475569", linespacing=1.35
    )

    fig.subplots_adjust(left=0.09, right=0.94, top=0.83, bottom=0.17, wspace=0.32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 1 successfully saved to: {output_path}")

    return fig


def generate_figure_crop_phenology(output_path: Path | str = "crop_phenology_vs_flood_timeline_clean.png") -> plt.Figure:
    """Generate Figure 2: Decluttered Staple Crop Phenology vs. Monthly Flood Risk.

    Parameters
    ----------
    output_path : Path or str
        Destination path for saving the PNG file.

    Returns
    -------
    plt.Figure
        The rendered matplotlib figure object.
    """
    set_academic_style()
    output_path = Path(output_path)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 8.0), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [1.05, 2.2], "hspace": 0.12}
    )

    month_indices = np.arange(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # 26-year MODIS Climatology values
    flood_prob = [36.2, 9.5, 7.0, 5.4, 13.9, 11.7, 21.6, 57.0, 80.4, 87.9, 91.8, 76.7]
    inundation_area = [0.1, 0.0, 0.1, 0.1, 0.1, 0.5, 0.3, 5.5, 31.4, 80.6, 53.4, 7.2]

    # -------------------------------------------------------------
    # Shared Shaded Zone: September through November (8.5 to 11.5)
    # -------------------------------------------------------------
    collision_start, collision_end = 8.5, 11.5
    ax_top.axvspan(collision_start, collision_end, color="#DC2626", alpha=0.12, zorder=1)
    ax_bot.axvspan(collision_start, collision_end, color="#DC2626", alpha=0.12, zorder=1)

    # Boundary guides
    for ax in (ax_top, ax_bot):
        ax.axvline(collision_start, color="#DC2626", linestyle="--", linewidth=1.1, alpha=0.65, zorder=2)
        ax.axvline(collision_end, color="#DC2626", linestyle="--", linewidth=1.1, alpha=0.65, zorder=2)

    # -------------------------------------------------------------
    # Upper Panel: Gantt-style Phenology Timeline
    # -------------------------------------------------------------
    ax_top.set_xlim(0.5, 12.5)
    ax_top.set_ylim(-0.6, 4.3)

    # Phenological stages for Sorghum and Pearl Millet
    stages = [
        {
            "name": "Stage 1: Planting",
            "desc": "Sowing & Germination",
            "start": 4.5, "end": 6.5,
            "color": "#16A34A",  # Forest Green
            "border": "#14532D",
            "y": 3,
        },
        {
            "name": "Stage 2: Vegetative",
            "desc": "Vegetative Growth",
            "start": 6.5, "end": 8.5,
            "color": "#2563EB",  # Royal Blue
            "border": "#1E3A8A",
            "y": 2,
        },
        {
            "name": "Stage 3: Ripening",
            "desc": "Grain Ripening",
            "start": 8.5, "end": 10.5,
            "color": "#D97706",  # Dark Amber / Orange
            "border": "#78350F",
            "y": 1,
        },
        {
            "name": "Stage 4: Harvest",
            "desc": "Harvest & Drying",
            "start": 10.0, "end": 11.5,
            "color": "#DC2626",  # Crimson Red
            "border": "#7F1D1D",
            "y": 0,
        },
    ]

    for st in stages:
        start, end, y = st["start"], st["end"], st["y"]
        width = end - start
        rect = mpatches.FancyBboxPatch(
            (start, y - 0.32), width, 0.64,
            boxstyle="round,pad=0.015,rounding_size=0.12",
            facecolor=st["color"], edgecolor=st["border"], linewidth=1.1,
            zorder=3
        )
        ax_top.add_patch(rect)

        # Centered text: Stage name and 2-word description
        center_x = start + width / 2.0
        name_size = 8.4 if st["name"].startswith("Stage 4") else 8.8
        desc_size = 7.4 if st["name"].startswith("Stage 4") else 7.8
        ax_top.text(
            center_x, y + 0.10, st["name"],
            ha="center", va="center", color="#FFFFFF",
            fontsize=name_size, fontweight="bold", zorder=4
        )
        ax_top.text(
            center_x, y - 0.12, st["desc"],
            ha="center", va="center", color="#F8FAFC",
            fontsize=desc_size, fontweight="medium", zorder=4
        )

    # Clean academic Gantt styling
    ax_top.set_yticks([])
    ax_top.tick_params(axis="both", which="both", length=0)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["bottom"].set_color("#CBD5E1")
    ax_top.grid(axis="x", linestyle=":", alpha=0.45, color="#CBD5E1")

    # Clean header tag at top of the shaded collision zone
    ax_top.text(
        10.0, 3.82, "Critical Harvest Window (Flood Risk >80%)",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#DC2626", lw=0.9),
        zorder=5
    )

    # -------------------------------------------------------------
    # Lower Panel: Monthly Flood Climatology (Probability & Area)
    # -------------------------------------------------------------
    # Secondary Y-axis: Inundation Area Bars (Soft Sky Blue)
    ax_area = ax_bot.twinx()
    bars = ax_area.bar(
        month_indices, inundation_area, width=0.44, color="#BAE6FD",
        edgecolor="#38BDF8", linewidth=1.0, alpha=0.62, zorder=2,
        label="Mean Weekly Inundation Area (km²)"
    )
    ax_area.set_ylabel("Mean Inundation Area (km²)", color="#0284C7", labelpad=8)
    ax_area.tick_params(axis="y", labelcolor="#0284C7", length=4)
    ax_area.set_ylim(0, 105)
    ax_area.yaxis.set_major_locator(MultipleLocator(20))
    ax_area.spines["top"].set_visible(False)
    ax_area.grid(False)

    # Milestone label: October Area Peak (clean badge inside bar shoulder)
    ax_area.text(
        10, 68.0, "80.6 km²\n(Area Peak)",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0284C7",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#38BDF8", lw=0.9, alpha=0.96),
        zorder=5
    )

    # Primary Y-axis: Flood Probability Line (Dark Navy)
    line_prob = ax_bot.plot(
        month_indices, flood_prob, color="#0F172A", linewidth=2.6,
        marker="o", markersize=6.0, markerfacecolor="#0F172A", markeredgecolor="#FFFFFF", markeredgewidth=1.4,
        label="Flood Detection Probability (%)", zorder=4
    )
    ax_bot.fill_between(month_indices, 0, flood_prob, color="#0F172A", alpha=0.06, zorder=3)

    ax_bot.set_ylabel("Flood Detection Probability (%)", color="#0F172A", labelpad=8)
    ax_bot.tick_params(axis="y", labelcolor="#0F172A", length=4)
    ax_bot.set_ylim(0, 110)
    ax_bot.yaxis.set_major_locator(MultipleLocator(20))
    ax_bot.set_xticks(month_indices)
    ax_bot.set_xticklabels(month_names, fontweight="bold")
    ax_bot.set_xlabel("Calendar Month", labelpad=6)
    ax_bot.grid(axis="both", linestyle=":", alpha=0.45, color="#CBD5E1")

    # Milestone label: November Probability Peak
    ax_bot.text(
        11, 91.8 + 3.5, "91.8%\n(Probability Peak)",
        ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#991B1B",
        zorder=5
    )

    ax_bot.spines["top"].set_visible(False)

    # -------------------------------------------------------------
    # Clean Unified Legend in Upper Left
    # -------------------------------------------------------------
    legend_handles = [
        line_prob[0],
        mpatches.Patch(facecolor="#BAE6FD", edgecolor="#38BDF8", alpha=0.7, label="Mean Weekly Inundation Area (km²)"),
        mpatches.Patch(facecolor="#FEE2E2", edgecolor="#DC2626", alpha=0.5, label="Critical Harvest Window (Sep-Nov)"),
    ]
    ax_bot.legend(
        handles=legend_handles, loc="upper left", frameon=True,
        facecolor="#FFFFFF", framealpha=0.92, edgecolor="#CBD5E1", fontsize=8.8,
        borderpad=0.5, labelspacing=0.4
    )

    # -------------------------------------------------------------
    # Overall Title
    # -------------------------------------------------------------
    fig.suptitle(
        "Figure: Staple Crop Phenology vs. Monthly Flood Risk in Northern Bahr el Ghazal",
        fontsize=12.5, fontweight="bold", y=0.98, color="#0F172A"
    )

    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.08, hspace=0.10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure 2 successfully saved to: {output_path}")

    return fig


def main() -> None:
    """Parse CLI arguments and generate both figures."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for JBG060 Capstone Project Plan."
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path.cwd(),
        help="Directory to save the generated figures (default: current directory)."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig1_path = args.output_dir / "basin_decoupling_2023_2024.png"
    fig2_path = args.output_dir / "crop_phenology_vs_flood_timeline_clean.png"

    print("Generating Figure 1: River Basin Decoupling...")
    fig1 = generate_figure_basin_decoupling(fig1_path)
    plt.close(fig1)

    print("Generating Figure 2: Decluttered Crop Phenology vs. Flood Timeline...")
    fig2 = generate_figure_crop_phenology(fig2_path)
    plt.close(fig2)

    print("Both figures successfully generated and saved at 300 DPI.")


if __name__ == "__main__":
    main()
