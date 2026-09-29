"""Crop Phenology vs. Flood Timeline Analysis for Northern Bahr el Ghazal, South Sudan.

This module computes empirical monthly flood probabilities from the 26-year
satellite record (2000-2025) and generates an integrated visualization comparing
staple crop phenological stages (Planting, Vegetative, Grain Ripening, Harvest)
against the monthly flood probability curve.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def get_default_paths(base_dir: Path | None = None) -> tuple[Path, Path, Path, Path]:
    """Resolve data input and output paths relative to repository root."""
    if base_dir is None:
        curr = Path(__file__).resolve().parent
        repo_root = curr.parent if curr.name == "EDA_crop_phenology" else curr
    else:
        repo_root = Path(base_dir)

    season_path = repo_root / "EDA_flood_masks" / "outputs" / "tables" / "monthly_seasonality.csv"
    weekly_path = repo_root / "EDA_flood_masks" / "outputs" / "tables" / "weekly_county_floods.csv"
    out_fig_dir = repo_root / "EDA_crop_phenology" / "outputs" / "figures"
    out_tab_dir = repo_root / "EDA_crop_phenology" / "outputs" / "tables"

    return season_path, weekly_path, out_fig_dir, out_tab_dir


def load_monthly_flood_probability(
    season_path: Path, weekly_path: Path
) -> pd.DataFrame:
    """Compute monthly flood probability and mean detected area across Northern Bahr el Ghazal.

    Parameters
    ----------
    season_path : Path
        Path to monthly_seasonality.csv from EDA_flood_masks.
    weekly_path : Path
        Path to weekly_county_floods.csv from EDA_flood_masks.

    Returns
    -------
    pd.DataFrame
        Table containing month, calendar_weeks, flood_weeks, flood_probability_pct,
        and mean_flood_area_km2.
    """
    if not season_path.exists():
        raise FileNotFoundError(f"Missing required seasonality file: {season_path}")
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing required weekly floods file: {weekly_path}")

    df_season = pd.read_csv(season_path)
    df_weekly = pd.read_csv(weekly_path)

    # Count distinct weeks with flood detections in NBeG per calendar month
    df_weekly["month"] = pd.to_datetime(df_weekly["week"]).dt.month
    unique_flood_weeks = df_weekly.groupby(["month", "week"]).size().reset_index()
    weeks_per_month_with_flood = unique_flood_weeks.groupby("month")["week"].nunique()

    # Calendar weeks per month (from 26-year reference calendar in monthly_seasonality.csv)
    cal_weeks = df_season.groupby("month")["calendar_weeks"].first()
    mean_area_sum = df_season.groupby("month")["mean_weekly_detected_area_km2"].sum()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for m in range(1, 13):
        total_w = int(cal_weeks.get(m, 115))
        flood_w = int(weeks_per_month_with_flood.get(m, 0))
        prob_pct = (flood_w / total_w) * 100.0 if total_w > 0 else 0.0
        rows.append({
            "month": m,
            "month_name": month_names[m - 1],
            "calendar_weeks": total_w,
            "weeks_with_flood": flood_w,
            "flood_probability_pct": prob_pct,
            "mean_flood_area_km2": float(mean_area_sum.get(m, 0.0)),
        })

    return pd.DataFrame(rows)


def get_crop_phenology_stages() -> list[dict]:
    """Define phenological stages for staple crops (Sorghum, Pearl Millet, Groundnuts) in NBeG."""
    return [
        {
            "stage_id": "0",
            "title": "LAND PREPARATION",
            "line2": "Tillage & seedbed prep",
            "line3": "Dry Season • Hazard: 5–9%",
            "start": 0.6,
            "end": 4.4,
            "color": "#F1F5F9",       # Soft Slate
            "border_color": "#94A3B8",
            "title_color": "#334155",
            "sub_color": "#475569",
            "badge_color": "#64748B",
            "y": 3,
        },
        {
            "stage_id": "1",
            "title": "STAGE 1: PLANTING",
            "line2": "Sowing & emergence",
            "line3": "Low Hazard: 12–14%",
            "start": 4.6,
            "end": 6.7,
            "color": "#16A34A",       # Forest Emerald
            "border_color": "#14532D",
            "title_color": "#FFFFFF",
            "sub_color": "#DCFCE7",
            "badge_color": "#BBF7D0",
            "y": 3,
        },
        {
            "stage_id": "2",
            "title": "STAGE 2: VEGETATIVE",
            "line2": "Tillering, elongation & flowering",
            "line3": "Rising Hazard: 22–57%",
            "start": 6.8,
            "end": 8.8,
            "color": "#2563EB",       # Royal Blue
            "border_color": "#1E3A8A",
            "title_color": "#FFFFFF",
            "sub_color": "#DBEAFE",
            "badge_color": "#BFDBFE",
            "y": 2,
        },
        {
            "stage_id": "3",
            "title": "STAGE 3: RIPENING",
            "line2": "Grain filling & dough",
            "line3": "Severe Hazard: 80–88%",
            "start": 8.7,
            "end": 10.3,
            "color": "#D97706",       # Deep Amber Gold
            "border_color": "#78350F",
            "title_color": "#FFFFFF",
            "sub_color": "#FEF3C7",
            "badge_color": "#FDE68A",
            "y": 1,
        },
        {
            "stage_id": "4",
            "title": "STAGE 4: HARVEST & DRYING",
            "line2": "Panicle cutting, threshing & field drying",
            "line3": "Critical Hazard: 77–92%",
            "start": 10.2,
            "end": 12.4,
            "color": "#DC2626",       # Crimson Red
            "border_color": "#7F1D1D",
            "title_color": "#FFFFFF",
            "sub_color": "#FEE2E2",
            "badge_color": "#FECACA",
            "y": 0,
        },
    ]


def plot_crop_phenology_vs_flood_timeline(
    df: pd.DataFrame,
    stages: list[dict],
    output_path: Path | None = None,
) -> plt.Figure:
    """Generate the refined, ultra-clear Crop Phenology vs. Flood Timeline chart."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 1.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14.2, 9.8), dpi=300, sharex=True,
        gridspec_kw={"height_ratios": [1.18, 2.22], "hspace": 0.08}
    )

    months = df["month"].to_numpy()
    month_names = df["month_name"].to_numpy()
    prob = df["flood_probability_pct"].to_numpy()
    area = df["mean_flood_area_km2"].to_numpy()

    # -------------------------------------------------------------
    # Background Banding & Shading Across Both Subplots
    # -------------------------------------------------------------
    for m in range(1, 13):
        if m % 2 == 0:
            ax_top.axvspan(m - 0.5, m + 0.5, color="#F8FAFC", zorder=0)
            ax_bot.axvspan(m - 0.5, m + 0.5, color="#F8FAFC", zorder=0)

    # Critical Hazard Collision Window (Sep 1 to Nov 30: month 8.5 to 11.5)
    collision_start, collision_end = 8.5, 11.5
    ax_top.axvspan(collision_start, collision_end, color="#FEE2E2", alpha=0.55, zorder=1)
    ax_bot.axvspan(collision_start, collision_end, color="#FEE2E2", alpha=0.55, zorder=1)

    # Vertical drop guidelines at key agricultural & flood transitions
    key_transitions = [
        (4.5, "#16A34A", ":", 1.0, 0.6),   # Planting begins
        (6.75, "#2563EB", ":", 1.0, 0.6),  # Vegetative begins
        (8.5, "#EF4444", "--", 1.5, 0.8),  # Collision window begins
        (10.25, "#D97706", ":", 1.0, 0.6), # Harvest begins
        (11.5, "#EF4444", "--", 1.5, 0.8), # Collision window ends
    ]
    for x_t, col, style, lw, alpha in key_transitions:
        ax_top.axvline(x_t, color=col, linestyle=style, linewidth=lw, alpha=alpha, zorder=2)
        ax_bot.axvline(x_t, color=col, linestyle=style, linewidth=lw, alpha=alpha, zorder=2)

    # -------------------------------------------------------------
    # 1. TOP PANEL: Crop Phenology Stages (Cascading Gantt)
    # -------------------------------------------------------------
    ax_top.set_xlim(0.5, 12.5)
    ax_top.set_ylim(-0.65, 3.65)

    for st in stages:
        start, end, y = st["start"], st["end"], st["y"]
        w = end - start

        # Main stage card
        rect = mpatches.FancyBboxPatch(
            (start, y - 0.39), w, 0.78,
            boxstyle="round,pad=0.012,rounding_size=0.15",
            facecolor=st["color"], edgecolor=st["border_color"], linewidth=1.4,
            zorder=4
        )
        ax_top.add_patch(rect)

        # Stage Title (Line 1)
        font_sz = 8.8 if st["stage_id"] == "3" else 9.2
        ax_top.text(
            start + w / 2.0, y + 0.16,
            st["title"],
            ha="center", va="center", color=st["title_color"],
            fontsize=font_sz, fontweight="bold", zorder=5
        )

        # Activity Description (Line 2)
        ax_top.text(
            start + w / 2.0, y - 0.04,
            st["line2"],
            ha="center", va="center", color=st["sub_color"],
            fontsize=7.8, fontweight="medium", zorder=5
        )

        # Hazard Badge (Line 3)
        ax_top.text(
            start + w / 2.0, y - 0.22,
            st["line3"],
            ha="center", va="center", color=st["badge_color"],
            fontsize=7.5, fontweight="bold", zorder=5
        )

    # Top panel y-ticks and labels
    ax_top.set_yticks([0, 1, 2, 3])
    ax_top.set_yticklabels(
        ["Stage 4: Harvest", "Stage 3: Ripening", "Stage 2: Vegetative", "Prep & Planting"],
        fontsize=10.5, fontweight="bold", color="#1E293B"
    )
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", length=0, pad=8)
    ax_top.grid(axis="x", linestyle=":", alpha=0.4, color="#94A3B8")
    ax_top.grid(axis="y", visible=False)

    # Title & Subtitle
    ax_top.text(
        0.0, 1.25,
        "Agricultural Phenology vs. Seasonal Flood Timeline",
        transform=ax_top.transAxes, fontsize=15.5, fontweight="bold", color="#0F172A", va="top"
    )
    ax_top.text(
        0.0, 1.09,
        "Northern Bahr el Ghazal, South Sudan  •  Staple Crops (Sorghum, Pearl Millet, Groundnuts) vs. 26-Year MODIS Satellite Record (2000–2025)",
        transform=ax_top.transAxes, fontsize=10.0, color="#475569", va="top"
    )

    # Banner across the top of collision window
    ax_top.text(
        10.0, 3.48,
        "CRITICAL HAZARD COLLISION WINDOW",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#B91C1C",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEE2E2", edgecolor="#EF4444", lw=1.0),
        zorder=6
    )

    # -------------------------------------------------------------
    # 2. BOTTOM PANEL: Flood Probability Curve & Inundation Area
    # -------------------------------------------------------------
    # Right Axis: Mean Inundation Area (km²)
    ax_area = ax_bot.twinx()
    bars = ax_area.bar(
        months, area, width=0.42, color="#93C5FD", alpha=0.50,
        edgecolor="#3B82F6", linewidth=1.3, zorder=2,
        label="Mean Weekly Inundation Area (km²)"
    )
    ax_area.set_ylabel("Mean Inundation Area (km²)", color="#1D4ED8", fontsize=11, fontweight="bold", labelpad=12)
    ax_area.tick_params(axis="y", labelcolor="#1D4ED8", labelsize=10, length=4)
    ax_area.set_ylim(0, 115)
    ax_area.grid(False)

    # Left Axis: Flood Probability (%)
    m_fine = np.linspace(1, 12, 350)
    interp = PchipInterpolator(months, prob)
    prob_fine = np.clip(interp(m_fine), 0, 100)

    # Area fill under the probability curve
    ax_bot.fill_between(m_fine, 0, prob_fine, color="#1D4ED8", alpha=0.12, zorder=3)

    # High-contrast bold curve
    line_prob = ax_bot.plot(
        m_fine, prob_fine, color="#1E3A8A", linewidth=3.2,
        label="Flood Detection Probability (%)", zorder=4
    )
    # Scatter markers with crisp white halos
    ax_bot.scatter(months, prob, color="#1E3A8A", s=64, edgecolor="#FFFFFF", linewidth=2.2, zorder=5)

    # Data value labels with refined spacing
    for m, p in zip(months, prob):
        is_peak = m in [9, 10, 11]
        col = "#B91C1C" if is_peak else "#1E3A8A"
        weight = "bold" if is_peak or m in [8, 12] else "medium"

        if m in [1, 2, 3, 4]:
            ax_bot.text(m, p + 3.8, f"{p:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight=weight, color=col, zorder=6)
        elif m == 5:
            # Shift May percentage slightly left to avoid green arrow
            ax_bot.text(m - 0.22, p + 2.5, f"{p:.1f}%", ha="right", va="bottom", fontsize=9.0, fontweight=weight, color=col, zorder=6)
        elif m == 6:
            ax_bot.text(m, p + 3.8, f"{p:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight=weight, color=col, zorder=6)
        elif m == 7:
            ax_bot.text(m - 0.16, p + 4.0, f"{p:.1f}%", ha="right", va="bottom", fontsize=9.0, fontweight=weight, color=col, zorder=6)
        elif m == 8:
            ax_bot.text(m - 0.22, p + 2.5, f"{p:.1f}%", ha="right", va="bottom", fontsize=9.4, fontweight=weight, color=col, zorder=6)
        elif m == 9:
            # September label centered cleanly above marker
            ax_bot.text(m, p + 4.2, f"{p:.1f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=col, zorder=6)
        elif m == 10:
            # October peak probability and area labels with crisp badge
            ax_bot.text(m, p + 4.2, f"{p:.1f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#B91C1C", zorder=6)
            ax_area.text(
                m, 72.0, f"{area[9]:.1f} km²\n(Area Peak)",
                ha="center", va="center", fontsize=8.4, fontweight="bold", color="#1E3A8A",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#3B82F6", lw=1.0, alpha=0.95),
                zorder=7
            )
        elif m == 11:
            ax_bot.text(m, p + 4.2, f"{p:.1f}%\n(Prob Peak)", ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=col, zorder=6)
        elif m == 12:
            ax_bot.text(m + 0.15, p + 3.5, f"{p:.1f}%", ha="left", va="bottom", fontsize=9.2, fontweight=weight, color=col, zorder=6)

    ax_bot.set_ylim(0, 115)
    ax_bot.set_ylabel("Flood Detection Probability (%)", color="#1E3A8A", fontsize=11, fontweight="bold", labelpad=12)
    ax_bot.tick_params(axis="y", labelcolor="#1E3A8A", labelsize=10, length=4)
    ax_bot.yaxis.set_major_locator(MultipleLocator(20))
    ax_bot.set_xticks(range(1, 13))
    ax_bot.set_xticklabels(month_names, fontsize=11, fontweight="bold", color="#0F172A")
    ax_bot.set_xlabel("Calendar Month", fontsize=11.5, fontweight="bold", labelpad=8, color="#0F172A")
    ax_bot.grid(axis="x", linestyle=":", alpha=0.4, color="#94A3B8")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5, color="#CBD5E1")

    # -------------------------------------------------------------
    # 3. Clean Contextual Callout Cards (In Non-Conflicting Areas)
    # -------------------------------------------------------------
    # Callout 1: Safe Planting Window (May–Jun)
    ax_bot.annotate(
        "Safe Planting Window (May–Jun)\n• Flood hazard minimal (11.7–13.9%)\n• Gentle rains assist seed germination\n  without causing waterlogging",
        xy=(5.0, 16.0), xytext=(1.8, 36),
        arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=1.6, mutation_scale=11),
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.3),
        fontsize=8.8, fontweight="medium", color="#14532D", zorder=7
    )

    # Callout 2: Critical Collision Impact Card (Clean placement at x=4.7, y=86)
    collision_badge = (
        "CRITICAL COLLISION FINDING (Sep–Nov):\n"
        "• Flood probability surges to 88–92% peak\n"
        "• Inundation area reaches 80.6 km² maximum\n"
        "• Direct collision with grain ripening & harvest\n"
        "• Severe food insecurity risk: >12% of cropland\n"
        "  flooded in Aweil South during peak harvest"
    )
    ax_bot.text(
        4.7, 86.0,
        collision_badge,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#FEF2F2", edgecolor="#EF4444", lw=1.5, alpha=0.96),
        fontsize=8.9, fontweight="medium", color="#991B1B", zorder=8
    )

    # -------------------------------------------------------------
    # 4. Clean Legend
    # -------------------------------------------------------------
    legend_handles = [
        line_prob[0],
        mpatches.Patch(facecolor="#93C5FD", edgecolor="#3B82F6", alpha=0.6, label="Mean Weekly Inundation Area (km²)"),
        mpatches.Patch(facecolor="#FEE2E2", edgecolor="#EF4444", label="Critical Hazard Collision Window (Sep–Nov)"),
    ]
    ax_bot.legend(
        handles=legend_handles, loc="upper left", frameon=True,
        facecolor="#FFFFFF", framealpha=0.96, edgecolor="#CBD5E1", fontsize=9.2,
        borderpad=0.55, labelspacing=0.45
    )

    # Figure margins
    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.08, hspace=0.09)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure successfully saved to: {output_path}")

    return fig


def main():
    """Run data loading, table generation, and figure plotting."""
    season_path, weekly_path, out_fig_dir, out_tab_dir = get_default_paths()
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_tab_dir.mkdir(parents=True, exist_ok=True)

    print("Computing monthly flood probabilities and agricultural phenology metrics...")
    df = load_monthly_flood_probability(season_path, weekly_path)
    stages = get_crop_phenology_stages()

    out_csv = out_tab_dir / "monthly_crop_phenology_flood_risk.csv"
    df.to_csv(out_csv, index=False)
    print(f"Metrics table saved to: {out_csv}")

    out_png = out_fig_dir / "crop_phenology_vs_flood_timeline.png"
    fig = plot_crop_phenology_vs_flood_timeline(df, stages, out_png)
    plt.close(fig)
    print("Done! Crop Phenology vs. Flood Timeline chart generated successfully.")


if __name__ == "__main__":
    main()
