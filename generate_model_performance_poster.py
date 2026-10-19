"""Generate A0-poster-proof performance overview visuals for both pipeline models.

Model 1 (Task A) - LightGBM county flood forecast:
    Source of truth: LightGBM_v1/outputs/outputs_fe/tables/cv_metrics_summary_aweil.csv
    (pooled out-of-fold CV, 1,810 county-weeks).

Model 2 (Task B) - climatology-downscaled ~232 m pixel forecast:
    Source of truth: climatology_maps/outputs/forecast_maps/PIXEL_PERFORMANCE.md
    (out-of-fold pixel-week evaluation, 211.5M pixel-weeks).

House style (consistent with the poster family):
    * titles #1E3F52 (deep slate blue), body text/axes #363432 (dark anthracite)
    * model colours: Model 1/2 highlight #1E3F52, persistence #D96B43,
      climatology reference #7A8B99
    * every text element >= 20 pt, clean sans-serif (DejaVu Sans)
    * transparent background, light horizontal gridlines #E0E0E0 (alpha 0.5)
    * exports 300 DPI PNG (bbox_inches='tight', transparent) + vector SVG
    * third panel in each figure: a small concrete-example table decoding
      the advertised percentages into actual score values

Outputs (poster_visuals/model_performance/):
    model1_county_forecast_performance.{png,svg}
    model2_pixel_performance_overview.{png,svg}
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

REPO_ROOT = Path(__file__).resolve().parent
METRICS_CSV = REPO_ROOT / "LightGBM_v1" / "outputs" / "outputs_fe" / "tables" / "cv_metrics_summary_aweil.csv"
PIXEL_PERF_MD = REPO_ROOT / "climatology_maps" / "outputs" / "forecast_maps" / "PIXEL_PERFORMANCE.md"
POSTER_DIR = REPO_ROOT / "poster_visuals" / "model_performance"

TEXT = "#363432"     # dark anthracite: body text and axes
TITLE = "#1E3F52"    # deep slate blue: titles
C_MODEL = "#1E3F52"  # model highlight bars
C_PERSIST = "#D96B43"  # persistence baseline bars
C_CLIM = "#7A8B99"   # climatology reference bars
GRID = "#E0E0E0"


def set_style() -> None:
    """House style for the performance overview figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "DejaVu Sans", "Arial"],
        "text.color": TEXT,
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    })


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT)
    ax.tick_params(axis="both", labelsize=20, colors=TEXT)
    ax.set_axisbelow(True)


def load_model1_pooled() -> pd.DataFrame:
    """Pooled out-of-fold metrics for LightGBM vs. baselines (county level)."""
    df = pd.read_csv(METRICS_CSV)
    pooled = df[df["scope"] == "pooled"].set_index("model")
    return pooled


def load_pixel_metrics() -> dict:
    """Parse the headline pixel-level metrics from PIXEL_PERFORMANCE.md."""
    text = PIXEL_PERF_MD.read_text()
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*\*{0,2}(.+?)\*{0,2}\s*\|\s*\*{0,2}([+-]?[0-9.]+)\*{0,2}\s*\|$", line)
        if m:
            metrics[m.group(1).strip()] = float(m.group(2))

    county_stats: list[tuple[str, float, float, float]] = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*(Aweil \w+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([+-]?[0-9.]+)\s*\|", line)
        if m:
            county_stats.append((m.group(1).strip(), float(m.group(2)),
                                 float(m.group(3)), float(m.group(4))))

    return {
        "auroc": metrics["AUROC (model map)"],
        "bss_vs_clim": metrics["BSS vs climatology"],
        "pr_auc": metrics["PR-AUC (model map)"],
        "county_stats": county_stats,  # (county, brier_model, brier_clim, bss)
    }


def _add_explanation_table(ax, header: list[str], rows: list[list[str]],
                           title: str, caption: str, col_widths: list[float]) -> None:
    """Third panel: small concrete-example table decoding the advertised %."""
    ax.axis("off")
    ax.set_title(title, pad=14, fontsize=20, fontweight="bold", color=TITLE)
    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center",
                   colWidths=col_widths, bbox=[0.0, 0.48, 1.0, 0.38])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(20)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor(TITLE)
            cell.set_text_props(color="#FFFFFF", fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r == 1 else "#F8FAFC")
            cell.set_text_props(color=TEXT, fontweight="bold" if c == 3 else "normal")
    ax.text(0.5, 0.30, caption, ha="center", va="top", fontsize=20,
            color=TEXT, linespacing=1.25, transform=ax.transAxes)


def generate_model1_overview(output_dir: Path) -> tuple[Path, Path]:
    """Model 1 (LightGBM county forecast): skill vs. baselines + area error."""
    pooled = load_model1_pooled()
    lgbm, persist, clim = pooled.loc["lightgbm_fe"], pooled.loc["persistence"], pooled.loc["climatology"]

    fig = plt.figure(figsize=(16.5, 7), dpi=300)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.30)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- Panel A: probabilistic skill vs. calendar climatology ----
    models = ["LightGBM", "Persistence", "Climatology"]
    bss_pct = [lgbm["bss_vs_climatology"] * 100, persist["bss_vs_climatology"] * 100, 0.0]
    colors = [C_MODEL, C_PERSIST, C_CLIM]
    bars = ax1.bar(models, bss_pct, width=0.55, color=colors, zorder=3)
    for bar, val in zip(bars, bss_pct):
        tag = f"+{val:.1f}%" if val > 0 else "reference"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.9, tag,
                 ha="center", va="bottom", fontsize=20, fontweight="bold", color=TEXT)
    ax1.axhline(0, color=TEXT, linewidth=1.4, zorder=2)
    ax1.set_ylim(-3, 42)
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.set_ylabel("BSS vs. Calendar Climatology (%)", fontsize=20, labelpad=10)
    ax1.grid(axis="y", color=GRID, alpha=0.5, linewidth=1.2, zorder=0)
    _clean_axes(ax1)
    ax1.set_title("A. Probabilistic Skill", pad=14, fontsize=20, fontweight="bold", color=TITLE)

    # ---- Panel B: flood-week area error (lower is better) ----
    mae = [lgbm["mae_flood_weeks"], persist["mae_flood_weeks"], clim["mae_flood_weeks"]]
    bars = ax2.bar(models, mae, width=0.55, color=colors, zorder=3)
    for bar, val in zip(bars, mae):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.35, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=20, fontweight="bold", color=TEXT)
    ax2.set_ylim(0, 19)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.set_ylabel("Area Error on Flood Weeks (km\u00b2)", fontsize=20, labelpad=10)
    ax2.grid(axis="y", color=GRID, alpha=0.5, linewidth=1.2, zorder=0)
    _clean_axes(ax2)
    ax2.set_title("B. Weekly Area Error", pad=14, fontsize=20, fontweight="bold", color=TITLE)

    # ---- Panel C: what the percentages mean concretely ----
    _add_explanation_table(
        ax3,
        header=["Measure", "Calendar", "Model", "Change"],
        rows=[["Prob. error", f"{clim['brier_score']:.4f}", f"{lgbm['brier_score']:.4f}",
               f"\u2212{lgbm['bss_vs_climatology'] * 100:.1f}%"],
              ["Area error", f"{clim['mae_flood_weeks']:.2f}", f"{lgbm['mae_flood_weeks']:.2f}",
               f"\u2212{(1 - lgbm['mae_flood_weeks'] / clim['mae_flood_weeks']) * 100:.1f}%"]],
        title="What the % Means",
        caption="Error scores: lower = better.\n"
                "Change = model vs. the\n"
                "calendar climatology.",
        col_widths=[0.34, 0.24, 0.20, 0.22],
    )

    fig.suptitle("Model 1 \u00b7 County Flood Forecast (LightGBM)",
                 fontsize=22, fontweight="bold", y=0.98, color=TITLE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12)

    png_path = output_dir / "model1_county_forecast_performance.png"
    svg_path = output_dir / "model1_county_forecast_performance.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  [Model 1] performance overview exported to:\n    - {png_path}\n    - {svg_path}")
    return png_path, svg_path


def generate_model2_overview(output_dir: Path) -> tuple[Path, Path]:
    """Model 2 (pixel downscaler): pixel ranking skill + per-county consistency."""
    px = load_pixel_metrics()

    fig = plt.figure(figsize=(16.5, 7), dpi=300)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.30)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- Panel A: pixel ranking skill (AUROC) vs. random chance ----
    bars = ax1.bar(["Pixel Forecast", "Random Chance"], [px["auroc"], 0.50],
                   width=0.5, color=[C_MODEL, C_CLIM], zorder=3)
    ax1.text(bars[0].get_x() + bars[0].get_width() / 2.0, px["auroc"] + 0.045, f"{px['auroc']:.2f}",
             ha="center", va="bottom", fontsize=20, fontweight="bold", color=TEXT)
    ax1.text(bars[1].get_x() + bars[1].get_width() / 2.0, 0.50 + 0.045, "Random guessing\n(0.50)",
             ha="center", va="bottom", fontsize=20, fontweight="bold", color=C_CLIM, linespacing=1.15)
    ax1.axhline(0.5, color=C_PERSIST, linewidth=2.2, linestyle=(0, (6, 3)), zorder=2)
    ax1.set_xlim(-0.55, 1.85)
    ax1.set_ylim(0, 1.08)
    ax1.yaxis.set_major_locator(MultipleLocator(0.25))
    ax1.set_ylabel("AUROC (Flooded vs. Dry Pixels)", fontsize=20, labelpad=10)
    ax1.grid(axis="y", color=GRID, alpha=0.5, linewidth=1.2, zorder=0)
    _clean_axes(ax1)
    ax1.set_title("A. Pixel Ranking Skill", pad=14, fontsize=20, fontweight="bold", color=TITLE)

    # ---- Panel B: positive skill in every county ----
    counties = [s[0] for s in px["county_stats"]]
    bss_pct = [s[3] * 100 for s in px["county_stats"]]
    short = [c.replace("Aweil", "").strip() for c in counties]
    bars = ax2.bar(short, bss_pct, width=0.55, color=C_MODEL, zorder=3)
    for bar, val in zip(bars, bss_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.25, f"+{val:.1f}%",
                 ha="center", va="bottom", fontsize=20, fontweight="bold", color=TEXT)
    ax2.axhline(0, color=TEXT, linewidth=1.4, zorder=2)
    ax2.set_ylim(-1.2, 10.5)
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.set_ylabel("BSS vs. Climatology (%)", fontsize=20, labelpad=10)
    ax2.grid(axis="y", color=GRID, alpha=0.5, linewidth=1.2, zorder=0)
    _clean_axes(ax2)
    ax2.set_title("B. Skill in Every County", pad=14, fontsize=20, fontweight="bold", color=TITLE)

    # ---- Panel C: what the scores mean concretely ----
    best_county, brier_model, brier_clim, best_bss = max(px["county_stats"], key=lambda s: s[3])
    _add_explanation_table(
        ax3,
        header=["Score", "Baseline", "Model", "Result"],
        rows=[["AUROC", "0.50", f"{px['auroc']:.2f}", f"{round(px['auroc'] * 100)}/100"],
              ["Brier error", f"{brier_clim:.4f}", f"{brier_model:.4f}",
               f"\u2212{best_bss * 100:.1f}%"]],
        title="What the Scores Mean",
        caption=f"AUROC: of 100 flooded/dry\n"
                f"pixel pairs, {round(px['auroc'] * 100)} ranked right.\n"
                f"Brier: {best_county} pixel error,\n"
                f"calendar vs. model per week.",
        col_widths=[0.34, 0.24, 0.20, 0.22],
    )

    fig.suptitle("Model 2 \u00b7 Pixel Downscaling Performance (~232 m Grid)",
                 fontsize=22, fontweight="bold", y=0.98, color=TITLE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12)

    png_path = output_dir / "model2_pixel_performance_overview.png"
    svg_path = output_dir / "model2_pixel_performance_overview.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  [Model 2] performance overview exported to:\n    - {png_path}\n    - {svg_path}")
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model performance overview visuals.")
    parser.add_argument("--output-dir", "-o", type=Path, default=POSTER_DIR,
                        help="Directory for the exported figures (default: poster_visuals/model_performance).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    generate_model1_overview(args.output_dir)
    generate_model2_overview(args.output_dir)


if __name__ == "__main__":
    main()
