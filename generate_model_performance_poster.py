"""Generate A0-poster-proof performance overview visuals for both pipeline models.

Design: attention-grabbing stat cards (the boast) + a small concrete-example
table (the proof/context). No charts, no number dumps.

Model 1 (Task A) - LightGBM county flood forecast:
    Source of truth: LightGBM_v1/outputs/outputs_fe/tables/cv_metrics_summary_aweil.csv
    (pooled out-of-fold CV, 1,810 county-weeks).

Model 2 (Task B) - climatology-downscaled ~232 m pixel forecast:
    Source of truth: climatology_maps/outputs/forecast_maps/PIXEL_PERFORMANCE.md
    (out-of-fold pixel-week evaluation, 211.5M pixel-weeks).

House style (consistent with the poster family):
    * titles #1E3F52 (deep slate blue), body text/axes #363432 (dark anthracite)
    * card accents: model #1E3F52, persistence #7A8B99, coral #D96B43
    * every text element >= 20 pt, clean sans-serif (DejaVu Sans)
    * transparent background, 300 DPI PNG (bbox_inches='tight') + vector SVG

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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
METRICS_CSV = REPO_ROOT / "LightGBM_v1" / "outputs" / "outputs_fe" / "tables" / "cv_metrics_summary_aweil.csv"
PIXEL_PERF_MD = REPO_ROOT / "climatology_maps" / "outputs" / "forecast_maps" / "PIXEL_PERFORMANCE.md"
POSTER_DIR = REPO_ROOT / "poster_visuals" / "model_performance"

TEXT = "#363432"     # dark anthracite: body text
TITLE = "#1E3F52"    # deep slate blue: titles
C_GREEN = "#059669"  # green card accent
C_BLUE = "#1E3F52"   # deep slate blue card accent
C_ORANGE = "#D96B43" # coral/amber card accent
TINT = {"#059669": "#ECFDF5", "#1E3F52": "#E8EEF2", "#D96B43": "#FAECE5"}


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


def load_model1_pooled() -> pd.DataFrame:
    """Pooled out-of-fold metrics for LightGBM vs. baselines (county level)."""
    df = pd.read_csv(METRICS_CSV)
    return df[df["scope"] == "pooled"].set_index("model")


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
        "county_stats": county_stats,  # (county, brier_model, brier_clim, bss)
    }


def draw_stat_card(ax, stat: str, label: str, accent: str,
                   stat_size: int = 52, label_size: int = 20) -> None:
    """One attention-grabbing stat card: big positive number + label."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=TINT[accent], edgecolor=accent, linewidth=2.5, zorder=2))
    ax.text(0.5, 0.60, stat, ha="center", va="center",
            fontsize=stat_size, fontweight="bold", color=accent, zorder=3)
    ax.text(0.5, 0.26, label, ha="center", va="center",
            fontsize=label_size, fontweight="bold", color=TEXT, linespacing=1.25, zorder=3)


def add_table_panel(ax, header: list[str], rows: list[list[str]],
                    col_widths: list[float], bbox: list[float]) -> None:
    """Supporting context: small concrete-example table decoding the boasted %."""
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, cellLoc="center",
                   colWidths=col_widths, bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(20)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor(TITLE)
            cell.set_text_props(color="#FFFFFF", fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r == 1 else "#F8FAFC")
            cell.set_text_props(color=TEXT)



def generate_model1_overview(output_dir: Path) -> tuple[Path, Path]:
    """Model 1 (LightGBM county forecast): stat cards + concrete-example table."""
    pooled = load_model1_pooled()
    lgbm, persist, clim = pooled.loc["lightgbm_fe"], pooled.loc["persistence"], pooled.loc["climatology"]
    bss_clim = lgbm["bss_vs_climatology"] * 100
    bss_pers = lgbm["bss_vs_persistence"] * 100
    pinball_cut = (1 - lgbm["mean_pinball_loss"] / clim["mean_pinball_loss"]) * 100
    pinball_cut_pers = (1 - lgbm["mean_pinball_loss"] / persist["mean_pinball_loss"]) * 100

    fig = plt.figure(figsize=(16.5, 10.25), dpi=300)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.12)

    draw_stat_card(fig.add_subplot(gs[0, 0]), f"+{bss_clim:.1f}%",
                   "BSS vs.\nClimatology", C_GREEN)
    draw_stat_card(fig.add_subplot(gs[0, 1]), f"+{pinball_cut:.1f}%",
                   "Pinball Skill Score\nvs. Climatology", C_BLUE)
    draw_stat_card(fig.add_subplot(gs[0, 2]), f"+{bss_pers:.1f}%",
                   "BSS vs.\nPersistence", C_ORANGE)
    draw_stat_card(fig.add_subplot(gs[0, 3]), f"+{pinball_cut_pers:.1f}%",
                   "Pinball Skill Score\nvs. Persistence", C_GREEN)

    ax_tbl = fig.add_subplot(gs[1, :])
    add_table_panel(
        ax_tbl,
        header=["Measure", "Climatology", "Persistence", "Model",
                "Model vs.\nClimatology", "Model vs.\nPersistence"],
        rows=[["Brier", f"{clim['brier_score']:.4f}", f"{persist['brier_score']:.4f}",
               f"{lgbm['brier_score']:.4f}", f"+{bss_clim:.1f}%", f"+{bss_pers:.1f}%"],
              ["Pinball loss", f"{clim['mean_pinball_loss']:.3f}", f"{persist['mean_pinball_loss']:.3f}",
               f"{lgbm['mean_pinball_loss']:.3f}", f"+{pinball_cut:.1f}%", f"+{pinball_cut_pers:.1f}%"]],
        col_widths=[0.13, 0.145, 0.145, 0.12, 0.23, 0.23],
        bbox=[0.03, 0.20, 0.94, 0.60],
    )

    fig.suptitle("Model 1 \u00b7 County Flood Forecast (LightGBM)",
                 fontsize=22, fontweight="bold", y=0.98, color=TITLE)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.06)

    png_path = output_dir / "model1_county_forecast_performance.png"
    svg_path = output_dir / "model1_county_forecast_performance.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  [Model 1] performance overview exported to:\n    - {png_path}\n    - {svg_path}")
    return png_path, svg_path


def generate_model2_overview(output_dir: Path) -> tuple[Path, Path]:
    """Model 2 (pixel downscaler): stat cards + county skill table with explanation."""
    px = load_pixel_metrics()
    auroc = px["auroc"]
    bss_all = px["bss_vs_clim"] * 100
    n_pos = sum(1 for s in px["county_stats"] if s[3] > 0)

    fig = plt.figure(figsize=(16.5, 9.0), dpi=300)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0],
                          height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.28)

    # Hero card: the headline ranking skill
    draw_stat_card(fig.add_subplot(gs[:, 0]), f"{auroc:.2f}",
                   "AUROC \u2014 Flood\nRanking Skill", C_GREEN,
                   stat_size=100, label_size=24)
    # Smaller supporting cards
    draw_stat_card(fig.add_subplot(gs[0, 1]), f"+{bss_all:.1f}%",
                   "BSS vs. Climatology\n(pixel probabilities)", C_BLUE)
    draw_stat_card(fig.add_subplot(gs[1, 1]), f"{n_pos} / 5",
                   "Counties with\nPositive Skill", C_ORANGE)

    fig.suptitle("Model 2 \u00b7 Pixel Downscaling Performance (~232 m Grid)",
                 fontsize=22, fontweight="bold", y=0.98, color=TITLE)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.05)

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
