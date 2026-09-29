"""Gate-0 audit: do the candidate hydro signals carry a multi-year signal
against observed flood pixels? (MODEL_RESEARCH.md experiment E1)

The EDA only measured 2024, where the Aweil correlations were weak and
*negative* — plausibly a single-year artifact (the only gauge sits
downstream of Bahr el Ghazal, and 2024 was an extreme year for the Sudd).
This audit re-measures the same relationships over 2015-2025 (default) and
reports:

1. daily lag correlations (lags 0-14 days) of every candidate signal against
   unique flood-pixel counts, for a pooled target;
2. detection coverage per year (weeks with at least one detection day, max
   day count) — "no detection" weeks may be coverage gaps, not dry ground,
   because ``cloud_frac`` is all zero in the supplied parquets;
3. a ranking of signals by max |correlation| across lags — the basis for
   feature selection in ``features.build_features``.

Everything is computed from real raw inputs only (this package never falls
back to synthetic data). Outputs: ``outputs/tables/audit_*.csv`` and
``outputs/figures/audit_*.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from modeling import config
from modeling import features as feat

__all__ = [
    "daily_flood_pixels",
    "main",
    "run_audit",
]

SIGNALS = [
    "local_tp",
    "local_ro",
    "up_tp",
    "up_ro",
    "gauge",
    "albert",
    "et0",
]
LAGS = list(range(15))


def _counties():
    """Admin-2 counties with county_code (reuses the national EDA loader)."""
    from EDA_flood_masks import national_flood_eda

    return national_flood_eda.load_south_sudan_counties(config.BOUNDARY_ADMIN2)


def daily_flood_pixels(scope: str, years: list[int]) -> pd.DataFrame:
    """Daily unique flood-pixel counts per county + a ``pooled`` column.

    Index: date. For the aweiL scope only the 5 Aweil counties are read
    (tile h20v08); for national scope all 79 counties (both tiles).
    """
    if scope == config.AWEIL_SCOPE:
        from EDA_hydrometeorology import hydrological_analysis

        counties = hydrological_analysis.load_aweil_counties(config.BOUNDARY_ADMIN2)
        obs = hydrological_analysis.load_aweil_flood_observations(
            config.FLOOD_ROOT, counties, years=years
        )
        daily = hydrological_analysis.daily_flood_pixels(obs)
        counts = daily.pivot(
            index="date", columns="county", values="active_flood_pixels"
        ).fillna(0)
    else:
        from EDA_flood_masks import national_flood_eda

        counties = _counties()
        bounds = tuple(counties.total_bounds)
        final_year = max(years)
        parts = []
        for year in years:
            print(f"  loading national flood observations for {year} ...")
            obs = national_flood_eda.load_analysis_year(
                config.FLOOD_ROOT, year, bounds, final_year
            )
            located = national_flood_eda.assign_counties(obs, counties)
            parts.append(located)
        located = pd.concat(parts, ignore_index=True)
        cells = located.drop_duplicates(["date", "county", "lat", "lon"])
        counts = (
            cells.pivot(index="date", columns="county", values="lat")
            .fillna(0)
        )
    counts = counts.astype(float)
    counts["pooled"] = counts.sum(axis=1)
    return counts


def _coverage(pixels: pd.Series) -> pd.DataFrame:
    """Per-year detection coverage from a daily pixel-count series."""
    active = (pixels > 0).astype(int)
    weekly = active.groupby(active.index.to_period("W-SUN")).sum()
    # count weeks with >=1 detection per year (NOT sum of detection days:
    # that would conflate flood frequency with magnitude)
    det = weekly[weekly > 0]
    weeks = det.groupby(det.index.year).size()
    return pd.DataFrame(
        {
            "weeks_with_detection": weeks,
            "max_day_pixels": pixels.groupby(pixels.index.year).max(),
            "active_days": active.groupby(active.index.year).sum(),
        }
    )


def run_audit(scope: str, years: list[int] | None = None) -> dict[str, pd.DataFrame]:
    years = years or config.AUDIT_YEARS
    print(f"audit scope={scope} years={years[0]}-{years[-1]}")

    signals = feat.daily_signals(scope, years=years)[SIGNALS]
    pixels = daily_flood_pixels(scope, years=years)["pooled"]
    df = signals.join(pixels.rename("flood_pixels")).dropna(subset=["flood_pixels"])
    df["flood_pixels"] = df["flood_pixels"].astype(float)
    print(f"  days with at least one detection: {len(df)} (of {len(signals)} signal days)")

    from EDA_hydrometeorology import hydrological_analysis

    corr = hydrological_analysis.lag_feature_correlations(
        df[SIGNALS], flood_target=df["flood_pixels"], lags=LAGS
    )
    corr = corr.rename(columns={"signal": "feature", "n_days": "n"}).copy()
    corr.insert(0, "scope", scope)

    ranking = (
        corr.assign(abs_corr=corr["correlation"].abs())
        .groupby("feature", observed=True)
        .agg(best_abs_corr=("abs_corr", "max"), n=("n", "first"))
        .reset_index()
        .sort_values("best_abs_corr", ascending=False)
    )
    print(f"\n  {scope}: signal ranking by max |lag 0-14 correlation|")
    print(ranking.to_string(index=False))

    return {
        "lag_corr": corr,
        "signal_ranking": ranking,
        "coverage": _coverage(pixels),
    }


def _write_heatmap(corr: pd.DataFrame, path: Path) -> None:
    if corr.empty:
        return
    wide = corr.pivot(index="lag_days", columns="feature", values="correlation")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        wide.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5
    )
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels(wide.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index)
    ax.set_ylabel("lag (days)")
    ax.set_title(f"{corr['scope'].iloc[0]}: daily signal vs flood pixels, lags 0-14")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=config.SCOPES, default=config.AWEIL_SCOPE)
    parser.add_argument("--from-year", type=int, default=config.AUDIT_YEARS[0])
    parser.add_argument("--to-year", type=int, default=config.AUDIT_YEARS[-1])
    args = parser.parse_args()

    years = list(range(args.from_year, args.to_year + 1))
    results = run_audit(args.scope, years=years)

    config.TABLES.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        out = config.TABLES / f"audit_{name}_{args.scope}.csv"
        frame.to_csv(out, index=False)
        print(f"wrote {out}")
    _write_heatmap(
        results["lag_corr"], config.FIGURES / f"audit_lag_heatmap_{args.scope}.png"
    )
    print("audit done — inspect the ranking before trusting any feature.")


if __name__ == "__main__":
    main()
