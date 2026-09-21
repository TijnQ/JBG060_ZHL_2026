"""Task C: deterministic impact-to-advisory layer.

Turns Task A weekly area predictions into plain-language crop/cattle
guidance, following the "impact-based flood forecasting" pattern
(MODEL_RESEARCH.md section 3C and reference 18). Everything here is a
transparent rule over three real, already-quantified ingredients:

1. the forecast: weekly q50 area + detection probability per county
   (from ``train_task_a`` outputs — any model's rows work);
2. the agricultural phase of the month (sowing -> vegetative -> maturation
   -> harvest/dry, the seasonal cycle quantified in the EDA);
3. the exposure: hectares of mapped crops / rangeland that actually lie in
   the county's historical flood footprint (the committed EDA exposure
   tables), scaled by how large this week's forecast is relative to that
   footprint's historical mean.

No learned impact model is attempted: the IPC outcome window (2022-2025)
is too short and too confounded to support one, and the EDA already shows
exposure scales with flood extent (r = 0.59-0.71). The advisory text is
deterministic, version-controlled and unit-tested (``check_modeling``),
so every sentence in a report can be traced to a rule.

v1 scope: the aweiL pilot (its exposure table carries crop+rangeland
hectares). The national exposure baseline contributes cattle counts only;
full national advisories are a follow-up once national crop exposure is
tabulated the same way.

Usage:
    # default: the primary backbone's Aweil predictions
    python -m modeling.advisory --case-week 2024-10-07 --case-county "Aweil East"
    # or point at any method's prediction CSV:
    python -m modeling.advisory --predictions modeling/outputs/methods/xgboost/tables/task_a_predictions_aweil.csv \
        --model xgboost --case-week 2024-10-07 --case-county "Aweil East"
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from modeling import config

__all__ = [
    "AG_PHASE_BY_MONTH",
    "advisory_text",
    "build_advisories",
    "case_study",
    "climatology_stats",
    "exposure_for_week",
    "magnitude_tier",
    "main",
    "month_phase",
]

AG_PHASE_BY_MONTH = {
    1: "dry season (late phase)",
    2: "dry season",
    3: "early season (sowing)",
    4: "crop establishment",
    5: "vegetative growth",
    6: "vegetative growth (peak)",
    7: "vegetative growth (peak)",
    8: "maturation",
    9: "maturation / grain filling",
    10: "harvest",
    11: "harvest / drying",
    12: "dry season",
}

_PHASE_GUIDANCE = {
    "early season (sowing)": "sowing windows are opening — delayed or drowned seedlings are the main crop risk",
    "crop establishment": "young crops are most vulnerable to being drowned",
    "vegetative growth (peak)": "crops are at peak water demand — short flooding can be tolerated, sustained submergence is not",
    "vegetative growth": "crops are at peak water demand — short flooding can be tolerated, sustained submergence is not",
    "maturation": "grain is filling; flooding now damages the yield that is already banked",
    "maturation / grain filling": "grain is filling; flooding now damages the yield that is already banked",
    "harvest": "harvest window — move produce and equipment to higher ground before water rises",
    "harvest / drying": "harvest window closing — move produce and equipment to higher ground before water rises",
    "dry season": "little active cropping; floodplain grazing and residual flooding dominate",
    "dry season (late phase)": "little active cropping; floodplain grazing and residual flooding dominate",
}


def month_phase(month: int) -> str:
    return AG_PHASE_BY_MONTH[int(month)]


def climatology_stats(features: pd.DataFrame) -> pd.DataFrame:
    """Per (county, month) statistics of the TRAIN-period weekly area.

    Returns mean / p80 / p95, the thresholds that define the magnitude tier.
    """
    train = features[features["split"] == "train"]
    stats = (
        train.groupby(["county", "month"], observed=True)["y_true"]
        .agg(
            clim_mean=lambda s: s.mean(),
            clim_p80=lambda s: s.quantile(0.80),
            clim_p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )
    return stats


def magnitude_tier(area: float, stats_row: pd.Series) -> int:
    """0 <= mean, 1 <= p80, 2 <= p95, 3 > p95 of the same county-month history."""
    if np.isnan(area):
        return 0
    if area <= stats_row["clim_mean"]:
        return 0
    if area <= stats_row["clim_p80"]:
        return 1
    if area <= stats_row["clim_p95"]:
        return 2
    return 3


def _load_exposure() -> pd.DataFrame:
    """Historical (county, month) exposure from the committed EDA table.

    Uses the *unusual*-flood rows (recurring rows describe the permanent
    wetland, which is not an incremental risk). Falls back to the merged
    rows where a county-month has no unusual detections.
    """
    df = pd.read_csv(config.AWAIL_MONTHLY_EXPOSURE_CSV)
    if not df["county"].isin(config.AWEIL_COUNTIES).all():
        df = df[df["county"].isin(config.AWEIL_COUNTIES)]
    unusual = df[df["flood_type"] == "unusual"].copy()
    missing = (
        set(df[["county", "month"]].itertuples(index=False, name=None))
        - set(unusual[["county", "month"]].itertuples(index=False, name=None))
    )
    if missing:
        fallback = df[
            df[["county", "month"]].apply(tuple, axis=1).isin(missing)
        ].drop_duplicates(["county", "month"])
        unusual = pd.concat([unusual, fallback], ignore_index=True)
    hist = unusual.groupby(["county", "month"], observed=True).agg(
        hist_crop_exposed_ha=("crop_exposed_hectares", "sum"),
        hist_rangeland_exposed_ha=("rangeland_exposed_hectares", "sum"),
    ).reset_index()
    return hist


def exposure_for_week(
    area: float,
    county: str,
    month: int,
    hist: pd.DataFrame,
    clim_mean: float | None,
) -> tuple[float, float]:
    """(crop, rangeland) exposed hectares implied by a forecast area.

    Scaling rule: exposure scales linearly with how large the forecast is
    relative to the historical mean area for that county-month, capped at 5x
    (beyond that the linear extrapolation of the overlap ratio is not
    trustworthy). No historical flood in that county-month -> 0.
    """
    row = hist[(hist["county"] == county) & (hist["month"] == month)]
    if row.empty or np.isnan(area) or not clim_mean or clim_mean <= 0:
        return 0.0, 0.0
    scale = min(5.0, max(0.0, area / clim_mean))
    crop = float(row["hist_crop_exposed_ha"].iloc[0] * scale)
    rangeland = float(row["hist_rangeland_exposed_ha"].iloc[0] * scale)
    return crop, rangeland


def advisory_text(
    county: str,
    month: int,
    q50: float,
    det_prob: float,
    tier: int,
    crop_ha: float,
    rangeland_ha: float,
    cattle: float | None = None,
) -> str:
    """The advisory sentence set for one county-week."""
    phase = month_phase(month)
    guidance = _PHASE_GUIDANCE.get(phase, "")
    if tier == 0 and det_prob < 0.5:
        head = f"No flood expected in {county} this week (q50 {q50:.1f} km2). Routine monitoring."
    elif tier == 1:
        head = f"Elevated flood probability in {county} this week (q50 {q50:.1f} km2, {det_prob:.0%} detection chance)."
    elif tier == 2:
        head = f"High flood risk in {county} this week: q50 {q50:.1f} km2, above most {month_name(month)} weeks since 2000."
    else:
        head = f"SEVERE flood expected in {county} this week: q50 {q50:.1f} km2, above the 95th percentile of all {month_name(month)} weeks since 2000."

    parts = [head]
    if guidance and tier >= 1:
        parts.append(f"{phase.capitalize()}: {guidance}.")
    if crop_ha >= 1.0:
        parts.append(f"Approx. {crop_ha:,.0f} ha of mapped crops in the {county} floodplain are at risk.")
    if rangeland_ha >= 1.0:
        parts.append(f"Approx. {rangeland_ha:,.0f} ha of rangeland at risk; plan early herd movement.")
    if cattle is not None and cattle > 0:
        parts.append(f"({county} maps ~{cattle:,.0f} cattle in the national exposure baseline.)")
    return " ".join(parts)


def month_name(month: int) -> str:
    return ["", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"][int(month)]


def _cattle_by_county() -> dict:
    df = pd.read_csv(config.NATIONAL_EXPOSURE_BASELINE_CSV)
    return dict(zip(df["county"], df["mapped_cattle"]))


def build_advisories(
    predictions: pd.DataFrame, model: str = "lightgbm"
) -> pd.DataFrame:
    """Advisory rows for a Task A prediction frame (one model's rows)."""
    pred = predictions[predictions["model"] == model].copy()
    pred["month"] = pd.to_datetime(pred["week"]).dt.month
    pred["year"] = pd.to_datetime(pred["week"]).dt.year

    # Training-period climatology from the committed weekly label table.
    labels = pd.read_csv(config.AWAIL_WEEKLY_FLOODS_CSV)
    labels["week"] = pd.to_datetime(labels["week"])
    labels["month"] = labels["week"].dt.month
    labels["year"] = labels["week"].dt.year
    train_labels = labels[labels["year"] <= 2014].copy()
    stats = train_labels.groupby(["county", "month"], observed=True)["detected_area_km2"].agg(
        clim_mean="mean", clim_p80=lambda s: s.quantile(0.80),
        clim_p95=lambda s: s.quantile(0.95),
    ).reset_index()

    hist = _load_exposure()
    cattle_map = _cattle_by_county()

    rows = []
    for _, r in pred.iterrows():
        s = stats[(stats["county"] == r["county"]) & (stats["month"] == r["month"])]
        clim_mean = float(s["clim_mean"].iloc[0]) if not s.empty else 0.0
        tier = 0 if s.empty else magnitude_tier(r["q50"], s.iloc[0])
        crop, rangeland = exposure_for_week(r["q50"], r["county"], r["month"], hist, clim_mean)
        cattle = cattle_map.get(r["county"])
        rows.append(
            {
                "week": r["week"],
                "county": r["county"],
                "model": model,
                "q50_km2": round(float(r["q50"]), 2),
                "det_prob": round(float(r["det_prob"]), 3),
                "tier": int(tier),
                "phase": month_phase(r["month"]),
                "crop_exposed_ha": round(crop, 1),
                "rangeland_exposed_ha": round(rangeland, 1),
                "cattle_mapped": cattle,
                "advisory": advisory_text(
                    r["county"], r["month"], float(r["q50"]), float(r["det_prob"]),
                    tier, crop, rangeland, cattle,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["week", "county"]).reset_index(drop=True)


def case_study(
    advisories: pd.DataFrame, week: str, county: str
) -> str:
    """Markdown block for a single county-week — paste into the report."""
    sub = advisories[
        (pd.to_datetime(advisories["week"]) == pd.Timestamp(week))
        & (advisories["county"] == county)
    ]
    if sub.empty:
        return f"No advisory for {county} on {week}."
    r = sub.iloc[0]
    return (
        f"### Case study: {county}, week of {week}\n\n"
        f"- Forecast: **{r['q50_km2']} km2** (q50), detection probability "
        f"**{r['det_prob']:.0%}**, magnitude tier **{r['tier']}/3**\n"
        f"- Agricultural phase: {r['phase']}\n"
        f"- Exposure: ~{r['crop_exposed_ha']:,.0f} ha crops, "
        f"~{r['rangeland_exposed_ha']:,.0f} ha rangeland\n\n"
        f"> {r['advisory']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default=str(config.method_dirs(config.TASK_A_METHODS[0])["tables"] / "task_a_predictions_aweil.csv"),
        help="Task A prediction CSV (default: the primary backbone's outputs)",
    )
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--case-week", default=None, help="e.g. 2024-10-07")
    parser.add_argument("--case-county", default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    advisories = build_advisories(predictions, model=args.model)

    config.TABLES.mkdir(parents=True, exist_ok=True)
    out = config.TABLES / "advisories.csv"
    advisories.to_csv(out, index=False)
    print(f"wrote {out} ({len(advisories)} rows)")

    if args.case_week and args.case_county:
        print()
        print(case_study(advisories, args.case_week, args.case_county))


if __name__ == "__main__":
    main()
