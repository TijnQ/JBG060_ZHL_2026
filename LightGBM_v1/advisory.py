"""Stakeholder advisory and cattle movement module for LightGBM_v1 pipeline.

Turns LightGBM flood area and duration forecasts into:
1. `advisories.csv`: Risk tier (0-3), agricultural phase, crop/rangeland exposure, and advisory text per county-week.
2. `movement_advice.csv`: Multi-horizon duration guidance recommending safe destination counties for cattle relocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from LightGBM_v1 import config

__all__ = [
    "AG_PHASE_BY_MONTH",
    "advisory_text",
    "climatology_threshold_stats",
    "exposure_for_week",
    "generate_advisories",
    "generate_movement_advice",
    "magnitude_tier",
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
    """Return agricultural phase description for calendar month."""
    return AG_PHASE_BY_MONTH[int(month)]


def climatology_threshold_stats(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per (county, month) training-period flood area thresholds (mean, p80, p95)."""
    train = features_df[features_df["split"] == "train"]
    stats = (
        train.groupby(["county", "month"], observed=True)["y_true"]
        .agg(
            clim_mean="mean",
            clim_p80=lambda s: float(s.quantile(0.80)),
            clim_p95=lambda s: float(s.quantile(0.95)),
        )
        .reset_index()
    )
    return stats


def magnitude_tier(area: float, stats_row: pd.Series) -> int:
    """Compute risk tier 0-3 by comparing forecast area to historical county-month thresholds."""
    if np.isnan(area) or area <= stats_row["clim_mean"]:
        return 0
    if area <= stats_row["clim_p80"]:
        return 1
    if area <= stats_row["clim_p95"]:
        return 2
    return 3


def _load_exposure_table() -> pd.DataFrame:
    """Load historical monthly exposure mapping."""
    if not config.AWEIL_MONTHLY_EXPOSURE_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(config.AWEIL_MONTHLY_EXPOSURE_CSV)
    df = df[df["county"].isin(config.AWEIL_COUNTIES)].copy()
    unusual = df[df["flood_type"] == "unusual"].copy()

    hist = unusual.groupby(["county", "month"], observed=True).agg(
        hist_crop_exposed_ha=("crop_exposed_hectares", "sum"),
        hist_rangeland_exposed_ha=("rangeland_exposed_hectares", "sum"),
    ).reset_index()

    return hist


def _cattle_by_county() -> dict[str, float]:
    """Load mapped cattle counts from national baseline table."""
    if not config.NATIONAL_EXPOSURE_BASELINE_CSV.exists():
        return {}
    df = pd.read_csv(config.NATIONAL_EXPOSURE_BASELINE_CSV)
    return dict(zip(df["county"], df["mapped_cattle"]))


def exposure_for_week(
    area: float, county: str, month: int, hist: pd.DataFrame, clim_mean: float | None
) -> tuple[float, float]:
    """Calculate implied crop and rangeland exposed hectares for week."""
    row = hist[(hist["county"] == county) & (hist["month"] == month)]
    if row.empty or np.isnan(area) or not clim_mean or clim_mean <= 0:
        return 0.0, 0.0

    scale = min(5.0, max(0.0, area / clim_mean))
    crop_ha = float(row["hist_crop_exposed_ha"].iloc[0] * scale)
    rangeland_ha = float(row["hist_rangeland_exposed_ha"].iloc[0] * scale)
    return crop_ha, rangeland_ha


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
    """Generate plain-language stakeholder advisory text."""
    phase = month_phase(month)
    guidance = _PHASE_GUIDANCE.get(phase, "")
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    if tier == 0 and det_prob < 0.5:
        head = f"No flood expected in {county} this week (q50 {q50:.1f} km²). Routine monitoring."
    elif tier == 1:
        head = f"Elevated flood probability in {county} this week (q50 {q50:.1f} km², {det_prob:.0%} detection chance)."
    elif tier == 2:
        head = f"High flood risk in {county} this week: q50 {q50:.1f} km², above most {month_names[month]} weeks since 2000."
    else:
        head = f"SEVERE flood expected in {county} this week: q50 {q50:.1f} km², above the 95th percentile of all {month_names[month]} weeks since 2000."

    parts = [head]
    if guidance and tier >= 1:
        parts.append(f"{phase.capitalize()}: {guidance}.")
    if crop_ha >= 1.0:
        parts.append(f"Approx. {crop_ha:,.0f} ha of mapped crops in the {county} floodplain are at risk.")
    if rangeland_ha >= 1.0:
        parts.append(f"Approx. {rangeland_ha:,.0f} ha of rangeland at risk; plan early herd movement.")
    if cattle is not None and cattle > 0:
        parts.append(f"({county} maps ~{cattle:,.0f} cattle in baseline.)")

    return " ".join(parts)


def generate_advisories(pred_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive advisories.csv DataFrame."""
    stats = climatology_threshold_stats(features_df)
    hist = _load_exposure_table()
    cattle_map = _cattle_by_county()

    rows = []
    for _, r in pred_df.iterrows():
        co = r["county"]
        mo = pd.to_datetime(r["week"]).month

        s = stats[(stats["county"] == co) & (stats["month"] == mo)]
        clim_mean = float(s["clim_mean"].iloc[0]) if not s.empty else 0.0

        tier = 0 if s.empty else magnitude_tier(r["q50"], s.iloc[0])
        crop, rangeland = exposure_for_week(r["q50"], co, mo, hist, clim_mean)
        cattle = cattle_map.get(co)

        adv_str = advisory_text(co, mo, float(r["q50"]), float(r["det_prob"]), tier, crop, rangeland, cattle)

        rows.append({
            "week": r["week"],
            "county": co,
            "q50_km2": round(float(r["q50"]), 2),
            "det_prob": round(float(r["det_prob"]), 3),
            "tier": int(tier),
            "phase": month_phase(mo),
            "crop_exposed_ha": round(crop, 1),
            "rangeland_exposed_ha": round(rangeland, 1),
            "cattle_mapped": cattle,
            "advisory": adv_str,
        })

    return pd.DataFrame(rows).sort_values(["week", "county"]).reset_index(drop=True)


def generate_movement_advice(advisories_df: pd.DataFrame, duration_df: pd.DataFrame) -> pd.DataFrame:
    """Generate movement_advice.csv recommending destination counties for cattle relocation."""
    merged = advisories_df.merge(duration_df, on=["week", "county"], how="left")

    rows = []
    for week_val, group in merged.groupby("week"):
        at_risk = group[group["tier"] >= 2]
        if at_risk.empty:
            continue

        group_copy = group.copy()
        group_copy["avg_duration_risk"] = (
            group_copy["det_prob_h1"] + group_copy["det_prob_h2"] + group_copy["det_prob_h3"] + group_copy["det_prob_h4"]
        ) / 4.0

        safe_destinations = group_copy.sort_values("avg_duration_risk")

        for _, origin in at_risk.iterrows():
            best_dest = safe_destinations[safe_destinations["county"] != origin["county"]].iloc[0]

            advice_msg = (
                f"Cattle in {origin['county']} face high flood risk (tier {origin['tier']}, "
                f"{origin['det_prob']:.0%} chance). Recommended relocation destination: {best_dest['county']} "
                f"(expected 4-week flood risk {best_dest['avg_duration_risk']:.0%})."
            )

            rows.append({
                "week": week_val,
                "origin_county": origin["county"],
                "origin_tier": origin["tier"],
                "origin_det_prob": origin["det_prob"],
                "recommended_destination": best_dest["county"],
                "destination_4wk_avg_risk": round(float(best_dest["avg_duration_risk"]), 3),
                "movement_advice": advice_msg,
            })

    return pd.DataFrame(rows).sort_values(["week", "origin_county"]).reset_index(drop=True)
