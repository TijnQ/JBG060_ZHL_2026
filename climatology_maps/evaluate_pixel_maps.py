"""Pixel-level evaluation of the climatology-downscaled forecast maps.

Scores the LightGBM x pixel-climatology maps (`forecast_maps.compose_week`,
`prob` layer) against the observed per-pixel flood masks on the out-of-fold
test weeks, and writes a Markdown performance report.

Honest-setup rules (same philosophy as `LightGBM_v1`):
* Only the OOF test weeks are scored (fold 1: 2010-11, fold 2: 2016-17,
  fold 3: 2022-24; 2025 excluded).
* The pixel climatology is rebuilt **per fold on that fold's training years
  only** (2000-2008 / 2000-2014 / 2000-2020), so neither the model probability
  nor the spatial prior ever sees a test year. The shipped gallery PNGs use
  the full 2000-2025 prior (explanatory only); this evaluation is the honest
  one.
* A week with no satellite observation is an all-zero label (censoring) — the
  same convention the EDA and LightGBM use. Scores are therefore conservative.

Metrics
-------
* pixel-week Brier score for (a) the model map `prob`, (b) the pure climatology
  map `clim`, (c) a hard persistence mask (last week's observed mask);
  Brier Skill Score vs climatology and vs persistence.
* AUROC and PR-AUC for the model map (computed from a 1000-bin histogram so the
  pooled 200M+ pixel-weeks never materialise in RAM).
* threshold classification family (accuracy, precision, recall, F1, CSI, FAR,
  TNR) at 0.05/0.1/0.2/0.3/0.5/0.7.
* weekly flooded-area: observed km2 vs map-implied (sum prob*area) and vs the
  severity layer (sum sev*area ~= q50) — MAE, bias, Pearson r, flood-week MAE.
* Brier + BSS(vs clim) per county and per calendar month.
* reliability (calibration) of the model map in deciles of predicted p.

Usage
-----
    python -m climatology_maps.evaluate_pixel_maps
    python -m climatology_maps.evaluate_pixel_maps --predictions PATH

Output: prints a console summary and writes
`climatology_maps/outputs/forecast_maps/PIXEL_PERFORMANCE.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climatology_maps import forecast_maps as fm
from climatology_maps import pixel_climatology as pc

OUT_DIR = PROJECT_ROOT / "climatology_maps" / "outputs" / "forecast_maps"
REPORT_NAME = "PIXEL_PERFORMANCE.md"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
N_BINS = 1000  # ROC/PR histogram resolution


class Tally:
    """Streaming accumulators for pooled pixel-week statistics."""

    def __init__(self) -> None:
        self.n = 0
        self.n_wet = 0
        self.s_p = 0.0        # sum of predicted prob
        self.s_p2 = 0.0       # sum of predicted prob^2
        self.s_py = 0.0       # sum of p * y
        self.s_clim = 0.0
        self.s_clim2 = 0.0
        self.s_climy = 0.0
        self.s_persist = 0.0
        self.s_persist2 = 0.0
        self.s_persisty = 0.0
        self.tp = np.zeros(len(THRESHOLDS))
        self.fp = np.zeros(len(THRESHOLDS))
        self.fn = np.zeros(len(THRESHOLDS))
        self.tn = np.zeros(len(THRESHOLDS))
        self.pos = np.zeros(N_BINS)   # wet pixels per prob bin
        self.neg = np.zeros(N_BINS)   # dry pixels per prob bin

    def add_week(self, p, clim, persist_bit, y) -> None:
        p = np.asarray(p, dtype=float)
        clim = np.asarray(clim, dtype=float)
        yf = np.asarray(y, dtype=float)
        yb = yf > 0
        n = len(yf)
        nwet = int(yb.sum())
        self.n += n
        self.n_wet += nwet
        self.s_p += p.sum(); self.s_p2 += (p * p).sum(); self.s_py += p[yb].sum()
        self.s_clim += clim.sum(); self.s_clim2 += (clim * clim).sum()
        self.s_climy += clim[yb].sum()
        ps = np.asarray(persist_bit, dtype=float)
        self.s_persist += ps.sum(); self.s_persist2 += (ps * ps).sum()
        self.s_persisty += ps[yb].sum()
        for ti, t in enumerate(THRESHOLDS):
            pred = p >= t
            self.tp[ti] += int((pred & yb).sum())
            self.fp[ti] += int((pred & ~yb).sum())
            self.fn[ti] += int((~pred & yb).sum())
            self.tn[ti] += int((~pred & ~yb).sum())
        bins = np.clip((p * N_BINS).astype(int), 0, N_BINS - 1)
        np.add.at(self.pos, bins, yf)
        np.add.at(self.neg, bins, 1 - yf)

    def brier(self, which: str) -> float:
        if which == "prob":
            s, s2, sy = self.s_p, self.s_p2, self.s_py
        elif which == "clim":
            s, s2, sy = self.s_clim, self.s_clim2, self.s_climy
        else:
            s, s2, sy = self.s_persist, self.s_persist2, self.s_persisty
        return (s2 - 2.0 * sy + self.n_wet) / self.n

    def roc_pr(self) -> tuple[float, float]:
        pos, neg = self.pos, self.neg
        P, Ntot = pos.sum(), neg.sum()
        if P == 0 or Ntot == 0:
            return float("nan"), float("nan")
        cum_pos = np.cumsum(pos[::-1])[::-1]   # pixels with p >= current bin
        cum_neg = np.cumsum(neg[::-1])[::-1]
        tpr = cum_pos / P
        fpr = cum_neg / Ntot
        # thresholds decrease from bin 0 to bin N-1; integrate low->high so the
        # trapezoid runs over an increasing x axis (otherwise AUC comes out negated)
        auroc = float(np.trapezoid(tpr[::-1], fpr[::-1]))
        precision = cum_pos / np.maximum(cum_pos + cum_neg, 1)
        order = np.argsort(recall := tpr)
        ap = float(np.trapezoid(precision[order], recall[order]))
        return auroc, ap


# --------------------------------------------------------------------------- #
def _pack(lat, lon) -> np.ndarray:
    """Encode (lat, lon) cells into int64 keys for fast vectorised lookups.
    Snapped coordinates have 6 decimal places; lat < 100 so lat*1e6 < 1e8.
    """
    lat6 = np.round(lat, 6) * 1_000_000
    lon6 = np.round(lon, 6) * 1_000_000
    return (lat6.astype(np.int64) * 100_000_000) + lon6.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--outdir", default=str(OUT_DIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    print("Loading predictions, observed masks and county boundaries ...")
    preds = fm.load_predictions(args.predictions)
    obs = pc.load_pixel_detections()
    obs["_key"] = _pack(obs["lat"].to_numpy(), obs["lon"].to_numpy())
    truth = {week: np.asarray(sorted(set(g["_key"])))
             for week, g in obs.groupby("week")}

    TRAIN_ENDS = {1: 2008, 2: 2014, 3: 2020}  # expanding-window protocol

    pooled = Tally()
    county_tally: dict[str, Tally] = {}
    month_tally: dict[int, Tally] = {}
    area_rows: list[dict] = []

    for fold, train_end in TRAIN_ENDS.items():
        weeks_fold = pd.Series(sorted(preds.loc[preds["fold"] == fold, "week"].unique()))
        print(f"\nFold {fold}: train climatology to {train_end}, testing {len(weeks_fold)} weeks")
        cache_dir = pc.OUT_DIR / f"train_{train_end}"
        cl = pc.get_climatology(cache_dir=cache_dir, years=range(2000, train_end + 1))
        assets = fm.county_month_assets(cl)
        prev_arr: np.ndarray | None = None

        for i, week in enumerate(weeks_fold):
            frame = fm.compose_week(week, preds, cl, assets, horizon=0)
            keys = _pack(frame["lat"].to_numpy(), frame["lon"].to_numpy())
            week_arr = truth.get(week)
            y = np.isin(keys, week_arr.astype(np.int64)) if week_arr is not None else np.zeros(len(keys), bool)
            persist = np.isin(keys, prev_arr) if prev_arr is not None else np.zeros(len(keys), bool)
            pooled.add_week(frame["prob"].to_numpy(), frame["clim"].to_numpy(), persist.astype(float), y.astype(float))

            p_all = frame["prob"].to_numpy()
            c_all = frame["clim"].to_numpy()
            for county in frame["county"].drop_duplicates():
                sel = (frame["county"] == county).to_numpy()
                county_tally.setdefault(county, Tally()).add_week(
                    p_all[sel], c_all[sel], persist[sel].astype(float), y[sel].astype(float))
                month_tally.setdefault(week.month, Tally()).add_week(
                    p_all[sel], c_all[sel], persist[sel].astype(float), y[sel].astype(float))

            o_km2 = float((y * frame["pixel_area_km2"].to_numpy()).sum())
            area_rows.append({
                "week": week.date(),
                "obs_km2": o_km2,
                "prob_km2": float((frame["prob"] * frame["pixel_area_km2"]).sum()),
                "sev_km2": float((frame["sev"] * frame["pixel_area_km2"]).sum()),
            })
            prev_arr = truth.get(week)
            if (i + 1) % 30 == 0:
                print(f"    ... {i + 1}/{len(weeks_fold)} weeks", flush=True)

    # ---------------- pooled scores ----------------
    g = {
        "n": pooled.n, "n_wet": pooled.n_wet,
        "prev": pooled.n_wet / pooled.n,
        "brier_model": pooled.brier("prob"),
        "brier_clim": pooled.brier("clim"),
        "brier_persist": pooled.brier("persist"),
    }
    g["bss_vs_clim"] = 1.0 - g["brier_model"] / g["brier_clim"]
    g["bss_vs_persist"] = 1.0 - g["brier_model"] / g["brier_persist"]
    auroc, ap = pooled.roc_pr()

    area = pd.DataFrame(area_rows)
    area = area.assign(diff_prob=area["prob_km2"] - area["obs_km2"],
                       diff_sev=area["sev_km2"] - area["obs_km2"])
    flood = area[area["obs_km2"] > 0]
    r2_sev = 1 - ((area["sev_km2"] - area["obs_km2"]) ** 2).sum() / \
        ((area["obs_km2"] - area["obs_km2"].mean()) ** 2).sum()

    # reliability deciles
    base = N_BINS // 10
    rel = []
    for b in range(10):
        lo_, hi_ = b * base, (b + 1) * base
        pos_b = pooled.pos[lo_:hi_]
        neg_b = pooled.neg[lo_:hi_]
        tot = pos_b.sum() + neg_b.sum()
        if tot:
            mid = (np.arange(lo_, hi_) + 0.5) / N_BINS
            pmean = (mid * (pos_b + neg_b)).sum() / tot   # mean predicted p over ALL pixels in the bin
            freq = pos_b.sum() / tot                       # observed wet frequency in the bin
            rel.append((f"{b/10:.1f}-{(b+1)/10:.1f}", pmean, freq, int(tot)))
    rel_df = pd.DataFrame(rel, columns=["bin", "mean_p", "observed_freq", "n"])

    # ---------------- markdown report ----------------
    lines = [
        "# Pixel-level performance of the climatology-downscaled forecast maps",
        "",
        "**Model under test:** LightGBM county forecast x per-pixel climatology "
        "(`forecast_maps.prob` = `det_prob x freq_given`; severity layer "
        "`sev` = climatology pattern scaled to the model's q50 area).",
        "",
        "**Honest setup:** out-of-fold test weeks only (2010-11, 2016-17, "
        "2022-24); the pixel climatology prior is rebuilt per fold on that "
        "fold's **training years only** (2000-2008 / 2000-2014 / 2000-2020), so "
        "neither the model probability nor the spatial prior ever saw a test "
        "year. Weeks without a satellite observation score as dry (censoring, "
        "conservative).",
        "",
        f"Scored: **{g['n']:,} pixel-weeks**, wet prevalence "
        f"**{100*g['prev']:.2f}%** ({g['n_wet']:,} wet).",
        "",
        "## 1. Probabilistic pixel scores",
        "",
        "| score | value |",
        "|---|---|",
        f"| Brier, model map `prob` | {g['brier_model']:.4f} |",
        f"| Brier, pure climatology `clim` | {g['brier_clim']:.4f} |",
        f"| Brier, persistence (last week's mask) | {g['brier_persist']:.4f} |",
        f"| **BSS vs climatology** | **{g['bss_vs_clim']:+.3f}** |",
        f"| BSS vs persistence | {g['bss_vs_persist']:+.3f} |",
        f"| AUROC (model map) | {auroc:.4f} |",
        f"| PR-AUC (model map) | {ap:.4f} |",
        "",
        "Brier = mean((p - y)^2), lower is better. "
        "BSS = 1 - Brier_model / Brier_reference (>0 = skill over the reference).",
        "",
        "## 2. Threshold classification (model map, pooled)",
        "",
        "| threshold | accuracy | precision | recall | F1 | CSI | FAR | TNR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ti, t in enumerate(THRESHOLDS):
        tp, fp_, fn, tn = pooled.tp[ti], pooled.fp[ti], pooled.fn[ti], pooled.tn[ti]
        acc = (tp + tn) / g["n"]
        prec = tp / (tp + fp_) if tp + fp_ else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        csi = tp / (tp + fp_ + fn) if tp + fp_ + fn else 0.0
        far = fp_ / (tp + fp_) if tp + fp_ else 0.0
        tnr = tn / (tn + fp_) if tn + fp_ else 0.0
        lines.append(
            f"| {t:.2f} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | "
            f"{csi:.4f} | {far:.4f} | {tnr:.4f} |"
        )

    lines += [
        "",
        "## 3. Weekly flooded area (km2, region-total per week)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| MAE map-implied area (Σ p x km2) vs observed, all {len(area)} weeks | {area['diff_prob'].abs().mean():.2f} km2 |",
        f"| MAE (probability map) on flood weeks only (obs > 0, n={len(flood)}) | {flood['diff_prob'].abs().mean():.2f} km2 |",
        f"| MAE severity layer (Σ sev x km2 ~ q50) on flood weeks | {flood['diff_sev'].abs().mean():.2f} km2 |",
        f"| bias (map-implied − observed), all weeks | {area['diff_prob'].mean():+.2f} km2 |",
        f"| Pearson r, weekly implied vs observed area | {area['prob_km2'].corr(area['obs_km2']):.3f} |",
        f"| R2 severity-layer weekly area | {r2_sev:.3f} |",
        "",
        "## 4. Per-county breakdown (pooled over test weeks)",
        "",
        "| county | Brier model | Brier climatology | BSS vs clim | wet pixel-weeks |",
        "|---|---|---|---|---|",
    ]
    for county in sorted(county_tally):
        ct = county_tally[county]
        bm, bc = ct.brier("prob"), ct.brier("clim")
        bss = 1 - bm / bc if bc > 0 else float("nan")
        lines.append(f"| {county} | {bm:.4f} | {bc:.4f} | {bss:+.3f} | {int(ct.n_wet):,} |")

    lines += [
        "",
        "## 5. Monthly breakdown (pooled over test years)",
        "",
        "| month | Brier model | Brier climatology | BSS vs clim |",
        "|---|---|---|---|",
    ]
    for m in sorted(month_tally):
        mt_ = month_tally[m]
        bm, bc = mt_.brier("prob"), mt_.brier("clim")
        bss = 1 - bm / bc if bc > 0 else float("nan")
        lines.append(f"| {MONTH_NAMES[m-1]} | {bm:.4f} | {bc:.4f} | {bss:+.3f} |")

    lines += [
        "",
        "## 6. Reliability (calibration of the model `prob`, pooled)",
        "",
        "| predicted p bin | mean predicted | observed wet freq | n |",
        "|---|---|---|---|",
    ]
    for row in rel_df.itertuples(index=False):
        lines.append(f"| {row.bin} | {row.mean_p:.3f} | {row.observed_freq:.3f} | {row.n:,} |")

    lines += [
        "",
        "## Notes & limitations",
        "",
        "* The labels are satellite detections; a week without a valid observation",
        "  is zero-filled (censored), which biases scores conservatively — some",
        "  'dry' pixels were unobserved, not truly dry.",
        "* The `clim` reference uses the same (k+0.5)/(n+1) formula as the LightGBM",
        "  protocol, rebuilt per fold on training years. The shipped gallery PNGs",
        "  use the full 2000-2025 prior (explanatory maps), so this table is the",
        "  honest out-of-sample evaluation.",
        "* Threshold metrics depend on the cutoff; the headline ranking is the",
        "  Brier family. Accuracy is dominated by the ~98% dry pixels.",
        f"* Regenerate with: `python -m climatology_maps.evaluate_pixel_maps` "
        f"-> {outdir / REPORT_NAME}",
    ]

    report = outdir / REPORT_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")

    # console summary
    print("\n" + "=" * 80)
    print("POOLED PIXEL-LEVEL PERFORMANCE (OOF test weeks, per-fold train climatology)")
    print("=" * 80)
    print(f"  pixel-weeks scored : {g['n']:,}   wet prevalence: {100*g['prev']:.2f}%")
    print(f"  Brier model/clim/persist: {g['brier_model']:.4f} / {g['brier_clim']:.4f} / {g['brier_persist']:.4f}")
    print(f"  BSS vs climatology: {g['bss_vs_clim']:+.3f}   |   vs persistence: {g['bss_vs_persist']:+.3f}")
    print(f"  AUROC: {auroc:.4f}   PR-AUC: {ap:.4f}")
    print(f"  weekly-area MAE (all/flood weeks): {area['diff_prob'].abs().mean():.2f} / {flood['diff_prob'].abs().mean():.2f} km2")
    print("=" * 80)
    print(f"[report] {report}")


if __name__ == "__main__":
    main()