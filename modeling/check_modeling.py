"""Known-answer checks for the ``modeling`` package — no raw data required.

Every rule the rest of the package relies on is pinned here with a hand-
computed expected value, so a refactor that changes embargo semantics,
split boundaries, CRPS arithmetic or advisory tiers fails loudly:

    python -m modeling.check_modeling

Exit code 0 = all checks pass; 1 = at least one failure. The heavy data
paths (audit, features, training, U-Net training) need the raw dataset;
they are import-checked only, plus the pure functions around them are
exercised here with synthetic inputs.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd

from modeling import (
    advisory,
    audit,
    baselines,
    config,
    features,
    metrics,
    splits,
    unet,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(b))


# --- splits -------------------------------------------------------------------


def test_splits() -> None:
    # Embargo: target week Monday 2024-10-07 -> features dated <= Friday 2024-10-04.
    check(
        "cutoff(2024-10-07) == 2024-10-04",
        splits.feature_cutoff(pd.Timestamp("2024-10-07")) == pd.Timestamp("2024-10-04"),
    )
    # Split boundaries (2014-12-29 is a Monday, 2020-01-06 a Monday).
    expect = {
        "2014-12-29": "train",
        "2015-01-05": "val",
        "2019-12-30": "val",
        "2020-01-06": "test",
        "2025-12-29": "test",
    }
    weeks = pd.DatetimeIndex([pd.Timestamp(w) for w in expect])
    got = splits.assign_split(weeks)
    check(
        "assign_split boundaries",
        all(got[i] == v for i, v in expect.items()),
        str(got.tolist()),
    )
    # Boundary purge: the first week of val/test maps to None.
    wmap = splits.week_split_map(
        pd.DatetimeIndex(
            [
                pd.Timestamp("2014-12-29"),
                pd.Timestamp("2015-01-05"),
                pd.Timestamp("2015-01-12"),
                pd.Timestamp("2020-01-06"),
                pd.Timestamp("2020-01-13"),
            ]
        )
    )
    check(
        "purge drops first val week (2015-01-05) -> None",
        wmap.iloc[1] is None,
        repr(wmap.iloc[1]),
    )
    check(
        "purge drops first test week (2020-01-06) -> None",
        wmap.iloc[3] is None,
        repr(wmap.iloc[3]),
    )
    check("non-boundary weeks keep their split", wmap.iloc[2] == "val" and wmap.iloc[4] == "test")
    # Weekly calendar.
    widx = splits.weekly_index()
    check(
        "weekly_index: first 2000-01-03, last 2025-12-29, 1357 Mondays",
        widx[0] == pd.Timestamp("2000-01-03")
        and widx[-1] == pd.Timestamp("2025-12-29")
        and len(widx) == 1357
        and (widx.dayofweek == 0).all(),
        f"first={widx[0]} last={widx[-1]} n={len(widx)}",
    )


# --- metrics --------------------------------------------------------------------


def test_metrics() -> None:
    # CRPS known answers for the 3-quantile CDF q=(1,2,3).
    expect = {
        0.0: 1 + (0.9**3 - 0.5**3) / 1.2 + (0.5**3 - 0.1**3) / 1.2,  # 1.6066667
        1.0: (0.9**3 - 0.5**3) / 1.2 + (0.5**3 - 0.1**3) / 1.2,  # 0.6066667
        2.0: 31.0 / 150.0,  # 0.2066667
        3.0: (0.5**3 - 0.1**3) / 1.2 + (0.9**3 - 0.5**3) / 1.2,  # 0.6066667
        5.0: (0.5**3 - 0.1**3) / 1.2 + (0.9**3 - 0.5**3) / 1.2 + 2.0,  # 2.6066667
    }
    for y, val in expect.items():
        got = metrics.crps_quantiles(y, 1.0, 2.0, 3.0)
        check(f"crps(y={y:g}, q=(1,2,3)) == {val:.7f}", close(got, val, 1e-9), f"got {got}")
    # No-NaN guarantee outside the quantile range (the bug this pins down).
    check(
        "crps finite for y < q10 and y > q90",
        np.isfinite(metrics.crps_quantiles(0.0, 1.0, 2.0, 3.0))
        and np.isfinite(metrics.crps_quantiles(9.0, 1.0, 2.0, 3.0)),
    )
    # Detection: 1 hit, 1 false alarm, 1 miss.
    pod, far, csi = metrics.detection_scores([1, 0, 1], [0.9, 0.8, 0.1], threshold=0.5)
    check(
        "detection (1 hit, 1 FA, 1 miss) = (0.5, 0.5, 1/3)",
        close(pod, 0.5) and close(far, 0.5) and close(csi, 1 / 3),
    )
    # summarise_forecast end-to-end on 2 hand-computed rows.
    df = pd.DataFrame(
        {
            "split": ["test", "test"],
            "y_true": [0.0, 10.0],
            "q10": [0.0, 0.0],
            "q50": [0.0, 0.0],
            "q90": [0.0, 2.0],
            "det_prob": [0.1, 0.9],
            "y_pred_baseline": [5.0, 5.0],
            "spike_threshold": [100.0, 5.0],
        }
    )
    table = metrics.summarise_forecast(df)
    all_row = table[table["slice"] == "all"].iloc[0]
    pos_row = table[(table["slice"] == "positive") & (table["split"] == "test")].iloc[0]
    spike_row = table[table["slice"] == "spike"].iloc[0]
    check(
        "skill_q50 == 1 - sqrt(50)/5 on 'all'",
        close(all_row["skill_q50"], 1.0 - 50.0**0.5 / 5.0, 1e-9),
        f"got {all_row['skill_q50']}",
    )
    check(
        "positive slice: csi 1.0, pod 1.0, far 0.0",
        close(pos_row["csi"], 1.0) and close(pos_row["pod"], 1.0) and close(pos_row["far"], 0.0),
    )
    check("spike slice keeps only the spiky row", int(spike_row["n"]) == 1)
    # y=10, q=(0,0,2): I1 = [0,2): ramp 0.5->0.9 -> (0.9^3-0.5^3)/0.6 = 1.0066667,
    # [2,10): 1 -> 8 ; I2 = 0  => 9.0066667
    check(
        "crps on positive row == 9.0066667 (y=10, q=(0,0,2))",
        close(pos_row["crps"], 9.0066667, 1e-6),
        f"got {pos_row['crps']}",
    )


# --- baselines --------------------------------------------------------------------


def _weekly(county: str, pairs: list[tuple[str, float]]) -> pd.DataFrame:
    rows = [{"county": county, "week": pd.Timestamp(w), "y_true": a, "split": "val"} for w, a in pairs]
    df = pd.DataFrame(rows).sort_values(["county", "week"])
    df["month"] = df["week"].dt.month
    df["year"] = df["week"].dt.year
    return df


def test_baselines() -> None:
    # Persistence: value of the previous week, 0 at the first week.
    df = _weekly("A", [("2014-12-29", 10.0), ("2015-01-05", 20.0), ("2015-01-12", 0.0), ("2015-01-19", 30.0)])
    df["split"] = ["train", "val", "val", "val"]
    got = baselines.persistence_baseline(df).tolist()
    check(
        "persistence = [0, 10, 20, 0]",
        got == [0.0, 10.0, 20.0, 0.0],
        str(got),
    )
    # Climatology: train-mean per county-month; missing county -> 0.
    df = _weekly("A", [("2014-01-06", 5.0), ("2014-02-03", 15.0), ("2015-01-05", 1.0), ("2015-02-02", 1.0)])
    df = pd.concat(
        [_weekly("B", [("2015-01-05", 1.0), ("2015-02-02", 1.0)]), df], ignore_index=True
    )
    df["split"] = ["val", "val", "train", "train", "val", "val"]
    df = df.sort_values(["county", "week"]).reset_index(drop=True)
    out = baselines.climatology_baseline(df, df["split"] == "train")
    a_jan = out.iloc[0]  # rows keep feature-frame order: A x4, then B x2
    a_feb = out.iloc[1]
    b_jan = out.iloc[4]
    check("climatology A jan = train mean 5.0", close(a_jan, 5.0), f"got {a_jan}")
    check("climatology A feb = train mean 15.0", close(a_feb, 15.0), f"got {a_feb}")
    check("climatology B jan = 0 (county never trained)", close(b_jan, 0.0), f"got {b_jan}")
    # Last detection: most recent non-zero within 4 weeks.
    df = _weekly("A", [("2014-12-29", 10.0), ("2015-01-05", 0.0), ("2015-01-12", 0.0), ("2015-01-19", 0.0)])
    df["split"] = ["train", "val", "val", "val"]
    got = baselines.last_detection_baseline(df).tolist()
    check("last-detection = [0, 10, 10, 10]", got == [0.0, 10.0, 10.0, 10.0], str(got))


# --- advisory -------------------------------------------------------------------


def test_advisory() -> None:
    stats = pd.Series({"clim_mean": 10.0, "clim_p80": 20.0, "clim_p95": 30.0})
    tiers = [
        (0.0, 0),
        (10.0, 0),
        (10.1, 1),
        (20.0, 1),
        (20.1, 2),
        (30.0, 2),
        (30.1, 3),
    ]
    for area, want in tiers:
        got = advisory.magnitude_tier(area, stats)
        check(f"tier(area={area}) == {want}", got == want, f"got {got}")
    check("month_phase(10) == harvest", advisory.month_phase(10) == "harvest")
    severe = advisory.advisory_text("Aweil East", 10, 500.0, 0.9, 3, 1200.0, 800.0, 12000)
    check(
        "tier-3 text says SEVERE + exposure + cattle",
        "SEVERE" in severe and "1,200 ha" in severe and "12,000 cattle" in severe,
        severe,
    )
    quiet = advisory.advisory_text("Aweil West", 3, 2.0, 0.2, 0, 0.0, 0.0)
    check("tier-0 quiet text", "No flood expected" in quiet, quiet)
    # Exposure scaling with cap.
    hist = pd.DataFrame(
        {
            "county": ["A", "A"],
            "month": [10, 11],
            "hist_crop_exposed_ha": [100.0, 50.0],
            "hist_rangeland_exposed_ha": [50.0, 25.0],
        }
    )
    crop, rng = advisory.exposure_for_week(20.0, "A", 10, hist, clim_mean=10.0)
    check("exposure scales 2x (area 20 vs mean 10)", close(crop, 200.0) and close(rng, 100.0))
    crop, rng = advisory.exposure_for_week(100.0, "A", 10, hist, clim_mean=10.0)
    check("exposure capped at 5x", close(crop, 500.0) and close(rng, 250.0))
    crop, rng = advisory.exposure_for_week(20.0, "A", 10, hist, clim_mean=0.0)
    check("no historical flood -> 0 exposure", crop == 0.0 and rng == 0.0)
    crop, rng = advisory.exposure_for_week(20.0, "B", 10, hist, clim_mean=10.0)
    check("unknown county -> 0 exposure", crop == 0.0 and rng == 0.0)


# --- features (pure helpers) ------------------------------------------------------


def test_features_helpers() -> None:
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    vals = pd.Series(np.arange(30.0), index=idx)
    daily = pd.DataFrame(
        {
            "local_tp": vals,
            "local_ro": vals,
            "up_tp": vals,
            "up_ro": vals,
            "gauge": vals,
            "albert": vals,
            "et0": vals,
        }
    )
    out = features._rolling_features(daily)
    check(
        "local_tp_w3 at day 2 = 3 (0+1+2)",
        close(out.loc[idx[2], "local_tp_w3"], 3.0),
        f"got {out.loc[idx[2], 'local_tp_w3']}",
    )
    check(
        "local_tp_w3 at day 4 = 9 (2+3+4)",
        close(out.loc[idx[4], "local_tp_w3"], 9.0),
        f"got {out.loc[idx[4], 'local_tp_w3']}",
    )
    check(
        "local_tp_w7 at day 6 = 21",
        close(out.loc[idx[6], "local_tp_w7"], 21.0),
        f"got {out.loc[idx[6], 'local_tp_w7']}",
    )
    check(
        "w30 is NaN at day 19 (min_periods=30)",
        np.isnan(out.loc[idx[19], "local_tp_w30"]),
    )
    check(
        "w30 at day 29 = sum of 0..29 = 435",
        close(out.loc[idx[29], "local_tp_w30"], 435.0),
        f"got {out.loc[idx[29], 'local_tp_w30']}",
    )
    check(
        "gauge_chg7 at day 7 = 7.0",
        close(out.loc[idx[7], "gauge_chg7"], 7.0),
        f"got {out.loc[idx[7], 'gauge_chg7']}",
    )
    check("gauge_chg7 is NaN before day 7", np.isnan(out.loc[idx[3], "gauge_chg7"]))
    check(
        "net_w7 at day 6 = tp_w7 (sum 21) - et0_w7 (mean 3) = 18",
        close(out.loc[idx[6], "net_w7"], 18.0),
        f"got {out.loc[idx[6], 'net_w7']}",
    )


# --- audit (pure helpers) ---------------------------------------------------------


def test_audit_helpers() -> None:
    days = pd.date_range("2020-01-01", periods=400, freq="D")
    pixels = pd.Series(0.0, index=days)
    june = np.where(days.month == 6)[0]
    values = [0.0, 5.0, 0.0, 3.0, 9.0, 1.0, 0.0, 2.0]  # Jun 1-8 (Jun 1 is a Monday)
    pixels.iloc[june[: len(values)]] = values
    cov = audit._coverage(pixels)
    check("coverage: 2020 active_days = 5", int(cov.loc[2020, "active_days"]) == 5, str(cov.to_dict()))
    check("coverage: 2020 max_day_pixels = 9", int(cov.loc[2020, "max_day_pixels"]) == 9)
    check(
        "coverage: 2020 weeks_with_detection = 2 (Jun 1-7 and Jun 8-14)",
        int(cov.loc[2020, "weeks_with_detection"]) == 2,
        f"got {int(cov.loc[2020, 'weeks_with_detection'])}",
    )


# --- unet (pure helpers) ------------------------------------------------------------


def test_unet_helpers() -> None:
    grid = unet.tile_grid()
    check("tile_grid is 40x40", len(grid) == 40)
    check(
        "tile_grid NW corner = (9.875 N, 20.125 E)",
        close(grid["lat"].iloc[0], 9.875) and close(grid["lon"].iloc[0], 20.125),
    )
    check(
        "tile_grid SE corner = (0.125 N, 29.875 E)",
        close(grid["lat"].iloc[-1], 0.125) and close(grid["lon"].iloc[-1], 29.875),
    )
    check("_pixel_to_cell(NW) = (0,0)", unet._pixel_to_cell(9.875, 20.125) == (0, 0))
    check("_pixel_to_cell(SE) = (39,39)", unet._pixel_to_cell(0.125, 29.875) == (39, 39))
    check("_pixel_to_cell(9.87, 20.13) = (0,0)", unet._pixel_to_cell(9.87, 20.13) == (0, 0))
    # CSI hand value: 1 hit, 1 miss, 1 false alarm on a 2x2 grid.
    p = np.array([[1, 1], [0, 0]], dtype=float)
    t = np.array([[1, 0], [0, 1]], dtype=float)
    check("csi (1 hit, 1 miss, 1 FA) == 1/3", close(unet.csi(p, t), 1 / 3), f"got {unet.csi(p, t)}")
    # Null baseline: threshold calibrated to the model's detection rate.
    ytr = np.zeros((1, 2, 2))
    ytr[0, 0, 0] = 1.0  # freq [[0.5, 0], [0, 0]]
    p_val = np.array([[0.9, 0.1], [0.2, 0.8]])  # 2 of 4 positive -> rate 0.5
    freq, thr = unet.null_baseline(ytr, p_val)
    check(
        "null threshold = 50th pct of [0.5,0,0,0] = 0.0",
        close(thr, 0.0),
        f"got {thr}",
    )
    check("null map marks only the flooded pixel", bool(freq[0, 0] > thr) and not bool((freq > thr).sum() > 1))
    # Shape + range test (needs torch; skipped in the CPU-only data room).
    try:
        import torch
    except ImportError:
        print("       (U-Net forward test skipped: torch not installed)")
        return
    model = unet.UNet(config.UNET_IN_CHANNELS)
    x = torch.randn(4, config.UNET_IN_CHANNELS, 40, 40)
    with torch.no_grad():
        out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    check(
        "UNet forward: (4,17,40,40) -> (4,1,40,40) in [0,1]",
        out.shape == (4, 1, 40, 40) and bool((out >= 0).all() and (out <= 1).all()),
        str(tuple(out.shape)),
    )
    check("UNet parameter count < 3M (fits a 3090 comfortably)", n_params < 3_000_000, f"{n_params}")


# --- family policy checks -----------------------------------------------------


def test_families() -> None:
    from modeling.train_task_a import FAMILY_PRIORITY, PRIMARY_FAMILY, resolve_families

    check(
        PRIMARY_FAMILY == "lgbm" and FAMILY_PRIORITY[0] == "lgbm",
        True,
        "LightGBM must be the primary backbone (first in priority order)",
    )
    check(resolve_families(None) == ["lgbm"], True, "default trains the primary only (LightGBM first)")
    check(resolve_families("") == ["lgbm"], True, "empty spec -> primary only")
    check(
        resolve_families("cat,lgbm,xgb") == ["lgbm", "xgb", "cat"],
        True,
        "spec is normalised to priority order",
    )
    check(resolve_families("xgb") == ["xgb"], True, "cross-check-only run allowed")
    try:
        resolve_families("lgbm,foo")
        check(False, "unknown family must raise")
    except ValueError:
        check(True, "unknown family raises ValueError")


# --- import checks ----------------------------------------------------------------


def test_imports() -> None:
    for mod in (
        "modeling.audit",
        "modeling.features",
        "modeling.data_check",
        "modeling.unet",
        "modeling.advisory",
        "modeling.train_task_a",
        "modeling.train_tabpfn",
        "modeling.shap_report",
        "modeling.baselines",
        "modeling.metrics",
        "modeling.splits",
        "modeling.config",
    ):
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"import {mod}", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("== modeling known-answer checks (no raw data needed) ==\n")
    print("-- splits --")
    test_splits()
    print("\n-- metrics --")
    test_metrics()
    print("\n-- baselines --")
    test_baselines()
    print("\n-- advisory --")
    test_advisory()
    print("\n-- features --")
    test_features_helpers()
    print("\n-- audit --")
    test_audit_helpers()
    print("\n-- unet --")
    test_unet_helpers()
    print("\n-- families --")
    test_families()
    print("\n-- imports --")
    test_imports()
    print(f"\n{'=' * 60}\n{len(FAILURES)} failure(s)")
    if FAILURES:
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
