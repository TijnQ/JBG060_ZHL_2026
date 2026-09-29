"""Task B: sub-county flood probability maps for the Aweil floodplain (E6).

A small U-Net predicts, for every 0.25-degree grid cell of HEC/USGS tile
h20v08 (40x40 pixels), the probability that the pixel is flooded during the
target week. It is the "where" complement to Task A's county-level "how
much": the 2024 report case study needs *which parts* of Aweil East/North/
West were at risk, not just the county total.

Protocol (same anti-leakage rules as Task A, applied to the pixel grid):
- target week starts Monday ``m``; label = union of all 3-day composites
  whose *date* falls in the week ``m..m+6`` (the EDA's W-SUN weekly-union
  convention; the newest composite dated ``m+6`` covers at most ``m+4..m+6``,
  still inside the week);
- features = last 7 days (``m-9..m-3``) of ERA5 rainfall and runoff on the
  tile, plus 3 static channels (crop / rangeland fraction, cattle density);
  the newest feature (``m-3``) predates the first composite day (``m-2``),
  so the 3-day embargo holds exactly;
- train 2000-2014 / val 2015-2019 / test 2020-2025 by week;
- comparison against a *null* baseline: each pixel's historical flood
  frequency (a climatological "where" map), threshold-calibrated to the
  model's detection rate so CSI comparison is fair.

Outputs (``outputs/methods/unet/``):
    figures/task_b_unet_maps.png      mean val prediction vs null, one example
    tables/task_b_unet_metrics.csv    CSI/POD/FAR, model vs null, per split

Usage:
    python -m modeling.unet.train --epochs 30

Read ``guide.md`` in this folder for what to test and the keep/kill rules.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from modeling import config
from modeling.splits import assign_split, weekly_index

__all__ = [
    "UNet",
    "build_tile_samples",
    "csi",
    "main",
    "null_baseline",
    "raster_to_grid",
    "tile_grid",
    "train_unet",
]


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "torch is required for Task B — install from requirements-ml.txt"
        ) from exc


def tile_grid() -> pd.DataFrame:
    """40x40 pixel-centre grid (north to south, west to east)."""
    half = config.ERA5_CELL_DEG / 2
    lats = (
        config.TILE_BBOX["lat_max"] - half - np.arange(config.TILE_GRID) * config.ERA5_CELL_DEG
    )
    lons = config.TILE_BBOX["lon_min"] + half + np.arange(config.TILE_GRID) * config.ERA5_CELL_DEG
    return pd.DataFrame({"lat": lats, "lon": lons})


def raster_to_grid(raster_path) -> np.ndarray:
    """Zonal mean of a raster band onto the tile grid (origin/transform agnostic)."""
    import rasterio

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        transform = src.transform
    xs = transform.c + (np.arange(src.width) + 0.5) * transform.a
    ys = transform.f + (np.arange(src.height) + 0.5) * transform.e
    xx, yy = np.meshgrid(xs, ys)
    col = np.floor((xx - config.TILE_BBOX["lon_min"]) / config.ERA5_CELL_DEG).astype(int)
    row = np.floor(
        (config.TILE_BBOX["lat_max"] - yy) / config.ERA5_CELL_DEG
    ).astype(int)
    valid = (
        np.isfinite(data)
        & (row >= 0)
        & (row < config.TILE_GRID)
        & (col >= 0)
        & (col < config.TILE_GRID)
    )
    flat = (row[valid] * config.TILE_GRID + col[valid]).astype(int)
    weights = np.nan_to_num(data[valid], nan=0.0)
    sums = np.bincount(flat, weights=weights, minlength=config.TILE_GRID**2)
    counts = np.bincount(flat, minlength=config.TILE_GRID**2)
    out = np.zeros(config.TILE_GRID**2)
    out[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return out.reshape(config.TILE_GRID, config.TILE_GRID)


def _pixel_to_cell(lat: float, lon: float) -> tuple[int, int]:
    half = config.ERA5_CELL_DEG / 2
    row = round((config.TILE_BBOX["lat_max"] - half - lat) / config.ERA5_CELL_DEG)
    col = round((lon - config.TILE_BBOX["lon_min"] - half) / config.ERA5_CELL_DEG)
    if not (0 <= row < config.TILE_GRID and 0 <= col < config.TILE_GRID):
        raise ValueError(f"pixel ({lat}, {lon}) outside tile grid")
    return row, col


def _era5_tile_daily(year: int) -> tuple[np.ndarray, np.ndarray]:
    """(tp, ro) in mm/day, each (365, 40, 40), north to south."""
    from processing_data import loading

    ds = loading.load_rainfall_runoff(np.array([year]))
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_idx = np.where((lat >= config.TILE_BBOX["lat_min"] + 0.1) & (lat <= config.TILE_BBOX["lat_max"] - 0.1))[0]
    lon_idx = np.where((lon >= config.TILE_BBOX["lon_min"] + 0.1) & (lon <= config.TILE_BBOX["lon_max"] - 0.1))[0]
    if len(lat_idx) != config.TILE_GRID or len(lon_idx) != config.TILE_GRID:
        raise ValueError(
            f"tile extraction for {year} gave {len(lat_idx)}x{len(lon_idx)}, "
            f"expected {config.TILE_GRID}x{config.TILE_GRID}"
        )
    tp = ds["tp"].values[:, lat_idx, lon_idx].astype(np.float32) * 1000.0
    ro = ds["ro"].values[:, lat_idx, lon_idx].astype(np.float32) * 1000.0
    return tp, ro


def _flood_parquet_pixels(year: int) -> list[tuple[pd.Timestamp, float, float]]:
    """(composite date, lat, lon) detections for the tile, both compact types."""
    frames = []
    for kind in ("compact_recurring", "compact_unusual"):
        path = config.FLOOD_ROOT / kind / f"flood_events_{config.TILE}_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path} — run `python -m modeling.data_check`"
            )
        df = pd.read_parquet(path, columns=["date", "lat", "lon"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return list(zip(out["date"], out["lat"], out["lon"]))


def build_tile_samples() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Assemble (X, y, weeks): X (n_weeks, 17, 40, 40) f32, y (n_weeks, 40, 40) f32.

    X channels: 14 daily (tp, ro for m-9..m-3) + 3 static.
    y: union of all 3-day composites whose date falls in the week (W-SUN).
    """
    weeks = weekly_index()
    week_index = {w: i for i, w in enumerate(weeks)}

    # ---- labels -------------------------------------------------------------
    labels: dict[int, np.ndarray] = {}
    print("loading flood parquets (labels)...")
    for year in config.YEARS:
        for d, lat, lon in _flood_parquet_pixels(year):
            week = d - pd.Timedelta(days=d.weekday())
            i = week_index.get(week)
            if i is None:
                continue  # week outside the 2000-01-03..2025-12-29 calendar
            mask = labels.setdefault(i, np.zeros((config.TILE_GRID, config.TILE_GRID), np.int8))
            mask[_pixel_to_cell(lat, lon)] = 1

    # ---- features ------------------------------------------------------------
    print("loading ERA5 tile daily fields (features)...")
    tp_days: dict[pd.Timestamp, np.ndarray] = {}
    ro_days: dict[pd.Timestamp, np.ndarray] = {}
    for year in config.YEARS:
        tp, ro = _era5_tile_daily(year)
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        for k in range(len(days)):
            tp_days[days[k]] = tp[k]
            ro_days[days[k]] = ro[k]

    print("assembling weekly tensors (embargo: features <= m-3)...")
    X = np.zeros((len(weeks), config.UNET_IN_CHANNELS, config.TILE_GRID, config.TILE_GRID), np.float32)
    y = np.zeros((len(weeks), config.TILE_GRID, config.TILE_GRID), np.float32)
    static = _static_channels()
    for i, m in enumerate(weeks):
        for k, day in enumerate(pd.date_range(m - pd.Timedelta(days=9), periods=7, freq="D")):
            tp_d, ro_d = tp_days.get(day), ro_days.get(day)
            if tp_d is None:  # pragma: no cover - only possible at the 2000 edge
                continue
            X[i, 2 * k] = tp_d
            X[i, 2 * k + 1] = ro_d
        X[i, config.UNET_DAILY_CHANNELS * config.UNET_HISTORY_DAYS :] = static
        mask = labels.get(i)
        if mask is not None:
            y[i] = mask
    return X, y, weeks


def _static_channels() -> np.ndarray:
    """3 static channels: crop fraction, rangeland fraction, log10(cattle+1)."""
    crops = raster_to_grid(config.FARMLAND_CROPS)
    rangeland = raster_to_grid(config.FARMLAND_RANGELAND)
    cattle = raster_to_grid(config.FARMLAND_CATTLE)
    return np.stack([crops, rangeland, np.log10(cattle + 1.0)]).astype(np.float32)


_UNET_CLS: type | None = None


def UNet(in_channels: int = config.UNET_IN_CHANNELS):
    """4-block UNet: 40x40 -> 20 -> 10 -> 5 -> 10 -> 20 -> 40 -> 1 channel.

    Built lazily so the module imports without torch installed.
    """
    global _UNET_CLS
    if _UNET_CLS is None:
        torch, nn, F = _torch_modules()

        def double_conv(cin: int, cout: int):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        class _UNet(nn.Module):
            def __init__(self, in_channels: int):
                super().__init__()
                self.down1 = double_conv(in_channels, 16)
                self.down2 = double_conv(16, 32)
                self.down3 = double_conv(32, 64)
                self.down4 = double_conv(64, 128)
                self.pool = nn.MaxPool2d(2)
                self.up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
                self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
                self.up2 = nn.ConvTranspose2d(16, 16, 2, stride=2)
                self.head = nn.Conv2d(16, 1, 1)
                for p in self.parameters():
                    if p.dim() > 1:
                        torch.nn.init.kaiming_normal_(p, nonlinearity="relu")

            def forward(self, x):
                d1 = self.down1(x)
                p1 = self.pool(d1)
                d2 = self.down2(p1)
                p2 = self.pool(d2)
                d3 = self.down3(p2)
                p3 = self.pool(d3)
                d4 = self.down4(p3)
                u4 = self.up4(d4) + p2
                u3 = self.up3(u4) + p1
                u2 = self.up2(u3) + d1
                return F.sigmoid(self.head(u2))

        _UNET_CLS = _UNet
    return _UNET_CLS(in_channels)


def _torch_modules():
    import torch
    import torch.nn.functional as F
    from torch import nn

    return torch, nn, F


def _to_tensor(arr, torch):
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def train_unet(
    Xtr, ytr, Xval, yval, epochs: int = 30, batch: int = 32,
    lr: float = 1e-3, device: str = "auto",
) -> dict:
    """Train the U-Net with BCE (positivity-weighted) + cosine schedule."""
    torch, nn, _ = _torch_modules()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    print(f"training on {device} (cuda available: {torch.cuda.is_available()})")

    torch.manual_seed(config.SEED)
    model = UNet(config.UNET_IN_CHANNELS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    pos = ytr[ytr > 0].size
    neg = ytr.size - pos
    weight = float(np.sqrt(neg / max(1, pos)))
    loss_fn = nn.BCELoss(pos_weight=torch.tensor(weight, device=device))

    use_amp = device == "cuda"
    Xtr_t, ytr_t = _to_tensor(Xtr, torch).to(device), _to_tensor(ytr, torch).to(device)
    Xval_t = _to_tensor(Xval, torch).to(device)

    best_csi, best_state, patience_left = -1.0, None, 5
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        total, seen = 0.0, 0
        for s in range(0, len(perm), batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        sched.step()
        # validation CSI (detection, threshold 0.5)
        model.eval()
        with torch.no_grad():
            pval = (model(Xval_t) > 0.5).float().cpu().numpy()
        csi_val = csi(pval, yval)
        history.append({"epoch": epoch, "loss": total / seen, "val_csi": csi_val})
        if csi_val > best_csi:
            best_csi, patience_left = csi_val, 5
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        if epoch % 5 == 0 or patience_left == 0:
            print(f"  epoch {epoch}: loss={total / seen:.4f} val_csi={csi_val:.3f}")
        if patience_left == 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": pd.DataFrame(history), "best_csi": best_csi,
            "device": device}


def csi(p: np.ndarray, t: np.ndarray) -> float:
    """Critical Success Index on a 0/1 array."""
    hits = float(np.sum((p > 0.5) & (t > 0.5)))
    miss = float(np.sum((p <= 0.5) & (t > 0.5)))
    false_alarm = float(np.sum((p > 0.5) & (t <= 0.5)))
    return hits / max(1e-12, hits + miss + false_alarm)


def null_baseline(ytr: np.ndarray, p_val: np.ndarray) -> tuple[np.ndarray, float]:
    """Historical-frequency map, threshold-calibrated to the model's detection rate."""
    freq = ytr.mean(axis=0)
    model_rate = float((p_val > 0.5).mean())
    threshold = np.quantile(freq, max(0.0, min(1.0, 1.0 - model_rate)))
    return freq, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    X, y, weeks = build_tile_samples()
    split = assign_split(weeks)
    masks = {k: (split.to_numpy() == k) for k in ("train", "val", "test")}

    out = train_unet(X[masks["train"]], y[masks["train"]],
                     X[masks["val"]], y[masks["val"]],
                     epochs=args.epochs, device=args.device)
    model, torch = out["model"], _torch()

    rows = []
    for name in ("val", "test"):
        model.eval()
        with torch.no_grad():
            p = model(_to_tensor(X[masks[name]], torch).to(out["device"])).cpu().numpy()
        freq, thr = null_baseline(y[masks["train"]], p)
        p_null = (freq > thr)
        for source, prob in (("model", p), ("null", p_null)):
            rows.append(
                {
                    "split": name,
                    "baseline": source,
                    "csi": csi(prob > 0.5, y[masks[name]]),
                    "pod": float(((prob > 0.5) & (y[masks[name]] > 0.5)).sum()
                                 / max(1, (y[masks[name]] > 0.5).sum())),
                    "far": float(((prob > 0.5) & (y[masks[name]] <= 0.5)).sum()
                                 / max(1, (prob > 0.5).sum())),
                }
            )
    metrics_df = pd.DataFrame(rows)
    dirs = config.method_dirs("unet")
    dirs["tables"].mkdir(parents=True, exist_ok=True)
    metrics_path = dirs["tables"] / "task_b_unet_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_df.to_string(index=False))
    print(f"wrote {metrics_path}")

    dirs["figures"].mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        p_val = model(_to_tensor(X[masks["val"]], torch).to(out["device"])).cpu().numpy()
    freq, thr = null_baseline(y[masks["train"]], p_val)
    mean_model = p_val.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(mean_model, origin="upper", cmap="Blues")
    axes[0].set_title("mean val probability (U-Net)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(freq > thr, origin="upper", cmap="Blues")
    axes[1].set_title("null: historical flood frequency (calibrated)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig_path = dirs["figures"] / "task_b_unet_maps.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
