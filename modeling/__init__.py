"""Forecasting and impact-advisory models for the South Sudan flood project.

Three tasks, mirroring MODEL_RESEARCH.md:

- Task A: per-county weekly detected flood area (km2) from tabular features,
  probabilistic (quantiles) — ``features``, ``baselines``, ``train_task_a``,
  ``train_tabpfn``, ``shap_report``.
- Task B: sub-county 2D "where" map on the 0.25 deg tile grid, small U-Net —
  ``unet``.
- Task C: deterministic impact-to-advisory layer turning Task A predictions
  into crop/cattle guidance text — ``advisory``.

Support modules: ``config`` (paths/constants), ``splits`` (weekly grid,
embargo, temporal splits), ``metrics`` (POD/FAR/CSI, CRPS, slicing),
``audit`` (multi-year gate-0 feature audit), ``data_check`` (raw-data
inventory, fails loudly — never falls back to synthetic data).

All modules import cleanly without raw data; ``python -m modeling.check_modeling``
runs the synthetic known-answer tests without data or GPU.
"""

__all__ = [
    "advisory",
    "audit",
    "baselines",
    "check_modeling",
    "config",
    "data_check",
    "features",
    "metrics",
    "shap_report",
    "splits",
    "train_tabpfn",
    "train_task_a",
    "unet",
]
