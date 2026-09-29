"""Forecasting and impact-advisory models for the South Sudan flood project.

This package is the *testing ground* for the model-development part of the
project: every candidate model method lives in its own subfolder with a
``guide.md`` (what to test, how to run it, keep/kill criteria) and the
losers get deleted. Nothing here falls back to synthetic data.

- Task A ("how much flood area next week", per county): one folder per
  model method — ``lightgbm/`` (the primary backbone, run first), then the
  cross-checks ``xgboost/``, ``catboost/``, ``randomforest/`` (untuned,
  degenerate point forecast) and ``tabpfn/`` (zero-shot reference). All
  share the pipeline core in ``methods_common``.
- Task B ("where will it flood?", 2D pixel map of the Aweil floodplain):
  ``unet/`` — a small U-Net on the 0.25 deg tile grid, compared against a
  null (historical-frequency) baseline.
- Task C (advisory): ``advisory`` — deterministic impact-to-guidance layer
  turning Task A predictions into crop/cattle text (not a competing model;
  it stays a module rather than a folder).

Support modules: ``config`` (paths/constants, the method registry),
``features`` (embargoed weekly features + labels), ``splits`` (weekly grid,
3-day embargo, purged temporal splits), ``metrics`` (POD/FAR/CSI, CRPS,
slicing), ``baselines`` (persistence / climatology / last-detection),
``audit`` (multi-year gate-0 feature audit), ``data_check`` (raw-data
inventory, fails loudly), ``shap_report`` (tree-attribution helper).

All modules import cleanly without raw data or GPU;
``python -m modeling.check_modeling`` runs the synthetic known-answer
tests.
"""

__all__ = [
    "advisory",
    "audit",
    "baselines",
    "check_modeling",
    "config",
    "data_check",
    "features",
    "methods_common",
    "metrics",
    "shap_report",
    "splits",
]
