"""SHAP attributions for the fitted tabular models (E4/E5 follow-up).

Answers "which features actually drive the forecast" on a fitted
LightGBM/XGBoost/CatBoost booster, so feature selection can be argued
from the trained model rather than from anecdote. Requires ``shap``
(see ``requirements-ml.txt``); degrades gracefully if it is not installed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["shap_plot", "shap_summary"]


def shap_summary(
    model, X: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature for a tree model."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # multi-output models: average over outputs
        values = np.stack(values, axis=0).mean(axis=0)
    out = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out


def shap_plot(
    summary: pd.DataFrame, path, title: str = "Mean |SHAP| by feature"
) -> None:
    top = summary.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_title(title)
    ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
