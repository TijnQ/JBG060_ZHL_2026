"""Train TabPFN and make predictions using the same setup as LightGBM.

We only fit the models on training data. TabPFN uses pretrained model weights,
so it does not need LightGBM's early stopping step.
We need three area estimates: q10 (lower), q50 (middle) and q90 (upper).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier, TabPFNRegressor

from LightGBM_v1 import config
from LightGBM_v1.features import FEATURE_COLUMNS
from LightGBM_v1.duration_model import build_duration_targets


def _classifier(X, y, seed):
    """Fit a model that predicts whether flooding occurs (0 = no, 1 = yes)."""
    # If every training label is the same, use that value as the probability.
    if y.nunique() == 1:
        return float(y.iloc[0])
    classifier = TabPFNClassifier(random_state=seed)
    classifier.fit(X, y)
    return classifier


def _probability(clf, X):
    """Return the flood probability for each input row."""
    # Handle the fixed probability returned when all training labels match.
    if isinstance(clf, float):
        return np.full(len(X), clf)
    # Find the column for class 1 (flooding). Class order can vary.
    flood_column = list(clf.classes_).index(1)
    probabilities = clf.predict_proba(X)
    return probabilities[:, flood_column]


def train_forecast(X_train, y_train_area, y_train_det,
                   X_val, y_val_area, y_val_det):
    """Fit one model for flooded area and one for flood detection."""
    # X contains input features. The y values contain the outcomes to predict.
    # Keep the validation arguments so the main script can call both models
    # in the same way. TabPFN does not use these rows for fitting or tuning.
    area_model = TabPFNRegressor(random_state=config.SEED)
    area_model.fit(X_train, y_train_area)
    detection_model = _classifier(X_train, y_train_det, config.SEED)
    return {"area": area_model, "det": detection_model}


def predict_forecast(models, X):
    """Predict flood probability and the three flooded-area estimates."""
    X = X[FEATURE_COLUMNS]
    values = models["area"].predict(
        X, output_type="quantiles", quantiles=list(config.QUANTILES)
    )
    # Put q10, q50 and q90 into three columns, with one row per observation.
    if isinstance(values, list):
        area_predictions = np.column_stack(values)
    else:
        area_predictions = np.asarray(values)
    # Stop if the model returns the wrong number of predictions or invalid values.
    if area_predictions.shape != (len(X), 3):
        raise ValueError(f"Unexpected TabPFN quantile shape: {area_predictions.shape}")
    if not np.isfinite(area_predictions).all():
        raise ValueError("TabPFN returned non-finite quantiles")

    # Flooded area cannot be negative. Keep the lower estimate below the
    # middle estimate and the upper estimate above it, just like LightGBM.
    area_predictions = np.clip(area_predictions, 0, None)
    return {
        "det_prob": _probability(models["det"], X),
        "q10": np.minimum(area_predictions[:, 0], area_predictions[:, 1]),
        "q50": area_predictions[:, 1],
        "q90": np.maximum(area_predictions[:, 2], area_predictions[:, 1]),
    }


def train_duration_models(df_train, df_val, horizons=config.DURATION_HORIZONS):
    """Fit a separate flood detection model for each future week (1 to 4)."""
    # Use LightGBM's function to create the same future flood labels.
    targets = build_duration_targets(df_train, horizons)
    models = {}
    for h in horizons:
        target_column = f"y_det_h{h}"
        # The last rows may have no future label, so leave them out of training.
        valid_rows = targets.dropna(subset=[target_column])
        models[h] = _classifier(
            valid_rows[FEATURE_COLUMNS],
            valid_rows[target_column].astype(int),
            config.SEED + h,
        )
    return models


def predict_duration(duration_models, X):
    """Return one probability column for each future week."""
    predictions = pd.DataFrame(index=X.index)
    for h, classifier in duration_models.items():
        predictions[f"det_prob_h{h}"] = _probability(classifier, X[FEATURE_COLUMNS])
    return predictions
