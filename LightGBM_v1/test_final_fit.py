"""Check the final-fit split boundary and forecast contract."""
import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd

from LightGBM_v1 import config, run, splits
from LightGBM_v1.features_fe import FEATURE_COLUMNS_ENGINEERED


def _weekly_frame(weeks, counties):
    frames = []
    for county in counties:
        frame = pd.DataFrame(0.0, index=range(len(weeks)), columns=FEATURE_COLUMNS_ENGINEERED)
        frame = frame.assign(week=weeks, county=county, y_true=0.0, y_det=1, complete=True)
        frame["year"] = weeks.year
        frame["month"] = weeks.month
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class FinalFitSplitTests(unittest.TestCase):
    def test_train_stops_before_final_year_and_val_purges_first_week(self):
        weeks = pd.date_range("2003-01-06", "2005-12-26", freq="W-MON")
        frame = _weekly_frame(weeks, ("Aweil East",))
        train_df, val_df, final_year = splits.final_fit_split(frame)
        self.assertEqual(final_year, 2005)
        self.assertEqual(sorted(train_df["week"].dt.year.unique()), [2003, 2004])
        self.assertEqual(sorted(val_df["week"].dt.year.unique()), [2005])
        self.assertEqual(val_df["week"].min(), pd.Timestamp("2005-01-10"))  # first 2005 week purged
        self.assertEqual(set(train_df["week"]) & set(val_df["week"]), set())


class FinalFitRunnerTests(unittest.TestCase):
    def test_final_fit_trains_pre_final_year_and_forecasts_latest_complete_week(self):
        weeks = pd.date_range("2003-01-06", "2005-12-26", freq="W-MON")
        frame = _weekly_frame(weeks, ("Aweil East", "Aweil North"))
        frame.loc[frame["week"] == weeks[-1], "complete"] = False  # latest week fails the gate
        latest_complete = weeks[-2]

        # Mirror the pipeline's own processing so indices can be asserted.
        feats = frame[frame["complete"]].reset_index(drop=True)

        calls, outputs = {}, {}

        def train(X_tr, y_tr, ydet_tr, X_val, y_val, ydet_val, **kwargs):
            calls["train_idx"] = X_tr.index.to_numpy()
            calls["val_idx"] = X_val.index.to_numpy()
            self.assertEqual(kwargs, {})  # fe_baseline keeps the historic params=None call shape
            return {}

        def predict(models, X, *args, **kwargs):
            calls["pred_idx"] = X.index.to_numpy()
            return {"det_prob": pd.Series(0.7, index=X.index),
                    "q10": pd.Series(0.0, index=X.index),
                    "q50": pd.Series(1.0, index=X.index),
                    "q90": pd.Series(2.0, index=X.index)}

        def train_duration(tr, val, feature_columns=None, params=None):
            calls["dur_train_years"] = sorted(tr["week"].dt.year.unique())
            calls["dur_val_years"] = sorted(val["week"].dt.year.unique())
            return {}

        def predict_duration(models, X, *args, **kwargs):
            calls["dur_pred_weeks"] = set(X["week"])
            return pd.DataFrame({f"det_prob_h{h}": 0.7 for h in config.DURATION_HORIZONS}, index=X.index)

        def save(df, path, **kwargs):
            outputs[path.name] = df.copy()

        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(run.features_fe, "build_weekly_features_fe", return_value=frame))
            stack.enter_context(patch.object(run.lgbm_model, "build_and_train_lightgbm", side_effect=train))
            stack.enter_context(patch.object(run.lgbm_model, "predict_lightgbm", side_effect=predict))
            stack.enter_context(patch.object(run.duration_model, "train_duration_models", side_effect=train_duration))
            stack.enter_context(patch.object(run.duration_model, "predict_duration", side_effect=predict_duration))
            stack.enter_context(patch.object(run.data_loader, "get_provenance_report",
                                             return_value=pd.DataFrame({"provenance": ["real"]})))
            stack.enter_context(patch("pathlib.Path.mkdir"))
            stack.enter_context(patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=save))
            run.final_fit("lightgbm_fe", "fe_baseline")

        np.testing.assert_array_equal(
            calls["train_idx"], feats.index[feats["week"].dt.year < 2005].to_numpy())
        np.testing.assert_array_equal(
            calls["val_idx"],
            feats.index[(feats["week"].dt.year == 2005)
                        & (feats["week"] > pd.Timestamp("2005-01-03"))].to_numpy())
        self.assertEqual(calls["dur_train_years"], [2003, 2004])
        self.assertEqual(calls["dur_val_years"], [2005])

        np.testing.assert_array_equal(
            calls["pred_idx"], feats.index[feats["week"] == latest_complete].to_numpy())
        self.assertEqual(calls["dur_pred_weeks"], {latest_complete})

        self.assertEqual(list(outputs), ["final_forecast_aweil.csv"])
        table = outputs["final_forecast_aweil.csv"]
        self.assertEqual(list(table.columns),
                         ["county", "week", "det_prob", "q10", "q50", "q90",
                          "det_prob_h1", "det_prob_h2"])
        self.assertEqual(len(table), 2)
        self.assertTrue(table["week"].eq(latest_complete).all())
        self.assertNotIn("det_prob_h3", table)  # h3/h4 lose to climatology & persistence
        self.assertNotIn("det_prob_h4", table)


if __name__ == "__main__":
    unittest.main()