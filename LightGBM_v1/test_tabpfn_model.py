"""Check the model swap's training boundary and prediction contract."""
import unittest
import contextlib
import io
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from LightGBM_v1 import config, tabpfn_model as backend
from LightGBM_v1.features import FEATURE_COLUMNS
from LightGBM_v1.duration_model import build_duration_targets
from LightGBM_v1 import run


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(0.0, index=range(8), columns=FEATURE_COLUMNS)
        self.y = pd.Series([0, 1] * 4)

    def test_validation_rows_are_never_fit(self):
        with patch.object(backend, "TabPFNRegressor") as reg, patch.object(backend, "TabPFNClassifier") as clf:
            backend.train_forecast(self.X, self.y, self.y,
                                             self.X + 100, self.y + 100, self.y + 100)
            pd.testing.assert_frame_equal(reg.return_value.fit.call_args.args[0], self.X)
            pd.testing.assert_frame_equal(clf.return_value.fit.call_args.args[0], self.X)
            self.assertEqual(reg.call_args.kwargs["random_state"], config.SEED)

    def test_quantiles_and_positive_class_match_shared_schema(self):
        reg = MagicMock()
        reg.predict.return_value = [np.full(8, -1), np.full(8, 2), np.full(8, 5)]
        clf = MagicMock()
        clf.classes_ = np.array([1, 0])
        clf.predict_proba.return_value = np.tile([0.7, 0.3], (8, 1))
        result = backend.predict_forecast({"area": reg, "det": clf}, self.X)
        self.assertEqual(set(result), {"det_prob", "q10", "q50", "q90"})
        np.testing.assert_allclose(result["det_prob"], 0.7)
        np.testing.assert_allclose(result["q10"], 0)
        np.testing.assert_allclose(result["q50"], 2)
        np.testing.assert_allclose(result["q90"], 5)
        self.assertEqual(reg.predict.call_args.kwargs,
                         {"output_type": "quantiles", "quantiles": [0.1, 0.5, 0.9]})

    def test_duration_training_matches_lightgbm_targets(self):
        frame = self.X.assign(county="Aweil East", y_det=self.y)
        expected = build_duration_targets(frame)
        with patch.object(backend, "_classifier", return_value=0.5) as fit:
            backend.train_duration_models(frame, frame.assign(y_det=99))
        for h, call in zip(config.DURATION_HORIZONS, fit.call_args_list):
            valid = expected.dropna(subset=[f"y_det_h{h}"])
            pd.testing.assert_frame_equal(call.args[0], valid[FEATURE_COLUMNS])
            pd.testing.assert_series_equal(call.args[1], valid[f"y_det_h{h}"].astype(int))
            self.assertEqual(call.args[2], config.SEED + h)

    def test_runner_preserves_folds_references_and_metrics_between_backends(self):
        weeks = pd.date_range("2000-01-03", "2025-12-29", freq="W-MON")
        frame = pd.DataFrame(0.0, index=range(len(weeks)), columns=FEATURE_COLUMNS)
        frame = frame.assign(week=weeks, county="Aweil East", y_true=1.0,
                             y_det=1, complete=True)
        frame["year"] = weeks.year
        frame["month"] = weeks.month
        captures = []
        fit_rows = []
        for name, train_name, predict_name, module in (
            ("lightgbm", "build_and_train_lightgbm", "predict_lightgbm", run.lgbm_model),
            ("tabpfn", "train_forecast", "predict_forecast", backend),
        ):
            outputs, calls = {}, []

            def train(X, *args):
                calls.append((X.copy(), args[2].copy()))
                return {}

            def predict(models, X, *args, **kwargs):
                return {"det_prob": np.full(len(X), 0.7),
                        "q10": np.zeros(len(X)), "q50": np.ones(len(X)),
                        "q90": np.full(len(X), 2.0)}

            def save(df, path, **kwargs):
                outputs[path.name] = df.copy()

            duration = run.duration_model if name == "lightgbm" else backend
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(run.features, "build_weekly_features", return_value=frame))
                stack.enter_context(patch.object(run.baselines, "add_baselines_to_dataframe",
                    side_effect=lambda df, mask: df.assign(det_prob_clim=0.5, area_clim_q50=1.0,
                                                          det_prob_persist=0.5, area_persist=1.0)))
                stack.enter_context(patch.object(module, train_name, side_effect=train))
                stack.enter_context(patch.object(module, predict_name, side_effect=predict))
                stack.enter_context(patch.object(duration, "train_duration_models", return_value={}))
                stack.enter_context(patch.object(duration, "predict_duration", side_effect=lambda models, X, *args, **kwargs:
                    pd.DataFrame({f"det_prob_h{h}": 0.7 for h in config.DURATION_HORIZONS}, index=X.index)))
                stack.enter_context(patch.object(run.advisory, "climatology_threshold_stats", return_value={}))
                stack.enter_context(patch.object(run.advisory, "generate_advisories", side_effect=lambda df, stats: df))
                stack.enter_context(patch.object(run.advisory, "generate_movement_advice", return_value=pd.DataFrame()))
                stack.enter_context(patch.object(run.data_loader, "get_provenance_report",
                                                return_value=pd.DataFrame({"provenance": ["real"]})))
                stack.enter_context(patch("pathlib.Path.mkdir"))
                stack.enter_context(patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=save))
                run.main(name)
            captures.append(outputs)
            fit_rows.append(calls)
        self.assertEqual(set(captures[0]), set(captures[1]))
        self.assertEqual(len(fit_rows[0]), 3)
        for left, right in zip(fit_rows[0], fit_rows[1]):
            for a, b in zip(left, right):
                pd.testing.assert_frame_equal(a, b)
        for filename in captures[0]:
            a, b = captures[0][filename], captures[1][filename]
            if "model" in a:
                b["model"] = b["model"].replace({"tabpfn": "lightgbm"})
            pd.testing.assert_frame_equal(a, b)
        self.assertNotIn(2025, captures[1]["cv_test_predictions_aweil.csv"]["test_year"].unique())


if __name__ == "__main__":
    unittest.main()
