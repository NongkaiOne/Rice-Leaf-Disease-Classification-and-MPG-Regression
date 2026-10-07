"""Verify inference integrity, validation, and UI without fitting any estimator."""

import importlib
import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
# Keep optional library caches and temporary files within the repository.
RUNTIME = PROJECT_DIR.parent / ".runtime"
for variable, folder in [("MPLCONFIGDIR", "matplotlib"), ("GRADIO_TEMP_DIR", "gradio"),
                         ("HF_HOME", "hf"), ("TEMP", "tmp"), ("TMP", "tmp")]:
    target = RUNTIME / folder
    target.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(target)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import app


class RegressionAppTests(unittest.TestCase):
    def test_saved_model_schema_and_preprocessing(self):
        self.assertIsInstance(app.model, Pipeline)
        self.assertEqual(list(app.model.feature_names_in_), app.FEATURES)
        preprocessing = app.model.named_steps["preprocess"]
        self.assertIsInstance(preprocessing, ColumnTransformer)
        numeric = preprocessing.named_transformers_["num"]
        self.assertIsInstance(numeric.named_steps["imputer"], SimpleImputer)
        self.assertEqual(numeric.named_steps["imputer"].strategy, "median")
        self.assertIsInstance(numeric.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(preprocessing.named_transformers_["cat"], OneHotEncoder)
        np.testing.assert_array_equal(preprocessing.named_transformers_["cat"].categories_[0], [1, 2, 3])
        self.assertIsInstance(app.model.named_steps["regressor"], LinearRegression)

    def test_notebook_and_examples_match_original_pipeline(self):
        self.assertEqual(app.predict_mpg(*app.EXAMPLES[0]), "14.94 MPG")
        for row in [app.DEFAULT_INPUTS, *app.EXAMPLES]:
            with self.subTest(row=row):
                expected = app.model.predict(pd.DataFrame([row], columns=app.FEATURES))[0]
                self.assertEqual(app.predict_mpg(*row), f"{expected:.2f} MPG")
                self.assertIn(f"# {expected:.2f} MPG", app.predict_for_display(*row))

    def test_all_complete_dataset_rows_are_accepted_unchanged(self):
        frame = pd.read_csv(PROJECT_DIR / "data/mpg.csv")
        frame["horsepower"] = pd.to_numeric(frame["horsepower"], errors="coerce")
        complete = frame.dropna(subset=app.FEATURES)
        self.assertEqual(len(complete), 392)
        predictions = app.model.predict(complete[app.FEATURES])
        for row, expected in zip(complete[app.FEATURES].itertuples(index=False, name=None), predictions):
            with self.subTest(row=row):
                self.assertEqual(app.predict_mpg(*row), f"{expected:.2f} MPG")

    def test_invalid_values_show_gradio_errors(self):
        for index in range(len(app.FEATURES)):
            for invalid in [None, "", " ", "not-a-number", float("nan"), float("inf"), float("-inf"), True, [], -1]:
                values = list(app.DEFAULT_INPUTS)
                values[index] = invalid
                with self.subTest(feature=app.FEATURES[index], invalid=invalid):
                    with self.assertRaises(gr.Error):
                        app.predict_mpg(*values)
        for index, invalid in [(0, 0), (0, 7), (0, 4.5), (1, 0), (2, 0), (3, 0), (4, 0),
                               (5, 1976), (5, 100), (5, 76.5), (6, 0), (6, 4), (6, 1.5)]:
            values = list(app.DEFAULT_INPUTS)
            values[index] = invalid
            with self.subTest(feature=app.FEATURES[index], invalid=invalid):
                with self.assertRaises(gr.Error):
                    app.predict_mpg(*values)

    def test_extrapolation_is_not_clamped_or_arbitrarily_rejected(self):
        values = list(app.DEFAULT_INPUTS)
        values[3] = 6000
        values[5] = 69
        expected = app.model.predict(pd.DataFrame([values], columns=app.FEATURES))[0]
        self.assertEqual(app.predict_mpg(*values), f"{expected:.2f} MPG")
        self.assertIn("Extrapolation", app.predict_for_display(*values))
        values[3] = 100000
        self.assertIn("Non-physical result", app.predict_for_display(*values))

    def test_numerical_failure_is_a_clear_error(self):
        with patch.object(app.model, "predict", return_value=[float("nan")]):
            with self.assertRaises(gr.Error):
                app.predict_mpg(*app.DEFAULT_INPUTS)
        with patch.object(app.model, "predict", side_effect=ValueError("overflow")):
            with self.assertRaises(gr.Error):
                app.predict_mpg(*app.DEFAULT_INPUTS)

    def test_example_selector_and_reset_preserve_input_values(self):
        for index, expected in enumerate(app.EXAMPLES):
            with self.subTest(example=index):
                loaded = app.load_example(index)
                self.assertEqual(loaded, (*expected, app.EMPTY_RESULT))
                self.assertEqual(app.predict_mpg(*loaded[:7]), app.predict_mpg(*expected))
        self.assertEqual(app.reset_form(), (*app.DEFAULT_INPUTS, app.EMPTY_RESULT, None))
        with self.assertRaises(gr.Error):
            app.load_example(len(app.EXAMPLES))

    def test_metrics_match_saved_model_on_original_test_split(self):
        metrics = app.load_metrics()
        self.assertEqual(metrics, json.loads(app.METRICS_PATH.read_text(encoding="utf-8")))
        frame = pd.read_csv(PROJECT_DIR / "data/mpg.csv")
        frame["horsepower"] = pd.to_numeric(frame["horsepower"], errors="coerce")
        x_train, x_test, _, y_test = train_test_split(frame[app.FEATURES], frame["mpg"], test_size=0.20, random_state=42)
        self.assertEqual((len(frame), len(x_train), len(x_test)),
                         (metrics["rows"], metrics["train_rows"], metrics["test_rows"]))
        predictions = app.model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        for key, actual in [("mae", mean_absolute_error(y_test, predictions)), ("mse", mse),
                            ("rmse", math.sqrt(mse)), ("r2", r2_score(y_test, predictions))]:
            self.assertAlmostEqual(metrics[key], actual, places=12)

    def test_startup_and_ui_never_fit_estimators(self):
        estimators = [Pipeline, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, LinearRegression]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for estimator in estimators:
                stack.enter_context(patch.object(estimator, "fit", side_effect=AssertionError("Startup must not train")))
            importlib.reload(app)
            demo = app.build_app()
            self.assertIsInstance(demo, gr.Blocks)
            config = demo.config
            self.assertFalse(any(c["type"] in ("tabs", "tabitem") for c in config["components"]))
            section_ids = {c["props"].get("elem_id") for c in config["components"]}
            self.assertTrue({"prediction", "evaluation", "about", "example-selector"} <= section_ids)
            header = next(c for c in config["components"] if c["props"].get("elem_id") == "site-header")
            for section in ("prediction", "evaluation", "about"):
                self.assertIn(f'href="#{section}"', header["props"]["value"])
            prediction_event = next(event for event in config["dependencies"] if event["api_name"] == "predict_mpg")
            self.assertEqual(len(prediction_event["inputs"]), 7)
            buttons = [c["props"]["value"] for c in config["components"] if c["type"] == "button"]
            self.assertFalse(any("flag" in label.lower() for label in buttons))
            self.assertEqual(app.reset_inputs(), (*app.DEFAULT_INPUTS, app.EMPTY_RESULT))
            with patch.object(demo, "queue", return_value=demo), patch.object(demo, "launch") as launch, \
                    patch.object(app, "build_app", return_value=demo), \
                    patch.dict(os.environ, {"PORT": "10000"}):
                os.environ.pop("GRADIO_SERVER_NAME", None)
                app.main()
                self.assertEqual(launch.call_args.kwargs["server_name"], "0.0.0.0")
                self.assertEqual(launch.call_args.kwargs["server_port"], 10000)


if __name__ == "__main__":
    unittest.main()
