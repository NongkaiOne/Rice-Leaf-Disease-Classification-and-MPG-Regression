"""Measurement conversions and unit-aware UI must preserve vehicle quantities."""

import math
import unittest
from unittest.mock import patch

from test_app import app, gr


class MeasurementUnitTests(unittest.TestCase):
    def test_lb_kg_exact_factor_and_round_trip(self):
        self.assertEqual(app.convert_input(1, "weight", "lb", "kg"), 0.45359237)
        self.assertAlmostEqual(app.convert_input(1, "weight", "kg", "lb"), 2.2046226218487757)
        kg = app.convert_input(3504, "weight", "lb", "kg")
        self.assertAlmostEqual(kg, 1589.38766448, places=10)
        self.assertAlmostEqual(app.convert_input(kg, "weight", "kg", "lb"), 3504, places=10)

    def test_displacement_all_units(self):
        self.assertEqual(app.convert_input(1, "displacement", "cu in", "cc"), 16.387064)
        self.assertAlmostEqual(app.convert_input(1, "displacement", "L", "cc"), 1000, places=12)
        self.assertAlmostEqual(app.convert_input(307, "displacement", "cu in", "L"), 5.030828648, places=12)
        for source in ("cu in", "cc", "L"):
            for target in ("cu in", "cc", "L"):
                with self.subTest(source=source, target=target):
                    value = app.convert_input(307, "displacement", "cu in", source)
                    converted = app.convert_input(value, "displacement", source, target)
                    original = app.convert_input(converted, "displacement", target, "cu in")
                    self.assertAlmostEqual(original, 307, places=11)

    def test_mechanical_hp_kw(self):
        self.assertEqual(app.convert_input(1, "horsepower", "hp", "kW"), 0.7456998715822702)
        self.assertAlmostEqual(app.convert_input(1, "horsepower", "kW", "hp"), 1.3410220895950277)
        kw = app.convert_input(130, "horsepower", "hp", "kW")
        self.assertAlmostEqual(kw, 96.94098330569513, places=12)
        self.assertAlmostEqual(app.convert_input(kw, "horsepower", "kW", "hp"), 130, places=12)

    def test_us_mpg_km_l_and_l_100km(self):
        self.assertAlmostEqual(app.convert_fuel_economy(10, "US MPG", "km/L"), 4.25143707430272, places=12)
        self.assertAlmostEqual(app.convert_fuel_economy(10, "US MPG", "L/100 km"), 23.52145833333333, places=12)
        for source in app.OUTPUT_UNITS:
            for target in app.OUTPUT_UNITS:
                with self.subTest(source=source, target=target):
                    value = app.convert_fuel_economy(14.94, "US MPG", source)
                    converted = app.convert_fuel_economy(value, source, target)
                    self.assertAlmostEqual(app.convert_fuel_economy(converted, target, "US MPG"), 14.94, places=12)

    def test_metric_us_and_mixed_prediction_equivalence(self):
        for values in [app.DEFAULT_INPUTS, *app.EXAMPLES]:
            expected = app.predict_mpg_value(*values)
            for units in [app.ORIGINAL_UNITS, app.UNIT_PRESETS["Metric"], ("cc", "hp", "kg")]:
                converted = [app.convert_input(value, feature, original, unit) for value, feature, original, unit in zip(
                    values[1:4], app.CONVERTIBLE_FEATURES, app.ORIGINAL_UNITS, units)]
                with self.subTest(values=values, units=units):
                    prediction = app.predict_with_units(values[0], *converted, *values[4:], *units)
                    self.assertAlmostEqual(prediction, expected, places=12)
        self.assertEqual(app.predict_mpg(*app.EXAMPLES[0]), "14.94 MPG")

    def test_repeated_unit_switching_has_no_cumulative_drift(self):
        state = app.make_unit_state(app.EXAMPLES[0][1:4])
        exact = list(state["model_values"])
        for turn in range(1000):
            target = "US" if turn % 2 == 0 else "Metric"
            switched = app.update_unit_controls(target, *state["units"], *state["display_values"], state)
            state = switched[-1]
            self.assertEqual(state["model_values"], exact)
        for turn in range(200):
            units = ("cc", "hp", "kg") if turn % 2 == 0 else ("L", "kW", "kg")
            state = app.update_unit_controls(state["system"], *units, *state["display_values"], state)[-1]
            self.assertEqual(state["model_values"], exact)
        result, _, prediction = app.predict_vehicle(8, *state["display_values"], 12, 70, 1, "US MPG", state)
        self.assertEqual(prediction["mpg"], app.predict_mpg_value(*app.EXAMPLES[0]))
        self.assertIn("14.94", result)

    def test_user_edit_is_converted_and_state_is_isolated(self):
        state = app.make_unit_state(app.EXAMPLES[0][1:4])
        before = list(state["model_values"])
        edited = list(state["display_values"])
        edited[2] += 10
        new_state = app.update_unit_controls("US", *state["units"], *edited, state)[-1]
        self.assertEqual(state["model_values"], before)
        self.assertEqual(new_state["model_values"][:2], before[:2])
        self.assertAlmostEqual(new_state["model_values"][2], edited[2] / 0.45359237, places=10)

    def test_examples_and_reset_use_selected_units(self):
        for units in [app.ORIGINAL_UNITS, app.UNIT_PRESETS["Metric"], ("cc", "hp", "kg")]:
            for index, values in enumerate(app.EXAMPLES):
                with self.subTest(units=units, example=index):
                    state = app.make_unit_state(units=units)
                    loaded = app.load_vehicle_example(index, state, "L/100 km")
                    state = loaded[-2]
                    self.assertEqual(state["model_values"], values[1:4])
                    self.assertEqual(tuple(state["units"]), units)
                    _, _, prediction = app.predict_vehicle(*loaded[:7], "L/100 km", state)
                    self.assertEqual(prediction["mpg"], app.predict_mpg_value(*values))
                    reset = app.reset_vehicle_form(state, "L/100 km")
                    self.assertEqual(reset[-2]["model_values"], app.DEFAULT_INPUTS[1:4])
                    self.assertEqual(tuple(reset[-2]["units"]), units)
                    self.assertIsNone(reset[8])
                    self.assertIsNone(reset[-1])
                    self.assertIn("L/100 km", reset[7])
        with self.assertRaises(gr.Error):
            app.load_vehicle_example(99, state, "US MPG")

    def test_output_switch_reuses_unrounded_cached_prediction(self):
        prediction = {"mpg": app.predict_mpg_value(*app.EXAMPLES[0]), "model_year": 70}
        with patch.object(app.model, "predict", side_effect=AssertionError("Changing output unit must not predict again")):
            for unit in app.OUTPUT_UNITS:
                result = app.update_output_unit(unit, prediction)
                expected = app.convert_fuel_economy(prediction["mpg"], "US MPG", unit)
                self.assertIn(app.fuel_display_number(expected), result)
                self.assertIn("US liquid gallon", result)
                self.assertIn("<svg", result)

    def test_zero_negative_and_nonfinite_fuel_economy_are_safe(self):
        for invalid in [None, True, "bad", 0, -1, math.inf, math.nan]:
            for unit in app.OUTPUT_UNITS:
                with self.subTest(value=invalid, unit=unit):
                    with self.assertRaises(gr.Error):
                        app.convert_fuel_economy(invalid, "US MPG", unit)
        for mpg in (0, -20):
            for unit in app.OUTPUT_UNITS:
                result = app.render_instrument({"mpg": mpg, "model_year": 70}, unit)
                self.assertIn("Non-physical result", result)
                self.assertNotIn("fuel-equivalents", result)
        for mpg in (math.inf, math.nan):
            with self.assertRaises(gr.Error):
                app.render_instrument({"mpg": mpg, "model_year": 70})
        with self.assertRaises(gr.Error):
            app.convert_fuel_economy(1e-320, "US MPG", "L/100 km")

    def test_invalid_input_units_and_values(self):
        for invalid in [None, True, "bad", math.nan, math.inf]:
            with self.assertRaises(gr.Error):
                app.convert_input(invalid, "weight", "kg", "lb")
        for feature, source, target in [("weight", "kg", "g"), ("displacement", "hp", "L"),
                                         ("horsepower", "PS", "hp"), ("cylinders", "count", "kg")]:
            with self.assertRaises(gr.Error):
                app.convert_input(1, feature, source, target)
        with self.assertRaises(gr.Error):
            app.convert_fuel_economy(10, "UK MPG", "US MPG")
        with self.assertRaises(gr.Error):
            app.make_unit_state([None, None, None], ("invalid", "hp", "lb"))
        for index, invalid in [(1, -1), (2, 0), (3, -100), (5, 1970), (6, 4)]:
            state = app.make_unit_state()
            values = list(app.DEFAULT_INPUTS)
            values[1:4] = state["display_values"]
            values[index] = invalid
            with self.subTest(index=index, invalid=invalid):
                with self.assertRaises(gr.Error):
                    app.predict_vehicle(*values, "km/L", state)

    def test_blank_field_remains_blank_through_unit_switch(self):
        state = app.make_unit_state()
        switched = app.update_unit_controls("US", *state["units"], None, *state["display_values"][1:], state)
        self.assertIsNone(switched[4]["value"])
        self.assertIsNone(switched[-1]["model_values"][0])
        with self.assertRaises(gr.Error):
            app.predict_vehicle(4, None, *switched[-1]["display_values"][1:], 15, 76, 1, "US MPG", switched[-1])

    def test_metric_defaults_and_dynamic_labels(self):
        state = app.make_unit_state()
        self.assertEqual(state["system"], "Metric")
        self.assertEqual(state["units"], ["L", "kW", "kg"])
        self.assertEqual(app.DEFAULT_OUTPUT_UNIT, "km/L")
        switched = app.update_unit_controls("US", *state["units"], *state["display_values"], state)
        self.assertIn("cu in", switched[4]["label"])
        self.assertIn("hp", switched[5]["label"])
        self.assertIn("lb", switched[6]["label"])

    def test_small_measurements_are_not_rounded_to_zero(self):
        state = app.make_unit_state([1e-10, 1e-10, 1e-10])
        self.assertTrue(all(value > 0 for value in state["display_values"]))
        switched = app.update_unit_controls("US", *state["units"], *state["display_values"], state)[-1]
        self.assertEqual(switched["model_values"], [1e-10, 1e-10, 1e-10])


if __name__ == "__main__":
    unittest.main()
