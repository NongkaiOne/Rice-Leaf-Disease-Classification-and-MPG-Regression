"""Replay Gradio's selection/blur requests through its real event/state engine."""

from copy import deepcopy
import unittest

from test_app import app, gr
from gradio.state_holder import SessionState


class UnitEventInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.demo = app.build_app()
        self.config = self.demo.config
        self.components = {component["id"]: component for component in self.config["components"]}
        self.front = {index: deepcopy(component["props"].get("value"))
                      for index, component in self.components.items()}
        self.ids = {component["props"].get("elem_id"): index for index, component in self.components.items()}
        self.unit_ids = [self.ids[name] for name in (
            "measurement-system", "displacement-unit", "power-unit", "weight-unit")]
        self.events = {event["api_name"]: event for event in self.config["dependencies"]}
        self.switch = self.events["switch_units"]
        self.unit_state_id = self.switch["inputs"][-1]
        self.session = SessionState(self.demo)

    async def asyncTearDown(self):
        self.demo.close()

    def payload(self, event):
        return [deepcopy(self.front[index]) for index in event["inputs"]]

    async def dispatch(self, event, payload, target=None):
        # process_api performs the actual preprocessing, EventData injection,
        # callback, output updates, and per-session gr.State synchronization.
        response = await self.demo.process_api(
            event["id"], payload, state=self.session,
            event_data=gr.EventData(self.demo.blocks[target], None) if target is not None else None,
        )
        for index, update in zip(event["outputs"], response["data"]):
            if self.components[index]["type"] == "state":
                continue  # Browser receives None; the server session owns State.
            if isinstance(update, dict) and update.get("__type__") == "update":
                if "value" in update:
                    self.front[index] = update["value"]
            else:
                self.front[index] = update
        return response

    async def choose(self, selector, value, duplicate_blur=True):
        target = self.unit_ids[selector]
        self.front[target] = value
        captured = self.payload(self.switch)
        await self.dispatch(self.switch, captured, target)
        if duplicate_blur:
            # Chromium captured two identical requests before the first reply:
            # selection followed by blur, with old siblings/numeric values.
            await self.dispatch(self.switch, deepcopy(captured), target)

    def controls(self):
        return [self.front[index] for index in self.unit_ids]

    def canonical(self):
        return self.session[self.unit_state_id]["model_values"]

    async def test_duplicate_selection_blur_keeps_metric_us_presets(self):
        exact = list(self.canonical())
        await self.choose(0, "US")
        self.assertEqual(self.controls(), ["US", "cu in", "hp", "lb"])
        self.assertEqual(self.canonical(), exact)
        await self.choose(0, "Metric")
        self.assertEqual(self.controls(), ["Metric", "L", "kW", "kg"])
        self.assertEqual(self.canonical(), exact)

    async def test_individual_choices_survive_duplicate_requests(self):
        exact = list(self.canonical())
        for selector, value in [(3, "lb"), (3, "kg"), (1, "cu in"), (1, "cc"),
                                (1, "L"), (2, "hp"), (2, "kW")]:
            with self.subTest(selector=selector, value=value):
                await self.choose(selector, value)
                self.assertEqual(self.controls()[selector], value)
                self.assertEqual(self.controls()[0], app.measurement_system(self.controls()[1:]))
                self.assertEqual(self.canonical(), exact)

    async def test_trigger_target_ignores_stale_sibling_selectors(self):
        await self.choose(0, "US")
        await self.choose(3, "kg")
        self.assertEqual(self.controls(), ["Custom", "cu in", "hp", "kg"])
        await self.choose(2, "kW")
        exact = list(self.canonical())
        self.front[self.unit_ids[1]] = "L"
        captured = self.payload(self.switch)
        captured[:4] = ["US", "L", "hp", "lb"]
        await self.dispatch(self.switch, captured, self.unit_ids[1])
        self.assertEqual(self.controls(), ["Metric", "L", "kW", "kg"])
        self.assertEqual(self.canonical(), exact)
        await self.dispatch(self.switch, deepcopy(captured), self.unit_ids[1])
        self.assertEqual(self.controls(), ["Metric", "L", "kW", "kg"])
        self.assertEqual(self.canonical(), exact)

    async def test_programmatic_changes_have_no_selector_change_callbacks(self):
        for event in self.config["dependencies"]:
            for target, event_name in event["targets"]:
                if target in self.unit_ids:
                    self.assertEqual(event_name, "input")
        await self.choose(0, "US")
        before = deepcopy(self.session[self.unit_state_id])
        # Numeric/selector outputs emit change, not user-input events. No unit
        # callback subscribes to those changes, so they cannot create a loop.
        for target in self.unit_ids:
            self.assertFalse(any([target, "change"] in event["targets"]
                                 for event in self.config["dependencies"]))
        self.assertEqual(self.session[self.unit_state_id], before)
        await self.choose(0, "Custom")
        self.assertEqual(self.controls(), ["Custom", "cu in", "hp", "lb"])

    async def test_repeated_event_switching_preserves_quantities_and_prediction(self):
        await self.dispatch(self.events["load_example"], [0, None, "US MPG"])
        exact = list(app.EXAMPLES[0][1:4])
        for turn in range(50):
            preset = "US" if turn % 2 == 0 else "Metric"
            await self.choose(0, preset)
            self.assertEqual(self.controls()[0], preset)
            self.assertEqual(self.canonical(), exact)
        prediction = self.events["predict_mpg"]
        output = prediction["inputs"][7]
        self.front[output] = "US MPG"
        response = await self.dispatch(prediction, self.payload(prediction))
        self.assertIn("14.94", response["data"][0])
        cached = self.session[prediction["outputs"][-1]]
        self.assertEqual(cached["mpg"], app.predict_mpg_value(*app.EXAMPLES[0]))

    async def test_examples_and_reset_keep_units_after_real_selector_events(self):
        await self.choose(0, "US")
        await self.choose(1, "cc")
        await self.choose(3, "kg")
        selected = list(self.controls())
        await self.dispatch(self.events["load_example"], [0, None, "US MPG"])
        self.assertEqual(self.controls(), selected)
        self.assertEqual(self.canonical(), app.EXAMPLES[0][1:4])
        prediction = self.events["predict_mpg"]
        self.front[prediction["inputs"][7]] = "US MPG"
        response = await self.dispatch(prediction, self.payload(prediction))
        self.assertIn("14.94", response["data"][0])
        await self.dispatch(self.events["reset_form"], [None, "US MPG"])
        self.assertEqual(self.controls(), selected)
        self.assertEqual(self.canonical(), app.DEFAULT_INPUTS[1:4])
        self.assertIsNone(self.front[self.ids["example-selector"]])
        self.assertIsNone(self.session[prediction["outputs"][-1]])


if __name__ == "__main__":
    unittest.main()
