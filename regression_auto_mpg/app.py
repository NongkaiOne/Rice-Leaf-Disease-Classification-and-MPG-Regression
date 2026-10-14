"""Auto MPG web application: inference with the existing fitted Pipeline only."""

import json
import math
import os
from copy import deepcopy
from pathlib import Path

# Disable Gradio telemetry before importing it; inference needs no network access.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import joblib
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
METRICS_PATH = APP_DIR / "metrics.json"
FEATURES = [
    "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin",
]
CYLINDERS = [3, 4, 5, 6, 8]
ORIGINS = [("USA (1)", 1), ("Europe (2)", 2), ("Japan (3)", 3)]
DEFAULT_INPUTS = [4, 140.0, 90.0, 2500.0, 15.0, 76, 1]
EXAMPLES = [
    [8, 307.0, 130.0, 3504, 12.0, 70, 1],
    [4, 97.0, 88.0, 2130, 14.5, 71, 3],
    [4, 120.0, 79.0, 2625, 18.6, 82, 1],
    [4, 104.0, 95.0, 2375, 17.5, 70, 2],
]
EXAMPLE_NAMES = ["1970 Chevrolet Chevelle Malibu", "1971 Datsun PL510",
                 "1982 Ford Ranger", "1970 Saab 99E"]
LB_TO_KG = 0.45359237
CU_IN_TO_CC = 16.387064
HP_TO_KW = 0.7456998715822702
MILE_TO_KM = 1.609344
US_GALLON_TO_L = 3.785411784  # US liquid gallon: 231 cubic inches.
KM_PER_L_PER_US_MPG = MILE_TO_KM / US_GALLON_TO_L
INPUT_UNIT_FACTORS = {
    "displacement": {"cu in": 1.0, "cc": CU_IN_TO_CC, "L": CU_IN_TO_CC / 1000},
    "horsepower": {"hp": 1.0, "kW": HP_TO_KW},
    "weight": {"lb": 1.0, "kg": LB_TO_KG},
}
CONVERTIBLE_FEATURES = ("displacement", "horsepower", "weight")
ORIGINAL_UNITS = ("cu in", "hp", "lb")
UNIT_PRESETS = {"Metric": ("L", "kW", "kg"), "US": ORIGINAL_UNITS}
OUTPUT_UNITS = ["US MPG", "km/L", "L/100 km"]
DEFAULT_OUTPUT_UNIT = "km/L"
CAR_SVG = """<svg class="car-outline" viewBox="0 0 96 32" aria-hidden="true" focusable="false">
    <path d="M8 23v-8l9-3 10-8h32l14 11 12 3 3 5h-9M17 23h49M27 4l-6 11h48M44 5v10"
          fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
    <circle cx="12" cy="24" r="5" fill="none" stroke="currentColor" stroke-width="1.7"/>
    <circle cx="73" cy="24" r="5" fill="none" stroke="currentColor" stroke-width="1.7"/>
</svg>"""
EMPTY_RESULT = (
    "### Estimated fuel economy\n\n# — MPG\n\n"
    "กรอกข้อมูลรถแล้วกดทำนาย เพื่อดูระยะทางโดยประมาณต่อน้ำมันหนึ่งแกลลอน"
)
CSS = """
body {background: #0f141b;}
.gradio-container {
    width: 100% !important;
    max-width: 1480px !important;
    margin-inline: auto;
    padding: clamp(18px, 3.2vw, 48px) !important;
    color-scheme: dark;
    --layout-gap: 18px;
}
.gradio-container, .gradio-container * {box-sizing: border-box;}
.gradio-container :is(.row, .column, .form, .block) {min-width: 0;}
.gradio-container :is(h1, h2, h3, p, dt, dd, label) {overflow-wrap: anywhere;}
.gradio-container :is(input, textarea) {width: 100%;}
.gradio-container :is(a, button, input):focus-visible {
    outline: 2px solid #72b3ff;
    outline-offset: 3px;
}
.gradio-container a {color: #8bbcff;}
.gradio-container .site-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
    padding-bottom: 24px;
    border-bottom: 1px solid #303b49;
}
.gradio-container .brand {display: grid; gap: 6px;}
.gradio-container .brand a {
    color: #f2f5fa;
    text-decoration: none;
    font-size: clamp(1.1rem, 1.8vw, 1.35rem);
    font-weight: 600;
    letter-spacing: -0.025em;
}
.gradio-container .brand span {color: #aab5c4; font-size: 0.82rem;}
.gradio-container .site-nav {display: flex; gap: clamp(16px, 2.5vw, 32px); flex-wrap: wrap;}
.gradio-container .site-nav a {color: #c2cbd7; text-decoration: none; padding: 8px 0; font-size: 0.9rem;}
.gradio-container .site-nav a:hover {color: #8bbcff;}
.gradio-container .hero {padding: clamp(24px, 3.5vw, 44px) 0 12px; max-width: 900px;}
.gradio-container .hero h1 {font-size: clamp(1.6rem, 3vw, 2.35rem); font-weight: 500; line-height: 1.4; margin: 0 0 14px; color: #f2f5fa;}
.gradio-container .hero p {max-width: 760px; margin: 0; color: #aab5c4; line-height: 1.85;}
.gradio-container .page-section {padding-top: clamp(24px, 3vw, 40px); gap: 22px; scroll-margin-top: 24px;}
.gradio-container .section-heading {display: flex; align-items: baseline; gap: 24px; flex-wrap: wrap;}
.gradio-container .section-heading h2 {font-size: clamp(1.25rem, 2vw, 1.55rem); color: #f2f5fa; font-weight: 500; margin: 0;}
.gradio-container .section-heading p {color: #aab5c4; font-size: 0.9rem; margin: 0;}
#prediction-layout {display: grid !important; grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr); gap: clamp(24px, 4vw, 56px); align-items: start;}
#prediction-layout > * {width: 100%; min-width: 0 !important;}
#vehicle-form {gap: 18px;}
#unit-controls {display: grid !important; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; padding-bottom: 18px; border-bottom: 1px solid #303b49;}
#unit-controls > * {min-width: 0 !important; width: 100%;}
#unit-controls label {font-size: 0.8rem;}
#vehicle-form .block.padded, #output-unit.block.padded, #example-selector.block.padded {padding: 14px 16px !important;}
#vehicle-form [data-testid="block-info"], #output-unit [data-testid="block-info"], #example-selector [data-testid="block-info"] {padding: 2px 4px 8px; margin: 0; line-height: 1.6; display: block;}
#vehicle-form .help-label, #example-selector .help-label {padding-right: 32px;}
.gradio-container .field-help-source {position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap;}
.gradio-container .field-help-button {position: absolute; width: 28px; height: 28px; padding: 0; border: 0; background: transparent; color: #aab5c4; cursor: pointer; display: grid; place-items: center; z-index: var(--layer-5, 50);}
.gradio-container .field-help-button::before {content: 'i'; width: 16px; height: 16px; border: 1px solid #65758b; border-radius: 50%; font: 500 11px/14px Georgia, serif; text-align: center;}
.gradio-container .field-help-button:hover, .gradio-container .field-help-button:focus-visible {color: #8bbcff;}
.gradio-container .field-help-tooltip {position: fixed; z-index: 1000; max-width: min(260px, calc(100vw - 24px)); padding: 9px 12px; border: 1px solid #455368; border-radius: 5px; background: #202a37; color: #d3dbe6; box-shadow: 0 6px 18px #0005; font-size: 0.8rem; line-height: 1.65; overflow-wrap: anywhere;}
.gradio-container .field-help-tooltip[hidden] {display: none;}
#output-unit {width: 100%; max-width: 280px; margin-inline: auto;}
.gradio-container .instrument {display: flex; flex-direction: column; text-align: center;}
.gradio-container .instrument-heading {display: flex; flex-direction: column; align-items: center; gap: 10px; order: 0;}
.gradio-container .instrument-heading p {margin: 0 !important; color: #aab5c4;}
.gradio-container .car-outline {width: 76px; height: 28px; flex: 0 0 auto; color: #82a8d9;}
.gradio-container .instrument-reading {display: flex; flex-direction: column; align-items: center; gap: 8px; margin: 24px 0 0; order: 1;}
.gradio-container .instrument-value {font: 500 clamp(2.8rem, 5vw, 4.2rem)/1.1 Consolas, monospace; color: #f2f5fa; letter-spacing: -0.04em; font-variant-numeric: tabular-nums; max-width: 100%; overflow-wrap: anywhere;}
.gradio-container .instrument-unit {color: #bac4d1; font-size: 0.9rem; overflow-wrap: anywhere;}
.gradio-container .fuel-equivalents {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; padding-top: 20px; margin: 26px 0 0; border-top: 1px solid #303b49; order: 2;}
.gradio-container .fuel-equivalents dt {font-size: 0.76rem; color: #aab5c4; line-height: 1.6;}
.gradio-container .fuel-equivalents dd {font-size: 1rem; color: #d3dbe6; margin: 6px 0 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere;}
#prediction-result .instrument-note {font-size: 0.75rem; color: #aab5c4; line-height: 1.65; max-width: 48ch; margin: 18px auto 0; order: 3;}
@media (max-width: 1200px) {#unit-controls {grid-template-columns: repeat(2, minmax(0, 1fr));}}
@media (prefers-reduced-motion: no-preference) {
    #prediction-result {transition: border-color 160ms ease;}
    .gradio-container .car-outline {transition: color 160ms ease;}
}
#vehicle-form :is(label, .label-wrap, .label-wrap span), #example-area label {
    white-space: normal;
    max-width: 100%;
}
.gradio-container .field-row {display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px;}
.gradio-container .field-row > * {width: 100%; min-width: 0 !important;}
#prediction-actions {display: flex; flex-wrap: wrap; gap: 12px; padding-top: 8px;}
#prediction-actions button {min-height: 46px; min-width: 0 !important; flex: 1 1 150px !important; white-space: normal;}
#prediction-result {
    border: 1px solid #303b49;
    border-top: 2px solid #659deb;
    border-radius: 4px;
    padding: clamp(24px, 3vw, 36px);
    background: #171e28;
    width: 100%;
}
#prediction-result h3 {font-size: 0.95rem; color: #aab5c4; font-weight: 400; margin: 0;}
#prediction-result h1 {font-size: clamp(2.7rem, 5vw, 4rem); color: #f2f5fa; line-height: 1.15; margin: 24px 0; font-weight: 500; font-variant-numeric: tabular-nums; letter-spacing: -0.035em;}
#prediction-result p {font-size: 0.93rem; color: #bac4d1; line-height: 1.85; margin-bottom: 0;}
#result-context {color: #aab5c4; font-size: 0.85rem; line-height: 1.8; text-align: center; max-width: 48ch; margin-inline: auto;}
#example-area {border-top: 1px solid #303b49; padding-top: 20px; max-width: 760px; gap: 10px;}
#example-area .prose p {margin: 0; font-size: 0.86rem; color: #aab5c4;}
#evaluation, #about {border-top: 1px solid #303b49; margin-top: clamp(20px, 3vw, 40px);}
.gradio-container .dataset-counts {color: #aab5c4; line-height: 1.8; margin: 0; font-size: 0.9rem;}
.gradio-container .metric-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: #303b49; border: 1px solid #303b49; margin: 0;}
.gradio-container .metric {background: #171e28; padding: clamp(16px, 2.2vw, 26px); min-width: 0;}
.gradio-container .metric dt {color: #bac4d1; font-size: 0.86rem; margin-bottom: 12px;}
.gradio-container .metric dd {margin: 0; color: #f2f5fa; font-size: clamp(1.6rem, 2.5vw, 2.2rem); font-variant-numeric: tabular-nums; line-height: 1.4;}
.gradio-container .metric .unit {font-size: 0.78rem; color: #aab5c4; margin-left: 6px; white-space: nowrap;}
.gradio-container .metric small {display: block; margin-top: 6px; color: #aab5c4; font-size: 0.78rem;}
.gradio-container .metric-primary dt {color: #8bbcff;}
.gradio-container .metric-notes, .gradio-container .about-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 48px;}
.gradio-container .metric-notes p {color: #bac4d1; font-size: 0.9rem; line-height: 1.9; margin: 0;}
.gradio-container .about-grid h3, .gradio-container .guide-title {color: #f2f5fa; font-weight: 500; font-size: 1.05rem; margin: 0 0 12px;}
.gradio-container .about-grid p, .gradio-container .about-grid li {color: #bac4d1; line-height: 1.9; font-size: 0.92rem;}
.gradio-container .about-grid p {margin: 0 0 10px;}
.gradio-container .about-grid ol {padding-left: 24px; margin: 0;}
.gradio-container .input-guide {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 48px; margin: 0;}
.gradio-container .input-guide > div {padding: 16px 0; border-bottom: 1px solid #27313e; min-width: 0;}
.gradio-container .input-guide dt {font-size: 0.9rem; font-weight: 500; color: #edf1f7; margin-bottom: 6px;}
.gradio-container .input-guide dd {margin: 0; color: #aab5c4; font-size: 0.86rem; line-height: 1.85;}
.gradio-container .limitations {border-left: 2px solid #58677b; padding-left: 20px; margin-top: 16px;}
.gradio-container .limitations h3 {font-size: 1rem; font-weight: 500; margin: 0 0 8px; color: #edf1f7;}
.gradio-container .limitations p {color: #aab5c4; font-size: 0.9rem; line-height: 1.85; margin: 0; max-width: 1050px;}
.gradio-container .project-footer {border-top: 1px solid #303b49; margin-top: 32px; padding: 24px 0 8px; color: #8796a9; font-size: 0.8rem; line-height: 1.9;}
.gradio-container .project-footer a {text-decoration: none;}
@media (max-width: 900px) {
    #prediction-layout {grid-template-columns: minmax(0, 1fr); gap: 24px;}
    .gradio-container .metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
}
@media (max-width: 600px) {
    .gradio-container .site-header {gap: 14px; padding-bottom: 16px;}
    .gradio-container .site-nav {width: 100%; justify-content: space-between; gap: 14px;}
    #vehicle-form .field-row, .gradio-container .about-grid,
    .gradio-container .metric-notes, .gradio-container .input-guide {grid-template-columns: minmax(0, 1fr);}
    #prediction-actions {gap: 10px;}
    .gradio-container .metric .unit {display: block; margin: 3px 0 0;}
}
@media (prefers-reduced-motion: no-preference) {html {scroll-behavior: smooth;}}
"""

HELP_JS = """
(async () => {
    let root;
    while (!(root = document.querySelector('.gradio-container'))) {
        await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (root.dataset.fieldHelpReady) return;
    root.dataset.fieldHelpReady = 'true';
    const tooltip = document.createElement('div');
    tooltip.id = 'vehicle-field-tooltip';
    tooltip.className = 'field-help-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.hidden = true;
    root.append(tooltip);
    const entries = new Map();
    let active = null, pinned = false, scheduled = false, nextId = 0;
    const close = () => { tooltip.hidden = true; active = null; pinned = false; };
    const show = (entry) => {
        active = entry;
        tooltip.textContent = entry.info.textContent.trim();
        tooltip.hidden = false;
        const anchor = entry.button.getBoundingClientRect();
        const box = tooltip.getBoundingClientRect();
        tooltip.style.left = Math.max(12, Math.min(anchor.left, innerWidth - box.width - 12)) + 'px';
        const below = anchor.bottom + 8;
        tooltip.style.top = Math.max(12, below + box.height <= innerHeight - 12 ? below : anchor.top - box.height - 8) + 'px';
    };
    const refresh = () => {
        scheduled = false;
        for (const [block, entry] of entries) {
            if (!entry.label.isConnected || !entry.info.isConnected) {
                entry.button.remove(); entries.delete(block);
                if (active === entry) close();
            }
        }
        root.querySelectorAll('#vehicle-form [data-testid="block-info"], #example-selector [data-testid="block-info"]').forEach(label => {
            const info = label.nextElementSibling;
            if (!info || !info.querySelector('.prose')) return;
            const block = label.closest('.block.padded');
            if (!block) return;
            let entry = entries.get(block);
            if (!entry) {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'field-help-button';
                if (!info.id) info.id = 'vehicle-field-help-' + (++nextId);
                button.setAttribute('aria-describedby', info.id);
                const input = block.querySelector('input');
                if (input) input.setAttribute('aria-describedby', info.id);
                entry = {label, info, button};
                button.addEventListener('pointerenter', event => { if (event.pointerType === 'mouse') show(entry); });
                button.addEventListener('pointerleave', () => { if (!pinned && document.activeElement !== button) close(); });
                button.addEventListener('focus', () => { pinned = false; show(entry); });
                button.addEventListener('blur', close);
                button.addEventListener('click', event => {
                    event.preventDefault(); event.stopPropagation();
                    if (pinned && active === entry) close();
                    else { pinned = true; show(entry); }
                });
                entries.set(block, entry);
                label.classList.add('help-label');
                info.classList.add('field-help-source');
                block.append(button);
            }
            entry.button.setAttribute('aria-label', 'ข้อมูลเพิ่มเติม: ' + label.textContent.trim());
            const area = block.getBoundingClientRect(), heading = label.getBoundingClientRect();
            const range = document.createRange();
            range.selectNodeContents(label);
            const text = range.getBoundingClientRect();
            entry.button.style.top = (heading.top - area.top) + 'px';
            entry.button.style.left = (Math.min(text.right + 4, heading.right - 30) - area.left) + 'px';
        });
    };
    const schedule = () => { if (!scheduled) { scheduled = true; requestAnimationFrame(refresh); } };
    new MutationObserver(schedule).observe(root, {childList: true, subtree: true, characterData: true});
    new ResizeObserver(schedule).observe(root);
    document.addEventListener('pointerdown', event => { if (active && !active.button.contains(event.target)) close(); }, true);
    document.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
    window.addEventListener('scroll', () => {
        if (!active) return;
        const anchor = active.button.getBoundingClientRect();
        if (anchor.bottom < 0 || anchor.top > innerHeight) close();
        else if (pinned || document.activeElement === active.button) show(active);
        else close();
    }, true);
    window.addEventListener('resize', () => { close(); schedule(); });
    refresh();
})();
"""

# Preprocessing is embedded in this artifact. Never fit or reconstruct it here.
model = joblib.load(MODEL_PATH)
if list(model.feature_names_in_) != FEATURES:
    raise ValueError("Saved model input columns do not match the application schema.")


def load_metrics():
    """Read the recorded evaluation, failing clearly on missing/invalid artifacts."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for key in ("mae", "mse", "rmse", "r2"):
        value = metrics.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (key != "r2" and value < 0)):
            raise ValueError(f"metrics.json must contain a finite evaluation value: {key}")
    for key in ("rows", "train_rows", "test_rows"):
        if key in metrics and (type(metrics[key]) is not int or metrics[key] <= 0):
            raise ValueError(f"metrics.json must contain a positive row count: {key}")
    return metrics


def validate_inputs(values):
    """Validate raw values without silently rounding discrete features."""
    data = {}
    for name, raw in zip(FEATURES, values, strict=True):
        if raw is None or isinstance(raw, bool) or (isinstance(raw, str) and not raw.strip()):
            raise gr.Error(f"{name}: กรุณากรอกตัวเลข / Please enter a number.")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise gr.Error(f"{name}: ต้องเป็นตัวเลข / Must be numeric.") from exc
        if not math.isfinite(value):
            raise gr.Error(f"{name}: ต้องเป็นตัวเลขที่มีค่าจำกัด / Must be finite.")
        if name in ("cylinders", "model_year", "origin"):
            if not value.is_integer():
                raise gr.Error(f"{name}: ต้องเป็นจำนวนเต็ม / Must be a whole number.")
            value = int(value)
        if name == "cylinders" and value not in CYLINDERS:
            raise gr.Error("cylinders: เลือก 3, 4, 5, 6 หรือ 8 ตามชุดข้อมูล / Choose 3, 4, 5, 6 or 8.")
        if name == "model_year" and not 0 <= value <= 99:
            raise gr.Error("model_year: ใช้รหัสปี 0–99 เช่น 76 = 1976 / Use a two-digit year, not 1976.")
        if name == "origin" and value not in (1, 2, 3):
            raise gr.Error("origin: เลือกรหัส 1, 2 หรือ 3 / Choose origin code 1, 2 or 3.")
        if name in ("displacement", "horsepower", "weight", "acceleration") and value <= 0:
            raise gr.Error(f"{name}: ต้องมากกว่า 0 / Must be greater than zero.")
        data[name] = value
    return data


def predict_mpg_value(cylinders, displacement, horsepower, weight,
                      acceleration, model_year, origin):
    """The original model call in its original units, without display rounding."""
    data = validate_inputs((cylinders, displacement, horsepower, weight,
                            acceleration, model_year, origin))
    try:
        prediction = float(model.predict(pd.DataFrame([data], columns=FEATURES))[0])
    except (ValueError, TypeError, OverflowError) as exc:
        raise gr.Error("ไม่สามารถทำนายจากค่าที่กรอกได้ กรุณาตรวจสอบข้อมูล / Please check the input values.") from exc
    if not math.isfinite(prediction):
        raise gr.Error("โมเดลไม่สามารถให้ผลลัพธ์ที่มีค่าจำกัด / The model could not produce a finite prediction.")
    return prediction


def predict_mpg(cylinders, displacement, horsepower, weight,
                acceleration, model_year, origin):
    prediction = predict_mpg_value(cylinders, displacement, horsepower, weight,
                                   acceleration, model_year, origin)
    return f"{prediction:.2f} MPG"


def predict_for_display(cylinders, displacement, horsepower, weight,
                        acceleration, model_year, origin):
    result = predict_mpg(cylinders, displacement, horsepower, weight,
                         acceleration, model_year, origin)
    note = ""
    if not 70 <= float(model_year) <= 82:
        note += "\n\nนอกช่วงข้อมูล (Extrapolation): โมเดลอ้างอิงรถปี 1970–1982 ค่าประมาณสำหรับปีอื่นควรตีความอย่างระมัดระวัง"
    if float(result.split()[0]) <= 0:
        note += "\n\nผลไม่สมเหตุสมผล (Non-physical result): โมเดลให้ MPG ที่ไม่เป็นบวกสำหรับข้อมูลชุดนี้ จึงไม่ควรใช้ค่านี้ประมาณรถจริง"
    return (f"### Estimated fuel economy\n\n# {result}\n\n"
            "ค่าประมาณระยะทางเป็นไมล์ต่อน้ำมันหนึ่งแกลลอน"
            + note)


def reset_inputs():
    return (*DEFAULT_INPUTS, EMPTY_RESULT)


def load_example(index):
    """Fill the unchanged seven-feature example and clear the previous result."""
    if index is None:
        return tuple(gr.skip() for _ in range(len(FEATURES) + 1))
    if index not in range(len(EXAMPLES)):
        raise gr.Error("กรุณาเลือกตัวอย่างรถจากรายการ")
    return (*EXAMPLES[int(index)], EMPTY_RESULT)


def reset_form():
    """Reset the input values, result, and example selector together."""
    return (*reset_inputs(), None)


def finite_number(value):
    if value is None or isinstance(value, bool):
        raise gr.Error("กรุณากรอกตัวเลขที่มีค่าจำกัด")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error("กรุณากรอกตัวเลขที่มีค่าจำกัด") from exc
    if not math.isfinite(value):
        raise gr.Error("กรุณากรอกตัวเลขที่มีค่าจำกัด")
    return value


def convert_input(value, feature, source_unit, target_unit):
    """Convert measurement units only; the saved Pipeline handles preprocessing."""
    value = finite_number(value)
    factors = INPUT_UNIT_FACTORS.get(feature, {})
    if source_unit not in factors or target_unit not in factors:
        raise gr.Error("หน่วยนำเข้าไม่ถูกต้อง")
    converted = value if source_unit == target_unit else value / factors[source_unit] * factors[target_unit]
    return finite_number(converted)


def convert_fuel_economy(value, source_unit, target_unit):
    """Convert positive fuel economy using US liquid gallons, never Imperial gallons."""
    value = finite_number(value)
    if source_unit not in OUTPUT_UNITS or target_unit not in OUTPUT_UNITS:
        raise gr.Error("หน่วยผลลัพธ์ไม่ถูกต้อง")
    if value <= 0:
        raise gr.Error("ค่าประหยัดเชื้อเพลิงต้องมากกว่า 0 จึงจะแปลงหน่วยได้")
    try:
        mpg = value if source_unit == "US MPG" else (
            value / KM_PER_L_PER_US_MPG if source_unit == "km/L"
            else 100 / (value * KM_PER_L_PER_US_MPG)
        )
        converted = mpg if target_unit == "US MPG" else (
            mpg * KM_PER_L_PER_US_MPG if target_unit == "km/L"
            else 100 / (mpg * KM_PER_L_PER_US_MPG)
        )
    except (ZeroDivisionError, OverflowError) as exc:
        raise gr.Error("ค่าผลลัพธ์อยู่นอกช่วงที่แปลงหน่วยได้") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise gr.Error("ค่าผลลัพธ์อยู่นอกช่วงที่แปลงหน่วยได้")
    return converted


def measurement_system(units):
    if tuple(units) == ORIGINAL_UNITS:
        return "US"
    if units[0] in ("L", "cc") and tuple(units[1:]) == ("kW", "kg"):
        return "Metric"
    return "Custom"


def make_unit_state(model_values=None, units=None):
    """Store original quantities; only the visible numeric values are rounded."""
    values = list(DEFAULT_INPUTS[1:4] if model_values is None else model_values)
    units = list(UNIT_PRESETS["Metric"] if units is None else units)
    if len(values) != 3 or len(units) != 3 or any(
            unit not in INPUT_UNIT_FACTORS[feature] for feature, unit in zip(CONVERTIBLE_FEATURES, units)):
        raise gr.Error("หน่วยนำเข้าไม่ถูกต้อง")
    displayed = []
    for feature, original, unit, value in zip(CONVERTIBLE_FEATURES, ORIGINAL_UNITS, units, values, strict=True):
        converted = None if value is None else convert_input(value, feature, original, unit)
        rounded = None if converted is None else round(converted, 6)
        # Do not turn a tiny nonzero measurement into a displayed zero.
        displayed.append(converted if converted and rounded == 0 else rounded)
    return {"units": units, "model_values": values, "display_values": displayed,
            "system": measurement_system(units)}


def sync_unit_state(displacement, horsepower, weight, state):
    """Keep exact canonical quantities unless the user changes a displayed value."""
    state = deepcopy(state) if state is not None else make_unit_state()
    for index, (feature, original, value) in enumerate(zip(
            CONVERTIBLE_FEATURES, ORIGINAL_UNITS, (displacement, horsepower, weight), strict=True)):
        if value is None:
            state["model_values"][index] = None
        else:
            value = finite_number(value)
            if value != state["display_values"][index]:
                state["model_values"][index] = convert_input(value, feature, state["units"][index], original)
        state["display_values"][index] = value
    return state


def unit_field_updates(state):
    displacement_unit, power_unit, weight_unit = state["units"]
    return (
        gr.update(value=state["display_values"][0], label=f"ปริมาตรกระบอกสูบ ({displacement_unit})",
                  info="ขนาดเครื่องยนต์ ไม่ใช่กำลังเครื่องยนต์"),
        gr.update(value=state["display_values"][1], label=f"กำลังเครื่องยนต์ ({power_unit})",
                  info="hp หมายถึง mechanical horsepower"),
        gr.update(value=state["display_values"][2], label=f"น้ำหนักรถ ({weight_unit})",
                  info="กิโลกรัม (kg) หรือปอนด์ (lb)"),
    )


def update_unit_controls(system, displacement_unit, power_unit, weight_unit,
                         displacement, horsepower, weight, state, event: gr.EventData = None):
    state = state if state is not None else make_unit_state()
    if system not in ("Metric", "US", "Custom"):
        raise gr.Error("ระบบหน่วยไม่ถูกต้อง")
    selectors = {"measurement-system": 0, "displacement-unit": 1, "power-unit": 2, "weight-unit": 3}
    selector = selectors.get(event.target.elem_id) if event is not None and event.target is not None else None
    selected = (system, displacement_unit, power_unit, weight_unit)
    if selector is not None:
        # Gradio 6 emits input on both selection and blur. A queued duplicate
        # carries old sibling controls/numbers, so skip BEFORE reading them.
        if selected[selector] == (state["system"], *state["units"])[selector]:
            return tuple(gr.skip() for _ in range(8))
        units = list(state["units"])
        if selector == 0:
            units = UNIT_PRESETS.get(system, units)
        else:
            units[selector - 1] = selected[selector]
    else:
        # Retain the existing direct-call/API behavior when no trigger is given.
        units = UNIT_PRESETS[system] if system != state["system"] and system in UNIT_PRESETS else (
            displacement_unit, power_unit, weight_unit)
    state = sync_unit_state(displacement, horsepower, weight, state)
    state = make_unit_state(state["model_values"], units)
    if selector == 0 and system == "Custom":
        state["system"] = "Custom"
    return (state["system"], *state["units"], *unit_field_updates(state), state)


def predict_with_units(cylinders, displacement, horsepower, weight, acceleration, model_year, origin,
                       displacement_unit="L", power_unit="kW", weight_unit="kg"):
    converted = [convert_input(value, feature, unit, original) for feature, unit, original, value in zip(
        CONVERTIBLE_FEATURES, (displacement_unit, power_unit, weight_unit), ORIGINAL_UNITS,
        (displacement, horsepower, weight), strict=True)]
    return predict_mpg_value(cylinders, *converted, acceleration, model_year, origin)


def fuel_display_number(value):
    return f"{value:.2f}" if 0.005 <= abs(value) < 1_000_000 else f"{value:.3g}"


def render_instrument(prediction=None, output_unit=DEFAULT_OUTPUT_UNIT):
    if output_unit not in OUTPUT_UNITS:
        raise gr.Error("หน่วยผลลัพธ์ไม่ถูกต้อง")
    number, equivalents, note = "—", "", "กรอกข้อมูลรถแล้วกดทำนาย"
    if prediction is not None:
        mpg = finite_number(prediction["mpg"])
        if mpg <= 0:
            note = (f"ผลไม่สมเหตุสมผล (Non-physical result): โมเดลให้ {fuel_display_number(mpg)} US MPG "
                    "จึงไม่แปลงหน่วยหรือใช้ค่านี้ประมาณรถจริง")
        else:
            values = {unit: convert_fuel_economy(mpg, "US MPG", unit) for unit in OUTPUT_UNITS}
            number = fuel_display_number(values[output_unit])
            equivalents = '<dl class="fuel-equivalents">' + ''.join(
                f'<div><dt>{unit}</dt><dd>{fuel_display_number(value)}</dd></div>'
                for unit, value in values.items()) + '</dl>'
            note = "ค่าประมาณจากโมเดล หน่วย MPG ใช้ US liquid gallon ไม่ใช่ UK Imperial gallon"
        if not 70 <= prediction["model_year"] <= 82:
            note += " · นอกช่วงข้อมูล (Extrapolation): โมเดลอ้างอิงรถปี 1970–1982"
    return (f'<div class="instrument" role="status" aria-live="polite">'
            f'<div class="instrument-heading">{CAR_SVG}<p>Estimated fuel economy</p></div>'
            f'<div class="instrument-reading"><span class="instrument-value">{number}</span>'
            f'<span class="instrument-unit">{output_unit}</span></div>'
            f'<p class="instrument-note">{note}</p>{equivalents}</div>')


def predict_vehicle(cylinders, displacement, horsepower, weight, acceleration, model_year, origin,
                    output_unit, state):
    state = sync_unit_state(displacement, horsepower, weight, state)
    mpg = predict_mpg_value(cylinders, *state["model_values"], acceleration, model_year, origin)
    prediction = {"mpg": mpg, "model_year": int(model_year)}
    return render_instrument(prediction, output_unit), state, prediction


def update_output_unit(output_unit, prediction):
    return render_instrument(prediction, output_unit)


def clear_vehicle_result(output_unit):
    return render_instrument(output_unit=output_unit), None


def load_vehicle_example(index, state, output_unit):
    if index is None:
        return tuple(gr.skip() for _ in range(10))
    values = list(load_example(index)[:7])
    state = make_unit_state(values[1:4], state["units"])
    values[1:4] = state["display_values"]
    return (*values, render_instrument(output_unit=output_unit), state, None)


def reset_vehicle_form(state, output_unit):
    values = list(reset_inputs()[:7])
    state = make_unit_state(values[1:4], state["units"])
    values[1:4] = state["display_values"]
    return (*values, render_instrument(output_unit=output_unit), None, state, None)


def build_theme():
    """Use the same graphite palette in both browser color modes."""
    colors = {
        "body_background_fill": "#0f141b",
        "body_text_color": "#edf1f7",
        "body_text_color_subdued": "#aab5c4",
        "background_fill_primary": "#171e28",
        "background_fill_secondary": "#202a37",
        "border_color_primary": "#303b49",
        "border_color_accent": "#72b3ff",
        "border_color_accent_subdued": "#3f5d83",
        "block_background_fill": "transparent",
        "block_border_color": "transparent",
        "block_label_background_fill": "transparent",
        "block_label_text_color": "#d3dbe6",
        "block_info_text_color": "#aab5c4",
        "block_title_background_fill": "transparent",
        "block_title_text_color": "#edf1f7",
        "block_shadow": "none",
        "input_background_fill": "#171e28",
        "input_background_fill_hover": "#1d2632",
        "input_background_fill_focus": "#171e28",
        "input_border_color": "#3d4a5e",
        "input_border_color_hover": "#65758d",
        "input_border_color_focus": "#72b3ff",
        "input_placeholder_color": "#aab5c4",
        "input_shadow": "none",
        "input_shadow_focus": "none",
        "button_primary_background_fill": "#3064a5",
        "button_primary_background_fill_hover": "#3a74bb",
        "button_primary_text_color": "#ffffff",
        "button_primary_text_color_hover": "#ffffff",
        "button_primary_border_color": "#487ebf",
        "button_primary_border_color_hover": "#72b3ff",
        "button_secondary_background_fill": "#202a37",
        "button_secondary_background_fill_hover": "#2a3749",
        "button_secondary_text_color": "#edf1f7",
        "button_secondary_text_color_hover": "#ffffff",
        "button_secondary_border_color": "#3d4a5e",
        "button_secondary_border_color_hover": "#65758d",
        "link_text_color": "#8bbcff",
        "link_text_color_hover": "#b6d6ff",
    }
    settings = {key + suffix: value for key, value in colors.items() for suffix in ("", "_dark")}
    return gr.themes.Base(
        primary_hue="blue", secondary_hue="slate", neutral_hue="slate", radius_size="sm",
        font=["Tahoma", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"],
    ).set(**settings, block_padding="0px", block_label_padding="0px",
          block_label_radius="0px", block_label_text_weight="400",
          block_border_width="0px", block_border_width_dark="0px",
          input_radius="4px", input_padding="12px", button_border_width="1px",
          button_border_width_dark="1px", button_large_radius="4px")


def build_app():
    metrics = load_metrics()
    initial_units = make_unit_state()
    counts = " / ".join(
        f"{label} {metrics[key]:,} รายการ"
        for key, label in [("rows", "ข้อมูลทั้งหมด"), ("train_rows", "Train"), ("test_rows", "Test")]
        if key in metrics
    )
    metric_cells = []
    for key, label, unit in [("mae", "MAE", "MPG"), ("mse", "MSE", "MPG²"),
                             ("rmse", "RMSE", "MPG"), ("r2", "R²", "")]:
        primary = " metric-primary" if key == "rmse" else ""
        note = "<small>เมตริกหลักของโปรเจกต์</small>" if key == "rmse" else ""
        metric_cells.append(
            f'<div class="metric{primary}"><dt>{label}</dt>'
            f'<dd>{metrics[key]:.3f}<span class="unit">{unit}</span></dd>{note}</div>'
        )
    with gr.Blocks(title="Auto MPG Regression", analytics_enabled=False) as app:
        unit_state = gr.State(initial_units)
        prediction_state = gr.State(None)
        gr.HTML("""
            <header class="site-header" id="top">
                <div class="brand">
                    <a href="#top">Auto MPG Regression</a>
                    <span>Vehicle fuel economy estimator</span>
                </div>
                <nav class="site-nav" aria-label="ส่วนต่าง ๆ ของหน้า">
                    <a href="#prediction">Prediction</a>
                    <a href="#evaluation">Evaluation</a>
                    <a href="#about">About</a>
                </nav>
            </header>
            """, elem_id="site-header", apply_default_css=False)
        gr.HTML("""
            <div class="hero">
                <h1>ประเมินความประหยัดเชื้อเพลิงของรถ</h1>
                <p>กรอกคุณลักษณะรถเพื่อประมาณค่า MPG ด้วย Linear Regression
                จากชุดข้อมูล Auto MPG แล้วดูผลประเมินและขอบเขตของโมเดลในหน้าเดียว</p>
            </div>
            """, apply_default_css=False)
        with gr.Column(elem_id="prediction", elem_classes="page-section", min_width=0):
            gr.HTML('<div class="section-heading"><h2>ทำนาย MPG</h2>'
                    '<p>เริ่มจากข้อมูลรถของคุณ หรือเลือกตัวอย่างด้านล่าง</p></div>',
                    apply_default_css=False)
            with gr.Row(elem_id="prediction-layout"):
                with gr.Column(elem_id="vehicle-form", min_width=0):
                    with gr.Row(elem_id="unit-controls"):
                        with gr.Column(min_width=0):
                            system = gr.Dropdown(["Metric", "US", "Custom"], value="Metric", min_width=0,
                                                 label="ระบบหน่วย", filterable=False, elem_id="measurement-system")
                        with gr.Column(min_width=0):
                            displacement_unit = gr.Dropdown(["L", "cc", "cu in"], value="L", min_width=0,
                                                            label="ปริมาตร", filterable=False, elem_id="displacement-unit")
                        with gr.Column(min_width=0):
                            power_unit = gr.Dropdown(["kW", "hp"], value="kW", min_width=0,
                                                     label="กำลัง", filterable=False, elem_id="power-unit")
                        with gr.Column(min_width=0):
                            weight_unit = gr.Dropdown(["kg", "lb"], value="kg", min_width=0,
                                                      label="น้ำหนัก", filterable=False, elem_id="weight-unit")
                    with gr.Row(elem_classes="field-row"):
                        with gr.Column(min_width=0):
                            cylinders = gr.Dropdown(CYLINDERS, value=DEFAULT_INPUTS[0],
                                                    label="กระบอกสูบ (Cylinders)", info="จำนวนกระบอกสูบของเครื่องยนต์")
                        with gr.Column(min_width=0):
                            displacement = gr.Number(value=initial_units["display_values"][0], label="ปริมาตรกระบอกสูบ (L)",
                                                     info="ขนาดเครื่องยนต์ ไม่ใช่กำลังเครื่องยนต์")
                    with gr.Row(elem_classes="field-row"):
                        with gr.Column(min_width=0):
                            horsepower = gr.Number(value=initial_units["display_values"][1], label="กำลังเครื่องยนต์ (kW)",
                                                   info="hp หมายถึง mechanical horsepower")
                        with gr.Column(min_width=0):
                            weight = gr.Number(value=initial_units["display_values"][2], label="น้ำหนักรถ (kg)",
                                               info="กิโลกรัม (kg) หรือปอนด์ (lb)")
                    with gr.Row(elem_classes="field-row"):
                        with gr.Column(min_width=0):
                            acceleration = gr.Number(value=DEFAULT_INPUTS[4], label="Acceleration (s)",
                                                     info="เวลาเร่งจาก 0 ถึง 60 mph เป็นวินาที")
                        with gr.Column(min_width=0):
                            model_year = gr.Number(value=DEFAULT_INPUTS[5], label="ปีรุ่น (Model year)",
                                                   info="70 = 1970, 76 = 1976, 82 = 1982", step=1)
                    origin = gr.Dropdown(ORIGINS, value=DEFAULT_INPUTS[6], label="แหล่งกำเนิด (Origin)",
                                         info="1 = USA, 2 = Europe, 3 = Japan")
                    with gr.Row(elem_id="prediction-actions"):
                        submit = gr.Button("ทำนาย MPG", variant="primary")
                        reset = gr.Button("คืนค่าเริ่มต้น")
                with gr.Column(min_width=0):
                    output_unit = gr.Dropdown(OUTPUT_UNITS, value=DEFAULT_OUTPUT_UNIT, label="หน่วยผลลัพธ์",
                                              filterable=False, min_width=0, elem_id="output-unit")
                    result = gr.HTML(render_instrument(), elem_id="prediction-result", apply_default_css=False)
                    gr.Markdown("US MPG และ km/L ยิ่งสูงยิ่งประหยัด ส่วน L/100 km ยิ่งต่ำยิ่งประหยัด",
                                elem_id="result-context")
            with gr.Column(elem_id="example-area", min_width=0):
                example = gr.Dropdown(choices=[(name, index) for index, name in enumerate(EXAMPLE_NAMES)],
                                      value=None, label="ตัวอย่างรถจาก Auto MPG",
                                      info="เลือกเพื่อเติมข้อมูล แล้วกดทำนาย MPG", filterable=False,
                                      elem_id="example-selector")
                gr.Markdown("ค่าต่อเนื่องนอกช่วงที่พบในข้อมูลยังกรอกได้ แต่ค่าที่ต่างจากข้อมูลมากอาจให้ผลที่ไม่น่าเชื่อถือ")
            inputs = [cylinders, displacement, horsepower, weight, acceleration, model_year, origin]
            submit.click(predict_vehicle, inputs=[*inputs, output_unit, unit_state],
                         outputs=[result, unit_state, prediction_state], api_name="predict_mpg",
                         concurrency_id="vehicle_units", concurrency_limit=1)
            reset.click(reset_vehicle_form, inputs=[unit_state, output_unit],
                        outputs=[*inputs, result, example, unit_state, prediction_state],
                        api_name="reset_form", api_visibility="private",
                        concurrency_id="vehicle_units", concurrency_limit=1)
            example.input(load_vehicle_example, inputs=[example, unit_state, output_unit],
                          outputs=[*inputs, result, unit_state, prediction_state],
                          api_name="load_example", api_visibility="private",
                          concurrency_id="vehicle_units", concurrency_limit=1)
            gr.on(triggers=[control.input for control in (system, displacement_unit, power_unit, weight_unit)],
                  fn=update_unit_controls,
                  inputs=[system, displacement_unit, power_unit, weight_unit, displacement, horsepower, weight, unit_state],
                  outputs=[system, displacement_unit, power_unit, weight_unit, displacement, horsepower, weight, unit_state],
                  api_name="switch_units", api_visibility="private", trigger_mode="once",
                  concurrency_id="vehicle_units", concurrency_limit=1)
            output_unit.input(update_output_unit, inputs=[output_unit, prediction_state], outputs=result,
                              api_name="output_unit", api_visibility="private",
                              concurrency_id="vehicle_units", concurrency_limit=1)
            for component in inputs:
                component.input(clear_vehicle_result, inputs=output_unit, outputs=[result, prediction_state],
                                api_visibility="private", concurrency_id="vehicle_units", concurrency_limit=1)
        with gr.Column(elem_id="evaluation", elem_classes="page-section", min_width=0):
            gr.HTML('<div class="section-heading"><h2>ผลประเมินโมเดล</h2>'
                    '<p>ผลบนชุดทดสอบที่ไม่ได้ใช้ฝึก</p></div>', apply_default_css=False)
            gr.HTML(f'<p class="dataset-counts">{counts}</p>', apply_default_css=False)
            gr.HTML('<dl class="metric-grid">' + ''.join(metric_cells) + '</dl>',
                    elem_id="evaluation-metrics", apply_default_css=False)
            gr.HTML(f"""
                <div class="metric-notes">
                    <p>RMSE {metrics['rmse']:.3f} MPG เป็นเมตริกหลัก สรุปขนาดความคลาดเคลื่อนบนชุดทดสอบ
                    และให้น้ำหนักกับความผิดพลาดขนาดใหญ่ ไม่ใช่ช่วงความเชื่อมั่นของรถแต่ละคัน</p>
                    <p>R² {metrics['r2']:.3f} วัดความแปรปรวนที่โมเดลอธิบายได้เทียบกับค่าเฉลี่ย
                    ไม่มีหน่วย และไม่ใช่เปอร์เซ็นต์ความแม่นยำในการจำแนกประเภท</p>
                    <p>MAE คือค่าเฉลี่ยขนาดความผิดพลาด มีหน่วย MPG ส่วน MSE
                    คือค่าเฉลี่ยความผิดพลาดยกกำลังสอง มีหน่วย MPG² ค่าต่ำกว่าหมายถึงความคลาดเคลื่อนน้อยลง</p>
                    <p>แบ่งข้อมูล Train/Test 80%/20% ด้วย random_state=42
                    ขั้นตอนเตรียมข้อมูลเรียนรู้จาก Train เท่านั้น คะแนนที่แสดงเป็นผลของโมเดลเดิม
                    หน่วย MPG ในผลประเมินเป็น US MPG และไม่เปลี่ยนตามหน่วยแสดงผลทำนาย</p>
                </div>
                """, apply_default_css=False)
        with gr.Column(elem_id="about", elem_classes="page-section", min_width=0):
            gr.HTML('<div class="section-heading"><h2>เกี่ยวกับและวิธีใช้งาน</h2>'
                    '<p>เข้าใจข้อมูลก่อนนำค่าประมาณไปใช้</p></div>', apply_default_css=False)
            gr.HTML(f"""
                <div class="about-grid">
                    <article>
                        <h3>โมเดลนี้ทำอะไร</h3>
                        <p>Linear Regression ประมาณอัตราการเดินทางต่อเชื้อเพลิงในหน่วย miles per gallon (MPG)
                        จากคุณลักษณะรถ 7 ค่า โดยไม่ใช้ชื่อรถเป็นตัวแปร</p>
                        <p>ใช้ชุดข้อมูล Auto MPG {metrics.get('rows', 398):,} รายการ
                        เว็บเรียกใช้โมเดลที่บันทึกไว้พร้อมการเติมค่าด้วย median การปรับสเกลตัวเลข
                        และการเข้ารหัส Origin ไม่ฝึกโมเดลใหม่เมื่อเริ่มทำงาน</p>
                    </article>
                    <article>
                        <h3>วิธีใช้งาน</h3>
                        <ol>
                            <li>เลือกระบบหน่วย แล้วกรอกข้อมูลทั้ง 7 ค่า หรือเลือกตัวอย่างรถ</li>
                            <li>ตรวจหน่วยและรหัสปี จากนั้นกดทำนาย MPG</li>
                            <li>เลือกผลเป็น US MPG, km/L หรือ L/100 km แล้วดูผลประเมินประกอบ</li>
                        </ol>
                        <p>การเปลี่ยนหน่วยคงข้อมูลรถเดิมไว้ เมื่อแก้ค่าตัวเลข ผลเดิมจะถูกล้าง
                        ปุ่มคืนค่าเริ่มต้นคืนข้อมูลเริ่มต้นตามหน่วยที่เลือก</p>
                    </article>
                </div>
                """, apply_default_css=False)
            gr.HTML("""
                <h3 class="guide-title">คู่มือข้อมูลรถ</h3>
                <dl class="input-guide">
                    <div><dt>Cylinders</dt><dd>จำนวนกระบอกสูบที่พบในข้อมูล: 3, 4, 5, 6 และ 8</dd></div>
                    <div><dt>Displacement (L / cc / cu in)</dt><dd>ปริมาตรกระบอกสูบเป็นลิตร ลูกบาศก์เซนติเมตร
                    หรือลูกบาศก์นิ้ว เป็นขนาดเครื่องยนต์ ไม่ใช่กำลังเครื่องยนต์ โมเดลรับ cu in</dd></div>
                    <div><dt>Engine power (kW / hp)</dt><dd>กำลังเครื่องยนต์เป็นกิโลวัตต์หรือ mechanical horsepower
                    โมเดลรับ hp (ไม่ใช่ metric horsepower / PS) ช่วงข้อมูล 46–230 hp
                    เว็บขอให้กรอกค่าที่ทราบให้ครบ</dd></div>
                    <div><dt>Weight (kg / lb)</dt><dd>น้ำหนักรถเป็นกิโลกรัมหรือปอนด์
                    โมเดลรับ lb ช่วงข้อมูล 1,613–5,140 lb</dd></div>
                    <div><dt>Acceleration (s)</dt><dd>เวลาเร่งจาก 0 ถึง 60 mph เป็นวินาที
                    ช่วงข้อมูล 8.0–24.8 ค่าน้อยหมายถึงใช้เวลาเร่งน้อยลง</dd></div>
                    <div><dt>Model year</dt><dd>รหัสปีรุ่น: 70 = 1970, 76 = 1976, 82 = 1982
                    ข้อมูลที่ใช้สร้างโมเดลมีปีรุ่น 70–82</dd></div>
                    <div><dt>Origin</dt><dd>แหล่งกำเนิดรถ: 1 = USA, 2 = Europe, 3 = Japan
                    แอปส่งรหัสตัวเลขเดิมให้โมเดล</dd></div>
                </dl>
                <aside class="limitations">
                    <h3>ขอบเขตของค่าประมาณ</h3>
                    <p>โครงงาน ML เพื่อการศึกษา ใช้ข้อมูลรถรุ่นเก่าปี 1970–1982
                    Linear Regression อาจอธิบายความสัมพันธ์ที่ไม่เป็นเส้นตรงได้ไม่ครบ
                    ค่าทำนายบางชุดอาจไม่สมเหตุสมผล และไม่ใช่ข้อมูลรับรองความประหยัดเชื้อเพลิงของรถรุ่นปัจจุบัน</p>
                </aside>
                """, apply_default_css=False)
        gr.HTML("""
            <footer class="project-footer">
                Auto MPG Regression — โครงงาน Machine Learning เพื่อการศึกษา<br>
                ข้อมูล: <a href="https://archive.ics.uci.edu/dataset/9/auto+mpg">UCI Auto MPG</a> (CC BY 4.0)
                / <a href="https://doi.org/10.24432/C5859H">DOI</a>
                / <a href="https://cran.r-project.org/web/packages/ISLR/ISLR.pdf#page=2">ความหมายและหน่วยของตัวแปร</a>
            </footer>
            """, apply_default_css=False)
    return app


def main():
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        theme=build_theme(),
        css=CSS,
        js=HELP_JS,
    )


if __name__ == "__main__":
    main()
