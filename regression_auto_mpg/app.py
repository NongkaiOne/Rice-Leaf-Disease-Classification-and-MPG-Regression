"""Auto MPG web application: inference with the existing fitted Pipeline only."""

import json
import math
import os
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
    border-left: 3px solid #659deb;
    border-radius: 4px;
    padding: clamp(20px, 3vw, 34px);
    background: #171e28;
    width: 100%;
}
#prediction-result h3 {font-size: 0.95rem; color: #aab5c4; font-weight: 400; margin: 0;}
#prediction-result h1 {font-size: clamp(2.7rem, 5vw, 4rem); color: #f2f5fa; line-height: 1.15; margin: 24px 0; font-weight: 500; font-variant-numeric: tabular-nums; letter-spacing: -0.035em;}
#prediction-result p {font-size: 0.93rem; color: #bac4d1; line-height: 1.85; margin-bottom: 0;}
#result-context {color: #aab5c4; font-size: 0.85rem; line-height: 1.8;}
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
    .gradio-container .field-row, .gradio-container .about-grid,
    .gradio-container .metric-notes, .gradio-container .input-guide {grid-template-columns: minmax(0, 1fr);}
    #prediction-actions {gap: 10px;}
    .gradio-container .metric .unit {display: block; margin: 3px 0 0;}
}
@media (prefers-reduced-motion: no-preference) {html {scroll-behavior: smooth;}}
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


def predict_mpg(cylinders, displacement, horsepower, weight,
                acceleration, model_year, origin):
    data = validate_inputs((cylinders, displacement, horsepower, weight,
                            acceleration, model_year, origin))
    try:
        prediction = float(model.predict(pd.DataFrame([data], columns=FEATURES))[0])
    except (ValueError, TypeError, OverflowError) as exc:
        raise gr.Error("ไม่สามารถทำนายจากค่าที่กรอกได้ กรุณาตรวจสอบข้อมูล / Please check the input values.") from exc
    if not math.isfinite(prediction):
        raise gr.Error("โมเดลไม่สามารถให้ผลลัพธ์ที่มีค่าจำกัด / The model could not produce a finite prediction.")
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
                    with gr.Row(elem_classes="field-row"):
                        with gr.Column(min_width=0):
                            cylinders = gr.Dropdown(CYLINDERS, value=DEFAULT_INPUTS[0],
                                                    label="กระบอกสูบ (Cylinders)", info="จำนวนกระบอกสูบของเครื่องยนต์")
                        with gr.Column(min_width=0):
                            displacement = gr.Number(value=DEFAULT_INPUTS[1], label="ปริมาตรกระบอกสูบ (cu in)",
                                                     info="ขนาดเครื่องยนต์เป็นลูกบาศก์นิ้ว ไม่ใช่แรงม้า")
                    with gr.Row(elem_classes="field-row"):
                        with gr.Column(min_width=0):
                            horsepower = gr.Number(value=DEFAULT_INPUTS[2], label="แรงม้า (Horsepower, hp)",
                                                   info="กำลังเครื่องยนต์ หน่วยแรงม้า")
                        with gr.Column(min_width=0):
                            weight = gr.Number(value=DEFAULT_INPUTS[3], label="น้ำหนักรถ (lb)",
                                               info="น้ำหนักเป็นปอนด์ ไม่ใช่กิโลกรัม")
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
                    result = gr.Markdown(EMPTY_RESULT, elem_id="prediction-result")
                    gr.Markdown("MPG คือ miles per gallon ค่ายิ่งสูง ยิ่งเดินทางได้ไกลต่อน้ำมันหนึ่งแกลลอน",
                                elem_id="result-context")
            with gr.Column(elem_id="example-area", min_width=0):
                example = gr.Dropdown(choices=[(name, index) for index, name in enumerate(EXAMPLE_NAMES)],
                                      value=None, label="ตัวอย่างรถจาก Auto MPG",
                                      info="เลือกเพื่อเติมข้อมูล แล้วกดทำนาย MPG", filterable=False,
                                      elem_id="example-selector")
                gr.Markdown("ค่าต่อเนื่องนอกช่วงที่พบในข้อมูลยังกรอกได้ แต่ค่าที่ต่างจากข้อมูลมากอาจให้ผลที่ไม่น่าเชื่อถือ")
            inputs = [cylinders, displacement, horsepower, weight, acceleration, model_year, origin]
            submit.click(predict_for_display, inputs=inputs, outputs=result, api_name="predict_mpg")
            reset.click(reset_form, outputs=[*inputs, result, example], queue=False,
                        api_name="reset_form", api_visibility="private")
            example.input(load_example, inputs=example, outputs=[*inputs, result], queue=False,
                          api_name="load_example", api_visibility="private")
            for component in inputs:
                component.change(lambda: EMPTY_RESULT, outputs=result, queue=False, api_visibility="private")
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
                    ขั้นตอนเตรียมข้อมูลเรียนรู้จาก Train เท่านั้น คะแนนที่แสดงเป็นผลของโมเดลเดิม</p>
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
                            <li>กรอกข้อมูลทั้ง 7 ค่า หรือเลือกตัวอย่างรถเพื่อเติมข้อมูล</li>
                            <li>ตรวจหน่วยและรหัสปี จากนั้นกดทำนาย MPG</li>
                            <li>อ่านค่าประมาณพร้อมหน่วย MPG แล้วดูผลประเมินประกอบ</li>
                        </ol>
                        <p>เมื่อแก้ข้อมูล ผลเดิมจะถูกล้าง ปุ่มคืนค่าเริ่มต้นจะล้างผลและคืนข้อมูลเริ่มต้น</p>
                    </article>
                </div>
                """, apply_default_css=False)
            gr.HTML("""
                <h3 class="guide-title">คู่มือข้อมูลรถ</h3>
                <dl class="input-guide">
                    <div><dt>Cylinders</dt><dd>จำนวนกระบอกสูบที่พบในข้อมูล: 3, 4, 5, 6 และ 8</dd></div>
                    <div><dt>Displacement (cu in)</dt><dd>ปริมาตรกระบอกสูบเป็นลูกบาศก์นิ้ว
                    เป็นขนาดเครื่องยนต์ ไม่ใช่กำลังเครื่องยนต์ ช่วงข้อมูล 68–455</dd></div>
                    <div><dt>Horsepower (hp)</dt><dd>กำลังเครื่องยนต์เป็นแรงม้า ช่วงข้อมูล 46–230
                    เว็บขอให้กรอกค่าที่ทราบให้ครบ</dd></div>
                    <div><dt>Weight (lb)</dt><dd>น้ำหนักรถเป็นปอนด์ ช่วงข้อมูล 1,613–5,140 lb</dd></div>
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
    )


if __name__ == "__main__":
    main()
