import os
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
model = joblib.load(MODEL_PATH)

def predict_mpg(cylinders, displacement, horsepower, weight,
                acceleration, model_year, origin):
    data = pd.DataFrame([{
        "cylinders": int(cylinders),
        "displacement": float(displacement),
        "horsepower": float(horsepower),
        "weight": float(weight),
        "acceleration": float(acceleration),
        "model_year": int(model_year),
        "origin": int(origin),
    }])

    prediction = float(model.predict(data)[0])
    return f"{prediction:.2f} MPG"

demo = gr.Interface(
    fn=predict_mpg,
    inputs=[
        gr.Number(label="Cylinders", value=4, precision=0),
        gr.Number(label="Displacement", value=140.0),
        gr.Number(label="Horsepower", value=90.0),
        gr.Number(label="Weight (lb)", value=2500.0),
        gr.Number(label="Acceleration", value=15.0),
        gr.Number(label="Model year (e.g. 76 = 1976)", value=76, precision=0),
        gr.Dropdown(
            choices=[1, 2, 3],
            value=1,
            label="Origin code (ตามชุดข้อมูล: 1 / 2 / 3)"
        ),
    ],
    outputs=gr.Textbox(label="Predicted fuel economy"),
    title="Auto MPG Regression",
    description=(
        "กรอกคุณลักษณะของรถเพื่อทำนายอัตราสิ้นเปลืองเชื้อเพลิง "
        "ผลลัพธ์มีหน่วยเป็น miles per gallon (MPG)."
    ),
    examples=[
        [8, 307.0, 130.0, 3504, 12.0, 70, 1],
        [4, 97.0, 88.0, 2130, 14.5, 71, 3],
        [4, 120.0, 79.0, 2625, 18.6, 82, 1],
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
    )
