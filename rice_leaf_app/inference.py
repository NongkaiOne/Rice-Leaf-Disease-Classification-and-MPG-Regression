from functools import lru_cache

import joblib
import numpy as np

from rice_leaf_app.config import MODEL_PATH, display_label
from rice_leaf_app.preprocessing import FEATURE_VERSION, features_from_processed, preprocess_image


class RicePredictor:
    def __init__(self, bundle):
        if bundle["feature_version"] != FEATURE_VERSION:
            raise ValueError("Model feature version does not match preprocessing code. Retrain the model.")
        self.model = bundle["model"]

    def predict(self, image):
        processed = preprocess_image(image)
        probabilities = self.model.predict_proba(features_from_processed(processed).reshape(1, -1))[0]
        winner = str(self.model.classes_[np.argmax(probabilities)])
        scores = {display_label(str(label)): float(prob) for label, prob in zip(self.model.classes_, probabilities)}
        rows = [[display_label(str(self.model.classes_[i])), round(float(probabilities[i])*100, 2)]
                for i in np.argsort(-probabilities)]
        result = f"### ผลจำแนก: {display_label(winner)}\nคะแนนโมเดล **{max(probabilities):.1%}**"
        result += "\n\nคะแนนเป็นค่าประมาณจากโมเดล ไม่ใช่ความแน่นอนว่าภาพมีโรคนั้น"
        return scores, processed, result, rows


@lru_cache(maxsize=1)
def get_predictor():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("ไม่พบโมเดล กรุณารัน python -m rice_leaf_app.train_model ก่อน")
    return RicePredictor(joblib.load(MODEL_PATH))
