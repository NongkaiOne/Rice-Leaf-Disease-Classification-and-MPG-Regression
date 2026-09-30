"""Meaningful artifact and inference checks: python -m rice_leaf_app.verify."""
import json

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from rice_leaf_app.config import APP_DIR, ARTIFACTS, CLASSES, MODEL_PATH, ROOT, display_label
from rice_leaf_app.inference import get_predictor
from rice_leaf_app.preprocessing import image_to_features, preprocess_image


def verify():
    manifest = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    assert not manifest.pixel_sha256.duplicated().any(), "Exact duplicates remain across splits"
    assert set(manifest.label) == set(CLASSES)
    assert set(manifest.development_split) == {"fit", "validation", "test"}
    assert manifest.loc[manifest.development_split.eq("test"), "split"].eq("test").all()
    assert manifest.loc[~manifest.development_split.eq("test"), "split"].eq("train").all()
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(ARTIFACTS / "test_predictions.csv")
    assert len(predictions) == metrics["test_count"] == np.asarray(metrics["confusion_matrix"]).sum()
    np.testing.assert_allclose(accuracy_score(predictions.label, predictions.predicted), metrics["accuracy"])
    np.testing.assert_allclose(f1_score(predictions.label, predictions.predicted, average="macro"), metrics["macro_f1"])
    predictor = get_predictor()
    model = joblib.load(MODEL_PATH)["model"]
    for label in CLASSES:
        row = predictions[predictions.label.eq(label)].iloc[0]
        path = ROOT / row.path
        scores, preview, result, table = predictor.predict(path)
        assert preview.shape == (128, 128, 3) and preview.dtype == np.uint8
        assert len(scores) == 10 and len(table) == 10
        assert max(scores, key=scores.get) == display_label(row.predicted)
        np.testing.assert_allclose(sum(scores.values()), 1, atol=1e-7)
        np.testing.assert_allclose(max(scores.values()), row.probability)
        with Image.open(path) as image:
            np.testing.assert_array_equal(image_to_features(path), image_to_features(image))
        x = image_to_features(path).reshape(1, -1)
        np.testing.assert_allclose(max(scores.values()), model.predict_proba(x).max())
        assert (APP_DIR / "examples" / f"{label}.jpg").exists()
    # Common real upload formats, including alpha and grayscale.
    for image in [Image.new("L", (33, 70), 128), Image.new("RGBA", (70, 33), (0, 200, 0, 100)),
                  np.ones((70, 33, 3), dtype=np.float32)]:
        assert np.isfinite(image_to_features(image)).all()
        assert len(image_to_features(image)) == 404
    try:
        preprocess_image(None)
    except ValueError:
        pass
    else:
        raise AssertionError("Empty upload must be rejected")
    from rice_leaf_app.app import build_app
    app = build_app()
    assert app.config["components"]
    print("PASS: split integrity, evaluation consistency, saved model, 10-class inference, preprocessing parity, upload formats, empty input, Gradio construction")


if __name__ == "__main__":
    verify()
