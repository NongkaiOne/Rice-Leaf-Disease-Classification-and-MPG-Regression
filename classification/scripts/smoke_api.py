"""End-to-end real HTTP upload/prediction test against a running Gradio app."""
import json
from pathlib import Path
import sys

import httpx
import numpy as np
from gradio_client import Client, handle_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rice_leaf_app.config import APP_DIR, CLASSES
from rice_leaf_app.inference import get_predictor

url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:7861"
client = Client(url, verbose=False)
results = []
for label in CLASSES:
    path = APP_DIR / "examples" / f"{label}.jpg"
    expected = get_predictor().predict(path)
    response = client.predict(handle_file(str(path)), api_name="/classify")
    assert len(response) == 4
    assert response[0]["label"] == max(expected[0], key=expected[0].get)
    assert len(response[0]["confidences"]) == 10
    np.testing.assert_allclose(sum(c["confidence"] for c in response[0]["confidences"]), 1)
    assert Path(response[1]).is_file(), "Processed preview must download successfully"
    assert len(response[3]["data"]) == 10
    results.append({"example": label, "prediction": response[0]["label"], "status": "pass"})
try:
    client.predict(None, api_name="/classify")
except Exception as exc:
    assert "กรุณาอัปโหลดภาพ" in str(exc), str(exc)
else:
    raise AssertionError("Empty upload must return an error")
with httpx.Client() as http:
    assert http.get(url).status_code == 200
    config = http.get(url + "/config").json()
    tabs = [c["props"]["label"] for c in config["components"] if c["type"] == "tabitem"]
    assert len(tabs) == 4
    # Exercise the exact comparison callback Gradio wires to both dropdowns.
    from rice_leaf_app.app import build_app
    app = build_app()
    comparison = next(f.fn for f in app.fns.values() if getattr(f.fn, "__name__", "") == "compare_classes")
    for left, right in [("healthy", "tungro"), ("neck_blast", "rice_hispa"), ("healthy", "healthy")]:
        image_left, image_right, rows, note = comparison(left, right)
        assert Path(image_left).is_file() and Path(image_right).is_file()
        assert len(rows) == 7 and note
report = {"status": "pass", "url": url, "examples": results, "empty_upload": "pass",
          "comparison_callback": "pass", "tabs": tabs,
          "visual_browser_check": "not performed: no browser surface available"}
(APP_DIR / "artifacts" / "smoke_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("PASS: real HTTP upload + prediction + preview for all 10 classes; empty upload; comparison callback; 4 tabs")
