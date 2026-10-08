"""Active five-class CNN settings. Legacy SVM experiments keep config.py."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
APP_DIR=Path(__file__).resolve().parent
ARTIFACTS=APP_DIR/'artifacts_cnn5'
EXAMPLES=APP_DIR/'examples_cnn5'
MODEL_PATH=APP_DIR/'models/rice_cnn.onnx'
METADATA_PATH=APP_DIR/'models/rice_cnn.json'
CLASSES={
 'rice_blast':('Rice Blast','โรคไหม้'),
 'bacterial_leaf_blight':('Bacterial Leaf Blight','โรคขอบใบแห้ง'),
 'sheath_blight':('Sheath Blight','โรคกาบใบแห้ง'),
 'brown_spot':('Brown Spot','โรคใบจุดสีน้ำตาล'),
 'healthy':('Healthy Rice Leaf','ใบข้าวปกติ'),
}
def display_label(label):
 english,thai=CLASSES[label];return f'{english} · {thai}'
