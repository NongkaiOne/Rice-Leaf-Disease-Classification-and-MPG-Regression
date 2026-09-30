from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Rice_Leaf_Diease" / "Rice_Leaf_Diease"
ARTIFACTS = APP_DIR / "artifacts"
MODEL_PATH = APP_DIR / "models" / "rice_classifier.joblib"
SEED = 42
CLASSES = {
    "bacterial_leaf_blight": ("Bacterial Leaf Blight", "ขอบใบแห้ง"),
    "brown_spot": ("Brown Spot", "ใบจุดสีน้ำตาล"),
    "healthy": ("Healthy Rice Leaf", "ใบข้าวปกติ"),
    "leaf_blast": ("Leaf Blast", "ไหม้ใบ"),
    "leaf_scald": ("Leaf Scald", "ใบลวก"),
    "narrow_brown_spot": ("Narrow Brown Leaf Spot", "ใบขีดสีน้ำตาล"),
    "neck_blast": ("Neck Blast", "ไหม้คอรวง"),
    "rice_hispa": ("Rice Hispa", "ใบเสียหายจากแมลงดำหนาม"),
    "sheath_blight": ("Sheath Blight", "กาบใบแห้ง"),
    "tungro": ("Tungro", "ทังโกร"),
}


def normalize_label(folder):
    label = folder.lower().replace(" ", "_")
    if label not in CLASSES:
        raise ValueError(f"Unknown class folder: {folder}")
    return label


def display_label(label):
    english, thai = CLASSES[label]
    return f"{english} · {thai}"
