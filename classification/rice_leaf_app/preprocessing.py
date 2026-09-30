"""The same deterministic RGB/color/texture features for training and inference."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (128, 128)
FEATURE_VERSION = "rgb-hsv-gradient-v1"


def preprocess_image(image):
    if image is None:
        raise ValueError("กรุณาอัปโหลดภาพใบข้าวก่อนจำแนก")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return preprocess_image(opened)
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("ข้อมูลภาพว่างหรือมีค่าที่ไม่ถูกต้อง")
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1:
            arr = arr * 255
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if not isinstance(image, Image.Image):
        raise TypeError("รองรับภาพ PIL, NumPy หรือเส้นทางไฟล์ภาพ")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    # Resize the entire frame, no segmentation or crop that might remove lesions.
    return np.array(image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS))


def features_from_processed(rgb):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255
    color = rgb.astype(np.float32) / 255
    features = []
    # Global and 2x2 local normalized HSV histograms: 180 features.
    for grid in (1, 2):
        step = 128 // grid
        for row in range(grid):
            for col in range(grid):
                cell = hsv[row*step:(row+1)*step, col*step:(col+1)*step]
                for channel in range(3):
                    hist = np.histogram(cell[..., channel], bins=12, range=(0, 1))[0]
                    features.extend(hist / cell.shape[0] / cell.shape[1])
    # Local RGB mean/std + gradient orientation histograms: 224 features.
    gray = color @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    for row in range(4):
        for col in range(4):
            region = np.s_[row*32:(row+1)*32, col*32:(col+1)*32]
            cell = color[region]
            features.extend(cell.mean(axis=(0, 1)))
            features.extend(cell.std(axis=(0, 1)))
            hist = np.histogram(orientation[region], bins=8, range=(0, np.pi),
                                weights=magnitude[region])[0]
            features.extend(hist / (np.linalg.norm(hist) + 1e-6))
    return np.asarray(features, dtype=np.float32)


def image_to_features(image):
    return features_from_processed(preprocess_image(image))
