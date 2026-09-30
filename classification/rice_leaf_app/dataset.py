"""Inventory and exact pixel deduplication without changing original files."""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image, ImageOps

from rice_leaf_app.config import ARTIFACTS, CLASSES, DATA_DIR, ROOT, normalize_label


def inspect_image(item):
    path, split, label = item
    record = dict(path=path.relative_to(ROOT).as_posix(), split=split, label=label)
    try:
        with Image.open(path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            record.update(width=rgb.width, height=rgb.height,
                          pixel_sha256=hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
                          status="candidate", reason="")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        record.update(status="excluded", reason=f"unreadable: {exc}", pixel_sha256="")
    return record


def audit_dataset(data_dir=DATA_DIR):
    items = []
    for split in ("test", "train"):
        folders = sorted((data_dir / split).iterdir())
        found = {normalize_label(folder.name) for folder in folders if folder.is_dir()}
        if found != set(CLASSES):
            raise ValueError(f"{split}: missing or unexpected classes: {found ^ set(CLASSES)}")
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    items.append((path, split, normalize_label(folder.name)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = []
        for i, row in enumerate(pool.map(inspect_image, items), 1):
            records.append(row)
            if i % 2000 == 0:
                print(f"Audited {i}/{len(items)} images", flush=True)
    frame = pd.DataFrame(records)
    valid = frame.status.eq("candidate")
    conflicts = frame.loc[valid].groupby("pixel_sha256").label.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    seen = {}
    # Test files come first: exclude matching training images, never fit on test.
    for idx in frame.index[valid]:
        row = frame.loc[idx]
        key = row.pixel_sha256
        if key in conflicts:
            frame.loc[idx, ["status", "reason"]] = ["excluded", "conflicting labels for identical pixels"]
        elif key in seen:
            frame.loc[idx, ["status", "reason"]] = ["excluded", f"exact duplicate of {seen[key]}"]
        else:
            frame.loc[idx, "status"] = "included"
            seen[key] = row.path
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ARTIFACTS / "dataset_manifest.csv", index=False)
    return frame
