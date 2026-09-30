"""Refresh dataset/status documentation from the new measured run."""
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'rice_leaf_app' / 'artifacts'
m = json.loads((A / 'metrics.json').read_text(encoding='utf-8'))
s = json.loads((A / 'more_import_summary.json').read_text(encoding='utf-8'))
counts = pd.read_csv(A / 'class_counts.csv')
report = pd.read_csv(A / 'per_class_metrics.csv').set_index('label')
manifest = pd.read_csv(A / 'dataset_manifest.csv').fillna('')
summary = (f"ภาพรวมหลังนำเข้า: **{m['raw_count']:,} ไฟล์** ก่อน audit "
           f"→ ใช้ฝึก **{m['train_count']:,} ภาพ** / ทดสอบ **{m['test_count']:,} ภาพ**; "
           f"Accuracy **{m['accuracy']:.2%}**, Macro F1 **{m['macro_f1']:.4f}**")
data = (ROOT / 'DATASET.md').read_text(encoding='utf-8')
data = re.sub(r'`Rice_Leaf_Diease/Rice_Leaf_Diease/train/<class>/\*`.*?Counts must',
              f"`Rice_Leaf_Diease/Rice_Leaf_Diease/train/<class>/*` and `test/<class>/*` now contain **{m['raw_count']:,} images** before filtering: "
              f"**{int(counts.raw_train.sum()):,} train** and **{int(counts.raw_test.sum()):,} test**. Counts must", data, count=1)
addition = f"""## Additional user-supplied images from more

Imported **{s['source_count']:,} source files**, retaining **{s['copied_unique_files']:,} distinct byte-content copies** in the existing class folders. There are **{s['new_pixel_groups']:,} new decoded-pixel groups** beyond the original dataset. New groups are split approximately 80/20 per class with seed 42. Exact matches retain the existing split (test takes precedence if the old dataset already contained a cross-split duplicate). No existing dataset file was overwritten. Source/destination paths, labels, split assignments, and byte/pixel checksums are recorded in `rice_leaf_app/artifacts/more_import_manifest.csv`.

The user supplied these additions in Healthy, Leaf Blast and Sheath Blight folders. The user identified [Rice Leaf Disease: An Images Dataset — alamshihab075, Kaggle](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) as their source. Kaggle public metadata was checked on 2026-09-30 and declares **MIT** for `alamshihab075/rice-leaf-disease-an-images-dataset`. This is separate from the original dataset's Apache 2.0 license; `DATASET-LICENSE-APACHE-2.0.txt` applies only to the original source. Preserve the upstream MIT license and copyright notice when redistributing the additional images; do not replace them with the Apache text. Images named `aug_*` still lack reliable original-image IDs, so near-duplicate/augmentation leakage remains a limitation.

The source folder `more` is **{'deleted after successful training and checksum verification' if s['source_deleted'] else 'retained until training and checksum verification finish'}**. Test composition expanded, so new and previous aggregate scores are not directly comparable.

"""
data = re.sub(r'## Additional user-supplied images from more\n.*?(?=## Publishing the dataset)', '', data, flags=re.S)
data = data.replace('## Publishing the dataset', addition + '## Publishing the dataset')
(ROOT / 'DATASET.md').write_text(data, encoding='utf-8')
status = (ROOT / 'STATUS.md').read_text(encoding='utf-8')
lines = status.splitlines()
replacements = {
    '| สำรวจ เตรียมข้อมูล และตรวจภาพซ้ำ |': f"| สำรวจ เตรียมข้อมูล และตรวจภาพซ้ำ | {m['raw_count']:,} ไฟล์ทั้งหมด; ไม่ใช้ซ้ำ/ไม่ผ่าน audit {m['excluded_count']:,} ไฟล์; ตรวจรายละเอียดได้จาก dataset_manifest.csv |",
    '| แยก train/test ชัดเจน |': f"| แยก train/test ชัดเจน | เก็บ split เดิมและเพิ่มข้อมูลจาก more; หลัง audit train {m['train_count']:,} / test {m['test_count']:,}; มี path, label และ SHA-256 |",
    '| ฝึกและบันทึกแบบจำลองจริง |': f"| ฝึกและบันทึกแบบจำลองจริง | ฝึกใหม่รวมภาพเพิ่มแล้ว บันทึก rice_classifier.joblib ({(ROOT/'rice_leaf_app/models/rice_classifier.joblib').stat().st_size/1e6:.2f} MB); โหลดกลับตรวจผลตรงกัน |",
    '| ผลประเมินหลักและรายคลาส |': f"| ผลประเมินหลักและรายคลาส | Accuracy **{m['accuracy']:.2%}**, Macro F1 **{m['macro_f1']:.4f}**, Weighted F1 **{m['weighted_f1']:.4f}**; CSV/JSON, confusion matrix และกราฟ |",
}
for i, line in enumerate(lines):
    for prefix, replacement in replacements.items():
        if line.startswith(prefix):
            lines[i] = replacement
status = '\n'.join(lines) + '\n'
best, worst = report['f1-score'].idxmax(), report['f1-score'].idxmin()
status = re.sub(r'- F1 สูงสุด:.*', f"- F1 สูงสุด: {best} **{report.loc[best, 'f1-score']:.4f}**; ต่ำสุด: {worst} **{report.loc[worst, 'f1-score']:.4f}**", status)
cm = m['confusion_matrix']
pairs = sorted(((cm[i][j], m['classes'][i], m['classes'][j]) for i in range(10) for j in range(10) if i != j), reverse=True)
status = re.sub(r'- Rice Hispa จริง.*', f"- คู่สับสนมากที่สุด: {pairs[0][1]} → {pairs[0][2]} **{pairs[0][0]} ภาพ** (รายละเอียดใน README)", status)
status = re.sub(r'## รอบเพิ่มข้อมูลจาก more\n.*?(?=## ทำแล้ว)', '', status, flags=re.S)
more_status = 'ลบ more แล้วหลังตรวจสำเนาและฝึกใหม่สำเร็จ' if s['source_deleted'] else 'ยังเก็บ more ไว้จนกว่าจะตรวจสำเนาครบ'
status = status.replace('## ทำแล้ว', f"## รอบเพิ่มข้อมูลจาก more\n\nนำเข้า {s['source_count']:,} ภาพ ({s['new_pixel_groups']:,} กลุ่มพิกเซลใหม่) โดยเก็บ mapping/checksum ครบ; **{more_status}**\n\n{summary}\n\nคะแนนใช้ test ที่ขยายแล้ว ไม่เทียบกับคะแนนเดิมโดยตรง แหล่งภาพเพิ่ม: [Rice Leaf Disease: An Images Dataset — alamshihab075, Kaggle](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) ซึ่งผู้ใช้ระบุสำหรับภาพจาก `more`; ตรวจ Kaggle metadata วันที่ 30 กันยายน 2026 พบใบอนุญาต **MIT** (แยกจาก Apache 2.0 ของชุดเดิม)\n\n## ทำแล้ว", 1)
(ROOT / 'STATUS.md').write_text(status, encoding='utf-8')
print('DATASET.md and STATUS.md updated from the retrained model and verified import ledger.')
