# Rice Leaf Disease Classification — CNN และ SVM

จำแนกภาพใบข้าวเพื่อช่วยคัดกรองเบื้องต้นและเปรียบเทียบประสิทธิภาพระหว่าง CNN ที่เรียนรู้ลักษณะภาพ กับ SVM ที่ใช้ฟีเจอร์สี/เนื้อสัมผัส เป็น supervised learning ไม่ใช่การยืนยันโรคในแปลง

**เว็บสำหรับผู้สอน:** https://rice-leaf-classification.onrender.com/

**การเผยแพร่เว็บ:** ดู [RENDER.md](RENDER.md) สำหรับการตั้งค่าและตรวจการทำนายหลัง deploy

## คลาสและข้อมูล

1. Rice Blast — โรคไหม้ใบ (ใช้เฉพาะ Leaf Blast)
2. Bacterial Leaf Blight — โรคขอบใบแห้ง
3. Sheath Blight — โรคกาบใบแห้ง
4. Brown Spot — โรคใบจุดสีน้ำตาล
5. Healthy Rice Leaf — ใบข้าวปกติ

เก็บภาพต้นทาง **16,533 ไฟล์** หลังลบคลาสที่ไม่ใช้ แบ่ง train/test ตามโฟลเดอร์; ไม่ใช้และลบ Rice_Leaf_AUG แล้ว หลังคัดภาพเสีย/ซ้ำ/กลุ่มเสี่ยงรั่ว ใช้ fit **8,186**, validation **1,368**, test **2,228** ภาพ

| label | raw_train | raw_test | used_train | validation | used_test |
|---|---|---|---|---|---|
| rice_blast | 4620 | 967 | 2235 | 377 | 605 |
| bacterial_leaf_blight | 1386 | 376 | 1182 | 195 | 318 |
| sheath_blight | 1853 | 641 | 1588 | 265 | 353 |
| brown_spot | 1480 | 380 | 1229 | 207 | 349 |
| healthy | 3836 | 994 | 1952 | 324 | 603 |

raw_train/raw_test = ภาพต้นทางที่เก็บ; used_train = fit ที่ใช้จริง; validation/used_test = ภาพประเมิน. Manifest ระบุ path, label, pixel hash และ split ชัดเจนที่ experiments/cnn5/manifest.csv

แหล่งข้อมูล: [loki4514 — Apache 2.0](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection) และ [alamshihab075 — MIT](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) ตาม metadata ที่บันทึกไว้ รายละเอียดสิทธิ์และที่มา: [DATASET.md](DATASET.md)

## โมเดลและผล

| โมเดล/นโยบาย | Accuracy | Macro F1 | Recall ต่ำสุด |
|---|---:|---:|---:|
| DenseNet121 คะแนนสูงสุด | 97.67% | 0.9808 | 96.68% |
| DenseNet121 เกณฑ์ 0.50 ที่ใช้ในแอป | 97.58% | 0.9814 | 96.68% |
| SVM คะแนนสูงสุด | 89.00% | 0.9033 | 82.44% |

CNN ปฏิเสธ 8/2,228 ภาพ; accuracy เฉพาะภาพที่ยอมตอบ 97.93% แต่ accuracy หลักนับภาพปฏิเสธเป็นจำแนกไม่ถูก. Macro F1 ให้น้ำหนักแต่ละคลาสเท่ากัน; weighted F1 ถ่วงตามจำนวนภาพ. [รายงานรายคลาสและข้อผิดพลาด](experiments/cnn5/RESULTS.md)

CNN ใช้ RGB resize 256/crop 224 + ImageNet normalize; SVM ใช้ RGB 128 และ HSV/RGB/gradient 404 ฟีเจอร์ พร้อม StandardScaler ใน pipeline. ทั้งสองใช้ preprocessing เดียวกันตอนฝึกกับตอนใช้งาน

CNN สับสนระหว่างใบปกติกับ Rice Blast มากที่สุด SVM recall ต่ำสุดเป็น Sheath Blight. ยังไม่มีผลภาพแปลงใหม่/ใบเดียวกันต่างพื้นหลัง กลุ่มแบ่งข้อมูลเป็นกลุ่มภาพคล้ายแทนรหัสใบจริง และ test เคยใช้รายงานในการทดลองก่อน จึงยังไม่ยืนยันการใช้งานภาคสนาม

## ติดตั้งและเปิดแอป

จาก root repository ติดตั้ง Git LFS และดาวน์โหลดภาพจริง:

```powershell
git lfs install
git lfs pull
cd classification
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

หรือใช้ runtime ที่มีใน workspace นี้จากโฟลเดอร์ classification: `../.runtime/python.exe app.py`

เปิด http://127.0.0.1:7861 เลือก CNN หรือ SVM อัปโหลด JPG/PNG แล้วกดจำแนก หรือเลือกภาพตัวอย่างทั้ง 5 คลาสในหน้าเว็บ ดูผลเปรียบเทียบใน Evaluation. เว็บใช้ ONNX Runtime CPU ไม่ต้องมี GPU; dataset ไม่จำเป็นสำหรับการเปิดเว็บที่ deploy แล้ว

## Notebook และทำซ้ำ

- [CNN notebook](notebooks/rice_leaf_classification.ipynb)
- [SVM notebook](notebooks/rice_leaf_svm.ipynb)

ทั้งสองรันตามลำดับในโหมดทบทวนโมเดลจริงโดยไม่ฝึกใหม่ ตั้ง RETRAIN=True เมื่อต้องการฝึกซ้ำ; CNN ต้องติดตั้ง PyTorch/torchvision ตามเครื่องและ requirements-training.txt ก่อน

จาก root repository:

```powershell
python classification/experiments/cnn5/prepare.py
python classification/experiments/svm/train.py
python -m pip install -r classification/experiments/cnn5/requirements-training.txt
python classification/experiments/cnn5/train.py --arch densenet121
python classification/experiments/cnn5/evaluate.py
python classification/experiments/cnn5/report.py
```

SVM อ่านภาพจริงและคำนวณฟีเจอร์เอง; ไม่มี dependency จากการทดลองที่ลบไป CNN ฝึกหัว 3 รอบแล้ว fine-tune 12 รอบ ใช้ class-weighted loss, label smoothing 0.05 และ augmentation เฉพาะ train เลือก checkpoint/เกณฑ์จาก validation. หากไม่มี CUDA ให้เพิ่ม `--allow-cpu` (อาจช้ามาก)

ตรวจหลังเปิดแอป: `python scripts/smoke_cnn_api.py http://127.0.0.1:7861` และ `python scripts/verify_submission.py`

## ไฟล์สำคัญ

| ตำแหน่ง | เนื้อหา |
|---|---|
| rice_leaf_app/app.py | Gradio: จำแนก เปรียบเทียบคลาส Evaluation และวิธีใช้ |
| rice_leaf_app/models/ | CNN ONNX + metadata และ SVM joblib รวม scaler |
| rice_leaf_app/cnn_inference.py, svm_inference.py, preprocessing.py | เตรียมภาพและทำนายเหมือนตอนฝึก |
| experiments/cnn5/ | manifest, โค้ดฝึก CNN, checkpoint, ประวัติและผลประเมิน |
| experiments/svm/ | โค้ดฝึก SVM, validation selection และ test predictions |
| notebooks/ | Notebook ที่มีผลรันจริงของ CNN/SVM |
| Rice_Leaf_Diease/Rice_Leaf_Diease/ | ภาพ 5 คลาส แยก train/test ผ่าน Git LFS |
| data_audit/ | หลักฐานคัดข้อมูล ที่มาภาพเพิ่มเติม และรายการลบ |
| rice_leaf_app/artifacts_cnn5/ | ผลที่แอปแสดงและหลักฐานการตรวจ |

## ผู้จัดทำ (เรียงรหัส)

| รหัสนิสิต | ชื่อ |
|---|---|
| 6730300213 | นายธนภัทร สีบุตดา |
| 6730300246 | นางสาวพัชญ์ปณดา ชัยเกตุธนพัฒน์ |
| 6730300299 | นางสาวน้ำพระทัย สาระกูล |
| 6730300868 | นายพัศสพล ราตรีวิจิตร์ |

Deployment: [RENDER.md](RENDER.md). โฟลเดอร์ MPG แยกจาก Classification และไม่ได้เปลี่ยนในรอบนี้
