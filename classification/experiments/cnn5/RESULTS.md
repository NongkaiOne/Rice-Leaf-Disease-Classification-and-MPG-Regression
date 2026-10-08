# ผลเปรียบเทียบ CNN กับ SVM — 5 คลาส

โมเดลที่เก็บและเปิดให้ทดลอง: DenseNet121 fine-tune กับ SVM RBF ทั้งสองใช้ fit 8,186 / validation 1,368 / test 2,228 ภาพเดียวกัน ไม่ฝึกหรือปรับเกณฑ์จาก test

## การเตรียมข้อมูลและฝึก

CNN: ImageNet pretrain, RGB resize 256 แล้ว crop 224, normalize mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. Random crop/flip/rotation/color jitter/blur เฉพาะ train. ฝึกหัว 3 รอบ lr 0.001 แล้ว fine-tune 12 รอบ lr 0.0001, AdamW, weighted cross entropy, label smoothing 0.05; เลือก checkpoint ด้วย validation macro F1

SVM: RGB 128×128 แบบ LANCZOS, HSV histogram global/2×2, RGB mean/std และ gradient histogram รวม 404 ฟีเจอร์; StandardScaler + RBF SVC, class_weight=balanced. เลือก C จาก 1 และ 10 ด้วย validation macro F1 (ได้ C=10) และใช้ probability=True, seed 42

Manifest เก็บกลุ่มความคล้ายจาก pHash/RGB หลังหมุน/พลิก กลุ่มไม่ข้ามชุดแต่ไม่ใช่รหัสใบจริง เก็บการแบ่งเดิมเพื่อทำซ้ำ; test เคยถูกใช้รายงานผลในงานทดลองก่อน จึงไม่ใช่ holdout ใหม่

## ผลบน test

| architecture | policy | accuracy | macro_f1 | minimum_recall | coverage | rejected |
|---|---|---|---|---|---|---|
| densenet121 | raw | 0.97666 | 0.98079 | 0.96683 | 1.0 | 0 |
| densenet121 | threshold | 0.97576 | 0.98141 | 0.96683 | 0.99641 | 8 |
| baseline_svm | raw | 0.89004 | 0.90329 | 0.82436 | 1.0 | 0 |

raw = เลือกคลาสคะแนนสูงสุด; threshold = CNN ปฏิเสธเมื่อคะแนน <0.50. เกณฑ์เลือกจาก validation โดย coverage ≥95% และ accuracy/recall ต่ำสุดลดไม่เกิน 1 จุดเปอร์เซ็นต์ ห้ามตีความคะแนนเป็นความน่าจะเป็นของโรคที่ผ่าน calibration

Macro F1 เฉลี่ยทุกคลาสเท่ากัน; weighted F1 ถ่วงตามจำนวนภาพ. Accuracy/recall หลักนับภาพไม่แน่ใจเป็นจำแนกไม่ถูก: CNN accuracy 97.58%, macro F1 0.9814, recall ต่ำสุด 96.68%; ยอมตอบ 99.64%, accuracy เฉพาะภาพที่ตอบ 97.93%

## ผล CNN รายคลาส

| label | precision | recall | f1-score | support |
|---|---|---|---|---|
| rice_blast | 0.9513 | 0.9686 | 0.95987 | 605.0 |
| bacterial_leaf_blight | 1.0 | 1.0 | 1.0 | 318.0 |
| sheath_blight | 0.99713 | 0.983 | 0.99001 | 353.0 |
| brown_spot | 0.99707 | 0.97421 | 0.98551 | 349.0 |
| healthy | 0.97655 | 0.96683 | 0.97167 | 603.0 |

## ผล SVM รายคลาส

| label | precision | recall | f1-score | support |
|---|---|---|---|---|
| rice_blast | 0.78009 | 0.86777 | 0.8216 | 605.0 |
| bacterial_leaf_blight | 0.99687 | 1.0 | 0.99843 | 318.0 |
| sheath_blight | 0.97324 | 0.82436 | 0.89264 | 353.0 |
| brown_spot | 0.94753 | 0.87966 | 0.91233 | 349.0 |
| healthy | 0.88418 | 0.89884 | 0.89145 | 603.0 |

## วิเคราะห์

CNN ก่อนปฏิเสธดีกว่า SVM 8.66 จุดเปอร์เซ็นต์. คลาส CNN ที่ recall ต่ำสุดคือ healthy; Bacterial Leaf Blight ถูก 318/318 เฉพาะ test นี้ คู่สับสนหลักคือใบปกติกับ Rice Blast (19 และ 14 ภาพ) เกณฑ์ 0.50 ปฏิเสธ 8 ภาพ: เดิมผิด 6 และถูก 2 จึงเพิ่มความแม่นยำเฉพาะภาพที่ตอบแต่ลด accuracy รวมเล็กน้อย

| label | actual | predicted |
|---|---|---|
| bacterial_leaf_blight | 318 | 318 |
| brown_spot | 349 | 341 |
| healthy | 603 | 597 |
| rice_blast | 605 | 616 |
| sheath_blight | 353 | 348 |
| uncertain | 0 | 8 |

ภาพฝึกคลาสใหญ่/เล็กต่างประมาณ 1.89 เท่า ใช้ class weights แล้ว ไม่จำเป็นต้องทิ้งภาพให้เท่ากัน ไม่มีชุดภาพภาคสนาม/ใบเดียวกันหลายพื้นหลัง จึงยังไม่ยืนยันว่าทนพื้นหลังใหม่ แม้ recall ภายใน dataset ผ่าน 85% ทุกคลาส; ข้าม accuracy ภาพจริงตามคำสั่งเดิม

## หลักฐาน

- checkpoint CNN: densenet121/best.pt; โมเดลใช้งาน: rice_leaf_app/models/rice_cnn.onnx + rice_cnn.json
- SVM: rice_leaf_app/models/rice_svm.joblib รวม scaler; โค้ดฟีเจอร์: rice_leaf_app/preprocessing.py
- ประวัติฝึกและ validation: densenet121/history.csv, validation_report.json; SVM: ../svm/selection.json
- train/test predictions: ไฟล์ CSV ในโฟลเดอร์โมเดล; ตัวอย่างข้อผิดพลาด: rice_leaf_app/artifacts_cnn5/errors.csv
- Notebook: ../../notebooks/rice_leaf_classification.ipynb และ rice_leaf_svm.ipynb

ประวัติการเลือก CNN เคยเทียบ EfficientNet-B0 ด้วย validation และเลือก DenseNet121 ก่อนดู test; โมเดลและการทดลองที่ไม่ได้ใช้ถูกลบตามคำขอ ไม่ย้อนเลือกจาก test
