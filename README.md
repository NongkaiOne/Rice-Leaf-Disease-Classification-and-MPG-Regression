# Rice Leaf Disease Classification and MPG Regression

โปรเจกต์ Machine Learning แยกเป็นสองโฟลเดอร์:

| งาน | โฟลเดอร์และวิธีใช้งาน |
|---|---|
| จำแนกภาพใบข้าว 5 คลาสด้วย CNN | [classification/](classification/README.md) — ใช้ DenseNet121 เป็นโมเดลหลัก พร้อมผลเปรียบเทียบกับ SVM, notebook และ dataset ภาพ 16,533 ไฟล์ |
| ทำนายอัตราประหยัดเชื้อเพลิงรถยนต์ (MPG) | [regression_auto_mpg/](regression_auto_mpg/README.md) — เว็บทำนาย MPG พร้อมส่วนผลประเมินและวิธีใช้งาน โมเดลเดิม notebook และข้อมูล Auto MPG |

## Image Classification: จำแนกโรคใบข้าวด้วย CNN

ระบบใช้ **CNN สถาปัตยกรรม DenseNet121 ที่ fine-tune จาก ImageNet เป็นโมเดลหลัก** สำหรับจำแนกภาพใบข้าว เพื่อช่วยคัดกรองเบื้องต้นและศึกษาความแตกต่างของโรค โดยจำแนกเป็น 5 คลาส:

1. **Rice Blast** — โรคไหม้ใบ ใช้เฉพาะ Leaf Blast
2. **Bacterial Leaf Blight** — โรคขอบใบแห้ง
3. **Sheath Blight** — โรคกาบใบแห้ง
4. **Brown Spot** — โรคใบจุดสีน้ำตาล
5. **Healthy Rice Leaf** — ใบข้าวปกติ

**SVM ที่เคยฝึกด้วยฟีเจอร์สีและเนื้อสัมผัสใช้เป็นโมเดลอ้างอิงสำหรับเปรียบเทียบประสิทธิภาพเท่านั้น** การเปรียบเทียบนี้ใช้ 5 คลาสและชุดแบ่งข้อมูลเดียวกัน: ฝึก 8,186 ภาพ, validation 1,368 ภาพ และ test 2,228 ภาพ โดยเลือกพารามิเตอร์และ checkpoint จาก validation

### ผลเปรียบเทียบและเหตุผลที่ใช้ CNN

ผลบน test เมื่อให้แต่ละโมเดลเลือกคลาสที่มีคะแนนสูงสุด:

| โมเดล | Accuracy | Macro F1 | Recall ต่ำสุดรายคลาส |
|---|---:|---:|---:|
| **CNN — DenseNet121 (โมเดลหลัก)** | **97.67%** | **0.9808** | **96.68%** |
| SVM (โมเดลอ้างอิง) | 89.00% | 0.9033 | 82.44% |

เลือกใช้ CNN เพราะผลทดสอบให้ความแม่นยำสูงกว่า SVM ประมาณ **8.66 จุดเปอร์เซ็นต์** และได้คะแนนดีขึ้นทั้ง macro F1 และ recall ของคลาสที่ทำได้ต่ำสุด แสดงว่าผลดีขึ้นทั้งภาพรวมและการครอบคลุมทุกคลาสในชุดทดสอบนี้ CNN เรียนรู้ลักษณะภาพผ่านการ fine-tune ส่วน SVM ใช้ฟีเจอร์สีและเนื้อสัมผัสที่กำหนดไว้ล่วงหน้า

Macro F1 ให้น้ำหนักทุกคลาสเท่ากัน จึงช่วยประเมินภาพรวมโดยไม่ให้คลาสที่มีภาพมากครอบงำคะแนน ส่วน recall ต่ำสุดช่วยตรวจว่ามีคลาสใดถูกจำแนกพลาดมากเป็นพิเศษ

ในการใช้งาน CNN มีเกณฑ์ความมั่นใจ **0.50** และตอบ **“ไม่แน่ใจ”** เมื่อคะแนนต่ำกว่าเกณฑ์นี้ ผลบน test คือ accuracy **97.58%**, macro F1 **0.9814** และไม่แน่ใจ **8 จาก 2,228 ภาพ** โดยนับภาพที่ไม่แน่ใจเป็นการจำแนกไม่ถูกใน accuracy หลัก

### ข้อมูลและการใช้งาน

ชุดข้อมูลต้นทางมี **16,533 ภาพใน 5 คลาส** จาก [Rice Leaf Diseases Detection](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection) และ [Rice Leaf Disease: An Images Dataset](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) ภาพที่ใช้ฝึกและประเมินผ่านการคัดภาพและแบ่งกลุ่มภาพคล้ายเพื่อลดการรั่วข้ามชุดแล้ว

ผู้ใช้สามารถอัปโหลดภาพ JPG/PNG หรือเลือกภาพตัวอย่าง แล้วอ่านผลจำแนกและคะแนนรายคลาส แอปใช้ CNN เป็นค่าเริ่มต้น พร้อมหน้าเปรียบเทียบคลาสและ Evaluation; ตัวเลือก SVM มีไว้ทดลองเทียบกับโมเดลอ้างอิง

โค้ด โมเดลที่บันทึก ชุดข้อมูล และ notebook พร้อมผลรันอยู่ในโฟลเดอร์ `classification/` ผ่านการติดตั้งใน environment ใหม่และทดสอบการอัปโหลดทั้งสองโมเดลแล้ว **การเผยแพร่ CNN รุ่นปัจจุบันบน Render ยังรอการยืนยัน** ดูรายละเอียดการเข้าถึงใน [สถานะงาน](classification/STATUS.md)

ข้อจำกัด: CNN ยังสับสนระหว่างใบปกติกับ Rice Blast และยังไม่มีชุดภาพภาคสนามอิสระสำหรับยืนยันความทนต่อพื้นหลังใหม่ กลุ่มภาพคล้ายไม่ใช่รหัสใบ/ต้นจริง และ test นี้เคยใช้รายงานผลในการทดลองก่อน จึงไม่ควรตีความคะแนนเป็นความแม่นยำในแปลงหรือการยืนยันโรค

[วิธีติดตั้งและใช้งาน](classification/README.md) · [รายงานผลรายคลาสและข้อผิดพลาด](classification/experiments/cnn5/RESULTS.md) · [CNN notebook](classification/notebooks/rice_leaf_classification.ipynb) · [SVM notebook](classification/notebooks/rice_leaf_svm.ipynb)

## ดาวน์โหลดพร้อม dataset

ติดตั้ง Git LFS แล้วรัน:

```powershell
git lfs install
git clone https://github.com/NongkaiOne/Rice-Leaf-Disease-Classification-and-MPG-Regression.git
cd Rice-Leaf-Disease-Classification-and-MPG-Regression
git lfs pull
```

ภาพ Classification ทั้ง 16,533 ไฟล์เก็บผ่าน Git LFS ข้อมูลไฟล์ไม่ซ้ำประมาณ 4.82 GB และใช้พื้นที่รวมประมาณ 6.19 GB เมื่อดาวน์โหลดครบทุก path ให้ใช้คำสั่งด้านบนเพื่อรับไฟล์ภาพจริง จากนั้นเข้าโฟลเดอร์ของงานและติดตั้ง dependencies ตาม README ของงานนั้น

## ผู้จัดทำ

| รหัสนิสิต | ชื่อ–นามสกุล |
|---|---|
| 6730300213 | นายธนภัทร สีบุตดา |
| 6730300246 | นางสาวพัชญ์ปณดา ชัยเกตุธนพัฒน์ |
| 6730300299 | นางสาวน้ำพระทัย สาระกูล |
| 6730300868 | นายพัศสพล ราตรีวิจิตร์ |

แหล่งข้อมูลและสิทธิ์ภาพ: [DATASET.md](classification/DATASET.md) · สถานะงาน Classification: [STATUS.md](classification/STATUS.md)



Classification บน Render: [https://rice-leaf-classification.onrender.com/](https://rice-leaf-classification.onrender.com/)

MPG deploy แยกบน Render: [https://auto-mpg-regression-opb7.onrender.com/](https://auto-mpg-regression-opb7.onrender.com/);
