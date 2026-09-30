# สถานะงานตาม Purpose.md — เฉพาะ Classification

ตรวจล่าสุด: **1 ตุลาคม 2026**

## รอบเพิ่มข้อมูลจาก more

นำเข้า 7,040 ภาพ (2,684 กลุ่มพิกเซลใหม่) โดยเก็บ mapping/checksum ครบ; **ลบ more แล้วหลังตรวจสำเนาและฝึกใหม่สำเร็จ**

ภาพรวมหลังนำเข้า: **25,445 ไฟล์** ก่อน audit → ใช้ฝึก **16,848 ภาพ** / ทดสอบ **3,792 ภาพ**; Accuracy **88.50%**, Macro F1 **0.8870**

คะแนนใช้ test ที่ขยายแล้ว ไม่เทียบกับคะแนนเดิมโดยตรง แหล่งภาพเพิ่ม: [Rice Leaf Disease: An Images Dataset — alamshihab075, Kaggle](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) ซึ่งผู้ใช้ระบุสำหรับภาพจาก `more`; ตรวจ Kaggle metadata วันที่ 30 กันยายน 2026 พบใบอนุญาต **MIT** (แยกจาก Apache 2.0 ของชุดเดิม)

## ทำแล้ว

| ข้อกำหนด | สถานะ / หลักฐาน |
|---|---|
| เผยแพร่ GitHub repository และ dataset เต็ม | [NongkaiOne/Rice-Leaf-Disease-Classification-and-MPG-Regression](https://github.com/NongkaiOne/Rice-Leaf-Disease-Classification-and-MPG-Regression) แบบ Public; ภาพ 25,445 ไฟล์ผ่าน Git LFS (20,640 objects ไม่ซ้ำ), พร้อมโค้ด โมเดล notebook และรายชื่อผู้จัดทำ |
| ชื่อและรหัสนิสิตผู้จัดทำ | เพิ่มครบ 4 คนใน README โดยเรียงรหัสนิสิตจากน้อยไปมากแล้ว |
| ระบุปัญหา เป้าหมาย ประโยชน์ และคลาส | README และ notebook: 10 คลาส ใช้ชื่อโฟลเดอร์เป็น label |
| แหล่งที่มาและสิทธิ์ใช้/เผยแพร่ | ผู้ใช้ให้ Kaggle loki4514/rice-leaf-diseases-detection; ชุดเดิม Apache 2.0; ชุดเพิ่ม alamshihab075/rice-leaf-disease-an-images-dataset เป็น MIT ตาม Kaggle metadata; บันทึกใน DATASET.md และแนบ DATASET-LICENSE-APACHE-2.0.txt |
| จำนวนภาพรายคลาส | `rice_leaf_app/artifacts/class_counts.csv` และตาราง README/แอป |
| สำรวจ เตรียมข้อมูล และตรวจภาพซ้ำ | 25,445 ไฟล์ทั้งหมด; ไม่ใช้ซ้ำ/ไม่ผ่าน audit 4,805 ไฟล์; ตรวจรายละเอียดได้จาก dataset_manifest.csv |
| แยก train/test ชัดเจน | เก็บ split เดิมและเพิ่มข้อมูลจาก more; หลัง audit train 16,848 / test 3,792; มี path, label และ SHA-256 |
| ไม่ใช้ test ฝึกหรือเลือกโมเดล | เลือก C=1/10 ด้วย validation ที่แยกจาก train แบบ stratified seed 42; จากนั้นฝึกโมเดลสุดท้ายด้วย train ทั้งหมด |
| อธิบาย preprocessing และเหตุผลเลือกวิธี | RGB 128×128, 404 features สี/texture, StandardScaler, RBF SVM; อธิบายใน README/notebook/แอป |
| ฝึกและบันทึกแบบจำลองจริง | ฝึกใหม่รวมภาพเพิ่มแล้ว บันทึก rice_classifier.joblib (16.06 MB); โหลดกลับตรวจผลตรงกัน |
| ผลประเมินหลักและรายคลาส | Accuracy **88.50%**, Macro F1 **0.8870**, Weighted F1 **0.8840**; CSV/JSON, confusion matrix และกราฟ |
| อธิบายการเฉลี่ยคะแนน | Macro ให้ทุกคลาสเท่ากันเป็นเมตริกหลัก; Weighted ถ่วงตามจำนวนภาพ |
| วิเคราะห์ข้อผิดพลาด | README และ notebook ระบุคลาสเด่น/อ่อน คู่สับสน ตัวอย่างถูก/ผิด; `errors.csv` เก็บข้อผิดพลาดทุกภาพ |
| Notebook มีคำอธิบาย รันตามลำดับ | `notebooks/rice_leaf_classification.ipynb`: รันจริง **11 code cells, 0 errors, 7 PNG outputs**; โหมดทบทวนโหลดโมเดลที่ฝึกจริงและประเมิน test ทั้งหมดซ้ำตรงกับรายงาน; มีโค้ดฝึกครบและตั้ง RETRAIN=True เพื่อฝึกใหม่ |
| Gradio รับภาพและจำแนก | `app.py`, `rice_leaf_app/app.py`, `inference.py`; แสดงชื่อคลาส คะแนนทุกคลาส ภาพที่เตรียมแล้ว และข้อความเมื่อไม่มีภาพ |
| หน้าตาคล้าย digit_svm_app | รูปแบบ Upload / Prediction / Processed image พร้อมชื่อหัวข้อ Rice Leaf Disease Classifier |
| เปรียบเทียบระหว่างคลาส | แท็บเลือกสองคลาส พร้อมภาพ ตัวเลข train/test, precision/recall/F1 และจำนวนสับสนกัน; แท็บผลประเมินมีครบทุกคลาส |
| วิธีใช้งานและตัวอย่าง input | มีใน README/แอป และตัวอย่างภาพ test อย่างละคลาสใน `examples/` |
| requirements และโค้ดพร้อมรัน | pin package versions; รัน pip install -r requirements.txt สำเร็จใน runtime ที่เตรียมไว้ และ pip check ผ่าน |
| ทดสอบ pipeline และแอป | `python -m rice_leaf_app.verify` ผ่าน; ตรวจ split/hash, metrics, preprocessing parity, saved model, formats, empty input, Gradio construction |
| ทดสอบผ่าน HTTP จริง | `scripts/smoke_api.py` ผ่าน: อัปโหลดภาพตัวอย่าง 10 คลาส → ผลทำนาย → ดาวน์โหลด preview; คะแนนตรงกับ inference; empty upload และ comparison callback ผ่าน; รายงาน `artifacts/smoke_test.json` |
| เตรียมไฟล์โฮสต์ | root app.py, requirements, metadata Hugging Face Spaces ใน README และ Dockerfile |

## ยังไม่เสร็จ / ต้องมีข้อมูลเพิ่มเติม

| รายการ | เหตุผล / สิ่งที่ต้องทำต่อ |
|---|---|
| **Public URL ของเว็บที่ไม่ต้องเปิดเครื่องนิสิต** | ยังไม่ได้ deploy ไปบัญชีโฮสต์ ไม่มี public URL; localhost ไม่ผ่านข้อกำหนดนี้ ไฟล์พร้อมนำไป deploy แล้ว |
| **ตรวจการเข้าถึงแบบผู้สอน** | GitHub เป็น Public และตรวจไฟล์พร้อมตัวอย่างภาพ LFS แบบไม่ล็อกอินแล้ว; ยังต้อง deploy เว็บแอปและตรวจการทำนายผ่าน public URL |
| **ตรวจภาพหน้าจอ/การคลิกผ่าน browser จริง** | เครื่องมือ browser รายงานว่าไม่มี browser surface; ลอง computer-use skill แล้ว native pipe ไม่พร้อม (`os error 2`); จึงยืนยันได้เฉพาะการสร้าง UI, callbacks และ HTTP end-to-end ไม่อ้างว่าตรวจภาพหน้าจอแล้ว |
| ติดตั้งบนเครื่องใหม่และ deploy Docker | ยังไม่ได้ทดสอบ fresh OS/venv หรือ Docker build; ได้ตรวจเวอร์ชัน dependencies ที่ติดตั้งและรันตามคำสั่งใน runtime ของ workspace นี้แล้ว |

## ข้อจำกัดของผลโมเดล

- F1 สูงสุด: bacterial_leaf_blight **0.9984**; ต่ำสุด: rice_hispa **0.6704**
- คู่สับสนมากที่สุด: sheath_blight → leaf_blast **57 ภาพ** (รายละเอียดใน README)
- ไม่ทราบรหัสภาพต้นฉบับ/ต้นข้าว/แปลง จึงยังไม่ยืนยันการแยก augmented relatives หรือ near duplicates ข้าม train/test; ควรประเมินกับแปลงใหม่เพิ่มเติม
- ไม่มีคลาส “ภาพอื่น” และไม่ใช้ผลนี้ยืนยันโรคจริง

## ใช้งานตอนนี้

จากโฟลเดอร์ `classification/`:

```powershell
..\.runtime\python.exe app.py
```

เปิด **http://127.0.0.1:7861** (เลือก 7861 เพราะเครื่องนี้มีแอปอื่นใช้ 7860 อยู่) ถ้าแอปกำลังทำงานอยู่ให้เปิด URL ได้เลย ไม่ต้องสั่งรันซ้ำ

รายงานนี้ครอบคลุมเฉพาะ Classification; รวมงาน MPG Regression ที่มีอยู่บน GitHub ไว้ใน `../regression_auto_mpg/` แล้ว โดยยังไม่ได้ทดสอบงาน Regression ในรอบนี้
