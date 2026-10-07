# Auto MPG Regression Project

## 1. แนวคิดและปัญหา
โปรเจกต์นี้สร้างแบบจำลอง **Regression** เพื่อทำนายค่าอัตราการประหยัดเชื้อเพลิงของรถยนต์
โดยค่าที่ต้องการทำนายคือ `mpg` ซึ่งมีหน่วยเป็น **miles per gallon (MPG)**

ประโยชน์ของแบบจำลองคือใช้ประมาณค่า MPG จากคุณลักษณะพื้นฐานของรถ เช่น
จำนวนกระบอกสูบ ขนาดเครื่องยนต์ แรงม้า น้ำหนัก อัตราเร่ง ปีรุ่น และรหัสแหล่งกำเนิด

> หมายเหตุ: เป็นแบบจำลองเพื่อการศึกษาและสะท้อนข้อมูลรถในชุดข้อมูลนี้
> ไม่ควรใช้เป็นค่ารับรองอัตราสิ้นเปลืองของรถจริงรุ่นปัจจุบัน

## 2. ชุดข้อมูล
ไฟล์ที่ใช้: `data/mpg.csv`

ชุดข้อมูลนี้ตรงกับ **Auto MPG** ของ UCI Machine Learning Repository:
- จำนวนข้อมูล: **398 แถว**
- Target: `mpg`
- Features ที่ใช้: `cylinders`, `displacement`, `horsepower`, `weight`,
  `acceleration`, `model_year`, `origin`
- `name` ไม่ใช้เป็นตัวแปรนำเข้า เพราะเป็นชื่อ/ตัวระบุรถและมีค่าหลากหลายมาก
- `horsepower` มีค่าที่เขียนเป็น `?` จำนวน **6 แถว**
  หลังแปลงเป็นตัวเลข จึงจัดการด้วย median imputation ที่เรียนรู้จาก training set เท่านั้น

แหล่งข้อมูล:
- UCI Auto MPG: https://archive.ics.uci.edu/dataset/9/auto+mpg
- DOI: https://doi.org/10.24432/C5859H
- License: Creative Commons Attribution 4.0 (CC BY 4.0)

## 3. การแบ่งข้อมูล
ใช้ `train_test_split`:
- Training set: **318 แถว (80%)**
- Test set: **80 แถว (20%)**
- `random_state=42` เพื่อให้แบ่งซ้ำได้เหมือนเดิม

ชุดทดสอบไม่ได้ถูกใช้ในการ fit preprocessing หรือฝึกแบบจำลอง

## 4. การเตรียมข้อมูล
- แปลง `horsepower` เป็นตัวเลข โดยค่าที่ไม่สามารถแปลงได้ (`?`) เป็น missing value
- Numeric features: เติม missing ด้วย median และทำ StandardScaler
- `origin`: ปฏิบัติเป็นตัวแปร categorical และทำ One-Hot Encoding
- ขั้นตอน preprocessing ทั้งหมดอยู่ใน `Pipeline` เดียวกับโมเดล เพื่อให้เว็บแอปใช้ขั้นตอนเดียวกับตอนฝึก

## 5. แบบจำลอง
ใช้ **Linear Regression**

เหตุผล:
1. Target (`mpg`) เป็นค่าต่อเนื่อง จึงเหมาะกับ Regression
2. Linear Regression เป็น baseline ที่อธิบายได้ง่ายและตรงกับเนื้อหาพื้นฐานของรายวิชา
3. สามารถรวม preprocessing และ model ใน scikit-learn Pipeline ได้ ทำให้ลดความเสี่ยงที่เว็บแอปจะเตรียมข้อมูลไม่เหมือนตอนฝึก

## 6. ผลประเมินบน Test Set
- **MAE:** 2.288 MPG
- **MSE:** 8.339
- **RMSE:** 2.888 MPG
- **R²:** 0.845

เมตริกหลักที่เลือกคือ **RMSE** เพราะบอกขนาดความคลาดเคลื่อนในหน่วย MPG
และลงโทษความผิดพลาดที่มีขนาดใหญ่มากกว่าความผิดพลาดเล็ก

ค่า RMSE ≈ **2.89 MPG** สรุปขนาดความคลาดเคลื่อนตามนิยาม RMSE บน test set นี้
ไม่ใช่ช่วงความเชื่อมั่นหรือขอบเขตความผิดพลาดของรถแต่ละคัน; MSE มีหน่วย MPG²

เว็บอ่านค่าประเมินและจำนวนแถวจาก `metrics.json` ของโมเดลเดิม
ไม่ฝึกโมเดลหรือคำนวณคะแนนใหม่เมื่อเริ่มทำงาน

R² ≈ **0.845** หมายถึงแบบจำลองอธิบายความแปรปรวนของค่า MPG ใน test set นี้ได้ประมาณ
**84.5%** (ไม่ควรตีความว่าเป็นเปอร์เซ็นต์ความแม่นยำโดยตรง)

## 7. ไฟล์สำคัญ
```text
regression_auto_mpg/
├── README.md
├── requirements.txt
├── regression_auto_mpg.ipynb
├── app.py
├── model.joblib
├── metrics.json
├── Dockerfile
├── tests/
│   └── test_app.py
└── data/
    └── mpg.csv
```

## 8. วิธีติดตั้งและรันบนเครื่อง
ใช้ Python 3.12 และ virtual environment แยกจาก Classification เพื่อคง dependencies ที่ pin ไว้
จากโฟลเดอร์ repository รันบน Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r regression_auto_mpg/requirements.txt
.\.venv\Scripts\python.exe regression_auto_mpg/app.py
```

บน Linux/macOS จาก repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r regression_auto_mpg/requirements.txt
.venv/bin/python regression_auto_mpg/app.py
```

เปิด [http://127.0.0.1:7860](http://127.0.0.1:7860).
แอปฟังที่ `0.0.0.0` โดยค่าเริ่มต้นสำหรับ Render; หากต้องการจำกัดเฉพาะเครื่องตนเอง
ตั้ง `$env:GRADIO_SERVER_NAME = "127.0.0.1"` ก่อนรันบน PowerShell.
ตั้ง `PORT` เพื่อเปลี่ยนพอร์ต (ค่าเริ่มต้น 7860).
เส้นทาง `model.joblib` และ `metrics.json` อ้างอิงตำแหน่ง `app.py` จึงไม่ขึ้นกับ working directory.

Dependencies เดิมไม่เปลี่ยน และใช้ Gradio 6.5.1 โดยส่ง theme/CSS ให้ `launch()`
ตาม [Gradio 6 migration guide](https://www.gradio.app/guides/gradio-6-migration-guide).

## 9. วิธีใช้งานเว็บแอป
เว็บใช้ `gr.Blocks` เป็นหน้าเดียวในโทน graphite พร้อมสีฟ้าเป็น accent
มี navigation ด้านบนเพื่อเลื่อนไปยังแต่ละส่วน:

- **Prediction / ทำนาย MPG:** กรอกข้อมูล กด **ทำนาย MPG** และอ่านผลพร้อมหน่วย MPG
  เลือกรถจาก dropdown ตัวอย่างเพื่อเติมข้อมูล; **คืนค่าเริ่มต้น** คืนข้อมูลเริ่มต้น ล้างผลและการเลือกตัวอย่าง
- **Evaluation / ผลประเมิน:** RMSE เป็นเมตริกหลัก พร้อม MAE, MSE, R² และจำนวน train/test
- **About & Usage / เกี่ยวกับและวิธีใช้งาน:** วิธีใช้ ความหมายตัวแปร วิธีสร้างโมเดล และข้อจำกัด

ไม่มีปุ่ม Flag. เมื่อแก้ input หรือเลือกตัวอย่าง ผลเก่าจะถูกล้างเพื่อให้กดทำนายใหม่

หน้าจอ Desktop แสดงฟอร์มและผลข้างกัน; ที่ความกว้างไม่เกิน 900px ผลจะอยู่ใต้ฟอร์ม
และที่ความกว้างไม่เกิน 600px ช่องข้อมูลจะเรียงคอลัมน์เดียว ตัวอย่างใช้ dropdown จึงไม่ต้องเลื่อนตารางแนวนอน

| Feature (เรียงตามที่โมเดลรับ) | ความหมาย | ค่าที่พบในข้อมูล |
|---|---|---|
| `cylinders` | จำนวนกระบอกสูบ | 3, 4, 5, 6, 8 |
| `displacement` | ปริมาตรกระบอกสูบเป็นลูกบาศก์นิ้ว (**cu in**) ไม่ใช่แรงม้า | 68–455 |
| `horsepower` | กำลังเครื่องยนต์เป็นแรงม้า (**hp**) | 46–230 |
| `weight` | น้ำหนักรถเป็นปอนด์ (**lb** ไม่ใช่ kg) | 1,613–5,140 |
| `acceleration` | เวลาเร่งจาก 0 ถึง 60 mph เป็นวินาที (**s**) | 8.0–24.8 |
| `model_year` | รหัสปี เช่น **76 = 1976** | 70–82 (1970–1982) |
| `origin` | แหล่งกำเนิด: **1 = USA, 2 = Europe, 3 = Japan** | 1, 2, 3 |

คำอธิบาย origin ตรวจเทียบกับ [เอกสาร Auto MPG ของ TensorFlow](https://www.tensorflow.org/tutorials/keras/regression).
หน่วย displacement, weight และ acceleration ตรวจเทียบกับคำอธิบายชุดข้อมูล Auto
ใน [เอกสาร ISLR หน้า 2](https://cran.r-project.org/web/packages/ISLR/ISLR.pdf#page=2)
ซึ่งใช้แหล่งข้อมูล StatLib เดียวกัน; ใช้อ้างอิงหน่วยเท่านั้น ไม่ได้เปลี่ยนข้อมูลหรือวิธีฝึกของโปรเจกต์นี้.
Dropdown แสดงชื่อ แต่ส่งรหัสตัวเลขเดิมให้ Pipeline; ไม่มีการเปลี่ยนวิธีฝึกของ notebook.

กรุณากรอกทั้ง 7 ค่าเป็นตัวเลขที่มีค่าจำกัด. ค่าต่อเนื่องต้องมากกว่า 0;
cylinders ต้องอยู่ในกลุ่มที่มีในข้อมูล, origin ต้องเป็น 1/2/3,
และ model_year ต้องเป็นจำนวนเต็ม 0–99 (ไม่ใช่ปีเต็ม เช่น 1976).
ปีนอกช่วง 70–82 ยังทำนายได้พร้อมข้อความเตือนเรื่อง extrapolation.
ช่วงค่าต่อเนื่องในตารางเป็นข้อมูลประกอบ ไม่ได้ใช้เป็นขอบเขตบังคับ.
แม้ Pipeline รองรับ median imputation เว็บขอให้กรอกค่าครบ;
ไม่มีการทำ preprocessing ซ้ำหรือปรับค่าทำนายเพื่อให้ดูสมเหตุสมผล.

ตัวอย่างข้อมูล:
```text
cylinders = 8
displacement = 307
horsepower = 130
weight = 3504
acceleration = 12
model_year = 70
origin = 1
```

ผลจากโมเดลเดิมสำหรับตัวอย่างนี้คือ **14.94 MPG** (ตรงกับผลใน notebook).
MPG สูงหมายถึงวิ่งได้ไกลขึ้นต่อน้ำมันหนึ่งแกลลอน.

ตรวจสอบแอปโดยไม่ retrain จาก repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s regression_auto_mpg/tests -v
.\.venv\Scripts\python.exe -m py_compile regression_auto_mpg/app.py
```

## 10. Deployment
เตรียม Docker และ Blueprint `../render-mpg.yaml` สำหรับ Render แยกจาก Rice แล้ว

ใน Render เลือก **New → Web Service** แล้วเลือก repository นี้ ตั้งค่า:

- Name: `auto-mpg-regression`
- Root Directory: `regression_auto_mpg`
- Runtime: Docker
- Dockerfile Path: `./Dockerfile`
- Docker Build Context: `.`
- Instance Type: Free
- Health Check Path: `/`
- Environment: `GIT_LFS_SKIP_SMUDGE=1`

หรือเลือก **New → Blueprint** แล้วตั้ง **Blueprint Path เป็น `render-mpg.yaml`** เพื่อสร้างเฉพาะ MPG
แอปอ่าน `PORT` จาก Render และฟังที่ `0.0.0.0` โหลดโมเดลเดิมโดยไม่ฝึกใหม่
Dockerfile บรรจุทั้ง `app.py`, `model.joblib` และ `metrics.json`; ไม่จำเป็นต้องมี notebook หรือ dataset ใน container.
Blueprint มี Regression เพียง service เดียว แยกจาก `render.yaml` ของ Classification.
เส้นทาง Dockerfile/context อ้างอิง `rootDir` ตาม [Render monorepo documentation](https://render.com/docs/monorepo-support).

หากมี Docker ทดสอบจาก repository root:

```bash
docker build -t auto-mpg-regression ./regression_auto_mpg
docker run --rm -p 127.0.0.1:10000:10000 -e PORT=10000 auto-mpg-regression
```

เปิด [http://127.0.0.1:10000](http://127.0.0.1:10000) แล้วลองตัวอย่างแรก (14.94 MPG).
Health check ใช้ HTTP GET `/`.

สถานะ: เตรียมไฟล์ deploy แล้ว ยังไม่ยืนยันการสร้าง service หรือ public URL.
ขั้นตอนที่เหลือ: นำไฟล์ที่แก้แล้วขึ้น GitHub ด้วยตนเอง, สร้าง Regression service ตาม Blueprint ด้านบน,
ตรวจ health check และทดลองทำนาย, แล้วเพิ่ม URL จริงใน README ทั้งสองไฟล์เมื่อ deploy สำเร็จ.
ไม่มีการ commit, push หรือ deploy อัตโนมัติจากการแก้เว็บนี้.
Render Free พัก service เมื่อไม่มีการใช้งาน จึงอาจต้องรอเปิดครั้งแรก

**Regression App URL:** ใส่ URL หลัง deploy สำเร็จ

## 11. ข้อจำกัด
- ชุดข้อมูลมีขนาดค่อนข้างเล็ก (398 แถว)
- เป็นข้อมูลรถยนต์รุ่นเก่า จึงไม่ควรสรุปแทนรถยนต์สมัยใหม่โดยตรง
- Linear Regression อาจไม่สามารถแทนความสัมพันธ์ที่ไม่เป็นเส้นตรงได้ทั้งหมด
- ตัวแปร `name` ไม่ได้ถูกนำมาใช้ในโมเดล

## 12. ผู้จัดทำ
ใส่ชื่อและรหัสนิสิตที่นี่
