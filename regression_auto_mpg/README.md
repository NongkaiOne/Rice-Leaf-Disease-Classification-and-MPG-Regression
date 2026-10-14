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
│   ├── test_app.py
│   └── test_units.py
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

- **Prediction / ทำนาย MPG:** กรอกข้อมูล กด **ทำนาย MPG** และอ่านผลในหน่วยที่เลือก
  เลือกรถจาก dropdown ตัวอย่างเพื่อเติมข้อมูล; **คืนค่าเริ่มต้น** คืนข้อมูลเริ่มต้น ล้างผลและการเลือกตัวอย่าง
- **Evaluation / ผลประเมิน:** RMSE เป็นเมตริกหลัก พร้อม MAE, MSE, R² และจำนวน train/test
- **About & Usage / เกี่ยวกับและวิธีใช้งาน:** วิธีใช้ ความหมายตัวแปร วิธีสร้างโมเดล และข้อจำกัด

ไม่มีปุ่ม Flag. เมื่อแก้ input หรือเลือกตัวอย่าง ผลเก่าจะถูกล้างเพื่อให้กดทำนายใหม่

หน้าจอ Desktop แสดงฟอร์มและผลข้างกัน; ที่ความกว้างไม่เกิน 900px ผลจะอยู่ใต้ฟอร์ม
และที่ความกว้างไม่เกิน 600px ช่องข้อมูลจะเรียงคอลัมน์เดียว ตัวอย่างใช้ dropdown จึงไม่ต้องเลื่อนตารางแนวนอน

### หน่วยวัด
ค่าเริ่มต้นคือ **Metric**: displacement เป็น L, engine power เป็น kW, weight เป็น kg
และผลทำนายเป็น km/L. เลือก **US** เพื่อใช้ cu in / mechanical hp / lb หรือเลือกแต่ละหน่วยเอง:

| ค่า | หน่วยที่เลือกได้ |
|---|---|
| Weight | kg, lb |
| Displacement | cc, L, cu in |
| Engine power | mechanical hp, kW (ไม่ใช่ metric horsepower / PS) |
| ผลทำนาย | US MPG, km/L, L/100 km |

กระบอกสูบยังเป็นจำนวน, acceleration ยังเป็นวินาทีสำหรับ 0–60 mph,
ปีรุ่นและรหัส origin ใช้รูปแบบเดิม. การเปลี่ยนหน่วยปรับทั้งป้ายและตัวเลข โดยคงรถคันเดิมไว้.
เว็บเก็บค่าปริมาตร/กำลัง/น้ำหนักในหน่วยเดิมของโมเดลอย่างละเอียดภายใน session;
ตัวเลขที่แสดงโดยทั่วไปปัดสูงสุด 6 ตำแหน่ง (ค่าที่เล็กมากจะคงค่าไม่เป็นศูนย์)
แต่ไม่ใช้ค่าที่ปัดนั้นแปลงซ้ำหากผู้ใช้ยังไม่ได้แก้ช่องข้อมูล.
เมื่อแก้ตัวเลข จะตีความตามหน่วยที่เลือกอยู่. ตัวอย่างและ Reset ใช้หน่วยที่เลือกไว้;
Reset ไม่เปลี่ยนระบบหน่วยหรือหน่วยผลลัพธ์. ถ้าใช้หน่วยผสม ระบบจะแสดง Custom.

สูตรแปลงนำเข้า (กลับสูตรด้วยการหาร):
- `kg = lb × 0.45359237`
- `cc = cu in × 16.387064`; `L = cc / 1000`
- `kW = mechanical hp × 0.7456998715822702`

ก่อนทำนาย เว็บแปลงกลับเป็น **cu in, hp, lb** แล้วส่ง DataFrame 7 feature เดิมให้ saved Pipeline.
ไม่มีการทำ preprocessing ซ้ำหรือเปลี่ยนโมเดล. การเปลี่ยนหน่วยผลลัพธ์ใช้ค่าทำนายเดิมที่ยังไม่ปัด
โดยไม่เรียกโมเดลใหม่:
- `km/L = US MPG × 1.609344 / 3.785411784`
- `L/100 km = 100 / (km/L)`

ใช้ international mile = 1.609344 km และ **US liquid gallon = 3.785411784 L**
(231 cu in); ไม่ใช้ UK Imperial gallon. อ้างอิง [NIST conversion factors](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b8).
US MPG และ km/L ยิ่งสูงยิ่งประหยัด ส่วน L/100 km ยิ่งต่ำยิ่งประหยัด.
ค่าทำนายที่เป็นศูนย์หรือติดลบจะแสดงคำอธิบายและค่าดิบ US MPG โดยไม่คำนวณค่าผกผัน;
ค่าที่ไม่เป็น finite จะถูกปฏิเสธ. ภาพรถเป็นเพียงสัญลักษณ์ ไม่ใช่มาตรวัดความเร็วหรือระดับประสิทธิภาพ.
เมตริกในส่วน Evaluation ยังคงค่าจริงในหน่วยเดิม (US MPG / US MPG²) และไม่เปลี่ยนตาม selector.

| Feature (เรียงตามที่โมเดลรับ) | ความหมาย | ค่าที่พบในข้อมูล (หน่วยเดิมของโมเดล) |
|---|---|---|
| `cylinders` | จำนวนกระบอกสูบ | 3, 4, 5, 6, 8 |
| `displacement` | ปริมาตรกระบอกสูบ เลือก L / cc / cu in ไม่ใช่แรงม้า | 68–455 cu in |
| `horsepower` | กำลังเครื่องยนต์ เลือก kW / mechanical hp | 46–230 hp |
| `weight` | น้ำหนักรถ เลือก kg / lb | 1,613–5,140 lb |
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

ตัวอย่างข้อมูลในหน่วยเดิมของโมเดล (เลือก US ก่อนกรอกเอง หรือเลือกตัวอย่าง Chevrolet ในแอป):
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
ค่าที่เทียบเท่าคือ **5.030828648 L / 96.94098330569513 kW / 1589.38766448 kg**;
เมื่อเลือกตัวอย่างใน Metric เว็บจะแสดงค่าที่ปัดสำหรับอ่านง่ายและคงค่าภายในไว้ จึงทำนายรถคันเดิม.

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

เว็บสาธารณะ: [Auto MPG Regression](https://auto-mpg-regression-opb7.onrender.com/)

ผลตรวจสอบวันที่ **14 ตุลาคม 2026**: `/config` ตอบ HTTP 200 และการทำนายตัวอย่าง Chevrolet
ผ่าน API ให้ **14.94 MPG**. อย่างไรก็ตาม service ยังแสดงอินเทอร์เฟซรุ่นเก่า:
prediction event มี 7 inputs และยังไม่มีตัวเลือกระบบหน่วย Metric/US, หน่วยผลลัพธ์ หรือ JavaScript สำหรับ tooltip.
จึงยังไม่ยืนยันว่า UI ล่าสุดใน repository ถูก deploy แล้ว.

ขั้นตอนที่เหลือ: อัปโหลดไฟล์ Regression ที่แก้แล้วขึ้น GitHub ด้วยตนเอง แล้ว deploy commit ล่าสุดของ `main`
ไปยัง Regression service เดิม โดยตรวจ Root Directory ให้เป็น `regression_auto_mpg` ตามการตั้งค่าด้านบน.
หลัง deploy ให้ตรวจ health check และทดลอง Metric/US, หน่วยรายช่อง, tooltip, ตัวอย่างรถ และ Reset
ผ่าน URL สาธารณะ. ตัวอย่าง Chevrolet ต้องให้ผลเทียบเท่า **14.94 US MPG / 6.35 km/L / 15.75 L/100 km**.
Render Free พัก service เมื่อไม่มีการใช้งาน จึงอาจต้องรอเปิดครั้งแรก.

## 11. ข้อจำกัด
- ชุดข้อมูลมีขนาดค่อนข้างเล็ก (398 แถว)
- เป็นข้อมูลรถยนต์รุ่นเก่า จึงไม่ควรสรุปแทนรถยนต์สมัยใหม่โดยตรง
- Linear Regression อาจไม่สามารถแทนความสัมพันธ์ที่ไม่เป็นเส้นตรงได้ทั้งหมด
- ตัวแปร `name` ไม่ได้ถูกนำมาใช้ในโมเดล

## ผู้จัดทำ (เรียงรหัส)

| รหัสนิสิต | ชื่อ |
|---|---|
| 6730300213 | นายธนภัทร สีบุตดา |
| 6730300246 | นางสาวพัชญ์ปณดา ชัยเกตุธนพัฒน์ |
| 6730300299 | นางสาวน้ำพระทัย สาระกูล |
| 6730300868 | นายพัศสพล ราตรีวิจิตร์ |