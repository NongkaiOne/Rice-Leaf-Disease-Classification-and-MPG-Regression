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

ค่า RMSE ≈ **2.89 MPG** หมายความว่าโดยทั่วไปค่าทำนายมีความคลาดเคลื่อน
ในระดับประมาณ 2.9 MPG ตามนิยามของ RMSE บน test set นี้

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
└── data/
    └── mpg.csv
```

## 8. วิธีติดตั้งและรันบนเครื่อง
```bash
pip install -r requirements.txt
python app.py
```

## 9. วิธีใช้งานเว็บแอป
กรอกข้อมูล:
- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model year
- Origin code

จากนั้นกด Submit แอปจะแสดงค่าทำนายเป็น `MPG`

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

## 10. Deployment
สามารถนำ `app.py`, `model.joblib` และ `requirements.txt`
ไป deploy บน Hugging Face Spaces (Gradio) ได้

**Regression App URL:** ใส่ URL หลัง deploy สำเร็จ

## 11. ข้อจำกัด
- ชุดข้อมูลมีขนาดค่อนข้างเล็ก (398 แถว)
- เป็นข้อมูลรถยนต์รุ่นเก่า จึงไม่ควรสรุปแทนรถยนต์สมัยใหม่โดยตรง
- Linear Regression อาจไม่สามารถแทนความสัมพันธ์ที่ไม่เป็นเส้นตรงได้ทั้งหมด
- ตัวแปร `name` ไม่ได้ถูกนำมาใช้ในโมเดล

## 12. ผู้จัดทำ
ใส่ชื่อและรหัสนิสิตที่นี่
