# Rice Leaf Disease Classification and MPG Regression

โปรเจกต์ Machine Learning แยกเป็นสองโฟลเดอร์:

| งาน | โฟลเดอร์และวิธีใช้งาน |
|---|---|
| จำแนกภาพโรคข้าว 10 คลาสด้วย SVM | [classification/](classification/README.md) — โค้ด โมเดล notebook และ dataset ภาพ 25,445 ไฟล์ |
| ทำนายอัตราประหยัดเชื้อเพลิงรถยนต์ (MPG) | [regression_auto_mpg/](regression_auto_mpg/README.md) — เว็บทำนาย MPG พร้อมส่วนผลประเมินและวิธีใช้งาน โมเดลเดิม notebook และข้อมูล Auto MPG |

## ดาวน์โหลดพร้อม dataset

ติดตั้ง Git LFS แล้วรัน:

```powershell
git lfs install
git clone https://github.com/NongkaiOne/Rice-Leaf-Disease-Classification-and-MPG-Regression.git
cd Rice-Leaf-Disease-Classification-and-MPG-Regression
git lfs pull
```

ภาพ Classification เก็บผ่าน Git LFS ประมาณ 7.45 GB ให้ใช้คำสั่งด้านบนเพื่อรับไฟล์ภาพจริงครบทุกไฟล์ จากนั้นเข้าโฟลเดอร์ของงานและติดตั้ง dependencies ตาม README ของงานนั้น

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
