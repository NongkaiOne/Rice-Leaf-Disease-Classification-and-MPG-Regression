# ตรวจตาม Purpose.md — เฉพาะ Classification

ตรวจวันที่ 9 ตุลาคม 2026 ขอบเขตนี้ไม่ตรวจ/แก้ MPG

| ข้อกำหนด | สถานะและหลักฐาน |
|---|---|
| ปัญหา เป้าหมาย ประโยชน์ และ 5 ประเภทภาพ | ครบ: README และ notebook ทั้งสอง |
| จำนวนภาพแต่ละคลาสและ split | ครบ: class_counts.csv, manifest.csv; ต้นทาง 16,533 / fit 8,186 / validation 1,368 / test 2,228 |
| เตรียมภาพ/สกัดคุณลักษณะ | ครบ: CNN RGB224+normalize, SVM RGB128/404 features; โค้ด inference ใช้เหมือนฝึก |
| ฝึกและบันทึกแบบจำลองจริง | ครบ: CNN ONNX+metadata, DenseNet checkpoint/history; SVM joblib รวม scaler |
| ไม่ใช้ test ฝึก/ปรับเลือกในรอบนี้ | ครบใน pipeline ปัจจุบัน: checkpoint/C/threshold ใช้ validation; เปิดเผยว่า test เคยรายงานในงานทดลองก่อน ไม่ใช่ holdout ใหม่ |
| ประเมิน test แสดงรายคลาสและวิธีเฉลี่ย | ครบ: Accuracy, macro/weighted F1, recall ต่ำสุด, confusion matrix, จำนวนไม่แน่ใจ; RESULTS.md และ Evaluation |
| ระบุคลาสที่ดี/สับสน วิเคราะห์ข้อผิดพลาด | ครบ: CNN สับสน Healthy/Rice Blast, SVM recall ต่ำสุด Sheath Blight; notebook มีภาพผิดจริง |
| Notebook รันตามลำดับ | ตรวจผ่าน: rice_leaf_classification.ipynb และ rice_leaf_svm.ipynb; โหมดทบทวนผลจริง ไม่ฝึกใหม่อัตโนมัติ |
| ชุดข้อมูลบน GitHub และ label ชัดเจน | เก็บ 5 คลาสใน train/test ผ่าน Git LFS; label/path/split อยู่ manifest; ตรวจ anonymous download action ครบ 12,111 objects รองรับภาพ 16,533 paths; หลักฐาน data_audit/public_lfs_check.json |
| แหล่งข้อมูลและสิทธิ์เผยแพร่ | มีแหล่งที่มาและประกาศ Apache 2.0/MIT ตาม metadata; ข้อจำกัด: ไม่ได้รับ upstream MIT copyright notice และยังไม่ตรวจเจ้าของภาพรายไฟล์ ดู DATASET.md |
| Gradio รับภาพและทำนายด้วยโมเดลที่บันทึก | ครบ: CNN เป็นค่าเริ่มต้น เลือก SVM เพื่อเทียบได้ |
| วิธีใช้และตัวอย่างข้อมูล | ครบ: README, About, ตัวอย่าง 5 คลาสในเว็บ |
| requirements และติดตั้ง/รันตาม README | ตรวจผ่านใน venv ใหม่: ติดตั้ง requirements, เปิดแอป, HTTP 2 โมเดล × 5 คลาส และ pip check |
| README ชื่อ/รหัสนิสิตเรียงน้อยไปมาก | ครบ 4 คน; รวมการติดตั้ง ผล ข้อจำกัด โฟลเดอร์ และ URL |
| GitHub ผู้สอนเปิดได้ | ตรวจ API ไม่ใช้บัญชี: public, HTTP 200; โค้ดรุ่นนี้จะอยู่ใน commit ที่ส่งขึ้น main; ตรวจ public files หลัง push |
| เว็บเปิดผ่านอินเทอร์เน็ตและทำนายได้จริง | **ยังไม่ผ่านการยืนยันรุ่น CNN ภายนอก**: Render ครั้งแรก timeout; ต้องตรวจ public URL หลัง deploy |
| พร้อมช่วงตรวจงานโดยไม่เปิดเครื่องนักศึกษา | ใช้ Render/Docker มี URL แยก ไม่พึ่งเครื่องนักศึกษา แต่ยังต้องยืนยัน service ออนไลน์จริงและดูแลไม่ให้บริการถูกระงับ |

## สิ่งที่ลบ/เก็บ

ลบ Leaf Scald, Narrow Brown Spot, Neck Blast, Rice Hispa, Tungro ทั้ง train/test; ลบ Rice_Leaf_AUG และการทดลอง tree/segmentation/augmentation/EfficientNet/โมเดล 10 คลาสที่ไม่ใช้แล้ว

เก็บ DenseNet121 เป็นโมเดลหลัก กับ SVM 5 คลาสเป็น baseline บน split เดียวกัน รวมฟีเจอร์ โค้ดฝึก checkpoint/ผลประเมิน/ตัวอย่าง และหลักฐาน provenance ที่จำเป็น ไม่ลบประวัติ Git หรือ rewrite LFS history

## ข้อจำกัดที่ไม่ควรอ้างว่าผ่านแล้ว

ไม่มีชุดภาพจริง/หลายพื้นหลังสำหรับยืนยัน field accuracy, consistency, leave-environment-out. Purpose ไม่ได้กำหนดตัวเลขเป้าของการทดสอบเหล่านี้ แต่ต้องเปิดเผยข้อจำกัด ไม่ควรอ้างว่าแก้ปัญหาพื้นหลังได้แล้ว
