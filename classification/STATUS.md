# สถานะ Classification — 9 ตุลาคม 2026

รุ่นปัจจุบันเหลือ CNN DenseNet121 และ SVM 5 คลาสสำหรับเปรียบเทียบ CNN เป็นค่าเริ่มต้น; ลบโมเดล 10 คลาสและข้อมูลคลาสที่เลิกใช้แล้ว

- ตรวจรายการตาม Purpose: [PURPOSE_CHECKLIST.md](PURPOSE_CHECKLIST.md)
- รายงานคะแนน: [RESULTS.md](experiments/cnn5/RESULTS.md)
- Notebook CNN/SVM รันครบ มีผลจริงและภาพตัวอย่างข้อผิดพลาด
- คำนวณ SVM จากภาพจริง 2,228 ภาพหลังลบ cache เก่า: accuracy 89.00% ตรงเดิม
- CNN ONNX เคยตรวจครบ 2,228 ภาพ ผลตรงกับรายงาน; ตรวจ manifest หลัง cleanup พบไฟล์ครบและมีเฉพาะ 5 คลาส
- คู่เปรียบเทียบ 25 คู่ผ่าน; ติดตั้งใน environment ใหม่ตาม README, pip check และ HTTP ทั้ง 2 โมเดล × 5 คลาสผ่าน
- GitHub เป็น public และ API เปิดได้แบบไม่ล็อกอิน; ตรวจ Git LFS มี download action ครบ 12,111 objects สำหรับภาพ 16,533 paths; ชุดนี้เตรียมส่งเฉพาะ classification/ ขึ้น main
- **Render ยังไม่ยืนยันรุ่น CNN จากภายนอก:** ครั้งแรก timeout; ต้องตรวจหลัง deploy ไม่อ้างว่ามี URL แล้วเท่ากับใช้งานผ่าน
- Browser tool เปิดไม่ได้ใน environment นี้ จึงยังไม่มีหลักฐานคลิกผ่าน browser ของรุ่นนี้
- ข้อมูล/ใบอนุญาตดู DATASET.md; upstream MIT copyright notice ยังไม่ได้รับ ต้องไม่แต่งข้อความเจ้าของขึ้นเอง

Public URL: https://rice-leaf-classification.onrender.com/
