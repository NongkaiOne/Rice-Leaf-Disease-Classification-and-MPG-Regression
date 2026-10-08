# สถานะ Classification — 9 ตุลาคม 2026

รุ่นปัจจุบันเหลือ CNN DenseNet121 และ SVM 5 คลาสสำหรับเปรียบเทียบ CNN เป็นค่าเริ่มต้น; ลบโมเดล 10 คลาสและข้อมูลคลาสที่เลิกใช้แล้ว

- ตรวจรายการตาม Purpose: [PURPOSE_CHECKLIST.md](PURPOSE_CHECKLIST.md)
- รายงานคะแนน: [RESULTS.md](experiments/cnn5/RESULTS.md)
- Notebook CNN/SVM รันครบ มีผลจริงและภาพตัวอย่างข้อผิดพลาด
- คำนวณ SVM จากภาพจริง 2,228 ภาพหลังลบ cache เก่า: accuracy 89.00% ตรงเดิม
- CNN ONNX เคยตรวจครบ 2,228 ภาพ ผลตรงกับรายงาน; ตรวจ manifest หลัง cleanup พบไฟล์ครบและมีเฉพาะ 5 คลาส
- คู่เปรียบเทียบ 25 คู่ผ่าน; ติดตั้งใน environment ใหม่ตาม README, pip check และ HTTP ทั้ง 2 โมเดล × 5 คลาสผ่าน
- GitHub เป็น public และ API เปิดได้แบบไม่ล็อกอิน; ตรวจ Git LFS มี download action ครบ 12,111 objects สำหรับภาพ 16,533 paths; push เฉพาะ classification/ ขึ้น main แล้วใน commit c35294b1 และตรวจ public API เห็น CNN DenseNet121/5 คลาสตรงกัน
- **Render ยังเป็นรุ่นเก่า:** ตรวจ /config หลัง push ได้ HTTP 200 แต่หัวข้อยังเป็น SVM 10 คลาสและไม่มีตัวเลือก CNN/SVM จึงยังไม่ผ่านการตรวจเผยแพร่รุ่นใหม่ ต้อง deploy latest commit ของ main แล้วตรวจการทำนายซ้ำ
- Browser tool เปิดไม่ได้ใน environment นี้ จึงยังไม่มีหลักฐานคลิกผ่าน browser ของรุ่นนี้
- ข้อมูล/ใบอนุญาตดู DATASET.md; upstream MIT copyright notice ยังไม่ได้รับ ต้องไม่แต่งข้อความเจ้าของขึ้นเอง

Public URL: https://rice-leaf-classification.onrender.com/

หลักฐานเพิ่มเติม: ดาวน์โหลดภาพตัวอย่างจาก GitHub แบบไม่ล็อกอินครบ 5 คลาส HTTP 200 และ SHA-256 ตรงกับไฟล์ในเครื่อง ดู data_audit/public_image_samples.json; โฟลเดอร์ train/test บน main เหลือ 5 คลาสจริง ดู data_audit/published_check.json

สิ่งที่ยังค้างตาม Purpose: ยืนยันการทำนาย CNN บน Render รุ่นใหม่และความพร้อมช่วงตรวจงาน; ต้นทางภาพเพิ่มเติมยังไม่มี upstream MIT copyright notice ที่ส่งมาด้วย ส่วนโค้ด/โมเดล/ข้อมูล/Notebook/การติดตั้งและทดสอบในเครื่องผ่านแล้ว
