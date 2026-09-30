import json
import os

import gradio as gr
import pandas as pd

from rice_leaf_app.config import APP_DIR, ARTIFACTS, CLASSES, display_label
from rice_leaf_app.inference import get_predictor


def predict_image(image):
    try:
        return get_predictor().predict(image)
    except (ValueError, TypeError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def load_reports():
    metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("กรุณาฝึกโมเดลก่อน: python -m rice_leaf_app.train_model")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    counts = pd.read_csv(ARTIFACTS / "class_counts.csv").set_index("label")
    per_class = pd.read_csv(ARTIFACTS / "per_class_metrics.csv").set_index("label")
    return metrics, counts, per_class


def build_app():
    metrics, counts, per_class = load_reports()
    options = [(display_label(label), label) for label in CLASSES]

    def compare_classes(left, right):
        rows = []
        for key, name in [("raw_train", "ภาพ train ที่ได้รับ"), ("raw_test", "ภาพ test ที่ได้รับ"),
                          ("used_train", "ภาพ train ที่ใช้จริง"), ("used_test", "ภาพ test ที่ใช้จริง")]:
            rows.append([name, str(int(counts.loc[left, key])), str(int(counts.loc[right, key]))])
        for key, name in [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1-score")]:
            rows.append([name, f"{per_class.loc[left, key]:.3f}", f"{per_class.loc[right, key]:.3f}"])
        labels = metrics["classes"]
        li, ri = labels.index(left), labels.index(right)
        cm = metrics["confusion_matrix"]
        note = (f"**{display_label(left)}** เทียบกับ **{display_label(right)}**\n\n"
                f"ภาพจริงคลาสซ้ายที่ทำนายเป็นคลาสขวา: **{cm[li][ri]} ภาพ** · "
                f"ภาพจริงคลาสขวาที่ทำนายเป็นคลาสซ้าย: **{cm[ri][li]} ภาพ**")
        if left == right:
            note = "เลือกคลาสต่างกันเพื่อดูการสับสนระหว่างคลาส (ขณะนี้กำลังแสดงคลาสเดียวกัน)"
        return (str(APP_DIR / "examples" / f"{left}.jpg"),
                str(APP_DIR / "examples" / f"{right}.jpg"), rows, note)

    initial = compare_classes("bacterial_leaf_blight", "brown_spot")
    class_table = []
    for label in CLASSES:
        class_table.append([display_label(label), int(counts.loc[label, "used_train"]),
                            int(counts.loc[label, "used_test"]),
                            *[round(float(per_class.loc[label, k]), 3) for k in ("precision", "recall", "f1-score")]])
    best = per_class["f1-score"].idxmax()
    worst = per_class["f1-score"].idxmin()
    css = """
    .gradio-container {max-width: 1180px !important;}
    #title {border-left: 5px solid #18815b; padding-left: 18px; margin-bottom: 12px;}
    #summary {
        --summary-bg: #edf8f1;
        --summary-text: #153c29;
        --summary-border: #bddccc;
        background: var(--summary-bg);
        border: 1px solid var(--summary-border);
        border-radius: 12px;
        padding: 16px;
    }
    .dark #summary {
        --summary-bg: #163329;
        --summary-text: #edf8f1;
        --summary-border: #3b6953;
    }
    #summary, #summary :is(h1, h2, h3, p, strong, span, li) {
        color: var(--summary-text) !important;
    }
    #summary *::selection {background: #245d43; color: #ffffff;}
    """
    with gr.Blocks(title="Rice Leaf Disease Classifier", theme=gr.themes.Soft(primary_hue="green"), css=css) as app:
        gr.Markdown("# Rice Leaf Disease Classifier\nจำแนกภาพข้าว 10 คลาสด้วย SVM · เปรียบเทียบผลระหว่างคลาส", elem_id="title")
        with gr.Tab("จำแนกภาพ · Classification"):
            gr.Markdown("อัปโหลดภาพข้าว 1 ภาพ แล้วกด **จำแนกภาพ** หรือเลือกตัวอย่างด้านล่าง")
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Upload a rice image · อัปโหลดภาพข้าว", sources=["upload"], height=310)
                    with gr.Row():
                        submit = gr.Button("จำแนกภาพ · Classify", variant="primary")
                        clear = gr.Button("ล้างภาพ")
                    preview = gr.Image(label="Processed 128 × 128 RGB image · ภาพที่เตรียมแล้ว", height=180, interactive=False)
                with gr.Column():
                    result = gr.Markdown("### ผลจำแนก\nรอภาพสำหรับจำแนก", elem_id="summary")
                    scores = gr.Label(label="Prediction · เปรียบเทียบคะแนนทั้ง 10 คลาส", num_top_classes=10)
                    with gr.Accordion("ดูคะแนนเป็นตาราง", open=False):
                        probabilities = gr.Dataframe(headers=["คลาส", "คะแนน (%)"], datatype=["str", "number"], interactive=False)
            submit.click(predict_image, image, [scores, preview, result, probabilities], api_name="classify")
            clear.click(lambda: (None, None, None, "### ผลจำแนก\nรอภาพสำหรับจำแนก", []),
                        outputs=[image, scores, preview, result, probabilities], api_name=False)
            gr.Examples(examples=[[str(APP_DIR / "examples" / f"{label}.jpg")] for label in CLASSES],
                        inputs=image, label="ตัวอย่างภาพจากชุดทดสอบ · เลือกแล้วกดจำแนกภาพ", examples_per_page=10)
            gr.Markdown("**ขอบเขต:** จำแนกได้เฉพาะ 10 คลาสในชุดข้อมูล รวมใบปกติ ความเสียหายจากแมลง และโรคข้าว "
                        "โมเดลยังบังคับเลือกคลาสเมื่ออัปโหลดภาพอื่น จึงควรใช้ภาพข้าวที่เห็นลักษณะชัดเจน "
                        "ผลนี้เป็นตัวอย่างงาน Machine Learning ไม่ใช่การยืนยันโรคในแปลง")
        with gr.Tab("เปรียบเทียบคลาส · Class Comparison"):
            with gr.Row():
                left = gr.Dropdown(options, value="bacterial_leaf_blight", label="คลาสซ้าย")
                right = gr.Dropdown(options, value="brown_spot", label="คลาสขวา")
            with gr.Row():
                left_img = gr.Image(value=initial[0], label="ตัวอย่างคลาสซ้าย", height=260, interactive=False)
                right_img = gr.Image(value=initial[1], label="ตัวอย่างคลาสขวา", height=260, interactive=False)
            pair_table = gr.Dataframe(value=initial[2], headers=["รายการ", "คลาสซ้าย", "คลาสขวา"], interactive=False)
            pair_note = gr.Markdown(initial[3])
            for dropdown in (left, right):
                dropdown.change(compare_classes, [left, right], [left_img, right_img, pair_table, pair_note], api_name=False)
        with gr.Tab("ผลประเมิน · Evaluation"):
            gr.Markdown(f"## ผลบนชุดทดสอบที่ไม่ได้ใช้ฝึกหรือเลือกโมเดล\n"
                        f"**Accuracy {metrics['accuracy']:.2%}** · **Macro F1 {metrics['macro_f1']:.4f}** · "
                        f"**Weighted F1 {metrics['weighted_f1']:.4f}**\n\n"
                        f"Train **{metrics['train_count']:,}** ภาพ · Test **{metrics['test_count']:,}** ภาพ\n\n"
                        "ใช้ **Macro F1 เป็นเมตริกหลัก** โดยเฉลี่ย F1 ทุกคลาสเท่ากัน ส่วน Weighted F1 ถ่วงตามจำนวนภาพ "
                        "Accuracy คือสัดส่วนที่ทำนายถูกทั้งหมด; Precision คือความแม่นเมื่อทำนายคลาสนั้น และ Recall คือสัดส่วนภาพจริงคลาสนั้นที่ค้นพบ")
            gr.Dataframe(value=class_table, headers=["Class", "Train", "Test", "Precision", "Recall", "F1"], interactive=False)
            gr.Markdown(f"คลาสที่มี F1 สูงสุด: **{display_label(best)}** · ต่ำสุด: **{display_label(worst)}**\n\n"
                        "ตาราง confusion matrix: แถวเป็นคลาสจริง คอลัมน์เป็นคลาสที่ทำนาย ช่องนอกแนวทแยงคือข้อผิดพลาด")
            gr.Image(str(ARTIFACTS / "confusion_matrix.png"), label="Confusion matrix", interactive=False)
            gr.Image(str(ARTIFACTS / "class_comparison.png"), label="จำนวนภาพและคะแนนรายคลาส", interactive=False)
            gr.Markdown("**ข้อจำกัดของการประเมิน:** ตัดภาพพิกเซลเหมือนกันและ label ขัดแย้งออกแล้ว "
                        "แต่ไม่มีรหัสภาพต้นฉบับ/แปลง จึงยังรับรองไม่ได้ว่าภาพดัดแปลงหรือภาพจากต้นเดียวกันไม่ข้ามชุด "
                        "ผลคะแนนนี้ไม่ใช่ผลรับรองกับแปลงใหม่")
        with gr.Tab("วิธีใช้งาน · About"):
            gr.Markdown("## วิธีใช้งาน\n1. เลือกภาพ JPG หรือ PNG ที่เห็นใบข้าวหรือส่วนข้าวตามคลาส\n"
                        "2. กดจำแนกภาพ แล้วอ่านคลาสที่ได้และคะแนนเปรียบเทียบ\n"
                        "3. เปิดแท็บเปรียบเทียบคลาสเพื่อดูภาพตัวอย่าง จำนวนข้อมูล และผลรายคลาส\n\n"
                        "## วิธีสร้างโมเดล\nปรับภาพทั้งเฟรมเป็น RGB 128 × 128 → สกัด histogram สี HSV, ค่าเฉลี่ย/ส่วนเบี่ยงเบนสี RGB "
                        "และทิศทางขอบภาพ รวม 404 ค่า → StandardScaler → RBF SVM\n\n"
                        "เลือก C จาก validation ที่แบ่งจาก train เท่านั้น แล้วฝึกโมเดลสุดท้ายด้วย train "
                        "และบันทึก scaler พร้อม SVM ใช้ขั้นตอนเตรียมภาพเดียวกับตอนฝึก\n\n"
                        "## ข้อมูล\nใช้ Rice_Leaf_Diease ที่ผู้จัดทำได้รับ: 10 คลาส ใช้โฟลเดอร์เป็น label "
                        "ไม่รวม Rice_Leaf_AUG ที่มี 9 คลาส · "
                        "[แหล่งข้อมูล Kaggle — loki4514](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection) "
                        "ระบุใบอนุญาต Apache 2.0 (ตรวจ metadata วันที่ 30 กันยายน 2026)")
    return app


def main():
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860" if os.environ.get("SPACE_ID") else "7861")), share=False)


if __name__ == "__main__":
    main()
