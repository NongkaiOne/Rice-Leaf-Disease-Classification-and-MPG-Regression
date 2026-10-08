import json
import os

import gradio as gr
import pandas as pd

from rice_leaf_app.cnn_config import APP_DIR, ARTIFACTS, EXAMPLES, CLASSES, display_label
from rice_leaf_app.cnn_inference import get_predictor
from rice_leaf_app.svm_inference import get_svm_predictor


def predict_image(image, model_name="CNN (DenseNet121)"):
    try:
        return (get_svm_predictor() if model_name == "SVM" else get_predictor()).predict(image)
    except (ValueError, TypeError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def load_reports():
    metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("ยังไม่มีผลประเมิน CNN กรุณาฝึกและ export โมเดลก่อน")
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
        return (str(EXAMPLES / f"{left}.jpg"),
                str(EXAMPLES / f"{right}.jpg"), rows, note)

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
        gr.Markdown("# Rice Leaf Disease Classifier\nจำแนกภาพใบข้าว 5 คลาสด้วย CNN · เปรียบเทียบผลระหว่างคลาส", elem_id="title")
        with gr.Tab("จำแนกภาพ · Classification"):
            gr.Markdown("อัปโหลดภาพข้าว 1 ภาพ แล้วกด **จำแนกภาพ** หรือเลือกตัวอย่างด้านล่าง")
            model_choice = gr.Radio(["CNN (DenseNet121)", "SVM"], value="CNN (DenseNet121)", label="โมเดลที่ใช้จำแนก")
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Upload a rice image · อัปโหลดภาพข้าว", sources=["upload"], height=310)
                    with gr.Row():
                        submit = gr.Button("จำแนกภาพ · Classify", variant="primary")
                        clear = gr.Button("ล้างภาพ")
                    preview = gr.Image(label="ภาพที่เตรียมแล้ว · CNN 224 × 224 / SVM 128 × 128", height=180, interactive=False)
                with gr.Column():
                    result = gr.Markdown("### ผลจำแนก\nรอภาพสำหรับจำแนก", elem_id="summary")
                    scores = gr.Label(label="คะแนนโมเดลทั้ง 5 คลาส · อ่านผลจำแนกด้านบน", num_top_classes=5)
                    with gr.Accordion("ดูคะแนนเป็นตาราง", open=False):
                        probabilities = gr.Dataframe(headers=["คลาส", "คะแนน (%)"], datatype=["str", "number"], interactive=False)
            submit.click(predict_image, [image, model_choice], [scores, preview, result, probabilities], api_name="classify")
            clear.click(lambda: (None, None, None, "### ผลจำแนก\nรอภาพสำหรับจำแนก", []),
                        outputs=[image, scores, preview, result, probabilities], api_name=False)
            gr.Examples(examples=[[str(EXAMPLES / f"{label}.jpg")] for label in CLASSES],
                        inputs=image, label="ตัวอย่างภาพจากชุดทดสอบ · เลือกแล้วกดจำแนกภาพ", examples_per_page=5)
            gr.Markdown("**ขอบเขต:** จำแนกโรคใบข้าว 4 โรคและใบข้าวปกติ รวม 5 คลาส "
                        "Rice Blast ใช้เฉพาะไหม้ใบ ไม่รวมไหม้คอรวง หากคะแนนต่ำกว่าเกณฑ์จะแสดงว่าไม่แน่ใจ "
                        "ผลนี้เป็นตัวอย่างงานจำแนกภาพ ไม่ใช่การยืนยันโรคในแปลง")
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
            gr.Markdown(f"## ผลบนชุดทดสอบ 5 คลาส\n"
                        f"**CNN: {metrics['architecture']}** · **Accuracy {metrics['accuracy']:.2%}** · "
                        f"**Macro F1 {metrics['macro_f1']:.4f}** · **Weighted F1 {metrics['weighted_f1']:.4f}**\n\n"
                        f"Fit **{metrics['train_count']:,}** · Validation **{metrics['validation_count']:,}** · Test **{metrics['test_count']:,}** ภาพ\n\n"
                        f"เกณฑ์ความมั่นใจ **{metrics['threshold']:.0%}** · ตอบไม่แน่ใจ **{metrics['rejected']} ภาพ** · "
                        f"สัดส่วนที่ยอมตอบ **{metrics['coverage']:.2%}** · ความแม่นยำเฉพาะภาพที่ตอบ **{metrics['accepted_accuracy']:.2%}**\n\n"
                        f"ก่อนใช้เกณฑ์ไม่แน่ใจ: Accuracy **{metrics['raw_accuracy']:.2%}**, Macro F1 **{metrics['raw_macro_f1']:.4f}**\n\n"
                        "Macro F1 ให้น้ำหนักทุกคลาสเท่ากัน; ตัวเลขหลักคิดจากทุกภาพ รวมภาพที่ตอบไม่แน่ใจเป็นการจำแนกไม่ถูก")
            comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
            gr.Markdown("### เปรียบเทียบ CNN กับ SVM บน test เดียวกัน\nraw = เลือกคะแนนสูงสุด; threshold = CNN ตอบไม่แน่ใจเมื่อคะแนนต่ำกว่า 0.50")
            gr.Dataframe(value=comparison[["architecture", "policy", "accuracy", "macro_f1", "minimum_recall"]], interactive=False)
            gr.Dataframe(value=class_table, headers=["Class", "Fit", "Test", "Precision", "Recall", "F1"], interactive=False)
            gr.Markdown(f"F1 สูงสุด: **{display_label(best)}** · ต่ำสุด: **{display_label(worst)}**\n\n"
                        "Confusion matrix: แถวเป็นคลาสจริง คอลัมน์เป็นคำตอบ รวมคอลัมน์ไม่แน่ใจ")
            gr.Image(str(ARTIFACTS / "confusion_matrix.png"), label="Confusion matrix", interactive=False)
            gr.Image(str(ARTIFACTS / "class_comparison.png"), label="จำนวนภาพและคะแนนรายคลาส", interactive=False)
            gr.Markdown("**ข้อจำกัด:** เลือก CNN และเกณฑ์จาก validation กันกลุ่มภาพคล้ายข้ามชุดแล้ว "
                        "แต่ไม่มีรหัสใบ/แปลงจริง และ test เป็นชุดเดิมที่เคยใช้รายงานผลทดลองมาก่อน "
                        "คะแนนนี้จึงยังไม่ยืนยันความแม่นยำกับพื้นหลังหรือแปลงใหม่")
        with gr.Tab("วิธีใช้งาน · About"):
            gr.Markdown("## วิธีใช้งาน\n1. อัปโหลดภาพ JPG หรือ PNG ที่เห็นใบข้าวชัดเจน\n"
                        "2. กดจำแนกภาพ อ่านผลหรือข้อความไม่แน่ใจ พร้อมคะแนนทั้ง 5 คลาส\n"
                        "3. เลือก CNN หรือ SVM เพื่อทดลอง แล้วใช้แท็บเปรียบเทียบคลาสและ Evaluation ดูผลรายคลาส\n\n"
                        "## วิธีสร้างโมเดล\nใช้ DenseNet121 ที่ pretrain จาก ImageNet และเทียบกับ SVM ฟีเจอร์สี/เนื้อสัมผัส 404 มิติ "
                        "ฝึกหัวจำแนก 3 รอบ แล้ว fine-tune ทั้งโมเดล 12 รอบ เลือกรอบและสถาปัตยกรรมด้วย validation macro F1\n\n"
                        "เตรียมภาพ RGB 256 × 256 แล้วครอปกลาง 224 × 224 และ normalize แบบ ImageNet เหมือนตอนฝึก "
                        "เพิ่มแสง สี มุม และการครอปแบบสุ่มเฉพาะ train; ไม่เพิ่มภาพให้ validation/test\n\n"
                        "## ข้อมูล\nใช้ 5 คลาสจาก dataset เดิม: Rice Blast (เฉพาะ Leaf Blast), Bacterial Leaf Blight, "
                        "Sheath Blight, Brown Spot และ Healthy Rice Leaf ไม่ใช้ Rice_Leaf_AUG\n\n"
                        "[แหล่งข้อมูลเดิม](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection) · "
                        "[แหล่งข้อมูลเพิ่มเติม](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset)")
    return app


def main():
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860" if os.environ.get("SPACE_ID") else "7861")), share=False)


if __name__ == "__main__":
    main()
