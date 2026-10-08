"""Generate and execute separate CNN and SVM notebooks with real saved outputs."""
from pathlib import Path
import sys,json,os
import nbformat as n
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]
def build(kind):
 cells=[]
 def md(s):cells.append(n.v4.new_markdown_cell(s))
 def code(s):cells.append(n.v4.new_code_cell(s))
 md(f'''# Rice Leaf Classification — {kind}

Supervised image classification: จำแนกโรคใบข้าว 4 โรคและใบปกติ เพื่อช่วยคัดกรองภาพเบื้องต้นและศึกษาความแตกต่างระหว่างคลาส ไม่ยืนยันโรคในแปลง

คลาส: Rice Blast (Leaf Blast เท่านั้น), Bacterial Leaf Blight, Sheath Blight, Brown Spot, Healthy Rice Leaf

ข้อมูลมาจาก [loki4514](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection) (ประกาศ Apache 2.0) และ [alamshihab075](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) (ประกาศ MIT). ดูรายละเอียดการอ้างอิงและข้อจำกัดใน DATASET.md

Notebook รันตามลำดับเพื่อทบทวนโมเดลที่ฝึกจริง เปิด RETRAIN เฉพาะเมื่อต้องการฝึกซ้ำ''')
 code('''from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from IPython.display import display,Markdown,Image
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
ROOT=Path.cwd().resolve()
if not (ROOT/'rice_leaf_app').is_dir():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from rice_leaf_app.cnn_config import CLASSES,EXAMPLES,ARTIFACTS
P=ROOT/'experiments/cnn5'
frame=pd.read_csv(P/'manifest.csv')
display(pd.DataFrame([{'label':k,'English':v[0],'Thai':v[1]} for k,v in CLASSES.items()]))
display(pd.crosstab(frame.label,frame.cnn_split))
display(pd.read_csv(ARTIFACTS/'class_counts.csv'))''')
 md('''## สำรวจ เตรียม และแบ่งข้อมูล

เก็บภาพต้นทาง 5 คลาส 16,533 ไฟล์ แยก train/test ตามโฟลเดอร์ ไม่ใช้ Rice_Leaf_AUG หลังคัดภาพเสีย/ซ้ำ/เสี่ยงรั่ว ใช้ fit 8,186, validation 1,368, test 2,228 ภาพ

แก้ EXIF orientation และวางภาพโปร่งใสบนพื้นขาว ตรวจ decoded RGB hash และกลุ่มภาพคล้ายจาก pHash/RGB หลังหมุน/พลิก เก็บ manifest ที่ระบุ path, label, hash และ split เพื่อทำซ้ำ validation ประมาณ 14.3% ด้วย StratifiedGroupKFold 7 folds seed 42 ไม่มีการใช้ test ฝึกหรือจูนในรอบนี้

กลุ่มความคล้ายไม่ใช่รหัสใบจริง ชุด test เดิมเคยรายงานผลในงานทดลองก่อน จึงไม่ใช่ชุดอิสระใหม่''')
 code('''for a,b in [('fit','validation'),('fit','test'),('validation','test')]:
    assert not set(frame.loc[frame.cnn_split.eq(a),'similarity_group']) & set(frame.loc[frame.cnn_split.eq(b),'similarity_group'])
assert all((ROOT/p).is_file() for p in frame.path)
print('All dataset paths exist; similarity groups do not cross splits')
audit=pd.read_csv(ROOT/'data_audit/dataset_manifest.csv')
display(audit.groupby(['label','status']).size().rename('images').reset_index())
for label in CLASSES:
    display(Markdown(CLASSES[label][0]));display(Image(filename=str(EXAMPLES/f'{label}.jpg'),width=180))''')
 if kind=='CNN':
  md('''## สกัดลักษณะและฝึก DenseNet121

เลือก CNN pretrained เพื่อเรียนรู้ลักษณะแผลและเนื้อสัมผัสจากภาพแทนฟีเจอร์สีที่กำหนดเอง ใช้ ImageNet, เปลี่ยนหัว 5 คลาส, RGB resize 256 → crop 224, normalize mean [0.485,0.456,0.406], std [0.229,0.224,0.225]

Train มี random crop/flip/rotation/color jitter/blur; validation/test ใช้ center crop ไม่สุ่ม ฝึกหัว 3 รอบ lr 0.001, fine-tune 12 รอบ lr 0.0001, AdamW, class weights, label smoothing 0.05. เลือก checkpoint ด้วย validation macro F1

โค้ดครบใน experiments/cnn5/common.py, train.py, evaluate.py. ฝึกใหม่ต้องติดตั้ง requirements-training.txt และ PyTorch สำหรับเครื่องนั้นก่อน''')
  code('''RETRAIN=False
if RETRAIN:
    import subprocess,os
    python=os.environ.get('CNN_PYTHON',sys.executable)
    for script in ['train.py','evaluate.py','report.py']:
        subprocess.run([python,str(P/script)],cwd=ROOT.parent,check=True)
display(pd.read_csv(P/'densenet121/history.csv'))
display(Image(filename=str(P/'training_curves.png')))
metadata=json.loads((ROOT/'rice_leaf_app/models/rice_cnn.json').read_text(encoding='utf-8'))
print('Selected checkpoint:',metadata['best_epoch'])
print('Saved ONNX model:',ROOT/'rice_leaf_app/models/rice_cnn.onnx')
print('Preprocessing:',{k:metadata[k] for k in ['resize','crop_size','mean','std','threshold']})''')
  modelpath="ARTIFACTS/'test_predictions.csv'";metricpath="ARTIFACTS/'metrics.json'"
 else:
  md('''## สกัดฟีเจอร์และฝึก SVM

SVM เป็น baseline สำหรับตรวจว่า CNN ให้ประโยชน์เพิ่มจากฟีเจอร์สี/เนื้อสัมผัสหรือไม่ RGB 128×128 LANCZOS, HSV histogram ทั้งภาพและ 2×2, RGB mean/std + gradient histogram รวม 404 มิติ

StandardScaler + RBF SVC, class_weight=balanced, gamma=scale. ลอง C=1 และ 10 แล้วเลือกด้วย validation macro F1 ได้ C=10. probability=True, seed 42; scaler อยู่ใน pipeline ที่บันทึก ไม่ใช้ test ปรับพารามิเตอร์

โค้ด train/evaluation ครบใน experiments/svm/train.py และฟีเจอร์อยู่ rice_leaf_app/preprocessing.py ไม่ใช้ cache จากการทดลองที่ลบไป''')
  code('''RETRAIN=False
if RETRAIN:
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'experiments/svm/train.py')],cwd=ROOT.parent,check=True)
import joblib
pack=joblib.load(ROOT/'rice_leaf_app/models/rice_svm.joblib')
display(pd.DataFrame(pack['candidates']))
print(pack['model'])
from rice_leaf_app.preprocessing import image_to_features
print('Feature dimensions:',image_to_features(EXAMPLES/'healthy.jpg').shape)
print('Saved pipeline includes StandardScaler and trained SVC')''')
  modelpath="ROOT/'experiments/svm/predictions.csv'";metricpath="ROOT/'experiments/svm/metrics.json'"
 md('''## ประเมิน test และความหมาย

คำนวณคะแนนใหม่จากผลทำนายที่โมเดลรันจริงและตรวจเทียบกับรายงาน Accuracy คือสัดส่วนถูก, macro F1 เฉลี่ยทุกคลาสเท่ากัน, weighted F1 ถ่วงตามจำนวนภาพ; recall บอกสัดส่วนภาพจริงของคลาสที่หาเจอ

CNN ตอบไม่แน่ใจเมื่อคะแนนต่ำกว่า 0.50 (เลือกจาก validation) และนับเป็นจำแนกไม่ถูกใน accuracy/recall หลัก SVM baseline เลือกคะแนนสูงสุด ไม่มีเกณฑ์ปฏิเสธ''')
 code(f'''pred=pd.read_csv({modelpath})
metrics=json.loads(({metricpath}).read_text(encoding='utf-8'))
labels=list(CLASSES)
assert abs(accuracy_score(pred.label,pred.predicted)-metrics['accuracy'])<1e-12
assert abs(f1_score(pred.label,pred.predicted,labels=labels,average='macro')-metrics['macro_f1'])<1e-12
display(pd.Series({{k:metrics[k] for k in ['accuracy','macro_f1','minimum_recall','minimum_recall_class']}}))
display(pd.DataFrame(classification_report(pred.label,pred.predicted,labels=labels,output_dict=True,zero_division=0)).T)
display(pd.DataFrame(confusion_matrix(pred.label,pred.predicted,labels=labels+['uncertain'])[:5],index=labels,columns=labels+['uncertain']))
display(pd.read_csv(ARTIFACTS/'model_comparison.csv'))''')
 md('''## ตัวอย่างผลทำนายและวิเคราะห์ข้อผิดพลาด

CNN raw accuracy 97.67% เทียบ SVM 89.00% บน split เดียวกัน; CNN เมื่อใช้เกณฑ์ accuracy 97.58%, recall ต่ำสุด 96.68%. CNN ยังสับสนระหว่าง Rice Blast และ Healthy เป็นหลัก ส่วน SVM recall ต่ำสุดเป็น Sheath Blight (82.44%). Bacterial Leaf Blight ที่ CNN ได้ 100% เป็นผลเฉพาะ test นี้ ไม่รับประกันภาพใหม่

คะแนนสูงยังไม่ยืนยันความทนพื้นหลังจริง ไม่มีชุดใบเดียวกันหลายพื้นหลังหรือ field IDs; การทดสอบภาพจริงถูกข้ามตามคำสั่ง''')
 code('''errors=pred[pred.label.ne(pred.predicted)]
display(errors.groupby(['label','predicted']).size().rename('images').reset_index())
for row in errors.head(3).itertuples():
    display(Markdown(f'True: {row.label}; predicted: {row.predicted}'))
    display(Image(filename=str(ROOT/row.path),width=220))''')
 code(('from rice_leaf_app.cnn_inference import get_predictor\nmodel=get_predictor()' if kind=='CNN' else 'from rice_leaf_app.svm_inference import get_svm_predictor\nmodel=get_svm_predictor()')+'''
for label in CLASSES:
    scores,preview,summary,rows=model.predict(EXAMPLES/f'{label}.jpg')
    display(Markdown(f'True: {CLASSES[label][0]}'))
    display(Markdown(summary))
    display(pd.DataFrame(rows,columns=['Class','Score (%)']))''')
 md('''## การนำไปใช้

Gradio app ใช้ไฟล์โมเดลที่บันทึกและ preprocessing เดียวกับตอนฝึก เลือก CNN หรือ SVM ในหน้า Classification แล้วดูผลเปรียบเทียบใน Evaluation เปิดเว็บจากลิงก์ README ได้โดยไม่ต้องเปิดเครื่องนักศึกษา ดูวิธีเผยแพร่และตรวจเว็บใน RENDER.md

ข้อมูลโมเดล/ผลรายคลาส/ข้อจำกัดและหลักฐานใน experiments/cnn5/RESULTS.md''')
 return n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'rice-python','display_name':'Rice Python','language':'python'}})

def main():
 kernel_root=ROOT.parent/'tmp/kernels';d=kernel_root/'rice-python';d.mkdir(parents=True,exist_ok=True);(d/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Rice Python','language':'python'}),encoding='utf-8');os.environ['IPYTHONDIR']=str(ROOT.parent/'tmp/ipython-cnn5')
 for kind,file in [('CNN','rice_leaf_classification.ipynb'),('SVM','rice_leaf_svm.ipynb')]:
  nb=build(kind);manager=KernelManager(kernel_name='rice-python',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]));client=NotebookClient(nb,km=manager,timeout=300,resources={'metadata':{'path':str(ROOT)}})
  try:client.execute();n.write(nb,ROOT/'notebooks'/file)
  finally:
   if manager.has_kernel:manager.shutdown_kernel(now=True)
  print('Executed',file,flush=True)
if __name__=='__main__':main()
