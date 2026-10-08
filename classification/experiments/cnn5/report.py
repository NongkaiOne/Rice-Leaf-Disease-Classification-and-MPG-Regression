"""Refresh the two retained models' reports and application artifacts."""
from pathlib import Path
import json,shutil
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parents[1];A=ROOT/'rice_leaf_app';ART=A/'artifacts_cnn5';S=P.parent/'svm'
def table(df):
 return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)])
def main():
 raw=json.loads((P/'densenet121_raw_metrics.json').read_text(encoding='utf-8'));m=json.loads((P/'densenet121_threshold_metrics.json').read_text(encoding='utf-8'));svm=json.loads((S/'metrics.json').read_text(encoding='utf-8'));meta=json.loads((A/'models/rice_cnn.json').read_text(encoding='utf-8'));m.update(raw_accuracy=raw['accuracy'],raw_macro_f1=raw['macro_f1'],model_sha256=meta['onnx_sha256'])
 (ART/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');shutil.copy2(P/'densenet121_threshold_predictions.csv',ART/'test_predictions.csv');shutil.copy2(P/'manifest.csv',ART/'split_manifest.csv')
 columns=['architecture','policy','accuracy','macro_f1','minimum_recall','coverage','rejected'];comparison=pd.DataFrame([{k:d[k] for k in columns} for d in [raw,m,svm]]);comparison.to_csv(P/'comparison.csv',index=False);comparison.to_csv(ART/'model_comparison.csv',index=False)
 per=pd.DataFrame([dict(label=c,**m['report'][c]) for c in m['classes']]);per.to_csv(ART/'per_class_metrics.csv',index=False)
 history=pd.read_csv(P/'densenet121/history.csv');fig,ax=plt.subplots(1,2,figsize=(11,4));x=np.arange(1,len(history)+1);ax[0].plot(x,history.train_loss);ax[1].plot(x,history.validation_macro_f1)
 for a in ax:a.axvline(3.5,linestyle=':',color='gray');a.set_xlabel('Epoch: head 1-3 / fine-tune 4-15')
 ax[0].set_ylabel('Training loss');ax[1].set_ylabel('Validation macro F1');fig.tight_layout();fig.savefig(P/'training_curves.png',dpi=130);plt.close(fig)
 cm=np.array(m['confusion_matrix']);labels=[c['english'] for c in meta['display_classes']];fig,ax=plt.subplots(figsize=(10,7));im=ax.imshow(cm,cmap='Greens');fig.colorbar(im,ax=ax);ax.set_xticks(range(6),labels+['Uncertain'],rotation=30,ha='right');ax.set_yticks(range(5),labels);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
 for i in range(5):
  for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
 fig.tight_layout();fig.savefig(ART/'confusion_matrix.png',dpi=130);plt.close(fig)
 preds=pd.read_csv(ART/'test_predictions.csv');errors=preds[preds.label.ne(preds.predicted)];errors.to_csv(ART/'errors.csv',index=False)
 volumes=pd.DataFrame({'actual':preds.label.value_counts(),'predicted':preds.predicted.value_counts()}).fillna(0).astype(int).rename_axis('label').reset_index();volumes.to_csv(ART/'prediction_counts.csv',index=False)
 text=f'''# ผลเปรียบเทียบ CNN กับ SVM — 5 คลาส

โมเดลที่เก็บและเปิดให้ทดลอง: DenseNet121 fine-tune กับ SVM RBF ทั้งสองใช้ fit 8,186 / validation 1,368 / test 2,228 ภาพเดียวกัน ไม่ฝึกหรือปรับเกณฑ์จาก test

## การเตรียมข้อมูลและฝึก

CNN: ImageNet pretrain, RGB resize 256 แล้ว crop 224, normalize mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. Random crop/flip/rotation/color jitter/blur เฉพาะ train. ฝึกหัว 3 รอบ lr 0.001 แล้ว fine-tune 12 รอบ lr 0.0001, AdamW, weighted cross entropy, label smoothing 0.05; เลือก checkpoint ด้วย validation macro F1

SVM: RGB 128×128 แบบ LANCZOS, HSV histogram global/2×2, RGB mean/std และ gradient histogram รวม 404 ฟีเจอร์; StandardScaler + RBF SVC, class_weight=balanced. เลือก C จาก 1 และ 10 ด้วย validation macro F1 (ได้ C=10) และใช้ probability=True, seed 42

Manifest เก็บกลุ่มความคล้ายจาก pHash/RGB หลังหมุน/พลิก กลุ่มไม่ข้ามชุดแต่ไม่ใช่รหัสใบจริง เก็บการแบ่งเดิมเพื่อทำซ้ำ; test เคยถูกใช้รายงานผลในงานทดลองก่อน จึงไม่ใช่ holdout ใหม่

## ผลบน test

{table(comparison.round(5))}

raw = เลือกคลาสคะแนนสูงสุด; threshold = CNN ปฏิเสธเมื่อคะแนน <{m['threshold']:.2f}. เกณฑ์เลือกจาก validation โดย coverage ≥95% และ accuracy/recall ต่ำสุดลดไม่เกิน 1 จุดเปอร์เซ็นต์ ห้ามตีความคะแนนเป็นความน่าจะเป็นของโรคที่ผ่าน calibration

Macro F1 เฉลี่ยทุกคลาสเท่ากัน; weighted F1 ถ่วงตามจำนวนภาพ. Accuracy/recall หลักนับภาพไม่แน่ใจเป็นจำแนกไม่ถูก: CNN accuracy {m['accuracy']:.2%}, macro F1 {m['macro_f1']:.4f}, recall ต่ำสุด {m['minimum_recall']:.2%}; ยอมตอบ {m['coverage']:.2%}, accuracy เฉพาะภาพที่ตอบ {m['accepted_accuracy']:.2%}

## ผล CNN รายคลาส

{table(per.round(5))}

## ผล SVM รายคลาส

{table(pd.DataFrame([dict(label=c,**svm['report'][c]) for c in m['classes']]).round(5))}

## วิเคราะห์

CNN ก่อนปฏิเสธดีกว่า SVM {(raw['accuracy']-svm['accuracy'])*100:.2f} จุดเปอร์เซ็นต์. คลาส CNN ที่ recall ต่ำสุดคือ {m['minimum_recall_class']}; Bacterial Leaf Blight ถูก 318/318 เฉพาะ test นี้ คู่สับสนหลักคือใบปกติกับ Rice Blast (19 และ 14 ภาพ) เกณฑ์ 0.50 ปฏิเสธ 8 ภาพ: เดิมผิด 6 และถูก 2 จึงเพิ่มความแม่นยำเฉพาะภาพที่ตอบแต่ลด accuracy รวมเล็กน้อย

{table(volumes)}

ภาพฝึกคลาสใหญ่/เล็กต่างประมาณ 1.89 เท่า ใช้ class weights แล้ว ไม่จำเป็นต้องทิ้งภาพให้เท่ากัน ไม่มีชุดภาพภาคสนาม/ใบเดียวกันหลายพื้นหลัง จึงยังไม่ยืนยันว่าทนพื้นหลังใหม่ แม้ recall ภายใน dataset ผ่าน 85% ทุกคลาส; ข้าม accuracy ภาพจริงตามคำสั่งเดิม

## หลักฐาน

- checkpoint CNN: densenet121/best.pt; โมเดลใช้งาน: rice_leaf_app/models/rice_cnn.onnx + rice_cnn.json
- SVM: rice_leaf_app/models/rice_svm.joblib รวม scaler; โค้ดฟีเจอร์: rice_leaf_app/preprocessing.py
- ประวัติฝึกและ validation: densenet121/history.csv, validation_report.json; SVM: ../svm/selection.json
- train/test predictions: ไฟล์ CSV ในโฟลเดอร์โมเดล; ตัวอย่างข้อผิดพลาด: rice_leaf_app/artifacts_cnn5/errors.csv
- Notebook: ../../notebooks/rice_leaf_classification.ipynb และ rice_leaf_svm.ipynb

ประวัติการเลือก CNN เคยเทียบ EfficientNet-B0 ด้วย validation และเลือก DenseNet121 ก่อนดู test; โมเดลและการทดลองที่ไม่ได้ใช้ถูกลบตามคำขอ ไม่ย้อนเลือกจาก test
'''
 (P/'RESULTS.md').write_text(text,encoding='utf-8');print('Refreshed CNN/SVM reports')
if __name__=='__main__':main()
