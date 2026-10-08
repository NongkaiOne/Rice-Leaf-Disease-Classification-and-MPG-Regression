"""Verify active CNN artifacts, all test predictions, and app construction."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
from rice_leaf_app.cnn_config import ROOT,APP_DIR,ARTIFACTS,EXAMPLES,CLASSES
from rice_leaf_app.cnn_inference import get_predictor

def main():
 model=get_predictor();metrics=json.loads((ARTIFACTS/'metrics.json').read_text());frame=pd.read_csv(ARTIFACTS/'split_manifest.csv');pred=pd.read_csv(ARTIFACTS/'test_predictions.csv');test=frame[frame.cnn_split.eq('test')]
 assert set(test.label)==set(CLASSES) and len(CLASSES)==5
 assert pred.path.tolist()==test.path.tolist()
 assert abs(np.mean(pred.label==pred.predicted)-metrics['accuracy'])<1e-12
 assert abs(f1_score(pred.label,pred.predicted,labels=list(CLASSES),average='macro')-metrics['macro_f1'])<1e-12
 for left,right in [('fit','validation'),('fit','test'),('validation','test')]:assert not set(frame.loc[frame.cnn_split.eq(left),'similarity_group'])&set(frame.loc[frame.cnn_split.eq(right),'similarity_group'])
 start=time.time();mismatches=[]
 for i,row in enumerate(pred.itertuples(),1):
  probs,rgb=model.probabilities(ROOT/row.path);assert rgb.shape==(224,224,3) and abs(float(probs.sum())-1)<1e-5
  actual=model.classes[int(probs.argmax())] if probs.max()>=model.threshold else 'uncertain'
  if actual!=row.predicted:mismatches.append(dict(path=row.path,expected=row.predicted,actual=actual))
  if i%500==0:print('CNN full-test inference verified',i,flush=True)
 assert not mismatches,str(mismatches[:10])
 try:model.predict(None)
 except ValueError as exc:assert 'กรุณาอัปโหลดภาพ' in str(exc)
 else:raise AssertionError('None image must fail')
 for label in CLASSES:
  scores,rgb,summary,rows=model.predict(EXAMPLES/f'{label}.jpg');assert len(scores)==len(rows)==5 and summary
 from rice_leaf_app.app import build_app
 app=build_app();fn=next(f.fn for f in app.fns.values() if getattr(f.fn,'__name__','')=='compare_classes')
 for left in CLASSES:
  for right in CLASSES:
   a,b,rows,note=fn(left,right);assert Path(a).exists() and Path(b).exists() and len(rows)==7 and note
 result=dict(status='pass',classes=list(CLASSES),test_images_verified=len(pred),onnx_predictions_match_report=True,empty_upload='pass',all_25_class_pairs='pass',seconds=time.time()-start,backend='ONNX Runtime CPU')
 (ARTIFACTS/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(result,flush=True)
if __name__=='__main__':main()
