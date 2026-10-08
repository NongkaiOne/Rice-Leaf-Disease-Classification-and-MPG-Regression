"""Reproduce the SVM baseline from images; never requires deleted study caches."""
from pathlib import Path
import sys,json,argparse
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np,joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from rice_leaf_app.preprocessing import image_to_features
from rice_leaf_app.cnn_config import CLASSES

def features(frame):
 with ThreadPoolExecutor(max_workers=4) as pool:return np.asarray(list(pool.map(image_to_features,[ROOT/p for p in frame.path])))
def evaluate(model,frame):
 prob=model.predict_proba(features(frame));pred=model.classes_[prob.argmax(1)];truth=frame.label.to_numpy();labels=list(CLASSES)
 report=classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0);recall={c:report[c]['recall'] for c in labels}
 result=dict(architecture='baseline_svm',policy='raw',threshold=0,accuracy=float(np.mean(truth==pred)),macro_f1=float(f1_score(truth,pred,labels=labels,average='macro')),weighted_f1=float(f1_score(truth,pred,labels=labels,average='weighted')),minimum_recall=min(recall.values()),minimum_recall_class=min(recall,key=recall.get),coverage=1.,accepted_accuracy=float(np.mean(truth==pred)),rejected=0,classes=labels,report=report,confusion_matrix=confusion_matrix(truth,pred,labels=labels).tolist(),train_count=8186,validation_count=1368,test_count=len(frame))
 (P/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');out=frame[['path','label']].copy();out['predicted']=pred;out['confidence']=prob.max(1);out.to_csv(P/'predictions.csv',index=False);print('SVM test',result['accuracy'],result['macro_f1'],flush=True)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args();frame=pd.read_csv(P.parent/'cnn5/manifest.csv');dest=ROOT/'rice_leaf_app/models/rice_svm.joblib'
 if not args.evaluate_only:
  dev=frame[frame.cnn_split.ne('test')];x=features(dev);y=dev.label.to_numpy();fit=dev.cnn_split.eq('fit');val=dev.cnn_split.eq('validation');options=[]
  for C in [1.,10.]:
   model=make_pipeline(StandardScaler(),SVC(C=C,class_weight='balanced',cache_size=512));model.fit(x[fit],y[fit]);options.append(dict(C=C,validation_macro_f1=float(f1_score(y[val],model.predict(x[val]),average='macro'))))
  C=max(options,key=lambda r:r['validation_macro_f1'])['C'];model=make_pipeline(StandardScaler(),SVC(C=C,class_weight='balanced',probability=True,random_state=42,cache_size=512));model.fit(x[fit],y[fit]);joblib.dump(dict(model=model,feature='original128',candidates=options),dest,compress=3)
  (P/'selection.json').write_text(json.dumps(dict(svm_C=C,svm_candidates=options,fit=int(fit.sum()),validation=int(val.sum()),selection='validation only'),indent=2),encoding='utf-8')
 evaluate(joblib.load(dest)['model'],frame[frame.cnn_split.eq('test')])
if __name__=='__main__':main()
