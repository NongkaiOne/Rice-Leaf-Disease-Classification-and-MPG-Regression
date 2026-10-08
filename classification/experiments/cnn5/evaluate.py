"""Choose CNN/thresholds on validation, then evaluate five-class test and export ONNX."""
from common import *
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
import joblib

def metrics(truth,pred):
 n=len(LABELS);report=classification_report(truth,pred,labels=np.arange(n),target_names=LABELS,output_dict=True,zero_division=0);cm=confusion_matrix(truth,pred,labels=list(range(n))+[-1])[:n];accepted=pred>=0;recalls=[report[c]['recall'] for c in LABELS]
 return dict(accuracy=float(np.mean(truth==pred)),macro_f1=float(f1_score(truth,pred,labels=np.arange(n),average='macro',zero_division=0)),weighted_f1=float(f1_score(truth,pred,labels=np.arange(n),average='weighted',zero_division=0)),minimum_recall=min(recalls),minimum_recall_class=LABELS[int(np.argmin(recalls))],coverage=float(accepted.mean()),accepted_accuracy=float(np.mean(truth[accepted]==pred[accepted])) if accepted.any() else None,rejected=int((~accepted).sum()),classes=LABELS,confusion_matrix=cm.tolist(),confusion_columns=LABELS+['uncertain'],report=report)
def decisions(prob,t):return np.where(prob.max(1)>=t,prob.argmax(1),-1)
def threshold_from_validation(prob,truth):
 raw=metrics(truth,prob.argmax(1));rows=[]
 for t in np.round(np.arange(.3,.91,.05),2):rows.append(dict(threshold=float(t),**metrics(truth,decisions(prob,t))))
 feasible=[r for r in rows if r['coverage']>=.95 and r['accuracy']>=raw['accuracy']-.01 and r['minimum_recall']>=raw['minimum_recall']-.01]
 assert feasible,'No validation threshold meets coverage/recall constraints'
 winner=max(feasible,key=lambda r:(r['accepted_accuracy'],r['coverage']))
 return winner['threshold'],rows


def main():
 import hashlib,shutil
 torch.set_num_threads(4);name='densenet121';frame=pd.read_csv(P/'manifest.csv');test=frame[frame.cnn_split.eq('test')];fit=frame[frame.cnn_split.eq('fit')];val=frame[frame.cnn_split.eq('validation')]
 z=np.load(P/name/'validation_predictions.npz');threshold,curve=threshold_from_validation(z['probabilities'],z['labels'])
 (P/name/'threshold_validation_curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
 selection={'selected_architecture':name,'threshold':threshold,'selection':'fixed retained architecture; checkpoint and threshold selected from validation, before test'}
 (P/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(cache_one,test.itertuples()))
 loader=DataLoader(Images(test),batch_size=16,num_workers=2);checkpoint=torch.load(P/name/'best.pt',map_location='cpu',weights_only=True);model=build(name,False);model.load_state_dict(checkpoint['state_dict']);model.eval();device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device);prob=[]
 with torch.inference_mode():
  for x,_ in loader:prob.extend(model(x.to(device)).softmax(1).cpu().numpy())
 prob=np.asarray(prob);truth=np.array([LABELS.index(c) for c in test.label])
 for policy,t in [('raw',0),('threshold',threshold)]:
  pred=decisions(prob,t);m=metrics(truth,pred);m.update(train_count=len(fit),validation_count=len(val),test_count=len(test),threshold=t,architecture=name,policy=policy)
  (P/f'{name}_{policy}_metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8');out=test[['path','label']].copy();out['predicted']=[LABELS[i] if i>=0 else 'uncertain' for i in pred];out['confidence']=prob.max(1);out.to_csv(P/f'{name}_{policy}_predictions.csv',index=False)
 model.cpu();dest=ROOT/'rice_leaf_app/models/rice_cnn.onnx';torch.onnx.export(model,torch.zeros(1,3,224,224),str(dest),input_names=['image'],output_names=['logits'],dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
 import onnxruntime as ort
 sample=next(iter(loader))[0][:5];session=ort.InferenceSession(str(dest),providers=['CPUExecutionProvider'])
 with torch.inference_mode():expected=model(sample).numpy()
 observed=session.run(None,{'image':sample.numpy()})[0];np.testing.assert_allclose(expected,observed,rtol=1e-3,atol=1e-4)
 meta=checkpoint['metadata'];meta.update(threshold=threshold,best_epoch=checkpoint['best_epoch'],selection=selection,onnx_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),onnx_parity_max_abs_error=float(np.max(np.abs(expected-observed))))
 dest.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print('CNN evaluated and exported. Run report.py to refresh app reports.')
if __name__=='__main__':main()
