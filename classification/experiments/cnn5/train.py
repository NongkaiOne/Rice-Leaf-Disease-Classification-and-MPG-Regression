"""Train head 3 epochs then fine-tune 12; select checkpoint only by validation macro F1."""
from pathlib import Path
import argparse,json,time,random,hashlib,os
from concurrent.futures import ThreadPoolExecutor
from common import *
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,classification_report

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def run(args):
 seed_all(42);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
 if not torch.cuda.is_available() and not args.allow_cpu:raise RuntimeError('CUDA required for this full training run; use --allow-cpu explicitly for a smoke test')
 frame=pd.read_csv(P/'manifest.csv');train=frame[frame.cnn_split.eq('fit')];val=frame[frame.cnn_split.eq('validation')]
 for split in ['fit','validation','test']:assert set(frame.loc[frame.cnn_split.eq(split),'label'])==set(LABELS)
 for left,right in [('fit','validation'),('fit','test'),('validation','test')]:assert not set(frame.loc[frame.cnn_split.eq(left),'similarity_group'])&set(frame.loc[frame.cnn_split.eq(right),'similarity_group'])
 # Prepare training/validation images only; do not read test images here.
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,_ in enumerate(pool.map(cache_one,pd.concat([train,val]).itertuples()),1):
   if i%1000==0:print('Cached train/validation',i,flush=True)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=build(args.arch).to(device)
 loaders={k:DataLoader(Images(part,k=='fit'),batch_size=args.batch_size,shuffle=k=='fit',num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0) for k,part in [('fit',train),('validation',val)]}
 counts=train.label.value_counts();weight=torch.tensor([len(train)/(len(LABELS)*counts[c]) for c in LABELS],dtype=torch.float32,device=device);criterion=nn.CrossEntropyLoss(weight=weight,label_smoothing=.05)
 scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');best=-1.;history=[];dest=P/args.arch;dest.mkdir(exist_ok=True)
 config=dict(**metadata(args.arch),seed=42,head_epochs=args.head_epochs,finetune_epochs=args.finetune_epochs,head_lr=.001,finetune_lr=.0001,batch_size=args.batch_size,fit_count=len(train),validation_count=len(val),class_weights=weight.cpu().tolist(),label_smoothing=.05,selection='maximum validation macro F1 across both phases',manifest_sha256=hashlib.sha256((P/'manifest.csv').read_bytes()).hexdigest(),device=str(device),torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None)
 (dest/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
 for phase,epochs,lr in [('head',args.head_epochs,.001),('finetune',args.finetune_epochs,.0001)]:
  for param in model.parameters():param.requires_grad=phase=='finetune'
  for param in model.classifier.parameters():param.requires_grad=True
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.0001)
  for epoch in range(1,epochs+1):
   start=time.time();model.train()
   if phase=='head':model.features.eval() # Freeze backbone batch-normalization statistics too.
   losses=[]
   for images,labels in loaders['fit']:
    images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):logits=model(images);loss=criterion(logits,labels)
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
   model.eval();truth=[];prob=[]
   with torch.inference_mode():
    for images,labels in loaders['validation']:
     with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):logits=model(images.to(device,non_blocking=True))
     prob.extend(logits.float().softmax(1).cpu().numpy());truth.extend(labels.numpy())
   prob=np.asarray(prob);pred=prob.argmax(1);score=float(f1_score(truth,pred,labels=np.arange(len(LABELS)),average='macro',zero_division=0));row=dict(phase=phase,epoch=epoch,train_loss=float(np.mean(losses)),validation_accuracy=float(accuracy_score(truth,pred)),validation_macro_f1=score,seconds=time.time()-start);history.append(row);pd.DataFrame(history).to_csv(dest/'history.csv',index=False);print(args.arch,row,flush=True)
   if score>best:
    best=score;checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=config,best_epoch=row)
    torch.save(checkpoint,dest/'best.pt');np.savez_compressed(dest/'validation_predictions.npz',probabilities=prob,labels=np.asarray(truth));(dest/'validation_report.json').write_text(json.dumps(classification_report(truth,pred,labels=np.arange(len(LABELS)),target_names=LABELS,output_dict=True,zero_division=0),indent=2),encoding='utf-8')
 (dest/'complete.json').write_text(json.dumps(dict(best_validation_macro_f1=best,epochs=len(history),finished=True),indent=2),encoding='utf-8');print('CNN TRAIN COMPLETE',args.arch,best,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--arch',choices=['densenet121'],default='densenet121');parser.add_argument('--head-epochs',type=int,default=3);parser.add_argument('--finetune-epochs',type=int,default=12);parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--allow-cpu',action='store_true');run(parser.parse_args())
