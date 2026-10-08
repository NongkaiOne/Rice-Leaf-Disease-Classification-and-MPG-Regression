"""ImageNet CNNs for the confirmed five-class rice-leaf task."""
from pathlib import Path
import json,sys,os,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
os.environ.setdefault('TORCH_HOME',str(ROOT.parent/'tmp/torch-models'))
import numpy as np,pandas as pd
from PIL import Image,ImageOps
import torch
from torch import nn
from torchvision import models,transforms
from torch.utils.data import Dataset
CLASSES=json.loads((P/'classes.json').read_text(encoding='utf-8-sig'));LABELS=[c['label'] for c in CLASSES]
MEAN=[.485,.456,.406];STD=[.229,.224,.225]
CACHE=ROOT.parent/'tmp/cnn5-rgb256';CACHE.mkdir(exist_ok=True)
def read_rgb(path):
 with Image.open(path) as im:
  im=ImageOps.exif_transpose(im)
  if im.mode in ('RGBA','LA') or 'transparency' in im.info:
   rgba=im.convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba)
  return im.convert('RGB')
def cache_path(row):return CACHE/(hashlib.sha256((row.path+row.pixel_sha256).encode()).hexdigest()+'.png')
def cache_one(row):
 path=cache_path(row)
 if not path.exists():read_rgb(ROOT/row.path).resize((256,256),Image.Resampling.BILINEAR).save(path)
 return path
def build(name,pretrained=True):
 if name=='densenet121':
  model=models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None);model.classifier=nn.Linear(model.classifier.in_features,len(LABELS))
 else:raise ValueError(name)
 return model
def transform(training):
 if training:return transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1.),ratio=(.9,1.1)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(15,fill=(128,128,128)),transforms.ColorJitter(.15,.15,.1,.03),transforms.RandomApply([transforms.GaussianBlur(3,sigma=(.1,1.))],p=.15),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
 return transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
class Images(Dataset):
 def __init__(self,frame,training=False):self.rows=[(str(cache_path(row)),LABELS.index(row.label)) for row in frame.itertuples()];self.tf=transform(training)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  path,label=self.rows[i]
  with Image.open(path) as im:x=self.tf(im.convert('RGB'))
  return x,label
def metadata(name):return dict(architecture=name,classes=LABELS,display_classes=CLASSES,resize=[256,256],crop_size=224,interpolation='bilinear',mean=MEAN,std=STD,alpha_background='white',exif_transpose=True,feature_scope='full image; no background removal',weights_source='ImageNet1K_V1')
