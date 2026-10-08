from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pandas as pd
from PIL import Image,ImageOps
p=Path('classification/experiments/cnn5');root=Path('classification');cache=Path('tmp/cnn5-rgb256');cache.mkdir(exist_ok=True)
f=pd.read_csv(p/'manifest.csv');f=f[f.cnn_split.ne('test')]
def one(row):
 target=cache/(hashlib.sha256((row.path+row.pixel_sha256).encode()).hexdigest()+'.png')
 if not target.exists():
  with Image.open(root/row.path) as im:
   im=ImageOps.exif_transpose(im)
   if im.mode in ('RGBA','LA') or 'transparency' in im.info:
    rgba=im.convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba)
   im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR).save(target)
with ThreadPoolExecutor(max_workers=6) as pool:
 for i,_ in enumerate(pool.map(one,f.itertuples()),1):
  if i%1000==0:print('CNN train/validation cache',i,'/',len(f),flush=True)
print('Cache complete; test images not read.')
