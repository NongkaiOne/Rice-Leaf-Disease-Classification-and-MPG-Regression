from pathlib import Path
import sys
P=Path('classification/experiments/cnn5').resolve();ROOT=P.parents[1];sys.path.insert(0,str(P));sys.path.insert(0,str(ROOT))
from common import read_rgb,transform,metadata,build
from rice_leaf_app.cnn_inference import prepare_image
import numpy as np,pandas as pd
f=pd.read_csv(P/'manifest.csv');samples=f[f.cnn_split.eq('validation')].groupby('label',group_keys=False).head(2)
for row in samples.itertuples():
 image=read_rgb(ROOT/row.path).resize((256,256));expected=transform(False)(image).numpy()
 # Explicit bilinear matches training cache (PIL resize defaults to bicubic, so specify it).
 from PIL import Image
 image=read_rgb(ROOT/row.path).resize((256,256),Image.Resampling.BILINEAR);expected=transform(False)(image).numpy()
 rgb,actual=prepare_image(ROOT/row.path,metadata('densenet121'));np.testing.assert_allclose(expected,actual[0],rtol=1e-6,atol=1e-6);assert rgb.shape==(224,224,3)
print('Training/serving preprocessing matches on 10 validation images.',flush=True)
model=build('densenet121');print('DenseNet121 ImageNet weights cached.',flush=True)
