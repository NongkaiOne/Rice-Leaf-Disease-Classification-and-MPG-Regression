"""CNN inference via ONNX Runtime CPU, with training-identical preprocessing."""
from functools import lru_cache
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageOps
import onnxruntime as ort
from rice_leaf_app.cnn_config import MODEL_PATH,METADATA_PATH,CLASSES,display_label

def prepare_image(image,metadata):
 if image is None:raise ValueError('กรุณาอัปโหลดภาพก่อนจำแนก')
 if isinstance(image,(str,Path)):
  with Image.open(image) as opened:image=ImageOps.exif_transpose(opened).copy()
 elif isinstance(image,np.ndarray):image=Image.fromarray(image)
 elif isinstance(image,Image.Image):image=ImageOps.exif_transpose(image).copy()
 else:raise TypeError('รูปแบบภาพไม่รองรับ กรุณาใช้ JPG หรือ PNG')
 if image.mode in ('RGBA','LA') or 'transparency' in image.info:
  rgba=image.convert('RGBA');image=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba)
 image=image.convert('RGB').resize(tuple(metadata['resize']),Image.Resampling.BILINEAR)
 size=int(metadata['crop_size']);left=round((image.width-size)/2);top=round((image.height-size)/2);image=image.crop((left,top,left+size,top+size));rgb=np.asarray(image)
 x=rgb.astype(np.float32)/255.;x=(x-np.asarray(metadata['mean'],np.float32))/np.asarray(metadata['std'],np.float32)
 return rgb,np.ascontiguousarray(x.transpose(2,0,1)[None],dtype=np.float32)

class CNNPredictor:
 def __init__(self,model_path=MODEL_PATH,metadata_path=METADATA_PATH):
  self.metadata=json.loads(Path(metadata_path).read_text(encoding='utf-8'));self.classes=self.metadata['classes']
  if self.classes!=list(CLASSES):raise ValueError('รายชื่อคลาสของโมเดลไม่ตรงกับแอป')
  if 'onnx_sha256' in self.metadata and hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=self.metadata['onnx_sha256']:raise ValueError('ไฟล์โมเดลไม่ตรงกับ metadata')
  opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
  self.session=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider']);self.threshold=float(self.metadata['threshold'])
 def probabilities(self,image):
  rgb,x=prepare_image(image,self.metadata);logits=self.session.run(None,{'image':x})[0][0];probs=np.exp(logits-logits.max());probs/=probs.sum()
  if len(probs)!=len(self.classes) or not np.isfinite(probs).all():raise ValueError('คะแนนจากโมเดลไม่ถูกต้อง')
  return probs,rgb
 def predict(self,image):
  probabilities,processed=self.probabilities(image);index=int(probabilities.argmax());winner=self.classes[index];confidence=float(probabilities[index]);uncertain=confidence<self.threshold
  scores={display_label(label):float(prob) for label,prob in zip(self.classes,probabilities)}
  rows=[[display_label(self.classes[i]),round(float(probabilities[i])*100,2)] for i in np.argsort(-probabilities)]
  if uncertain:result=f'### ผลจำแนก: ไม่แน่ใจ\nคะแนนสูงสุด **{confidence:.1%}** ต่ำกว่าเกณฑ์ **{self.threshold:.0%}**\n\nคลาสที่ได้คะแนนสูงสุด: {display_label(winner)}'
  else:result=f'### ผลจำแนก: {display_label(winner)}\nคะแนนโมเดล **{confidence:.1%}**'
  result+='\n\nคะแนนเป็นค่าประมาณจากโมเดล ไม่ใช่ความแน่นอนว่าภาพมีโรคนั้น'
  return scores,processed,result,rows

@lru_cache(maxsize=1)
def get_predictor():
 if not MODEL_PATH.exists() or not METADATA_PATH.exists():raise FileNotFoundError('ยังไม่มีโมเดล CNN ที่ฝึกและประเมินเสร็จ')
 return CNNPredictor()
