"""Five-class SVM using the exact saved 404-dimensional feature pipeline."""
from functools import lru_cache
import joblib,numpy as np
from rice_leaf_app.cnn_config import APP_DIR,CLASSES,display_label
from rice_leaf_app.preprocessing import preprocess_image,features_from_processed
class SVMPredictor:
 def __init__(self):
  self.model=joblib.load(APP_DIR/'models/rice_svm.joblib')['model'];self.classes=list(self.model.classes_)
  assert set(self.classes)==set(CLASSES)
 def predict(self,image):
  rgb=preprocess_image(image);probs=self.model.predict_proba(features_from_processed(rgb)[None])[0]
  scores={display_label(k):float(v) for k,v in zip(self.classes,probs)};i=int(probs.argmax())
  summary=f'### SVM: {display_label(self.classes[i])}\nคะแนนโมเดล **{probs[i]:.1%}**\n\nSVM baseline เลือกคลาสคะแนนสูงสุด ไม่มีเกณฑ์ไม่แน่ใจ; คะแนนไม่ใช่ความแน่นอนของโรค'
  rows=[[display_label(self.classes[i]),round(float(probs[i])*100,2)] for i in np.argsort(-probs)]
  return scores,rgb,summary,rows
@lru_cache(maxsize=1)
def get_svm_predictor():return SVMPredictor()
