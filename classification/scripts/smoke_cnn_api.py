"""Exercise both models via real HTTP, including preview and empty input."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx,numpy as np
from gradio_client import Client,handle_file
from rice_leaf_app.cnn_config import CLASSES,EXAMPLES,ARTIFACTS
from rice_leaf_app.cnn_inference import get_predictor
from rice_leaf_app.svm_inference import get_svm_predictor
url=sys.argv[1].rstrip('/') if len(sys.argv)>1 else 'http://127.0.0.1:7861'
client=Client(url,verbose=False);results=[]
for name,model in [('CNN (DenseNet121)',get_predictor()),('SVM',get_svm_predictor())]:
 for label in CLASSES:
  path=EXAMPLES/f'{label}.jpg';expected=model.predict(path);response=client.predict(handle_file(str(path)),name,api_name='/classify');assert len(response)==4;assert response[0]['label']==max(expected[0],key=expected[0].get);assert len(response[0]['confidences'])==5;assert Path(response[1]).is_file();assert len(response[3]['data'])==5;np.testing.assert_allclose(sum(c['confidence'] for c in response[0]['confidences']),1)
  results.append({'model':name,'example':label,'prediction':response[0]['label'],'status':'pass'})
 try:client.predict(None,name,api_name='/classify')
 except Exception as e:assert 'กรุณาอัปโหลดภาพ' in str(e)
 else:raise AssertionError('Empty image accepted')
with httpx.Client(timeout=30) as c:
 assert c.get(url).status_code==200;cfg=c.get(url+'/config').json();tabs=[x['props']['label'] for x in cfg['components'] if x['type']=='tabitem'];assert len(tabs)==4
result={'status':'pass','url':url,'examples':results,'empty_upload':'both models pass','tabs':tabs,'visual_browser_check':'not performed: browser tool failed to initialize'}
name='public_smoke_test.json' if url.startswith('https://') else 'smoke_test.json'
(ARTIFACTS/name).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('PASS: 2 models x 5 HTTP uploads, previews, empty inputs and 4 tabs')
