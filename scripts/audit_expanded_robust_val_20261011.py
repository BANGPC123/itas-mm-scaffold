"""Candidate regression audit across clean/blur/small/small+blur VAL views.
No TEST. These transformations are synthetic, not field-quality proof.
"""
from pathlib import Path
import json,importlib.util,torch,pandas as pd
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('robust',root/'scripts/finetune_recognizer_robustness.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
df=pd.read_csv(root/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False)
val=df[df.split=='VAL'];torch.set_num_threads(4)
loaders={k:DataLoader(mod.Crops(val,condition=k),batch_size=64,shuffle=False,num_workers=0)
 for k in ['clean','blur','small','small_blur']}
output={}
for name,path in [('deployed',root/'models/sign_recognizer/retrained_20261009.pt'),
                  ('expanded_candidate',root/'output/retraining_20261011/recognizer_hardneg_expanded/candidate.pt')]:
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    net=mobilenet_v3_small(num_classes=len(ckpt['classes'])).cuda()
    net.load_state_dict(ckpt['state_dict'])
    output[name]=mod.evaluate(net,loaders,'cuda:0')
    print('METRICS '+name+' '+json.dumps(output[name]),flush=True)
result={'purpose':'regression_validation_only', 'split':'VAL','test_touched':False,
 'synthetic_conditions':True,'metrics':output,'candidate_deployed':False}
dest=root/'output/evaluation/hardneg_expanded_robust_val_20261011.json'
dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
print('WRITTEN '+str(dest),flush=True)
