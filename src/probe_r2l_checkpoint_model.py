import inspect
from pathlib import Path
import torch
import direct_pose
p=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt")
x=torch.load(p,map_location='cpu',weights_only=False)
print('MODEL_KWARGS',repr(x.get('model_kwargs')))
print('CLASSES')
for n,v in sorted(vars(direct_pose).items()):
    if inspect.isclass(v) and getattr(v,'__module__',None)=='direct_pose':
        print(n, inspect.signature(v))
