import inspect
import json
from pathlib import Path
import torch
from canvas_data import CanvasDataset
split_path = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
split = json.loads(split_path.read_text(encoding="utf-8"))
name = split["splits"]["fit"][0]
ds = CanvasDataset([name], real_prob=0.0, seed=20260814)
s = ds[0]
print("signature", inspect.signature(CanvasDataset))
print("getitem_source")
print(inspect.getsource(CanvasDataset.__getitem__))
print("sample_type", type(s).__name__)
if isinstance(s, dict):
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape), v.dtype, "minmax", float(v.min()), float(v.max()))
        else:
            print(k, repr(v))
else:
    print(repr(s))
