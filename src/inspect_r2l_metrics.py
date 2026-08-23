import json
from pathlib import Path
import torch

checkpoint = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
result = {
    "keys": sorted(payload.keys()),
    "args": payload.get("args"),
    "metrics": payload.get("metrics"),
    "step": payload.get("step"),
    "selection": payload.get("selection"),
}
print(json.dumps(result, indent=2, default=str))
