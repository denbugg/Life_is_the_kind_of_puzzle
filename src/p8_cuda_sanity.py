"""Minimal non-training CUDA sanity probe for P8 recovery diagnosis."""
import json
import torch

out={"torch_cuda_available":torch.cuda.is_available(),"torch_version":torch.__version__}
if out["torch_cuda_available"]:
    out["device_name"]=torch.cuda.get_device_name(0)
    x=torch.randn((128,128),device="cuda",dtype=torch.float32)
    out["finite"]=bool(torch.isfinite((x@x.T).mean()).item())
    torch.cuda.synchronize()
print(json.dumps(out),flush=True)
