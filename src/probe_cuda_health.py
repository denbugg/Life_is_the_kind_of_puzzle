from __future__ import annotations

import torch

print({"cuda_available": torch.cuda.is_available(), "cuda_count": torch.cuda.device_count()})
if torch.cuda.is_available():
    x = torch.randn((128, 128), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print({"device": torch.cuda.get_device_name(0), "finite": bool(torch.isfinite(y).all().item())})
