from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return (completed.stdout + completed.stderr).strip()


print(json.dumps({"event": "python", "version": sys.version}), flush=True)
print(run(["nvidia-smi"]), flush=True)

import torch

payload = {
    "event": "torch_probe",
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "compiled_cuda": torch.version.cuda,
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
}
if torch.cuda.is_available():
    payload.update(
        {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )
print(json.dumps(payload, sort_keys=True), flush=True)

success = False
error = None
try:
    left = torch.randn(1024, 1024, device="cuda")
    right = torch.randn(1024, 1024, device="cuda")
    value = float((left @ right).mean().cpu())
    torch.cuda.synchronize()
    success = True
except Exception as exc:
    value = None
    error = repr(exc)
print(json.dumps({"event": "cuda_matmul", "success": success, "value": value, "error": error}), flush=True)

input_roots = [str(path) for path in sorted(Path("/kaggle/input").glob("*/*/*"))]
print(json.dumps({"event": "input_roots", "paths": input_roots[:20]}), flush=True)
Path("/kaggle/working/probe_result.json").write_text(
    json.dumps({"torch_probe": payload, "matmul_success": success, "matmul_error": error}, indent=2),
    encoding="utf-8",
)
if not success:
    raise SystemExit(2)
