from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


started = time.time()
command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--upgrade",
    "torch==2.6.0",
    "--index-url",
    "https://download.pytorch.org/whl/cu124",
]
print(json.dumps({"event": "install_start", "command": command}), flush=True)
completed = subprocess.run(command, check=False, text=True)
print(json.dumps({"event": "install_end", "returncode": completed.returncode, "seconds": time.time() - started}), flush=True)
if completed.returncode:
    raise SystemExit(completed.returncode)

import torch

probe = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "compiled_cuda": torch.version.cuda,
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
print(json.dumps({"event": "torch_probe", **probe}, sort_keys=True), flush=True)

success = False
error = None
throughput = None
try:
    left = torch.randn(2048, 2048, device="cuda")
    right = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    bench_started = time.time()
    for _ in range(5):
        output = left @ right
    torch.cuda.synchronize()
    throughput = 5 * 2 * 2048**3 / (time.time() - bench_started) / 1e12
    value = float(output.mean().cpu())
    success = True
except Exception as exc:
    value = None
    error = repr(exc)

result = {
    "torch_probe": probe,
    "matmul_success": success,
    "matmul_mean": value,
    "matmul_tflops": throughput,
    "matmul_error": error,
    "total_seconds": time.time() - started,
}
print(json.dumps({"event": "result", **result}, sort_keys=True), flush=True)
Path("/kaggle/working/torch26_probe_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
if not success:
    raise SystemExit(2)
