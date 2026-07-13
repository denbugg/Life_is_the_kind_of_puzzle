from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


started = time.time()
install = [
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
print(json.dumps({"event": "torch_install_start", "command": install}), flush=True)
subprocess.run(install, check=True)

import torch

probe = {
    "torch": torch.__version__,
    "compiled_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
if not probe["cuda_available"]:
    raise SystemExit("GPU kernel has no CUDA device")
probe["matmul_mean"] = float(
    (torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda"))
    .mean()
    .cpu()
)
print(json.dumps({"event": "hardware_probe", **probe}, sort_keys=True), flush=True)

input_root = Path("/kaggle/input")
packages = sorted(input_root.rglob("src/puzzle_assembly/__init__.py"))
checkpoints = sorted(input_root.rglob("selected_tilenaf_synth_50k.pt"))
targets = sorted(input_root.rglob("train/targets"))
if len(packages) != 1 or len(checkpoints) != 1 or len(targets) != 1:
    raise SystemExit(
        f"input discovery failed: packages={packages}, checkpoints={checkpoints}, targets={targets}"
    )
code_root = packages[0].parents[2]
data_root = targets[0].parent.parent
command = [
    sys.executable,
    str(code_root / "scripts/train_pair_reranker.py"),
    "--data-root",
    str(data_root),
    "--denoiser",
    str(checkpoints[0]),
    "--manifest",
    str(code_root / "configs/denoise_splits_seed20260710.json"),
    "--quarantine",
    str(code_root / "configs/denoise_validation_quarantine_v1.json"),
    "--train-sources",
    "64",
    "--val-sources",
    "4",
    "--epochs",
    "1",
    "--queries-per-source",
    "128",
    "--device",
    "cuda",
    "--output",
    "/kaggle/working/l0_gpu_smoke.pt",
]
environment = dict(os.environ)
environment["PYTHONPATH"] = str(code_root / "src")
print(json.dumps({"event": "training_start", "command": command}), flush=True)
subprocess.run(command, check=True, env=environment)
result = {
    "schema_version": 1,
    "kind": "assembly_v1_l0_gpu_smoke_wrapper",
    "probe": probe,
    "seconds": time.time() - started,
    "training_report": "/kaggle/working/l0_gpu_smoke.json",
}
Path("/kaggle/working/gpu_smoke_wrapper.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({"event": "wrapper_complete", **result}, sort_keys=True), flush=True)
