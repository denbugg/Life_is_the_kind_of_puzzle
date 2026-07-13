from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


started = time.time()
install_command = [
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
print(json.dumps({"event": "torch_install_start", "command": install_command}), flush=True)
subprocess.run(install_command, check=True)

import torch


probe = {
    "torch": torch.__version__,
    "compiled_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    "capabilities": [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())],
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
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
initializers = sorted(input_root.rglob("selected_tilenaf_synth_50k.pt"))
targets = sorted(input_root.rglob("train/targets"))
if not all(len(values) == 1 for values in (packages, initializers, targets)):
    raise SystemExit(
        f"input discovery failed: packages={packages}, initializers={initializers}, targets={targets}"
    )
code_root = packages[0].parents[2]
data_root = targets[0].parent.parent
command = [
    sys.executable,
    str(code_root / "scripts/train_denoise_v2.py"),
    "--data-root", str(data_root),
    "--manifest", str(code_root / "configs/denoise_splits_seed20260710.json"),
    "--output", "/kaggle/working/seam_denoiser_gpu.pt",
    "--model", "tile-naf",
    "--init-weights", str(initializers[0]),
    "--train-images", "1024",
    "--val-images", "16",
    "--val-tiles-per-image", "576",
    "--steps", "3000",
    "--batch-size", "256",
    "--eval-batch-size", "512",
    "--learning-rate", "2e-5",
    "--eval-interval", "500",
    "--log-interval", "50",
    "--ssim-start-fraction", "0.0",
    "--loss-ssim", "0.10",
    "--loss-gradient", "0.15",
    "--loss-boundary-extra", "2.0",
    "--device", "cuda",
]
environment = dict(os.environ)
environment["PYTHONPATH"] = str(code_root / "src")
print(json.dumps({"event": "training_start", "command": command}), flush=True)
subprocess.run(command, check=True, env=environment)
result = {
    "schema_version": 1,
    "kind": "assembly_v1_seam_denoiser_gpu_wrapper",
    "probe": probe,
    "seconds": time.time() - started,
    "checkpoint": "/kaggle/working/seam_denoiser_gpu.pt",
}
Path("/kaggle/working/gpu_full_wrapper.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({"event": "wrapper_complete", **result}, sort_keys=True), flush=True)
