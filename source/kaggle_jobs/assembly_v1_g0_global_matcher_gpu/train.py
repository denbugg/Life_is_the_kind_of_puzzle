from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


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
    "devices": [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ],
    "capabilities": [
        list(torch.cuda.get_device_capability(index))
        for index in range(torch.cuda.device_count())
    ],
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
denoisers = sorted(input_root.rglob("selected_tilenaf_synth_50k.pt"))
embeddings = sorted(input_root.rglob("hbt_d320_denoised_rgb_sobel.pt"))
targets = sorted(input_root.rglob("train/targets"))
if not all(len(values) == 1 for values in (packages, denoisers, embeddings, targets)):
    raise SystemExit(
        "input discovery failed: "
        f"packages={packages}, denoisers={denoisers}, "
        f"embeddings={embeddings}, targets={targets}"
    )
code_root = packages[0].parents[2]
data_root = targets[0].parent.parent
environment = dict(os.environ)
environment["PYTHONPATH"] = str(code_root / "src")
output = "/kaggle/working/g0_global_matcher_512x2.pt"
command = [
    sys.executable,
    str(code_root / "scripts/train_global_successor_matcher.py"),
    "--data-root",
    str(data_root),
    "--denoiser",
    str(denoisers[0]),
    "--embedding-checkpoint",
    str(embeddings[0]),
    "--manifest",
    str(code_root / "configs/denoise_splits_seed20260710.json"),
    "--quarantine",
    str(code_root / "configs/denoise_validation_quarantine_v1.json"),
    "--panel",
    "primary_kornia",
    "--view",
    "denoised",
    "--train-sources",
    "512",
    "--val-sources",
    "32",
    "--epochs",
    "2",
    "--model-dim",
    "128",
    "--layers",
    "3",
    "--heads",
    "4",
    "--feedforward-dim",
    "256",
    "--sinkhorn-iterations",
    "20",
    "--learning-rate",
    "0.0001",
    "--weight-decay",
    "0.0001",
    "--grad-clip",
    "1.0",
    "--device",
    "cuda",
    "--output",
    output,
]
print(json.dumps({"event": "g0_start", "command": command}, sort_keys=True), flush=True)
subprocess.run(command, check=True, env=environment)
print(
    json.dumps(
        {
            "event": "g0_wrapper_complete",
            "checkpoint": output,
            "report": str(Path(output).with_suffix(".json")),
            "probe": probe,
        },
        sort_keys=True,
    ),
    flush=True,
)
