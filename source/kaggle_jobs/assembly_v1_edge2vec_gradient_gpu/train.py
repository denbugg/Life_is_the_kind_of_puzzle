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


device_count = torch.cuda.device_count()
probe = {
    "torch": torch.__version__,
    "compiled_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": device_count,
    "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
    "capabilities": [list(torch.cuda.get_device_capability(index)) for index in range(device_count)],
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
warm_starts = sorted(input_root.rglob("l1_gpu_full.pt"))
targets = sorted(input_root.rglob("train/targets"))
if not all(len(values) == 1 for values in (packages, denoisers, warm_starts, targets)):
    raise SystemExit(
        "input discovery failed: "
        f"packages={packages}, denoisers={denoisers}, "
        f"warm_starts={warm_starts}, targets={targets}"
    )
code_root = packages[0].parents[2]
data_root = targets[0].parent.parent
environment = dict(os.environ)
environment["PYTHONPATH"] = str(code_root / "src")

experiments = [
    {
        "name": "hbt_d320_denoised_rgb_norm",
        "view": "denoised",
        "input_mode": "rgb_norm",
        "train_sources": 1024,
        "epochs": 2,
    },
    {
        "name": "hbt_d320_denoised_rgb_sobel",
        "view": "denoised",
        "input_mode": "rgb_sobel",
        "train_sources": 2048,
        "epochs": 2,
    },
    {
        "name": "hbt_d320_raw_rgb_sobel",
        "view": "raw",
        "input_mode": "rgb_sobel",
        "train_sources": 2048,
        "epochs": 2,
    },
    {
        "name": "hbt_d320_denoised_sobel_only",
        "view": "denoised",
        "input_mode": "sobel_only",
        "train_sources": 512,
        "epochs": 1,
    },
    {
        "name": "hbt_d320_raw_sobel_only",
        "view": "raw",
        "input_mode": "sobel_only",
        "train_sources": 512,
        "epochs": 1,
    },
    {
        "name": "hbt_d320_denoised_binary_edges",
        "view": "denoised",
        "input_mode": "binary_edges",
        "train_sources": 512,
        "epochs": 1,
    },
    {
        "name": "hbt_d320_raw_binary_edges",
        "view": "raw",
        "input_mode": "binary_edges",
        "train_sources": 512,
        "epochs": 1,
    },
]

reports = []
for experiment in experiments:
    output = f"/kaggle/working/{experiment['name']}.pt"
    command = [
        sys.executable,
        str(code_root / "scripts/train_side_embeddings.py"),
        "--data-root", str(data_root),
        "--denoiser", str(denoisers[0]),
        "--warm-start-stem", str(warm_starts[0]),
        "--manifest", str(code_root / "configs/denoise_splits_seed20260710.json"),
        "--quarantine", str(code_root / "configs/denoise_validation_quarantine_v1.json"),
        "--panel", "primary_kornia",
        "--view", experiment["view"],
        "--train-sources", str(experiment["train_sources"]),
        "--val-sources", "32",
        "--epochs", str(experiment["epochs"]),
        "--channels", "64",
        "--embedding-dim", "320",
        "--side-band", "4",
        "--tangent-bins", "10",
        "--input-mode", experiment["input_mode"],
        "--edge-threshold", "0.12",
        "--loss", "hard_triplet",
        "--triplet-margin", "0.2",
        "--cross-entropy-weight", "0.25",
        "--embedding-l2-weight", "0.0001",
        "--outside-weight", "0.2",
        "--learning-rate", "0.0003",
        "--weight-decay", "0.0001",
        "--device", "cuda",
        "--output", output,
    ]
    print(
        json.dumps(
            {"event": "experiment_start", "experiment": experiment, "command": command},
            sort_keys=True,
        ),
        flush=True,
    )
    experiment_started = time.time()
    subprocess.run(command, check=True, env=environment)
    report = {
        "name": experiment["name"],
        "checkpoint": output,
        "report": str(Path(output).with_suffix(".json")),
        "seconds": time.time() - experiment_started,
    }
    reports.append(report)
    print(json.dumps({"event": "experiment_complete", **report}, sort_keys=True), flush=True)

result = {
    "schema_version": 1,
    "kind": "assembly_v1_edge2vec_gradient_factorial_gpu",
    "probe": probe,
    "experiments": experiments,
    "reports": reports,
    "seconds": time.time() - started,
}
Path("/kaggle/working/gpu_full_wrapper.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({"event": "wrapper_complete", **result}, sort_keys=True), flush=True)
