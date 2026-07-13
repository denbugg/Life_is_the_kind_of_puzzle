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
test_dirs = sorted(
    path
    for path in input_root.rglob("test")
    if path.is_dir() and len(list(path.glob("*.png"))) == 700
)
if not all(len(values) == 1 for values in (packages, denoisers, test_dirs)):
    raise SystemExit(
        "input discovery failed: "
        f"packages={packages}, denoisers={denoisers}, test_dirs={test_dirs}"
    )
code_root = packages[0].parents[2]
environment = dict(os.environ)
environment["PYTHONPATH"] = str(code_root / "src")
output = "/kaggle/working/submission.zip"
report = "/kaggle/working/submission.json"
command = [
    sys.executable,
    str(code_root / "scripts/build_assembly_submission.py"),
    "--input-dir",
    str(test_dirs[0]),
    "--denoiser",
    str(denoisers[0]),
    "--output",
    output,
    "--report",
    report,
    "--expected-count",
    "700",
    "--device",
    "cuda",
]
print(json.dumps({"event": "submission_start", "command": command}, sort_keys=True), flush=True)
subprocess.run(command, check=True, env=environment)
print(
    json.dumps(
        {
            "event": "submission_wrapper_complete",
            "output": output,
            "report": report,
            "probe": probe,
        },
        sort_keys=True,
    ),
    flush=True,
)
