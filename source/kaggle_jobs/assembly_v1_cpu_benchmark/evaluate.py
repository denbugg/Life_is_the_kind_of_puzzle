from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
started = time.time()
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
probe = {
    "python": sys.version,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
}
print(json.dumps({"event": "cpu_probe", **probe}, sort_keys=True), flush=True)
command = [
    sys.executable,
    str(code_root / "scripts/evaluate_assembly_baselines.py"),
    "--data-root",
    str(data_root),
    "--checkpoint",
    str(checkpoints[0]),
    "--manifest",
    str(code_root / "configs/denoise_splits_seed20260710.json"),
    "--quarantine",
    str(code_root / "configs/denoise_validation_quarantine_v1.json"),
    "--split",
    "edge_development",
    "--panel",
    "primary_kornia",
    "--view",
    "denoised",
    "--device",
    "cpu",
    "--offset",
    "0",
    "--limit",
    "4",
    "--component-scores",
    "pbc,fusion",
    "--skip-component-refine",
    "--lp-scores",
    "pbc,fusion",
    "--output",
    "/kaggle/working/cpu_primary_4.json",
]
environment = dict(os.environ)
environment["PYTHONPATH"] = str(code_root / "src")
print(json.dumps({"event": "cpu_benchmark_start", "command": command}), flush=True)
subprocess.run(command, check=True, env=environment)
wrapper = {
    "schema_version": 1,
    "kind": "assembly_v1_cpu_benchmark_wrapper",
    "probe": probe,
    "seconds": time.time() - started,
    "report": "/kaggle/working/cpu_primary_4.json",
}
Path("/kaggle/working/cpu_benchmark_wrapper.json").write_text(
    json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({"event": "cpu_wrapper_complete", **wrapper}, sort_keys=True), flush=True)
