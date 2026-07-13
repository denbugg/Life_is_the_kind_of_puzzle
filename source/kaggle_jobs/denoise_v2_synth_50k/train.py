from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def install_runtime() -> None:
    started = time.time()
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--upgrade",
            "torch==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "kornia==0.8.3"],
    ]
    for command in commands:
        print(json.dumps({"event": "install", "command": command}), flush=True)
        subprocess.run(command, check=True)
    print(json.dumps({"event": "runtime_ready", "seconds": time.time() - started}), flush=True)


install_runtime()

package_candidates = sorted(Path("/kaggle/input").rglob("src/puzzle_denoise_v2/__init__.py"))
if len(package_candidates) != 1:
    raise SystemExit(f"expected one puzzle_denoise_v2 package, found {package_candidates}")
package_init = package_candidates[0]
package_root = package_init.parents[1]
sys.path.insert(0, str(package_root))

source_digest = hashlib.sha256()
for source_file in sorted(package_init.parent.glob("*.py")):
    source_digest.update(source_file.name.encode("utf-8"))
    source_digest.update(source_file.read_bytes())

manifest_candidates = sorted(Path("/kaggle/input").rglob("denoise_splits_seed20260710.json"))
if len(manifest_candidates) != 1:
    raise SystemExit(f"expected one split manifest, found {manifest_candidates}")
manifest_path = manifest_candidates[0]

import torch

from puzzle_denoise_v2.training import TrainConfig, train


probe = {
    "event": "gpu_probe",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    "source_sha256": source_digest.hexdigest(),
    "manifest": str(manifest_path),
}
print(json.dumps(probe, sort_keys=True), flush=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (6, 0):
    raise SystemExit("compatible CUDA GPU unavailable")

data_root = Path("/kaggle/input/datasets/pasha883/vsos-ai-initiative-pazzle")
if not (data_root / "train" / "targets").is_dir():
    candidates = list(Path("/kaggle/input").rglob("train/targets"))
    if not candidates:
        raise SystemExit("could not locate train/targets")
    data_root = candidates[0].parent.parent

config = TrainConfig(
    data_root=str(data_root),
    manifest=str(manifest_path),
    output="/kaggle/working/tilenaf_synth_50k.pt",
    model="tile-naf",
    train_images=4900,
    val_images=24,
    val_tiles_per_image=576,
    steps=50_000,
    batch_size=256,
    eval_batch_size=512,
    learning_rate=3e-4,
    weight_decay=1e-4,
    ema_decay=0.999,
    seed=20260710,
    device="cuda",
    log_interval=500,
    eval_interval=5_000,
    ssim_start_fraction=0.50,
    variant_weights=(1.0, 0.0, 0.0),
    libjpeg_val_images=4,
)
result = train(config)
Path("/kaggle/working/synth_50k_result.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps({"event": "job_complete", **result}, sort_keys=True), flush=True)
