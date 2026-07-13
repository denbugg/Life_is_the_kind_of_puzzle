#!/usr/bin/env python3
"""Two-device real16 tuning grid for the promoted directional QAP solver."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


def single(paths: list[Path], label: str) -> Path:
    candidates = sorted(set(paths))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {label}, found {candidates}")
    return candidates[0]


def find_data_root() -> Path:
    return single(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and (path.parent / "targets").is_dir()
        ],
        "data root",
    )


def find_runtime_root() -> Path:
    return single(
        [
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "runtime root",
    )


def find_code_root() -> Path:
    preferred = INPUT / "vsos-solver-rework-night-code"
    if (
        (preferred / "src" / "puzzle_assembly" / "qap.py").is_file()
        and (preferred / "scripts" / "evaluate_real_assembly.py").is_file()
    ):
        return preferred
    return single(
        [
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_assembly/qap.py")
            if (path.parent.parent.parent / "scripts" / "evaluate_real_assembly.py").is_file()
        ],
        "code root",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe() -> dict:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "device_count": count,
        "devices": [torch.cuda.get_device_name(index) for index in range(count)],
        "capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(count)
        ],
        "arch_list": torch.cuda.get_arch_list(),
        "matmul_means": [],
    }
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        left = torch.randn(128, 128, device=device)
        right = torch.randn(128, 128, device=device)
        result["matmul_means"].append(float((left @ right).mean().item()))
    return result


def run_one(name: str, command: list[str], gpu: int, code_root: Path) -> dict:
    output = WORKING / f"{name}.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(code_root / "src")
    started = time.perf_counter()
    print(json.dumps({"event": "start", "name": name, "gpu": gpu}), flush=True)
    completed = subprocess.run(
        command,
        cwd=code_root,
        env=environment,
        check=False,
    )
    record = {
        "name": name,
        "gpu": gpu,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "output": str(output),
    }
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        record["sha256"] = sha256(output)
        record["source_names"] = report.get("source_names")
        record["macro"] = report.get("macro")
    print(json.dumps({"event": "complete", **record}, default=str), flush=True)
    return record


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    code_root = find_code_root()
    hardware = probe()
    print(json.dumps({"event": "hardware", **hardware}, sort_keys=True), flush=True)
    gpu_count = max(1, int(hardware["device_count"]))

    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    hbt = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    script = str(code_root / "scripts" / "evaluate_real_assembly.py")
    common = [
        sys.executable, script,
        "--data-root", str(data_root),
        "--manifest", str(code_root / "configs" / "denoise_splits_seed20260710.json"),
        "--quarantine", str(code_root / "configs" / "denoise_validation_quarantine_v1.json"),
        "--device", "cuda",
        "--batch-size", "512",
        "--denoiser", str(denoiser),
        "--split", "assembly_cal",
        "--offset", "0",
        "--limit", "16",
        "--embedding-checkpoint", str(hbt),
        "--embedding-view", "denoised",
        "--soft-cycle-topk", "8",
        "--soft-cycle-scores", "l1",
        "--soft-cycle-keep-per-tile", "1",
        "--soft-cycle-keep-fraction", "0.5",
    ]
    settings = [
        {
            "name": "qap_l1w4_multiseed_real16",
            "score": "l1w4",
            "seeds": "softcycle_l1_k8,component_l1fusion_q50,denoised_component_fusion",
            "iterations": 25,
            "restarts": 2,
            "boundary": 0.0,
            "initial_weight": 0.75,
            "noisy_components": 3,
            "noise_scale": 1.0,
        },
        {
            "name": "qap_cross_multiseed_real16",
            "score": "cross_l1w4",
            "seeds": "softcycle_l1_k8,component_cross_l1w4_q50",
            "iterations": 25,
            "restarts": 2,
            "boundary": 0.0,
            "initial_weight": 0.75,
            "noisy_components": 3,
            "noise_scale": 1.0,
        },
        {
            "name": "qap_l1w4_heavy_real16",
            "score": "l1w4",
            "seeds": "softcycle_l1_k8",
            "iterations": 40,
            "restarts": 4,
            "boundary": 0.0,
            "initial_weight": 0.50,
            "noisy_components": 5,
            "noise_scale": 0.50,
        },
        {
            "name": "qap_l1w4_boundary_real16",
            "score": "l1w4",
            "seeds": "softcycle_l1_k8",
            "iterations": 25,
            "restarts": 2,
            "boundary": 0.05,
            "initial_weight": 0.75,
            "noisy_components": 3,
            "noise_scale": 1.0,
        },
    ]

    queues: list[list[tuple[str, list[str]]]] = [[] for _ in range(gpu_count)]
    for index, setting in enumerate(settings):
        name = str(setting["name"])
        command = [
            *common,
            "--qap-iterations", str(setting["iterations"]),
            "--qap-restarts", str(setting["restarts"]),
            "--qap-boundary-weight", str(setting["boundary"]),
            "--qap-initial-weight", str(setting["initial_weight"]),
            "--qap-noisy-components", str(setting["noisy_components"]),
            "--qap-noise-scale", str(setting["noise_scale"]),
            "--qap-refine-swaps", "8",
            "--qap-seeds", str(setting["seeds"]),
            "--qap-score", str(setting["score"]),
            "--output", str(WORKING / f"{name}.json"),
            "--overwrite",
        ]
        queues[index % gpu_count].append((name, command))

    def run_queue(gpu: int, queue: list[tuple[str, list[str]]]) -> list[dict]:
        return [run_one(name, command, gpu, code_root) for name, command in queue]

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        futures = [
            executor.submit(run_queue, gpu, queue)
            for gpu, queue in enumerate(queues)
            if queue
        ]
        for future in futures:
            records.extend(future.result())

    wrapper = {
        "schema_version": 1,
        "kind": "qap_real16_tuning_grid",
        "probe": hardware,
        "settings": settings,
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    (WORKING / "qap_tuning_night_wrapper.json").write_text(
        json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"failed QAP runs: {[record['name'] for record in failures]}")


if __name__ == "__main__":
    main()
