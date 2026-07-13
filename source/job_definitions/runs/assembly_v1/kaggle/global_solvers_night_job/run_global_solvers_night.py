#!/usr/bin/env python3
"""Two-GPU leakage-safe gate for global tile-layout solvers."""

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


def _single(candidates: list[Path], kind: str) -> Path:
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"expected one {kind}, found {unique}")
    return unique[0]


def find_data_root() -> Path:
    return _single(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and (path.parent / "targets").is_dir()
        ],
        "puzzle data root",
    )


def find_runtime_root() -> Path:
    return _single(
        [
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "checkpoint runtime root",
    )


def find_code_root() -> Path:
    preferred = INPUT / "vsos-solver-rework-night-code"
    required = preferred / "src" / "puzzle_assembly" / "qap.py"
    if required.is_file():
        return preferred
    return _single(
        [
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_assembly/qap.py")
            if (path.parent / "particle.py").is_file()
            and (path.parent.parent.parent / "scripts" / "evaluate_real_assembly.py").is_file()
        ],
        "global solver code root",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_probe() -> dict:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "capabilities": [
            list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
        ],
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"expected T4x2, got {result}")
    means = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        left = torch.randn(256, 256, device=device)
        right = torch.randn(256, 256, device=device)
        means.append(float((left @ right).mean().item()))
    result["matmul_means"] = means
    return result


def run_experiment(
    name: str,
    command: list[str],
    *,
    gpu: int,
    code_root: Path,
) -> dict:
    output = WORKING / f"{name}.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(code_root / "src")
    started = time.perf_counter()
    print(
        json.dumps({"event": "start", "name": name, "gpu": gpu, "command": command}),
        flush=True,
    )
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
        payload = json.loads(output.read_text(encoding="utf-8"))
        record["sha256"] = sha256(output)
        record["source_names"] = payload.get("source_names")
        record["macro"] = payload.get("macro")
    print(json.dumps({"event": "complete", **record}, default=str), flush=True)
    return record


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    code_root = find_code_root()
    probe = hardware_probe()
    print(json.dumps({"event": "hardware", **probe}, sort_keys=True), flush=True)

    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    hbt = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    exact_script = str(code_root / "scripts" / "evaluate_assembly_baselines.py")
    real_script = str(code_root / "scripts" / "evaluate_real_assembly.py")
    common = [
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--quarantine", str(quarantine),
        "--device", "cuda",
        "--batch-size", "512",
    ]

    def exact_command(panel: str, name: str) -> list[str]:
        return [
            sys.executable, exact_script,
            *common,
            "--checkpoint", str(denoiser),
            "--panel", panel,
            "--view", "denoised",
            "--split", "edge_development",
            "--offset", "0",
            "--limit", "2",
            "--embedding-checkpoint", str(hbt),
            "--global-score", "l1w4",
            "--component-scores", "l1",
            "--soft-cycle-topk", "8",
            "--soft-cycle-keep-per-tile", "1",
            "--soft-cycle-keep-fraction", "0.5",
            "--faithful-rl-phases", "24",
            "--faithful-rl-topk", "17",
            "--faithful-rl-max-iterations", "12",
            "--particle-beam-particles", "16",
            "--particle-beam-topk", "4",
            "--particle-beam-seeds", "component_l1_loops,component_l1_softcycle_k8_p1",
            "--qap-iterations", "15",
            "--qap-restarts", "2",
            "--qap-seeds", "component_l1_softcycle_k8_p1",
            "--qap-refine-swaps", "8",
            "--skip-component-refine",
            "--beam-width", "1",
            "--anneal-evaluations", "0",
            "--output", str(WORKING / f"{name}.json"),
            "--overwrite",
        ]

    exact_specs = [
        ("global_exact_primary2", "primary_kornia", 0),
        ("global_exact_independent2", "independent_libjpeg", 1),
    ]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_experiment,
                name,
                exact_command(panel, name),
                gpu=gpu,
                code_root=code_root,
            )
            for name, panel, gpu in exact_specs
        ]
        records.extend(future.result() for future in futures)

    real_name = "global_real4"
    real_command = [
        sys.executable, real_script,
        *common,
        "--denoiser", str(denoiser),
        "--split", "assembly_cal",
        "--offset", "0",
        "--limit", "4",
        "--embedding-checkpoint", str(hbt),
        "--embedding-view", "denoised",
        "--soft-cycle-topk", "8",
        "--soft-cycle-scores", "l1",
        "--soft-cycle-keep-per-tile", "1",
        "--soft-cycle-keep-fraction", "0.5",
        "--faithful-rl-phases", "24",
        "--faithful-rl-topk", "17",
        "--faithful-rl-max-iterations", "12",
        "--faithful-rl-score", "l1w4",
        "--particle-beam-particles", "16",
        "--particle-beam-topk", "4",
        "--particle-beam-seeds", "component_l1,softcycle_l1_k8",
        "--particle-beam-score", "l1w4",
        "--qap-iterations", "15",
        "--qap-restarts", "2",
        "--qap-seeds", "softcycle_l1_k8",
        "--qap-score", "l1w4",
        "--qap-refine-swaps", "8",
        "--output", str(WORKING / f"{real_name}.json"),
        "--overwrite",
    ]
    records.append(
        run_experiment(real_name, real_command, gpu=0, code_root=code_root)
    )

    wrapper = {
        "schema_version": 1,
        "kind": "global_solver_small_gate",
        "probe": probe,
        "data_root": str(data_root),
        "runtime_root": str(runtime_root),
        "code_root": str(code_root),
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    output = WORKING / "global_solvers_night_wrapper.json"
    output.write_text(json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8")
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"failed experiments: {[item['name'] for item in failures]}")


if __name__ == "__main__":
    main()
