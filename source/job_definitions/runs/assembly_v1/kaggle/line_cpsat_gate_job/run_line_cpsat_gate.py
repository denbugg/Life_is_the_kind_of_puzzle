#!/usr/bin/env python3
"""Kaggle gate for line-continuation scores and optional CP-SAT search."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
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


def roots() -> tuple[Path, Path, Path]:
    data = single(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and (path.parent / "targets").is_dir()
        ],
        "data root",
    )
    runtime = single(
        [
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "runtime root",
    )
    preferred = INPUT / "vsos-solver-rework-night-code"
    if (preferred / "src" / "puzzle_assembly" / "line_seam.py").is_file():
        code = preferred
    else:
        code = single(
            [
                path.parent.parent.parent
                for path in INPUT.glob("**/src/puzzle_assembly/line_seam.py")
                if (path.parent / "cpsat.py").is_file()
            ],
            "code root",
        )
    return data, runtime, code


def install_ortools() -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "ortools==9.14.6206",
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"OR-Tools installation failed with {completed.returncode}")
    return importlib.metadata.version("ortools")


def hardware_probe() -> dict:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    result = {
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
        values = torch.randn(128, 128, device=device)
        result["matmul_means"].append(float((values @ values).mean().item()))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(name: str, command: list[str], gpu: int, code: Path) -> dict:
    output = WORKING / f"{name}.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(code / "src")
    started = time.perf_counter()
    print(json.dumps({"event": "start", "name": name, "gpu": gpu}), flush=True)
    completed = subprocess.run(command, cwd=code, env=environment, check=False)
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
        record["macro"] = report.get("macro")
        record["source_names"] = report.get("source_names")
    print(json.dumps({"event": "complete", **record}, default=str), flush=True)
    return record


def main() -> None:
    started = time.perf_counter()
    data, runtime, code = roots()
    ortools_version = install_ortools()
    hardware = hardware_probe()
    gpu_count = max(1, int(hardware["device_count"]))
    denoiser = runtime / "selected_tilenaf_synth_50k.pt"
    hbt = runtime / "hbt_d320_denoised_rgb_sobel.pt"
    manifest = code / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code / "configs" / "denoise_validation_quarantine_v1.json"
    common = [
        "--data-root", str(data),
        "--manifest", str(manifest),
        "--quarantine", str(quarantine),
        "--device", "cuda",
        "--batch-size", "512",
    ]

    def exact(panel: str, name: str) -> list[str]:
        return [
            sys.executable, str(code / "scripts" / "evaluate_assembly_baselines.py"),
            *common,
            "--checkpoint", str(denoiser),
            "--embedding-checkpoint", str(hbt),
            "--panel", panel,
            "--view", "denoised",
            "--split", "edge_development",
            "--offset", "0",
            "--limit", "2",
            "--line-seam",
            "--global-score", "l1w4line",
            "--component-scores", "line,l1w4line",
            "--soft-cycle-topk", "8",
            "--soft-cycle-keep-per-tile", "1",
            "--soft-cycle-keep-fraction", "0.5",
            "--qap-iterations", "15",
            "--qap-restarts", "2",
            "--qap-seeds", "component_l1w4line_softcycle_k8_p1",
            "--cpsat-time-seconds", "60",
            "--cpsat-topk", "8",
            "--cpsat-workers", "2",
            "--cpsat-square-terms", "1024",
            "--cpsat-seeds", "qap_component_l1w4line_softcycle_k8_p1",
            "--skip-component-refine",
            "--beam-width", "1",
            "--anneal-evaluations", "0",
            "--output", str(WORKING / f"{name}.json"),
            "--overwrite",
        ]

    exact_specs = [
        ("line_cpsat_exact_primary2", "primary_kornia", 0),
        ("line_cpsat_exact_independent2", "independent_libjpeg", 1 % gpu_count),
    ]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(gpu_count, 2)) as executor:
        futures = [
            executor.submit(run, name, exact(panel, name), gpu, code)
            for name, panel, gpu in exact_specs
        ]
        records.extend(future.result() for future in futures)

    real_script = str(code / "scripts" / "evaluate_real_assembly.py")
    real_common = [
        sys.executable, real_script,
        *common,
        "--denoiser", str(denoiser),
        "--embedding-checkpoint", str(hbt),
        "--embedding-view", "denoised",
        "--split", "assembly_cal",
        "--offset", "0",
        "--limit", "4",
        "--soft-cycle-topk", "8",
        "--soft-cycle-keep-per-tile", "1",
        "--soft-cycle-keep-fraction", "0.5",
        "--qap-iterations", "15",
        "--qap-restarts", "2",
        "--cpsat-time-seconds", "60",
        "--cpsat-topk", "8",
        "--cpsat-workers", "2",
        "--cpsat-square-terms", "1024",
    ]
    real_specs = [
        (
            "line_cpsat_real4",
            [
                *real_common,
                "--line-seam",
                "--soft-cycle-scores", "l1w4line",
                "--qap-score", "l1w4line",
                "--qap-seeds", "softcycle_l1w4line_k8",
                "--cpsat-score", "l1w4line",
                "--cpsat-seeds", "qap_softcycle_l1w4line_k8",
                "--preview-dir", str(WORKING / "line_real4_previews"),
                "--output", str(WORKING / "line_cpsat_real4.json"),
                "--overwrite",
            ],
            0,
        ),
        (
            "base_cpsat_real4",
            [
                *real_common,
                "--soft-cycle-scores", "l1",
                "--qap-score", "l1w4",
                "--qap-seeds", "softcycle_l1_k8",
                "--cpsat-score", "l1w4",
                "--cpsat-seeds", "qap_softcycle_l1_k8",
                "--preview-dir", str(WORKING / "base_real4_previews"),
                "--output", str(WORKING / "base_cpsat_real4.json"),
                "--overwrite",
            ],
            1 % gpu_count,
        ),
    ]
    with ThreadPoolExecutor(max_workers=min(gpu_count, 2)) as executor:
        futures = [
            executor.submit(run, name, command, gpu, code)
            for name, command, gpu in real_specs
        ]
        records.extend(future.result() for future in futures)

    wrapper = {
        "schema_version": 1,
        "kind": "line_continuation_cpsat_gate",
        "ortools": ortools_version,
        "probe": hardware,
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    (WORKING / "line_cpsat_gate_wrapper.json").write_text(
        json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"failed line/CP-SAT runs: {[item['name'] for item in failures]}")


if __name__ == "__main__":
    main()
