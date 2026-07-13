#!/usr/bin/env python3
"""Nightly Kaggle matrix for solver-only reordering experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


JOB_ROOT = Path(__file__).resolve().parent
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
sys.path.insert(0, str(JOB_ROOT / "src"))


def mount_snapshot() -> dict[str, list[str]]:
    snapshot: dict[str, list[str]] = {}
    for mount in sorted(INPUT.iterdir()):
        if mount.is_dir():
            snapshot[mount.name] = [path.name for path in sorted(mount.iterdir())[:32]]
    return snapshot


def find_data_root() -> Path:
    candidates = sorted(
        {
            inputs.parent.parent
            for inputs in INPUT.glob("**/train/inputs")
            if inputs.is_dir() and (inputs.parent / "targets").is_dir()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one puzzle data root, found {candidates}; mounts={mount_snapshot()}"
        )
    return candidates[0]


def find_runtime_root() -> Path:
    candidates = sorted(
        {
            checkpoint.parent
            for checkpoint in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (checkpoint.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one runtime root, found {candidates}; mounts={mount_snapshot()}"
        )
    return candidates[0]


def find_code_root() -> Path:
    preferred = INPUT / "vsos-solver-rework-night-code"
    if (
        (preferred / "scripts" / "evaluate_assembly_baselines.py").is_file()
        and (preferred / "src" / "puzzle_assembly" / "solvers.py").is_file()
    ):
        return preferred
    candidates = sorted(
        {
            script.parent.parent
            for script in INPUT.glob("**/scripts/evaluate_assembly_baselines.py")
            if (script.parent / "evaluate_real_assembly.py").is_file()
            and (script.parent.parent / "src" / "puzzle_assembly" / "solvers.py").is_file()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one solver code root, found {candidates}; mounts={mount_snapshot()}"
        )
    return candidates[0]


def probe() -> dict:
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
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available")
    first_capability = torch.cuda.get_device_capability(0)
    if f"sm_{first_capability[0]}{first_capability[1]}" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch build does not support device capability {first_capability}")
    left = torch.randn(256, 256, device="cuda")
    right = torch.randn(256, 256, device="cuda")
    result["matmul_mean"] = float((left @ right).mean().item())
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(name: str, command: list[str], records: list[dict]) -> None:
    output = WORKING / f"{name}.json"
    started = time.perf_counter()
    print(json.dumps({"event": "experiment_start", "name": name, "command": command}), flush=True)
    child_env = os.environ.copy()
    code_root = Path(child_env["SOLVER_CODE_ROOT"])
    packaged_src = str(code_root / "src")
    inherited_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        packaged_src
        if not inherited_pythonpath
        else packaged_src + os.pathsep + inherited_pythonpath
    )
    completed = subprocess.run(
        command,
        cwd=code_root,
        check=False,
        env=child_env,
    )
    record = {
        "name": name,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "output": str(output),
    }
    if output.is_file():
        record["sha256"] = sha256(output)
        report = json.loads(output.read_text(encoding="utf-8"))
        record["macro"] = report.get("macro")
        record["source_names"] = report.get("source_names")
    records.append(record)
    print(json.dumps({"event": "experiment_complete", **record}, default=str), flush=True)


def main() -> None:
    started = time.perf_counter()
    print(json.dumps({"event": "mounts", "snapshot": mount_snapshot()}, sort_keys=True), flush=True)
    code_root = find_code_root()
    os.environ["SOLVER_CODE_ROOT"] = str(code_root)
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    hbt = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    hardware = probe()
    print(json.dumps({"event": "hardware_probe", **hardware}, sort_keys=True), flush=True)
    common = [
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--quarantine", str(quarantine),
        "--device", "cuda",
        "--batch-size", "512",
    ]
    exact_script = str(code_root / "scripts" / "evaluate_assembly_baselines.py")
    real_script = str(code_root / "scripts" / "evaluate_real_assembly.py")
    records: list[dict] = []

    # Exact transfer grid: adjacency, LCC and SSIM on both corruption engines.
    for panel in ("primary_kornia", "independent_libjpeg"):
        for rl_top_k in (4, 16):
            name = f"exact_{panel}_rl_k{rl_top_k}"
            run(
                name,
                [
                    sys.executable, exact_script,
                    *common,
                    "--checkpoint", str(denoiser),
                    "--panel", panel,
                    "--view", "denoised",
                    "--split", "edge_development",
                    "--offset", "0",
                    "--limit", "4",
                    "--embedding-checkpoint", str(hbt),
                    "--global-score", "l1w4",
                    "--component-scores", "l1",
                    "--soft-cycle-topk", "8",
                    "--soft-cycle-keep-per-tile", "1",
                    "--soft-cycle-keep-fraction", "0.5",
                    "--multi-phase-rl-phases", "12",
                    "--multi-phase-rl-topk", str(rl_top_k),
                    "--multi-phase-rl-iterations", "3",
                    "--multi-phase-rl-anchor-batch", "48",
                    "--multi-phase-rl-seeds", "component_l1_loops,component_l1_softcycle_k8_p1",
                    "--skip-component-refine",
                    "--beam-width", "1",
                    "--anneal-evaluations", "0",
                    "--output", str(WORKING / f"{name}.json"),
                    "--overwrite",
                ],
                records,
            )

    # Input-only real16 gates. Targets are opened only after all layouts freeze.
    for rl_top_k in (4, 8, 16):
        name = f"real16_rl_k{rl_top_k}"
        run(
            name,
            [
                sys.executable, real_script,
                *common,
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
                "--multi-phase-rl-phases", "12",
                "--multi-phase-rl-topk", str(rl_top_k),
                "--multi-phase-rl-iterations", "3",
                "--multi-phase-rl-anchor-batch", "48",
                "--multi-phase-rl-seeds", "component_l1,softcycle_l1_k8",
                "--multi-phase-rl-score", "l1w4",
                "--output", str(WORKING / f"{name}.json"),
                "--overwrite",
            ],
            records,
        )

    for subset in (64, 192):
        name = f"real16_lns_s{subset}"
        run(
            name,
            [
                sys.executable, real_script,
                *common,
                "--denoiser", str(denoiser),
                "--split", "assembly_cal",
                "--offset", "0",
                "--limit", "16",
                "--embedding-checkpoint", str(hbt),
                "--embedding-view", "denoised",
                "--lns-iterations", "60",
                "--lns-subset-size", str(subset),
                "--lns-seeds", "component_l1,denoised_component_fusion",
                "--lns-score", "l1w4",
                "--output", str(WORKING / f"{name}.json"),
                "--overwrite",
            ],
            records,
        )

    name = "real16_cross_softcycle"
    run(
        name,
        [
            sys.executable, real_script,
            *common,
            "--denoiser", str(denoiser),
            "--split", "assembly_cal",
            "--offset", "0",
            "--limit", "16",
            "--embedding-checkpoint", str(hbt),
            "--embedding-view", "denoised",
            "--soft-cycle-topk", "4",
            "--soft-cycle-scores", "cross_c1,cross_c1_dn2",
            "--soft-cycle-keep-per-tile", "1",
            "--soft-cycle-keep-fraction", "0.5",
            "--output", str(WORKING / f"{name}.json"),
            "--overwrite",
        ],
        records,
    )

    name = "real16_anneal20k"
    run(
        name,
        [
            sys.executable, real_script,
            *common,
            "--denoiser", str(denoiser),
            "--split", "assembly_cal",
            "--offset", "0",
            "--limit", "16",
            "--embedding-checkpoint", str(hbt),
            "--embedding-view", "denoised",
            "--anneal-refine-seeds", "component_l1,denoised_component_fusion",
            "--anneal-refine-evaluations", "20000",
            "--anneal-score", "l1w4",
            "--output", str(WORKING / f"{name}.json"),
            "--overwrite",
        ],
        records,
    )

    wrapper = {
        "schema_version": 1,
        "kind": "solver_rework_night_matrix",
        "data_root": str(data_root),
        "runtime_root": str(runtime_root),
        "probe": hardware,
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    (WORKING / "solver_rework_night_wrapper.json").write_text(
        json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8"
    )
    failures = [record for record in records if record["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} experiments failed: {[r['name'] for r in failures]}")


if __name__ == "__main__":
    main()
