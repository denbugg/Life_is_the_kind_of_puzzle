#!/usr/bin/env python3
"""Fail-closed staging, T4x2 preflight, smoke, and layout-energy pilot runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WRAPPER_PATH = WORKING / "layout_energy_pilot_wrapper.json"
EXPECTED_OVERLAY_SHA256 = {
    "src/puzzle_assembly/layout_energy_transformer.py": "ebcaab5ecd77dc54e7a7c1f9bf7c282931b4ccbddda414b28e9f07872aa7e6e1",
    "scripts/train_evaluate_layout_energy.py": "4206f121398015405aed1f6fb7438fd9d1bd3d1a98662ae0664d9028d96331d1",
    "tests/test_layout_energy_transformer.py": "144405eff3238bfb6b46879a467ca9473720adf81d009d385397931c16181c2c",
}


def one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise RuntimeError(f"unsafe archive member in {archive}")
        handle.extractall(destination)
    return destination


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_data_root() -> Path:
    return one(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir()
            and (path.parent / "targets").is_dir()
            and len(list((path.parent / "targets").glob("*.png"))) == 7000
        ],
        "puzzle data root",
    )


def find_base_root() -> Path:
    direct = sorted(
        {
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_assembly/qap.py")
            if (
                path.parent.parent.parent
                / "configs"
                / "denoise_splits_seed20260710.json"
            ).is_file()
            and (
                path.parent.parent.parent
                / "src"
                / "puzzle_denoise_v2"
                / "degradation.py"
            ).is_file()
        }
    )
    if len(direct) == 1:
        return direct[0]
    if direct:
        raise RuntimeError(f"ambiguous direct base code roots: {direct}")
    archive = one(list(INPUT.glob("**/solver_rework_code.zip")), "base code archive")
    extracted = safe_extract(archive, WORKING / "layout_energy_base_extracted")
    if not (extracted / "src" / "puzzle_assembly" / "qap.py").is_file():
        raise RuntimeError("base archive is missing puzzle_assembly/qap.py")
    return extracted


def find_overlay_root() -> tuple[Path, Path | None]:
    direct = sorted(
        {
            path.parent.parent
            for path in INPUT.glob("**/scripts/train_evaluate_layout_energy.py")
            if (
                path.parent.parent
                / "src"
                / "puzzle_assembly"
                / "layout_energy_transformer.py"
            ).is_file()
            and (
                path.parent.parent
                / "tests"
                / "test_layout_energy_transformer.py"
            ).is_file()
        }
    )
    archives = sorted(INPUT.glob("**/layout_energy_code.zip"))
    if len(direct) == 1:
        return direct[0], one(archives, "layout-energy overlay archive") if archives else None
    if direct:
        raise RuntimeError(f"ambiguous direct layout-energy overlays: {direct}")
    archive = one(archives, "layout-energy overlay archive")
    extraction = safe_extract(archive, WORKING / "layout_energy_overlay_extracted")
    if not (
        extraction / "src" / "puzzle_assembly" / "layout_energy_transformer.py"
    ).is_file():
        raise RuntimeError("extracted overlay is missing model source")
    return extraction, archive


def copy_minimal_code(base: Path, overlay: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    required_base = (
        "src/puzzle_assembly/__init__.py",
        "src/puzzle_assembly/geometry.py",
        "src/puzzle_assembly/metrics.py",
        "src/puzzle_assembly/compatibility.py",
        "src/puzzle_assembly/components.py",
        "src/puzzle_assembly/solvers.py",
        "src/puzzle_assembly/panels.py",
        "src/puzzle_assembly/protocol.py",
        "src/puzzle_denoise_v2/__init__.py",
        "src/puzzle_denoise_v2/degradation.py",
        "src/puzzle_denoise_v2/model.py",
        "src/puzzle_denoise_v2/tiles.py",
        "configs/denoise_splits_seed20260710.json",
        "configs/denoise_validation_quarantine_v1.json",
    )
    required_overlay = (
        "src/puzzle_assembly/layout_energy_transformer.py",
        "scripts/train_evaluate_layout_energy.py",
        "tests/test_layout_energy_transformer.py",
    )
    for relative in (*required_base, *required_overlay):
        source_root = base if relative in required_base else overlay
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    actual = {
        relative: sha256(destination / relative)
        for relative in EXPECTED_OVERLAY_SHA256
    }
    if actual != EXPECTED_OVERLAY_SHA256:
        raise RuntimeError(
            f"layout-energy overlay hash mismatch: expected "
            f"{EXPECTED_OVERLAY_SHA256}, found {actual}"
        )


def verify_hash_manifest(path: Path, expected: dict[str, Path]) -> None:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name:
            raise RuntimeError(f"malformed SHA256SUMS line: {line!r}")
        records[name] = digest
    wanted = {name: sha256(value) for name, value in expected.items()}
    if records != wanted:
        raise RuntimeError(f"SHA256SUMS mismatch: expected {wanted}, found {records}")


def hardware_probe() -> dict[str, object]:
    subprocess.run(["nvidia-smi"], check=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    if count != 2:
        raise RuntimeError(f"layout-energy pilot requires exactly 2 GPUs, found {count}")
    devices: list[dict[str, object]] = []
    for index in range(count):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if "T4" not in name.upper():
            raise RuntimeError(f"GPU {index} is not a T4: {name}")
        if capability != (7, 5):
            raise RuntimeError(f"GPU {index} has unexpected capability {capability}")
        device = torch.device(f"cuda:{index}")
        torch.manual_seed(20260711 + index)
        left = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        right = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        product = left @ right
        loss = product.float().square().mean()
        if product.dtype != torch.float16 or not torch.isfinite(product).all():
            raise RuntimeError(f"GPU {index} failed real fp16 matmul")
        torch.cuda.synchronize(device)
        devices.append(
            {
                "index": index,
                "name": name,
                "capability": list(capability),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
                "fp16_matmul_shape": [1024, 1024, 1024],
                "fp16_matmul_mean": float(product.float().mean().item()),
                "fp16_loss": float(loss.item()),
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        del left, right, product, loss
        torch.cuda.empty_cache()
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "device_count": count,
        "devices": devices,
    }


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    label: str,
    telemetry: list[dict[str, object]],
) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
    record: dict[str, object] = {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
    }
    telemetry.append(record)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")
    return record


def execute(wrapper: dict[str, object]) -> None:
    data_root = find_data_root()
    base_root = find_base_root()
    overlay_root, overlay_archive = find_overlay_root()
    code_root = WORKING / "layout_energy_code"
    copy_minimal_code(base_root, overlay_root, code_root)
    puzzle_link = code_root / "puzzle"
    puzzle_link.symlink_to(data_root, target_is_directory=True)

    model = code_root / "src" / "puzzle_assembly" / "layout_energy_transformer.py"
    trainer = code_root / "scripts" / "train_evaluate_layout_energy.py"
    tests = code_root / "tests" / "test_layout_energy_transformer.py"
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    wrapper["inputs"] = {
        "data_root": str(data_root),
        "base_root": str(base_root),
        "overlay_root": str(overlay_root),
        "overlay_archive": None if overlay_archive is None else str(overlay_archive),
        "overlay_archive_sha256": None
        if overlay_archive is None
        else sha256(overlay_archive),
        "model_sha256": sha256(model),
        "trainer_sha256": sha256(trainer),
        "tests_sha256": sha256(tests),
        "manifest_sha256": sha256(manifest),
        "quarantine_sha256": sha256(quarantine),
        "runner_sha256": sha256(Path(__file__)),
    }
    hardware_started = time.perf_counter()
    wrapper["hardware"] = hardware_probe()
    wrapper["hardware_seconds"] = time.perf_counter() - hardware_started
    print(json.dumps({"event": "hardware_passed", **wrapper["hardware"]}, sort_keys=True), flush=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    commands: list[dict[str, object]] = []
    wrapper["commands"] = commands
    run_checked(
        [sys.executable, "-m", "py_compile", str(model), str(trainer), str(tests)],
        cwd=code_root,
        environment=environment,
        label="pycompile",
        telemetry=commands,
    )
    run_checked(
        [sys.executable, "-m", "pytest", "-q", str(tests)],
        cwd=code_root,
        environment=environment,
        label="pytest",
        telemetry=commands,
    )

    smoke_dir = WORKING / "layout_energy_smoke"
    smoke_command = [
        sys.executable,
        str(trainer),
        "--synthetic-smoke",
        "--device",
        "cuda:0",
        "--smoke-steps",
        "2",
        "--output-dir",
        str(smoke_dir),
        "--overwrite",
    ]
    run_checked(
        smoke_command,
        cwd=code_root,
        environment=environment,
        label="single_gpu_synthetic_smoke",
        telemetry=commands,
    )
    smoke_report_path = smoke_dir / "layout_energy_report.json"
    if not smoke_report_path.is_file():
        raise RuntimeError("smoke returned success without a report")
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    if (
        smoke_report.get("kind") != "layout_energy_small_grid_smoke"
        or smoke_report.get("status") != "smoke_passed"
        or smoke_report.get("safe_for_submission") is not False
    ):
        raise RuntimeError(f"invalid smoke report: {smoke_report.get('status')}")

    output_dir = WORKING / "layout_energy_pilot"
    pilot_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(trainer),
        "--data-root",
        str(data_root),
        "--manifest",
        str(manifest),
        "--quarantine",
        str(quarantine),
        "--amp",
        "fp16",
        "--amp-init-scale",
        "1024",
        "--max-consecutive-amp-skips",
        "8",
        "--max-total-amp-skips",
        "32",
        "--train-sources",
        "512",
        "--selection-sources",
        "16",
        "--holdout-sources",
        "16",
        "--epochs",
        "4",
        "--negatives-per-source",
        "4",
        "--eval-negatives",
        "8",
        "--eval-replicas",
        "2",
        "--d-model",
        "256",
        "--heads",
        "8",
        "--local-layers",
        "6",
        "--window-size",
        "6",
        "--global-layers",
        "2",
        "--global-tokens",
        "6",
        "--feedforward-dim",
        "1024",
        "--repair-steps",
        "6",
        "--repair-beam-width",
        "3",
        "--repair-hot-positions",
        "32",
        "--repair-proposals",
        "64",
        "--score-batch-size",
        "6",
        "--gate-min-relative-repair-error-reduction",
        "0.25",
        "--gate-min-control-win-fraction",
        "0.60",
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    run_checked(
        pilot_command,
        cwd=code_root,
        environment=environment,
        label="torchrun_t4x2_full_default",
        telemetry=commands,
    )

    report_path = output_dir / "layout_energy_report.json"
    checkpoint_path = output_dir / "layout_energy_checkpoint.pt"
    hashes_path = output_dir / "SHA256SUMS.txt"
    resume_path = output_dir / "layout_energy_resume_epoch.pt"
    for required in (report_path, checkpoint_path, hashes_path, resume_path):
        if not required.is_file():
            raise RuntimeError(f"pilot returned success without required artifact {required}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection_passed = report.get("selection_gate_passed")
    holdout_passed = report.get("holdout_gate_passed")
    expected_status = (
        "holdout_gate_passed"
        if selection_passed is True and holdout_passed is True
        else "holdout_gate_failed"
    )
    if (
        report.get("kind") != "raw_layout_energy_transformer_bounded_pilot"
        or report.get("safe_for_submission") is not False
        or not isinstance(selection_passed, bool)
        or not isinstance(holdout_passed, bool)
        or report.get("status") != expected_status
    ):
        raise RuntimeError("pilot report violates kind/status/gate/safety contract")
    checkpoint_record = report.get("checkpoint") or {}
    resume_record = report.get("epoch_boundary_resume") or {}
    if (
        checkpoint_record.get("sha256") != sha256(checkpoint_path)
        or resume_record.get("sha256") != sha256(resume_path)
    ):
        raise RuntimeError("report artifact hashes do not match written artifacts")
    verify_hash_manifest(
        hashes_path,
        {
            checkpoint_path.name: checkpoint_path,
            report_path.name: report_path,
            resume_path.name: resume_path,
        },
    )
    wrapper["pilot"] = {
        "status": report.get("status"),
        "selection_gate_passed": report.get("selection_gate_passed"),
        "holdout_gate_passed": report.get("holdout_gate_passed"),
        "selected_epoch": report.get("selected_epoch"),
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "hashes": str(hashes_path),
        "hashes_sha256": sha256(hashes_path),
        "resume_checkpoint": str(resume_path),
        "resume_checkpoint_sha256": sha256(resume_path),
    }
    wrapper["smoke"] = {
        "status": smoke_report.get("status"),
        "report": str(smoke_report_path),
        "report_sha256": sha256(smoke_report_path),
    }
    wrapper["status"] = "complete"


def main() -> None:
    started = time.perf_counter()
    wrapper: dict[str, object] = {
        "schema_version": 1,
        "kind": "layout_energy_t4x2_pilot_wrapper",
        "status": "running",
        "safe_for_submission": False,
    }
    exit_code = 0
    try:
        execute(wrapper)
    except BaseException as error:
        exit_code = 1
        wrapper["status"] = "error"
        wrapper["error_type"] = type(error).__name__
        wrapper["error"] = str(error)
        wrapper["traceback"] = traceback.format_exc()
    finally:
        wrapper["seconds"] = time.perf_counter() - started
        atomic_json(WRAPPER_PATH, wrapper)
        wrapper["wrapper_sha256"] = sha256(WRAPPER_PATH)
        print(json.dumps({"event": "wrapper_final", **wrapper}, default=str), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
