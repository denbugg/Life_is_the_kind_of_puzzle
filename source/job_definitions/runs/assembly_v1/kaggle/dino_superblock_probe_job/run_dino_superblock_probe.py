#!/usr/bin/env python3
"""Run the bounded frozen-DINOv2 4x4-superblock gate on Kaggle T4x2."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
AUTHORITATIVE_REPORT_SHA256 = (
    "cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60"
)
AUTHORITATIVE_MANIFEST_SHA256 = (
    "92233fc5343aac3049ce0327417b645998bf477c6db91a4a852659312949ced6"
)
EMBEDDED_PAYLOAD_SHA256 = (
    "64045be1bcb26926704d10daf34b0668baf2db54ed9d0fc91da56aebfd4b96f3"
)
EMBEDDED_PAYLOAD_B64 = "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"
SUBPROCESS_TIMEOUT_SECONDS = 2640.0
WALL_CAP_SECONDS = 2700.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single(paths: list[Path], label: str) -> Path:
    candidates = sorted(set(path.resolve() for path in paths if path.exists()))
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
        "puzzle data root",
    )


def find_runtime_root() -> Path:
    return single(
        [
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "runtime checkpoint root",
    )


def valid_base_code_root(path: Path) -> bool:
    return (
        (path / "src" / "puzzle_assembly" / "qap.py").is_file()
        and (path / "src" / "puzzle_assembly" / "context_reorg.py").is_file()
        and (path / "configs" / "denoise_splits_seed20260710.json").is_file()
    )


def find_base_code_root() -> Path:
    for preferred in (
        INPUT / "datasets" / "pasha883" / "vsos-solver-rework-night-code",
        INPUT / "vsos-solver-rework-night-code",
    ):
        if valid_base_code_root(preferred):
            return preferred.resolve()
    return single(
        [
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_assembly/qap.py")
            if valid_base_code_root(path.parent.parent.parent)
        ],
        "base solver code root",
    )


def valid_expanded_payload(path: Path) -> bool:
    return (
        (path / "src" / "puzzle_assembly" / "dino_superblock.py").is_file()
        and (path / "scripts" / "train_evaluate_dino_superblock.py").is_file()
        and (
            path
            / "reference"
            / "qap_l1w4_boundary_real16_manifest.json"
        ).is_file()
    )


def materialize_embedded_payload() -> Path:
    payload = base64.b64decode(EMBEDDED_PAYLOAD_B64, validate=True)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EMBEDDED_PAYLOAD_SHA256:
        raise RuntimeError(
            f"embedded payload hash mismatch: {digest} != {EMBEDDED_PAYLOAD_SHA256}"
        )
    path = WORKING / "embedded_dino_superblock_payload.zip"
    path.write_bytes(payload)
    return path


def find_payload() -> Path:
    # Standard Kaggle kernels push uploads only code_file.  The canonical
    # payload is therefore embedded in this runner; external copies are never
    # required for correctness.
    embedded = materialize_embedded_payload()
    if embedded.is_file():
        return embedded
    for direct in (
        Path(__file__).resolve().parent / "dino_superblock_payload.zip",
        Path.cwd() / "dino_superblock_payload.zip",
        WORKING / "dino_superblock_payload.zip",
    ):
        if direct.is_file():
            return direct.resolve()
    for expanded in (
        Path(__file__).resolve().parent / "dino_superblock_payload",
        Path.cwd() / "dino_superblock_payload",
        Path.cwd(),
    ):
        if valid_expanded_payload(expanded):
            return expanded.resolve()
    archives = list(INPUT.glob("**/dino_superblock_payload.zip"))
    if archives:
        return single(archives, "DINO superblock payload archive")
    expanded_roots = [
        path.parent.parent.parent
        for path in INPUT.glob("**/src/puzzle_assembly/dino_superblock.py")
        if valid_expanded_payload(path.parent.parent.parent)
    ]
    return single(expanded_roots, "expanded DINO superblock payload")


def extract_payload(base_code_root: Path, payload: Path) -> Path:
    code_root = WORKING / "dino_superblock_code"
    if code_root.exists():
        shutil.rmtree(code_root)
    shutil.copytree(base_code_root, code_root)
    if payload.is_file():
        with zipfile.ZipFile(payload) as archive:
            root = code_root.resolve()
            for info in archive.infolist():
                destination = (code_root / info.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise RuntimeError(f"unsafe payload member: {info.filename}")
            archive.extractall(code_root)
    else:
        for relative in (
            Path("src/puzzle_assembly/dino_superblock.py"),
            Path("scripts/train_evaluate_dino_superblock.py"),
            Path("reference/qap_l1w4_boundary_real16_manifest.json"),
        ):
            destination = code_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(payload / relative, destination)
    required = [
        code_root / "src" / "puzzle_assembly" / "dino_superblock.py",
        code_root / "scripts" / "train_evaluate_dino_superblock.py",
        code_root / "reference" / "qap_l1w4_boundary_real16_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"payload extraction is incomplete: {missing}")
    reference = required[-1]
    if sha256(reference) != AUTHORITATIVE_MANIFEST_SHA256:
        raise RuntimeError("payload authoritative manifest hash mismatch")
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    if reference_payload.get("source_report_sha256") != AUTHORITATIVE_REPORT_SHA256:
        raise RuntimeError("payload manifest points to the wrong v2 report")
    return code_root


def hardware_probe() -> dict[str, Any]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "tensor_probe_means": [],
    }
    if not result["cuda_available"] or result["device_count"] < 2:
        raise RuntimeError(f"DINO superblock gate requires two CUDA GPUs: {result}")
    if any("T4" not in name.upper() for name in result["devices"][:2]):
        raise RuntimeError(f"DINO superblock gate requires T4x2: {result['devices']}")
    for index in range(2):
        left = torch.randn(128, 128, device=f"cuda:{index}")
        right = torch.randn(128, 128, device=f"cuda:{index}")
        result["tensor_probe_means"].append(float((left @ right).mean().item()))
    return result


def run_and_tee(
    command: list[str],
    *,
    environment: dict[str, str],
    code_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    log = WORKING / "dino_superblock_probe.log"
    started = time.perf_counter()
    timed_out = threading.Event()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=code_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def terminate() -> None:
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(timeout_seconds, terminate)
        timer.daemon = True
        timer.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        finally:
            timer.cancel()
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out.is_set(),
        "timeout_seconds": timeout_seconds,
        "seconds": time.perf_counter() - started,
        "log": str(log),
        "log_sha256": sha256(log),
    }


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    base_code_root = find_base_code_root()
    payload = find_payload()
    code_root = extract_payload(base_code_root, payload)
    probe = hardware_probe()
    print(json.dumps({"event": "hardware", **probe}, sort_keys=True), flush=True)

    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    embedding = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    reference = (
        code_root / "reference" / "qap_l1w4_boundary_real16_manifest.json"
    )
    checkpoint = WORKING / "dino_superblock_head.pt"
    report = WORKING / "dino_superblock_probe_report.json"

    cache_root = WORKING / "dino_model_cache"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["PYTHONHASHSEED"] = "0"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["TORCH_HOME"] = str(cache_root / "torch")
    environment["HF_HOME"] = str(cache_root / "huggingface")
    environment["HUGGINGFACE_HUB_CACHE"] = str(
        cache_root / "huggingface" / "hub"
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    command = [
        sys.executable,
        str(code_root / "scripts" / "train_evaluate_dino_superblock.py"),
        "--data-root",
        str(data_root),
        "--denoiser",
        str(denoiser),
        "--embedding-checkpoint",
        str(embedding),
        "--manifest",
        str(manifest),
        "--quarantine",
        str(quarantine),
        "--authoritative-reference",
        str(reference),
        "--train-sources",
        "512",
        "--dev-sources",
        "64",
        "--exact-sources",
        "8",
        "--real-sources",
        "16",
        "--runtime-cap-seconds",
        "2520",
        "--device",
        "cuda",
        "--output",
        str(checkpoint),
        "--report",
        str(report),
        "--overwrite",
    ]
    elapsed_before_run = time.perf_counter() - started
    timeout_seconds = min(
        SUBPROCESS_TIMEOUT_SECONDS,
        max(1.0, WALL_CAP_SECONDS - elapsed_before_run - 30.0),
    )
    run = run_and_tee(
        command,
        environment=environment,
        code_root=code_root,
        timeout_seconds=timeout_seconds,
    )
    payload_report = (
        json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
    )
    wrapper = {
        "schema_version": 1,
        "kind": "puzzle_dino_vits14_superblock_probe_kaggle_wrapper",
        "probe": probe,
        "mounts": {
            "data_root": str(data_root),
            "runtime_root": str(runtime_root),
            "base_code_root": str(base_code_root),
            "code_root": str(code_root),
            "payload": str(payload),
        },
        "input_hashes": {
            "payload": sha256(payload) if payload.is_file() else None,
            "denoiser": sha256(denoiser),
            "embedding": sha256(embedding),
            "manifest": sha256(manifest),
            "quarantine": sha256(quarantine),
            "authoritative_reference": sha256(reference),
            "dino_superblock_module": sha256(
                code_root / "src" / "puzzle_assembly" / "dino_superblock.py"
            ),
            "train_evaluate_script": sha256(
                code_root / "scripts" / "train_evaluate_dino_superblock.py"
            ),
            "qap_module": sha256(code_root / "src" / "puzzle_assembly" / "qap.py"),
        },
        "run": run,
        "result": {
            "report": str(report) if report.is_file() else None,
            "report_sha256": sha256(report) if report.is_file() else None,
            "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
            "status": payload_report.get("status") if payload_report else "missing_report",
            "accepted": bool(payload_report and payload_report.get("accepted", False)),
            "dino": (
                payload_report.get("frozen_models", {}).get("dino")
                if payload_report
                else None
            ),
        },
        "hard_runtime_cap": {
            "wall_seconds": WALL_CAP_SECONDS,
            "subprocess_timeout_seconds": SUBPROCESS_TIMEOUT_SECONDS,
            "resolved_subprocess_timeout_seconds": timeout_seconds,
            "internal_cap_seconds": 2520,
            "timed_out": run["timed_out"],
        },
        "seconds": time.perf_counter() - started,
    }
    wrapper_path = WORKING / "dino_superblock_probe_wrapper.json"
    wrapper_path.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_paths = [
        path
        for path in (
            Path(run["log"]),
            report,
            checkpoint,
            wrapper_path,
        )
        if path.is_file()
    ]
    hashes = {
        "schema_version": 1,
        "kind": "puzzle_dino_superblock_probe_artifact_hashes",
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
    }
    hash_path = WORKING / "dino_superblock_probe_hashes.json"
    hash_path.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "dino_superblock_wrapper_complete",
                "returncode": run["returncode"],
                "timed_out": run["timed_out"],
                "status": wrapper["result"]["status"],
                "accepted": wrapper["result"]["accepted"],
                "wrapper": str(wrapper_path),
                "wrapper_sha256": sha256(wrapper_path),
                "seconds": wrapper["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if run["returncode"] != 0 or not report.is_file():
        raise SystemExit("DINO superblock probe failed before producing a valid report")


if __name__ == "__main__":
    main()
