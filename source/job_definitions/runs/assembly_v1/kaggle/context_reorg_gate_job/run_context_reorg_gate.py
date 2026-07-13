#!/usr/bin/env python3
"""Train and gate contextual reorganization on a Kaggle T4x2 session."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
AUTHORITATIVE_QAP_REAL16_SSIM = 0.18281991502795386
AUTHORITATIVE_QAP_REPORT_SHA256 = (
    "cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single(candidates: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in candidates if path.exists()))
    if len(values) != 1:
        raise RuntimeError(f"expected one {label}, found {values}")
    return values[0]


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
        "checkpoint runtime root",
    )


def find_base_code_root() -> Path:
    preferred = INPUT / "vsos-solver-rework-night-code"
    if (
        (preferred / "src" / "puzzle_assembly" / "qap.py").is_file()
        and (preferred / "configs" / "denoise_splits_seed20260710.json").is_file()
    ):
        return preferred
    candidates = [
        path.parent.parent.parent
        for path in INPUT.glob("**/src/puzzle_assembly/qap.py")
        if (path.parent.parent.parent / "configs" / "denoise_splits_seed20260710.json").is_file()
    ]
    return single(candidates, "base solver code root")


def find_payload() -> Path:
    direct = [
        Path(__file__).resolve().parent / "context_reorg_payload.zip",
        Path.cwd() / "context_reorg_payload.zip",
        WORKING / "context_reorg_payload.zip",
    ]
    for path in direct:
        if path.is_file():
            return path.resolve()
    archives = list(INPUT.glob("**/context_reorg_payload.zip"))
    if archives:
        return single(archives, "context-reorg payload archive")
    # Kaggle recursively expands a ZIP nested inside an uploaded dataset ZIP.
    # In that representation the payload archive itself disappears, while its
    # deterministic T0 checkpoint and source tree remain under a directory.
    return single(
        list(INPUT.glob("**/context_reorg_payload/checkpoints/t0_gpu_full.pt")),
        "expanded context-reorg payload checkpoint",
    )


def extract_payload(base_code_root: Path, payload: Path) -> Path:
    code_root = WORKING / "context_reorg_code"
    if code_root.exists():
        shutil.rmtree(code_root)
    shutil.copytree(base_code_root, code_root)
    if payload.suffix.lower() == ".zip":
        with zipfile.ZipFile(payload) as archive:
            root = code_root.resolve()
            for info in archive.infolist():
                destination = (code_root / info.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise RuntimeError(f"unsafe payload member: {info.filename}")
            archive.extractall(code_root)
    else:
        # The current code dataset already contains the three context-reorg
        # source files at their normal paths.  Only the extracted frozen T0
        # checkpoint must be restored to the payload's expected location.
        checkpoint = code_root / "checkpoints" / "t0_gpu_full.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(payload, checkpoint)
    required = [
        code_root / "src" / "puzzle_assembly" / "context_reorg.py",
        code_root / "scripts" / "train_context_reorg.py",
        code_root / "scripts" / "evaluate_context_reorg.py",
        code_root / "checkpoints" / "t0_gpu_full.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"payload extraction is incomplete: {missing}")
    return code_root


def hardware_probe() -> dict:
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
        "matmul_means": [],
    }
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"context gate requires a two-GPU session, got {result}")
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        left = torch.randn(256, 256, device=device)
        right = torch.randn(256, 256, device=device)
        result["matmul_means"].append(float((left @ right).mean().item()))
    return result


def run_command(
    name: str,
    command: list[str],
    *,
    gpu: int,
    code_root: Path,
) -> dict:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(code_root / "src")
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    started = time.perf_counter()
    print(
        json.dumps(
            {"event": "context_reorg_job_start", "name": name, "gpu": gpu},
            sort_keys=True,
        ),
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
        "command": command,
    }
    print(
        json.dumps(
            {"event": "context_reorg_job_complete", **record}, sort_keys=True
        ),
        flush=True,
    )
    return record


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
    context = code_root / "checkpoints" / "t0_gpu_full.pt"
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    checkpoint = WORKING / "context_reorg_r0.pt"
    training_report = WORKING / "context_reorg_r0_training.json"
    exact_report = WORKING / "context_reorg_exact8.json"
    real_report = WORKING / "context_reorg_real16.json"

    common = [
        "--data-root",
        str(data_root),
        "--denoiser",
        str(denoiser),
        "--embedding-checkpoint",
        str(embedding),
        "--context-checkpoint",
        str(context),
        "--manifest",
        str(manifest),
        "--quarantine",
        str(quarantine),
        "--device",
        "cuda",
        "--denoise-batch-size",
        "512",
        "--chunk-size",
        "64",
    ]
    train_command = [
        sys.executable,
        str(code_root / "scripts" / "train_context_reorg.py"),
        *common,
        "--train-split",
        "edge_train",
        "--train-offset",
        "0",
        "--train-sources",
        "24",
        "--val-split",
        "edge_development",
        "--val-offset",
        "0",
        "--val-sources",
        "4",
        "--panel",
        "primary_kornia",
        "--replicas-per-source",
        "1",
        "--epochs",
        "3",
        "--corruptions-per-source",
        "4",
        "--correction-rounds",
        "2",
        "--qap-iterations",
        "8",
        "--qap-restarts",
        "1",
        "--qap-boundary-weight",
        "0.05",
        "--qap-refine-swaps",
        "4",
        "--model-dim",
        "96",
        "--layers",
        "2",
        "--heads",
        "4",
        "--feedforward-dim",
        "256",
        "--match-dim",
        "32",
        "--amp",
        "--output",
        str(checkpoint),
        "--report",
        str(training_report),
        "--overwrite",
    ]
    training = run_command(
        "train_context_reorg", train_command, gpu=0, code_root=code_root
    )
    if training["returncode"] != 0 or not checkpoint.is_file():
        raise RuntimeError(f"context-reorg training failed: {training}")

    evaluation_common = [
        sys.executable,
        str(code_root / "scripts" / "evaluate_context_reorg.py"),
        *common,
        "--reorg-checkpoint",
        str(checkpoint),
        "--rounds",
        "2",
        "--qap-iterations",
        "25",
        "--qap-restarts",
        "2",
        "--qap-boundary-weight",
        "0.05",
        "--qap-refine-swaps",
        "8",
        "--min-real-ssim-delta",
        "0.02",
        "--min-exact-wrong-reduction",
        "0.10",
        "--overwrite",
    ]
    exact_command = [
        *evaluation_common,
        "--exact-split",
        "edge_development",
        "--exact-offset",
        "8",
        "--exact-sources",
        "8",
        "--real-sources",
        "0",
        "--output",
        str(exact_report),
    ]
    real_command = [
        *evaluation_common,
        "--exact-sources",
        "0",
        "--real-split",
        "assembly_cal",
        "--real-offset",
        "0",
        "--real-sources",
        "16",
        "--output",
        str(real_report),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        exact_future = executor.submit(
            run_command,
            "evaluate_context_reorg_exact8",
            exact_command,
            gpu=0,
            code_root=code_root,
        )
        real_future = executor.submit(
            run_command,
            "evaluate_context_reorg_real16",
            real_command,
            gpu=1,
            code_root=code_root,
        )
        evaluations = [exact_future.result(), real_future.result()]
    failures = [record for record in evaluations if record["returncode"] != 0]
    if failures or not exact_report.is_file() or not real_report.is_file():
        raise RuntimeError(f"context-reorg evaluation failed: {failures}")

    training_payload = json.loads(training_report.read_text(encoding="utf-8"))
    exact_payload = json.loads(exact_report.read_text(encoding="utf-8"))
    real_payload = json.loads(real_report.read_text(encoding="utf-8"))
    exact_reduction = float(
        exact_payload["exact"]["macro"]["wrong_position_reduction"]
    )
    real_seed_ssim = float(
        real_payload["real"]["macro"]["seed_image"]["predicted_layout_ssim"]
    )
    real_delta = float(real_payload["real"]["macro"]["ssim_delta"])
    exact_pass = exact_reduction >= 0.10
    real_pass = real_delta >= 0.02
    train_names = set(
        training_payload["whole_source_split"]["train_names"]
        + training_payload["whole_source_split"]["validation_names"]
    )
    exact_names = set(exact_payload["source_names"]["exact"])
    real_names = set(real_payload["source_names"]["real"])
    split_intersections = {
        "train_or_val_vs_exact": sorted(train_names & exact_names),
        "train_or_val_vs_real": sorted(train_names & real_names),
        "exact_vs_real": sorted(exact_names & real_names),
    }
    split_safe = not any(split_intersections.values())
    leakage_safe = bool(
        real_payload["anti_leakage"][
            "real_layouts_frozen_before_any_target_read"
        ]
    )
    baseline_delta = real_seed_ssim - AUTHORITATIVE_QAP_REAL16_SSIM
    baseline_reproduced = abs(baseline_delta) <= 1e-6
    promote = bool(
        exact_pass and real_pass and split_safe and leakage_safe and baseline_reproduced
    )

    wrapper = {
        "schema_version": 1,
        "kind": "puzzle_context_reorganization_r0_kaggle_gate",
        "probe": probe,
        "paths": {
            "data_root": str(data_root),
            "runtime_root": str(runtime_root),
            "base_code_root": str(base_code_root),
            "code_root": str(code_root),
            "payload": str(payload),
        },
        "hashes": {
            "payload": sha256(payload),
            "context_reorg_module": sha256(
                code_root / "src" / "puzzle_assembly" / "context_reorg.py"
            ),
            "trainer": sha256(code_root / "scripts" / "train_context_reorg.py"),
            "evaluator": sha256(
                code_root / "scripts" / "evaluate_context_reorg.py"
            ),
            "denoiser": sha256(denoiser),
            "embedding": sha256(embedding),
            "context": sha256(context),
            "checkpoint": sha256(checkpoint),
            "training_report": sha256(training_report),
            "exact_report": sha256(exact_report),
            "real_report": sha256(real_report),
        },
        "commands": {
            "training": training,
            "evaluations": evaluations,
        },
        "authoritative_qap_reference": {
            "report": "qap_l1w4_boundary_real16.json",
            "report_sha256": AUTHORITATIVE_QAP_REPORT_SHA256,
            "denoised_render_ssim": AUTHORITATIVE_QAP_REAL16_SSIM,
            "gate_seed_ssim": real_seed_ssim,
            "delta": baseline_delta,
            "reproduced_within_1e-6": baseline_reproduced,
        },
        "whole_source_split": {
            "intersections": split_intersections,
            "safe": split_safe,
        },
        "anti_leakage_pass": leakage_safe,
        "gate": {
            "exact_wrong_position_reduction": exact_reduction,
            "exact_threshold": 0.10,
            "exact_pass": exact_pass,
            "real_ssim_delta_vs_qap": real_delta,
            "real_threshold": 0.02,
            "real_pass": real_pass,
            "promote": promote,
            "rule": (
                "promote only when exact wrong positions fall by >=10%, real16 "
                "SSIM rises by >=0.02 over the reproduced fixed QAP seed, and "
                "split/leakage audits pass"
            ),
        },
        "seconds": time.perf_counter() - started,
    }
    wrapper_path = WORKING / "context_reorg_gate_report.json"
    wrapper_path.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "context_reorg_gate_finished",
                "report": str(wrapper_path),
                "gate": wrapper["gate"],
                "seconds": wrapper["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
