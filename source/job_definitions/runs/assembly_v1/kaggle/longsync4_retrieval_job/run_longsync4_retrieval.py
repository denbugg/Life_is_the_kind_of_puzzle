#!/usr/bin/env python3
"""Run the frozen LongSync-4 retrieval-only diagnostic on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WRAPPER = WORKING / "longsync4_retrieval_wrapper.json"
REPORT = WORKING / "longsync4_retrieval_report.json"
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "full_union_tabular.joblib": "c5929a76c843f7541119f622bf1c5b6774006ad79e3811407e36edfe60bd0f10",
}
EXPECTED_CODE = {
    "scripts/evaluate_longsync4_retrieval.py": "78a3e7019d84d669bf514b0343da9109a226c10fca1c6c1e7680d3d60dede6d3",
    "src/puzzle_assembly/longsync_translation.py": "7592d1e5c986d430aba5139abdfb5774804e4c63b8d05ae2ba97dbb56744871f",
    "scripts/train_binary_edge_verifier.py": "ef3686ebc015d6647ddcc8878d3ac4b9cafb558ab4408667206da282bdaebab9",
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
    "configs/assembly_audit_exclusion_v1.json": "772e89ad4f633d2050f8ad3806cd24bffed132bcd8914951b7b8edff3f608ab6",
}
EXPECTED_SOURCE_NAMES_SHA256 = (
    "c0f9548268a4e72a07e987cfdedf98047313e61967758401b297ff60f82ff7c7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def find_asset(name: str) -> Path:
    path = exactly_one(list(INPUT_ROOT.rglob(name)), name)
    actual = sha256(path)
    if actual != EXPECTED_ASSETS[name]:
        raise RuntimeError(f"asset hash mismatch for {name}: {actual}")
    return path


def write_wrapper(payload: dict) -> None:
    WRAPPER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def hardware_probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    devices = []
    for index in range(torch.cuda.device_count()):
        value = torch.randn(128, 128, device=f"cuda:{index}")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "tensor_op": float((value @ value).mean().cpu()),
            }
        )
    smi = subprocess.run(
        ["nvidia-smi"], capture_output=True, check=False, text=True
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "nvidia_smi": smi.stdout,
    }


def verify_code(bundle_root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in EXPECTED_CODE.items():
        path = (bundle_root / relative).resolve()
        try:
            path.relative_to(bundle_root.resolve())
        except ValueError as error:
            raise RuntimeError(f"code path escaped bundle: {relative}") from error
        if not path.is_file():
            raise RuntimeError(f"missing frozen code/config: {relative}")
        actual[relative] = sha256(path)
        if actual[relative] != expected:
            raise RuntimeError(
                f"code/config hash mismatch for {relative}: {actual[relative]}"
            )
    return actual


def verify_report(payload: dict) -> None:
    if payload.get("kind") != "longsync4_translation_hgb_top2_retrieval_diagnostic":
        raise RuntimeError("unexpected report kind")
    if payload.get("safe_for_submission") is not False:
        raise RuntimeError("retrieval diagnostic must not be submission-safe")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("missing report protocol")
    expected_protocol = {
        "split": "edge_development",
        "source_offset": 316,
        "source_count": 8,
        "source_names_sha256": EXPECTED_SOURCE_NAMES_SHA256,
        "top_k": 2,
        "iterations": 10,
        "parameter_sweeps": 0,
        "assembly_targets_opened": False,
        "whole_source_disjoint_from_hgb_fit_calibration": True,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(
                f"report protocol drift for {key}: {protocol.get(key)!r}"
            )
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 16:
        raise RuntimeError("report must contain exactly 16 source-panel records")
    panels = {record.get("panel") for record in records if isinstance(record, dict)}
    if panels != {"primary_kornia", "independent_libjpeg"}:
        raise RuntimeError(f"report panel drift: {panels}")
    gate = payload.get("gate")
    if not isinstance(gate, dict) or gate.get("decision") not in {
        "continue_to_disjoint_assembly_gate",
        "stop_no_retrieval_signal",
    }:
        raise RuntimeError("invalid frozen gate decision")


def main() -> None:
    started = time.time()
    wrapper = {
        "schema_version": 1,
        "kind": "longsync4_retrieval_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write_wrapper(wrapper)
    try:
        evaluator = exactly_one(
            list(INPUT_ROOT.rglob("scripts/evaluate_longsync4_retrieval.py")),
            "LongSync retrieval evaluator",
        )
        bundle_root = evaluator.parents[1]
        code_hashes = verify_code(bundle_root)
        targets = exactly_one(list(INPUT_ROOT.rglob("train/targets")), "train targets")
        data_root = targets.parent.parent
        denoiser = find_asset("selected_tilenaf_synth_50k.pt")
        embedding = find_asset("hbt_d320_denoised_rgb_sobel.pt")
        model = find_asset("full_union_tabular.joblib")
        probe = hardware_probe()
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(bundle_root / "src"),
                "PYTHONHASHSEED": "20260713",
                "PYTHONUNBUFFERED": "1",
                "OPENBLAS_NUM_THREADS": "4",
                "OMP_NUM_THREADS": "4",
            }
        )
        command = [
            sys.executable,
            str(evaluator),
            "--model",
            str(model),
            "--data-root",
            str(data_root),
            "--denoiser",
            str(denoiser),
            "--embedding-checkpoint",
            str(embedding),
            "--manifest",
            str(bundle_root / "configs/denoise_splits_seed20260710.json"),
            "--quarantine",
            str(bundle_root / "configs/denoise_validation_quarantine_v1.json"),
            "--audit-exclusion",
            str(bundle_root / "configs/assembly_audit_exclusion_v1.json"),
            "--split",
            "edge_development",
            "--source-offset",
            "316",
            "--sources",
            "8",
            "--top-k",
            "2",
            "--iterations",
            "10",
            "--device",
            "cuda",
            "--denoise-batch-size",
            "512",
            "--output",
            str(REPORT),
            "--overwrite",
        ]
        wrapper.update(
            {
                "status": "running",
                "hardware": probe,
                "bundle_root": str(bundle_root),
                "data_root": str(data_root),
                "code_hashes": code_hashes,
                "assets": {
                    "denoiser": {
                        "path": str(denoiser),
                        "sha256": sha256(denoiser),
                    },
                    "embedding": {
                        "path": str(embedding),
                        "sha256": sha256(embedding),
                    },
                    "model": {"path": str(model), "sha256": sha256(model)},
                },
                "command": command,
            }
        )
        write_wrapper(wrapper)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            raise RuntimeError(f"evaluator exited with code {completed.returncode}")
        result = json.loads(REPORT.read_text())
        verify_report(result)
        wrapper.update(
            {
                "status": "complete",
                "gate": result["gate"],
                "report": {"path": str(REPORT), "sha256": sha256(REPORT)},
                "seconds": time.time() - started,
            }
        )
        write_wrapper(wrapper)
        print(json.dumps(wrapper, sort_keys=True), flush=True)
    except Exception as error:
        wrapper.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "seconds": time.time() - started,
            }
        )
        write_wrapper(wrapper)
        raise


if __name__ == "__main__":
    main()
