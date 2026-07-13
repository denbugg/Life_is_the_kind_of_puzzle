#!/usr/bin/env python3
"""Run the frozen two-panel exact axis path-cover prerequisite on Kaggle."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
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
WRAPPER = WORKING / "path_cover_gate_wrapper.json"
FINAL_REPORT = WORKING / "path_cover_gate_report.json"
PANELS = ("primary_kornia", "independent_libjpeg")
EXPECTED_SOURCE_NAMES_SHA256 = (
    "93a429dec71ad1abd28df5b981b9142ac89525a0d3d092dc0078a4a0d27f128c"
)
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
EXPECTED_CODE = {
    "scripts/evaluate_path_cover_gate.py": "8ddd30f3b1259a9f56154750c0049b0d62b92a1f7f908de42bfb8e673fa97b3e",
    "scripts/train_binary_edge_verifier.py": "ef3686ebc015d6647ddcc8878d3ac4b9cafb558ab4408667206da282bdaebab9",
    "src/puzzle_assembly/path_cover.py": "9d2ed84621beaa90cba11158cae2a4c610c73a408311ae444b47f2490e191023",
    "tests/test_path_cover.py": "dce4a30681e1dcedc47d4fa118412afa7d7dc11958fd7370cba2648767c837ad",
    "tests/test_evaluate_path_cover_gate.py": "77e2095d3cea7d9eddf591cbdc8641e8dcf1001b805b8feba5c227b673e03881",
    "src/puzzle_assembly/components.py": "655af23f2705e0a22cb334fb5d5b282795f8c0de35c22f265966118cce0c34b0",
    "src/puzzle_assembly/qap.py": "b8a5e1da67387fd04effd979270ca16925aceab23d37083eda108c6e3e349c32",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_denoise_v2/inference.py": "20767cc26270cfde7472cf33a0247b1ea6d96e5b5c8ff5d705b785ae710dd6da",
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
    "configs/assembly_audit_exclusion_v1.json": "772e89ad4f633d2050f8ad3806cd24bffed132bcd8914951b7b8edff3f608ab6",
}


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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    if completed.returncode:
        raise RuntimeError(f"OR-Tools install failed: {completed.returncode}")
    return importlib.metadata.version("ortools")


def hardware_probe() -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("frozen job requires two visible CUDA GPUs")
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
    smi = subprocess.run(["nvidia-smi"], capture_output=True, check=False, text=True)
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
            raise RuntimeError(f"code hash mismatch for {relative}: {actual[relative]}")
    return actual


def run_correctness_tests(bundle_root: Path, environment: dict[str, str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_path_cover.py",
        "tests/test_evaluate_path_cover_gate.py",
    ]
    completed = subprocess.run(
        command,
        cwd=bundle_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode:
        raise RuntimeError(f"path-cover correctness tests failed: {record}")
    if "skipped" in completed.stdout.lower():
        raise RuntimeError(f"OR-Tools correctness tests were skipped: {record}")
    return record


def run_panel(
    panel: str,
    gpu: int,
    *,
    evaluator: Path,
    bundle_root: Path,
    data_root: Path,
    denoiser: Path,
    embedding: Path,
    base_environment: dict[str, str],
) -> dict:
    output = WORKING / f"path_cover_gate_{panel}.json"
    command = [
        sys.executable,
        str(evaluator),
        "--panel",
        panel,
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
        "332",
        "--sources",
        "8",
        "--outgoing-top-k",
        "16",
        "--incoming-top-k",
        "16",
        "--time-limit-seconds",
        "30",
        "--device",
        "cuda",
        "--denoise-batch-size",
        "512",
        "--output",
        str(output),
        "--overwrite",
    ]
    environment = dict(base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    completed = subprocess.run(command, cwd=bundle_root, env=environment, check=False)
    record = {
        "panel": panel,
        "gpu": gpu,
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.time() - started,
        "output": str(output),
    }
    if completed.returncode:
        raise RuntimeError(f"panel evaluator failed: {record}")
    report = json.loads(output.read_text())
    if report.get("kind") != "exact_24x24_axis_path_cover_prerequisite_panel":
        raise RuntimeError(f"unexpected panel report kind: {panel}")
    if report.get("panel") != panel or len(report.get("records", [])) != 8:
        raise RuntimeError(f"panel report schema drift: {panel}")
    protocol = report.get("protocol", {})
    expected_protocol = {
        "source_names_sha256": EXPECTED_SOURCE_NAMES_SHA256,
        "source_offset": 332,
        "source_count": 8,
        "outgoing_top_k": 16,
        "incoming_top_k": 16,
        "path_count": 24,
        "path_length": 24,
        "parameter_sweeps": 0,
        "assembly_layout_constructed": False,
        "layout_ssim_opened": False,
        "solver_accepts_target_or_truth": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"panel protocol drift {panel}.{key}")
    record["sha256"] = sha256(output)
    record["gate"] = report["gate"]
    return record


def main() -> None:
    started = time.time()
    wrapper = {
        "schema_version": 1,
        "kind": "exact_axis_path_cover_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write_json(WRAPPER, wrapper)
    try:
        evaluator = exactly_one(
            list(INPUT_ROOT.rglob("scripts/evaluate_path_cover_gate.py")),
            "path-cover evaluator",
        )
        bundle_root = evaluator.parents[1]
        code_hashes = verify_code(bundle_root)
        targets = exactly_one(list(INPUT_ROOT.rglob("train/targets")), "train targets")
        data_root = targets.parent.parent
        denoiser = find_asset("selected_tilenaf_synth_50k.pt")
        embedding = find_asset("hbt_d320_denoised_rgb_sobel.pt")
        ortools_version = install_ortools()
        probe = hardware_probe()
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(bundle_root / "src"),
                "PYTHONHASHSEED": "20260713",
                "PYTHONUNBUFFERED": "1",
                "OPENBLAS_NUM_THREADS": "2",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            }
        )
        tests = run_correctness_tests(bundle_root, environment)
        wrapper.update(
            {
                "status": "running",
                "ortools": ortools_version,
                "hardware": probe,
                "bundle_root": str(bundle_root),
                "data_root": str(data_root),
                "code_hashes": code_hashes,
                "correctness_tests": tests,
                "assets": {
                    "denoiser": {"path": str(denoiser), "sha256": sha256(denoiser)},
                    "embedding": {"path": str(embedding), "sha256": sha256(embedding)},
                },
            }
        )
        write_json(WRAPPER, wrapper)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_panel,
                    panel,
                    gpu,
                    evaluator=evaluator,
                    bundle_root=bundle_root,
                    data_root=data_root,
                    denoiser=denoiser,
                    embedding=embedding,
                    base_environment=environment,
                )
                for gpu, panel in enumerate(PANELS)
            ]
            panel_records = [future.result() for future in futures]
        panel_reports = {
            record["panel"]: json.loads(Path(record["output"]).read_text())
            for record in panel_records
        }
        panel_pass = {
            panel: bool(panel_reports[panel]["gate"]["passed"]) for panel in PANELS
        }
        passed = all(panel_pass.values())
        final = {
            "schema_version": 1,
            "kind": "exact_24x24_axis_path_cover_prerequisite_two_panel",
            "safe_for_submission": False,
            "protocol": {
                "panels": list(PANELS),
                "source_names_sha256": EXPECTED_SOURCE_NAMES_SHA256,
                "parameter_sweeps": 0,
                "layout_or_ssim_constructed": False,
                "decision_requires_both_panels": True,
            },
            "panels": {
                panel: {
                    "summary": panel_reports[panel]["summary"],
                    "gate": panel_reports[panel]["gate"],
                    "report_sha256": sha256(Path(next(
                        record["output"] for record in panel_records if record["panel"] == panel
                    ))),
                }
                for panel in PANELS
            },
            "gate": {
                "passed": passed,
                "checks": {f"{panel}_passed": value for panel, value in panel_pass.items()},
                "decision": (
                    "continue_to_disjoint_cross_axis_reconciliation"
                    if passed
                    else "stop_path_cover_no_axis_signal"
                ),
                "scope": "axis paths only; no 2D candidate or SSIM was constructed",
            },
            "records": [
                item
                for panel in PANELS
                for item in panel_reports[panel]["records"]
            ],
            "seconds": time.time() - started,
        }
        write_json(FINAL_REPORT, final)
        wrapper.update(
            {
                "status": "complete",
                "panel_records": panel_records,
                "gate": final["gate"],
                "report": {"path": str(FINAL_REPORT), "sha256": sha256(FINAL_REPORT)},
                "seconds": time.time() - started,
            }
        )
        write_json(WRAPPER, wrapper)
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
        write_json(WRAPPER, wrapper)
        raise


if __name__ == "__main__":
    main()
