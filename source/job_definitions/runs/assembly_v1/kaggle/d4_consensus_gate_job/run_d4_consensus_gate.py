#!/usr/bin/env python3
"""Run the frozen two-panel D4 compatibility-consensus gate on Kaggle."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
WRAPPER = WORKING / "d4_consensus_gate_wrapper.json"
FINAL_REPORT = WORKING / "d4_consensus_gate_report.json"
PANELS = ("primary_kornia", "independent_libjpeg")
EXPECTED_SOURCE_NAMES_SHA256 = (
    "ddc0a394fbac1f5674a1f724de1b0617e32e0e26d308c6ff5ee369d6452055be"
)
EXPECTED_ASSETS = {
    "selected_tilenaf_synth_50k.pt": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "seam_denoiser_gpu.pt": "f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9",
    "hbt_d320_denoised_rgb_sobel.pt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
EXPECTED_CODE = {
    # Filled and pinned before upload.  The wrapper refuses any semantic drift.
    "scripts/evaluate_d4_consensus_gate.py": "9e7f592f42d9ed917941fdc5b1e78323f3ce6991624bbb211e3ffaab05616443",
    "scripts/build_assembly_submission.py": "8433c0e545edfeb49f2512208a3ea062fb1a248a64bcde3f87037cdf30d6ac97",
    "references/d4_consensus_slow_phase_a_v2_primary_kornia.json": "66710a9e2e42b98658ca7513d8bb08b5458c84152552114f79640c8bd1afd45b",
    "references/d4_consensus_slow_phase_a_v2_independent_libjpeg.json": "f9c132f45b1520cb07d03b405509efdf5802cffd71dcf0546c2033387554e845",
    "src/puzzle_assembly/d4_consensus.py": "9f495ab0fee98e4e7af89d06be089c480874bf29549dbb159f8dec70bf2a8539",
    "tests/test_d4_consensus.py": "20361d9ebd6c1ae4fe56a02a1c73a1d70aeb87b5d02180a48c0fe1194892e22a",
    "tests/test_evaluate_d4_consensus_gate.py": "c0934a40e4b8bbac3c9e0e7a7950d11cf0720ca7ccbccb388e4135d6a668efd4",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/components.py": "655af23f2705e0a22cb334fb5d5b282795f8c0de35c22f265966118cce0c34b0",
    "src/puzzle_assembly/qap.py": "b8a5e1da67387fd04effd979270ca16925aceab23d37083eda108c6e3e349c32",
    "src/puzzle_assembly/postassembly_harmonizer.py": "cd67d99a43f40932821e31e732c8988edb4e930661dd37314505e5128e75aec0",
    "src/puzzle_assembly/learned.py": "9e3dba673aa85eaab5698dbeb63b3d94f88e3ea92b5e5979bde4b0273642697b",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_assembly/protocol.py": "7651d4405ce4dd35203a0cae7bfdd591621044f9e90dc522a314262727c86eca",
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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def exactly_one(paths: list[Path], label: str) -> Path:
    values = sorted(set(path.resolve() for path in paths))
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, got {values}")
    return values[0]


def find_asset(name: str) -> Path:
    expected = EXPECTED_ASSETS[name]
    matches = sorted(
        {path.resolve() for path in INPUT_ROOT.rglob(name) if sha256(path) == expected}
    )
    if not matches:
        raise RuntimeError(f"no hash-pinned asset found for {name}")
    # Multiple mounted datasets may carry the identical promoted asset.  The
    # content hash, not the mount path, is the frozen identity.
    return matches[0]


def verify_code(bundle_root: Path) -> dict[str, str]:
    actual = {}
    for relative, expected in EXPECTED_CODE.items():
        if expected.startswith("__"):
            raise RuntimeError(f"unresolved code-hash placeholder: {relative}")
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


def hardware_probe() -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("frozen D4 job requires two visible CUDA GPUs")
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


def run_tests(bundle_root: Path, environment: dict[str, str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_d4_consensus.py",
        "tests/test_evaluate_d4_consensus_gate.py",
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
    if completed.returncode or "skipped" in completed.stdout.lower():
        raise RuntimeError(f"D4 correctness tests failed or skipped: {record}")
    return record


def freshness_scan(bundle_root: Path, environment: dict[str, str]) -> dict:
    command = [
        sys.executable,
        "-c",
        (
            "from puzzle_assembly.protocol import source_names_for_split as f;"
            "n=f('edge_development',manifest_path='configs/denoise_splits_seed20260710.json',"
            "quarantine_path='configs/denoise_validation_quarantine_v1.json',"
            "audit_exclusion_path='configs/assembly_audit_exclusion_v1.json')[340:348];"
            "print('\\n'.join(n))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=bundle_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"source-name resolution failed: {completed.stderr}")
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    digest = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    if len(names) != 8 or digest != EXPECTED_SOURCE_NAMES_SHA256:
        raise RuntimeError("frozen source-name slice drift")
    excluded = {
        (bundle_root / "configs/denoise_splits_seed20260710.json").resolve(),
        (bundle_root / "configs/denoise_validation_quarantine_v1.json").resolve(),
        (bundle_root / "configs/assembly_audit_exclusion_v1.json").resolve(),
    }
    excluded.update(
        (bundle_root / relative).resolve()
        for relative in EXPECTED_CODE
        if relative.startswith("references/")
    )
    hits = []
    for path in bundle_root.rglob("*"):
        if not path.is_file() or path.resolve() in excluded or path.suffix in {".pyc", ".zip"}:
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for name in names:
            if name in text:
                hits.append({"path": str(path.relative_to(bundle_root)), "name": name})
    if hits:
        raise RuntimeError(f"fresh exact8 basenames occur in semantic bundle files: {hits}")
    return {
        "names": names,
        "names_sha256": digest,
        "semantic_hits": hits,
        "excluded_hash_pinned_input_only_references": sorted(
            str(path.relative_to(bundle_root)) for path in excluded if "references" in path.parts
        ),
    }


def panel_base_command(
    panel: str,
    *,
    evaluator: Path,
    bundle_root: Path,
    data_root: Path,
    selected: Path,
    seam: Path,
    embedding: Path,
    slow_reference: Path,
    output: Path,
    artifact: Path,
) -> list[str]:
    return [
        sys.executable,
        str(evaluator),
        "--panel",
        panel,
        "--data-root",
        str(data_root),
        "--selected-denoiser",
        str(selected),
        "--seam-denoiser",
        str(seam),
        "--embedding-checkpoint",
        str(embedding),
        "--slow-phase-a-reference",
        str(slow_reference),
        "--manifest",
        str(bundle_root / "configs/denoise_splits_seed20260710.json"),
        "--quarantine",
        str(bundle_root / "configs/denoise_validation_quarantine_v1.json"),
        "--audit-exclusion",
        str(bundle_root / "configs/assembly_audit_exclusion_v1.json"),
        "--split",
        "edge_development",
        "--source-offset",
        "340",
        "--sources",
        "8",
        "--device",
        "cuda",
        "--denoise-batch-size",
        "512",
        "--phase-a-artifact",
        str(artifact),
        "--output",
        str(output),
        "--overwrite",
    ]


def run_phase_a(
    panel: str,
    gpu: int,
    *,
    evaluator: Path,
    bundle_root: Path,
    data_root: Path,
    selected: Path,
    seam: Path,
    embedding: Path,
    slow_reference: Path,
    base_environment: dict[str, str],
) -> dict:
    output = WORKING / f"d4_consensus_phase_a_{panel}.json"
    artifact = WORKING / f"d4_consensus_phase_a_{panel}.npz"
    command = panel_base_command(
        panel,
        evaluator=evaluator,
        bundle_root=bundle_root,
        data_root=data_root,
        selected=selected,
        seam=seam,
        embedding=embedding,
        slow_reference=slow_reference,
        output=output,
        artifact=artifact,
    )
    command.extend(["--phase", "phase-a"])
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
        raise RuntimeError(f"Phase-A panel evaluator failed: {record}")
    report = json.loads(output.read_text())
    if report.get("kind") != "d4_compatibility_consensus_phase_a":
        raise RuntimeError(f"unexpected D4 Phase-A report kind: {panel}")
    if report.get("panel") != panel:
        raise RuntimeError(f"panel report mismatch: {panel}")
    protocol = report.get("protocol", {})
    expected = {
        "source_offset": 340,
        "source_count": 8,
        "source_names_sha256": EXPECTED_SOURCE_NAMES_SHA256,
        "parameter_sweeps": 0,
        "phase_a_target_metrics_opened": False,
        "solver_accepts_target_or_truth": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"panel protocol drift {panel}.{key}")
    if report.get("artifact", {}).get("sha256") != sha256(artifact):
        raise RuntimeError(f"Phase-A artifact hash mismatch: {panel}")
    record["report_sha256"] = sha256(output)
    record["artifact"] = str(artifact)
    record["artifact_sha256"] = sha256(artifact)
    record["phase_a"] = report["phase_a"]
    return record


def run_phase_b(
    phase_a_record: dict,
    gpu: int,
    *,
    authorization: Path,
    evaluator: Path,
    bundle_root: Path,
    data_root: Path,
    selected: Path,
    seam: Path,
    embedding: Path,
    slow_reference: Path,
    base_environment: dict[str, str],
) -> dict:
    panel = phase_a_record["panel"]
    output = WORKING / f"d4_consensus_gate_{panel}.json"
    command = panel_base_command(
        panel,
        evaluator=evaluator,
        bundle_root=bundle_root,
        data_root=data_root,
        selected=selected,
        seam=seam,
        embedding=embedding,
        slow_reference=slow_reference,
        output=output,
        artifact=Path(phase_a_record["artifact"]),
    )
    command.extend(
        [
            "--phase",
            "phase-b",
            "--phase-a-report",
            phase_a_record["output"],
            "--phase-b-authorization",
            str(authorization),
        ]
    )
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
        raise RuntimeError(f"Phase-B panel evaluator failed: {record}")
    report = json.loads(output.read_text())
    if report.get("kind") != "d4_compatibility_consensus_exact_panel_gate":
        raise RuntimeError(f"unexpected D4 Phase-B report kind: {panel}")
    if report.get("panel") != panel or len(report.get("records", [])) != 8:
        raise RuntimeError(f"Phase-B panel report drift: {panel}")
    if report.get("phase_b_authorization_sha256") != sha256(authorization):
        raise RuntimeError(f"Phase-B authorization receipt drift: {panel}")
    record["sha256"] = sha256(output)
    record["gate"] = report["gate"]
    return record


def main() -> None:
    started = time.time()
    wrapper = {
        "schema_version": 1,
        "kind": "d4_compatibility_consensus_kaggle_wrapper",
        "status": "starting",
        "safe_for_submission": False,
        "started_unix": started,
    }
    write_json(WRAPPER, wrapper)
    try:
        evaluator = exactly_one(
            list(INPUT_ROOT.rglob("scripts/evaluate_d4_consensus_gate.py")),
            "D4 evaluator",
        )
        bundle_root = evaluator.parents[1]
        code_hashes = verify_code(bundle_root)
        targets = exactly_one(list(INPUT_ROOT.rglob("train/targets")), "train targets")
        data_root = targets.parent.parent
        selected = find_asset("selected_tilenaf_synth_50k.pt")
        seam = find_asset("seam_denoiser_gpu.pt")
        embedding = find_asset("hbt_d320_denoised_rgb_sobel.pt")
        slow_references = {
            panel: bundle_root
            / "references"
            / f"d4_consensus_slow_phase_a_v2_{panel}.json"
            for panel in PANELS
        }
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
        tests = run_tests(bundle_root, environment)
        freshness = freshness_scan(bundle_root, environment)
        wrapper.update(
            {
                "status": "running",
                "hardware": probe,
                "bundle_root": str(bundle_root),
                "data_root": str(data_root),
                "code_hashes": code_hashes,
                "correctness_tests": tests,
                "freshness": freshness,
                "assets": {
                    "selected_denoiser": {"path": str(selected), "sha256": sha256(selected)},
                    "seam_denoiser": {"path": str(seam), "sha256": sha256(seam)},
                    "embedding": {"path": str(embedding), "sha256": sha256(embedding)},
                },
            }
        )
        write_json(WRAPPER, wrapper)
        authorization = WORKING / "d4_global_phase_b_authorization.json"
        if authorization.exists():
            authorization.unlink()
        if FINAL_REPORT.exists():
            FINAL_REPORT.unlink()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_phase_a,
                    panel,
                    gpu,
                    evaluator=evaluator,
                    bundle_root=bundle_root,
                    data_root=data_root,
                    selected=selected,
                    seam=seam,
                    embedding=embedding,
                    slow_reference=slow_references[panel],
                    base_environment=environment,
                )
                for gpu, panel in enumerate(PANELS)
            ]
            phase_a_records = [future.result() for future in futures]
        phase_a_reports = {
            record["panel"]: json.loads(Path(record["output"]).read_text())
            for record in phase_a_records
        }
        changed = sum(
            phase_a_reports[panel]["phase_a"]["different_layouts"] for panel in PANELS
        )
        phase_a_checks = {
            "both_phase_a_pass": all(
                phase_a_reports[p]["phase_a"]["passed"] for p in PANELS
            ),
            "different_layouts_ge_4_of_16": changed >= 4,
            "authoritative_score_checks_eq_32": sum(
                phase_a_reports[p]["phase_a"][
                    "authoritative_score_checks_passed"
                ]
                for p in PANELS
            )
            == 32,
            "all_phase_a_artifacts_hash_sealed": all(
                record["artifact_sha256"]
                == phase_a_reports[record["panel"]]["artifact"]["sha256"]
                for record in phase_a_records
            ),
            "no_target_metrics_opened": all(
                phase_a_reports[p]["phase_a"]["target_metrics_opened"] is False
                for p in PANELS
            ),
        }
        phase_b_records = []
        reports = {}
        macro_ssim_delta = None
        if all(phase_a_checks.values()):
            auth_payload = {
                "schema_version": 1,
                "kind": "d4_global_phase_b_authorization",
                "authorized": True,
                "created_unix": time.time(),
                "source_names_sha256": EXPECTED_SOURCE_NAMES_SHA256,
                "checks": phase_a_checks,
                "panels": {
                    record["panel"]: {
                        "phase_a_report_sha256": record["report_sha256"],
                        "phase_a_artifact_sha256": record["artifact_sha256"],
                    }
                    for record in phase_a_records
                },
            }
            atomic_write_json(authorization, auth_payload)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        run_phase_b,
                        record,
                        gpu,
                        authorization=authorization,
                        evaluator=evaluator,
                        bundle_root=bundle_root,
                        data_root=data_root,
                        selected=selected,
                        seam=seam,
                        embedding=embedding,
                        slow_reference=slow_references[record["panel"]],
                        base_environment=environment,
                    )
                    for gpu, record in enumerate(phase_a_records)
                ]
                phase_b_records = [future.result() for future in futures]
            reports = {
                record["panel"]: json.loads(Path(record["output"]).read_text())
                for record in phase_b_records
            }
            all_records = [
                row for panel in PANELS for row in reports[panel].get("records", [])
            ]
            macro_ssim_delta = (
                sum(row["delta"]["harmonized_ssim"] for row in all_records)
                / len(all_records)
                if all_records
                else None
            )
        else:
            all_records = []
        scientific_checks = {
            **phase_a_checks,
            "global_authorization_written_before_phase_b": (
                authorization.is_file() and bool(phase_b_records)
            ),
            "all_16_targets_attached_after_authorization": len(all_records) == 16,
            "both_panel_gates_pass": bool(reports)
            and all(reports[p]["gate"]["passed"] for p in PANELS),
            "macro_harmonized_ssim_delta_ge_0.003": (
                macro_ssim_delta is not None and macro_ssim_delta >= 0.003
            ),
        }
        passed = all(scientific_checks.values())
        final = {
            "schema_version": 1,
            "kind": "d4_compatibility_consensus_two_panel_gate",
            "safe_for_submission": False,
            "protocol": {
                "panels": list(PANELS),
                "source_names_sha256": EXPECTED_SOURCE_NAMES_SHA256,
                "parameter_sweeps": 0,
                "formula": "0.50*identity_row_rank + 0.40*median4_row_rank + 0.10*MAD4",
                "real_or_test_inputs_opened": False,
            },
            "phase_a_runs": phase_a_records,
            "phase_a_checks": phase_a_checks,
            "phase_b_authorization": (
                {"path": str(authorization), "sha256": sha256(authorization)}
                if authorization.is_file()
                else None
            ),
            "phase_b_runs": phase_b_records,
            "panel_summaries": (
                {panel: reports[panel]["summary"] for panel in PANELS}
                if reports
                else None
            ),
            "phase_a_different_layouts": changed,
            "macro_harmonized_ssim_delta": macro_ssim_delta,
            "gate": {
                "passed": passed,
                "checks": scientific_checks,
                "decision": (
                    "promote_to_fresh_real16_confirmation"
                    if passed
                    else "retire_d4_consensus"
                ),
            },
            "seconds": time.time() - started,
        }
        write_json(FINAL_REPORT, final)
        wrapper.update(
            {
                "status": "complete",
                "scientific_status": final["gate"]["decision"],
                "final_report": {"path": str(FINAL_REPORT), "sha256": sha256(FINAL_REPORT)},
                "phase_a_runs": phase_a_records,
                "phase_b_runs": phase_b_records,
                "seconds": time.time() - started,
            }
        )
        write_json(WRAPPER, wrapper)
        print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        wrapper.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "seconds": time.time() - started,
            }
        )
        write_json(WRAPPER, wrapper)
        raise


if __name__ == "__main__":
    main()
