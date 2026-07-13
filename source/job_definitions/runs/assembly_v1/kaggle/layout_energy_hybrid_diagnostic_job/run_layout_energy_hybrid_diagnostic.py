#!/usr/bin/env python3
"""Fail-closed runner for the inference-only frozen-critic real16 diagnostic."""

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


INPUT = Path("/kaggle/input/datasets/pasha883")
WORKING = Path("/kaggle/working")
PUZZLE_INPUT = INPUT / "vsos-ai-initiative-pazzle"
OVERLAY_INPUT = INPUT / "vsos-layout-energy-hybrid-diagnostic-code"
WRAPPER_PATH = WORKING / "layout_energy_hybrid_diagnostic_wrapper.json"
OUTPUT_DIR = WORKING / "layout_energy_hybrid_diagnostic"
EXPECTED_ARCHIVE_SHA256 = (
    "346cf65e58a7971a54abaf7dcb1d1bf202d9b40afe70d6c5826ac7cac5679771"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "039cd7638731006665a62064f658211fd288d8cdcae6df79347a2f038f5cb717"
)
EXPECTED_LAYOUT_MANIFEST_SHA256 = (
    "d5eb0f71668be726cea84f6b8a2c9e6ea42c551fe72dbaff60d90aa13d6f4b00"
)
EXPECTED_CODE_SHA256 = {
    "scripts/evaluate_layout_energy_hybrid.py": "7311a05790c63234bd7b4cc76b8227bf9f7055189463efe8ce0ff6d3181fb020",
    "scripts/export_frozen_real16_layouts.py": "b3ed2de08e23ccca4ed609f526849dc3f020e3ba381860d722734a6a5176b4fa",
    "src/puzzle_assembly/__init__.py": "09e051b7555471aafca03cd666d789f033aca47f1c82f6e2af9c0cce50afe9d5",
    "src/puzzle_assembly/layout_energy_transformer.py": "ebcaab5ecd77dc54e7a7c1f9bf7c282931b4ccbddda414b28e9f07872aa7e6e1",
    "src/puzzle_assembly/layout_energy_hybrid.py": "05e476e0d9c5bcbcaef636f1f6c530f38842a6b8b4799da29cf9eec6832eb3bd",
    "src/puzzle_assembly/geometry.py": "1e16bec6fb98a33060558d5d28062334d9114b12424733ef103a40393ef1ba86",
    "src/puzzle_assembly/metrics.py": "84857ef92c382cc0964c21bfec67c13308014a1674aebf8686b17514784dae69",
    "src/puzzle_assembly/compatibility.py": "aff2149b161c4fded4e5d91fbea49a8a62967886148d3ad374467331e0416a9f",
    "src/puzzle_assembly/components.py": "53fcc7c4fd23956db884ee45060e47f8e94a931c16e497e426d67549621bd367",
    "src/puzzle_assembly/solvers.py": "23f9e32200748349d0da8558b7b44053a758e1c1eb306d8f31ce59feae03fe8e",
    "src/puzzle_denoise_v2/__init__.py": "30849e0f937ba4a50e85ce2eee0d2b930db06fbcc0b7dff84547e121ef2f30b7",
    "src/puzzle_denoise_v2/tiles.py": "21270e283e50ea0b155ef194de889222fb0c4f6954437eb1526342c006eefaa7",
    "src/puzzle_denoise_v2/degradation.py": "7e314081c143a1c7846a9777eaea8716092a85595f856769efd3704a2c583a75",
    "src/puzzle_denoise_v2/model.py": "37db32fb83ece0f122757bdbec19ffc6a17c5e5e00ef92a26328247d95c55d11",
    "tests/test_layout_energy_hybrid.py": "bfebd11f2ca929bae20c3a1a04194992927a75d4a7aa575867b2384498b50d60",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(paths: list[Path], label: str) -> Path:
    values = sorted({path.resolve() for path in paths})
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {values}")
    return values[0]


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_extract(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise RuntimeError("unsafe archive member")
        handle.extractall(destination)
    return destination


def require_exact_direct_tree(root: Path) -> None:
    expected = set(EXPECTED_CODE_SHA256)
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in direct code mount: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise RuntimeError(
            "direct code tree mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def find_data_root() -> Path:
    if not PUZZLE_INPUT.is_dir():
        raise FileNotFoundError(f"missing exact puzzle mount: {PUZZLE_INPUT}")
    candidates = [PUZZLE_INPUT] + [
        path for path in PUZZLE_INPUT.iterdir() if path.is_dir()
    ]
    return one(
        [
            candidate
            for candidate in candidates
            if (candidate / "train" / "inputs").is_dir()
            and len(list((candidate / "train" / "inputs").glob("*.png"))) == 7000
        ],
        "puzzle data root",
    )


def hardware_probe() -> dict[str, object]:
    import torch

    subprocess.run(["nvidia-smi"], check=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is unavailable")
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "T4" not in name.upper() or capability != (7, 5):
        raise RuntimeError(f"diagnostic requires a T4 sm_75, found {name} {capability}")
    left = torch.randn(512, 512, device="cuda:0", dtype=torch.float16)
    right = torch.randn(512, 512, device="cuda:0", dtype=torch.float16)
    product = left @ right
    if product.dtype != torch.float16 or not torch.isfinite(product).all():
        raise RuntimeError("real fp16 T4 matmul failed")
    torch.cuda.synchronize()
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_0": name,
        "capability_0": list(capability),
        "fp16_matmul_mean": float(product.float().mean().item()),
        "peak_allocated": int(torch.cuda.max_memory_allocated(0)),
    }


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    label: str,
    telemetry: list[dict[str, object]],
) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
    record = {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
    }
    telemetry.append(record)
    atomic_json(WRAPPER_PATH, {"status": "running", "telemetry": telemetry})
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def validate_report(report_path: Path, predictions_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("kind") != "frozen_failed_layout_energy_hybrid_real16_diagnostic":
        raise RuntimeError("wrong diagnostic report kind")
    if report.get("status") not in {"actionable_signal", "no_actionable_signal"}:
        raise RuntimeError("unexpected diagnostic status")
    if report.get("safe_for_submission") is not False:
        raise RuntimeError("diagnostic report became submission-safe")
    if report.get("checkpoint", {}).get("real16_source_overlap_count") != 0:
        raise RuntimeError("real16 overlaps critic training/selection/holdout sources")
    anti = report.get("anti_leakage", {})
    required_anti = {
        "predictor_accepts_target": False,
        "predictions_atomically_frozen_before_target_access": True,
        "prediction_artifact_unchanged": True,
        "target_used_only_for_posthoc_metrics": True,
    }
    for key, expected in required_anti.items():
        if anti.get(key) is not expected:
            raise RuntimeError(f"anti-leakage assertion failed: {key}")
    if sha256(predictions_path) != report.get("prediction_artifact", {}).get("sha256"):
        raise RuntimeError("prediction artifact hash mismatch")
    if len(report.get("per_source", [])) != 16:
        raise RuntimeError("report does not contain fixed real16")
    aggregates = report.get("aggregates", [])
    if len(aggregates) != 24:
        raise RuntimeError(f"expected 24 base/K/method aggregates, found {len(aggregates)}")
    no_ops = [row for row in aggregates if row.get("method") == "no_op_budget_matched"]
    if len(no_ops) != 6:
        raise RuntimeError("expected six no-op aggregates")
    for row in no_ops:
        delta = row.get("ssim_delta_vs_base", {})
        if any(abs(float(delta.get(key, 1.0))) > 1e-12 for key in ("mean", "lower_95", "upper_95")):
            raise RuntimeError("budget-matched no-op changed SSIM")
    if len(report.get("actionability_gates", [])) != 12:
        raise RuntimeError("expected 12 learned actionability gate records")
    if (WORKING / "submission.zip").exists() or (OUTPUT_DIR / "submission.zip").exists():
        raise RuntimeError("diagnostic must not create a submission")
    return {
        "status": report["status"],
        "actionable_signal": report["actionable_signal"],
        "report_sha256": sha256(report_path),
        "predictions_sha256": sha256(predictions_path),
    }


def main() -> None:
    started = time.perf_counter()
    telemetry: list[dict[str, object]] = []
    wrapper: dict[str, object] = {
        "kind": "layout_energy_hybrid_diagnostic_wrapper",
        "safe_for_submission": False,
        "status": "running",
        "telemetry": telemetry,
    }
    atomic_json(WRAPPER_PATH, wrapper)
    try:
        if not OVERLAY_INPUT.is_dir():
            raise FileNotFoundError(f"missing exact overlay mount: {OVERLAY_INPUT}")
        archives = list(OVERLAY_INPUT.glob("**/layout_energy_hybrid_code.zip"))
        checkpoint = one(list(OVERLAY_INPUT.glob("**/layout_energy_checkpoint.pt")), "checkpoint")
        layouts = one(
            list(OVERLAY_INPUT.glob("**/frozen_real16_hbt_qap_layouts.json")),
            "frozen layout manifest",
        )
        actual_assets = {
            "checkpoint": sha256(checkpoint),
            "layouts": sha256(layouts),
        }
        expected_assets = {
            "checkpoint": EXPECTED_CHECKPOINT_SHA256,
            "layouts": EXPECTED_LAYOUT_MANIFEST_SHA256,
        }
        if actual_assets != expected_assets:
            raise RuntimeError(
                f"diagnostic asset hash mismatch: expected {expected_assets}, got {actual_assets}"
            )
        if archives:
            archive = one(archives, "code archive")
            archive_sha256 = sha256(archive)
            if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
                raise RuntimeError(
                    "diagnostic code archive hash mismatch: "
                    f"expected {EXPECTED_ARCHIVE_SHA256}, got {archive_sha256}"
                )
            root = safe_extract(archive, WORKING / "layout_energy_hybrid_code")
            actual_assets["archive"] = archive_sha256
            actual_assets["code_mount_mode"] = "pinned_zip"
        else:
            root = one(
                [
                    path.parent.parent
                    for path in OVERLAY_INPUT.glob(
                        "**/scripts/evaluate_layout_energy_hybrid.py"
                    )
                    if (
                        path.parent.parent
                        / "src"
                        / "puzzle_assembly"
                        / "layout_energy_hybrid.py"
                    ).is_file()
                ],
                "Kaggle-expanded direct code root",
            )
            actual_assets["archive"] = None
            actual_assets["expected_source_archive_sha256"] = EXPECTED_ARCHIVE_SHA256
            actual_assets["code_mount_mode"] = "kaggle_expanded_direct_files"
            require_exact_direct_tree(root)
        actual_code = {relative: sha256(root / relative) for relative in EXPECTED_CODE_SHA256}
        if actual_code != EXPECTED_CODE_SHA256:
            raise RuntimeError("exact code hash pin failed")
        data_root = find_data_root()
        hardware = hardware_probe()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        environment["PYTHONHASHSEED"] = "20260711"
        run_checked(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_layout_energy_hybrid.py",
                "-k",
                "not target_scoring_reads_an_immutable_prediction_artifact",
            ],
            cwd=root,
            environment=environment,
            label="seven_pre_prediction_contract_tests_without_target_paths",
            telemetry=telemetry,
        )
        run_checked(
            [
                sys.executable,
                "scripts/evaluate_layout_energy_hybrid.py",
                "--data-root",
                str(data_root),
                "--checkpoint",
                str(checkpoint),
                "--frozen-layouts",
                str(layouts),
                "--output-dir",
                str(OUTPUT_DIR),
                "--expected-checkpoint-sha256",
                EXPECTED_CHECKPOINT_SHA256,
                "--expected-layout-manifest-sha256",
                EXPECTED_LAYOUT_MANIFEST_SHA256,
                "--device",
                "cuda:0",
                "--require-t4",
                "--suspect-k",
                "8,16,32",
                "--proposal-budget",
                "96",
                "--rerank-budget",
                "4",
                "--score-batch-size",
                "5",
                "--bootstrap-samples",
                "5000",
                "--limit",
                "16",
            ],
            cwd=root,
            environment=environment,
            label="frozen_real16_inference_only_diagnostic",
            telemetry=telemetry,
        )
        run_checked(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_layout_energy_hybrid.py::test_target_scoring_reads_an_immutable_prediction_artifact",
            ],
            cwd=root,
            environment=environment,
            label="post_prediction_target_scoring_contract_test",
            telemetry=telemetry,
        )
        report_path = OUTPUT_DIR / "layout_energy_hybrid_report.json"
        predictions_path = OUTPUT_DIR / "layout_energy_hybrid_predictions_frozen.json"
        validation = validate_report(report_path, predictions_path)
        wrapper.update(
            {
                "status": "complete",
                "safe_for_submission": False,
                "elapsed_seconds": time.perf_counter() - started,
                "assets": actual_assets,
                "code_sha256": actual_code,
                "hardware": hardware,
                "data_root": str(data_root),
                "validation": validation,
                "telemetry": telemetry,
            }
        )
    except BaseException as error:
        wrapper.update(
            {
                "status": "failed",
                "safe_for_submission": False,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "telemetry": telemetry,
            }
        )
        atomic_json(WRAPPER_PATH, wrapper)
        raise
    atomic_json(WRAPPER_PATH, wrapper)
    print(json.dumps(wrapper, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
