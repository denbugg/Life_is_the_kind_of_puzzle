#!/usr/bin/env python3
"""Hash-pinned Kaggle launcher for the target-free masked-gap DDP v2 benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import zipfile


INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
SELECTION_REPORT = WORKING / "masked_gap_t4_ddp_selection_v2.json"
WRAPPER_REPORT = WORKING / "masked_gap_t4_ddp_benchmark_wrapper_v2.json"
EXPECTED_BENCHMARK_SHA256 = (
    "2ee1f73992df440c90949e71edca9aa5e5a7289b5811486852a25ab79def07c5"
)
EXPECTED_BUNDLE_SHA256 = (
    "5237d7f033122248029c4f01277af17022306a4a2ab3b33d35241b32b060e1f4"
)
EXPECTED_CONTRACT_SHA256 = (
    "3b396bb6fd8a2945e6cd43fc82a9b36a92d4eba7fe3de89babed88b416bc2be6"
)
EXPECTED_KIND = "masked_gap_t4x2_amp_ddp_capacity_selection_v2"
EXPECTED_CAPACITIES = [
    {"width": 64, "generator_blocks": 6, "ranker_blocks": 5},
    {"width": 48, "generator_blocks": 4, "ranker_blocks": 4},
    {"width": 32, "generator_blocks": 3, "ranker_blocks": 3},
    {"width": 24, "generator_blocks": 2, "ranker_blocks": 2},
    {"width": 16, "generator_blocks": 2, "ranker_blocks": 2},
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def extract_benchmark() -> tuple[Path, Path]:
    matches = sorted(
        {
            path.resolve()
            for path in INPUT_ROOT.rglob("masked_gap_benchmark_code.bin")
            if path.is_file() and sha256(path) == EXPECTED_BUNDLE_SHA256
        }
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one hash-pinned benchmark code bundle, got {matches}"
        )
    archive = matches[0]
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if [info.filename for info in infos] != ["benchmark_masked_gap_t4.py"]:
            raise RuntimeError("benchmark bundle member contract mismatch")
        info = infos[0]
        if info.is_dir() or info.file_size <= 0 or info.file_size > 1_000_000:
            raise RuntimeError("benchmark bundle member size/type mismatch")
        payload = bundle.read(info)
    if hashlib.sha256(payload).hexdigest() != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("benchmark source hash mismatch inside code bundle")
    destination = WORKING / "hash_pinned_benchmark" / "benchmark_masked_gap_t4.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    if sha256(destination) != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("roundtrip-extracted benchmark source hash mismatch")
    return destination, archive


def validate_selection(report: dict) -> None:
    required_false = (
        "safe_for_submission",
        "launches_scientific_training",
        "scientific_images_labels_targets_opened",
    )
    if report.get("kind") != EXPECTED_KIND or report.get("status") != "complete":
        raise RuntimeError("selection report kind/status mismatch")
    if any(report.get(key) is not False for key in required_false):
        raise RuntimeError("selection report fail-closed flags drift")
    if report.get("synthetic_only") is not True:
        raise RuntimeError("selection report is not synthetic-only")
    if report.get("synthetic_optimizer_steps") is not True:
        raise RuntimeError("synthetic optimizer-step disclosure missing")
    if report.get("weights_discarded") is not True:
        raise RuntimeError("discarded-weight disclosure missing")
    if report.get("benchmark_source_sha256") != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("selection report source hash mismatch")
    contract = report.get("contract", {})
    if report.get("contract_sha256") != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("selection report contract hash mismatch")
    if canonical_json_sha256(contract) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("selection report contract payload mismatch")
    if contract.get("capacities_largest_first") != EXPECTED_CAPACITIES:
        raise RuntimeError("precommitted capacity list/order drift")
    if contract.get("selection", {}).get("safety_factor") != 1.35:
        raise RuntimeError("safety factor drift")
    if contract.get("selection", {}).get("max_projected_hours") != 5.5:
        raise RuntimeError("time threshold drift")
    if contract.get("selection", {}).get("max_peak_bytes_per_gpu") != 13_500_000_000:
        raise RuntimeError("memory threshold drift")
    workload = contract.get("workload", {})
    if workload.get("development_dense_pairs_per_model_per_pass") != 5_299_200:
        raise RuntimeError("development dense workload drift")
    if workload.get("checkpoint_selection_dense_pairs_per_model_two_epochs") != 10_598_400:
        raise RuntimeError("checkpoint-selection dense workload drift")
    if workload.get("calibration_b_dense_pairs_per_model") != 5_299_200:
        raise RuntimeError("calibration-B dense workload drift")
    if workload.get("final_dense_pairs_per_model") != 10_598_400:
        raise RuntimeError("final dense workload drift")
    if workload.get("all_dense_pairs_per_model") != 26_496_000:
        raise RuntimeError("full dense workload drift")
    if workload.get("all_source_panel_preparations_tilenaf") != 808:
        raise RuntimeError("TileNAF source-preparation workload drift")
    if workload.get("all_source_panel_preparations_w4") != 424:
        raise RuntimeError("w4 source-preparation workload drift")
    if contract.get("selection", {}).get(
        "fixed_source_preparation_reserve_seconds_before_safety"
    ) != 3600:
        raise RuntimeError("fixed source-preparation reserve drift")
    timing = contract.get("timing", {})
    if timing.get("final_decision") != "largest DDP-feasible capacity in precommitted order":
        raise RuntimeError("DDP-only final-decision drift")
    if timing.get("data_parallel_route") != "not executed by protocol v2":
        raise RuntimeError("retired DataParallel route drift")
    if "data_parallel_confirmation" in report:
        raise RuntimeError("v2 report unexpectedly contains legacy DataParallel confirmation")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(EXPECTED_CAPACITIES):
        raise RuntimeError("candidate result count mismatch")
    if [value.get("capacity") for value in candidates] != EXPECTED_CAPACITIES:
        raise RuntimeError("candidate capacity list/order drift")
    feasible = [value for value in candidates if value.get("feasible") is True]
    if not feasible:
        raise RuntimeError("benchmark returned no feasible capacity")
    selected = report.get("selected_capacity", {})
    if selected.get("capacity_key") != feasible[0].get("capacity_key"):
        raise RuntimeError("largest-feasible selection mismatch")
    if selected.get("capacity") != feasible[0].get("capacity"):
        raise RuntimeError("selected capacity config mismatch")
    projected_seconds = float(selected.get("projected_seconds_with_1p35_safety"))
    projected_hours = float(selected.get("projected_hours_with_1p35_safety"))
    if not math.isfinite(projected_seconds) or not math.isfinite(projected_hours):
        raise RuntimeError("selected DDP time projection is non-finite")
    if projected_seconds != float(feasible[0].get("projected_seconds_with_1p35_safety")):
        raise RuntimeError("selected seconds do not match first feasible DDP candidate")
    if not math.isclose(projected_seconds / 3600.0, projected_hours, rel_tol=1e-12):
        raise RuntimeError("selected projected seconds/hours mismatch")
    if projected_hours > 5.5:
        raise RuntimeError("selected capacity exceeds time threshold")
    if int(selected.get("max_peak_reserved_bytes")) > 13_500_000_000:
        raise RuntimeError("selected capacity exceeds memory threshold")
    if selected.get("execution_route") != "DDP_T4x2_AMP_v2":
        raise RuntimeError("selected execution route drift")
    if projected_hours != float(feasible[0].get("projected_hours_with_1p35_safety")):
        raise RuntimeError("selected time does not match first feasible DDP candidate")
    if int(selected["max_peak_reserved_bytes"]) != int(feasible[0].get("max_peak_reserved_bytes")):
        raise RuntimeError("selected peak does not match first feasible DDP candidate")
    for candidate in candidates:
        if candidate.get("status") == "oom":
            if candidate.get("feasible") is not False:
                raise RuntimeError("OOM candidate was not rejected")
            continue
        if candidate.get("status") != "complete":
            raise RuntimeError("unexpected candidate status")
        if candidate.get("throughput_aggregation") != "2*minimum_per_rank_rate":
            raise RuntimeError("optimistic two-rank throughput aggregation")
        if candidate.get("ddp_all_reduce_cost_measured_in_training_rates") is not True:
            raise RuntimeError("DDP all-reduce was not measured")
        if candidate.get("ddp_buckets_in_peak_memory") is not True:
            raise RuntimeError("DDP buckets missing from peak memory")
        if candidate.get("isolated_fresh_process_pair") is not True:
            raise RuntimeError("capacity did not run in an isolated process pair")
        if candidate.get("allocator_cleared_before_capacity") is not True:
            raise RuntimeError("capacity allocator-clean proof missing")

    devices = report.get("hardware", {}).get("devices")
    if not isinstance(devices, list) or len(devices) != 2:
        raise RuntimeError("exact two-GPU hardware evidence missing")
    for index, device in enumerate(devices):
        if device.get("index") != index:
            raise RuntimeError("GPU index order drift")
        if "T4" not in str(device.get("name", "")).upper():
            raise RuntimeError("non-T4 device in benchmark report")
        if device.get("capability") != [7, 5]:
            raise RuntimeError("non-sm75 device in benchmark report")
        if not math.isfinite(float(device.get("actual_tensor_op"))):
            raise RuntimeError("real tensor operation evidence is non-finite")


def run() -> dict:
    benchmark, bundle = extract_benchmark()
    if SELECTION_REPORT.exists():
        SELECTION_REPORT.unlink()
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    command = [sys.executable, str(benchmark), "--output", str(SELECTION_REPORT)]
    completed = subprocess.run(
        command,
        cwd=WORKING,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    execution = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-20_000:],
        "stderr_tail": completed.stderr[-20_000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark process failed: {execution}")
    if not SELECTION_REPORT.is_file():
        raise RuntimeError("benchmark succeeded without its atomic selection report")
    report = json.loads(SELECTION_REPORT.read_text(encoding="utf-8"))
    validate_selection(report)
    wrapper = {
        "kind": "masked_gap_t4x2_ddp_benchmark_wrapper_v2",
        "status": "complete",
        "safe_for_submission": False,
        "launches_scientific_training": False,
        "synthetic_optimizer_steps": True,
        "weights_discarded": True,
        "synthetic_only": True,
        "scientific_images_labels_targets_opened": False,
        "code_bundle": str(bundle),
        "code_bundle_sha256": sha256(bundle),
        "benchmark_source": str(benchmark),
        "benchmark_source_sha256": sha256(benchmark),
        "selection_report": SELECTION_REPORT.name,
        "selection_report_sha256": sha256(SELECTION_REPORT),
        "selected_capacity": report["selected_capacity"],
        "execution": execution,
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    atomic_json(WRAPPER_REPORT, wrapper)
    return wrapper


def main() -> None:
    try:
        wrapper = run()
    except Exception as error:
        atomic_json(
            WRAPPER_REPORT,
            {
                "kind": "masked_gap_t4x2_ddp_benchmark_wrapper_v2",
                "status": "failed",
                "safe_for_submission": False,
                "launches_scientific_training": False,
                "synthetic_optimizer_steps": True,
                "weights_discarded": True,
                "synthetic_only": True,
                "scientific_images_labels_targets_opened": False,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(wrapper, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
