"""Executable, process-separated runner for frozen E24 CRS-v1.

No target phase is implicit.  ``preflight`` is the only mode allowed without
an already frozen ledger; every other mode authenticates that ledger before
loading a target scene or creating a target artifact.  ``orchestrate`` invokes
each capability as a fresh Python process in this order:

1. a trusted upstream replay worker commits only detached corrupted tile bytes,
2. an authenticated two-member raw/spatial broker commits label-free inputs,
3. eight label-free feature workers,
4. one six-scene label broker and separate trainer/predictor workers per fold,
5. a global four-fold commit barrier, then label-only structural evaluation.

The staged board/SSIM/NLM gate is intentionally absent and remains sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


STORAGE_ROOT = Path("E:/pazzle_work/posegraph_e24_selector")
_RUNTIME_DIR = STORAGE_ROOT / "tmp"
_PYCACHE_DIR = STORAGE_ROOT / "pycache"
sys.pycache_prefix = str(_PYCACHE_DIR)
for _key in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "LIGHTGBM_TMPDIR"):
    os.environ[_key] = str(_RUNTIME_DIR)

import numpy as np

import e24_context_relation_selector as selector
import e23_i21_residual_candidate_oracle as e23_core
import eval_e24_context_relation_selector as evaluator


class E24RunnerError(RuntimeError):
    """The E24 executable pipeline, provenance, or capability boundary failed."""


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOCUMENT = ROOT / "E24_CONTEXT_RELATION_SELECTOR.md"
PLAN_DOCUMENT = ROOT / "autoresearch-runs/pazzle-solution-20260806/PLAN.md"
BUDGET_DOCUMENT = ROOT / "autoresearch-runs/pazzle-solution-20260806/BUDGET.md"

RUNNER_SCHEMA = "pazzle-e24-crs-v1-runner-preflight-v1"
UPSTREAM_PROJECTION_SCHEMA = "pazzle-e24-crs-v1-label-free-source-projection-v1"
TILE_LINEAGE_SCHEMA = "pazzle-e24-crs-v1-detached-tile-lineage-v1"
INPUT_SCHEMA = "pazzle-e24-crs-v1-label-free-input-bundle-v2"
FEATURE_MANIFEST_SCHEMA = "pazzle-e24-crs-v1-feature-artifact-v1"
LABEL_MANIFEST_SCHEMA = "pazzle-e24-crs-v1-fold-train-label-v1"
MODEL_MANIFEST_SCHEMA = "pazzle-e24-crs-v1-fold-model-v1"
STRUCTURAL_REPORT_SCHEMA = "pazzle-e24-crs-v1-structural-oof-report-v1"
FEATURE_RECEIPT_SCHEMA = "pazzle-e24-crs-v1-feature-worker-receipt-v1"
ORCHESTRATION_RECEIPT_SCHEMA = "pazzle-e24-crs-v1-orchestration-receipt-v1"

CANARY_IMAGE = 17
CANARY_EXPECTED_HYPOTHESES = 333_080
CANARY_WALL_SECONDS_MAX = 30 * 60
CANARY_PEAK_RSS_BYTES_MAX = 4 * 1024**3
CANARY_FEATURE_BYTES_MAX = 480 * 1024**2
CANARY_FOLD_OUTPUT_RESERVE_BYTES = 512 * 1024**2

EXPECTED_E23_REPORT_SHA256 = (
    "9043a52fd746558d4a9a4eb047b83724abf225d3c00d71e1413e6e8e58698c20"
)
DEFAULT_E23_REPORT = Path(
    "E:/pazzle_work/posegraph_e23/cc96_i21_residual_k64_candidate_ceiling_v1.json"
)
DEFAULT_RAW_CACHE_ROOT = Path("E:/pazzle_work/edge_confidence/full_graph_cache")
RAW_CACHE_TAG = "k64"
DEFAULT_LEDGER = STORAGE_ROOT / "preflight" / "e24_crs_v1_preflight.json"
INPUT_ROOT = STORAGE_ROOT / "label_free_inputs_v1"
FEATURE_ROOT = STORAGE_ROOT / "feature_cache_v1"
FOLD_ROOT = STORAGE_ROOT / "folds_v1"
STRUCTURAL_REPORT = STORAGE_ROOT / "contextual_relation_selector_oof_v1.json"
CANARY_GATE_PATH = STORAGE_ROOT / "canary" / "scene_0017_gate.json"
ORCHESTRATION_RECEIPT_PATH = STORAGE_ROOT / "oof_orchestration_receipt.json"

SOURCE_FILES = (
    ROOT / "src/e24_context_relation_selector.py",
    ROOT / "src/eval_e24_context_relation_selector.py",
    ROOT / "src/run_e24_context_relation_selector.py",
    ROOT / "src/e23_i21_residual_candidate_oracle.py",
    ROOT / "src/eval_e23_i21_residual_candidate_ceiling.py",
    ROOT / "src/eval_clean_score_oracle.py",
    ROOT / "src/eval_e14_cc192_discovery.py",
    ROOT / "src/eval_buddies_ssim_budget.py",
    ROOT / "src/e22_rcce4_candidate_oracle.py",
    ROOT / "src/e21_posegraph_candidate_oracle.py",
    ROOT / "src/rank96_lab_selector.py",
    ROOT / "src/solve_buddies.py",
    ROOT / "src/eval_seeded_qap.py",
    ROOT / "src/candidate_rank.py",
    ROOT / "src/canvas_data.py",
    ROOT / "src/distort.py",
    ROOT / "src/imgio.py",
    ROOT / "src/config.py",
    PROTOCOL_DOCUMENT,
    PLAN_DOCUMENT,
    BUDGET_DOCUMENT,
    ROOT / "tests/test_e24_context_relation_selector.py",
    ROOT / "tests/test_e24_context_relation_evaluator.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    """Return the process peak working set without creating an artifact."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = ()
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        success = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise E24RunnerError("Windows peak-working-set query failed")
        return int(counters.PeakWorkingSetSize)
    try:  # pragma: no cover - Windows is the frozen target runtime
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except Exception as exc:  # pragma: no cover
        raise E24RunnerError("peak resident-memory query failed") from exc


def _resource_snapshot(started_wall: float, started_cpu: float) -> dict[str, Any]:
    return {
        "wall_seconds": float(time.perf_counter() - started_wall),
        "process_cpu_seconds": float(time.process_time() - started_cpu),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _enforce_process_resource_caps(resource: Mapping[str, Any]) -> None:
    peak = int(resource["peak_rss_bytes"])
    cpu = float(resource["process_cpu_seconds"])
    if peak > evaluator.PEAK_RAM_BYTES_MAX:
        raise E24RunnerError("process exceeded the frozen 16 GiB peak-RAM cap")
    if cpu > evaluator.OOF_CPU_SECONDS_MAX:
        raise E24RunnerError("process exceeded the frozen 8 CPU-hour OOF cap")


def _require_image(image: int) -> int:
    if type(image) is not int or image not in evaluator.CALIBRATION_IDS:
        raise E24RunnerError("image must be one of the frozen E24 IDs 10..17")
    return image


def _artifact_files(paths: Sequence[Path]) -> set[Path]:
    files: set[Path] = set()
    for path in paths:
        resolved = _require_storage(path, label="artifact accounting path")
        if resolved.is_file():
            files.add(resolved)
        elif resolved.is_dir():
            for candidate in resolved.rglob("*"):
                if candidate.is_file():
                    files.add(_require_storage(candidate, label="accounted artifact"))
    return files


def enforce_aggregate_artifact_caps(
    *,
    ledger_path: Path,
    additional_feature_bytes: int = 0,
    additional_total_bytes: int = 0,
) -> dict[str, int]:
    """Enforce the frozen aggregate 4 GiB feature / 8 GiB run caps."""

    if (
        type(additional_feature_bytes) is not int
        or type(additional_total_bytes) is not int
        or additional_feature_bytes < 0
        or additional_total_bytes < 0
    ):
        raise E24RunnerError("additional artifact byte estimates must be nonnegative ints")
    feature_files = _artifact_files((FEATURE_ROOT,))
    # The all-artifact cap includes runtime temp, pycache, interrupted orphans,
    # test residues, and unknown files anywhere below the frozen storage root.
    all_files = _artifact_files((STORAGE_ROOT, ledger_path))
    feature_bytes = sum(path.stat().st_size for path in feature_files)
    total_bytes = sum(path.stat().st_size for path in all_files)
    if feature_bytes + additional_feature_bytes > evaluator.FEATURE_CACHE_BYTES_MAX:
        raise E24RunnerError("aggregate E24 feature cache exceeds the frozen 4 GiB cap")
    if total_bytes + additional_total_bytes > evaluator.ALL_ARTIFACT_BYTES_MAX:
        raise E24RunnerError("aggregate E24 artifacts exceed the frozen 8 GiB cap")
    return {"feature_bytes": feature_bytes, "total_bytes": total_bytes}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return evaluator._canonical_json_bytes(dict(value))


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_spatial_logits(value: object) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.shape != (4, 576, 576)
        or value.dtype != np.float32
        or not value.flags.c_contiguous
        or not bool(np.isfinite(value).all())
    ):
        raise E24RunnerError(
            "spatial_logits must be contiguous finite float32[4,576,576]"
        )
    return value


def _lightgbm_contract_sha256(ledger: Mapping[str, Any], fold: int) -> str:
    lightgbm = ledger.get("lightgbm")
    if not isinstance(lightgbm, Mapping):
        raise E24RunnerError("preflight LightGBM contract is malformed")
    payload = {
        "version": lightgbm.get("version"),
        "config": evaluator.frozen_lightgbm_config(fold),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _ledger_source_sha256(ledger: Mapping[str, Any], path: Path) -> str:
    sources = ledger.get("sources")
    key = str(path.resolve())
    if not isinstance(sources, Mapping) or key not in sources:
        raise E24RunnerError(f"preflight source hash is absent for {path.name}")
    try:
        return evaluator._validate_lower_hex_sha256(
            sources[key], label=f"source SHA for {path.name}"
        )
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError(str(exc)) from exc


def _require_storage(path: Path, *, label: str) -> Path:
    try:
        return evaluator._require_e24_storage_path(path, label=label)
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError(str(exc)) from exc


def _load_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise E24RunnerError(f"{label} is unreadable") from exc
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise E24RunnerError(f"{label} must be canonical JSON")
    return value


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in SOURCE_FILES:
        if not path.is_file():
            raise E24RunnerError(f"frozen source/protocol input is missing: {path}")
        output[str(path.resolve())] = _sha256_file(path.resolve())
    return output


def _build_preflight_base_payload() -> dict[str, Any]:
    """Build fields that target workers can re-authenticate without report parsing."""

    try:
        runtime_paths = evaluator.validate_e24_runtime_paths()
        lightgbm_version = evaluator.validate_lightgbm_runtime_version()
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError(str(exc)) from exc
    if not DEFAULT_E23_REPORT.is_file() or _sha256_file(DEFAULT_E23_REPORT) != EXPECTED_E23_REPORT_SHA256:
        raise E24RunnerError("frozen E23 report byte provenance is absent or changed")
    return {
        "schema": RUNNER_SCHEMA,
        "status": "frozen_preflight_only",
        "metrics_opened": False,
        "target_artifacts_created": False,
        "evaluator_protocol_sha256": evaluator.PROTOCOL_SHA256,
        # Normalize tuples and any other JSON-compatible containers now, not
        # only while writing. The verified in-memory contract must compare
        # equal to the canonical JSON object loaded in a fresh process.
        "core_protocol": json.loads(
            _canonical_json_bytes(selector.PROTOCOL).decode("ascii")
        ),
        "ordered_feature_names": list(selector.FEATURE_NAMES),
        "ordered_feature_names_sha256": hashlib.sha256(
            _canonical_json_bytes({"feature_names": list(selector.FEATURE_NAMES)})
        ).hexdigest(),
        "lightgbm": {
            "version": lightgbm_version,
            "config": dict(selector.LIGHTGBM_CONFIG),
            "fold_seeds": {
                str(fold): {
                    key: evaluator.frozen_lightgbm_config(fold)[key]
                    for key in (
                        "random_state",
                        "data_random_seed",
                        "feature_fraction_seed",
                    )
                }
                for fold in evaluator.OOF_FOLDS
            },
        },
        "folds": {str(key): list(value) for key, value in evaluator.OOF_FOLDS.items()},
        "structural_gates": dict(evaluator.STRUCTURAL_GATES),
        "end_to_end_gates_sealed": dict(evaluator.END_TO_END_GATES),
        "e25": {
            "ids": list(evaluator.E25_SEALED_IDS),
            "newline_list_sha256": evaluator.E25_NEWLINE_LIST_SHA256,
            "canonical_records_sha256": evaluator.E25_CANONICAL_RECORDS_SHA256,
            "opened": False,
        },
        "upstream": {
            "e23_report": str(DEFAULT_E23_REPORT.resolve()),
            "e23_report_sha256": EXPECTED_E23_REPORT_SHA256,
        },
        "runtime_paths": runtime_paths,
        "runtime_versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "scikit-image": importlib.metadata.version("scikit-image"),
            "scipy": importlib.metadata.version("scipy"),
            "opencv-python": importlib.metadata.version("opencv-python"),
            "Pillow": importlib.metadata.version("Pillow"),
            "lightgbm": lightgbm_version,
        },
        "sources": _source_hashes(),
        "storage_root": str(STORAGE_ROOT.resolve(strict=False)),
        "board_authorization": (
            "external append-only preflight event pins this ledger SHA; "
            "mutable board bytes are deliberately excluded from the run contract"
        ),
        "staged_board_ssim_nlm": "sealed",
    }


def _project_e23_label_free_inputs() -> dict[str, Any]:
    """Project only deployable source fields from the already-open frozen E23 run."""

    rows = _e23_report_rows()
    records: list[dict[str, Any]] = []
    for image in evaluator.CALIBRATION_IDS:
        row = rows[image]
        validation_name = row.get("validation_name")
        if type(validation_name) is not str or not validation_name:
            raise E24RunnerError("E23 validation-name provenance is malformed")
        raw_path = (
            DEFAULT_RAW_CACHE_ROOT / f"image_{image:04d}_{RAW_CACHE_TAG}.npz"
        ).resolve()
        raw_sha = evaluator._validate_lower_hex_sha256(
            row.get("raw_cache_sha256"), label="E23 raw-cache SHA"
        )
        if (
            not raw_path.is_file()
            or _sha256_file(raw_path) != raw_sha
        ):
            raise E24RunnerError("E23 raw-cache path/SHA provenance is absent")
        spatial = row.get("spatial_cache")
        if type(spatial) is not dict:
            raise E24RunnerError("E23 spatial-cache record is malformed")
        spatial_path = evaluator._require_e_drive(
            Path(str(spatial.get("array_path", ""))),
            label="E23 spatial-cache array",
        )
        spatial_metadata_path = evaluator._require_e_drive(
            Path(str(spatial.get("metadata_path", ""))),
            label="E23 spatial-cache metadata",
        )
        spatial_sha = evaluator._validate_lower_hex_sha256(
            spatial.get("array_file_sha256"), label="E23 spatial-cache file SHA"
        )
        spatial_array_sha = evaluator._validate_lower_hex_sha256(
            spatial.get("array_sha256"), label="E23 spatial-cache array SHA"
        )
        spatial_bytes = spatial.get("array_file_bytes")
        if (
            type(spatial_bytes) is not int
            or spatial_bytes <= 0
            or not spatial_path.is_file()
            or spatial_path.stat().st_size != spatial_bytes
            or _sha256_file(spatial_path) != spatial_sha
            or not spatial_metadata_path.is_file()
        ):
            raise E24RunnerError("E23 spatial-cache path/size/SHA provenance is absent")
        records.append(
            {
                "image": image,
                "validation_name": validation_name,
                "raw_cache": {
                    "path": str(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "file_sha256": raw_sha,
                },
                "candidate_ids_sha256": evaluator._validate_lower_hex_sha256(
                    row.get("candidate_ids_sha256"), label="E23 candidate-ID SHA"
                ),
                "raw_logits_sha256": evaluator._validate_lower_hex_sha256(
                    row.get("raw_logits_sha256"), label="E23 raw-logit SHA"
                ),
                "tiles_uint8_sha256": evaluator._validate_lower_hex_sha256(
                    row.get("tiles_uint8_sha256"), label="E23 tile-array SHA"
                ),
                "spatial_logits": {
                    "path": str(spatial_path),
                    "bytes": spatial_bytes,
                    "file_sha256": spatial_sha,
                    "array_sha256": spatial_array_sha,
                    "metadata_path": str(spatial_metadata_path),
                    "metadata_bytes": spatial_metadata_path.stat().st_size,
                    "metadata_file_sha256": _sha256_file(spatial_metadata_path),
                },
            }
        )
    projection = {
        "schema": UPSTREAM_PROJECTION_SCHEMA,
        "e23_report_sha256": EXPECTED_E23_REPORT_SHA256,
        "records": records,
    }
    projection["records_sha256"] = hashlib.sha256(
        _canonical_json_bytes({"records": records})
    ).hexdigest()
    return projection


def _validated_upstream_projection(
    ledger: Mapping[str, Any], image: int
) -> dict[str, Any]:
    """Validate a ledger-bound projection without opening the E23 report."""

    image = _require_image(image)
    upstream = ledger.get("upstream")
    if type(upstream) is not dict:
        raise E24RunnerError("preflight upstream record is malformed")
    projection = upstream.get("label_free_input_projection")
    if type(projection) is not dict or set(projection) != {
        "schema",
        "e23_report_sha256",
        "records",
        "records_sha256",
    }:
        raise E24RunnerError("preflight label-free projection is malformed")
    records = projection.get("records")
    if (
        projection["schema"] != UPSTREAM_PROJECTION_SCHEMA
        or projection["e23_report_sha256"] != EXPECTED_E23_REPORT_SHA256
        or type(records) is not list
        or hashlib.sha256(_canonical_json_bytes({"records": records})).hexdigest()
        != projection["records_sha256"]
    ):
        raise E24RunnerError("preflight label-free projection digest drifted")
    expected_record_keys = {
        "image",
        "validation_name",
        "raw_cache",
        "candidate_ids_sha256",
        "raw_logits_sha256",
        "tiles_uint8_sha256",
        "spatial_logits",
    }
    if [record.get("image") for record in records if type(record) is dict] != list(
        evaluator.CALIBRATION_IDS
    ):
        raise E24RunnerError("preflight label-free projection scene order drifted")
    for record in records:
        if type(record) is not dict or set(record) != expected_record_keys:
            raise E24RunnerError("preflight label-free source record is malformed")
        if type(record["validation_name"]) is not str or not record["validation_name"]:
            raise E24RunnerError("preflight validation-name record is malformed")
        raw = record["raw_cache"]
        spatial = record["spatial_logits"]
        if (
            type(raw) is not dict
            or set(raw) != {"path", "bytes", "file_sha256"}
            or type(spatial) is not dict
            or set(spatial)
            != {
                "path",
                "bytes",
                "file_sha256",
                "array_sha256",
                "metadata_path",
                "metadata_bytes",
                "metadata_file_sha256",
            }
            or type(raw["bytes"]) is not int
            or raw["bytes"] <= 0
            or type(spatial["bytes"]) is not int
            or spatial["bytes"] <= 0
            or type(spatial["metadata_bytes"]) is not int
            or spatial["metadata_bytes"] <= 0
        ):
            raise E24RunnerError("preflight label-free path record is malformed")
        expected_raw_path = (
            DEFAULT_RAW_CACHE_ROOT
            / f"image_{int(record['image']):04d}_{RAW_CACHE_TAG}.npz"
        ).resolve()
        if Path(str(raw["path"])).resolve() != expected_raw_path:
            raise E24RunnerError("preflight raw-cache path drifted")
        try:
            evaluator._require_e_drive(spatial["path"], label="projected spatial cache")
            evaluator._require_e_drive(
                spatial["metadata_path"], label="projected spatial-cache metadata"
            )
            for key in (
                "candidate_ids_sha256",
                "raw_logits_sha256",
                "tiles_uint8_sha256",
            ):
                evaluator._validate_lower_hex_sha256(record[key], label=key)
            evaluator._validate_lower_hex_sha256(raw["file_sha256"], label="raw file SHA")
            evaluator._validate_lower_hex_sha256(
                spatial["file_sha256"], label="spatial file SHA"
            )
            evaluator._validate_lower_hex_sha256(
                spatial["array_sha256"], label="spatial array SHA"
            )
            evaluator._validate_lower_hex_sha256(
                spatial["metadata_file_sha256"], label="spatial metadata SHA"
            )
        except evaluator.E24EvaluatorContractError as exc:
            raise E24RunnerError(str(exc)) from exc
    return dict(records[list(evaluator.CALIBRATION_IDS).index(image)])


def build_preflight_payload() -> dict[str, Any]:
    """Build the no-target run contract and a label-free E23 source projection."""

    payload = _build_preflight_base_payload()
    payload["upstream"]["label_free_input_projection"] = (
        _project_e23_label_free_inputs()
    )
    payload["run_contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def write_preflight_ledger(path: Path) -> dict[str, Any]:
    destination = _require_storage(path, label="preflight ledger")
    payload = build_preflight_payload()
    ledger_bytes = _canonical_json_bytes(payload)
    enforce_aggregate_artifact_caps(
        ledger_path=destination,
        additional_total_bytes=(0 if destination.exists() else len(ledger_bytes)),
    )
    evaluator._atomic_write_create_or_verify(destination, ledger_bytes)
    enforce_aggregate_artifact_caps(ledger_path=destination)
    return payload


def verify_preflight_ledger(path: Path, expected_sha256: str) -> dict[str, Any]:
    source = _require_storage(path, label="preflight ledger")
    expected = evaluator._validate_lower_hex_sha256(
        expected_sha256, label="preflight ledger SHA"
    )
    if not source.is_file() or _sha256_file(source) != expected:
        raise E24RunnerError("preflight ledger SHA mismatch/absence")
    payload = _load_canonical_json(source, label="preflight ledger")
    contract_sha = payload.get("run_contract_sha256")
    try:
        evaluator._validate_lower_hex_sha256(
            contract_sha, label="preflight run-contract SHA"
        )
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError(str(exc)) from exc
    contract_body = dict(payload)
    contract_body.pop("run_contract_sha256", None)
    if hashlib.sha256(_canonical_json_bytes(contract_body)).hexdigest() != contract_sha:
        raise E24RunnerError("preflight run-contract digest mismatch")
    for image in evaluator.CALIBRATION_IDS:
        _validated_upstream_projection(payload, image)
    comparable = dict(contract_body)
    stored_upstream = comparable.get("upstream")
    if type(stored_upstream) is not dict:
        raise E24RunnerError("preflight upstream record is malformed")
    stored_upstream = dict(stored_upstream)
    stored_upstream.pop("label_free_input_projection", None)
    comparable["upstream"] = stored_upstream
    current = _build_preflight_base_payload()
    if comparable != current:
        raise E24RunnerError("preflight ledger no longer matches frozen sources/runtime")
    return payload


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    destination = _require_storage(path, label="detached array")
    evaluator._atomic_write_create_or_verify(destination, evaluator._npy_bytes(value))


def _load_exact_npy(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    file_sha256: str,
) -> np.ndarray:
    source = _require_storage(path, label="detached array")
    if not source.is_file() or _sha256_file(source) != file_sha256:
        raise E24RunnerError("detached array file provenance mismatch")
    try:
        value = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise E24RunnerError("detached array is unreadable") from exc
    if (
        type(value) is not np.ndarray
        or value.shape != shape
        or value.dtype != dtype
        or not value.flags.c_contiguous
    ):
        raise E24RunnerError("detached array shape/dtype/order drifted")
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _e23_report_rows() -> dict[int, Mapping[str, Any]]:
    if _sha256_file(DEFAULT_E23_REPORT) != EXPECTED_E23_REPORT_SHA256:
        raise E24RunnerError("E23 report SHA mismatch")
    try:
        report = json.loads(DEFAULT_E23_REPORT.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E24RunnerError("E23 report is unreadable") from exc
    if (
        type(report) is not dict
        or report.get("status") != "complete"
        or report.get("completed_images") != list(evaluator.CALIBRATION_IDS)
        or type(report.get("rows")) is not list
    ):
        raise E24RunnerError("E23 report status/scene set drifted")
    rows = {int(row["image"]): row for row in report["rows"] if type(row) is dict}
    if set(rows) != set(evaluator.CALIBRATION_IDS):
        raise E24RunnerError("E23 report row identity drifted")
    return rows


def _replay_detached_tiles(image: int) -> tuple[str, np.ndarray]:
    """Trusted CanvasDataset lineage; return no sample, label, or report object."""

    import random

    import torch

    from canvas_data import CanvasDataset
    from config import SEED
    from imgio import train_val_split

    image = _require_image(image)
    replay_start, replay_count = 10, 12
    try:
        _train_names, validation_names = train_val_split()
        if replay_start + replay_count > len(validation_names):
            raise E24RunnerError("fixed tile replay exceeds the validation pool")
        random.seed(int(SEED))
        np.random.seed(int(SEED))
        torch.manual_seed(int(SEED))
        dataset = CanvasDataset(
            validation_names[replay_start : replay_start + replay_count],
            real_prob=0.0,
            seed=int(SEED) + 400_000,
        )
        selected: Mapping[str, Any] | None = None
        for local in range(image - replay_start + 1):
            sample = dataset[local]
            if replay_start + local == image:
                selected = sample
        if selected is None or not bool(selected["has_perm"]):
            raise E24RunnerError("tile replay did not return the requested synthetic bag")
        tile_tensor = selected["tiles"]
        tiles = np.rint(tile_tensor.permute(0, 2, 3, 1).numpy() * 255.0)
        tiles = np.ascontiguousarray(tiles.clip(0, 255), dtype=np.uint8)
    except E24RunnerError:
        raise
    except Exception as exc:
        raise E24RunnerError("frozen CanvasDataset tile replay failed") from exc
    return str(validation_names[image]), tiles


def _input_manifest_path(image: int) -> Path:
    return INPUT_ROOT / f"image_{_require_image(image):04d}" / "input_manifest.json"


def _tile_lineage_paths(image: int) -> tuple[Path, Path]:
    directory = INPUT_ROOT / f"image_{_require_image(image):04d}"
    return directory / "tiles_uint8.npy", directory / "tile_lineage.json"


def _feature_paths(image: int) -> tuple[Path, Path]:
    image = _require_image(image)
    return (
        FEATURE_ROOT / f"image_{image:04d}_features.npz",
        FEATURE_ROOT / f"image_{image:04d}_features.json",
    )


def _feature_receipt_path(image: int) -> Path:
    return FEATURE_ROOT / f"image_{_require_image(image):04d}_receipt.json"


def prepare_upstream_tile_bytes(
    image: int, ledger_path: Path, ledger_sha256: str
) -> None:
    """Trusted lineage worker; export only detached corrupted tile bytes."""

    image = _require_image(image)
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    if image != CANARY_IMAGE:
        verify_feature_canary(ledger_path, ledger_sha256)
    source_record = _validated_upstream_projection(ledger, image)
    validation_name, tiles = _replay_detached_tiles(image)
    raw_record = source_record["raw_cache"]
    if (
        validation_name != source_record["validation_name"]
        or not Path(raw_record["path"]).resolve().is_file()
        or Path(raw_record["path"]).resolve().stat().st_size != raw_record["bytes"]
        or _sha256_file(Path(raw_record["path"]).resolve())
        != raw_record["file_sha256"]
    ):
        raise E24RunnerError("tile-lineage replay differs from preflight projection")
    if tiles.shape != (576, 20, 20, 3) or _array_sha256(tiles) != source_record[
        "tiles_uint8_sha256"
    ]:
        raise E24RunnerError("tile-lineage replay digest/shape differs from E23")
    tiles_path, lineage_path = _tile_lineage_paths(image)
    tiles_bytes = evaluator._npy_bytes(tiles)
    lineage_estimate = 64 * 1024
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=sum(
            (
                0 if tiles_path.exists() else len(tiles_bytes),
                0 if lineage_path.exists() else lineage_estimate,
            )
        ),
    )
    evaluator._atomic_write_create_or_verify(tiles_path, tiles_bytes)
    projection = ledger["upstream"]["label_free_input_projection"]
    payload = {
        "schema": TILE_LINEAGE_SCHEMA,
        "status": "complete",
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "image": image,
        "validation_name": source_record["validation_name"],
        "source_projection_sha256": projection["records_sha256"],
        "raw_cache": dict(raw_record),
        "tiles": {
            "path": str(tiles_path.resolve()),
            "bytes": tiles_path.stat().st_size,
            "file_sha256": _sha256_file(tiles_path),
            "array_sha256": _array_sha256(tiles),
            "shape": [576, 20, 20, 3],
            "dtype": "uint8",
        },
        "output_capability": ["tiles_uint8"],
        "sealed_fields_exported": False,
    }
    lineage_bytes = _canonical_json_bytes(payload)
    if len(lineage_bytes) > lineage_estimate:
        raise E24RunnerError("tile-lineage manifest exceeded 64 KiB")
    evaluator._atomic_write_create_or_verify(lineage_path, lineage_bytes)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)


def _load_upstream_tile_bytes(
    image: int,
    ledger_sha256: str,
    run_contract_sha256: str,
    source_record: Mapping[str, Any],
    projection_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    tiles_path, lineage_path = _tile_lineage_paths(image)
    payload = _load_canonical_json(lineage_path, label="tile-lineage manifest")
    expected = {
        "schema",
        "status",
        "ledger_sha256",
        "run_contract_sha256",
        "image",
        "validation_name",
        "source_projection_sha256",
        "raw_cache",
        "tiles",
        "output_capability",
        "sealed_fields_exported",
    }
    tiles_record = payload.get("tiles")
    if (
        set(payload) != expected
        or payload["schema"] != TILE_LINEAGE_SCHEMA
        or payload["status"] != "complete"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["image"] != image
        or payload["validation_name"] != source_record["validation_name"]
        or payload["source_projection_sha256"] != projection_sha256
        or payload["raw_cache"] != source_record["raw_cache"]
        or payload["output_capability"] != ["tiles_uint8"]
        or payload["sealed_fields_exported"] is not False
        or type(tiles_record) is not dict
        or set(tiles_record)
        != {"path", "bytes", "file_sha256", "array_sha256", "shape", "dtype"}
        or Path(str(tiles_record["path"])).resolve() != tiles_path.resolve()
        or not tiles_path.is_file()
        or tiles_record["bytes"] != tiles_path.stat().st_size
        or tiles_record["shape"] != [576, 20, 20, 3]
        or tiles_record["dtype"] != "uint8"
        or tiles_record["array_sha256"] != source_record["tiles_uint8_sha256"]
    ):
        raise E24RunnerError("tile-lineage capability identity drifted")
    tiles = _load_exact_npy(
        tiles_path,
        shape=(576, 20, 20, 3),
        dtype=np.dtype(np.uint8),
        file_sha256=str(tiles_record["file_sha256"]),
    )
    if _array_sha256(tiles) != tiles_record["array_sha256"]:
        raise E24RunnerError("tile-lineage array digest mismatch")
    return payload, tiles


def prepare_label_free_input(
    image: int, ledger_path: Path, ledger_sha256: str
) -> None:
    """Strict broker: open only raw IDs/scores and projected spatial artifacts."""

    image = _require_image(image)
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    if image != CANARY_IMAGE:
        verify_feature_canary(ledger_path, ledger_sha256)
    source_record = _validated_upstream_projection(ledger, image)
    projection_sha = ledger["upstream"]["label_free_input_projection"][
        "records_sha256"
    ]
    lineage, tiles = _load_upstream_tile_bytes(
        image,
        ledger_sha256,
        ledger["run_contract_sha256"],
        source_record,
        projection_sha,
    )
    directory = INPUT_ROOT / f"image_{image:04d}"
    raw_npz = directory / "raw_candidates.npz"
    raw_manifest = directory / "raw_candidates.json"
    spatial_path = directory / "spatial_logits.npy"
    spatial_metadata_path = directory / "spatial_source.json"
    input_manifest = directory / "input_manifest.json"
    raw_record = source_record["raw_cache"]
    original_raw = Path(raw_record["path"]).resolve()
    if (
        not original_raw.is_file()
        or original_raw.stat().st_size != raw_record["bytes"]
        or _sha256_file(original_raw) != raw_record["file_sha256"]
    ):
        raise E24RunnerError("projected original raw-cache provenance mismatch")
    try:
        original_arrays = evaluator.load_original_raw_candidate_members(
            original_raw,
            expected_sha256=raw_record["file_sha256"],
        )
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError("two-member original raw-cache read failed") from exc
    candidate_ids = original_arrays.candidate_ids
    candidate_scores = original_arrays.candidate_scores
    if (
        _array_sha256(candidate_ids) != source_record["candidate_ids_sha256"]
        or _array_sha256(candidate_scores) != source_record["raw_logits_sha256"]
    ):
        raise E24RunnerError("allowlisted original raw arrays differ from E23")
    scene_record = {
        "image": image,
        "validation_name": source_record["validation_name"],
        "raw_cache_path": str(original_raw),
        "raw_cache_sha256": raw_record["file_sha256"],
        "candidate_ids_sha256": _array_sha256(candidate_ids),
        "raw_logits_sha256": _array_sha256(candidate_scores),
        "tiles_uint8_sha256": _array_sha256(tiles),
    }
    scene_contract_sha = _canonical_digest(scene_record)
    raw_bytes_estimate = evaluator._canonical_raw_npz_bytes(
        candidate_ids, candidate_scores
    )
    spatial_record = source_record["spatial_logits"]
    upstream_spatial = Path(spatial_record["path"]).resolve()
    upstream_metadata = Path(spatial_record["metadata_path"]).resolve()
    if (
        not upstream_spatial.is_file()
        or upstream_spatial.stat().st_size != spatial_record["bytes"]
        or _sha256_file(upstream_spatial) != spatial_record["file_sha256"]
        or not upstream_metadata.is_file()
        or upstream_metadata.stat().st_size != spatial_record["metadata_bytes"]
        or _sha256_file(upstream_metadata) != spatial_record["metadata_file_sha256"]
    ):
        raise E24RunnerError("projected E23 spatial payload/sidecar drifted")
    spatial_payload_bytes = upstream_spatial.read_bytes()
    spatial_metadata_bytes = upstream_metadata.read_bytes()
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=sum(
            (
                0 if raw_npz.exists() else len(raw_bytes_estimate),
                0 if raw_manifest.exists() else evaluator.SANITIZED_RAW_MANIFEST_BYTES_MAX,
                0 if spatial_path.exists() else len(spatial_payload_bytes),
                0 if spatial_metadata_path.exists() else len(spatial_metadata_bytes),
                0 if input_manifest.exists() else 64 * 1024,
            )
        ),
    )
    raw_artifact = evaluator.sanitize_raw_candidate_cache(
        scene_id=image,
        original_raw_cache_path=original_raw,
        expected_original_sha256=raw_record["file_sha256"],
        source_scene_contract_sha256=scene_contract_sha,
        candidate_ids=candidate_ids,
        candidate_scores=candidate_scores,
        sanitized_npz_path=raw_npz,
        manifest_path=raw_manifest,
    )
    evaluator._atomic_write_create_or_verify(spatial_path, spatial_payload_bytes)
    evaluator._atomic_write_create_or_verify(
        spatial_metadata_path, spatial_metadata_bytes
    )
    try:
        spatial = np.load(spatial_path, allow_pickle=False)
    except Exception as exc:
        raise E24RunnerError("detached spatial logits are unreadable") from exc
    _validate_spatial_logits(spatial)
    if _array_sha256(spatial) != spatial_record["array_sha256"]:
        raise E24RunnerError("detached spatial array digest mismatch")
    tiles_path, lineage_path = _tile_lineage_paths(image)
    payload = {
        "schema": INPUT_SCHEMA,
        "status": "complete",
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "image": image,
        "validation_name": source_record["validation_name"],
        "orientation_degrees": 0,
        "reflection": False,
        "source_projection_sha256": projection_sha,
        "raw_member_allowlist": ["candidate_ids", "candidate_scores"],
        "raw_manifest": {
            "path": str(raw_artifact.manifest_path),
            "sha256": raw_artifact.manifest_sha256,
        },
        "tile_lineage_manifest": {
            "path": str(lineage_path.resolve()),
            "sha256": _sha256_file(lineage_path),
        },
        "tiles": dict(lineage["tiles"]),
        "spatial_logits": {
            "path": str(spatial_path.resolve()),
            "file_sha256": _sha256_file(spatial_path),
            "array_sha256": _array_sha256(spatial),
            "shape": [4, 576, 576],
            "dtype": "float32",
            "upstream_path": str(upstream_spatial),
            "upstream_file_sha256": spatial_record["file_sha256"],
            "metadata_path": str(spatial_metadata_path.resolve()),
            "metadata_file_sha256": _sha256_file(spatial_metadata_path),
            "upstream_metadata_path": str(upstream_metadata),
            "upstream_metadata_file_sha256": spatial_record[
                "metadata_file_sha256"
            ],
        },
        "source_scene_contract_sha256": scene_contract_sha,
        "e23_report_sha256": EXPECTED_E23_REPORT_SHA256,
    }
    input_manifest_bytes = _canonical_json_bytes(payload)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=(0 if input_manifest.exists() else len(input_manifest_bytes)),
    )
    evaluator._atomic_write_create_or_verify(input_manifest, input_manifest_bytes)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)


def _load_projected_permutation(
    ledger: Mapping[str, Any], image: int
) -> np.ndarray:
    """Label capability: open exactly one permutation member from a pinned cache."""

    source_record = _validated_upstream_projection(ledger, image)
    raw_record = source_record["raw_cache"]
    try:
        return evaluator.load_original_permutation_member(
            raw_record["path"], expected_sha256=raw_record["file_sha256"]
        )
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError("authenticated permutation-only read failed") from exc


def _load_input_bundle(
    image: int,
    ledger_sha256: str,
    run_contract_sha256: str,
    source_projection_sha256: str,
) -> tuple[dict[str, Any], evaluator.SanitizedRawArtifact, np.ndarray, np.ndarray]:
    image = _require_image(image)
    path = _input_manifest_path(image)
    payload = _load_canonical_json(path, label=f"scene {image} input manifest")
    expected_keys = {
        "schema",
        "status",
        "ledger_sha256",
        "run_contract_sha256",
        "image",
        "validation_name",
        "orientation_degrees",
        "reflection",
        "source_projection_sha256",
        "raw_member_allowlist",
        "raw_manifest",
        "tile_lineage_manifest",
        "tiles",
        "spatial_logits",
        "source_scene_contract_sha256",
        "e23_report_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload["schema"] != INPUT_SCHEMA
        or payload["status"] != "complete"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["image"] != image
        or payload["orientation_degrees"] != 0
        or payload["reflection"] is not False
        or payload["source_projection_sha256"] != source_projection_sha256
        or payload["raw_member_allowlist"]
        != ["candidate_ids", "candidate_scores"]
        or payload["e23_report_sha256"] != EXPECTED_E23_REPORT_SHA256
    ):
        raise E24RunnerError("input bundle identity drifted")
    raw_record = payload["raw_manifest"]
    if type(raw_record) is not dict or set(raw_record) != {"path", "sha256"}:
        raise E24RunnerError("input raw-manifest record drifted")
    raw_manifest = _require_storage(Path(raw_record["path"]), label="raw manifest")
    expected_raw_manifest = (
        INPUT_ROOT / f"image_{image:04d}" / "raw_candidates.json"
    ).resolve()
    if (
        raw_manifest != expected_raw_manifest
        or not raw_manifest.is_file()
        or _sha256_file(raw_manifest) != raw_record["sha256"]
    ):
        raise E24RunnerError("input raw-manifest SHA mismatch")
    raw = evaluator.verify_sanitized_raw_artifact(raw_manifest)
    lineage_record = payload["tile_lineage_manifest"]
    expected_lineage = _tile_lineage_paths(image)[1].resolve()
    if (
        type(lineage_record) is not dict
        or set(lineage_record) != {"path", "sha256"}
        or Path(str(lineage_record["path"])).resolve() != expected_lineage
        or not expected_lineage.is_file()
        or _sha256_file(expected_lineage) != lineage_record["sha256"]
    ):
        raise E24RunnerError("input tile-lineage manifest binding drifted")
    lineage_payload = _load_canonical_json(
        expected_lineage, label="input-bound tile-lineage manifest"
    )
    if (
        set(lineage_payload)
        != {
            "schema",
            "status",
            "ledger_sha256",
            "run_contract_sha256",
            "image",
            "validation_name",
            "source_projection_sha256",
            "raw_cache",
            "tiles",
            "output_capability",
            "sealed_fields_exported",
        }
        or lineage_payload["schema"] != TILE_LINEAGE_SCHEMA
        or lineage_payload["status"] != "complete"
        or lineage_payload["ledger_sha256"] != ledger_sha256
        or lineage_payload["run_contract_sha256"] != run_contract_sha256
        or lineage_payload["image"] != image
        or lineage_payload["validation_name"] != payload["validation_name"]
        or lineage_payload["source_projection_sha256"]
        != source_projection_sha256
        or lineage_payload["output_capability"] != ["tiles_uint8"]
        or lineage_payload["sealed_fields_exported"] is not False
    ):
        raise E24RunnerError("input-bound tile-lineage capability drifted")
    tiles_record = payload["tiles"]
    spatial_record = payload["spatial_logits"]
    if (
        type(tiles_record) is not dict
        or set(tiles_record)
        != {"path", "bytes", "file_sha256", "array_sha256", "shape", "dtype"}
        or type(spatial_record) is not dict
        or set(spatial_record)
        != {
            "path",
            "file_sha256",
            "array_sha256",
            "shape",
            "dtype",
            "upstream_path",
            "upstream_file_sha256",
            "metadata_path",
            "metadata_file_sha256",
            "upstream_metadata_path",
            "upstream_metadata_file_sha256",
        }
    ):
        raise E24RunnerError("input array records are malformed")
    if lineage_payload["tiles"] != tiles_record:
        raise E24RunnerError("input tiles differ from tile-lineage receipt")
    expected_tiles = (
        INPUT_ROOT / f"image_{image:04d}" / "tiles_uint8.npy"
    ).resolve()
    expected_spatial = (
        INPUT_ROOT / f"image_{image:04d}" / "spatial_logits.npy"
    ).resolve()
    if (
        Path(tiles_record["path"]).resolve() != expected_tiles
        or not expected_tiles.is_file()
        or tiles_record["bytes"] != expected_tiles.stat().st_size
        or tiles_record["shape"] != [576, 20, 20, 3]
        or tiles_record["dtype"] != "uint8"
        or Path(spatial_record["path"]).resolve() != expected_spatial
        or spatial_record["shape"] != [4, 576, 576]
        or spatial_record["dtype"] != "float32"
    ):
        raise E24RunnerError("input detached array identity drifted")
    try:
        upstream_spatial = evaluator._require_e_drive(
            Path(spatial_record["upstream_path"]), label="upstream E23 spatial cache"
        )
        upstream_metadata = evaluator._require_e_drive(
            Path(spatial_record["upstream_metadata_path"]),
            label="upstream E23 spatial-cache metadata",
        )
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError(str(exc)) from exc
    expected_metadata = (
        INPUT_ROOT / f"image_{image:04d}" / "spatial_source.json"
    ).resolve()
    if (
        not upstream_spatial.is_file()
        or _sha256_file(upstream_spatial)
        != spatial_record["upstream_file_sha256"]
        or not upstream_metadata.is_file()
        or _sha256_file(upstream_metadata)
        != spatial_record["upstream_metadata_file_sha256"]
        or Path(spatial_record["metadata_path"]).resolve() != expected_metadata
        or not expected_metadata.is_file()
        or _sha256_file(expected_metadata)
        != spatial_record["metadata_file_sha256"]
        or expected_metadata.read_bytes() != upstream_metadata.read_bytes()
    ):
        raise E24RunnerError("upstream E23 spatial cache/sidecar provenance mismatch")
    tiles = _load_exact_npy(
        Path(tiles_record["path"]),
        shape=(576, 20, 20, 3),
        dtype=np.dtype(np.uint8),
        file_sha256=str(tiles_record["file_sha256"]),
    )
    spatial = _load_exact_npy(
        Path(spatial_record["path"]),
        shape=(4, 576, 576),
        dtype=np.dtype(np.float32),
        file_sha256=str(spatial_record["file_sha256"]),
    )
    if (
        _array_sha256(tiles) != tiles_record["array_sha256"]
        or _array_sha256(spatial) != spatial_record["array_sha256"]
    ):
        raise E24RunnerError("input detached array digest mismatch")
    if (
        raw.scene_id != image
        or raw.manifest_path != expected_raw_manifest
        or raw.npz_path
        != (INPUT_ROOT / f"image_{image:04d}" / "raw_candidates.npz").resolve()
        or raw.manifest_sha256 != raw_record["sha256"]
        or raw.source_scene_contract_sha256
        != payload["source_scene_contract_sha256"]
        or lineage_payload["raw_cache"]
        != {
            "path": str(raw.original_path),
            "bytes": raw.original_path.stat().st_size,
            "file_sha256": raw.original_sha256,
        }
    ):
        raise E24RunnerError("sanitized raw/input scene binding drifted")
    reconstructed_scene_contract = _canonical_digest(
        {
            "image": image,
            "validation_name": payload["validation_name"],
            "raw_cache_path": str(raw.original_path),
            "raw_cache_sha256": raw.original_sha256,
            "candidate_ids_sha256": _array_sha256(raw.arrays.candidate_ids),
            "raw_logits_sha256": _array_sha256(raw.arrays.candidate_scores),
            "tiles_uint8_sha256": _array_sha256(tiles),
        }
    )
    if reconstructed_scene_contract != payload["source_scene_contract_sha256"]:
        raise E24RunnerError("input bundle does not reproduce its source-scene contract")
    return payload, raw, tiles, spatial


def _recompute_candidate_pool(
    image: int,
    ledger_sha256: str,
    run_contract_sha256: str,
    source_projection_sha256: str,
) -> tuple[dict[str, Any], evaluator.SanitizedRawArtifact, np.ndarray, np.ndarray, e23_core.CandidatePoolResult]:
    payload, raw, tiles, spatial = _load_input_bundle(
        image, ledger_sha256, run_contract_sha256, source_projection_sha256
    )
    try:
        result = e23_core.run_i21_residual_candidate_oracle(
            raw.arrays.candidate_ids, raw.arrays.candidate_scores, spatial
        )
    except Exception as exc:
        raise E24RunnerError("E23 candidate-pool replay failed") from exc
    if len(result.hypotheses) > evaluator.GEOMETRY_HYPOTHESES_MAX_EACH:
        raise E24RunnerError("E23 candidate-pool replay exceeded 450000 hypotheses")
    return payload, raw, tiles, spatial, result


def run_feature_worker(
    image: int, ledger_path: Path, ledger_sha256: str
) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    image = _require_image(image)
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    if image != CANARY_IMAGE:
        verify_feature_canary(ledger_path, ledger_sha256)
    receipt_path = _feature_receipt_path(image)
    if receipt_path.is_file():
        _table, _manifest = _load_feature_artifact(
            image, ledger_sha256, ledger["run_contract_sha256"]
        )
        receipt = _load_feature_receipt(
            image, ledger_sha256, ledger["run_contract_sha256"]
        )
        if image == CANARY_IMAGE:
            _commit_or_verify_feature_canary_gate(ledger_path, ledger_sha256)
            verify_feature_canary(ledger_path, ledger_sha256)
        return receipt
    input_payload, raw, tiles, spatial, result = _recompute_candidate_pool(
        image,
        ledger_sha256,
        ledger["run_contract_sha256"],
        ledger["upstream"]["label_free_input_projection"]["records_sha256"],
    )
    try:
        table = selector.extract_relation_features(
            result,
            raw.arrays.candidate_ids,
            raw.arrays.candidate_scores,
            spatial,
            tiles,
        )
    except Exception as exc:
        raise E24RunnerError("label-free CRS feature extraction failed") from exc
    feature_path, manifest_path = _feature_paths(image)
    _require_storage(feature_path, label="feature cache")
    _require_storage(manifest_path, label="feature manifest")
    feature_bytes = selector.feature_table_npz_bytes(table)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_feature_bytes=(0 if feature_path.exists() else len(feature_bytes)),
        additional_total_bytes=(0 if feature_path.exists() else len(feature_bytes)),
    )
    evaluator._atomic_write_create_or_verify(feature_path, feature_bytes)
    payload = {
        "schema": FEATURE_MANIFEST_SCHEMA,
        "status": "complete",
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "image": image,
        "input_manifest": {
            "path": str(_input_manifest_path(image).resolve()),
            "sha256": _sha256_file(_input_manifest_path(image)),
        },
        "feature_file": {
            "path": str(feature_path.resolve()),
            "bytes": feature_path.stat().st_size,
            "sha256": _sha256_file(feature_path),
        },
        "feature_names_sha256": ledger["ordered_feature_names_sha256"],
        "rows": table.rows,
        "queries": table.queries,
        "hypotheses": len(result.hypotheses),
        "finite": bool(np.isfinite(table.features).all()),
        "candidate_pool_input_sha256": input_payload["source_scene_contract_sha256"],
    }
    if not payload["finite"] or feature_path.stat().st_size > evaluator.FEATURE_CACHE_BYTES_MAX:
        raise E24RunnerError("feature artifact finite/size gate failed")
    manifest_bytes = _canonical_json_bytes(payload)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_feature_bytes=(
            0 if manifest_path.exists() else len(manifest_bytes)
        ),
        additional_total_bytes=(0 if manifest_path.exists() else len(manifest_bytes)),
    )
    evaluator._atomic_write_create_or_verify(
        manifest_path, manifest_bytes
    )
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    resource = _resource_snapshot(started_wall, started_cpu)
    receipt = {
        "schema": FEATURE_RECEIPT_SCHEMA,
        "status": "complete_measurement",
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "image": image,
        "feature_manifest_sha256": _sha256_file(manifest_path),
        "feature_file_sha256": _sha256_file(feature_path),
        "feature_file_bytes": feature_path.stat().st_size,
        "hypotheses": len(result.hypotheses),
        "resource": resource,
    }
    receipt_bytes = _canonical_json_bytes(receipt)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_feature_bytes=len(receipt_bytes),
        additional_total_bytes=len(receipt_bytes),
    )
    evaluator._atomic_write_create(receipt_path, receipt_bytes)
    _enforce_process_resource_caps(resource)
    if image == CANARY_IMAGE:
        _commit_or_verify_feature_canary_gate(ledger_path, ledger_sha256)
        verify_feature_canary(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    return receipt


def _load_feature_artifact(
    image: int, ledger_sha256: str, run_contract_sha256: str
) -> tuple[selector.RelationFeatureTable, dict[str, Any]]:
    image = _require_image(image)
    feature_path, manifest_path = _feature_paths(image)
    payload = _load_canonical_json(manifest_path, label="feature manifest")
    expected = {
        "schema",
        "status",
        "ledger_sha256",
        "run_contract_sha256",
        "image",
        "input_manifest",
        "feature_file",
        "feature_names_sha256",
        "rows",
        "queries",
        "hypotheses",
        "finite",
        "candidate_pool_input_sha256",
    }
    if (
        set(payload) != expected
        or payload["schema"] != FEATURE_MANIFEST_SCHEMA
        or payload["status"] != "complete"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["image"] != image
        or payload["finite"] is not True
        or payload["feature_names_sha256"]
        != hashlib.sha256(
            _canonical_json_bytes({"feature_names": list(selector.FEATURE_NAMES)})
        ).hexdigest()
    ):
        raise E24RunnerError("feature manifest identity drifted")
    record = payload["feature_file"]
    if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
        raise E24RunnerError("feature file record drifted")
    recorded_path = _require_storage(Path(record["path"]), label="feature cache")
    if recorded_path != feature_path.resolve() or (
        not feature_path.is_file()
        or feature_path.stat().st_size != record["bytes"]
        or _sha256_file(feature_path) != record["sha256"]
    ):
        raise E24RunnerError("feature file provenance mismatch")
    input_record = payload["input_manifest"]
    expected_input = _input_manifest_path(image).resolve()
    if (
        type(input_record) is not dict
        or set(input_record) != {"path", "sha256"}
        or _require_storage(Path(input_record["path"]), label="input manifest")
        != expected_input
        or not expected_input.is_file()
        or _sha256_file(expected_input) != input_record["sha256"]
    ):
        raise E24RunnerError("feature/input manifest provenance mismatch")
    input_payload = _load_canonical_json(
        expected_input, label=f"scene {image} input manifest"
    )
    if (
        input_payload.get("image") != image
        or input_payload.get("ledger_sha256") != ledger_sha256
        or input_payload.get("run_contract_sha256") != run_contract_sha256
        or payload["candidate_pool_input_sha256"]
        != input_payload.get("source_scene_contract_sha256")
    ):
        raise E24RunnerError("feature/input source binding drifted")
    try:
        table = selector.load_feature_table_npz(feature_path)
    except Exception as exc:
        raise E24RunnerError("feature table strict replay failed") from exc
    if table.rows != payload["rows"] or table.queries != payload["queries"]:
        raise E24RunnerError("feature row/query counts drifted")
    if (
        type(payload["hypotheses"]) is not int
        or payload["hypotheses"] < 0
        or payload["hypotheses"] > evaluator.GEOMETRY_HYPOTHESES_MAX_EACH
        or int(np.count_nonzero(table.row_kind == selector.ROW_OFFSET))
        != payload["hypotheses"]
    ):
        raise E24RunnerError("feature hypothesis count drifted")
    return table, payload


def _load_feature_receipt(
    image: int, ledger_sha256: str, run_contract_sha256: str
) -> dict[str, Any]:
    image = _require_image(image)
    receipt_path = _feature_receipt_path(image)
    payload = _load_canonical_json(receipt_path, label="feature-worker receipt")
    expected = {
        "schema",
        "status",
        "ledger_sha256",
        "run_contract_sha256",
        "image",
        "feature_manifest_sha256",
        "feature_file_sha256",
        "feature_file_bytes",
        "hypotheses",
        "resource",
    }
    feature_path, manifest_path = _feature_paths(image)
    if (
        set(payload) != expected
        or payload["schema"] != FEATURE_RECEIPT_SCHEMA
        or payload["status"] != "complete_measurement"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["image"] != image
        or not manifest_path.is_file()
        or payload["feature_manifest_sha256"] != _sha256_file(manifest_path)
        or not feature_path.is_file()
        or payload["feature_file_sha256"] != _sha256_file(feature_path)
        or payload["feature_file_bytes"] != feature_path.stat().st_size
    ):
        raise E24RunnerError("feature-worker receipt identity/provenance drifted")
    resource = payload["resource"]
    if (
        type(resource) is not dict
        or set(resource)
        != {"wall_seconds", "process_cpu_seconds", "peak_rss_bytes"}
        or not isinstance(resource["wall_seconds"], (int, float))
        or not isinstance(resource["process_cpu_seconds"], (int, float))
        or type(resource["peak_rss_bytes"]) is not int
        or not np.isfinite(float(resource["wall_seconds"]))
        or not np.isfinite(float(resource["process_cpu_seconds"]))
        or float(resource["wall_seconds"]) < 0.0
        or float(resource["process_cpu_seconds"]) < 0.0
        or resource["peak_rss_bytes"] <= 0
    ):
        raise E24RunnerError("feature-worker resource receipt is malformed")
    if type(payload["hypotheses"]) is not int or payload["hypotheses"] < 0:
        raise E24RunnerError("feature-worker hypothesis receipt is malformed")
    _enforce_process_resource_caps(resource)
    return payload


def _canary_thresholds() -> dict[str, int]:
    return {
        "image": CANARY_IMAGE,
        "hypotheses": CANARY_EXPECTED_HYPOTHESES,
        "wall_seconds_max": CANARY_WALL_SECONDS_MAX,
        "peak_rss_bytes_max": CANARY_PEAK_RSS_BYTES_MAX,
        "feature_file_bytes_max": CANARY_FEATURE_BYTES_MAX,
        "projected_feature_bytes_max": evaluator.FEATURE_CACHE_BYTES_MAX,
        "projected_total_bytes_max": evaluator.ALL_ARTIFACT_BYTES_MAX,
        "fold_output_reserve_bytes": CANARY_FOLD_OUTPUT_RESERVE_BYTES,
    }


def _canary_observed_and_checks(
    receipt: Mapping[str, Any],
    *,
    input_bytes: int,
    feature_bundle_bytes: int,
    aggregate_total_bytes_at_gate: int,
    first_target_state: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    projected_feature_bytes = feature_bundle_bytes * len(evaluator.CALIBRATION_IDS)
    projected_total_bytes = (
        aggregate_total_bytes_at_gate
        + (len(evaluator.CALIBRATION_IDS) - 1)
        * (input_bytes + feature_bundle_bytes)
        + CANARY_FOLD_OUTPUT_RESERVE_BYTES
    )
    observed = {
        "image": CANARY_IMAGE,
        "hypotheses": int(receipt["hypotheses"]),
        "wall_seconds": float(receipt["resource"]["wall_seconds"]),
        "peak_rss_bytes": int(receipt["resource"]["peak_rss_bytes"]),
        "feature_file_bytes": int(receipt["feature_file_bytes"]),
        "input_bundle_bytes": input_bytes,
        "feature_bundle_bytes": feature_bundle_bytes,
        "aggregate_total_bytes_at_gate": aggregate_total_bytes_at_gate,
        "projected_feature_bytes": projected_feature_bytes,
        "projected_total_bytes": projected_total_bytes,
        "first_target_state": first_target_state,
    }
    checks = {
        "first_target_is_scene17_only": first_target_state,
        "exact_frozen_hypothesis_count": observed["hypotheses"]
        == CANARY_EXPECTED_HYPOTHESES,
        "wall_seconds_at_most_1800": observed["wall_seconds"]
        <= CANARY_WALL_SECONDS_MAX,
        "peak_rss_at_most_4gib": observed["peak_rss_bytes"]
        <= CANARY_PEAK_RSS_BYTES_MAX,
        "feature_file_at_most_480mib": observed["feature_file_bytes"]
        <= CANARY_FEATURE_BYTES_MAX,
        "projected_feature_at_most_4gib": projected_feature_bytes
        <= evaluator.FEATURE_CACHE_BYTES_MAX,
        "projected_total_at_most_8gib": projected_total_bytes
        <= evaluator.ALL_ARTIFACT_BYTES_MAX,
    }
    return observed, checks


def _commit_or_verify_feature_canary_gate(
    ledger_path: Path, ledger_sha256: str
) -> dict[str, Any]:
    if CANARY_GATE_PATH.is_file():
        return _load_canonical_json(CANARY_GATE_PATH, label="feature canary gate")
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    receipt = _load_feature_receipt(
        CANARY_IMAGE, ledger_sha256, ledger["run_contract_sha256"]
    )
    input_files = _artifact_files(
        (INPUT_ROOT / f"image_{CANARY_IMAGE:04d}",)
    )
    feature_path, feature_manifest_path = _feature_paths(CANARY_IMAGE)
    expected_feature_files = {
        feature_path.resolve(),
        feature_manifest_path.resolve(),
        _feature_receipt_path(CANARY_IMAGE).resolve(),
    }
    all_input_files = _artifact_files((INPUT_ROOT,))
    all_feature_files = _artifact_files((FEATURE_ROOT,))
    first_target_state = bool(
        input_files
        and all_input_files == input_files
        and all_feature_files == expected_feature_files
        and not _artifact_files((FOLD_ROOT, STRUCTURAL_REPORT))
    )
    input_bytes = sum(path.stat().st_size for path in input_files)
    feature_bundle_bytes = sum(path.stat().st_size for path in expected_feature_files)
    aggregate = enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    observed, checks = _canary_observed_and_checks(
        receipt,
        input_bytes=input_bytes,
        feature_bundle_bytes=feature_bundle_bytes,
        aggregate_total_bytes_at_gate=aggregate["total_bytes"],
        first_target_state=first_target_state,
    )
    passed = all(checks.values())
    payload = {
        "schema": "pazzle-e24-crs-v1-feature-canary-gate-v1",
        "status": "pass" if passed else "infrastructure_stop",
        "passed": passed,
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "receipt_sha256": _sha256_file(_feature_receipt_path(CANARY_IMAGE)),
        "thresholds": _canary_thresholds(),
        "observed": observed,
        "checks": checks,
        "labels_or_metrics_opened": False,
    }
    gate_bytes = _canonical_json_bytes(payload)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path, additional_total_bytes=len(gate_bytes)
    )
    evaluator._atomic_write_create(CANARY_GATE_PATH, gate_bytes)
    return payload


def verify_feature_canary(
    ledger_path: Path, ledger_sha256: str
) -> dict[str, Any]:
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    receipt = _load_feature_receipt(
        CANARY_IMAGE, ledger_sha256, ledger["run_contract_sha256"]
    )
    gate = _load_canonical_json(CANARY_GATE_PATH, label="feature canary gate")
    expected_keys = {
        "schema",
        "status",
        "passed",
        "ledger_sha256",
        "run_contract_sha256",
        "receipt_sha256",
        "thresholds",
        "observed",
        "checks",
        "labels_or_metrics_opened",
    }
    input_files = _artifact_files(
        (INPUT_ROOT / f"image_{CANARY_IMAGE:04d}",)
    )
    feature_path, feature_manifest_path = _feature_paths(CANARY_IMAGE)
    expected_feature_files = {
        feature_path.resolve(),
        feature_manifest_path.resolve(),
        _feature_receipt_path(CANARY_IMAGE).resolve(),
    }
    if not input_files or not all(path.is_file() for path in expected_feature_files):
        raise E24RunnerError("scene-17 canary artifacts are incomplete")
    stored_observed = gate.get("observed")
    if type(stored_observed) is not dict:
        raise E24RunnerError("scene-17 canary observed payload is malformed")
    aggregate_at_gate = stored_observed.get("aggregate_total_bytes_at_gate")
    first_target_state = stored_observed.get("first_target_state")
    if type(aggregate_at_gate) is not int or type(first_target_state) is not bool:
        raise E24RunnerError("scene-17 canary gate snapshot is malformed")
    expected_observed, expected_checks = _canary_observed_and_checks(
        receipt,
        input_bytes=sum(path.stat().st_size for path in input_files),
        feature_bundle_bytes=sum(
            path.stat().st_size for path in expected_feature_files
        ),
        aggregate_total_bytes_at_gate=aggregate_at_gate,
        first_target_state=first_target_state,
    )
    if (
        set(gate) != expected_keys
        or gate["schema"] != "pazzle-e24-crs-v1-feature-canary-gate-v1"
        or gate["ledger_sha256"] != ledger_sha256
        or gate["run_contract_sha256"] != ledger["run_contract_sha256"]
        or gate["receipt_sha256"]
        != _sha256_file(_feature_receipt_path(CANARY_IMAGE))
        or gate["thresholds"] != _canary_thresholds()
        or gate["observed"] != expected_observed
        or gate["checks"] != expected_checks
        or gate["labels_or_metrics_opened"] is not False
        or gate["passed"] is not all(expected_checks.values())
        or gate["status"]
        != ("pass" if all(expected_checks.values()) else "infrastructure_stop")
        or not all(expected_checks.values())
    ):
        raise E24RunnerError("scene-17 label-free canary did not pass exact frozen gates")
    aggregate_now = enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    downstream_exists = bool(
        _artifact_files((INPUT_ROOT,)) - input_files
        or _artifact_files((FEATURE_ROOT,)) - expected_feature_files
        or _artifact_files((FOLD_ROOT, STRUCTURAL_REPORT))
    )
    if not downstream_exists:
        expected_at_gate = aggregate_now["total_bytes"] - CANARY_GATE_PATH.stat().st_size
        if not first_target_state or aggregate_at_gate != expected_at_gate:
            raise E24RunnerError("scene-17 canary first-target snapshot drifted")
    return gate


def _label_paths(fold: int, image: int) -> tuple[Path, Path]:
    evaluator.fold_boundary(fold)
    image = _require_image(image)
    root = FOLD_ROOT / f"fold_{fold}" / "train_labels"
    return root / f"image_{image:04d}.npy", root / f"image_{image:04d}.json"


def prepare_fold_train_labels(
    fold: int, ledger_path: Path, ledger_sha256: str
) -> None:
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    verify_feature_canary(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    boundary = evaluator.fold_boundary(fold)
    # All eight immutable feature artifacts must exist before any label package.
    feature_artifacts = {
        image: _load_feature_artifact(
            image, ledger_sha256, ledger["run_contract_sha256"]
        )
        for image in evaluator.CALIBRATION_IDS
    }
    for image in boundary.train_ids:
        label_path, manifest_path = _label_paths(fold, image)
        table, feature_manifest = feature_artifacts[image]
        _input, _raw, _tiles, _spatial, result = _recompute_candidate_pool(
            image,
            ledger_sha256,
            ledger["run_contract_sha256"],
            ledger["upstream"]["label_free_input_projection"]["records_sha256"],
        )
        # This is the only permutation access in a fold-label broker, and the
        # loop is exactly the fold's six training IDs.
        truth = evaluator.build_label_only_relation_truth(
            result, table, _load_projected_permutation(ledger, image)
        )
        label_bytes = evaluator._npy_bytes(truth.relevance)
        enforce_aggregate_artifact_caps(
            ledger_path=ledger_path,
            additional_total_bytes=(
                (0 if label_path.exists() else len(label_bytes))
                + (0 if manifest_path.exists() else 64 * 1024)
            ),
        )
        _atomic_npy(label_path, truth.relevance)
        payload = {
            "schema": LABEL_MANIFEST_SCHEMA,
            "status": "complete",
            "ledger_sha256": ledger_sha256,
            "run_contract_sha256": ledger["run_contract_sha256"],
            "fold": fold,
            "image": image,
            "train_ids": list(boundary.train_ids),
            "heldout_ids": list(boundary.heldout_ids),
            "feature_sha256": feature_manifest["feature_file"]["sha256"],
            "label_file": {
                "path": str(label_path.resolve()),
                "bytes": label_path.stat().st_size,
                "sha256": _sha256_file(label_path),
                "dtype": "int8",
                "shape": [table.rows],
            },
            "onehot_queries": table.queries,
        }
        label_manifest_bytes = _canonical_json_bytes(payload)
        enforce_aggregate_artifact_caps(
            ledger_path=ledger_path,
            additional_total_bytes=(
                0 if manifest_path.exists() else len(label_manifest_bytes)
            ),
        )
        evaluator._atomic_write_create_or_verify(manifest_path, label_manifest_bytes)
        enforce_aggregate_artifact_caps(ledger_path=ledger_path)


def _verify_fold_label_manifest(
    fold: int,
    image: int,
    *,
    ledger_sha256: str,
    run_contract_sha256: str,
    rows: int,
    queries: int,
    feature_sha256: str,
    verify_label_file: bool,
) -> tuple[dict[str, Any], str]:
    boundary = evaluator.fold_boundary(fold)
    if image not in boundary.train_ids:
        raise E24RunnerError("fold trainer attempted to open a held-out label")
    label_path, manifest_path = _label_paths(fold, image)
    payload = _load_canonical_json(manifest_path, label="fold label manifest")
    expected = {
        "schema",
        "status",
        "ledger_sha256",
        "run_contract_sha256",
        "fold",
        "image",
        "train_ids",
        "heldout_ids",
        "feature_sha256",
        "label_file",
        "onehot_queries",
    }
    if (
        set(payload) != expected
        or payload["schema"] != LABEL_MANIFEST_SCHEMA
        or payload["status"] != "complete"
        or payload["ledger_sha256"] != ledger_sha256
        or payload["run_contract_sha256"] != run_contract_sha256
        or payload["fold"] != fold
        or payload["image"] != image
        or payload["train_ids"] != list(boundary.train_ids)
        or payload["heldout_ids"] != list(boundary.heldout_ids)
        or payload["feature_sha256"] != feature_sha256
    ):
        raise E24RunnerError("fold label manifest identity drifted")
    record = payload["label_file"]
    if type(record) is not dict or set(record) != {
        "path",
        "bytes",
        "sha256",
        "dtype",
        "shape",
    }:
        raise E24RunnerError("fold label file record drifted")
    if (
        record["dtype"] != "int8"
        or record["shape"] != [rows]
        or payload["onehot_queries"] != queries
    ):
        raise E24RunnerError("fold label shape/dtype manifest drifted")
    label_file = _require_storage(Path(record["path"]), label="fold label file")
    if (
        label_file != label_path.resolve()
    ):
        raise E24RunnerError("fold label file path drifted")
    if verify_label_file and (
        not label_file.is_file()
        or label_file.stat().st_size != record["bytes"]
        or _sha256_file(label_file) != record["sha256"]
    ):
        raise E24RunnerError("fold label file provenance mismatch")
    return payload, _sha256_file(manifest_path)


def _load_fold_label(
    fold: int,
    image: int,
    *,
    ledger_sha256: str,
    run_contract_sha256: str,
    table: selector.RelationFeatureTable,
    feature_sha256: str,
) -> tuple[np.ndarray, str]:
    payload, manifest_sha256 = _verify_fold_label_manifest(
        fold,
        image,
        ledger_sha256=ledger_sha256,
        run_contract_sha256=run_contract_sha256,
        rows=table.rows,
        queries=table.queries,
        feature_sha256=feature_sha256,
        verify_label_file=True,
    )
    label_file = Path(payload["label_file"]["path"]).resolve()
    labels = _load_exact_npy(
        label_file,
        shape=(table.rows,),
        dtype=np.dtype(np.int8),
        file_sha256=payload["label_file"]["sha256"],
    )
    return labels, manifest_sha256


def _fold_paths(fold: int) -> tuple[Path, Path, Path]:
    evaluator.fold_boundary(fold)
    root = FOLD_ROOT / f"fold_{fold}"
    return root / "model.txt", root / "predictions.npz", root / "commit.json"


def _model_manifest_path(fold: int) -> Path:
    evaluator.fold_boundary(fold)
    return FOLD_ROOT / f"fold_{fold}" / "model.json"


def _serialize_model_bytes(model: Any) -> bytes:
    booster = getattr(model, "booster_", None)
    if booster is None or not hasattr(booster, "num_trees") or booster.num_trees() != 256:
        raise E24RunnerError("trained model is not an exact 256-tree LightGBM booster")
    try:
        text = booster.model_to_string(num_iteration=256)
    except Exception as exc:
        raise E24RunnerError("LightGBM model serialization failed") from exc
    if not isinstance(text, str) or not text:
        raise E24RunnerError("LightGBM serialized model is empty")
    return text.encode("utf-8")


def _fold_run_provenance(
    *,
    fold: int,
    ledger: Mapping[str, Any],
    ledger_sha256: str,
    feature_manifests: Mapping[int, Mapping[str, Any]],
    label_manifest_sha256: Mapping[int, str],
) -> dict[str, Any]:
    boundary = evaluator.fold_boundary(fold)
    if set(feature_manifests) != set(evaluator.CALIBRATION_IDS):
        raise E24RunnerError("run provenance requires all eight feature manifests")
    value = {
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "core_source_sha256": _ledger_source_sha256(
            ledger, ROOT / "src/e24_context_relation_selector.py"
        ),
        "ordered_feature_schema_sha256": ledger[
            "ordered_feature_names_sha256"
        ],
        "lightgbm_contract_sha256": _lightgbm_contract_sha256(ledger, fold),
        "canary_gate_sha256": _sha256_file(CANARY_GATE_PATH),
        "train_feature_sha256": {
            image: feature_manifests[image]["feature_file"]["sha256"]
            for image in boundary.train_ids
        },
        "train_label_manifest_sha256": dict(label_manifest_sha256),
    }
    try:
        return evaluator.normalize_fold_run_provenance(fold, value)
    except evaluator.E24EvaluatorContractError as exc:
        raise E24RunnerError(str(exc)) from exc


def _load_model_manifest(
    fold: int, expected_run_provenance: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    boundary = evaluator.fold_boundary(fold)
    model_path, _prediction, _commit = _fold_paths(fold)
    manifest_path = _model_manifest_path(fold)
    payload = _load_canonical_json(manifest_path, label="fold model manifest")
    if (
        set(payload)
        != {
            "schema",
            "status",
            "fold",
            "train_ids",
            "heldout_ids",
            "run_provenance",
            "model",
        }
        or payload["schema"] != MODEL_MANIFEST_SCHEMA
        or payload["status"] != "complete"
        or payload["fold"] != fold
        or payload["train_ids"] != list(boundary.train_ids)
        or payload["heldout_ids"] != list(boundary.heldout_ids)
        or evaluator.normalize_fold_run_provenance(
            fold, payload["run_provenance"]
        )
        != evaluator.normalize_fold_run_provenance(fold, expected_run_provenance)
    ):
        raise E24RunnerError("fold model manifest identity/provenance drifted")
    record = payload["model"]
    if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
        raise E24RunnerError("fold model file record drifted")
    recorded = _require_storage(Path(record["path"]), label="fold model")
    if (
        recorded != model_path.resolve()
        or not recorded.is_file()
        or recorded.stat().st_size != record["bytes"]
        or _sha256_file(recorded) != record["sha256"]
    ):
        raise E24RunnerError("fold model file provenance mismatch")
    return recorded, payload


def _reload_committed_predictor(
    fold: int, expected_run_provenance: Mapping[str, Any]
) -> tuple[Any, Path, dict[str, Any]]:
    """Authenticate the model transaction, then reload its exact text bytes."""

    model_path, manifest = _load_model_manifest(fold, expected_run_provenance)
    try:
        from lightgbm import Booster

        predictor = Booster(model_file=str(model_path))
    except Exception as exc:
        raise E24RunnerError("committed LightGBM model reload failed") from exc
    if predictor.num_trees() != 256 or predictor.num_feature() != len(
        selector.FEATURE_NAMES
    ):
        raise E24RunnerError(
            "committed predictor tree-count/feature-schema contract drifted"
        )
    return predictor, model_path, manifest


def train_fold_model(
    fold: int, ledger_path: Path, ledger_sha256: str
) -> None:
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    verify_feature_canary(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    boundary = evaluator.fold_boundary(fold)
    tables: dict[int, selector.RelationFeatureTable] = {}
    manifests: dict[int, dict[str, Any]] = {}
    for image in evaluator.CALIBRATION_IDS:
        tables[image], manifests[image] = _load_feature_artifact(
            image, ledger_sha256, ledger["run_contract_sha256"]
        )
    labels: dict[int, np.ndarray] = {}
    label_manifest_hashes: dict[int, str] = {}
    for image in boundary.train_ids:
        labels[image], label_manifest_hashes[image] = _load_fold_label(
            fold,
            image,
            ledger_sha256=ledger_sha256,
            run_contract_sha256=ledger["run_contract_sha256"],
            table=tables[image],
            feature_sha256=manifests[image]["feature_file"]["sha256"],
        )
    run_provenance = _fold_run_provenance(
        fold=fold,
        ledger=ledger,
        ledger_sha256=ledger_sha256,
        feature_manifests=manifests,
        label_manifest_sha256=label_manifest_hashes,
    )
    try:
        model, _batch = evaluator.fit_oof_fold(
            fold, tables_by_scene=tables, relevance_by_scene=labels
        )
    except Exception as exc:
        raise E24RunnerError("fold training failed") from exc
    model_path, _prediction_path, _commit_path = _fold_paths(fold)
    model_bytes = _serialize_model_bytes(model)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=(0 if model_path.exists() else len(model_bytes)),
    )
    evaluator._atomic_write_create_or_verify(model_path, model_bytes)
    model_manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "status": "complete",
        "fold": fold,
        "train_ids": list(boundary.train_ids),
        "heldout_ids": list(boundary.heldout_ids),
        "run_provenance": run_provenance,
        "model": {
            "path": str(model_path.resolve()),
            "bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
        },
    }
    model_manifest_path = _model_manifest_path(fold)
    model_manifest_bytes = _canonical_json_bytes(model_manifest)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=(
            0 if model_manifest_path.exists() else len(model_manifest_bytes)
        ),
    )
    evaluator._atomic_write_create_or_verify(
        model_manifest_path, model_manifest_bytes
    )
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)


def predict_commit_fold(
    fold: int, ledger_path: Path, ledger_sha256: str
) -> None:
    """Reload a committed model in this label-free process, then predict."""

    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    verify_feature_canary(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    boundary = evaluator.fold_boundary(fold)
    tables: dict[int, selector.RelationFeatureTable] = {}
    manifests: dict[int, dict[str, Any]] = {}
    for image in evaluator.CALIBRATION_IDS:
        tables[image], manifests[image] = _load_feature_artifact(
            image, ledger_sha256, ledger["run_contract_sha256"]
        )
    label_manifest_hashes: dict[int, str] = {}
    for image in boundary.train_ids:
        _payload, label_manifest_hashes[image] = _verify_fold_label_manifest(
            fold,
            image,
            ledger_sha256=ledger_sha256,
            run_contract_sha256=ledger["run_contract_sha256"],
            rows=manifests[image]["rows"],
            queries=manifests[image]["queries"],
            feature_sha256=manifests[image]["feature_file"]["sha256"],
            verify_label_file=False,
        )
    run_provenance = _fold_run_provenance(
        fold=fold,
        ledger=ledger,
        ledger_sha256=ledger_sha256,
        feature_manifests=manifests,
        label_manifest_sha256=label_manifest_hashes,
    )
    predictor, model_path, _model_manifest = _reload_committed_predictor(
        fold, run_provenance
    )
    scene_ids: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    counts: dict[int, int] = {}
    for image in boundary.heldout_ids:
        one = selector.predict_scores(predictor, tables[image])
        counts[image] = tables[image].rows
        scene_ids.append(np.full(tables[image].rows, image, dtype=np.int16))
        row_indices.append(np.arange(tables[image].rows, dtype=np.int64))
        scores.append(one)
    rows = evaluator.PredictionRows(
        scene_ids=np.ascontiguousarray(np.concatenate(scene_ids), dtype=np.int16),
        row_indices=np.ascontiguousarray(np.concatenate(row_indices), dtype=np.int64),
        scores=np.ascontiguousarray(np.concatenate(scores), dtype=np.float64),
    )
    model_path, prediction_path, commit_path = _fold_paths(fold)
    prediction_bytes = evaluator._prediction_npz_bytes(
        fold=fold, model_sha256=_sha256_file(model_path), rows=rows
    )
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=(
            (0 if prediction_path.exists() else len(prediction_bytes))
            + (0 if commit_path.exists() else 64 * 1024)
        ),
    )
    evaluator.commit_fold_predictions(
        fold=fold,
        model_path=model_path,
        prediction_path=prediction_path,
        commit_path=commit_path,
        run_provenance=run_provenance,
        feature_sha256={
            image: manifests[image]["feature_file"]["sha256"]
            for image in boundary.heldout_ids
        },
        row_counts=counts,
        rows=rows,
    )
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)


def run_structural_evaluation(ledger_path: Path, ledger_sha256: str) -> dict[str, Any]:
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    verify_feature_canary(ledger_path, ledger_sha256)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    commit_paths = {fold: _fold_paths(fold)[2] for fold in evaluator.OOF_FOLDS}
    feature_manifests: dict[int, dict[str, Any]] = {}
    for image in evaluator.CALIBRATION_IDS:
        _table, feature_manifests[image] = _load_feature_artifact(
            image, ledger_sha256, ledger["run_contract_sha256"]
        )
    expected_run_provenance: dict[int, dict[str, Any]] = {}
    for fold in evaluator.OOF_FOLDS:
        boundary = evaluator.fold_boundary(fold)
        label_hashes: dict[int, str] = {}
        for image in boundary.train_ids:
            manifest = feature_manifests[image]
            _payload, label_hashes[image] = _verify_fold_label_manifest(
                fold,
                image,
                ledger_sha256=ledger_sha256,
                run_contract_sha256=ledger["run_contract_sha256"],
                rows=manifest["rows"],
                queries=manifest["queries"],
                feature_sha256=manifest["feature_file"]["sha256"],
                verify_label_file=False,
            )
        expected_run_provenance[fold] = _fold_run_provenance(
            fold=fold,
            ledger=ledger,
            ledger_sha256=ledger_sha256,
            feature_manifests=feature_manifests,
            label_manifest_sha256=label_hashes,
        )
    model_records: dict[int, tuple[Path, str]] = {}
    for fold in evaluator.OOF_FOLDS:
        predictor, model_path, model_manifest = _reload_committed_predictor(
            fold, expected_run_provenance[fold]
        )
        del predictor
        model_records[fold] = (
            model_path,
            str(model_manifest["model"]["sha256"]),
        )
    # Global barrier comes before raw-scene loading and therefore before the
    # first held-out permutation can even be materialized by this process.
    verified_set = evaluator.verify_all_oof_commits(
        commit_paths, expected_run_provenance=expected_run_provenance
    )
    for fold, commit in verified_set.commits.items():
        expected_model_path, expected_model_sha = model_records[fold]
        if (
            commit.model_path != expected_model_path
            or commit.model_sha256 != expected_model_sha
        ):
            raise E24RunnerError(
                "prediction commit is not bound to the authenticated model manifest"
            )
    rows: list[evaluator.StructuralSceneCounts] = []
    for image in evaluator.CALIBRATION_IDS:
        fold = next(key for key, ids in evaluator.OOF_FOLDS.items() if image in ids)
        input_payload, _raw, _tiles, _spatial, result = _recompute_candidate_pool(
            image,
            ledger_sha256,
            ledger["run_contract_sha256"],
            ledger["upstream"]["label_free_input_projection"]["records_sha256"],
        )
        feature_path, _feature_manifest_path_value = _feature_paths(image)
        feature_manifest = feature_manifests[image]
        candidate_pool_provenance_ok = bool(
            feature_manifest["candidate_pool_input_sha256"]
            == input_payload["source_scene_contract_sha256"]
            and feature_manifest["input_manifest"]["sha256"]
            == _sha256_file(_input_manifest_path(image))
            and feature_manifest["hypotheses"] == len(result.hypotheses)
            and verified_set.commits[fold].feature_sha256[image]
            == feature_manifest["feature_file"]["sha256"]
        )
        rows.append(
            evaluator.evaluate_committed_structural_scene(
                verified_set.commits[fold],
                image=image,
                result=result,
                feature_path=feature_path,
                permutation=_load_projected_permutation(ledger, image),
                candidate_pool_provenance_ok=candidate_pool_provenance_ok,
            )
        )
    summary = evaluator.summarize_structural(rows)
    decision = evaluator.structural_decision(summary)
    payload = {
        "schema": STRUCTURAL_REPORT_SCHEMA,
        "status": "complete",
        "stage": decision["stage"],
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "fold_commit_sha256": {
            str(fold): _sha256_file(path) for fold, path in commit_paths.items()
        },
        "rows": [asdict(row) for row in rows],
        "summary": summary,
        "decision": decision,
        "staged_board_ssim_nlm": "sealed_not_run",
        "e25_opened": False,
    }
    report = _require_storage(STRUCTURAL_REPORT, label="structural report")
    report_bytes = _canonical_json_bytes(payload)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path,
        additional_total_bytes=(0 if report.exists() else len(report_bytes)),
    )
    evaluator._atomic_write_create_or_verify(report, report_bytes)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    return payload


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER", "LIGHTGBM_TMPDIR"):
        environment[key] = str(_RUNTIME_DIR)
    environment["PYTHONPYCACHEPREFIX"] = str(_PYCACHE_DIR)
    return environment


def orchestrate(ledger_path: Path, ledger_sha256: str) -> dict[str, Any]:
    ledger = verify_preflight_ledger(ledger_path, ledger_sha256)
    if ORCHESTRATION_RECEIPT_PATH.is_file():
        existing = _load_canonical_json(
            ORCHESTRATION_RECEIPT_PATH, label="OOF orchestration receipt"
        )
        existing_resource = existing.get("resource")
        expected_checks = (
            {
                "oof_cpu_at_most_8h": float(
                    existing_resource["child_process_cpu_seconds"]
                )
                <= evaluator.OOF_CPU_SECONDS_MAX,
                "peak_rss_at_most_16gib": int(
                    existing_resource["maximum_child_peak_rss_bytes"]
                )
                <= evaluator.PEAK_RAM_BYTES_MAX,
                "aggregate_artifacts_at_most_8gib": True,
            }
            if type(existing_resource) is dict
            and set(existing_resource)
            == {
                "child_process_cpu_seconds",
                "maximum_child_peak_rss_bytes",
                "cpu_seconds_max",
                "peak_rss_bytes_max",
            }
            else {}
        )
        if (
            set(existing)
            != {
                "schema",
                "status",
                "ledger_sha256",
                "run_contract_sha256",
                "canary_gate_sha256",
                "structural_report_sha256",
                "resource",
                "checks",
            }
            or existing["schema"] != ORCHESTRATION_RECEIPT_SCHEMA
            or existing["status"] != "pass"
            or existing["ledger_sha256"] != ledger_sha256
            or existing["run_contract_sha256"] != ledger["run_contract_sha256"]
            or existing["canary_gate_sha256"] != _sha256_file(CANARY_GATE_PATH)
            or existing["structural_report_sha256"]
            != _sha256_file(STRUCTURAL_REPORT)
            or type(existing_resource) is not dict
            or existing_resource.get("cpu_seconds_max")
            != evaluator.OOF_CPU_SECONDS_MAX
            or existing_resource.get("peak_rss_bytes_max")
            != evaluator.PEAK_RAM_BYTES_MAX
            or existing["checks"] != expected_checks
            or not expected_checks
            or not all(expected_checks.values())
        ):
            raise E24RunnerError("OOF orchestration receipt identity drifted")
        enforce_aggregate_artifact_caps(ledger_path=ledger_path)
        return dict(existing["resource"])
    runner = Path(__file__).resolve()
    total_cpu_seconds = 0.0
    maximum_peak_rss = 0

    def invoke(*arguments: str) -> dict[str, Any]:
        nonlocal total_cpu_seconds, maximum_peak_rss
        command = [
            sys.executable,
            str(runner),
            *arguments,
            "--ledger",
            str(ledger_path.resolve()),
            "--ledger-sha256",
            ledger_sha256,
        ]
        completed = subprocess.run(
            command,
            env=_subprocess_environment(),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise E24RunnerError(
                f"worker failed ({completed.returncode}): {arguments}; {detail}"
            )
        try:
            records = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.strip().startswith("{")
            ]
            output = records[-1]
            resource = output["resource"]
            _enforce_process_resource_caps(resource)
        except Exception as exc:
            raise E24RunnerError(
                f"worker emitted no valid resource receipt: {arguments}"
            ) from exc
        total_cpu_seconds += float(resource["process_cpu_seconds"])
        maximum_peak_rss = max(maximum_peak_rss, int(resource["peak_rss_bytes"]))
        if total_cpu_seconds > evaluator.OOF_CPU_SECONDS_MAX:
            raise E24RunnerError("cumulative E24 run exceeded the frozen 8 CPU-hour cap")
        if maximum_peak_rss > evaluator.PEAK_RAM_BYTES_MAX:
            raise E24RunnerError("an E24 worker exceeded the frozen 16 GiB RAM cap")
        return output

    invoke("prepare-tile-bytes", "--image", str(CANARY_IMAGE))
    invoke("prepare-inputs", "--image", str(CANARY_IMAGE))
    invoke("feature-worker", "--image", str(CANARY_IMAGE))
    for image in evaluator.CALIBRATION_IDS:
        if image == CANARY_IMAGE:
            continue
        invoke("prepare-tile-bytes", "--image", str(image))
        invoke("prepare-inputs", "--image", str(image))
        invoke("feature-worker", "--image", str(image))
    for fold in evaluator.OOF_FOLDS:
        invoke("prepare-fold-labels", "--fold", str(fold))
        invoke("train-fold", "--fold", str(fold))
        invoke("predict-fold", "--fold", str(fold))
    invoke("structural-eval")
    resource = {
        "child_process_cpu_seconds": total_cpu_seconds,
        "maximum_child_peak_rss_bytes": maximum_peak_rss,
        "cpu_seconds_max": evaluator.OOF_CPU_SECONDS_MAX,
        "peak_rss_bytes_max": evaluator.PEAK_RAM_BYTES_MAX,
    }
    receipt = {
        "schema": ORCHESTRATION_RECEIPT_SCHEMA,
        "status": "pass",
        "ledger_sha256": ledger_sha256,
        "run_contract_sha256": ledger["run_contract_sha256"],
        "canary_gate_sha256": _sha256_file(CANARY_GATE_PATH),
        "structural_report_sha256": _sha256_file(STRUCTURAL_REPORT),
        "resource": resource,
        "checks": {
            "oof_cpu_at_most_8h": total_cpu_seconds
            <= evaluator.OOF_CPU_SECONDS_MAX,
            "peak_rss_at_most_16gib": maximum_peak_rss
            <= evaluator.PEAK_RAM_BYTES_MAX,
            "aggregate_artifacts_at_most_8gib": True,
        },
    }
    receipt_bytes = _canonical_json_bytes(receipt)
    enforce_aggregate_artifact_caps(
        ledger_path=ledger_path, additional_total_bytes=len(receipt_bytes)
    )
    evaluator._atomic_write_create(ORCHESTRATION_RECEIPT_PATH, receipt_bytes)
    enforce_aggregate_artifact_caps(ledger_path=ledger_path)
    return resource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen process-separated E24 CRS-v1 runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    for mode in (
        "prepare-tile-bytes",
        "prepare-inputs",
        "feature-worker",
        "prepare-fold-labels",
        "train-fold",
        "predict-fold",
        "structural-eval",
        "orchestrate",
    ):
        command = subparsers.add_parser(mode)
        command.add_argument("--ledger", type=Path, required=True)
        command.add_argument("--ledger-sha256", required=True)
        if mode in {"prepare-tile-bytes", "prepare-inputs", "feature-worker"}:
            command.add_argument("--image", type=int, required=True)
        if mode in {"prepare-fold-labels", "train-fold", "predict-fold"}:
            command.add_argument("--fold", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    args = build_parser().parse_args(argv)
    if args.mode == "preflight":
        payload = write_preflight_ledger(args.ledger)
        output = {
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": _sha256_file(args.ledger.resolve()),
            "run_contract_sha256": payload["run_contract_sha256"],
            "target_artifacts_created": False,
            "metrics_opened": False,
        }
    elif args.mode == "prepare-tile-bytes":
        prepare_upstream_tile_bytes(args.image, args.ledger, args.ledger_sha256)
        output = {
            "mode": args.mode,
            "image": args.image,
            "status": "complete",
            "output_capability": ["tiles_uint8"],
            "metrics_opened": False,
        }
    elif args.mode == "prepare-inputs":
        prepare_label_free_input(args.image, args.ledger, args.ledger_sha256)
        output = {
            "mode": args.mode,
            "image": args.image,
            "status": "complete",
            "metrics_opened": False,
        }
    elif args.mode == "feature-worker":
        receipt = run_feature_worker(args.image, args.ledger, args.ledger_sha256)
        output = {
            "mode": args.mode,
            "image": args.image,
            "status": "complete",
            "resource": receipt["resource"],
        }
    elif args.mode == "prepare-fold-labels":
        prepare_fold_train_labels(args.fold, args.ledger, args.ledger_sha256)
        output = {"mode": args.mode, "fold": args.fold, "status": "complete"}
    elif args.mode == "train-fold":
        train_fold_model(args.fold, args.ledger, args.ledger_sha256)
        output = {"mode": args.mode, "fold": args.fold, "status": "complete"}
    elif args.mode == "predict-fold":
        predict_commit_fold(args.fold, args.ledger, args.ledger_sha256)
        output = {"mode": args.mode, "fold": args.fold, "status": "complete"}
    elif args.mode == "structural-eval":
        result = run_structural_evaluation(args.ledger, args.ledger_sha256)
        output = {
            "mode": args.mode,
            "stage": result["stage"],
            "report": str(STRUCTURAL_REPORT.resolve()),
            "staged_board_ssim_nlm": "sealed_not_run",
        }
    elif args.mode == "orchestrate":
        child_resource = orchestrate(args.ledger, args.ledger_sha256)
        output = {
            "mode": args.mode,
            "status": "complete",
            "child_resource": child_resource,
        }
    else:  # pragma: no cover - argparse exhaustiveness
        raise AssertionError(args.mode)
    if args.mode != "preflight" and "resource" not in output:
        resource = _resource_snapshot(started_wall, started_cpu)
        _enforce_process_resource_caps(resource)
        output["resource"] = resource
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
