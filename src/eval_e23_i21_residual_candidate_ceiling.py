"""Frozen E23 I21-residual K64 candidate-availability ceiling.

The only changed candidate source relative to E22 is the frozen directional
edge head in the byte-pinned I21 checkpoint.  The model sees the corrupted,
upright tile bag only.  Labels are opened after the complete combined pool has
returned and passed an independent label-free replay.

This module deliberately constructs no board and computes no image metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The executable entry point must not create local ``src/__pycache__`` files.
# A caller may provide an E:-resident prefix; otherwise imports following this
# assignment use the frozen E23 location.  (The __main__ module itself is not
# cached by CPython.)
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(Path("E:/pazzle_work/posegraph_e23/pycache"))

import numpy as np
import scipy
import skimage
import torch

import e21_posegraph_candidate_oracle as e21_pose
import e22_rcce4_candidate_oracle as e22_core
import e23_i21_residual_candidate_oracle as e23_core
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
import eval_e22_rcce4_candidate_ceiling as e22_eval
import positional_ddpm


class E23ContractError(RuntimeError):
    """The frozen E23 protocol, lineage, cache, or oracle algebra drifted."""


class E23ScientificGuardFailure(E23ContractError):
    """A predeclared scientific cost guard killed E23 without execution error."""

    def __init__(
        self,
        *,
        image: int,
        guard: str,
        observed: int,
        maximum: int,
        phase: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.payload = {
            "image": int(image),
            "guard": str(guard),
            "observed": int(observed),
            "maximum": int(maximum),
            "phase": str(phase),
            "evidence": _jsonable(dict(evidence or {})),
        }
        super().__init__(
            f"scientific guard {guard} failed for image {image}: "
            f"{observed} > {maximum} ({phase})"
        )


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e23-i21-residual-k64-candidate-ceiling-report-v1"
EXPERIMENT = "e23_i21_residual_k64_candidate_ceiling_v1"
CACHE_SCHEMA = "pazzle-e23-spatial-logits-cpu-f32-cache-v1"

EXPECTED_E22_REPORT_SHA256 = (
    "a594bdd64a8b786b261175f3d6f071f6afe91c7ede92a33b0d7e9ac9edf30281"
)
EXPECTED_E22_RUN_CONTRACT_SHA256 = (
    "55398bc0a268cf23394fe18bab5238735d9f0d68b0651c5ea9365b9a3fc150e2"
)
EXPECTED_E22_PROTOCOL_SHA256 = (
    "9956030b0e16797f2fd7588c58d23c04a4d828c1f6fabd10eda42b48757634f9"
)
EXPECTED_E22_STAGE = "kill_existing_affinity_full_union_generator"

EXPECTED_CHECKPOINT_SIZE = 29_677_382
EXPECTED_CHECKPOINT_SHA256 = (
    "54b13fa3bc594ca8739cb948c68a3725aa29b34bcc8406f94fd2a332db3992c1"
)
EXPECTED_CHECKPOINT_STEP = 6000
EXPECTED_MODEL_ARGS: dict[str, int] = {
    "side": 24,
    "tile_dim": 128,
    "d_model": 192,
    "layers": 4,
    "heads": 6,
    "diffusion_steps": 300,
}
EXPECTED_POSITIONAL_SOURCE_SHA256 = (
    "a41c8abfb9a47954fcb4d500812b2fff62f797109de7b0488706729fe0ecfbbf"
)
EXPECTED_ALIGNMENT_SOURCE_SHA256 = (
    "564b879c892c4bac7cb93d02a7b7cc095e030bf7e8c91b7da281bc73131feda4"
)
EXPECTED_CONFIG_SOURCE_SHA256 = (
    "824165ab03dbf3171aa3a2e8817f084058ecf9bbd4192eed3acbbe0bf73e0a83"
)
EXPECTED_E22_RUNTIME_PROVENANCE = dict(e22_eval.EXPECTED_RUNTIME_PROVENANCE)
EXPECTED_CPU_RUNTIME_CONFIGURATION: dict[str, Any] = {
    "device": "cpu",
    "dtype": "float32",
    "evaluation_mode": True,
    "inference_mode": True,
    "deterministic_algorithms": True,
    "mkldnn_enabled": False,
    "torch_intraop_threads": 1,
    "torch_interop_threads": 1,
}

NUM_DIRECTIONS = 4
SPATIAL_K = 64
SPATIAL_LOGIT_VALUES = NUM_DIRECTIONS * e12.NFRAG * e12.NFRAG
SPATIAL_SELECTIONS = NUM_DIRECTIONS * e12.NFRAG * SPATIAL_K
MAX_DIRECTED_MEMBERSHIPS = e12.NFRAG * 128
MAX_UNORDERED_PAIRS = e12.NFRAG * (e12.NFRAG - 1) // 2
MAX_NEW_SPATIAL_PAIRS = SPATIAL_SELECTIONS
MAX_NEW_LITERAL_CLAIMS = 4 * MAX_NEW_SPATIAL_PAIRS
MAX_COMBINED_LITERAL_CLAIMS = 4 * MAX_UNORDERED_PAIRS
CACHE_PAYLOAD_NBYTES = SPATIAL_LOGIT_VALUES * np.dtype(np.float32).itemsize
CACHE_METADATA_MAX_BYTES = 64 * 1024
CACHE_NPY_HEADER_BYTES = 128
CACHE_NPY_FILE_BYTES = CACHE_NPY_HEADER_BYTES + CACHE_PAYLOAD_NBYTES

DECISION_RULE: dict[str, float | int] = {
    "completed_scenes": 8,
    "emitters_each": 576,
    "exact_e22_prefix_replay_scenes": 8,
    "provenance_replay_scenes": 8,
    "all_bounds_scenes": 8,
    "true_relation_scenes": 8,
    "legal_origin_scenes": 8,
    "positive_eligible_denominator_scenes": 8,
    "incremental_eligible_hit_scenes": 8,
    "null_complete_bounds_survival_scenes": 8,
    "nonzero_null_efficiency_scenes": 8,
    "exact_postfilter_survival_scenes": 8,
    "mean_eligible_contact_recall_min": 0.90,
    "worst_eligible_contact_recall_min": 0.80,
    "mean_exact_connected_coverage_min": 0.30,
    "worst_exact_connected_coverage_min": 0.20,
    "mean_selected_cycle_rank_ratio_min": 0.05,
    "worst_selected_cycle_rank_ratio_min": 0.01,
    "mean_spatial_minus_null_combined_recall_min": 0.020,
    "strict_spatial_recall_win_scenes_min": 6,
    "mean_incremental_hit_efficiency_ratio_min": 1.10,
    "spatial_new_pairs_max_each": 100_000,
    "spatial_geometry_valid_hypotheses_max_each": 450_000,
}

E23_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e23-i21-residual-k64-candidate-ceiling-v1",
    "role": "CPU_label_after_complete_combined_core_discovery_ceiling",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "authorization": {
        "e22_report_sha256": EXPECTED_E22_REPORT_SHA256,
        "e22_run_contract_sha256": EXPECTED_E22_RUN_CONTRACT_SHA256,
        "e22_protocol_sha256": EXPECTED_E22_PROTOCOL_SHA256,
        "required_status": "complete",
        "required_stage": EXPECTED_E22_STAGE,
        "e22_failure_scope": "eligible_pair_recall_only",
    },
    "checkpoint": {
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "size_bytes": EXPECTED_CHECKPOINT_SIZE,
        "step": EXPECTED_CHECKPOINT_STEP,
        "model_args": dict(EXPECTED_MODEL_ARGS),
        "model_dependency_sha256": {
            "positional_ddpm.py": EXPECTED_POSITIONAL_SOURCE_SHA256,
            "eval_paired_alignment.py": EXPECTED_ALIGNMENT_SOURCE_SHA256,
            "config.py": EXPECTED_CONFIG_SOURCE_SHA256,
        },
        "load": "torch_weights_only_true_map_location_cpu",
    },
    "inference": {
        "input": "exact_E12_corrupted_upright_uint8_tile_bag",
        "conversion": "one_contiguous_NCHW_float32_div255",
        "method_order": ["encode_tiles", "directional_edge_scores"],
        "device": "cpu",
        "dtype": "float32",
        "evaluation_mode": True,
        "autocast": False,
        "sampling": False,
        "coordinate_prediction": False,
        "denoising": False,
        "training": False,
        "gpu": False,
    },
    "core_boundary": {
        "arguments": ["candidate_ids", "raw_logits", "spatial_logits"],
        "candidate_ids": "contiguous_int64_576x128",
        "raw_logits": "contiguous_float32_4x576x128_UDLR",
        "spatial_logits": "contiguous_finite_float32_4x576x576_UDLR",
        "forbidden_inputs": [
            "labels",
            "permutation",
            "clean_target",
            "target_pixels",
            "board",
        ],
    },
    "residual_selection": {
        "directions": ["U", "D", "L", "R"],
        "exclude": ["self", "every_existing_E22_unordered_pair"],
        "k_per_anchor_direction": SPATIAL_K,
        "rank": "score_desc_then_tile_id_asc",
        "threshold": False,
        "direction_role": "nomination_metadata_only_not_physical_side",
        "deduplicate": "canonical_unordered_OR",
        "combined_order": "exact_E22_prefix_then_new_pairs_lexicographic",
        "score_fusion_or_rerank": False,
    },
    "pair_lift": {
        "new_pair_variants": ["a_b_R", "b_a_R", "a_b_D", "b_a_D"],
        "tile_rotation": False,
        "reflection": False,
        "same_component_claims": "removed",
        "signed_offset_grouping": "unchanged_E22_exact_no_offset_collapse",
        "geometry_filter": "unchanged_E22_adjacency_collision_24x24_span",
        "post_union_truncation": False,
    },
    "bounds": {
        "spatial_logit_values_exact": SPATIAL_LOGIT_VALUES,
        "spatial_selections_exact": SPATIAL_SELECTIONS,
        "directed_memberships_max": MAX_DIRECTED_MEMBERSHIPS,
        "base_pairs_max": MAX_DIRECTED_MEMBERSHIPS,
        "new_pairs_max_formula": "min(147456,165600-base_pairs)",
        "combined_pairs_max": MAX_UNORDERED_PAIRS,
        "new_literal_claims_exact_four_per_new_pair": True,
        "new_literal_claims_max": MAX_NEW_LITERAL_CLAIMS,
        "combined_literal_claims_exact_four_per_pair": True,
        "combined_literal_claims_max": MAX_COMBINED_LITERAL_CLAIMS,
        "cross_claims_relations_hypotheses_max": MAX_COMBINED_LITERAL_CLAIMS,
        "truncate": False,
    },
    "cache": {
        "schema": CACHE_SCHEMA,
        "format": "atomic_deterministic_npy_plus_canonical_json",
        "payload": "label_free_contiguous_finite_float32_spatial_logits_only",
        "key": ["tile_bytes", "checkpoint", "model_code", "runtime", "inference"],
        "hit_validation": [
            "metadata_exact",
            "file_bytes_exact",
            "file_sha256_exact",
            "array_sha256_exact",
            "shape_dtype_order_finite_exact",
        ],
        "drive": "E",
        "existing_cache_policy": "always_recompute_checkpoint_logits_and_byte_compare",
    },
    "measurement": {
        "labels": "first_after_complete_core_and_independent_label_free_validation",
        "definitions": "exact_E22_purity_denominator_relation_DSU_geometry_selection",
        "required_unchanged": [
            "component_digest",
            "eligible_denominator_digest",
            "E22_pair_prefix",
            "E22_eligible_hits",
        ],
        "incremental_hit": "eligible_true_seam_in_new_spatial_pairs_not_base_pairs",
        "postfilter_survival": "exact_physical_seam_in_geometry_valid_hypothesis",
    },
    "matched_budget_null": {
        "record": "E23-hash-null-v1|tiles_sha256|anchor|direction|target",
        "encoding": "ASCII_no_newline_lowercase_hex_unpadded_decimal",
        "directions": "0_1_2_3_equal_U_D_L_R",
        "order": "full_32_digest_bytes_ascending_then_target_id_collision_tie_break",
        "logits": "rank_0_through_575_gets_exact_float32_575_minus_rank",
        "same_core": True,
        "same_K64_RCCE4_geometry": True,
        "uses_permutation_target_checkpoint_spatial_logits_or_labels": False,
        "seed_or_rule_sweep": False,
    },
    "comparative_decision": {
        "mean_spatial_minus_null_combined_recall_min": 0.020,
        "strict_spatial_recall_win_scenes_min": 6,
        "mean_incremental_hit_efficiency_ratio_min": 1.10,
        "zero_null_efficiency_denominator_allowed": False,
        "spatial_new_pairs_max_each": 100_000,
        "spatial_combined_geometry_valid_hypotheses_max_each": 450_000,
    },
    "decision": dict(DECISION_RULE),
    "routing": {
        "pass": "open_separately_frozen_source_group_disjoint_confirmation",
        "fail": "close_exact_I21_residual_K64_without_sweep",
        "training_authorized": False,
    },
    "excluded": [
        "K32",
        "K128",
        "alternate_directional_interpretation",
        "threshold",
        "alpha",
        "score_fusion",
        "checkpoint_selection",
        "geometry_filter_change",
        "cap_change",
        "clean_pixels",
        "labels_in_core",
        "board",
        "SSIM",
        "NLM",
        "placement",
        "neighbour",
        "GPU",
        "diffusion_sampling",
        "denoising",
        "target_submission_data",
        "alternate_null",
        "null_seed_sweep",
    ],
    "runtime_provenance": EXPECTED_E22_RUNTIME_PROVENANCE,
    "runtime_configuration": EXPECTED_CPU_RUNTIME_CONFIGURATION,
}

DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_E22_REPORT = Path(
    "E:/pazzle_work/posegraph_e22/cc96_all_emitter_full_union_candidate_ceiling_v1.json"
)
DEFAULT_CHECKPOINT = Path(
    "E:/pazzle_work/positional_ddpm/positional_ddpm_train_latest.pt"
)
DEFAULT_SPATIAL_CACHE_DIR = Path(
    "E:/pazzle_work/posegraph_e23/spatial_logits_cpu_f32_v1"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/posegraph_e23/cc96_i21_residual_k64_candidate_ceiling_v1.json"
)


@dataclass(frozen=True)
class E23Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e22_report: Path
    checkpoint: Path
    spatial_cache_dir: Path
    report: Path


@dataclass(frozen=True)
class SpatialCacheRecord:
    key: str
    array_path: Path
    metadata_path: Path
    array_file_sha256: str
    array_file_bytes: int
    array_sha256: str
    hit: bool
    verified_by_recompute: bool


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise E23ContractError(f"{label} must be an integer")
    return int(value)


def _finite(
    value: object,
    *,
    label: str,
    minimum: float = -float("inf"),
    maximum: float = float("inf"),
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise E23ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise E23ContractError(f"{label} is non-finite or outside bounds")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E23ContractError(f"{label} must reside on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E23ContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E23ContractError(f"{label} is not a JSON object")
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="JSON output")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def _atomic_write_npy(path: Path, values: np.ndarray) -> None:
    resolved = _require_e_drive(path, label="spatial-logit cache array")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.lib.format.write_array(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise E23ContractError("payload contains a non-finite float")
        return value
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    raise E23ContractError(f"payload contains unsupported type {type(value)}")


def _stream_digest(values: Sequence[Any] | Any) -> str:
    digest = hashlib.sha256(b"pazzle-e23-stream-v1\0")
    count = 0
    for value in values:
        encoded = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def _check_forbidden_payload_keys(value: Any) -> None:
    forbidden = {
        "board",
        "canvas",
        "placement",
        "neighbour",
        "ssim",
        "nlm",
        "target",
        "target_uint8",
        "target_pixels",
        "ground_truth",
        "permutation",
        "clean_pixels",
        "rotation",
        "reflection",
        "spatial_logits",
        "raw_logits",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                raise E23ContractError(f"core payload contains forbidden key {key}")
            _check_forbidden_payload_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_forbidden_payload_keys(item)


def _base_runtime_provenance() -> dict[str, str]:
    try:
        observed = dict(e22_eval._runtime_provenance())
    except Exception as exc:
        raise E23ContractError(f"cannot verify E22 package runtime: {exc}") from exc
    if observed != EXPECTED_E22_RUNTIME_PROVENANCE:
        raise E23ContractError(
            "E23 package runtime drifted: "
            f"expected {EXPECTED_E22_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _configure_frozen_cpu_runtime() -> dict[str, Any]:
    """Idempotently configure and verify the frozen CPU inference manifest."""

    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    except Exception as exc:
        raise E23ContractError(f"cannot configure frozen CPU runtime: {exc}") from exc
    observed: dict[str, Any] = {
        "device": "cpu",
        "dtype": "float32",
        "evaluation_mode": True,
        "inference_mode": True,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }
    if observed != EXPECTED_CPU_RUNTIME_CONFIGURATION:
        raise E23ContractError(
            "frozen CPU runtime configuration drifted: "
            f"expected {EXPECTED_CPU_RUNTIME_CONFIGURATION}, got {observed}"
        )
    return observed


def _runtime_provenance() -> dict[str, Any]:
    return {
        "packages": _base_runtime_provenance(),
        "configuration": _configure_frozen_cpu_runtime(),
    }


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e21_posegraph_candidate_oracle.py": source / "e21_posegraph_candidate_oracle.py",
        "e22_rcce4_candidate_oracle.py": source / "e22_rcce4_candidate_oracle.py",
        "e23_i21_residual_candidate_oracle.py": source
        / "e23_i21_residual_candidate_oracle.py",
        "config.py": source / "config.py",
        "eval_paired_alignment.py": source / "eval_paired_alignment.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_e22_rcce4_candidate_ceiling.py": source
        / "eval_e22_rcce4_candidate_ceiling.py",
        "eval_e23_i21_residual_candidate_ceiling.py": Path(__file__).resolve(),
        "positional_ddpm.py": source / "positional_ddpm.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise E23ContractError("E23 source file is missing: " + ", ".join(missing))
    result = {name: e12.sha256_file(path) for name, path in sorted(paths.items())}
    if result["positional_ddpm.py"] != EXPECTED_POSITIONAL_SOURCE_SHA256:
        raise E23ContractError("frozen positional_ddpm.py bytes drifted")
    if result["eval_paired_alignment.py"] != EXPECTED_ALIGNMENT_SOURCE_SHA256:
        raise E23ContractError("frozen eval_paired_alignment.py bytes drifted")
    if result["config.py"] != EXPECTED_CONFIG_SOURCE_SHA256:
        raise E23ContractError("frozen config.py bytes drifted")
    return result


def _verify_e22_kill(path: Path) -> Mapping[str, Any]:
    resolved = _require_e_drive(path, label="E22 authorization report")
    if not resolved.is_file():
        raise E23ContractError(f"E22 authorization report is missing: {resolved}")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E22_REPORT_SHA256:
        raise E23ContractError(
            "E22 report SHA256 mismatch: "
            f"expected {EXPECTED_E22_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E22 authorization report")
    rows = report.get("rows")
    contract = report.get("run_contract")
    if (
        _integer(report.get("schema_version"), label="E22 schema version")
        != e22_eval.SCHEMA_VERSION
        or report.get("schema") != e22_eval.REPORT_SCHEMA
        or report.get("experiment") != e22_eval.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("stage") != EXPECTED_E22_STAGE
        or report.get("protocol") != e22_eval.E22_PROTOCOL
        or report.get("protocol_sha256") != EXPECTED_E22_PROTOCOL_SHA256
        or e12.canonical_digest(report.get("protocol"))
        != EXPECTED_E22_PROTOCOL_SHA256
        or report.get("run_contract_sha256")
        != EXPECTED_E22_RUN_CONTRACT_SHA256
        or not isinstance(contract, Mapping)
        or e12.canonical_digest(contract) != EXPECTED_E22_RUN_CONTRACT_SHA256
        or report.get("completed_images") != list(e12.CALIBRATION_IDS)
        or not isinstance(rows, list)
        or len(rows) != len(e12.CALIBRATION_IDS)
    ):
        raise E23ContractError("E22 authorization contract drifted")
    if [row.get("image") for row in rows if isinstance(row, Mapping)] != list(
        e12.CALIBRATION_IDS
    ):
        raise E23ContractError("E22 authorization rows are incomplete or reordered")
    for row in rows:
        if not isinstance(row, Mapping):
            raise E23ContractError("E22 authorization row is malformed")
        core = row.get("core")
        oracle = row.get("oracle")
        if (
            not isinstance(core, Mapping)
            or not isinstance(oracle, Mapping)
            or row.get("core_sha256") != e12.canonical_digest(core)
            or row.get("oracle_sha256") != e12.canonical_digest(oracle)
        ):
            raise E23ContractError("E22 authorization row digest drifted")
    try:
        expected_summary = e22_eval.summarize(rows)
        expected_decision = e22_eval.decision(expected_summary)
    except Exception as exc:
        raise E23ContractError(f"E22 terminal payload is malformed: {exc}") from exc
    if (
        report.get("summary") != expected_summary
        or report.get("decision") != expected_decision
        or expected_decision.get("passed") is not False
        or expected_decision.get("status") != EXPECTED_E22_STAGE
    ):
        raise E23ContractError("E22 recall-only KILL decision drifted")
    frozen_sources = contract.get("source_provenance")
    if not isinstance(frozen_sources, Mapping):
        raise E23ContractError("E22 source provenance is malformed")
    source = Path(__file__).resolve().parent
    for name, expected in frozen_sources.items():
        if not isinstance(name, str) or Path(name).name != name or not _is_sha256(expected):
            raise E23ContractError("E22 source provenance entry is malformed")
        source_path = source / name
        if not source_path.is_file() or e12.sha256_file(source_path) != expected:
            raise E23ContractError(f"source shared with E22 drifted for {name}")
    if contract.get("runtime_provenance") != _base_runtime_provenance():
        raise E23ContractError("E22-to-E23 runtime provenance drifted")
    return report


def _checkpoint_record(path: Path) -> dict[str, Any]:
    resolved = _require_e_drive(path, label="I21 checkpoint")
    if not resolved.is_file():
        raise E23ContractError(f"I21 checkpoint is missing: {resolved}")
    size = resolved.stat().st_size
    digest = e12.sha256_file(resolved)
    source_path = Path(positional_ddpm.__file__).resolve()
    source_dir = source_path.parent
    dependency_paths = {
        "positional_ddpm.py": source_path,
        "eval_paired_alignment.py": source_dir / "eval_paired_alignment.py",
        "config.py": source_dir / "config.py",
    }
    expected_dependencies = {
        "positional_ddpm.py": EXPECTED_POSITIONAL_SOURCE_SHA256,
        "eval_paired_alignment.py": EXPECTED_ALIGNMENT_SOURCE_SHA256,
        "config.py": EXPECTED_CONFIG_SOURCE_SHA256,
    }
    if any(not value.is_file() for value in dependency_paths.values()):
        raise E23ContractError("I21 model dependency is missing")
    dependency_digests = {
        name: e12.sha256_file(value) for name, value in dependency_paths.items()
    }
    if size != EXPECTED_CHECKPOINT_SIZE or digest != EXPECTED_CHECKPOINT_SHA256:
        raise E23ContractError("I21 checkpoint byte identity drifted")
    if dependency_digests != expected_dependencies:
        raise E23ContractError("I21 model dependency byte identity drifted")
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": digest,
        "step": EXPECTED_CHECKPOINT_STEP,
        "model_args": dict(EXPECTED_MODEL_ARGS),
        "model_dependencies": {
            name: {
                "path": str(dependency_paths[name]),
                "sha256": dependency_digests[name],
            }
            for name in sorted(dependency_paths)
        },
    }


def load_frozen_i21_model(path: Path) -> tuple[positional_ddpm.PositionalDDPM, dict[str, Any]]:
    """Authenticate and load the checkpoint without executable pickle objects."""

    _configure_frozen_cpu_runtime()
    record = _checkpoint_record(path)
    try:
        payload = torch.load(record["path"], map_location="cpu", weights_only=True)
    except Exception as exc:
        raise E23ContractError(f"safe I21 checkpoint load failed: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "optimizer",
        "step",
        "metrics",
        "model_args",
    }:
        raise E23ContractError("I21 checkpoint payload fields drifted")
    if (
        _integer(payload.get("step"), label="I21 checkpoint step")
        != EXPECTED_CHECKPOINT_STEP
        or payload.get("model_args") != EXPECTED_MODEL_ARGS
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise E23ContractError("I21 checkpoint step/model arguments drifted")
    model = positional_ddpm.PositionalDDPM(**EXPECTED_MODEL_ARGS)
    try:
        incompatible = model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:
        raise E23ContractError(f"I21 checkpoint state dict drifted: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise E23ContractError("I21 checkpoint state dict is not exact")
    model.to(device=torch.device("cpu"), dtype=torch.float32)
    model.eval()
    for tensor in tuple(model.parameters()) + tuple(model.buffers()):
        if tensor.device.type != "cpu":
            raise E23ContractError("I21 model escaped CPU")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise E23ContractError("I21 model escaped float32")
    del payload
    return model, record


def _validate_tiles_uint8(value: object) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (e12.NFRAG, e12.FS, e12.FS, 3)
        or value.dtype != np.uint8
        or not value.flags.c_contiguous
    ):
        raise E23ContractError("corrupted tiles must be contiguous uint8[576,20,20,3]")
    return value


def infer_spatial_logits(
    tiles_uint8: np.ndarray,
    model: positional_ddpm.PositionalDDPM,
) -> np.ndarray:
    """Run exactly ``encode_tiles -> directional_edge_scores`` on CPU."""

    _configure_frozen_cpu_runtime()
    tiles = _validate_tiles_uint8(tiles_uint8)
    if model.training:
        raise E23ContractError("I21 model must already be in evaluation mode")
    for tensor in tuple(model.parameters()) + tuple(model.buffers()):
        if tensor.device.type != "cpu" or (
            tensor.is_floating_point() and tensor.dtype != torch.float32
        ):
            raise E23ContractError("I21 model must be CPU float32")
    # The uint8 bag is rearranged once, converted once and normalized in-place.
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(device="cpu", dtype=torch.float32).div_(255.0).unsqueeze(0)
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != (1, e12.NFRAG, 3, e12.FS, e12.FS)
        or float(tensor.min()) < 0.0
        or float(tensor.max()) > 1.0
    ):
        raise E23ContractError("I21 model-input conversion drifted")
    try:
        with torch.inference_mode():
            features = model.encode_tiles(tensor)
            output = model.directional_edge_scores(features)
    except Exception as exc:
        raise E23ContractError(f"I21 edge-only inference failed: {exc}") from exc
    if (
        not isinstance(output, torch.Tensor)
        or output.device.type != "cpu"
        or output.dtype != torch.float32
        or tuple(output.shape) != (1, 4, e12.NFRAG, e12.NFRAG)
        or not bool(torch.isfinite(output).all())
    ):
        raise E23ContractError("I21 directional output contract drifted")
    result = np.ascontiguousarray(output[0].detach().numpy(), dtype=np.float32)
    if result.shape != (4, e12.NFRAG, e12.NFRAG) or not np.isfinite(result).all():
        raise E23ContractError("I21 spatial logits are malformed")
    return result


def hash_null_spatial_logits(tiles_sha256: str) -> np.ndarray:
    """Build the one frozen matched-budget, label-free SHA256 null tensor."""

    _configure_frozen_cpu_runtime()
    if not _is_sha256(tiles_sha256) or tiles_sha256 != tiles_sha256.lower():
        raise E23ContractError("hash-null tiles SHA256 must be lowercase 64-hex")
    logits = np.empty(
        (NUM_DIRECTIONS, e12.NFRAG, e12.NFRAG), dtype=np.float32, order="C"
    )
    targets = range(e12.NFRAG)
    for direction in range(NUM_DIRECTIONS):
        for anchor in range(e12.NFRAG):
            keyed = []
            for target in targets:
                record = (
                    f"E23-hash-null-v1|{tiles_sha256}|{anchor:d}|"
                    f"{direction:d}|{target:d}"
                ).encode("ascii")
                keyed.append((hashlib.sha256(record).digest(), target))
            keyed.sort(key=lambda value: (value[0], value[1]))
            if len(keyed) != e12.NFRAG:
                raise E23ContractError("hash-null row length drifted")
            for rank, (_digest, target) in enumerate(keyed):
                logits[direction, anchor, target] = np.float32(e12.NFRAG - 1 - rank)
    _validate_spatial_logits(logits)
    expected_scores = np.arange(e12.NFRAG - 1, -1, -1, dtype=np.float32)
    for direction in range(NUM_DIRECTIONS):
        for anchor in range(e12.NFRAG):
            if not np.array_equal(np.sort(logits[direction, anchor])[::-1], expected_scores):
                raise E23ContractError("hash-null rank logits are not the exact 575..0 set")
    return logits


def _cache_identity(
    *,
    image_id: int,
    validation_name: str,
    tiles_uint8: np.ndarray,
    checkpoint_record: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    tiles = _validate_tiles_uint8(tiles_uint8)
    frozen_runtime = _runtime_provenance()
    if dict(runtime_provenance) != frozen_runtime:
        raise E23ContractError("spatial cache runtime manifest drifted")
    identity = {
        "schema": CACHE_SCHEMA,
        "image": _integer(image_id, label="cache image"),
        "validation_name": str(validation_name),
        "tiles_uint8_sha256": e12.array_sha256(tiles),
        "checkpoint": _jsonable(checkpoint_record),
        "model_dependency_sha256": {
            "positional_ddpm.py": EXPECTED_POSITIONAL_SOURCE_SHA256,
            "eval_paired_alignment.py": EXPECTED_ALIGNMENT_SOURCE_SHA256,
            "config.py": EXPECTED_CONFIG_SOURCE_SHA256,
        },
        "runtime_provenance": frozen_runtime,
        "inference_protocol_sha256": e12.canonical_digest(E23_PROTOCOL["inference"]),
        "direction_order": ["U", "D", "L", "R"],
        "array_contract": {
            "shape": [4, e12.NFRAG, e12.NFRAG],
            "dtype": "float32",
            "order": "C",
            "finite": True,
        },
    }
    _check_forbidden_payload_keys(
        {key: value for key, value in identity.items() if key != "tiles_uint8_sha256"}
    )
    return identity


def _spatial_cache_paths(
    cache_dir: Path, *, image_id: int, identity: Mapping[str, Any]
) -> tuple[str, Path, Path]:
    directory = _require_e_drive(cache_dir, label="spatial-logit cache directory")
    key = e12.canonical_digest(identity)
    stem = f"image_{int(image_id):04d}_{key}"
    return key, directory / f"{stem}.npy", directory / f"{stem}.json"


def _validate_spatial_logits(value: object, *, readonly: bool = False) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (4, e12.NFRAG, e12.NFRAG)
        or value.dtype != np.float32
        or not value.flags.c_contiguous
        or (readonly and value.flags.writeable)
        or not bool(np.isfinite(value).all())
    ):
        suffix = " read-only" if readonly else ""
        raise E23ContractError(
            "spatial_logits must be contiguous finite"
            f"{suffix} float32[4,576,576]"
        )
    if value.size != SPATIAL_LOGIT_VALUES:
        raise E23ContractError("spatial-logit value count drifted")
    return value


def _load_spatial_cache(
    array_path: Path,
    metadata_path: Path,
    *,
    key: str,
    identity: Mapping[str, Any],
) -> tuple[np.ndarray, SpatialCacheRecord]:
    if not array_path.is_file() or not metadata_path.is_file():
        raise E23ContractError("spatial cache is partial or missing")
    metadata_bytes = metadata_path.stat().st_size
    if not 1 <= metadata_bytes <= CACHE_METADATA_MAX_BYTES:
        raise E23ContractError("spatial cache metadata exceeds the absolute size cap")
    metadata = _load_json(metadata_path, label="spatial cache metadata")
    expected_keys = {"schema", "key", "identity", "array"}
    array_record = metadata.get("array")
    if (
        set(metadata) != expected_keys
        or metadata.get("schema") != CACHE_SCHEMA
        or metadata.get("key") != key
        or metadata.get("identity") != identity
        or not isinstance(array_record, Mapping)
        or set(array_record)
        != {
            "filename",
            "format",
            "shape",
            "dtype",
            "order",
            "finite",
            "payload_nbytes",
            "file_bytes",
            "file_sha256",
            "array_sha256",
        }
    ):
        raise E23ContractError("spatial cache metadata drifted")
    if (
        array_record.get("filename") != array_path.name
        or array_record.get("format") != "npy-v1"
        or array_record.get("shape") != [4, e12.NFRAG, e12.NFRAG]
        or array_record.get("dtype") != "float32"
        or array_record.get("order") != "C"
        or array_record.get("finite") is not True
        or _integer(array_record.get("payload_nbytes"), label="cache payload bytes")
        != CACHE_PAYLOAD_NBYTES
    ):
        raise E23ContractError("spatial cache array contract drifted")
    observed_bytes = array_path.stat().st_size
    if observed_bytes != CACHE_NPY_FILE_BYTES:
        raise E23ContractError("spatial cache NPY file exceeds its absolute envelope")
    try:
        with array_path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version != (1, 0):
                raise E23ContractError("spatial cache uses an unsupported NPY version")
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                handle, max_header_size=CACHE_NPY_HEADER_BYTES
            )
            payload_offset = handle.tell()
    except E23ContractError:
        raise
    except Exception as exc:
        raise E23ContractError(f"spatial cache NPY header is malformed: {exc}") from exc
    if (
        tuple(shape) != (4, e12.NFRAG, e12.NFRAG)
        or bool(fortran_order)
        or np.dtype(dtype) != np.dtype(np.float32)
        or payload_offset != CACHE_NPY_HEADER_BYTES
        or payload_offset + CACHE_PAYLOAD_NBYTES != observed_bytes
    ):
        raise E23ContractError("spatial cache NPY header/payload envelope drifted")
    observed_file_sha = e12.sha256_file(array_path)
    if (
        observed_bytes
        != _integer(array_record.get("file_bytes"), label="cache file bytes")
        or observed_file_sha != array_record.get("file_sha256")
        or not _is_sha256(observed_file_sha)
    ):
        raise E23ContractError("spatial cache bytes failed authentication")
    try:
        loaded = np.load(array_path, allow_pickle=False)
    except Exception as exc:
        raise E23ContractError(f"cannot decode spatial cache array: {exc}") from exc
    logits = np.ascontiguousarray(loaded) if isinstance(loaded, np.ndarray) else loaded
    _validate_spatial_logits(logits)
    if not isinstance(loaded, np.ndarray) or not loaded.flags.c_contiguous:
        raise E23ContractError("spatial cache stored a non-C array")
    observed_array_sha = e12.array_sha256(logits)
    if observed_array_sha != array_record.get("array_sha256"):
        raise E23ContractError("spatial cache array payload hash drifted")
    logits.setflags(write=False)
    return logits, SpatialCacheRecord(
        key=key,
        array_path=array_path,
        metadata_path=metadata_path,
        array_file_sha256=observed_file_sha,
        array_file_bytes=observed_bytes,
        array_sha256=observed_array_sha,
        hit=True,
        verified_by_recompute=False,
    )


def load_or_compute_spatial_logits(
    *,
    cache_dir: Path,
    image_id: int,
    validation_name: str,
    tiles_uint8: np.ndarray,
    model: positional_ddpm.PositionalDDPM,
    checkpoint_record: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    force_recompute: bool = False,
    infer: Callable[[np.ndarray, positional_ddpm.PositionalDDPM], np.ndarray]
    = infer_spatial_logits,
) -> tuple[np.ndarray, SpatialCacheRecord]:
    """Load a byte-authenticated cache or atomically write exact CPU logits."""

    tiles = _validate_tiles_uint8(tiles_uint8)
    tiles_digest_before = e12.array_sha256(tiles)
    tiles_writeable_before = bool(tiles.flags.writeable)
    identity = _cache_identity(
        image_id=image_id,
        validation_name=validation_name,
        tiles_uint8=tiles_uint8,
        checkpoint_record=checkpoint_record,
        runtime_provenance=runtime_provenance,
    )
    key, array_path, metadata_path = _spatial_cache_paths(
        cache_dir, image_id=image_id, identity=identity
    )
    existing: tuple[np.ndarray, SpatialCacheRecord] | None = None
    if array_path.exists() or metadata_path.exists():
        existing = _load_spatial_cache(
            array_path, metadata_path, key=key, identity=identity
        )
        if not force_recompute:
            if (
                e12.array_sha256(tiles) != tiles_digest_before
                or bool(tiles.flags.writeable) != tiles_writeable_before
            ):
                raise E23ContractError("spatial cache path mutated the tile input")
            return existing
    computed = infer(tiles_uint8, model)
    if (
        e12.array_sha256(tiles) != tiles_digest_before
        or bool(tiles.flags.writeable) != tiles_writeable_before
    ):
        raise E23ContractError("spatial inference mutated the frozen tile input")
    logits = np.ascontiguousarray(computed, dtype=np.float32)
    if logits is not computed or computed.dtype != np.float32:
        # The inference boundary itself, including injected audit functions,
        # must return the exact frozen representation without coercion.
        raise E23ContractError("spatial inference did not return exact contiguous float32")
    _validate_spatial_logits(logits)
    if existing is not None:
        cached, old_record = existing
        if (
            e12.array_sha256(logits) != old_record.array_sha256
            or not np.array_equal(logits, cached)
        ):
            raise E23ContractError(
                "forced spatial recomputation differs from authenticated cache"
            )
        return cached, SpatialCacheRecord(
            key=old_record.key,
            array_path=old_record.array_path,
            metadata_path=old_record.metadata_path,
            array_file_sha256=old_record.array_file_sha256,
            array_file_bytes=old_record.array_file_bytes,
            array_sha256=old_record.array_sha256,
            hit=True,
            verified_by_recompute=True,
        )
    _atomic_write_npy(array_path, logits)
    file_bytes = array_path.stat().st_size
    file_sha = e12.sha256_file(array_path)
    array_sha = e12.array_sha256(logits)
    metadata = {
        "schema": CACHE_SCHEMA,
        "key": key,
        "identity": identity,
        "array": {
            "filename": array_path.name,
            "format": "npy-v1",
            "shape": [4, e12.NFRAG, e12.NFRAG],
            "dtype": "float32",
            "order": "C",
            "finite": True,
            "payload_nbytes": logits.nbytes,
            "file_bytes": file_bytes,
            "file_sha256": file_sha,
            "array_sha256": array_sha,
        },
    }
    _atomic_write_json(metadata_path, metadata)
    replayed, record = _load_spatial_cache(
        array_path, metadata_path, key=key, identity=identity
    )
    if not np.array_equal(replayed, logits):
        raise E23ContractError("fresh spatial cache replay changed logits")
    return replayed, SpatialCacheRecord(
        key=record.key,
        array_path=record.array_path,
        metadata_path=record.metadata_path,
        array_file_sha256=record.array_file_sha256,
        array_file_bytes=record.array_file_bytes,
        array_sha256=record.array_sha256,
        hit=False,
        verified_by_recompute=False,
    )


def _validate_raw_arrays(
    candidate_ids: object, raw_logits: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return e22_eval._validate_raw_arrays(candidate_ids, raw_logits)
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc


def _validate_components(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return e22_eval._validate_components(result)
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc


def _expected_residual_selection(
    spatial_logits: np.ndarray,
    base_pair_records: Sequence[tuple[int, int, int | None, int | None]],
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    scores = _validate_spatial_logits(spatial_logits)
    excluded: list[set[int]] = [set() for _ in range(e12.NFRAG)]
    for a, b, _a_slot, _b_slot in base_pair_records:
        excluded[a].add(b)
        excluded[b].add(a)
    tile_ids = np.arange(e12.NFRAG, dtype=np.int64)
    selected = np.empty(
        (NUM_DIRECTIONS, e12.NFRAG, SPATIAL_K), dtype=np.int64, order="C"
    )
    nominations: dict[tuple[int, int], int] = {}
    for direction in range(NUM_DIRECTIONS):
        for source in range(e12.NFRAG):
            mask = np.ones(e12.NFRAG, dtype=np.bool_)
            mask[source] = False
            if excluded[source]:
                mask[np.fromiter(sorted(excluded[source]), dtype=np.int64)] = False
            eligible = tile_ids[mask]
            if eligible.size < SPATIAL_K:
                raise E23ContractError("residual K64 has fewer than 64 eligible targets")
            order = np.lexsort((eligible, -scores[direction, source, eligible]))
            chosen = eligible[order[:SPATIAL_K]]
            if chosen.size != SPATIAL_K or np.unique(chosen).size != SPATIAL_K:
                raise E23ContractError("residual K64 selection drifted")
            selected[direction, source] = chosen
            for target_value in chosen.tolist():
                target = int(target_value)
                a, b = (source, target) if source < target else (target, source)
                nominations[(a, b)] = nominations.get((a, b), 0) + 1
    if selected.size != SPATIAL_SELECTIONS or sum(nominations.values()) != SPATIAL_SELECTIONS:
        raise E23ContractError("residual selection accounting drifted")
    return selected, nominations


def preflight_spatial_deployability(
    *,
    image_id: int,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
) -> int:
    """Count spatial residual pairs independently before allocating the core pool."""

    candidate_ids, raw_logits, valid = _validate_raw_arrays(candidate_ids, raw_logits)
    del raw_logits
    expected_base = e22_eval._expected_affinity_pairs(candidate_ids, valid)
    _selected, nominations = _expected_residual_selection(
        _validate_spatial_logits(spatial_logits), expected_base
    )
    count = len(nominations)
    maximum = int(DECISION_RULE["spatial_new_pairs_max_each"])
    if count > maximum:
        raise E23ScientificGuardFailure(
            image=_integer(image_id, label="preflight image"),
            guard="spatial_new_pairs_max_each",
            observed=count,
            maximum=maximum,
            phase="before_combined_core_construction",
            evidence={"spatial_selection_count": SPATIAL_SELECTIONS},
        )
    return count


def _relation_tuple(value: Any, *, label: str) -> tuple[int, int, int, int]:
    try:
        return e22_eval._relation_tuple(value, label=label)
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc


def _component_entries(component: Any) -> tuple[tuple[int, int, int], ...]:
    try:
        return e22_eval._component_entries(component)
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc


def _independent_geometry_reason(
    relation: tuple[int, int, int, int],
    claim_ids: Sequence[int],
    *,
    claims: Sequence[Any],
    components: Sequence[Any],
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> str | None:
    try:
        return e22_eval._independent_geometry_reason(
            relation,
            claim_ids,
            claims=claims,
            components=components,
            owner=owner,
            local_rows=local_rows,
            local_cols=local_cols,
        )
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc


def validate_candidate_pool(
    result: Any,
    *,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
    e22_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently replay residual admission, RCCE-4 grouping and geometry."""

    if not isinstance(result, e23_core.CandidatePoolResult):
        raise E23ContractError("candidate core returned the wrong result type")
    candidate_ids, raw_logits, valid = _validate_raw_arrays(candidate_ids, raw_logits)
    spatial_logits = _validate_spatial_logits(spatial_logits)
    owner, local_rows, local_cols = _validate_components(result)
    try:
        (
            expected_components,
            expected_owner,
            expected_local_rows,
            expected_local_cols,
            expected_nontrivial,
        ) = e22_eval._independent_raw_cc96_partition(candidate_ids, raw_logits, valid)
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc
    if (
        result.components != expected_components
        or not np.array_equal(owner, expected_owner)
        or not np.array_equal(local_rows, expected_local_rows)
        or not np.array_equal(local_cols, expected_local_cols)
        or result.nontrivial_component_ids != expected_nontrivial
    ):
        raise E23ContractError("combined core CC96 differs from independent E22 replay")

    expected_base = e22_eval._expected_affinity_pairs(candidate_ids, valid)
    if (
        not isinstance(result.base_affinity_pairs, tuple)
        or len(result.base_affinity_pairs) != len(expected_base)
    ):
        raise E23ContractError("E22 base affinity-pair prefix is malformed")
    for pair_id, (pair, expected) in enumerate(zip(result.base_affinity_pairs, expected_base)):
        if not isinstance(pair, e22_core.AffinityPair):
            raise E23ContractError("E22 base affinity pair has the wrong type")
        observed = (
            _integer(pair.a, label="base pair a"),
            _integer(pair.b, label="base pair b"),
            None
            if pair.a_to_b_slot is None
            else _integer(pair.a_to_b_slot, label="base a-to-b slot"),
            None
            if pair.b_to_a_slot is None
            else _integer(pair.b_to_a_slot, label="base b-to-a slot"),
        )
        if _integer(pair.pair_id, label="base pair ID") != pair_id or observed != expected:
            raise E23ContractError("exact E22 unordered pair prefix drifted")

    expected_selected, nomination_counts = _expected_residual_selection(
        spatial_logits, expected_base
    )
    selected = result.spatial_selected_ids
    if (
        not isinstance(selected, np.ndarray)
        or selected.shape != (NUM_DIRECTIONS, e12.NFRAG, SPATIAL_K)
        or selected.dtype != np.int64
        or not selected.flags.c_contiguous
        or selected.flags.writeable
        or not np.array_equal(selected, expected_selected)
    ):
        raise E23ContractError("spatial_selected_ids differs from frozen K64 replay")

    expected_spatial_identities = tuple(sorted(nomination_counts))
    base_identities = {(a, b) for a, b, _a_slot, _b_slot in expected_base}
    if base_identities.intersection(expected_spatial_identities):
        raise E23ContractError("new spatial pairs intersect exact E22 pairs")
    if (
        not isinstance(result.spatial_pairs, tuple)
        or len(result.spatial_pairs) != len(expected_spatial_identities)
    ):
        raise E23ContractError("spatial pair inventory is malformed")
    base_count = len(expected_base)
    for offset, (pair, identity) in enumerate(
        zip(result.spatial_pairs, expected_spatial_identities)
    ):
        if (
            not isinstance(pair, e23_core.SpatialPair)
            or _integer(pair.pair_id, label="spatial pair ID") != base_count + offset
            or pair.identity != identity
            or _integer(pair.nomination_count, label="spatial nomination count")
            != nomination_counts[identity]
        ):
            raise E23ContractError("spatial pair canonical OR/metadata drifted")
    if sum(pair.nomination_count for pair in result.spatial_pairs) != SPATIAL_SELECTIONS:
        raise E23ContractError("spatial pair nomination accounting drifted")

    if (
        not isinstance(result.affinity_pairs, tuple)
        or result.affinity_pairs[:base_count] != result.base_affinity_pairs
        or result.affinity_pairs[base_count:] != result.spatial_pairs
        or len(result.affinity_pairs) != base_count + len(result.spatial_pairs)
    ):
        raise E23ContractError("combined pair inventory lost exact prefix/suffix order")
    for pair_id, pair in enumerate(result.affinity_pairs):
        if _integer(pair.pair_id, label="combined pair ID") != pair_id:
            raise E23ContractError("combined pair IDs are not contiguous")

    same_base = sum(int(owner[a]) == int(owner[b]) for a, b, *_ in expected_base)
    cross_base = base_count - same_base
    same_spatial = sum(
        int(owner[a]) == int(owner[b]) for a, b in expected_spatial_identities
    )
    cross_spatial = len(expected_spatial_identities) - same_spatial
    expected_claim_count = 4 * (cross_base + cross_spatial)
    if not isinstance(result.claims, tuple) or len(result.claims) != expected_claim_count:
        raise E23ContractError("combined cross-component claim inventory drifted")

    claim_cursor = 0
    claim_observations = 0
    pair_specs: list[tuple[int, int, int | None, int | None, bool]] = [
        (a, b, a_slot, b_slot, True) for a, b, a_slot, b_slot in expected_base
    ] + [
        (a, b, None, None, False) for a, b in expected_spatial_identities
    ]
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    physical_seams: set[tuple[int, int, int, int]] = set()
    for pair_id, (a, b, a_slot, b_slot, is_base) in enumerate(pair_specs):
        if int(owner[a]) == int(owner[b]):
            continue
        specs = (
            (a, b, 0, 1, a_slot, e22_core.RIGHT, b_slot, e22_core.LEFT),
            (b, a, 0, 1, b_slot, e22_core.RIGHT, a_slot, e22_core.LEFT),
            (a, b, 1, 0, a_slot, e22_core.DOWN, b_slot, e22_core.UP),
            (b, a, 1, 0, b_slot, e22_core.DOWN, a_slot, e22_core.UP),
        )
        for first, second, dy, dx, f_slot, f_dir, r_slot, r_dir in specs:
            if claim_cursor >= len(result.claims):
                raise E23ContractError("combined RCCE-4 claims were truncated")
            claim = result.claims[claim_cursor]
            if not isinstance(claim, e22_core.RCCE4Claim):
                raise E23ContractError("combined RCCE-4 claim has the wrong type")
            seam = (first, second, dy, dx)
            if (
                _integer(claim.claim_id, label="claim ID") != claim_cursor
                or _integer(claim.pair_id, label="claim pair ID") != pair_id
                or _integer(claim.first, label="claim first") != first
                or _integer(claim.second, label="claim second") != second
                or _integer(claim.dy, label="claim dy") != dy
                or _integer(claim.dx, label="claim dx") != dx
                or _integer(claim.first_component, label="claim first component")
                != int(owner[first])
                or _integer(claim.second_component, label="claim second component")
                != int(owner[second])
                or int(owner[first]) == int(owner[second])
                or tuple(claim.physical_seam) != seam
            ):
                raise E23ContractError("literal combined RCCE-4 claim algebra drifted")
            if is_base:
                forward = (
                    None
                    if f_slot is None
                    else (first, second, f_dir, f_slot, float(raw_logits[f_dir, first, f_slot]))
                )
                reverse = (
                    None
                    if r_slot is None
                    else (second, first, r_dir, r_slot, float(raw_logits[r_dir, second, r_slot]))
                )
                try:
                    e22_eval._validate_observation(
                        claim.forward_observation, forward, label="base forward observation"
                    )
                    e22_eval._validate_observation(
                        claim.reverse_observation, reverse, label="base reverse observation"
                    )
                except Exception as exc:
                    raise E23ContractError(str(exc)) from exc
                claim_observations += int(forward is not None) + int(reverse is not None)
            elif claim.forward_observation is not None or claim.reverse_observation is not None:
                raise E23ContractError("spatial direction illegally selected a physical side")
            if seam in physical_seams:
                raise E23ContractError("combined physical seam claim is duplicated")
            physical_seams.add(seam)
            try:
                relation = e22_eval.seam_relation(
                    seam,
                    owner=owner,
                    local_rows=local_rows,
                    local_cols=local_cols,
                )
            except Exception as exc:
                raise E23ContractError(str(exc)) from exc
            grouped.setdefault(relation, []).append(claim_cursor)
            claim_cursor += 1
    if claim_cursor != len(result.claims):
        raise E23ContractError("combined claim inventory contains extra claims")

    expected_relations = tuple(sorted(grouped))
    relations = result.relation_candidates
    if not isinstance(relations, tuple) or len(relations) != len(expected_relations):
        raise E23ContractError("combined relation inventory drifted")
    component_pair_offsets: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for relation_id, (candidate, relation) in enumerate(zip(relations, expected_relations)):
        if (
            not isinstance(candidate, e22_core.RelationCandidate)
            or _integer(candidate.relation_id, label="relation ID") != relation_id
            or _relation_tuple(candidate, label="relation candidate") != relation
            or candidate.claim_ids != tuple(grouped[relation])
        ):
            raise E23ContractError("canonical signed relation grouping drifted")
        component_pair_offsets.setdefault(relation[:2], set()).add(relation[2:])

    hypotheses = result.hypotheses
    rejections = result.geometry_rejections
    if not isinstance(hypotheses, tuple) or not isinstance(rejections, tuple):
        raise E23ContractError("combined geometry-filter output is malformed")
    hypothesis_cursor = 0
    rejection_cursor = 0
    rejection_counts = {"adjacency": 0, "collision": 0, "span": 0}
    for relation_id, candidate in enumerate(relations):
        relation = _relation_tuple(candidate, label="relation candidate")
        reason = _independent_geometry_reason(
            relation,
            candidate.claim_ids,
            claims=result.claims,
            components=result.components,
            owner=owner,
            local_rows=local_rows,
            local_cols=local_cols,
        )
        if reason is None:
            if hypothesis_cursor >= len(hypotheses):
                raise E23ContractError("geometry-valid hypotheses were truncated")
            hypothesis = hypotheses[hypothesis_cursor]
            if (
                not isinstance(hypothesis, e22_core.PoseHypothesis)
                or _integer(hypothesis.hypothesis_id, label="hypothesis ID")
                != hypothesis_cursor
                or _integer(hypothesis.relation_id, label="hypothesis relation ID")
                != relation_id
                or _relation_tuple(hypothesis, label="hypothesis") != relation
                or hypothesis.claim_ids != candidate.claim_ids
            ):
                raise E23ContractError("post-geometry hypothesis algebra drifted")
            hypothesis_cursor += 1
        else:
            if rejection_cursor >= len(rejections):
                raise E23ContractError("geometry rejections were truncated")
            rejection = rejections[rejection_cursor]
            if (
                not isinstance(rejection, e22_core.GeometryRejection)
                or _integer(rejection.relation_id, label="rejection relation ID")
                != relation_id
                or rejection.reason != reason
            ):
                raise E23ContractError("geometry rejection reason/order drifted")
            rejection_counts[reason] += 1
            rejection_cursor += 1
    if hypothesis_cursor != len(hypotheses) or rejection_cursor != len(rejections):
        raise E23ContractError("combined geometry output contains extra records")

    directed_memberships = int(valid.sum())
    one_way = sum(
        (a_slot is None) != (b_slot is None)
        for _a, _b, a_slot, b_slot in expected_base
    )
    reciprocal = base_count - one_way
    nontrivial_tiles = sum(
        len(_component_entries(result.components[cid]))
        for cid in result.nontrivial_component_ids
    )
    expected_diagnostics = {
        "component_count": len(result.components),
        "nontrivial_components": len(result.nontrivial_component_ids),
        "singleton_components": len(result.components) - len(result.nontrivial_component_ids),
        "total_tiles": e12.NFRAG,
        "nontrivial_tiles": nontrivial_tiles,
        "singleton_tiles": e12.NFRAG - nontrivial_tiles,
        "emitter_tiles": e12.NFRAG,
        "directed_valid_memberships": directed_memberships,
        "input_logit_observations": 4 * directed_memberships,
        "spatial_logit_values": SPATIAL_LOGIT_VALUES,
        "spatial_selections": SPATIAL_SELECTIONS,
        "spatial_pair_nominations": SPATIAL_SELECTIONS,
        "base_affinity_pairs": base_count,
        "spatial_pairs": len(expected_spatial_identities),
        "unordered_affinity_pairs": base_count + len(expected_spatial_identities),
        "one_way_affinity_pairs": one_way,
        "reciprocal_affinity_pairs": reciprocal,
        "base_pre_component_filter_claims": 4 * base_count,
        "spatial_pre_component_filter_claims": 4 * len(expected_spatial_identities),
        "pre_component_filter_claims": 4
        * (base_count + len(expected_spatial_identities)),
        "base_same_component_pairs": same_base,
        "spatial_same_component_pairs": same_spatial,
        "same_component_pairs": same_base + same_spatial,
        "same_component_claims_removed": 4 * (same_base + same_spatial),
        "base_cross_component_pairs": cross_base,
        "spatial_cross_component_pairs": cross_spatial,
        "cross_component_pairs": cross_base + cross_spatial,
        "base_claims": 4 * cross_base,
        "spatial_claims": 4 * cross_spatial,
        "claims": claim_cursor,
        "claim_logit_observations": claim_observations,
        "relation_candidates": len(relations),
        "geometry_valid_hypotheses": hypothesis_cursor,
        "geometry_rejected_relations": rejection_cursor,
        "geometry_rejected_adjacency": rejection_counts["adjacency"],
        "geometry_rejected_collision": rejection_counts["collision"],
        "geometry_rejected_span": rejection_counts["span"],
        "component_pairs": len(component_pair_offsets),
        "component_pairs_with_alternative_offsets": sum(
            len(offsets) > 1 for offsets in component_pair_offsets.values()
        ),
    }
    diagnostics = _jsonable(result.diagnostics)
    _check_forbidden_payload_keys(diagnostics)
    if diagnostics != expected_diagnostics:
        raise E23ContractError("combined candidate-pool diagnostics drifted")

    spatial_count = len(expected_spatial_identities)
    combined_count = base_count + spatial_count
    if (
        not e12.NFRAG <= directed_memberships <= MAX_DIRECTED_MEMBERSHIPS
        or base_count > MAX_DIRECTED_MEMBERSHIPS
        or spatial_count > min(SPATIAL_SELECTIONS, MAX_UNORDERED_PAIRS - base_count)
        or combined_count > MAX_UNORDERED_PAIRS
        or 4 * spatial_count > MAX_NEW_LITERAL_CLAIMS
        or 4 * combined_count > MAX_COMBINED_LITERAL_CLAIMS
        or claim_cursor > MAX_COMBINED_LITERAL_CLAIMS
        or len(relations) > MAX_COMBINED_LITERAL_CLAIMS
        or hypothesis_cursor > MAX_COMBINED_LITERAL_CLAIMS
    ):
        raise E23ContractError("one or more theoretical fail-not-truncate bounds failed")

    component_e22_digest = e22_eval._stream_digest(result.components)
    base_pairs_e22_digest = e22_eval._stream_digest(result.base_affinity_pairs)
    prefix_replay = True
    if e22_row is not None:
        core = e22_row.get("core")
        if not isinstance(core, Mapping):
            raise E23ContractError("authorized E22 row core is malformed")
        if (
            core.get("components_sha256") != component_e22_digest
            or core.get("owner_sha256") != e12.array_sha256(owner)
            or core.get("local_rows_sha256") != e12.array_sha256(local_rows)
            or core.get("local_cols_sha256") != e12.array_sha256(local_cols)
            or core.get("affinity_pairs_sha256") != base_pairs_e22_digest
            or _integer(core.get("affinity_pair_count"), label="E22 affinity pair count")
            != base_count
        ):
            raise E23ContractError("exact authorized E22 component/pair prefix replay drifted")
    return {
        "owner": owner,
        "local_rows": local_rows,
        "local_cols": local_cols,
        "valid": valid,
        "base_pair_count": base_count,
        "spatial_pair_count": spatial_count,
        "combined_pair_count": combined_count,
        "directed_memberships": directed_memberships,
        "component_e22_sha256": component_e22_digest,
        "base_pairs_e22_sha256": base_pairs_e22_digest,
        "exact_e22_prefix_replay": prefix_replay,
        "all_theoretical_bounds": True,
    }


def _core_payload(
    result: Any,
    *,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
    e22_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = validate_candidate_pool(
        result,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        spatial_logits=spatial_logits,
        e22_row=e22_row,
    )
    diagnostics = _jsonable(result.diagnostics)
    _check_forbidden_payload_keys(diagnostics)
    payload = {
        "component_count": len(result.components),
        "nontrivial_component_ids": _jsonable(result.nontrivial_component_ids),
        "base_affinity_pair_count": len(result.base_affinity_pairs),
        "spatial_pair_count": len(result.spatial_pairs),
        "combined_pair_count": len(result.affinity_pairs),
        "spatial_selection_count": int(result.spatial_selected_ids.size),
        "claim_count": len(result.claims),
        "relation_candidate_count": len(result.relation_candidates),
        "hypothesis_count": len(result.hypotheses),
        "geometry_rejection_count": len(result.geometry_rejections),
        "components_sha256": _stream_digest(result.components),
        "components_e22_sha256": context["component_e22_sha256"],
        "owner_sha256": e12.array_sha256(result.owner),
        "local_rows_sha256": e12.array_sha256(result.local_rows),
        "local_cols_sha256": e12.array_sha256(result.local_cols),
        "base_affinity_pairs_sha256": _stream_digest(result.base_affinity_pairs),
        "base_affinity_pairs_e22_sha256": context["base_pairs_e22_sha256"],
        "spatial_selected_ids_sha256": e12.array_sha256(result.spatial_selected_ids),
        "spatial_pairs_sha256": _stream_digest(result.spatial_pairs),
        "combined_pairs_sha256": _stream_digest(result.affinity_pairs),
        "claims_sha256": _stream_digest(result.claims),
        "relation_candidates_sha256": _stream_digest(result.relation_candidates),
        "hypotheses_sha256": _stream_digest(result.hypotheses),
        "geometry_rejections_sha256": _stream_digest(result.geometry_rejections),
        "exact_e22_prefix_replay": context["exact_e22_prefix_replay"],
        "diagnostics": diagnostics,
        "diagnostic_semantics": {
            "one_way_affinity_pairs": "exact_E22_prefix_only",
            "reciprocal_affinity_pairs": "exact_E22_prefix_only",
        },
    }
    return payload, context


def _pair_key(seam: tuple[int, int, int, int]) -> tuple[int, int]:
    first, second = seam[:2]
    return (first, second) if first < second else (second, first)


def _measure_arm_after_core(
    result: e23_core.CandidatePoolResult,
    *,
    permutation: object,
    authorized_e22_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply labels only after both complete cores were validated by the caller."""

    try:
        shifts, true_hypotheses, clusters, selected = e22_eval.build_oracle_ceiling(
            result, permutation
        )
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc
    owner, local_rows, local_cols = _validate_components(result)
    pure_ids = tuple(sorted(cid for cid, shift in shifts.items() if shift is not None))
    pure_tiles = sum(len(_component_entries(result.components[cid])) for cid in pure_ids)
    try:
        all_seams = e22_eval.ground_truth_seams(permutation)
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc
    unconditional = tuple(
        seam for seam in all_seams if int(owner[seam[0]]) != int(owner[seam[1]])
    )
    eligible = tuple(
        seam
        for seam in unconditional
        if shifts[int(owner[seam[0]])] is not None
        and shifts[int(owner[seam[1]])] is not None
    )
    base_pairs = {(int(pair.a), int(pair.b)) for pair in result.base_affinity_pairs}
    spatial_pairs = {(int(pair.a), int(pair.b)) for pair in result.spatial_pairs}
    combined_pairs = {(int(pair.a), int(pair.b)) for pair in result.affinity_pairs}
    if base_pairs.intersection(spatial_pairs) or combined_pairs != base_pairs | spatial_pairs:
        raise E23ContractError("label-only pair inventory lost base/new disjoint union")
    base_hits = tuple(seam for seam in eligible if _pair_key(seam) in base_pairs)
    combined_hits = tuple(seam for seam in eligible if _pair_key(seam) in combined_pairs)
    incremental_hits = tuple(
        seam for seam in eligible if _pair_key(seam) in spatial_pairs
    )
    if not set(base_hits).issubset(combined_hits):
        raise E23ContractError("an authorized E22 eligible hit was not preserved")
    if set(base_hits).intersection(incremental_hits):
        raise E23ContractError("incremental eligible hits overlap E22 hits")
    if set(combined_hits) != set(base_hits) | set(incremental_hits):
        raise E23ContractError("combined eligible hits are not base plus incremental")
    unconditional_hits = tuple(
        seam for seam in unconditional if _pair_key(seam) in combined_pairs
    )

    hypothesis_by_relation = {
        _relation_tuple(value, label="hypothesis"): value
        for value in result.hypotheses
    }
    survivors: list[tuple[int, int, int, int]] = []
    for seam in combined_hits:
        try:
            relation = e22_eval.seam_relation(
                seam,
                owner=owner,
                local_rows=local_rows,
                local_cols=local_cols,
            )
        except Exception as exc:
            raise E23ContractError(str(exc)) from exc
        hypothesis = hypothesis_by_relation.get(relation)
        if hypothesis is None:
            continue
        supported = {
            tuple(map(int, result.claims[int(claim_id)].physical_seam))
            for claim_id in hypothesis.claim_ids
        }
        if seam in supported:
            survivors.append(seam)

    eligible_digest = e12.canonical_digest(_jsonable(eligible))
    base_hits_digest = e12.canonical_digest(_jsonable(base_hits))
    authorized_oracle = authorized_e22_row.get("oracle")
    authorized_metrics = authorized_e22_row.get("metrics")
    if not isinstance(authorized_oracle, Mapping) or not isinstance(
        authorized_metrics, Mapping
    ):
        raise E23ContractError("authorized E22 label-only row is malformed")
    contact = authorized_oracle.get("contact_inventory")
    if not isinstance(contact, Mapping):
        raise E23ContractError("authorized E22 contact inventory is malformed")
    if (
        contact.get("eligible_whole_pure_cross_component_seams_sha256")
        != eligible_digest
        or contact.get("eligible_pair_or_hits_sha256") != base_hits_digest
        or _integer(authorized_metrics.get("eligible_contacts"), label="E22 eligible contacts")
        != len(eligible)
        or _integer(authorized_metrics.get("eligible_pair_hits"), label="E22 eligible hits")
        != len(base_hits)
    ):
        raise E23ContractError("E22 eligible denominator/hit replay drifted")

    eligible_count = len(eligible)
    combined_hit_count = len(combined_hits)
    new_pair_count = len(result.spatial_pairs)
    if new_pair_count <= 0:
        raise E23ContractError("incremental hit efficiency has zero pair denominator")
    survival = (
        float(len(survivors) / combined_hit_count) if combined_hit_count else 0.0
    )
    combined_recall = (
        float(combined_hit_count / eligible_count) if eligible_count else 0.0
    )
    base_recall = float(len(base_hits) / eligible_count) if eligible_count else 0.0
    unconditional_recall = (
        float(len(unconditional_hits) / len(unconditional)) if unconditional else 0.0
    )
    incremental_efficiency = float(len(incremental_hits) / new_pair_count)
    inventory = {
        "ground_truth_upright_rd_seam_count": len(all_seams),
        "ground_truth_upright_rd_seams_sha256": e12.canonical_digest(
            _jsonable(all_seams)
        ),
        "unconditional_cross_component_seams_sha256": e12.canonical_digest(
            _jsonable(unconditional)
        ),
        "eligible_whole_pure_cross_component_seams_sha256": eligible_digest,
        "e22_base_eligible_hits_sha256": base_hits_digest,
        "combined_eligible_hits_sha256": e12.canonical_digest(
            _jsonable(combined_hits)
        ),
        "incremental_eligible_hits_sha256": e12.canonical_digest(
            _jsonable(incremental_hits)
        ),
        "postfilter_exact_physical_seam_survivors_sha256": e12.canonical_digest(
            _jsonable(tuple(survivors))
        ),
    }
    true_identities = (
        (
            int(value.hypothesis_id),
            int(value.relation_id),
            *_relation_tuple(value, label="true hypothesis"),
        )
        for value in true_hypotheses
    )
    oracle = {
        "pure_component_ids": _jsonable(pure_ids),
        "pure_shifts_sha256": e12.canonical_digest(_jsonable(shifts)),
        "true_hypotheses_sha256": _stream_digest(true_identities),
        "cluster_count": len(clusters),
        "clusters_sha256": _stream_digest(
            e22_eval._cluster_payload(value) for value in clusters
        ),
        "contact_inventory": inventory,
        "selected": e22_eval._cluster_payload(selected),
    }
    diagnostics = result.diagnostics
    metrics = {
        "tile_orientation_degrees": 0,
        "emitter_tiles": int(diagnostics.emitter_tiles),
        "directed_valid_memberships": int(diagnostics.directed_valid_memberships),
        "spatial_logit_values": int(diagnostics.spatial_logit_values),
        "spatial_selections": int(diagnostics.spatial_selections),
        "base_affinity_pairs": len(result.base_affinity_pairs),
        "spatial_pairs": new_pair_count,
        "combined_pairs": len(result.affinity_pairs),
        "new_literal_rcce4_preclaims": int(
            diagnostics.spatial_pre_component_filter_claims
        ),
        "combined_literal_rcce4_preclaims": int(
            diagnostics.pre_component_filter_claims
        ),
        "cross_component_claims": int(diagnostics.claims),
        "relation_candidates": int(diagnostics.relation_candidates),
        "geometry_valid_hypotheses": int(diagnostics.geometry_valid_hypotheses),
        "theoretical_bounds_passed": True,
        "spatial_pair_deployability_guard": new_pair_count <= 100_000,
        "spatial_geometry_deployability_guard": len(result.hypotheses) <= 450_000,
        "component_count": len(result.components),
        "nontrivial_component_count": len(result.nontrivial_component_ids),
        "pure_component_count": len(pure_ids),
        "pure_component_tiles": int(pure_tiles),
        "eligible_contacts": eligible_count,
        "e22_base_eligible_hits": len(base_hits),
        "e22_base_eligible_recall": base_recall,
        "combined_eligible_hits": combined_hit_count,
        "combined_eligible_recall": combined_recall,
        "incremental_eligible_hits": len(incremental_hits),
        "incremental_hit_efficiency": incremental_efficiency,
        "unconditional_cross_component_contacts": len(unconditional),
        "unconditional_combined_hits": len(unconditional_hits),
        "unconditional_combined_recall": unconditional_recall,
        "postfilter_eligible_hits": combined_hit_count,
        "postfilter_exact_physical_seam_survivors": len(survivors),
        "postfilter_eligible_true_survival": survival,
        "true_hypotheses": len(true_hypotheses),
        "selected_components": len(selected.component_ids),
        "selected_accepted_relations": selected.accepted_relation_count,
        "selected_cycle_rank": selected.cycle_rank,
        "selected_cycle_rank_ratio": selected.cycle_rank_ratio,
        "selected_exact_connected_tiles": selected.exact_connected_tiles,
        "selected_exact_connected_coverage": selected.exact_connected_coverage,
        "legal_origin_count": selected.legal_origin_count,
        "exact_e22_denominator_and_hits_replay": True,
    }
    state = {
        "eligible": eligible,
        "base_hits": base_hits,
        "combined_hits": combined_hits,
        "incremental_hits": incremental_hits,
    }
    return oracle, metrics, state


def _cache_record_payload(record: SpatialCacheRecord) -> dict[str, Any]:
    return {
        "key": record.key,
        "array_path": str(record.array_path),
        "metadata_path": str(record.metadata_path),
        "array_file_sha256": record.array_file_sha256,
        "array_file_bytes": record.array_file_bytes,
        "array_sha256": record.array_sha256,
        "byte_validated": True,
    }


def evaluate_scene_pair(
    scene: e12.RawScene,
    spatial_result: e23_core.CandidatePoolResult,
    null_result: e23_core.CandidatePoolResult,
    *,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
    null_logits: np.ndarray,
    spatial_cache_record: SpatialCacheRecord,
    authorized_e22_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate both complete label-free pools, then open labels once."""

    # No permutation/target/label field is touched before both calls return.
    spatial_core, spatial_context = _core_payload(
        spatial_result,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        spatial_logits=spatial_logits,
        e22_row=authorized_e22_row,
    )
    maximum_pairs = int(DECISION_RULE["spatial_new_pairs_max_each"])
    if spatial_context["spatial_pair_count"] > maximum_pairs:
        raise E23ScientificGuardFailure(
            image=int(scene.image_id),
            guard="spatial_new_pairs_max_each",
            observed=int(spatial_context["spatial_pair_count"]),
            maximum=maximum_pairs,
            phase="independent_post_core_validation",
        )
    maximum_hypotheses = int(
        DECISION_RULE["spatial_geometry_valid_hypotheses_max_each"]
    )
    if len(spatial_result.hypotheses) > maximum_hypotheses:
        raise E23ScientificGuardFailure(
            image=int(scene.image_id),
            guard="spatial_geometry_valid_hypotheses_max_each",
            observed=len(spatial_result.hypotheses),
            maximum=maximum_hypotheses,
            phase="independent_post_core_validation",
        )
    null_core, null_context = _core_payload(
        null_result,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        spatial_logits=null_logits,
        e22_row=authorized_e22_row,
    )
    if (
        spatial_context["component_e22_sha256"]
        != null_context["component_e22_sha256"]
        or spatial_context["base_pairs_e22_sha256"]
        != null_context["base_pairs_e22_sha256"]
    ):
        raise E23ContractError("spatial/null unchanged E22 prefixes disagree")

    expected_image = _integer(authorized_e22_row.get("image"), label="E22 row image")
    if (
        expected_image != int(scene.image_id)
        or authorized_e22_row.get("validation_name") != str(scene.validation_name)
        or authorized_e22_row.get("raw_cache_sha256") != str(scene.cache_sha256)
        or authorized_e22_row.get("candidate_ids_sha256")
        != e12.array_sha256(candidate_ids)
        or authorized_e22_row.get("raw_logits_sha256") != e12.array_sha256(raw_logits)
    ):
        raise E23ContractError("scene provenance differs from authorized E22 row")

    # First label access in the complete spatial+null evaluation path.
    permutation = scene.permutation
    spatial_oracle, spatial_metrics, spatial_state = _measure_arm_after_core(
        spatial_result,
        permutation=permutation,
        authorized_e22_row=authorized_e22_row,
    )
    null_oracle, null_metrics, null_state = _measure_arm_after_core(
        null_result,
        permutation=permutation,
        authorized_e22_row=authorized_e22_row,
    )
    if (
        spatial_state["eligible"] != null_state["eligible"]
        or spatial_state["base_hits"] != null_state["base_hits"]
    ):
        raise E23ContractError("spatial/null E22 denominator or base hits disagree")
    null_efficiency = float(null_metrics["incremental_hit_efficiency"])
    spatial_efficiency = float(spatial_metrics["incremental_hit_efficiency"])
    null_denominator_positive = null_efficiency > 0.0
    efficiency_ratio = (
        float(spatial_efficiency / null_efficiency)
        if null_denominator_positive
        else 0.0
    )
    comparison = {
        "S_spatial": int(spatial_metrics["spatial_pairs"]),
        "S_null": int(null_metrics["spatial_pairs"]),
        "spatial_incremental_eligible_hits": int(
            spatial_metrics["incremental_eligible_hits"]
        ),
        "null_incremental_eligible_hits": int(
            null_metrics["incremental_eligible_hits"]
        ),
        "spatial_incremental_hit_efficiency": spatial_efficiency,
        "null_incremental_hit_efficiency": null_efficiency,
        "null_efficiency_denominator_positive": null_denominator_positive,
        "incremental_hit_efficiency_ratio": efficiency_ratio,
        "spatial_minus_null_combined_recall": float(
            spatial_metrics["combined_eligible_recall"]
            - null_metrics["combined_eligible_recall"]
        ),
        "spatial_strict_recall_win": bool(
            spatial_metrics["combined_eligible_recall"]
            > null_metrics["combined_eligible_recall"]
        ),
    }
    spatial_payload = {
        "core": spatial_core,
        "core_sha256": e12.canonical_digest(spatial_core),
        "oracle": spatial_oracle,
        "oracle_sha256": e12.canonical_digest(spatial_oracle),
        "metrics": spatial_metrics,
    }
    null_payload = {
        "core": null_core,
        "core_sha256": e12.canonical_digest(null_core),
        "oracle": null_oracle,
        "oracle_sha256": e12.canonical_digest(null_oracle),
        "metrics": null_metrics,
    }
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "raw_cache_sha256": str(scene.cache_sha256),
        "candidate_ids_sha256": e12.array_sha256(candidate_ids),
        "raw_logits_sha256": e12.array_sha256(raw_logits),
        "tiles_uint8_sha256": e12.array_sha256(scene.tiles_uint8),
        "orientation": "upright_0_degrees_no_rotation_no_reflection",
        "arm": "E23_I21_residual_K64_vs_matched_hash_null_candidate_ceiling",
        "spatial_logits_sha256": e12.array_sha256(spatial_logits),
        "spatial_cache": _cache_record_payload(spatial_cache_record),
        "null_rule": "E23-hash-null-v1",
        "null_logits_sha256": e12.array_sha256(null_logits),
        "spatial": spatial_payload,
        "hash_null": null_payload,
        "comparison": comparison,
        "exact_e22_prefix_and_provenance_replay": True,
    }


def _arm_metrics(row: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    payload = row.get(arm)
    if not isinstance(payload, Mapping):
        raise E23ContractError(f"row {arm} payload is malformed")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise E23ContractError(f"row {arm} metrics are malformed")
    return metrics


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E23ContractError("E23 summary requires exactly eight rows")
    images = [_integer(row.get("image"), label="summary image") for row in rows]
    if tuple(images) != e12.CALIBRATION_IDS or len(set(images)) != len(images):
        raise E23ContractError("E23 summary image IDs are incomplete or reordered")
    spatial = [_arm_metrics(row, "spatial") for row in rows]
    null = [_arm_metrics(row, "hash_null") for row in rows]
    comparisons: list[Mapping[str, Any]] = []
    for row in rows:
        value = row.get("comparison")
        if not isinstance(value, Mapping):
            raise E23ContractError("E23 comparison payload is malformed")
        comparisons.append(value)

    def ints(metrics: Sequence[Mapping[str, Any]], key: str) -> list[int]:
        return [_integer(value.get(key), label=key) for value in metrics]

    def finite(
        metrics: Sequence[Mapping[str, Any]],
        key: str,
        *,
        minimum: float = 0.0,
        maximum: float = float("inf"),
    ) -> list[float]:
        return [
            _finite(value.get(key), label=key, minimum=minimum, maximum=maximum)
            for value in metrics
        ]

    spatial_emitters = ints(spatial, "emitter_tiles")
    null_emitters = ints(null, "emitter_tiles")
    spatial_memberships = ints(spatial, "directed_valid_memberships")
    null_memberships = ints(null, "directed_valid_memberships")
    spatial_values = ints(spatial, "spatial_logit_values")
    null_values = ints(null, "spatial_logit_values")
    spatial_selections = ints(spatial, "spatial_selections")
    null_selections = ints(null, "spatial_selections")
    spatial_base_pairs = ints(spatial, "base_affinity_pairs")
    null_base_pairs = ints(null, "base_affinity_pairs")
    spatial_new_pairs = ints(spatial, "spatial_pairs")
    null_new_pairs = ints(null, "spatial_pairs")
    spatial_combined_pairs = ints(spatial, "combined_pairs")
    null_combined_pairs = ints(null, "combined_pairs")
    spatial_new_preclaims = ints(spatial, "new_literal_rcce4_preclaims")
    null_new_preclaims = ints(null, "new_literal_rcce4_preclaims")
    spatial_preclaims = ints(spatial, "combined_literal_rcce4_preclaims")
    null_preclaims = ints(null, "combined_literal_rcce4_preclaims")
    spatial_claims = ints(spatial, "cross_component_claims")
    null_claims = ints(null, "cross_component_claims")
    spatial_relations = ints(spatial, "relation_candidates")
    null_relations = ints(null, "relation_candidates")
    spatial_hypotheses = ints(spatial, "geometry_valid_hypotheses")
    null_hypotheses = ints(null, "geometry_valid_hypotheses")

    def algebraic_bounds(index: int, arm: Sequence[Mapping[str, Any]], *, spatial_arm: bool) -> bool:
        memberships = spatial_memberships[index] if spatial_arm else null_memberships[index]
        values = spatial_values[index] if spatial_arm else null_values[index]
        selections = spatial_selections[index] if spatial_arm else null_selections[index]
        base = spatial_base_pairs[index] if spatial_arm else null_base_pairs[index]
        new = spatial_new_pairs[index] if spatial_arm else null_new_pairs[index]
        combined = spatial_combined_pairs[index] if spatial_arm else null_combined_pairs[index]
        new_pre = spatial_new_preclaims[index] if spatial_arm else null_new_preclaims[index]
        pre = spatial_preclaims[index] if spatial_arm else null_preclaims[index]
        claims = spatial_claims[index] if spatial_arm else null_claims[index]
        relations = spatial_relations[index] if spatial_arm else null_relations[index]
        hypotheses = spatial_hypotheses[index] if spatial_arm else null_hypotheses[index]
        return (
            arm[index].get("theoretical_bounds_passed") is True
            and e12.NFRAG <= memberships <= MAX_DIRECTED_MEMBERSHIPS
            and values == SPATIAL_LOGIT_VALUES
            and selections == SPATIAL_SELECTIONS
            and base <= MAX_DIRECTED_MEMBERSHIPS
            and new <= min(SPATIAL_SELECTIONS, MAX_UNORDERED_PAIRS - base)
            and combined == base + new <= MAX_UNORDERED_PAIRS
            and new_pre == 4 * new <= MAX_NEW_LITERAL_CLAIMS
            and pre == 4 * combined <= MAX_COMBINED_LITERAL_CLAIMS
            and 0 <= claims <= pre
            and claims % 4 == 0
            and 0 <= relations <= claims <= MAX_COMBINED_LITERAL_CLAIMS
            and 0 <= hypotheses <= relations <= MAX_COMBINED_LITERAL_CLAIMS
        )

    spatial_bounds = [algebraic_bounds(i, spatial, spatial_arm=True) for i in range(8)]
    null_bounds = [algebraic_bounds(i, null, spatial_arm=False) for i in range(8)]
    spatial_eligible = ints(spatial, "eligible_contacts")
    null_eligible = ints(null, "eligible_contacts")
    spatial_base_hits = ints(spatial, "e22_base_eligible_hits")
    null_base_hits = ints(null, "e22_base_eligible_hits")
    spatial_combined_hits = ints(spatial, "combined_eligible_hits")
    null_combined_hits = ints(null, "combined_eligible_hits")
    spatial_incremental = ints(spatial, "incremental_eligible_hits")
    null_incremental = ints(null, "incremental_eligible_hits")
    spatial_survivors = ints(spatial, "postfilter_exact_physical_seam_survivors")
    null_survivors = ints(null, "postfilter_exact_physical_seam_survivors")
    spatial_true = ints(spatial, "true_hypotheses")
    null_true = ints(null, "true_hypotheses")
    spatial_legal = ints(spatial, "legal_origin_count")
    null_legal = ints(null, "legal_origin_count")
    spatial_survival = finite(
        spatial, "postfilter_eligible_true_survival", minimum=0.0, maximum=1.0
    )
    null_survival = finite(
        null, "postfilter_eligible_true_survival", minimum=0.0, maximum=1.0
    )
    spatial_recall = finite(
        spatial, "combined_eligible_recall", minimum=0.0, maximum=1.0
    )
    null_recall = finite(null, "combined_eligible_recall", minimum=0.0, maximum=1.0)
    spatial_base_recall = finite(
        spatial, "e22_base_eligible_recall", minimum=0.0, maximum=1.0
    )
    null_base_recall = finite(
        null, "e22_base_eligible_recall", minimum=0.0, maximum=1.0
    )
    spatial_metric_efficiency = finite(
        spatial, "incremental_hit_efficiency", minimum=0.0, maximum=1.0
    )
    null_metric_efficiency = finite(
        null, "incremental_hit_efficiency", minimum=0.0, maximum=1.0
    )
    spatial_coverage = finite(
        spatial, "selected_exact_connected_coverage", minimum=0.0, maximum=1.0
    )
    spatial_tiles = ints(spatial, "selected_exact_connected_tiles")
    spatial_selected_components = ints(spatial, "selected_components")
    spatial_cycle_ranks = ints(spatial, "selected_cycle_rank")
    spatial_cycles = finite(spatial, "selected_cycle_rank_ratio", minimum=0.0)
    lifts = [
        _finite(
            value.get("spatial_minus_null_combined_recall"),
            label="spatial-minus-null recall",
            minimum=-1.0,
            maximum=1.0,
        )
        for value in comparisons
    ]
    efficiency_ratios = [
        _finite(
            value.get("incremental_hit_efficiency_ratio"),
            label="incremental efficiency ratio",
            minimum=0.0,
        )
        for value in comparisons
    ]
    spatial_efficiencies = [
        _finite(
            value.get("spatial_incremental_hit_efficiency"),
            label="spatial incremental efficiency",
            minimum=0.0,
            maximum=1.0,
        )
        for value in comparisons
    ]
    null_efficiencies = [
        _finite(
            value.get("null_incremental_hit_efficiency"),
            label="null incremental efficiency",
            minimum=0.0,
            maximum=1.0,
        )
        for value in comparisons
    ]
    null_denominator_positive = [
        value.get("null_efficiency_denominator_positive") is True
        for value in comparisons
    ]
    strict_wins = [value.get("spatial_strict_recall_win") is True for value in comparisons]
    prefix_replays = [
        row.get("exact_e22_prefix_and_provenance_replay") is True for row in rows
    ]
    for index in range(8):
        if spatial_new_pairs[index] <= 0 or null_new_pairs[index] <= 0:
            raise E23ContractError("comparison has a zero new-pair denominator")
        expected_spatial_efficiency = float(
            spatial_incremental[index] / spatial_new_pairs[index]
        )
        expected_null_efficiency = float(
            null_incremental[index] / null_new_pairs[index]
        )
        expected_ratio = (
            float(expected_spatial_efficiency / expected_null_efficiency)
            if expected_null_efficiency > 0.0
            else 0.0
        )
        expected_spatial_recall = float(
            spatial_combined_hits[index] / spatial_eligible[index]
        ) if spatial_eligible[index] > 0 else 0.0
        expected_null_recall = float(
            null_combined_hits[index] / null_eligible[index]
        ) if null_eligible[index] > 0 else 0.0
        expected_spatial_base_recall = float(
            spatial_base_hits[index] / spatial_eligible[index]
        ) if spatial_eligible[index] > 0 else 0.0
        expected_null_base_recall = float(
            null_base_hits[index] / null_eligible[index]
        ) if null_eligible[index] > 0 else 0.0
        expected_spatial_survival = float(
            spatial_survivors[index] / spatial_combined_hits[index]
        ) if spatial_combined_hits[index] > 0 else 0.0
        expected_null_survival = float(
            null_survivors[index] / null_combined_hits[index]
        ) if null_combined_hits[index] > 0 else 0.0
        expected_coverage = float(spatial_tiles[index] / e12.NFRAG)
        expected_cycle_ratio = float(
            spatial_cycle_ranks[index]
            / max(1, spatial_selected_components[index] - 1)
        )
        if (
            spatial_base_pairs[index] != null_base_pairs[index]
            or spatial_memberships[index] != null_memberships[index]
            or spatial_eligible[index] != null_eligible[index]
            or spatial_base_hits[index] != null_base_hits[index]
            or not 0
            <= spatial_base_hits[index]
            <= spatial_combined_hits[index]
            <= spatial_eligible[index]
            or not 0
            <= null_base_hits[index]
            <= null_combined_hits[index]
            <= null_eligible[index]
            or spatial_incremental[index]
            != spatial_combined_hits[index] - spatial_base_hits[index]
            or null_incremental[index]
            != null_combined_hits[index] - null_base_hits[index]
            or spatial_recall[index] != expected_spatial_recall
            or null_recall[index] != expected_null_recall
            or spatial_base_recall[index] != expected_spatial_base_recall
            or null_base_recall[index] != expected_null_base_recall
            or spatial_metric_efficiency[index] != expected_spatial_efficiency
            or null_metric_efficiency[index] != expected_null_efficiency
            or spatial_survival[index] != expected_spatial_survival
            or null_survival[index] != expected_null_survival
            or not 1 <= spatial_tiles[index] <= e12.NFRAG
            or spatial_selected_components[index] < 1
            or spatial_cycle_ranks[index] < 0
            or spatial_coverage[index] != expected_coverage
            or spatial_cycles[index] != expected_cycle_ratio
            or not math.isclose(
                lifts[index], spatial_recall[index] - null_recall[index], rel_tol=0.0, abs_tol=0.0
            )
            or _integer(comparisons[index].get("S_spatial"), label="S_spatial")
            != spatial_new_pairs[index]
            or _integer(comparisons[index].get("S_null"), label="S_null")
            != null_new_pairs[index]
            or _integer(
                comparisons[index].get("spatial_incremental_eligible_hits"),
                label="spatial incremental hits",
            )
            != spatial_incremental[index]
            or _integer(
                comparisons[index].get("null_incremental_eligible_hits"),
                label="null incremental hits",
            )
            != null_incremental[index]
            or spatial_efficiencies[index] != expected_spatial_efficiency
            or null_efficiencies[index] != expected_null_efficiency
            or efficiency_ratios[index] != expected_ratio
            or null_denominator_positive[index]
            != (expected_null_efficiency > 0.0)
            or strict_wins[index]
            != (spatial_recall[index] > null_recall[index])
        ):
            raise E23ContractError("spatial/null comparison accounting drifted")

    spatial_guard = [
        spatial_new_pairs[index]
        <= int(DECISION_RULE["spatial_new_pairs_max_each"])
        and spatial_hypotheses[index]
        <= int(DECISION_RULE["spatial_geometry_valid_hypotheses_max_each"])
        and spatial[index].get("spatial_pair_deployability_guard") is True
        and spatial[index].get("spatial_geometry_deployability_guard") is True
        for index in range(8)
    ]
    null_complete = [
        null_emitters[index] == e12.NFRAG
        and null_bounds[index]
        and null_eligible[index] >= 1
        and null_incremental[index] >= 1
        and null_true[index] >= 1
        and null_legal[index] >= 1
        and null_survival[index] == 1.0
        for index in range(8)
    ]
    return {
        "images": len(rows),
        "completed_scenes": len(rows),
        "upright_orientation_scenes": int(
            sum(row.get("orientation") == "upright_0_degrees_no_rotation_no_reflection" for row in rows)
        ),
        "spatial_emitters_exact_scenes": int(sum(value == e12.NFRAG for value in spatial_emitters)),
        "null_emitters_exact_scenes": int(sum(value == e12.NFRAG for value in null_emitters)),
        "exact_e22_prefix_replay_scenes": int(sum(prefix_replays)),
        "provenance_replay_scenes": int(sum(prefix_replays)),
        "spatial_all_theoretical_bounds_scenes": int(sum(spatial_bounds)),
        "null_all_theoretical_bounds_scenes": int(sum(null_bounds)),
        "spatial_true_relation_scenes": int(sum(value >= 1 for value in spatial_true)),
        "null_true_relation_scenes": int(sum(value >= 1 for value in null_true)),
        "spatial_legal_origin_scenes": int(sum(value >= 1 for value in spatial_legal)),
        "null_legal_origin_scenes": int(sum(value >= 1 for value in null_legal)),
        "spatial_positive_eligible_denominator_scenes": int(
            sum(value >= 1 for value in spatial_eligible)
        ),
        "null_positive_eligible_denominator_scenes": int(
            sum(value >= 1 for value in null_eligible)
        ),
        "spatial_incremental_eligible_hit_scenes": int(
            sum(value >= 1 for value in spatial_incremental)
        ),
        "null_incremental_eligible_hit_scenes": int(
            sum(value >= 1 for value in null_incremental)
        ),
        "spatial_exact_postfilter_survival_scenes": int(
            sum(value == 1.0 for value in spatial_survival)
        ),
        "null_exact_postfilter_survival_scenes": int(
            sum(value == 1.0 for value in null_survival)
        ),
        "null_complete_bounds_survival_scenes": int(sum(null_complete)),
        "nonzero_null_efficiency_scenes": int(sum(null_denominator_positive)),
        "spatial_deployability_guard_scenes": int(sum(spatial_guard)),
        "mean_spatial_combined_eligible_recall": float(np.mean(spatial_recall)),
        "worst_spatial_combined_eligible_recall": float(min(spatial_recall)),
        "mean_null_combined_eligible_recall": float(np.mean(null_recall)),
        "worst_null_combined_eligible_recall": float(min(null_recall)),
        "mean_spatial_minus_null_combined_recall": float(np.mean(lifts)),
        "strict_spatial_recall_win_scenes": int(sum(strict_wins)),
        "mean_incremental_hit_efficiency_ratio": float(np.mean(efficiency_ratios)),
        "mean_spatial_exact_connected_tiles": float(np.mean(spatial_tiles)),
        "mean_spatial_exact_connected_coverage": float(np.mean(spatial_coverage)),
        "worst_spatial_exact_connected_coverage": float(min(spatial_coverage)),
        "mean_spatial_selected_cycle_rank_ratio": float(np.mean(spatial_cycles)),
        "worst_spatial_selected_cycle_rank_ratio": float(min(spatial_cycles)),
        "max_spatial_new_pairs": max(spatial_new_pairs),
        "max_null_new_pairs": max(null_new_pairs),
        "max_spatial_geometry_valid_hypotheses": max(spatial_hypotheses),
        "max_null_geometry_valid_hypotheses": max(null_hypotheses),
        "total_spatial_incremental_eligible_hits": int(sum(spatial_incremental)),
        "total_null_incremental_eligible_hits": int(sum(null_incremental)),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    target = int(DECISION_RULE["completed_scenes"])
    observed = {
        "completed_scenes": _integer(summary.get("completed_scenes"), label="completed scenes"),
        "upright_orientation_scenes": _integer(
            summary.get("upright_orientation_scenes"), label="upright scenes"
        ),
        "spatial_emitters_exact_scenes": _integer(
            summary.get("spatial_emitters_exact_scenes"), label="spatial emitters"
        ),
        "null_emitters_exact_scenes": _integer(
            summary.get("null_emitters_exact_scenes"), label="null emitters"
        ),
        "exact_e22_prefix_replay_scenes": _integer(
            summary.get("exact_e22_prefix_replay_scenes"), label="E22 prefix replays"
        ),
        "provenance_replay_scenes": _integer(
            summary.get("provenance_replay_scenes"), label="provenance replays"
        ),
        "spatial_all_theoretical_bounds_scenes": _integer(
            summary.get("spatial_all_theoretical_bounds_scenes"), label="spatial bounds"
        ),
        "null_all_theoretical_bounds_scenes": _integer(
            summary.get("null_all_theoretical_bounds_scenes"), label="null bounds"
        ),
        "spatial_true_relation_scenes": _integer(
            summary.get("spatial_true_relation_scenes"), label="spatial true relations"
        ),
        "null_true_relation_scenes": _integer(
            summary.get("null_true_relation_scenes"), label="null true relations"
        ),
        "spatial_legal_origin_scenes": _integer(
            summary.get("spatial_legal_origin_scenes"), label="spatial legal origins"
        ),
        "null_legal_origin_scenes": _integer(
            summary.get("null_legal_origin_scenes"), label="null legal origins"
        ),
        "spatial_positive_eligible_denominator_scenes": _integer(
            summary.get("spatial_positive_eligible_denominator_scenes"),
            label="spatial eligible denominators",
        ),
        "null_positive_eligible_denominator_scenes": _integer(
            summary.get("null_positive_eligible_denominator_scenes"),
            label="null eligible denominators",
        ),
        "spatial_incremental_eligible_hit_scenes": _integer(
            summary.get("spatial_incremental_eligible_hit_scenes"),
            label="spatial incremental-hit scenes",
        ),
        "null_incremental_eligible_hit_scenes": _integer(
            summary.get("null_incremental_eligible_hit_scenes"),
            label="null incremental-hit scenes",
        ),
        "spatial_exact_postfilter_survival_scenes": _integer(
            summary.get("spatial_exact_postfilter_survival_scenes"),
            label="spatial survival scenes",
        ),
        "null_exact_postfilter_survival_scenes": _integer(
            summary.get("null_exact_postfilter_survival_scenes"),
            label="null survival scenes",
        ),
        "null_complete_bounds_survival_scenes": _integer(
            summary.get("null_complete_bounds_survival_scenes"),
            label="null complete replay scenes",
        ),
        "nonzero_null_efficiency_scenes": _integer(
            summary.get("nonzero_null_efficiency_scenes"),
            label="nonzero null efficiency scenes",
        ),
        "spatial_deployability_guard_scenes": _integer(
            summary.get("spatial_deployability_guard_scenes"),
            label="spatial deployability scenes",
        ),
        "mean_spatial_combined_eligible_recall": _finite(
            summary.get("mean_spatial_combined_eligible_recall"),
            label="mean spatial recall",
            minimum=0.0,
            maximum=1.0,
        ),
        "worst_spatial_combined_eligible_recall": _finite(
            summary.get("worst_spatial_combined_eligible_recall"),
            label="worst spatial recall",
            minimum=0.0,
            maximum=1.0,
        ),
        "mean_spatial_minus_null_combined_recall": _finite(
            summary.get("mean_spatial_minus_null_combined_recall"),
            label="mean spatial-minus-null recall",
            minimum=-1.0,
            maximum=1.0,
        ),
        "strict_spatial_recall_win_scenes": _integer(
            summary.get("strict_spatial_recall_win_scenes"),
            label="strict spatial wins",
        ),
        "mean_incremental_hit_efficiency_ratio": _finite(
            summary.get("mean_incremental_hit_efficiency_ratio"),
            label="mean efficiency ratio",
            minimum=0.0,
        ),
        "mean_spatial_exact_connected_coverage": _finite(
            summary.get("mean_spatial_exact_connected_coverage"),
            label="mean spatial coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        "worst_spatial_exact_connected_coverage": _finite(
            summary.get("worst_spatial_exact_connected_coverage"),
            label="worst spatial coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        "mean_spatial_selected_cycle_rank_ratio": _finite(
            summary.get("mean_spatial_selected_cycle_rank_ratio"),
            label="mean spatial cycle ratio",
            minimum=0.0,
        ),
        "worst_spatial_selected_cycle_rank_ratio": _finite(
            summary.get("worst_spatial_selected_cycle_rank_ratio"),
            label="worst spatial cycle ratio",
            minimum=0.0,
        ),
    }
    checks = {
        "completed_scenes": observed["completed_scenes"] == target,
        "upright_orientation_only": observed["upright_orientation_scenes"] == target,
        "spatial_emitters_each": observed["spatial_emitters_exact_scenes"] == target,
        "null_emitters_each": observed["null_emitters_exact_scenes"] == target,
        "exact_e22_prefix_replays": observed["exact_e22_prefix_replay_scenes"] == target,
        "provenance_replays": observed["provenance_replay_scenes"] == target,
        "spatial_all_theoretical_bounds": observed[
            "spatial_all_theoretical_bounds_scenes"
        ]
        == target,
        "null_all_theoretical_bounds": observed["null_all_theoretical_bounds_scenes"]
        == target,
        "spatial_true_relations": observed["spatial_true_relation_scenes"] == target,
        "null_true_relations": observed["null_true_relation_scenes"] == target,
        "spatial_legal_origins": observed["spatial_legal_origin_scenes"] == target,
        "null_legal_origins": observed["null_legal_origin_scenes"] == target,
        "spatial_positive_eligible_denominators": observed[
            "spatial_positive_eligible_denominator_scenes"
        ]
        == target,
        "null_positive_eligible_denominators": observed[
            "null_positive_eligible_denominator_scenes"
        ]
        == target,
        "spatial_incremental_hits_each": observed[
            "spatial_incremental_eligible_hit_scenes"
        ]
        == target,
        "null_incremental_hits_each": observed["null_incremental_eligible_hit_scenes"]
        == target,
        "spatial_exact_survival": observed[
            "spatial_exact_postfilter_survival_scenes"
        ]
        == target,
        "null_exact_survival": observed["null_exact_postfilter_survival_scenes"]
        == target,
        "null_complete_bounds_survival": observed[
            "null_complete_bounds_survival_scenes"
        ]
        == target,
        "nonzero_null_efficiency": observed["nonzero_null_efficiency_scenes"]
        == target,
        "spatial_deployability_guards": observed["spatial_deployability_guard_scenes"]
        == target,
        "mean_spatial_recall": observed["mean_spatial_combined_eligible_recall"]
        >= float(DECISION_RULE["mean_eligible_contact_recall_min"]),
        "worst_spatial_recall": observed["worst_spatial_combined_eligible_recall"]
        >= float(DECISION_RULE["worst_eligible_contact_recall_min"]),
        "mean_spatial_minus_null_recall": observed[
            "mean_spatial_minus_null_combined_recall"
        ]
        >= float(DECISION_RULE["mean_spatial_minus_null_combined_recall_min"]),
        "strict_spatial_recall_wins": observed["strict_spatial_recall_win_scenes"]
        >= int(DECISION_RULE["strict_spatial_recall_win_scenes_min"]),
        "mean_incremental_efficiency_ratio": observed[
            "mean_incremental_hit_efficiency_ratio"
        ]
        >= float(DECISION_RULE["mean_incremental_hit_efficiency_ratio_min"]),
        "mean_spatial_coverage": observed["mean_spatial_exact_connected_coverage"]
        >= float(DECISION_RULE["mean_exact_connected_coverage_min"]),
        "worst_spatial_coverage": observed["worst_spatial_exact_connected_coverage"]
        >= float(DECISION_RULE["worst_exact_connected_coverage_min"]),
        "mean_spatial_cycle_ratio": observed[
            "mean_spatial_selected_cycle_rank_ratio"
        ]
        >= float(DECISION_RULE["mean_selected_cycle_rank_ratio_min"]),
        "worst_spatial_cycle_ratio": observed[
            "worst_spatial_selected_cycle_rank_ratio"
        ]
        >= float(DECISION_RULE["worst_selected_cycle_rank_ratio_min"]),
    }
    passed = all(checks.values())
    return {
        "status": (
            "go_source_group_disjoint_confirmation_same_generator"
            if passed
            else "kill_i21_residual_k64_and_matched_hash_null"
        ),
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "candidate_availability_only_no_training_or_production_authority",
    }


def _scientific_guard_terminal(
    payload: Mapping[str, Any], *, completed_scenes: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {"image", "guard", "observed", "maximum", "phase", "evidence"}
    if set(payload) != expected_keys or not isinstance(payload.get("evidence"), Mapping):
        raise E23ContractError("scientific guard payload is malformed")
    observed = _integer(payload.get("observed"), label="guard observed")
    maximum = _integer(payload.get("maximum"), label="guard maximum")
    if observed <= maximum:
        raise E23ContractError("scientific guard payload does not exceed its maximum")
    summary = {
        "images": _integer(completed_scenes, label="guard completed scenes"),
        "completed_scenes": int(completed_scenes),
        "scientific_guard_failure": _jsonable(payload),
    }
    terminal_decision = {
        "status": "kill_i21_residual_k64_scientific_deployability_guard",
        "passed": False,
        "thresholds": dict(DECISION_RULE),
        "observed": {
            "completed_scenes_before_guard": int(completed_scenes),
            "guard": str(payload["guard"]),
            "guard_observed": observed,
            "guard_maximum": maximum,
        },
        "checks": {"scientific_deployability_guard": False},
        "guard_failure": _jsonable(payload),
        "scope": "predeclared_scientific_KILL_not_execution_failure_no_training_authority",
    }
    return summary, terminal_decision


def _load_verified_raw_inputs(
    paths: E23Paths,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[e12.RawScene]]:
    try:
        return e14.load_verified_e12_inputs(
            e14.E14Paths(
                raw_cache_dir=paths.raw_cache_dir,
                calibration_report=paths.calibration_report,
                e12_report=paths.e12_report,
                report=paths.report,
            )
        )
    except Exception as exc:
        raise E23ContractError(str(exc)) from exc


def _scene_arrays(scene: e12.RawScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _validate_raw_arrays(scene.candidate_ids, scene.base_scores)


def _array_states(**arrays: np.ndarray) -> dict[str, tuple[str, bool]]:
    return {
        name: (e12.array_sha256(value), bool(value.flags.writeable))
        for name, value in arrays.items()
    }


def _assert_array_states(
    expected: Mapping[str, tuple[str, bool]], **arrays: np.ndarray
) -> None:
    observed = _array_states(**arrays)
    if observed != dict(expected):
        raise E23ContractError(
            f"frozen candidate/core input mutated: expected {expected}, got {observed}"
        )


def _label_free_scene_provenance(scene: e12.RawScene) -> dict[str, Any]:
    """Hash only E23 inputs; never touch permutation or clean target fields."""

    candidate_ids, raw_logits, _valid = _scene_arrays(scene)
    tiles = _validate_tiles_uint8(scene.tiles_uint8)
    cache_path = Path(scene.cache_path).resolve()
    record = {
        "image": _integer(scene.image_id, label="scene image"),
        "validation_name": str(scene.validation_name),
        "raw_cache_path": str(cache_path),
        "raw_cache_sha256": str(scene.cache_sha256),
        "candidate_ids_sha256": e12.array_sha256(candidate_ids),
        "raw_logits_sha256": e12.array_sha256(raw_logits),
        "tiles_uint8_sha256": e12.array_sha256(tiles),
    }
    if not _is_sha256(record["raw_cache_sha256"]):
        raise E23ContractError("scene raw-cache SHA256 is malformed")
    return record


def _run_scene_pair(
    scene: e12.RawScene,
    *,
    authorized_e22_row: Mapping[str, Any],
    model: positional_ddpm.PositionalDDPM,
    checkpoint_record: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    spatial_cache_dir: Path,
    force_recompute_spatial_cache: bool,
) -> dict[str, Any]:
    candidate_ids, raw_logits, _valid = _scene_arrays(scene)
    tiles = _validate_tiles_uint8(scene.tiles_uint8)
    tiles_sha256 = e12.array_sha256(tiles)
    tiles_writeable = bool(tiles.flags.writeable)
    spatial_logits, cache_record = load_or_compute_spatial_logits(
        cache_dir=spatial_cache_dir,
        image_id=int(scene.image_id),
        validation_name=str(scene.validation_name),
        tiles_uint8=tiles,
        model=model,
        checkpoint_record=checkpoint_record,
        runtime_provenance=runtime_provenance,
        force_recompute=force_recompute_spatial_cache,
    )
    if (
        e12.array_sha256(tiles) != tiles_sha256
        or bool(tiles.flags.writeable) != tiles_writeable
    ):
        raise E23ContractError("cache/inference changed the frozen corrupted tile bag")
    null_logits = hash_null_spatial_logits(tiles_sha256)

    frozen_inputs = {
        "candidate_ids": candidate_ids,
        "raw_logits": raw_logits,
        "spatial_logits": spatial_logits,
        "null_logits": null_logits,
    }
    frozen_states = _array_states(**frozen_inputs)

    # Pinned deployability guard occurs before the potentially large core pool.
    try:
        preflight_spatial_pairs = preflight_spatial_deployability(
            image_id=int(scene.image_id),
            candidate_ids=candidate_ids,
            raw_logits=raw_logits,
            spatial_logits=spatial_logits,
        )
    except Exception:
        _assert_array_states(frozen_states, **frozen_inputs)
        raise
    _assert_array_states(frozen_states, **frozen_inputs)
    spatial_digest = frozen_states["spatial_logits"][0]
    try:
        spatial_result = e23_core.run_i21_residual_candidate_oracle(
            candidate_ids, raw_logits, spatial_logits
        )
    except Exception as exc:
        _assert_array_states(frozen_states, **frozen_inputs)
        raise E23ContractError(f"spatial candidate core failed: {exc}") from exc
    _assert_array_states(frozen_states, **frozen_inputs)
    if len(spatial_result.spatial_pairs) != preflight_spatial_pairs:
        raise E23ContractError("spatial pre-core pair count differs from returned core")
    geometry_maximum = int(
        DECISION_RULE["spatial_geometry_valid_hypotheses_max_each"]
    )
    if len(spatial_result.hypotheses) > geometry_maximum:
        raise E23ScientificGuardFailure(
            image=int(scene.image_id),
            guard="spatial_geometry_valid_hypotheses_max_each",
            observed=len(spatial_result.hypotheses),
            maximum=geometry_maximum,
            phase="immediately_after_spatial_core_before_null_and_labels",
            evidence={
                "spatial_pair_count": len(spatial_result.spatial_pairs),
                "spatial_logits_sha256": spatial_digest,
            },
        )
    try:
        null_result = e23_core.run_i21_residual_candidate_oracle(
            candidate_ids, raw_logits, null_logits
        )
    except Exception as exc:
        _assert_array_states(frozen_states, **frozen_inputs)
        raise E23ContractError(f"matched hash-null candidate core failed: {exc}") from exc
    _assert_array_states(frozen_states, **frozen_inputs)
    row = evaluate_scene_pair(
        scene,
        spatial_result,
        null_result,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        spatial_logits=spatial_logits,
        null_logits=null_logits,
        spatial_cache_record=cache_record,
        authorized_e22_row=authorized_e22_row,
    )
    if row["comparison"]["S_spatial"] != preflight_spatial_pairs:
        raise E23ContractError("reported spatial pair count differs from preflight")
    return row


def _validate_report_header(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_digest: str,
) -> tuple[list[Mapping[str, Any]], list[int]]:
    required = {
        "schema_version",
        "schema",
        "experiment",
        "status",
        "stage",
        "protocol",
        "protocol_sha256",
        "run_contract",
        "run_contract_sha256",
        "rows",
        "completed_images",
        "decision",
    }
    if not required.issubset(report):
        raise E23ContractError("existing E23 report header is incomplete")
    if (
        _integer(report.get("schema_version"), label="E23 schema version")
        != SCHEMA_VERSION
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("protocol") != E23_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E23_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E23ContractError("existing E23 report belongs to different bytes")
    rows = report.get("rows")
    completed = report.get("completed_images")
    if not isinstance(rows, list) or not isinstance(completed, list):
        raise E23ContractError("existing E23 progress fields are malformed")
    ids = [_integer(value, label="completed image") for value in completed]
    if (
        len(rows) != len(ids)
        or ids != list(e12.CALIBRATION_IDS[: len(ids)])
        or [row.get("image") for row in rows if isinstance(row, Mapping)] != ids
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise E23ContractError("existing E23 rows are not an exact ordered prefix")
    return list(rows), ids


def run_gate(
    paths: E23Paths,
) -> Mapping[str, Any]:
    """Run or fully replay the one frozen E23 spatial-vs-null ceiling."""

    started = time.perf_counter()
    runtime = _runtime_provenance()
    if sys.pycache_prefix is None:
        raise E23ContractError("E23 Python bytecode prefix is not configured")
    _require_e_drive(Path(sys.pycache_prefix), label="Python bytecode prefix")
    report_path = _require_e_drive(paths.report, label="E23 report")
    spatial_cache_dir = _require_e_drive(
        paths.spatial_cache_dir, label="spatial-logit cache directory"
    )
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e22_report_path = _require_e_drive(paths.e22_report, label="E22 report")
    checkpoint_path = _require_e_drive(paths.checkpoint, label="I21 checkpoint")
    calibration_path = paths.calibration_report.resolve()
    if report_path.suffix.lower() != ".json":
        raise E23ContractError("E23 report must be a .json file")
    if report_path in {e12_report_path, e22_report_path, checkpoint_path, calibration_path}:
        raise E23ContractError("E23 report must not overwrite an input")
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(
        spatial_cache_dir
    ):
        raise E23ContractError("E23 report must not be inside an input/cache directory")

    e22_report = _verify_e22_kill(e22_report_path)
    e12_report, calibration, scenes = _load_verified_raw_inputs(paths)
    if not isinstance(calibration, Mapping):
        raise E23ContractError("verified calibration payload is malformed")
    if tuple(int(scene.image_id) for scene in scenes) != e12.CALIBRATION_IDS:
        raise E23ContractError("E23 inputs are not exact E12 scenes 10..17")
    model, checkpoint_record = load_frozen_i21_model(checkpoint_path)
    scene_records = [_label_free_scene_provenance(scene) for scene in scenes]
    source_provenance = _source_provenance()
    contract = {
        "protocol_sha256": e12.canonical_digest(E23_PROTOCOL),
        "e22_report": {
            "path": str(e22_report_path),
            "sha256": EXPECTED_E22_REPORT_SHA256,
            "run_contract_sha256": EXPECTED_E22_RUN_CONTRACT_SHA256,
            "stage": EXPECTED_E22_STAGE,
        },
        "e12_report": {
            "path": str(e12_report_path),
            "sha256": e14.EXPECTED_E12_REPORT_SHA256,
        },
        "calibration_report": {
            "path": str(calibration_path),
            "sha256": e12.CALIBRATION_REPORT_SHA256,
        },
        "raw_cache_dir": str(raw_cache_dir),
        "raw_scenes": scene_records,
        "raw_scenes_sha256": e12.canonical_digest(scene_records),
        "checkpoint": checkpoint_record,
        "spatial_cache_dir": str(spatial_cache_dir),
        "null_rule_sha256": e12.canonical_digest(E23_PROTOCOL["matched_budget_null"]),
        "orientation": {
            "tile_orientation_degrees": [0],
            "forbidden_tile_orientation_degrees": [90, 180, 270],
            "reflection": False,
            "rcce4_variants_are_adjacency_orderings_not_rotations": True,
        },
        "report": str(report_path),
        "source_provenance": source_provenance,
        "runtime_provenance": runtime,
    }
    contract_digest = e12.canonical_digest(contract)
    e22_rows = {
        _integer(row.get("image"), label="E22 row image"): row
        for row in e22_report["rows"]
    }

    existing: Mapping[str, Any] | None = None
    existing_rows: list[Mapping[str, Any]] = []
    existing_ids: list[int] = []
    if report_path.is_file():
        existing = _load_json(report_path, label="existing E23 report")
        existing_rows, existing_ids = _validate_report_header(
            existing, contract=contract, contract_digest=contract_digest
        )

    # A complete report is accepted only after a full cache/core/label replay.
    replayed_rows: list[Mapping[str, Any]] = []
    for index, image in enumerate(existing_ids):
        replayed = _run_scene_pair(
            scenes[index],
            authorized_e22_row=e22_rows[image],
            model=model,
            checkpoint_record=checkpoint_record,
            runtime_provenance=runtime,
            spatial_cache_dir=spatial_cache_dir,
            force_recompute_spatial_cache=True,
        )
        if replayed != existing_rows[index]:
            raise E23ContractError(f"existing E23 row replay drifted for image {image}")
        replayed_rows.append(replayed)
    if existing is not None and existing.get("status") == "complete":
        guard_failure = existing.get("guard_failure")
        if guard_failure is not None:
            if not isinstance(guard_failure, Mapping):
                raise E23ContractError("complete scientific guard payload is malformed")
            if len(existing_rows) >= len(e12.CALIBRATION_IDS):
                raise E23ContractError("scientific guard report has no failing scene")
            failing_index = len(existing_rows)
            failing_image = int(scenes[failing_index].image_id)
            try:
                _run_scene_pair(
                    scenes[failing_index],
                    authorized_e22_row=e22_rows[failing_image],
                    model=model,
                    checkpoint_record=checkpoint_record,
                    runtime_provenance=runtime,
                    spatial_cache_dir=spatial_cache_dir,
                    force_recompute_spatial_cache=True,
                )
            except E23ScientificGuardFailure as exc:
                if exc.payload != guard_failure:
                    raise E23ContractError(
                        "scientific guard complete replay payload drifted"
                    ) from exc
            else:
                raise E23ContractError(
                    "scientific guard did not recur during complete replay"
                )
            expected_summary, expected_decision = _scientific_guard_terminal(
                guard_failure, completed_scenes=len(existing_rows)
            )
            if (
                set(existing)
                != {
                    "schema_version",
                    "schema",
                    "experiment",
                    "status",
                    "stage",
                    "protocol",
                    "protocol_sha256",
                    "run_contract",
                    "run_contract_sha256",
                    "rows",
                    "completed_images",
                    "summary",
                    "decision",
                    "guard_failure",
                    "runtime_seconds",
                }
                or existing.get("summary") != expected_summary
                or existing.get("decision") != expected_decision
                or existing.get("stage") != expected_decision["status"]
                or existing.get("completed_images")
                != list(e12.CALIBRATION_IDS[: len(existing_rows)])
                or _integer(guard_failure.get("image"), label="guard image")
                != failing_image
            ):
                raise E23ContractError("complete scientific KILL payload drifted")
            _finite(
                existing.get("runtime_seconds"),
                label="existing E23 guard runtime",
                minimum=0.0,
            )
            return existing
        if len(existing_rows) != len(e12.CALIBRATION_IDS):
            raise E23ContractError("complete E23 report has incomplete rows")
        expected_summary = summarize(existing_rows)
        expected_decision = decision(expected_summary)
        if (
            set(existing)
            != {
                "schema_version",
                "schema",
                "experiment",
                "status",
                "stage",
                "protocol",
                "protocol_sha256",
                "run_contract",
                "run_contract_sha256",
                "rows",
                "completed_images",
                "summary",
                "decision",
                "runtime_seconds",
            }
            or existing.get("summary") != expected_summary
            or existing.get("decision") != expected_decision
            or existing.get("stage") != expected_decision["status"]
            or existing.get("completed_images") != list(e12.CALIBRATION_IDS)
        ):
            raise E23ContractError("complete E23 terminal payload drifted")
        _finite(
            existing.get("runtime_seconds"),
            label="existing E23 runtime",
            minimum=0.0,
        )
        return existing

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "upright_i21_residual_k64_vs_hash_null_candidate_ceiling",
        "protocol": E23_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E23_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rows": list(existing_rows),
        "completed_images": list(existing_ids),
        "decision": {"status": "not_run"},
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _atomic_write_json(report_path, output)
    try:
        for index in range(len(existing_rows), len(scenes)):
            scene = scenes[index]
            image = int(scene.image_id)
            row = _run_scene_pair(
                scene,
                authorized_e22_row=e22_rows[image],
                model=model,
                checkpoint_record=checkpoint_record,
                runtime_provenance=runtime,
                spatial_cache_dir=spatial_cache_dir,
                force_recompute_spatial_cache=True,
            )
            output["rows"].append(row)
            output["completed_images"].append(image)
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
        
        result_summary = summarize(output["rows"])
        result_decision = decision(result_summary)
        output["summary"] = result_summary
        output["decision"] = result_decision
        output["status"] = "complete"
        output["stage"] = result_decision["status"]
        output["runtime_seconds"] = float(time.perf_counter() - started)
        _atomic_write_json(report_path, output)
        return output
    except E23ScientificGuardFailure as exc:
        guard_summary, guard_decision = _scientific_guard_terminal(
            exc.payload, completed_scenes=len(output["rows"])
        )
        output["summary"] = guard_summary
        output["decision"] = guard_decision
        output["guard_failure"] = exc.payload
        output["status"] = "complete"
        output["stage"] = guard_decision["status"]
        output["runtime_seconds"] = float(time.perf_counter() - started)
        _atomic_write_json(report_path, output)
        return output
    except Exception as exc:
        output["status"] = "failed"
        output["error"] = f"{type(exc).__name__}: {exc}"
        output["runtime_seconds"] = float(time.perf_counter() - started)
        _atomic_write_json(report_path, output)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen CPU-only E23 I21-residual K64 vs hash-null ceiling."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e22-report", type=Path, default=DEFAULT_E22_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--spatial-cache-dir", type=Path, default=DEFAULT_SPATIAL_CACHE_DIR
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gate(
        E23Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e22_report=args.e22_report,
            checkpoint=args.checkpoint,
            spatial_cache_dir=args.spatial_cache_dir,
            report=args.report,
        )
    )
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "stage": result["stage"],
                "passed": bool(result["decision"]["passed"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
