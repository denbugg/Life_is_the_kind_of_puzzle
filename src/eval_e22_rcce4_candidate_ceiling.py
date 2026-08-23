"""Frozen E22 RCCE-4 full-union all-emitter candidate ceiling.

The label-free core receives exactly the byte-pinned raw affinity candidate
IDs and their four U/D/L/R score rows.  Permutation labels are opened only
after the complete RCCE-4 pool has returned.  This evaluator measures contact
retention and exact relative connectivity; it never builds an absolute board.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import skimage

import e22_rcce4_candidate_oracle as rcce
import e21_posegraph_candidate_oracle as e21_pose
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
import eval_e21_posegraph_candidate_ceiling as e21_eval


class E22ContractError(RuntimeError):
    """The frozen E22 protocol, lineage, or oracle algebra drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e22-rcce4-candidate-ceiling-report-v1"
EXPERIMENT = "e22_rcce4_full_union_all_emitter_candidate_ceiling_v1"

EXPECTED_E12_REPORT_SHA256 = e14.EXPECTED_E12_REPORT_SHA256
EXPECTED_E21_REPORT_SHA256 = (
    "0c43099860c7a16f5e968a8ea6cf637293cd639d9b86e342797ef68c5d53e724"
)
EXPECTED_E21_RUN_CONTRACT_SHA256 = (
    "1cff1e4ca733a24d69e9b68b410e75ef453f6db712b2709bad6db9f3ed73a992"
)
EXPECTED_E21_PROTOCOL_SHA256 = (
    "134b1192fcdeb3d63583af938b53b6906930ab725a53df01015836047cd2a04f"
)
EXPECTED_E21_STAGE = "kill_raw_CC96_anchor_top8_candidate_pool"
EXPECTED_RUNTIME_PROVENANCE = dict(e21_eval.EXPECTED_RUNTIME_PROVENANCE)

MAX_DIRECTED_MEMBERSHIPS = 576 * 128
MAX_UNORDERED_PAIRS = 576 * 128
MAX_DIRECTIONAL_OBSERVATIONS = 4 * 576 * 128
MAX_RCCE4_PRECLAIMS = 4 * 576 * 128
MAX_GEOMETRY_VALID_HYPOTHESES = 4 * 576 * 128

DECISION_RULE: dict[str, float | int] = {
    "completed_scenes": 8,
    "emitters_each": 576,
    "all_bounds_scenes": 8,
    "true_relation_scenes": 8,
    "legal_origin_scenes": 8,
    "positive_eligible_denominator_scenes": 8,
    "exact_postfilter_survival_scenes": 8,
    "mean_eligible_contact_recall_min": 0.90,
    "worst_eligible_contact_recall_min": 0.80,
    "mean_exact_connected_coverage_min": 0.30,
    "worst_exact_connected_coverage_min": 0.20,
    "mean_selected_cycle_rank_ratio_min": 0.05,
    "worst_selected_cycle_rank_ratio_min": 0.01,
}

E22_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e22-rcce4-full-union-all-emitter-ceiling-v1",
    "role": "CPU_label_after_core_discovery_ceiling_not_isolated_ablation",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "authorization": {
        "e21_report_sha256": EXPECTED_E21_REPORT_SHA256,
        "e21_run_contract_sha256": EXPECTED_E21_RUN_CONTRACT_SHA256,
        "e21_protocol_sha256": EXPECTED_E21_PROTOCOL_SHA256,
        "required_status": "complete",
        "required_stage": EXPECTED_E21_STAGE,
    },
    "core_boundary": {
        "arguments": ["candidate_ids", "raw_logits"],
        "candidate_ids": "contiguous_int64_576x128",
        "raw_logits": "contiguous_float32_4x576x128_UDLR",
        "common_finite_mask": True,
        "derives_dense_and_CC96_internally": True,
        "forbidden_inputs": [
            "labels",
            "pixels",
            "permutation",
            "dense_matrices",
            "components",
            "target",
        ],
    },
    "upstream_support": {
        "source": "already_frozen_ordered_dual_affinity_K64_plus_K64_union",
        "storage_width": 128,
        "additional_E22_truncation": False,
        "raw_logits": (
            "row_listwise_metadata_preserved_per_finite_slot_never_averaged_"
            "summed_ranked_or_thresholded_for_admission"
        ),
    },
    "components": {
        "dense_conversion": "frozen_Rank96_CPU_float32",
        "builder": "corrected_exact_buddies",
        "max_edges": 96,
        "min_margin": 0.0,
        "full_partition_including_singletons": True,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
    },
    "upright_orientation_clarification": {
        "tile_orientation_degrees": [0],
        "forbidden_tile_orientation_degrees": [90, 180, 270],
        "reflection": False,
        "four_pair_variants": (
            "upright_adjacency_orderings_b_right_of_a_a_right_of_b_"
            "b_below_a_a_below_b_not_tile_rotations"
        ),
    },
    "pair_lift": {
        "emitters": 576,
        "pair": "canonical_unordered_OR_of_either_directed_membership",
        "pair_order": "a_then_b_ascending",
        "upright_adjacency_order": ["a_b_R", "b_a_R", "a_b_D", "b_a_D"],
        "metadata": [
            "RIGHT_a_b_and_optional_LEFT_b_a",
            "RIGHT_b_a_and_optional_LEFT_a_b",
            "DOWN_a_b_and_optional_UP_b_a",
            "DOWN_b_a_and_optional_UP_a_b",
        ],
        "reverse_membership": "metadata_not_duplicate_or_admission_average",
        "same_component_claims": "removed_after_literal_four_per_pair_lift",
    },
    "relations": {
        "key": "u_lt_v_plus_exact_signed_offset",
        "alternative_offsets_retained": True,
        "physical_seams": "unique",
        "geometry_filter": [
            "all_supporting_endpoints_adjacent",
            "two_component_coordinate_sets_no_collision",
            "pair_bbox_height_width_at_most_24",
        ],
        "incidental_contacts": "ignored_neither_evidence_nor_rejection",
    },
    "bounds": {
        "directed_memberships_max": MAX_DIRECTED_MEMBERSHIPS,
        "unordered_pairs_max": MAX_UNORDERED_PAIRS,
        "finite_directional_logit_observations_max": MAX_DIRECTIONAL_OBSERVATIONS,
        "rcce4_preclaims_equal_four_per_pair": True,
        "rcce4_preclaims_max": MAX_RCCE4_PRECLAIMS,
        "geometry_valid_hypotheses_max": MAX_GEOMETRY_VALID_HYPOTHESES,
        "truncate": False,
    },
    "measurement": {
        "labels": "evaluator_only_after_core_returns",
        "component_purity": "whole_component_one_exact_truth_minus_local_translation",
        "eligible_denominator": (
            "all_GT_canonical_undirected_cardinal_seams_crossing_two_distinct_"
            "whole_pure_CC96_components"
        ),
        "eligible_denominator_must_be_positive": True,
        "candidate_hit": "unordered_affinity_pair_OR_membership",
        "also_report": "unconditional_all_cross_component_true_seam_recall",
        "postfilter_survival": (
            "hit_eligible_seam_exact_relation_and_physical_seam_in_geometry_valid_pool"
        ),
        "required_postfilter_survival": 1.0,
    },
    "oracle_connectivity": {
        "relation_truth": "both_whole_components_pure_and_signed_delta_exact",
        "union": "all_true_hypotheses_once_in_canonical_order",
        "potential": "translation_node_minus_translation_parent",
        "pure_isolated_components_included": True,
        "collision_allowed": False,
        "bbox_height_width_max": 24,
        "selection_rank": [
            "exact_connected_tiles_desc",
            "accepted_relations_desc",
            "cycle_rank_desc",
            "minimum_tile_asc",
            "canonical_translations_asc",
        ],
        "legal_origins": "analytic_after_normalisation",
        "absolute_board": False,
    },
    "decision": dict(DECISION_RULE),
    "routing": {
        "pass": "open_separately_frozen_E23_source_group_disjoint_confirmation",
        "fail": "close_exact_existing_affinity_full_union_generator_without_resweep",
        "immediate_GPU_training": False,
    },
    "excluded": [
        "clean_scores",
        "clean_pixels",
        "labels_in_core",
        "learned_shortlist",
        "topk_or_threshold_sweep",
        "triangle_filter",
        "iterative_growth",
        "board",
        "residual",
        "placement",
        "neighbour",
        "SSIM",
        "NLM",
        "absolute_origin_choice",
        "rotation",
        "reflection",
        "GPU",
        "diffusion",
        "target_submission_data",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_E21_REPORT = Path(
    "E:/pazzle_work/posegraph_e21/cc96_top8_anchor_candidate_ceiling_v1.json"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/posegraph_e22/cc96_all_emitter_full_union_candidate_ceiling_v1.json"
)


@dataclass(frozen=True)
class E22Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e21_report: Path
    report: Path


@dataclass(frozen=True)
class OracleCluster:
    component_ids: tuple[int, ...]
    translations: tuple[tuple[int, int, int], ...]
    relative_entries: tuple[tuple[int, int, int], ...]
    accepted_relations: tuple[tuple[int, int, int, int], ...]
    exact_connected_tiles: int
    exact_connected_coverage: float
    accepted_relation_count: int
    cycle_rank: int
    cycle_rank_ratio: float
    minimum_tile: int
    bbox: tuple[int, int, int, int]
    bbox_height: int
    bbox_width: int
    legal_origin_bounds: tuple[int, int, int, int]
    legal_origin_count: int


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise E22ContractError(f"{label} is not an integer")
    return int(value)


def _finite(
    value: object, *, label: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise E22ContractError(f"{label} is not numeric")
    observed = float(value)
    if not math.isfinite(observed):
        raise E22ContractError(f"{label} is not finite")
    if not minimum <= observed <= maximum:
        raise E22ContractError(f"{label} is outside [{minimum}, {maximum}]")
    return observed


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E22ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E22ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E22ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E22 report")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime_provenance() -> dict[str, str]:
    import cv2

    observed = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scikit_image": str(skimage.__version__),
        "opencv": str(cv2.__version__),
        "opencv_build_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "torch": str(e12.torch.__version__),
        "execution": "CPU_only",
        "scipy": str(scipy.__version__),
    }
    if observed != EXPECTED_RUNTIME_PROVENANCE:
        raise E22ContractError(
            f"E22 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e22_rcce4_candidate_oracle.py": source / "e22_rcce4_candidate_oracle.py",
        "e21_posegraph_candidate_oracle.py": source
        / "e21_posegraph_candidate_oracle.py",
        "eval_buddies_ssim_budget.py": source / "eval_buddies_ssim_budget.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_e21_posegraph_candidate_ceiling.py": source
        / "eval_e21_posegraph_candidate_ceiling.py",
        "eval_e22_rcce4_candidate_ceiling.py": Path(__file__).resolve(),
        "eval_seeded_qap.py": source / "eval_seeded_qap.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise E22ContractError("E22 source file is missing: " + ", ".join(missing))
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _verify_e21_kill(path: Path) -> Mapping[str, Any]:
    """Authenticate the exact E21 KILL bytes and their replayable lineage."""

    resolved = _require_e_drive(path, label="E21 report")
    if not resolved.is_file():
        raise E22ContractError(f"E21 report is missing: {resolved}")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E21_REPORT_SHA256:
        raise E22ContractError(
            "E21 report SHA256 mismatch: "
            f"expected {EXPECTED_E21_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E21 report")
    contract = report.get("run_contract")
    rows = report.get("rows")
    if (
        _integer(report.get("schema_version"), label="E21 schema version")
        != e21_eval.SCHEMA_VERSION
        or report.get("schema") != e21_eval.REPORT_SCHEMA
        or report.get("experiment") != e21_eval.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("stage") != EXPECTED_E21_STAGE
        or report.get("protocol") != e21_eval.E21_PROTOCOL
        or report.get("protocol_sha256") != EXPECTED_E21_PROTOCOL_SHA256
        or e12.canonical_digest(report.get("protocol"))
        != EXPECTED_E21_PROTOCOL_SHA256
        or report.get("run_contract_sha256")
        != EXPECTED_E21_RUN_CONTRACT_SHA256
        or not isinstance(contract, Mapping)
        or e12.canonical_digest(contract) != EXPECTED_E21_RUN_CONTRACT_SHA256
        or report.get("completed_images") != list(e12.CALIBRATION_IDS)
        or not isinstance(rows, list)
        or len(rows) != len(e12.CALIBRATION_IDS)
    ):
        raise E22ContractError("E21 authorization contract drifted")
    _finite(
        report.get("runtime_seconds"),
        label="E21 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    try:
        expected_summary = e21_eval.summarize(rows)
        expected_decision = e21_eval.decision(expected_summary)
    except Exception as exc:
        raise E22ContractError(f"E21 terminal payload is malformed: {exc}") from exc
    if (
        report.get("summary") != expected_summary
        or report.get("decision") != expected_decision
        or expected_decision.get("passed") is not False
        or expected_decision.get("status") != EXPECTED_E21_STAGE
    ):
        raise E22ContractError("E21 KILL decision drifted")

    frozen_sources = contract.get("source_provenance")
    if not isinstance(frozen_sources, Mapping):
        raise E22ContractError("E21 source provenance is malformed")
    source = Path(__file__).resolve().parent
    for name, expected in frozen_sources.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not _is_sha256(expected)
        ):
            raise E22ContractError("E21 source provenance entry is malformed")
        source_path = source / name
        if not source_path.is_file():
            raise E22ContractError(f"source shared with E21 is missing: {source_path}")
        observed = e12.sha256_file(source_path)
        if observed != expected:
            raise E22ContractError(
                f"source shared with E21 drifted for {name}: "
                f"expected {expected}, got {observed}"
            )
    if contract.get("runtime_provenance") != _runtime_provenance():
        raise E22ContractError("E21-to-E22 runtime provenance drifted")
    return report


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
            raise E22ContractError("core payload contains a non-finite float")
        return value
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    raise E22ContractError(f"core payload contains unsupported type {type(value)}")


def _stream_digest(values: Sequence[Any] | Any) -> str:
    """Canonical length-framed digest without materialising a large JSON list."""

    digest = hashlib.sha256(b"pazzle-e22-stream-v1\0")
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
    forbidden_exact = {
        "board",
        "canvas",
        "residual",
        "placement",
        "neighbour",
        "ssim",
        "nlm",
        "target",
        "target_uint8",
        "target_pixels",
        "ground_truth",
        "permutation",
        "pixel",
        "pixels",
        "clean_score",
        "clean_scores",
        "clean_pixels",
        "rotation",
        "reflection",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in forbidden_exact:
                raise E22ContractError(f"core payload contains forbidden key {key}")
            _check_forbidden_payload_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_forbidden_payload_keys(item)


def _strict_vector(
    value: object, *, label: str, readonly: bool = False
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (e12.NFRAG,)
        or value.dtype != np.int64
        or not value.flags.c_contiguous
        or (readonly and value.flags.writeable)
    ):
        suffix = " read-only" if readonly else ""
        raise E22ContractError(
            f"{label} must be a contiguous{suffix} int64 ({e12.NFRAG},) vector"
        )
    return value


def _validate_permutation(value: object) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (e12.NFRAG,)
        or array.dtype.kind not in "iu"
        or isinstance(value, (bool, np.bool_))
    ):
        raise E22ContractError("permutation must be an integer (576,) vector")
    permutation = np.ascontiguousarray(array.astype(np.int64, copy=False))
    if not np.array_equal(
        np.sort(permutation), np.arange(e12.NFRAG, dtype=np.int64)
    ):
        raise E22ContractError("permutation is not an input-tile to cell bijection")
    return permutation


def _validate_raw_arrays(
    candidate_ids: object, raw_logits: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        not isinstance(candidate_ids, np.ndarray)
        or candidate_ids.shape != (e12.NFRAG, 128)
        or candidate_ids.dtype != np.int64
        or not candidate_ids.flags.c_contiguous
    ):
        raise E22ContractError(
            "candidate_ids must be contiguous int64[576,128] without coercion"
        )
    if (
        not isinstance(raw_logits, np.ndarray)
        or raw_logits.shape != (4, e12.NFRAG, 128)
        or raw_logits.dtype != np.float32
        or not raw_logits.flags.c_contiguous
    ):
        raise E22ContractError(
            "raw_logits must be contiguous float32[4,576,128] in U,D,L,R order"
        )
    if bool(np.isnan(raw_logits).any()) or bool(np.isposinf(raw_logits).any()):
        raise E22ContractError(
            "raw_logits may contain only finite values or -inf padding"
        )
    masks = np.isfinite(raw_logits)
    if any(not np.array_equal(masks[0], masks[index]) for index in range(1, 4)):
        raise E22ContractError("raw U,D,L,R logits do not share one finite mask")
    valid = masks[0]
    expanded_valid = np.broadcast_to(valid, raw_logits.shape)
    if not bool(np.isneginf(raw_logits[~expanded_valid]).all()):
        raise E22ContractError("invalid raw logit padding must be -inf")
    if not bool(valid.any(axis=1).all()):
        raise E22ContractError("every affinity row must contain a valid membership")
    for anchor in range(e12.NFRAG):
        values = candidate_ids[anchor, valid[anchor]]
        if (
            bool((values < 0).any())
            or bool((values >= e12.NFRAG).any())
            or bool((values == anchor).any())
            or len(set(map(int, values))) != int(values.size)
        ):
            raise E22ContractError(
                f"valid candidate IDs for emitter {anchor} are not unique non-self 0..575"
            )
    return candidate_ids, raw_logits, valid


def _component_entries(component: Any) -> tuple[tuple[int, int, int], ...]:
    raw = getattr(component, "entries", None)
    if not isinstance(raw, tuple) or not raw:
        raise E22ContractError("component entries are missing or mutable")
    entries: list[tuple[int, int, int]] = []
    for entry in raw:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise E22ContractError("component contains a malformed entry")
        entries.append(tuple(_integer(item, label="component entry") for item in entry))
    return tuple(entries)


def _validate_components(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = getattr(result, "components", None)
    if not isinstance(components, tuple) or not components:
        raise E22ContractError("candidate component partition is malformed")
    owner = _strict_vector(result.owner, label="component owner", readonly=True)
    local_rows = _strict_vector(
        result.local_rows, label="component local rows", readonly=True
    )
    local_cols = _strict_vector(
        result.local_cols, label="component local cols", readonly=True
    )
    seen_tiles: set[int] = set()
    observed_order: list[tuple[int, int, tuple[tuple[int, int, int], ...]]] = []
    expected_nontrivial: list[int] = []
    for expected_id, component in enumerate(components):
        component_id = _integer(
            getattr(component, "component_id", None), label="component ID"
        )
        if component_id != expected_id:
            raise E22ContractError("component IDs are not dense deterministic IDs")
        entries = _component_entries(component)
        if entries != tuple(sorted(entries)):
            raise E22ContractError("component entries are not canonical")
        tiles = [tile for tile, _row, _col in entries]
        positions = [(row, col) for _tile, row, col in entries]
        if (
            len(set(tiles)) != len(tiles)
            or len(set(positions)) != len(positions)
            or any(not 0 <= tile < e12.NFRAG for tile in tiles)
            or any(row < 0 or col < 0 for row, col in positions)
            or min(row for row, _col in positions) != 0
            or min(col for _row, col in positions) != 0
            or max(row for row, _col in positions) >= 24
            or max(col for _row, col in positions) >= 24
        ):
            raise E22ContractError("component geometry is invalid or not normalized")
        if seen_tiles.intersection(tiles):
            raise E22ContractError("component partition duplicates a tile")
        seen_tiles.update(tiles)
        for tile, row, col in entries:
            if (
                int(owner[tile]) != component_id
                or int(local_rows[tile]) != row
                or int(local_cols[tile]) != col
            ):
                raise E22ContractError("owner/local coordinate arrays drifted")
        observed_order.append((-len(entries), min(tiles), entries))
        if len(entries) >= 2:
            expected_nontrivial.append(component_id)
    if seen_tiles != set(range(e12.NFRAG)):
        raise E22ContractError("component partition does not cover all 576 tiles")
    if observed_order != sorted(observed_order):
        raise E22ContractError("component ordering drifted")
    if (
        not isinstance(result.nontrivial_component_ids, frozenset)
        or result.nontrivial_component_ids != frozenset(expected_nontrivial)
    ):
        raise E22ContractError("nontrivial component IDs drifted")
    return owner, local_rows, local_cols


def _directed_memberships(
    candidate_ids: np.ndarray, valid: np.ndarray
) -> dict[tuple[int, int], int]:
    output: dict[tuple[int, int], int] = {}
    for source in range(e12.NFRAG):
        for slot in np.flatnonzero(valid[source]):
            output[(source, int(candidate_ids[source, slot]))] = int(slot)
    return output


def _expected_affinity_pairs(
    candidate_ids: np.ndarray, valid: np.ndarray
) -> tuple[tuple[int, int, int | None, int | None], ...]:
    directed = _directed_memberships(candidate_ids, valid)
    keys = sorted({tuple(sorted((source, target))) for source, target in directed})
    return tuple(
        (a, b, directed.get((a, b)), directed.get((b, a))) for a, b in keys
    )


def component_truth_shifts(
    result: Any, permutation: object
) -> dict[int, tuple[int, int] | None]:
    _validate_components(result)
    truth = _validate_permutation(permutation)
    shifts: dict[int, tuple[int, int] | None] = {}
    for component in result.components:
        component_id = int(component.component_id)
        offsets = {
            (
                int(truth[tile] // 24) - local_row,
                int(truth[tile] % 24) - local_col,
            )
            for tile, local_row, local_col in _component_entries(component)
        }
        shifts[component_id] = next(iter(offsets)) if len(offsets) == 1 else None
    return shifts


def ground_truth_seams(
    permutation: object,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return every upright physical R/D seam, independent of candidate output."""

    truth = _validate_permutation(permutation)
    tile_at_cell = np.empty(e12.NFRAG, dtype=np.int64)
    tile_at_cell[truth] = np.arange(e12.NFRAG, dtype=np.int64)
    seams: list[tuple[int, int, int, int]] = []
    for row in range(24):
        for col in range(24):
            cell = row * 24 + col
            first = int(tile_at_cell[cell])
            if col < 23:
                seams.append((first, int(tile_at_cell[cell + 1]), 0, 1))
            if row < 23:
                seams.append((first, int(tile_at_cell[cell + 24]), 1, 0))
    if len(seams) != 2 * 24 * 23 or len(set(seams)) != len(seams):
        raise E22ContractError("upright ground-truth seam enumeration drifted")
    return tuple(seams)


def _relation_tuple(value: Any, *, label: str) -> tuple[int, int, int, int]:
    relation = getattr(value, "relation", None)
    if not isinstance(relation, tuple) or len(relation) != 4:
        relation = tuple(
            getattr(value, name, None) for name in ("u", "v", "dr", "dc")
        )
    if not isinstance(relation, tuple) or len(relation) != 4:
        raise E22ContractError(f"{label} relation is malformed")
    u, v, dr, dc = tuple(_integer(item, label=f"{label} relation") for item in relation)
    return u, v, dr, dc


def seam_relation(
    seam: tuple[int, int, int, int],
    *,
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> tuple[int, int, int, int]:
    first, second, dy, dx = seam
    first_component = int(owner[first])
    second_component = int(owner[second])
    if first_component == second_component:
        raise E22ContractError("same-component seam has no lifted relation")
    delta = (
        int(local_rows[first]) + dy - int(local_rows[second]),
        int(local_cols[first]) + dx - int(local_cols[second]),
    )
    if first_component < second_component:
        return first_component, second_component, delta[0], delta[1]
    return second_component, first_component, -delta[0], -delta[1]


def true_pose_hypotheses(
    result: Any,
    shifts: Mapping[int, tuple[int, int] | None],
) -> tuple[Any, ...]:
    output: list[Any] = []
    for hypothesis in result.hypotheses:
        u, v, dr, dc = _relation_tuple(hypothesis, label="hypothesis")
        left = shifts.get(u)
        right = shifts.get(v)
        if (
            left is not None
            and right is not None
            and (right[0] - left[0], right[1] - left[1]) == (dr, dc)
        ):
            output.append(hypothesis)
    identities = [
        (*_relation_tuple(value, label="true hypothesis"), int(value.hypothesis_id))
        for value in output
    ]
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise E22ContractError("oracle-true hypotheses are not canonical and unique")
    return tuple(output)


class _PotentialDSU:
    """Independent signed-translation DSU with merge-time geometry checks."""

    def __init__(
        self,
        components: Sequence[Any],
        active_component_ids: Sequence[int],
    ) -> None:
        self.components = tuple(components)
        self.active = frozenset(map(int, active_component_ids))
        self.parent = {component_id: component_id for component_id in self.active}
        self.size = {component_id: 1 for component_id in self.active}
        self.delta = {component_id: (0, 0) for component_id in self.active}
        self.members = {
            component_id: {component_id} for component_id in self.active
        }

    def find(self, component_id: int) -> tuple[int, tuple[int, int]]:
        if component_id not in self.parent:
            raise E22ContractError("relation touches an impure component")
        parent = self.parent[component_id]
        if parent == component_id:
            return component_id, (0, 0)
        root, parent_delta = self.find(parent)
        own_delta = self.delta[component_id]
        total = (
            own_delta[0] + parent_delta[0],
            own_delta[1] + parent_delta[1],
        )
        self.parent[component_id] = root
        self.delta[component_id] = total
        return root, total

    def _validate_root_geometry(self, root: int) -> None:
        positions: dict[tuple[int, int], int] = {}
        for component_id in sorted(self.members[root]):
            observed_root, shift = self.find(component_id)
            if observed_root != root:
                raise E22ContractError("DSU member/root algebra drifted")
            for tile, row, col in _component_entries(self.components[component_id]):
                position = (row + shift[0], col + shift[1])
                if position in positions and positions[position] != tile:
                    raise E22ContractError("oracle union creates a tile collision")
                positions[position] = tile
        if not positions:
            raise E22ContractError("oracle DSU root is empty")
        rows = [row for row, _col in positions]
        cols = [col for _row, col in positions]
        if max(rows) - min(rows) + 1 > 24 or max(cols) - min(cols) + 1 > 24:
            raise E22ContractError("oracle union exceeds the upright 24x24 span")

    def union(self, u: int, v: int, dr: int, dc: int) -> bool:
        u = _integer(u, label="union u")
        v = _integer(v, label="union v")
        dr = _integer(dr, label="union dr")
        dc = _integer(dc, label="union dc")
        root_u, shift_u = self.find(u)
        root_v, shift_v = self.find(v)
        if root_u == root_v:
            observed = (
                shift_v[0] - shift_u[0],
                shift_v[1] - shift_u[1],
            )
            if observed != (dr, dc):
                raise E22ContractError("oracle cycle contradicts signed translations")
            return False

        # T(root_v)-T(root_u), independently derived from T(v)-T(u).
        root_v_from_u = (
            dr + shift_u[0] - shift_v[0],
            dc + shift_u[1] - shift_v[1],
        )
        keep_u = (self.size[root_u], -root_u) >= (self.size[root_v], -root_v)
        if keep_u:
            self.parent[root_v] = root_u
            self.delta[root_v] = root_v_from_u
            self.size[root_u] += self.size[root_v]
            self.members[root_u].update(self.members.pop(root_v))
            root = root_u
        else:
            self.parent[root_u] = root_v
            self.delta[root_u] = (-root_v_from_u[0], -root_v_from_u[1])
            self.size[root_v] += self.size[root_u]
            self.members[root_v].update(self.members.pop(root_u))
            root = root_v
        self._validate_root_geometry(root)
        return True


def _make_cluster(
    dsu: _PotentialDSU,
    component_ids: Sequence[int],
    relations: Sequence[tuple[int, int, int, int]],
) -> OracleCluster:
    ids = tuple(sorted(map(int, component_ids)))
    if not ids:
        raise E22ContractError("cannot create an empty oracle cluster")
    root, _ = dsu.find(ids[0])
    raw_translations: dict[int, tuple[int, int]] = {}
    raw_entries: list[tuple[int, int, int]] = []
    for component_id in ids:
        observed_root, translation = dsu.find(component_id)
        if observed_root != root:
            raise E22ContractError("oracle cluster is disconnected")
        raw_translations[component_id] = translation
        for tile, row, col in _component_entries(dsu.components[component_id]):
            raw_entries.append((tile, row + translation[0], col + translation[1]))
    if len({tile for tile, _row, _col in raw_entries}) != len(raw_entries):
        raise E22ContractError("oracle cluster duplicates a tile")
    if len({(row, col) for _tile, row, col in raw_entries}) != len(raw_entries):
        raise E22ContractError("oracle cluster contains a collision")
    min_row = min(row for _tile, row, _col in raw_entries)
    min_col = min(col for _tile, _row, col in raw_entries)
    translations = tuple(
        (component_id, row - min_row, col - min_col)
        for component_id, (row, col) in sorted(raw_translations.items())
    )
    entries = tuple(
        sorted(
            (tile, row - min_row, col - min_col)
            for tile, row, col in raw_entries
        )
    )
    rows = [row for _tile, row, _col in entries]
    cols = [col for _tile, _row, col in entries]
    bbox = (min(rows), max(rows), min(cols), max(cols))
    height = bbox[1] - bbox[0] + 1
    width = bbox[3] - bbox[2] + 1
    if bbox[0] != 0 or bbox[2] != 0 or not (1 <= height <= 24 and 1 <= width <= 24):
        raise E22ContractError("oracle cluster normalization/span drifted")
    accepted = tuple(sorted(tuple(map(int, relation)) for relation in relations))
    if len(set(accepted)) != len(accepted):
        raise E22ContractError("oracle cluster duplicates a relation")
    if any(u not in ids or v not in ids for u, v, _dr, _dc in accepted):
        raise E22ContractError("oracle relation leaves its connected cluster")
    translation_map = {cid: (row, col) for cid, row, col in translations}
    for u, v, dr, dc in accepted:
        observed = (
            translation_map[v][0] - translation_map[u][0],
            translation_map[v][1] - translation_map[u][1],
        )
        if observed != (dr, dc):
            raise E22ContractError("accepted relation contradicts cluster translations")
    cycle_rank = len(accepted) - len(ids) + 1
    if cycle_rank < 0:
        raise E22ContractError("oracle relation graph is not connected")
    cycle_ratio = float(cycle_rank / max(1, len(ids) - 1))
    tile_count = len(entries)
    return OracleCluster(
        component_ids=ids,
        translations=translations,
        relative_entries=entries,
        accepted_relations=accepted,
        exact_connected_tiles=tile_count,
        exact_connected_coverage=float(tile_count / e12.NFRAG),
        accepted_relation_count=len(accepted),
        cycle_rank=cycle_rank,
        cycle_rank_ratio=cycle_ratio,
        minimum_tile=min(tile for tile, _row, _col in entries),
        bbox=bbox,
        bbox_height=height,
        bbox_width=width,
        legal_origin_bounds=(0, 23 - bbox[1], 0, 23 - bbox[3]),
        legal_origin_count=(25 - height) * (25 - width),
    )


def select_oracle_cluster(clusters: Sequence[OracleCluster]) -> OracleCluster:
    if not clusters:
        raise E22ContractError("no exact pure component cluster exists")
    return min(
        clusters,
        key=lambda cluster: (
            -cluster.exact_connected_tiles,
            -cluster.accepted_relation_count,
            -cluster.cycle_rank,
            cluster.minimum_tile,
            cluster.translations,
        ),
    )


def build_oracle_ceiling(
    result: Any, permutation: object
) -> tuple[
    dict[int, tuple[int, int] | None],
    tuple[Any, ...],
    tuple[OracleCluster, ...],
    OracleCluster,
]:
    _validate_components(result)
    shifts = component_truth_shifts(result, permutation)
    pure_ids = tuple(sorted(cid for cid, shift in shifts.items() if shift is not None))
    if not pure_ids:
        raise E22ContractError("candidate partition contains no exact pure component")
    true_hypotheses = true_pose_hypotheses(result, shifts)
    dsu = _PotentialDSU(result.components, pure_ids)
    ordered_true = tuple(
        sorted(
            true_hypotheses,
            key=lambda value: (
                *_relation_tuple(value, label="true hypothesis"),
                int(value.hypothesis_id),
            ),
        )
    )
    for hypothesis in ordered_true:
        dsu.union(*_relation_tuple(hypothesis, label="true hypothesis"))

    relations_by_root: dict[int, list[tuple[int, int, int, int]]] = {}
    for hypothesis in ordered_true:
        relation = _relation_tuple(hypothesis, label="true hypothesis")
        root_u, _ = dsu.find(relation[0])
        root_v, _ = dsu.find(relation[1])
        if root_u != root_v:
            raise E22ContractError("true relation endpoints remained disconnected")
        relations_by_root.setdefault(root_u, []).append(relation)
    clusters: list[OracleCluster] = []
    roots = sorted({dsu.find(component_id)[0] for component_id in pure_ids})
    for root in roots:
        ids = tuple(sorted(dsu.members[root]))
        clusters.append(_make_cluster(dsu, ids, relations_by_root.get(root, ())))
    clusters.sort(key=lambda cluster: (cluster.minimum_tile, cluster.translations))
    selected = select_oracle_cluster(clusters)
    return shifts, ordered_true, tuple(clusters), selected


def _cluster_payload(cluster: OracleCluster) -> dict[str, Any]:
    return _jsonable(asdict(cluster))


def _validate_observation(
    observed: Any,
    expected: tuple[int, int, int, int, float] | None,
    *,
    label: str,
) -> None:
    if expected is None:
        if observed is not None:
            raise E22ContractError(f"{label} must explicitly preserve missing membership")
        return
    if not isinstance(observed, rcce.LogitObservation):
        raise E22ContractError(f"{label} has the wrong observation type")
    source, target, direction, slot, logit = expected
    if (
        _integer(observed.source, label=f"{label} source") != source
        or _integer(observed.target, label=f"{label} target") != target
        or _integer(observed.direction, label=f"{label} direction") != direction
        or _integer(observed.slot, label=f"{label} slot") != slot
        or _finite(
            observed.logit,
            label=f"{label} raw logit",
            minimum=-float("inf"),
            maximum=float("inf"),
        )
        != logit
    ):
        raise E22ContractError(f"{label} raw-logit metadata drifted")


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
    u, v, dr, dc = relation
    shifts = {u: (0, 0), v: (dr, dc)}
    for claim_id in claim_ids:
        claim = claims[claim_id]
        if {int(claim.first_component), int(claim.second_component)} != {u, v}:
            raise E22ContractError("relation claim leaves its component pair")
        first_shift = shifts[int(owner[int(claim.first)])]
        second_shift = shifts[int(owner[int(claim.second)])]
        first_position = (
            int(local_rows[int(claim.first)]) + first_shift[0],
            int(local_cols[int(claim.first)]) + first_shift[1],
        )
        second_position = (
            int(local_rows[int(claim.second)]) + second_shift[0],
            int(local_cols[int(claim.second)]) + second_shift[1],
        )
        if (
            second_position[0] - first_position[0],
            second_position[1] - first_position[1],
        ) != (int(claim.dy), int(claim.dx)):
            return "adjacency"

    occupied: set[tuple[int, int]] = set()
    for component_id in (u, v):
        shift = shifts[component_id]
        for _tile, row, col in _component_entries(components[component_id]):
            position = (row + shift[0], col + shift[1])
            if position in occupied:
                return "collision"
            occupied.add(position)
    rows = [row for row, _col in occupied]
    cols = [col for _row, col in occupied]
    if max(rows) - min(rows) + 1 > 24 or max(cols) - min(cols) + 1 > 24:
        return "span"
    return None


def _independent_raw_cc96_partition(
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    valid: np.ndarray,
) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray, np.ndarray, frozenset[int]]:
    """Rebuild frozen dense+CC96 from private copies, independently of E22 core."""

    private_ids = np.array(candidate_ids, dtype=np.int64, order="C", copy=True)
    private_logits = np.array(raw_logits, dtype=np.float32, order="C", copy=True)
    private_ids[~valid] = 0
    try:
        right, down = e12.dense_from_graph(private_ids, private_logits)
        return e21_pose.build_components(right, down)
    except Exception as exc:
        raise E22ContractError(f"independent raw CC96 replay failed: {exc}") from exc


def _find_relation_candidate(
    candidates: Sequence[Any], relation: tuple[int, int, int, int]
) -> int:
    """Binary-search prevalidated canonical relations without a key-cache copy."""

    lower = 0
    upper = len(candidates)
    while lower < upper:
        middle = (lower + upper) // 2
        candidate = candidates[middle]
        observed = (candidate.u, candidate.v, candidate.dr, candidate.dc)
        if observed < relation:
            lower = middle + 1
        else:
            upper = middle
    if lower >= len(candidates):
        return -1
    candidate = candidates[lower]
    return lower if (candidate.u, candidate.v, candidate.dr, candidate.dc) == relation else -1


def validate_candidate_pool(
    result: Any,
    *,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
) -> None:
    """Independently replay every admission/grouping/filter algebra invariant."""

    if not isinstance(result, rcce.CandidatePoolResult):
        raise E22ContractError("candidate core returned the wrong result type")
    candidate_ids, raw_logits, valid = _validate_raw_arrays(candidate_ids, raw_logits)
    owner, local_rows, local_cols = _validate_components(result)
    (
        expected_components,
        expected_owner,
        expected_local_rows,
        expected_local_cols,
        expected_nontrivial,
    ) = _independent_raw_cc96_partition(candidate_ids, raw_logits, valid)
    if (
        result.components != expected_components
        or not np.array_equal(owner, expected_owner)
        or not np.array_equal(local_rows, expected_local_rows)
        or not np.array_equal(local_cols, expected_local_cols)
        or result.nontrivial_component_ids != expected_nontrivial
    ):
        raise E22ContractError(
            "returned CC96 partition differs from independent frozen dense replay"
        )

    expected_pairs = _expected_affinity_pairs(candidate_ids, valid)
    if (
        not isinstance(result.affinity_pairs, tuple)
        or len(result.affinity_pairs) != len(expected_pairs)
    ):
        raise E22ContractError("affinity-pair inventory is malformed")
    for pair_id, (pair, expected) in enumerate(
        zip(result.affinity_pairs, expected_pairs)
    ):
        if not isinstance(pair, rcce.AffinityPair):
            raise E22ContractError("affinity pair has the wrong type")
        observed = (
            _integer(pair.a, label="affinity pair a"),
            _integer(pair.b, label="affinity pair b"),
            None
            if pair.a_to_b_slot is None
            else _integer(pair.a_to_b_slot, label="a-to-b slot"),
            None
            if pair.b_to_a_slot is None
            else _integer(pair.b_to_a_slot, label="b-to-a slot"),
        )
        if (
            _integer(pair.pair_id, label="affinity pair ID") != pair_id
            or observed != expected
        ):
            raise E22ContractError("canonical unordered affinity-pair OR drifted")

    same_component_pairs = 0
    cross_component_pairs = 0
    for a, b, _a_slot, _b_slot in expected_pairs:
        if int(owner[a]) == int(owner[b]):
            same_component_pairs += 1
        else:
            cross_component_pairs += 1
    expected_claim_count = 4 * cross_component_pairs
    if not isinstance(result.claims, tuple) or len(result.claims) != expected_claim_count:
        raise E22ContractError("cross-component RCCE-4 claim inventory drifted")

    relations = result.relation_candidates
    if not isinstance(relations, tuple):
        raise E22ContractError("relation-candidate inventory drifted")
    previous_relation: tuple[int, int, int, int] | None = None
    component_pairs = 0
    component_pairs_with_alternative_offsets = 0
    previous_component_pair: tuple[int, int] | None = None
    offsets_for_pair = 0
    for relation_id, candidate in enumerate(relations):
        if not isinstance(candidate, rcce.RelationCandidate):
            raise E22ContractError("relation candidate has the wrong type")
        relation = _relation_tuple(candidate, label="relation candidate")
        u, v, _dr, _dc = relation
        claim_ids = candidate.claim_ids
        if (
            _integer(candidate.relation_id, label="relation candidate ID")
            != relation_id
            or not 0 <= u < v < len(result.components)
            or previous_relation is not None
            and relation <= previous_relation
            or not isinstance(claim_ids, tuple)
            or not claim_ids
        ):
            raise E22ContractError("canonical signed relation grouping drifted")
        checked_claim_ids = tuple(
            _integer(value, label="relation claim ID") for value in claim_ids
        )
        if (
            checked_claim_ids != tuple(sorted(checked_claim_ids))
            or len(set(checked_claim_ids)) != len(checked_claim_ids)
            or checked_claim_ids[0] < 0
            or checked_claim_ids[-1] >= len(result.claims)
        ):
            raise E22ContractError("relation claim IDs are not canonical")
        component_pair = (u, v)
        if component_pair != previous_component_pair:
            if offsets_for_pair > 1:
                component_pairs_with_alternative_offsets += 1
            component_pairs += 1
            offsets_for_pair = 1
            previous_component_pair = component_pair
        else:
            offsets_for_pair += 1
        previous_relation = relation
    if offsets_for_pair > 1:
        component_pairs_with_alternative_offsets += 1

    relation_positions = np.zeros(len(relations), dtype=np.int64)
    claim_cursor = 0
    claim_observations = 0
    for pair_id, (a, b, a_slot, b_slot) in enumerate(expected_pairs):
        if int(owner[a]) == int(owner[b]):
            continue
        specs = (
            (a, b, 0, 1, a_slot, rcce.RIGHT, b_slot, rcce.LEFT),
            (b, a, 0, 1, b_slot, rcce.RIGHT, a_slot, rcce.LEFT),
            (a, b, 1, 0, a_slot, rcce.DOWN, b_slot, rcce.UP),
            (b, a, 1, 0, b_slot, rcce.DOWN, a_slot, rcce.UP),
        )
        for first, second, dy, dx, f_slot, f_dir, r_slot, r_dir in specs:
            if claim_cursor >= len(result.claims):
                raise E22ContractError("cross-component RCCE-4 claims were truncated")
            claim = result.claims[claim_cursor]
            if not isinstance(claim, rcce.RCCE4Claim):
                raise E22ContractError("RCCE-4 claim has the wrong type")
            first_c = int(owner[first])
            second_c = int(owner[second])
            forward = (
                None
                if f_slot is None
                else (
                    first,
                    second,
                    f_dir,
                    f_slot,
                    float(raw_logits[f_dir, first, f_slot]),
                )
            )
            reverse = (
                None
                if r_slot is None
                else (
                    second,
                    first,
                    r_dir,
                    r_slot,
                    float(raw_logits[r_dir, second, r_slot]),
                )
            )
            physical_seam = (first, second, dy, dx)
            if (
                _integer(claim.claim_id, label="claim ID") != claim_cursor
                or _integer(claim.pair_id, label="claim pair ID") != pair_id
                or _integer(claim.first, label="claim first") != first
                or _integer(claim.second, label="claim second") != second
                or _integer(claim.dy, label="claim dy") != dy
                or _integer(claim.dx, label="claim dx") != dx
                or _integer(claim.first_component, label="claim first component")
                != first_c
                or _integer(claim.second_component, label="claim second component")
                != second_c
                or first_c == second_c
                or claim.physical_seam != physical_seam
            ):
                raise E22ContractError("literal ordered RCCE-4 claim algebra drifted")
            _validate_observation(
                claim.forward_observation,
                forward,
                label="claim forward observation",
            )
            _validate_observation(
                claim.reverse_observation,
                reverse,
                label="claim reverse observation",
            )
            claim_observations += int(forward is not None) + int(reverse is not None)
            relation = seam_relation(
                physical_seam,
                owner=owner,
                local_rows=local_rows,
                local_cols=local_cols,
            )
            relation_id = _find_relation_candidate(relations, relation)
            if relation_id < 0:
                raise E22ContractError("claim relation is missing from grouped output")
            position = int(relation_positions[relation_id])
            claim_ids = relations[relation_id].claim_ids
            if position >= len(claim_ids) or int(claim_ids[position]) != claim_cursor:
                raise E22ContractError("canonical signed relation grouping drifted")
            relation_positions[relation_id] += 1
            claim_cursor += 1
    if claim_cursor != len(result.claims):
        raise E22ContractError("cross-component RCCE-4 claim inventory drifted")
    for relation_id, candidate in enumerate(relations):
        if int(relation_positions[relation_id]) != len(candidate.claim_ids):
            raise E22ContractError("relation grouping omitted or duplicated a claim")

    if not isinstance(result.hypotheses, tuple) or not isinstance(
        result.geometry_rejections, tuple
    ):
        raise E22ContractError("geometry-filter output is malformed")
    hypothesis_cursor = 0
    rejection_cursor = 0
    rejection_counts = {"adjacency": 0, "collision": 0, "span": 0}
    for relation_id, candidate in enumerate(relations):
        relation = _relation_tuple(candidate, label="relation candidate")
        claim_ids = candidate.claim_ids
        reason = _independent_geometry_reason(
            relation,
            claim_ids,
            claims=result.claims,
            components=result.components,
            owner=owner,
            local_rows=local_rows,
            local_cols=local_cols,
        )
        if reason is None:
            if hypothesis_cursor >= len(result.hypotheses):
                raise E22ContractError("geometry-valid hypotheses were truncated")
            hypothesis = result.hypotheses[hypothesis_cursor]
            if (
                not isinstance(hypothesis, rcce.PoseHypothesis)
                or _integer(hypothesis.hypothesis_id, label="hypothesis ID")
                != hypothesis_cursor
                or _integer(hypothesis.relation_id, label="hypothesis relation ID")
                != relation_id
                or _relation_tuple(hypothesis, label="hypothesis") != relation
                or hypothesis.claim_ids != claim_ids
            ):
                raise E22ContractError("post-geometry hypothesis algebra drifted")
            hypothesis_cursor += 1
        else:
            if rejection_cursor >= len(result.geometry_rejections):
                raise E22ContractError("geometry rejections were truncated")
            rejection = result.geometry_rejections[rejection_cursor]
            if (
                not isinstance(rejection, rcce.GeometryRejection)
                or _integer(rejection.relation_id, label="rejected relation ID")
                != relation_id
                or rejection.reason != reason
            ):
                raise E22ContractError("geometry rejection reason/order drifted")
            rejection_counts[reason] += 1
            rejection_cursor += 1
    if hypothesis_cursor != len(result.hypotheses):
        raise E22ContractError("geometry-valid hypothesis inventory drifted")
    if rejection_cursor != len(result.geometry_rejections):
        raise E22ContractError("geometry rejection inventory drifted")

    directed_memberships = int(valid.sum())
    input_observations = 4 * directed_memberships
    one_way = sum((a_slot is None) != (b_slot is None) for _a, _b, a_slot, b_slot in expected_pairs)
    reciprocal = len(expected_pairs) - one_way
    nontrivial_tiles = sum(
        len(_component_entries(result.components[cid]))
        for cid in result.nontrivial_component_ids
    )
    expected_diagnostics = {
        "component_count": len(result.components),
        "nontrivial_components": len(result.nontrivial_component_ids),
        "singleton_components": len(result.components)
        - len(result.nontrivial_component_ids),
        "total_tiles": e12.NFRAG,
        "nontrivial_tiles": nontrivial_tiles,
        "singleton_tiles": e12.NFRAG - nontrivial_tiles,
        "emitter_tiles": e12.NFRAG,
        "directed_valid_memberships": directed_memberships,
        "input_logit_observations": input_observations,
        "unordered_affinity_pairs": len(expected_pairs),
        "one_way_affinity_pairs": one_way,
        "reciprocal_affinity_pairs": reciprocal,
        "pre_component_filter_claims": 4 * len(expected_pairs),
        "same_component_pairs": same_component_pairs,
        "same_component_claims_removed": 4 * same_component_pairs,
        "cross_component_pairs": cross_component_pairs,
        "claims": claim_cursor,
        "claim_logit_observations": claim_observations,
        "relation_candidates": len(relations),
        "geometry_valid_hypotheses": hypothesis_cursor,
        "geometry_rejected_relations": rejection_cursor,
        "geometry_rejected_adjacency": rejection_counts["adjacency"],
        "geometry_rejected_collision": rejection_counts["collision"],
        "geometry_rejected_span": rejection_counts["span"],
        "component_pairs": component_pairs,
        "component_pairs_with_alternative_offsets": component_pairs_with_alternative_offsets,
    }
    diagnostics = _jsonable(result.diagnostics)
    _check_forbidden_payload_keys(diagnostics)
    if diagnostics != expected_diagnostics:
        raise E22ContractError("candidate-pool diagnostics drifted")
    if (
        not e12.NFRAG <= directed_memberships <= MAX_DIRECTED_MEMBERSHIPS
        or len(expected_pairs) > MAX_UNORDERED_PAIRS
        or input_observations > MAX_DIRECTIONAL_OBSERVATIONS
        or 4 * len(expected_pairs) > MAX_RCCE4_PRECLAIMS
        or claim_cursor > MAX_RCCE4_PRECLAIMS
        or len(relations) > claim_cursor
        or hypothesis_cursor > len(relations)
        or hypothesis_cursor > MAX_GEOMETRY_VALID_HYPOTHESES
    ):
        raise E22ContractError("one or more theoretical fail-not-truncate bounds failed")


def _core_payload(
    result: Any,
    *,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
) -> dict[str, Any]:
    validate_candidate_pool(
        result, candidate_ids=candidate_ids, raw_logits=raw_logits
    )
    diagnostics = _jsonable(result.diagnostics)
    _check_forbidden_payload_keys(diagnostics)
    return {
        "component_count": len(result.components),
        "nontrivial_component_ids": _jsonable(result.nontrivial_component_ids),
        "affinity_pair_count": len(result.affinity_pairs),
        "claim_count": len(result.claims),
        "relation_candidate_count": len(result.relation_candidates),
        "hypothesis_count": len(result.hypotheses),
        "geometry_rejection_count": len(result.geometry_rejections),
        "components_sha256": _stream_digest(result.components),
        "owner_sha256": e12.array_sha256(result.owner),
        "local_rows_sha256": e12.array_sha256(result.local_rows),
        "local_cols_sha256": e12.array_sha256(result.local_cols),
        "affinity_pairs_sha256": _stream_digest(result.affinity_pairs),
        "claims_sha256": _stream_digest(result.claims),
        "relation_candidates_sha256": _stream_digest(result.relation_candidates),
        "hypotheses_sha256": _stream_digest(result.hypotheses),
        "geometry_rejections_sha256": _stream_digest(result.geometry_rejections),
        "diagnostics": diagnostics,
    }


def _contact_measurement(
    result: Any,
    *,
    permutation: object,
    shifts: Mapping[int, tuple[int, int] | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    owner, local_rows, local_cols = _validate_components(result)
    all_seams = ground_truth_seams(permutation)
    # These denominators are formed before consulting candidate pairs or
    # hypotheses.  Only the already-returned CC96 partition/purity is used.
    unconditional = tuple(
        seam for seam in all_seams if int(owner[seam[0]]) != int(owner[seam[1]])
    )
    eligible = tuple(
        seam
        for seam in unconditional
        if shifts[int(owner[seam[0]])] is not None
        and shifts[int(owner[seam[1]])] is not None
    )
    pair_inventory = {(int(pair.a), int(pair.b)) for pair in result.affinity_pairs}

    def pair_key(seam: tuple[int, int, int, int]) -> tuple[int, int]:
        first, second = seam[:2]
        return (first, second) if first < second else (second, first)

    eligible_hits = tuple(seam for seam in eligible if pair_key(seam) in pair_inventory)
    unconditional_hits = tuple(
        seam for seam in unconditional if pair_key(seam) in pair_inventory
    )
    hypothesis_by_relation = {
        _relation_tuple(value, label="postfilter hypothesis"): value
        for value in result.hypotheses
    }
    if len(hypothesis_by_relation) != len(result.hypotheses):
        raise E22ContractError("postfilter hypotheses duplicate a signed relation")
    survivors: list[tuple[int, int, int, int]] = []
    for seam in eligible_hits:
        relation = seam_relation(
            seam, owner=owner, local_rows=local_rows, local_cols=local_cols
        )
        hypothesis = hypothesis_by_relation.get(relation)
        if hypothesis is None:
            continue
        supported_seams = {
            tuple(map(int, result.claims[int(claim_id)].physical_seam))
            for claim_id in hypothesis.claim_ids
        }
        # A matching component relation alone is insufficient: the exact GT
        # physical seam must be one of that post-geometry hypothesis's claims.
        if seam in supported_seams:
            survivors.append(seam)
    eligible_count = len(eligible)
    unconditional_count = len(unconditional)
    eligible_hit_count = len(eligible_hits)
    eligible_recall = float(eligible_hit_count / eligible_count) if eligible_count else 0.0
    unconditional_recall = (
        float(len(unconditional_hits) / unconditional_count)
        if unconditional_count
        else 0.0
    )
    survival = (
        float(len(survivors) / eligible_hit_count) if eligible_hit_count else 0.0
    )
    inventory = {
        "ground_truth_upright_rd_seam_count": len(all_seams),
        "ground_truth_upright_rd_seams_sha256": e12.canonical_digest(
            _jsonable(all_seams)
        ),
        "unconditional_cross_component_seams_sha256": e12.canonical_digest(
            _jsonable(unconditional)
        ),
        "eligible_whole_pure_cross_component_seams_sha256": e12.canonical_digest(
            _jsonable(eligible)
        ),
        "eligible_pair_or_hits_sha256": e12.canonical_digest(
            _jsonable(eligible_hits)
        ),
        "postfilter_exact_physical_seam_survivors_sha256": e12.canonical_digest(
            _jsonable(tuple(survivors))
        ),
    }
    metrics = {
        "eligible_contacts": eligible_count,
        "eligible_pair_hits": eligible_hit_count,
        "eligible_contact_recall": eligible_recall,
        "unconditional_cross_component_contacts": unconditional_count,
        "unconditional_pair_hits": len(unconditional_hits),
        "unconditional_cross_component_recall": unconditional_recall,
        "postfilter_eligible_hits": eligible_hit_count,
        "postfilter_exact_physical_seam_survivors": len(survivors),
        "postfilter_eligible_true_survival": survival,
    }
    return inventory, metrics


def evaluate_scene(
    scene: e12.RawScene,
    result: Any,
    *,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
) -> dict[str, Any]:
    core = _core_payload(
        result, candidate_ids=candidate_ids, raw_logits=raw_logits
    )
    # The first label access is intentionally after the complete core payload
    # has returned and passed independent label-free validation.
    shifts, true_hypotheses, clusters, selected = build_oracle_ceiling(
        result, scene.permutation
    )
    pure_ids = tuple(sorted(cid for cid, shift in shifts.items() if shift is not None))
    pure_tiles = sum(len(_component_entries(result.components[cid])) for cid in pure_ids)
    contact_inventory, contact_metrics = _contact_measurement(
        result, permutation=scene.permutation, shifts=shifts
    )
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
            _cluster_payload(value) for value in clusters
        ),
        "contact_inventory": contact_inventory,
        "selected": _cluster_payload(selected),
    }
    diagnostics = result.diagnostics
    metrics = {
        "tile_orientation_degrees": 0,
        "emitter_tiles": int(diagnostics.emitter_tiles),
        "directed_valid_memberships": int(diagnostics.directed_valid_memberships),
        "unordered_affinity_pairs": int(diagnostics.unordered_affinity_pairs),
        "finite_directional_logit_observations": int(
            diagnostics.input_logit_observations
        ),
        "rcce4_preclaims": int(diagnostics.pre_component_filter_claims),
        "cross_component_claims": int(diagnostics.claims),
        "relation_candidates": int(diagnostics.relation_candidates),
        "geometry_valid_hypotheses": int(diagnostics.geometry_valid_hypotheses),
        "bounds_passed": True,
        "component_count": len(result.components),
        "nontrivial_component_count": len(result.nontrivial_component_ids),
        "pure_component_count": len(pure_ids),
        "pure_component_tiles": int(pure_tiles),
        **contact_metrics,
        "true_hypotheses": len(true_hypotheses),
        "selected_components": len(selected.component_ids),
        "selected_accepted_relations": selected.accepted_relation_count,
        "selected_cycle_rank": selected.cycle_rank,
        "selected_cycle_rank_ratio": selected.cycle_rank_ratio,
        "selected_exact_connected_tiles": selected.exact_connected_tiles,
        "selected_exact_connected_coverage": selected.exact_connected_coverage,
        "legal_origin_count": selected.legal_origin_count,
    }
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "raw_cache_sha256": str(scene.cache_sha256),
        "candidate_ids_sha256": e12.array_sha256(candidate_ids),
        "raw_logits_sha256": e12.array_sha256(raw_logits),
        "orientation": "upright_0_degrees_no_rotation_no_reflection",
        "arm": "E22_upright_RCCE4_full_union_all_emitter_candidate_ceiling",
        "core": core,
        "core_sha256": e12.canonical_digest(core),
        "oracle": oracle,
        "oracle_sha256": e12.canonical_digest(oracle),
        "metrics": metrics,
    }


def _scene_arrays(
    scene: e12.RawScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _validate_raw_arrays(scene.candidate_ids, scene.base_scores)


def _validate_success_row(
    row: Mapping[str, Any],
    *,
    scene: e12.RawScene,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    expected_result: Any | None = None,
) -> None:
    expected_keys = {
        "image",
        "validation_name",
        "raw_cache_sha256",
        "candidate_ids_sha256",
        "raw_logits_sha256",
        "orientation",
        "arm",
        "core",
        "core_sha256",
        "oracle",
        "oracle_sha256",
        "metrics",
    }
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise E22ContractError("E22 row fields drifted")
    for label, observed, expected in (
        ("raw cache", row.get("raw_cache_sha256"), scene.cache_sha256),
        (
            "candidate IDs",
            row.get("candidate_ids_sha256"),
            e12.array_sha256(candidate_ids),
        ),
        (
            "raw logits",
            row.get("raw_logits_sha256"),
            e12.array_sha256(raw_logits),
        ),
    ):
        if observed != expected or not _is_sha256(observed):
            raise E22ContractError(f"E22 {label} lineage drifted")
    if (
        _integer(row.get("image"), label="row image") != int(scene.image_id)
        or row.get("validation_name") != str(scene.validation_name)
        or row.get("orientation")
        != "upright_0_degrees_no_rotation_no_reflection"
        or row.get("arm")
        != "E22_upright_RCCE4_full_union_all_emitter_candidate_ceiling"
    ):
        raise E22ContractError(f"E22 row provenance drifted for image {scene.image_id}")
    if expected_result is not None:
        result = expected_result
    else:
        candidate_digest_before = e12.array_sha256(candidate_ids)
        logit_digest_before = e12.array_sha256(raw_logits)
        result = rcce.run_rcce4_candidate_oracle(candidate_ids, raw_logits)
        if (
            e12.array_sha256(candidate_ids) != candidate_digest_before
            or e12.array_sha256(raw_logits) != logit_digest_before
        ):
            raise E22ContractError("candidate core mutated its frozen raw inputs")
    expected = evaluate_scene(
        scene,
        result,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
    )
    if row != expected:
        raise E22ContractError(f"E22 row replay drifted for image {scene.image_id}")
    if row.get("core_sha256") != e12.canonical_digest(row["core"]):
        raise E22ContractError("E22 core hash drifted")
    if row.get("oracle_sha256") != e12.canonical_digest(row["oracle"]):
        raise E22ContractError("E22 oracle hash drifted")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E22ContractError("E22 summary requires exactly eight rows")
    images = [_integer(row.get("image"), label="summary row image") for row in rows]
    if tuple(sorted(images)) != e12.CALIBRATION_IDS or len(set(images)) != len(images):
        raise E22ContractError("E22 summary image IDs drifted")
    metrics: list[Mapping[str, Any]] = []
    for row in rows:
        value = row.get("metrics")
        if not isinstance(value, Mapping):
            raise E22ContractError("E22 row metrics are malformed")
        metrics.append(value)

    orientation = [
        _integer(value.get("tile_orientation_degrees"), label="tile orientation")
        for value in metrics
    ]
    emitters = [
        _integer(value.get("emitter_tiles"), label="emitter tiles")
        for value in metrics
    ]
    memberships = [
        _integer(value.get("directed_valid_memberships"), label="memberships")
        for value in metrics
    ]
    pairs = [
        _integer(value.get("unordered_affinity_pairs"), label="affinity pairs")
        for value in metrics
    ]
    observations = [
        _integer(
            value.get("finite_directional_logit_observations"),
            label="directional logit observations",
        )
        for value in metrics
    ]
    preclaims = [
        _integer(value.get("rcce4_preclaims"), label="RCCE-4 preclaims")
        for value in metrics
    ]
    retained_claims = [
        _integer(
            value.get("cross_component_claims"),
            label="retained cross-component RCCE-4 claims",
        )
        for value in metrics
    ]
    relation_candidates = [
        _integer(value.get("relation_candidates"), label="relation candidates")
        for value in metrics
    ]
    hypotheses = [
        _integer(
            value.get("geometry_valid_hypotheses"),
            label="geometry-valid hypotheses",
        )
        for value in metrics
    ]
    explicit_bounds = [value.get("bounds_passed") is True for value in metrics]
    algebraic_bounds = [
        e12.NFRAG <= memberships[index] <= MAX_DIRECTED_MEMBERSHIPS
        and pairs[index] <= MAX_UNORDERED_PAIRS
        and observations[index] == 4 * memberships[index]
        and observations[index] <= MAX_DIRECTIONAL_OBSERVATIONS
        and preclaims[index] == 4 * pairs[index]
        and preclaims[index] <= MAX_RCCE4_PRECLAIMS
        and 0 <= retained_claims[index] <= preclaims[index]
        and retained_claims[index] % 4 == 0
        and 0 <= relation_candidates[index] <= retained_claims[index]
        and 0 <= hypotheses[index] <= relation_candidates[index]
        and hypotheses[index] <= MAX_GEOMETRY_VALID_HYPOTHESES
        for index in range(len(metrics))
    ]
    true_counts = [
        _integer(value.get("true_hypotheses"), label="true hypothesis count")
        for value in metrics
    ]
    legal_counts = [
        _integer(value.get("legal_origin_count"), label="legal origin count")
        for value in metrics
    ]
    eligible_counts = [
        _integer(value.get("eligible_contacts"), label="eligible contacts")
        for value in metrics
    ]
    eligible_hits = [
        _integer(value.get("eligible_pair_hits"), label="eligible pair hits")
        for value in metrics
    ]
    survivors = [
        _integer(
            value.get("postfilter_exact_physical_seam_survivors"),
            label="postfilter exact seam survivors",
        )
        for value in metrics
    ]
    eligible_recall = [
        _finite(
            value.get("eligible_contact_recall"),
            label="eligible contact recall",
            minimum=0.0,
            maximum=1.0,
        )
        for value in metrics
    ]
    unconditional_recall = [
        _finite(
            value.get("unconditional_cross_component_recall"),
            label="unconditional cross-component recall",
            minimum=0.0,
            maximum=1.0,
        )
        for value in metrics
    ]
    survival = [
        _finite(
            value.get("postfilter_eligible_true_survival"),
            label="postfilter eligible-true survival",
            minimum=0.0,
            maximum=1.0,
        )
        for value in metrics
    ]
    coverage = [
        _finite(
            value.get("selected_exact_connected_coverage"),
            label="exact connected coverage",
            minimum=0.0,
            maximum=1.0,
        )
        for value in metrics
    ]
    tiles = [
        _integer(
            value.get("selected_exact_connected_tiles"),
            label="exact connected tiles",
        )
        for value in metrics
    ]
    cycle_ratios = [
        _finite(
            value.get("selected_cycle_rank_ratio"),
            label="selected cycle-rank ratio",
            minimum=0.0,
            maximum=float("inf"),
        )
        for value in metrics
    ]
    if any(
        count < 0 or hit < 0 or hit > count or kept < 0 or kept > hit
        for count, hit, kept in zip(eligible_counts, eligible_hits, survivors)
    ):
        raise E22ContractError("eligible contact/survival counts are inconsistent")
    return {
        "images": len(rows),
        "completed_scenes": len(rows),
        "upright_orientation_scenes": int(sum(value == 0 for value in orientation)),
        "emitters_exact_scenes": int(sum(value == e12.NFRAG for value in emitters)),
        "all_bounds_scenes": int(
            sum(explicit and algebraic for explicit, algebraic in zip(explicit_bounds, algebraic_bounds))
        ),
        "true_relation_scenes": int(sum(value >= 1 for value in true_counts)),
        "legal_origin_scenes": int(sum(value >= 1 for value in legal_counts)),
        "positive_eligible_denominator_scenes": int(
            sum(value >= 1 for value in eligible_counts)
        ),
        "exact_postfilter_survival_scenes": int(
            sum(value == 1.0 for value in survival)
        ),
        "mean_eligible_contact_recall": float(np.mean(eligible_recall)),
        "worst_eligible_contact_recall": float(min(eligible_recall)),
        "mean_unconditional_cross_component_recall": float(
            np.mean(unconditional_recall)
        ),
        "worst_unconditional_cross_component_recall": float(
            min(unconditional_recall)
        ),
        "mean_exact_connected_tiles": float(np.mean(tiles)),
        "mean_exact_connected_coverage": float(np.mean(coverage)),
        "worst_exact_connected_coverage": float(min(coverage)),
        "mean_selected_cycle_rank_ratio": float(np.mean(cycle_ratios)),
        "worst_selected_cycle_rank_ratio": float(min(cycle_ratios)),
        "max_directed_valid_memberships": max(memberships),
        "max_unordered_affinity_pairs": max(pairs),
        "max_finite_directional_logit_observations": max(observations),
        "max_rcce4_preclaims": max(preclaims),
        "max_cross_component_claims": max(retained_claims),
        "max_relation_candidates": max(relation_candidates),
        "max_geometry_valid_hypotheses": max(hypotheses),
        "total_geometry_valid_hypotheses": int(sum(hypotheses)),
        "total_true_hypotheses": int(sum(true_counts)),
        "total_eligible_contacts": int(sum(eligible_counts)),
        "total_eligible_pair_hits": int(sum(eligible_hits)),
        "total_postfilter_exact_physical_seam_survivors": int(sum(survivors)),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "completed_scenes": _integer(
            summary.get("completed_scenes"), label="completed scenes"
        ),
        "upright_orientation_scenes": _integer(
            summary.get("upright_orientation_scenes"),
            label="upright orientation scenes",
        ),
        "emitters_exact_scenes": _integer(
            summary.get("emitters_exact_scenes"), label="exact emitter scenes"
        ),
        "all_bounds_scenes": _integer(
            summary.get("all_bounds_scenes"), label="all-bounds scenes"
        ),
        "true_relation_scenes": _integer(
            summary.get("true_relation_scenes"), label="true relation scenes"
        ),
        "legal_origin_scenes": _integer(
            summary.get("legal_origin_scenes"), label="legal origin scenes"
        ),
        "positive_eligible_denominator_scenes": _integer(
            summary.get("positive_eligible_denominator_scenes"),
            label="positive eligible denominator scenes",
        ),
        "exact_postfilter_survival_scenes": _integer(
            summary.get("exact_postfilter_survival_scenes"),
            label="exact postfilter survival scenes",
        ),
        "mean_eligible_contact_recall": _finite(
            summary.get("mean_eligible_contact_recall"),
            label="mean eligible contact recall",
            minimum=0.0,
            maximum=1.0,
        ),
        "worst_eligible_contact_recall": _finite(
            summary.get("worst_eligible_contact_recall"),
            label="worst eligible contact recall",
            minimum=0.0,
            maximum=1.0,
        ),
        "mean_exact_connected_coverage": _finite(
            summary.get("mean_exact_connected_coverage"),
            label="mean exact connected coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        "worst_exact_connected_coverage": _finite(
            summary.get("worst_exact_connected_coverage"),
            label="worst exact connected coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        "mean_selected_cycle_rank_ratio": _finite(
            summary.get("mean_selected_cycle_rank_ratio"),
            label="mean selected cycle-rank ratio",
            minimum=0.0,
            maximum=float("inf"),
        ),
        "worst_selected_cycle_rank_ratio": _finite(
            summary.get("worst_selected_cycle_rank_ratio"),
            label="worst selected cycle-rank ratio",
            minimum=0.0,
            maximum=float("inf"),
        ),
    }
    target = int(DECISION_RULE["completed_scenes"])
    checks = {
        "completed_scenes": observed["completed_scenes"] == target,
        "upright_orientation_only": observed["upright_orientation_scenes"] == target,
        "emitters_each": observed["emitters_exact_scenes"] == target,
        "all_bounds_scenes": observed["all_bounds_scenes"]
        == int(DECISION_RULE["all_bounds_scenes"]),
        "true_relation_scenes": observed["true_relation_scenes"]
        == int(DECISION_RULE["true_relation_scenes"]),
        "legal_origin_scenes": observed["legal_origin_scenes"]
        == int(DECISION_RULE["legal_origin_scenes"]),
        "positive_eligible_denominator_scenes": observed[
            "positive_eligible_denominator_scenes"
        ]
        == int(DECISION_RULE["positive_eligible_denominator_scenes"]),
        "exact_postfilter_survival_scenes": observed[
            "exact_postfilter_survival_scenes"
        ]
        == int(DECISION_RULE["exact_postfilter_survival_scenes"]),
        "mean_eligible_contact_recall": observed["mean_eligible_contact_recall"]
        >= float(DECISION_RULE["mean_eligible_contact_recall_min"]),
        "worst_eligible_contact_recall": observed[
            "worst_eligible_contact_recall"
        ]
        >= float(DECISION_RULE["worst_eligible_contact_recall_min"]),
        "mean_exact_connected_coverage": observed[
            "mean_exact_connected_coverage"
        ]
        >= float(DECISION_RULE["mean_exact_connected_coverage_min"]),
        "worst_exact_connected_coverage": observed[
            "worst_exact_connected_coverage"
        ]
        >= float(DECISION_RULE["worst_exact_connected_coverage_min"]),
        "mean_selected_cycle_rank_ratio": observed[
            "mean_selected_cycle_rank_ratio"
        ]
        >= float(DECISION_RULE["mean_selected_cycle_rank_ratio_min"]),
        "worst_selected_cycle_rank_ratio": observed[
            "worst_selected_cycle_rank_ratio"
        ]
        >= float(DECISION_RULE["worst_selected_cycle_rank_ratio_min"]),
    }
    passed = all(checks.values())
    return {
        "status": (
            "go_E23_source_group_disjoint_confirmation_same_generator"
            if passed
            else "kill_existing_affinity_full_union_generator"
        ),
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": (
            "candidate_availability_discovery_ceiling_not_deployable_"
            "no_GPU_training_authority"
        ),
    }


def _validate_complete_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_digest: str,
    scenes: Sequence[e12.RawScene],
) -> None:
    expected_keys = {
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
    if set(report) != expected_keys:
        raise E22ContractError("existing E22 complete report fields drifted")
    if (
        _integer(report.get("schema_version"), label="E22 schema version")
        != SCHEMA_VERSION
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != E22_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E22_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E22ContractError("existing E22 complete report contract drifted")
    _finite(
        report.get("runtime_seconds"),
        label="existing E22 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    rows = report.get("rows")
    if (
        not isinstance(rows, list)
        or report.get("completed_images") != list(e12.CALIBRATION_IDS)
        or [row.get("image") for row in rows if isinstance(row, Mapping)]
        != list(e12.CALIBRATION_IDS)
    ):
        raise E22ContractError("existing E22 rows/completion IDs drifted")
    by_image = {
        _integer(row.get("image"), label="existing row image"): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if (
        len(rows) != len(e12.CALIBRATION_IDS)
        or len(by_image) != len(rows)
        or tuple(sorted(by_image)) != e12.CALIBRATION_IDS
    ):
        raise E22ContractError("existing E22 rows are incomplete or duplicated")
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    for image in e12.CALIBRATION_IDS:
        scene = scene_by_image[image]
        candidate_ids, raw_logits, _valid = _scene_arrays(scene)
        _validate_success_row(
            by_image[image],
            scene=scene,
            candidate_ids=candidate_ids,
            raw_logits=raw_logits,
        )
    expected_summary = summarize(rows)
    expected_decision = decision(expected_summary)
    if report.get("summary") != expected_summary:
        raise E22ContractError("existing E22 summary drifted")
    if report.get("decision") != expected_decision:
        raise E22ContractError("existing E22 decision drifted")
    if report.get("stage") != expected_decision["status"]:
        raise E22ContractError("existing E22 terminal stage drifted")


def _load_verified_raw_inputs(
    paths: E22Paths,
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
        raise E22ContractError(str(exc)) from exc


def run_gate(paths: E22Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E22 report")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e21_report_path = _require_e_drive(paths.e21_report, label="E21 report")
    calibration_path = paths.calibration_report.resolve()
    if report_path.suffix.lower() != ".json":
        raise E22ContractError("E22 report must be a .json file")
    if report_path in {e12_report_path, e21_report_path, calibration_path}:
        raise E22ContractError("E22 report must not overwrite an input")
    if report_path.is_relative_to(raw_cache_dir):
        raise E22ContractError("E22 report must not be written inside the raw cache")

    e21_report = _verify_e21_kill(e21_report_path)
    e12_report, calibration, scenes = _load_verified_raw_inputs(paths)
    if tuple(int(scene.image_id) for scene in scenes) != e12.CALIBRATION_IDS:
        raise E22ContractError("E22 inputs are not exact E12 scenes 10..17")
    scene_records = [e12.scene_provenance(scene) for scene in scenes]
    contract = {
        "protocol_sha256": e12.canonical_digest(E22_PROTOCOL),
        "e21_report": {
            "path": str(e21_report_path),
            "sha256": EXPECTED_E21_REPORT_SHA256,
            "run_contract_sha256": str(e21_report["run_contract_sha256"]),
            "stage": str(e21_report["stage"]),
        },
        "e12_report": {
            "path": str(e12_report_path),
            "sha256": EXPECTED_E12_REPORT_SHA256,
            "scene_provenance_digest": str(e12_report["scene_provenance_digest"]),
        },
        "calibration_report": {
            "path": str(calibration_path),
            "sha256": e12.CALIBRATION_REPORT_SHA256,
        },
        "raw_cache_dir": str(raw_cache_dir),
        "raw_scenes": scene_records,
        "raw_scenes_sha256": e12.canonical_digest(scene_records),
        "orientation": {
            "tile_orientation_degrees": [0],
            "forbidden_tile_orientation_degrees": [90, 180, 270],
            "reflection": False,
            "rcce4_variants_are_adjacency_orderings_not_rotations": True,
        },
        "report": str(report_path),
        "source_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }
    if not isinstance(calibration, Mapping):
        raise E22ContractError("verified calibration payload is malformed")
    contract_digest = e12.canonical_digest(contract)
    if report_path.is_file():
        existing = _load_json(report_path, label="existing E22 report")
        if existing.get("run_contract_sha256") != contract_digest:
            raise E22ContractError("existing E22 report belongs to different bytes")
        if existing.get("run_contract") != contract:
            raise E22ContractError("existing E22 report contract payload drifted")
        if existing.get("status") == "complete":
            _validate_complete_report(
                existing,
                contract=contract,
                contract_digest=contract_digest,
                scenes=scenes,
            )
            return existing

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "upright_rcce4_candidate_availability_ceiling",
        "protocol": E22_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E22_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rows": [],
        "completed_images": [],
        "decision": {"status": "not_run"},
    }
    _atomic_write_json(report_path, output)
    try:
        for scene in scenes:
            candidate_ids, raw_logits, _valid = _scene_arrays(scene)
            # Frozen label-free boundary: exactly two positional array inputs.
            candidate_digest_before = e12.array_sha256(candidate_ids)
            logit_digest_before = e12.array_sha256(raw_logits)
            result = rcce.run_rcce4_candidate_oracle(candidate_ids, raw_logits)
            if (
                e12.array_sha256(candidate_ids) != candidate_digest_before
                or e12.array_sha256(raw_logits) != logit_digest_before
            ):
                raise E22ContractError("candidate core mutated its frozen raw inputs")
            row = evaluate_scene(
                scene,
                result,
                candidate_ids=candidate_ids,
                raw_logits=raw_logits,
            )
            # ``evaluate_scene`` already performed the complete independent
            # replay.  Avoid a duplicate near-cap evaluation on fresh output;
            # existing complete reports still replay every row from raw bytes.
            if row["core_sha256"] != e12.canonical_digest(row["core"]):
                raise E22ContractError("fresh E22 core hash drifted")
            if row["oracle_sha256"] != e12.canonical_digest(row["oracle"]):
                raise E22ContractError("fresh E22 oracle hash drifted")
            output["rows"].append(row)
            output["completed_images"].append(int(scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
            # Release the potentially near-cap pool before the next core call;
            # assignment would otherwise keep it alive while constructing RHS.
            del result
        summary = summarize(output["rows"])
        result_decision = decision(summary)
        output["summary"] = summary
        output["decision"] = result_decision
        output["status"] = "complete"
        output["stage"] = result_decision["status"]
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
        description=(
            "Run fixed CPU-only upright E22 RCCE-4 full-union candidate ceiling."
        )
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e21-report", type=Path, default=DEFAULT_E21_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gate(
        E22Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e21_report=args.e21_report,
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
