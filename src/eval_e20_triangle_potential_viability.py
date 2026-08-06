"""Frozen E20 structure-only evaluator for triangle-supported pose DSU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import skimage

import e18_absolute_frame_beam as e18_core
import e20_triangle_potential_dsu as pose
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
import eval_e17_cc192_rigid_viability as e17
import eval_e18_absolute_frame_oracle as e18_eval
import eval_e19_relative_frame_viability as e19_eval


class E20ContractError(RuntimeError):
    """The frozen E20 protocol, input bytes, or report drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e20-cc192-triangle-potential-viability-report-v1"
EXPERIMENT = "e20_cc192_top8_triangle_signed_potential_dsu_v1"
EXPECTED_E12_REPORT_SHA256 = e14.EXPECTED_E12_REPORT_SHA256
EXPECTED_E19_REPORT_SHA256 = (
    "9a881793cbbfaa7f4da616e5a283d9f4cb4ad28a13e5605ff88aa05939bc3314"
)
EXPECTED_E19_RUN_CONTRACT_SHA256 = (
    "da327f546803f4efad2cfb07d5dd669123b74376ef73f34a010e5394921c14d1"
)
EXPECTED_E19_PROTOCOL_SHA256 = (
    "f9c5de6e9618991cde255b1e1387bed0f8113415eaffc4a572fad8542dc6bb9f"
)
EXPECTED_RUNTIME_PROVENANCE = dict(e18_eval.EXPECTED_RUNTIME_PROVENANCE)
E19_SHARED_SOURCE_NAMES = (
    "e15_frame_consensus.py",
    "e18_absolute_frame_beam.py",
    "eval_clean_score_oracle.py",
    "eval_e14_cc192_discovery.py",
    "eval_e17_cc192_rigid_viability.py",
    "eval_e18_absolute_frame_oracle.py",
    "eval_e19_relative_frame_viability.py",
    "rank96_lab_selector.py",
    "solve_buddies.py",
)

DECISION_RULE: dict[str, float | int] = {
    "completed_scenes": 8,
    "legal_origin_scenes": 8,
    "mean_rigid_coverage_min": 0.35,
    "worst_rigid_coverage_min": 0.25,
    "mean_exact_pose_coverage_min": 0.30,
    "worst_exact_pose_coverage_min": 0.20,
    "mean_exact_relative_pose_precision_min": 0.90,
    "worst_exact_relative_pose_precision_min": 0.80,
    "mean_accepted_relation_precision_min": 0.85,
    "worst_accepted_relation_precision_min": 0.70,
    "mean_accepted_cross_seam_precision_min": 0.85,
    "worst_accepted_cross_seam_precision_min": 0.70,
    "mean_component_cycle_rank_ratio_min": 0.05,
}

E20_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e20-cc192-top8-triangle-potential-dsu-v1",
    "role": "target_derived_clean_score_structure_gate_no_board",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "authorization": {
        "e19_report_sha256": EXPECTED_E19_REPORT_SHA256,
        "e19_run_contract_sha256": EXPECTED_E19_RUN_CONTRACT_SHA256,
        "e19_protocol_sha256": EXPECTED_E19_PROTOCOL_SHA256,
        "required_status": "complete",
        "required_stage": "kill_relative_cap",
        "required_cap_image": 10,
        "required_proposal_evaluations": 500000,
        "required_rounds": 32,
    },
    "input_graph": {
        "components": "exact_E18_E19_CC192_nontrivial_rigid",
        "claims": "exact_positive_dense_top8_U_D_L_R_cross_component",
        "top_k": 8,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
    },
    "hypotheses": {
        "key": "u_lt_v_plus_exact_signed_offset",
        "physical_seams": "canonical_deduplicated",
        "reciprocal_same_physical_seam": "rank_evidence_not_independent_path",
        "direct_score": "sum_and_max_of_once_only_physical_seam_maxima",
        "component_pair_offset_precollapse": False,
    },
    "triangles": {
        "incident_hypotheses_per_component": 8,
        "incident_rank": [
            "physical_seams_desc",
            "reciprocal_seams_desc",
            "direct_sum_desc",
            "direct_max_desc",
            "other_component_asc",
            "hypothesis_key_asc",
        ],
        "leg_pairs_per_intermediate_max": 28,
        "composition": "exact_signed_two_leg_offset_to_existing_direct_hypothesis",
        "witness_dedupe": "one_per_distinct_intermediate_component",
        "strong_witness": "both_legs_each_have_two_unique_physical_seams",
        "bottleneck": "min_leg_direct_sum",
        "same_intermediate_tie": [
            "bottleneck_desc",
            "min_leg_direct_max_desc",
            "leg_ids_asc",
        ],
    },
    "selection": {
        "independent_paths": "unique_physical_seams_plus_distinct_triangle_intermediates",
        "minimum_independent_paths": 2,
        "kruskal_rank": [
            "independent_paths_desc",
            "triangle_intermediates_desc",
            "strong_triangle_intermediates_desc",
            "physical_seams_desc",
            "reciprocal_seams_desc",
            "triangle_bottleneck_sum_desc",
            "direct_sum_desc",
            "direct_max_desc",
            "u_v_dr_dc_asc",
        ],
        "potential": "translation_node_minus_translation_parent",
        "union": "one_irreversible_union_by_size_with_path_compression",
        "merge_geometry": "all_claimed_contacts_cardinal_no_collision_bbox_each_at_most_24",
        "connected_exact_offset": "cycle_evidence",
        "connected_different_offset": "pose_conflict",
        "weak_hypotheses": "diagnostic_only_never_selected_evidence",
        "rollback": False,
        "beam": False,
    },
    "output": {
        "cluster_normalisation": "subtract_minimum_occupied_row_and_column",
        "one_cluster_label_free_rank": [
            "rigid_tiles_desc",
            "component_cycle_rank_desc",
            "accepted_physical_seams_desc",
            "accepted_once_only_neural_sum_desc",
            "minimum_tile_asc",
            "translations_asc",
        ],
        "sparse_only": True,
        "absolute_board": False,
        "legal_origins": "analytic_after_selection",
    },
    "measurement": {
        "labels": "evaluator_only_after_core_returns",
        "exact_pose_bin": "modal_truth_coordinate_minus_relative_coordinate",
        "modal_tie": "lexicographically_smallest_offset",
        "exact_pose_coverage": "modal_tiles_divided_by_576",
        "exact_relative_pose_precision": "modal_tiles_divided_by_selected_tiles",
        "relation_truth": "both_whole_components_exact_and_signed_delta_exact",
        "empty_relation_or_seam_precision": 0.0,
    },
    "decision": dict(DECISION_RULE),
    "routing": {
        "pass": "open_separately_frozen_E21_one_cluster_absolute_origin_residual",
        "fail": "close_exact_top8_triangle_potential_route",
    },
    "excluded": [
        "absolute_board",
        "residual_completion",
        "placement",
        "neighbour",
        "SSIM",
        "NLM",
        "labels_inside_selection",
        "modal_trim_inside_algorithm",
        "threshold_topk_support_sweep",
        "rotation",
        "reflection",
        "GPU",
        "diffusion",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_E19_REPORT = Path(
    "E:/pazzle_work/relative_frame_e19/cc192_origin_quotient_viability_v1.json"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/triangle_pose_e20/cc192_triangle_potential_viability_v1.json"
)


@dataclass(frozen=True)
class E20Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e19_report: Path
    report: Path


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E20ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E20ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E20ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E20 report")
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
        raise E20ContractError(
            f"E20 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e15_frame_consensus.py": source / "e15_frame_consensus.py",
        "e18_absolute_frame_beam.py": source / "e18_absolute_frame_beam.py",
        "e20_triangle_potential_dsu.py": source / "e20_triangle_potential_dsu.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_e17_cc192_rigid_viability.py": source
        / "eval_e17_cc192_rigid_viability.py",
        "eval_e18_absolute_frame_oracle.py": source
        / "eval_e18_absolute_frame_oracle.py",
        "eval_e19_relative_frame_viability.py": source
        / "eval_e19_relative_frame_viability.py",
        "eval_e20_triangle_potential_viability.py": Path(__file__).resolve(),
        "rank96_lab_selector.py": source / "rank96_lab_selector.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _verify_e19_cap_kill(path: Path) -> Mapping[str, Any]:
    resolved = _require_e_drive(path, label="E19 report")
    if not resolved.is_file():
        raise E20ContractError(f"E19 report is missing: {resolved}")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E19_REPORT_SHA256:
        raise E20ContractError(
            "E19 report SHA256 mismatch: "
            f"expected {EXPECTED_E19_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E19 report")
    cap = report.get("cap_failure")
    contract = report.get("run_contract")
    if (
        report.get("schema_version") != e19_eval.SCHEMA_VERSION
        or report.get("schema") != e19_eval.REPORT_SCHEMA
        or report.get("experiment") != e19_eval.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("stage") != "kill_relative_cap"
        or report.get("protocol") != e19_eval.E19_PROTOCOL
        or report.get("protocol_sha256") != EXPECTED_E19_PROTOCOL_SHA256
        or report.get("run_contract_sha256")
        != EXPECTED_E19_RUN_CONTRACT_SHA256
        or report.get("rows") != []
        or report.get("completed_images") != []
        or "summary" in report
        or not isinstance(cap, Mapping)
        or set(cap)
        != {
            "image",
            "validation_name",
            "clean_score_cache_sha256",
            "proposal_evaluations",
            "rounds",
            "initial_states",
            "cap_hit",
            "error",
        }
        or _integer(cap.get("image"), label="E19 cap image") != 10
        or _integer(
            cap.get("proposal_evaluations"), label="E19 cap evaluations"
        )
        != 500000
        or _integer(cap.get("rounds"), label="E19 cap rounds") != 32
        or _integer(cap.get("initial_states"), label="E19 initial states") != 1
        or cap.get("cap_hit") is not True
        or cap.get("validation_name") != "img_006710.png"
        or cap.get("clean_score_cache_sha256")
        != "3cbc4006fa43643c57668a0932e3b2e945e86b1d38eca95d55e3b5725797ce13"
        or cap.get("error")
        != (
            "RelativeFrameCapError: relative beam reached the frozen cumulative "
            "proposal cap"
        )
        or report.get("decision") != e19_eval.cap_decision(cap)
        or not isinstance(contract, Mapping)
        or contract.get("protocol_sha256") != EXPECTED_E19_PROTOCOL_SHA256
    ):
        raise E20ContractError("E19 relative-cap KILL contract drifted")
    frozen_sources = contract.get("source_provenance")
    if not isinstance(frozen_sources, Mapping):
        raise E20ContractError("E19 source provenance is malformed")
    source_dir = Path(__file__).resolve().parent
    for name in E19_SHARED_SOURCE_NAMES:
        observed = e12.sha256_file(source_dir / name)
        if frozen_sources.get(name) != observed:
            raise E20ContractError(
                f"shared E19-to-E20 source drifted for {name}: "
                f"expected {frozen_sources.get(name)}, got {observed}"
            )
    runtime = report.get("runtime_seconds")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise E20ContractError("E19 report runtime is invalid")
    return report


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise E20ContractError(f"{label} is not an integer")
    return int(value)


def _finite(
    value: object, *, label: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise E20ContractError(f"{label} is not numeric")
    observed = float(value)
    if not math.isfinite(observed):
        raise E20ContractError(f"{label} is not finite")
    if not minimum <= observed <= maximum:
        raise E20ContractError(f"{label} is outside [{minimum}, {maximum}]")
    return observed


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _tuple_rows(
    value: object, *, width: int, label: str
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise E20ContractError(f"{label} is not a sequence")
    output: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise E20ContractError(f"{label} contains a malformed row")
        output.append(
            tuple(_integer(item, label=f"{label} value") for item in row)
        )
    return tuple(output)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    raise E20ContractError(f"core payload contains unsupported type {type(value)}")


def _cluster_payload(cluster: pose.PoseCluster) -> dict[str, Any]:
    return {
        "translations": _jsonable(cluster.translations),
        "relative_entries": _jsonable(cluster.relative_entries),
        "bbox": _jsonable(cluster.bbox),
        "bbox_height": _jsonable(cluster.bbox_height),
        "bbox_width": _jsonable(cluster.bbox_width),
        "legal_origin_bounds": _jsonable(cluster.legal_origin_bounds),
        "legal_origin_count": _jsonable(cluster.legal_origin_count),
        "component_ids": _jsonable(cluster.component_ids),
        "tree_hypothesis_ids": _jsonable(cluster.tree_hypothesis_ids),
        "cycle_hypothesis_ids": _jsonable(cluster.cycle_hypothesis_ids),
        "accepted_hypothesis_ids": _jsonable(cluster.accepted_hypothesis_ids),
        "accepted_relations": _jsonable(cluster.accepted_relations),
        "component_contacts": _jsonable(cluster.component_contacts),
        "accepted_cross_seams": _jsonable(cluster.accepted_cross_seams),
        "rigid_tiles": _jsonable(cluster.rigid_tiles),
        "rigid_coverage": _jsonable(cluster.rigid_coverage),
        "component_cycle_rank": _jsonable(cluster.component_cycle_rank),
        "component_cycle_rank_ratio": _jsonable(cluster.component_cycle_rank_ratio),
        "cross_neural_sum": _jsonable(cluster.cross_neural_sum),
        "minimum_tile": _jsonable(cluster.minimum_tile),
    }


def _core_payload(result: pose.TrianglePoseResult) -> dict[str, Any]:
    payload = {
        "selected": _cluster_payload(result.selected),
        "diagnostics": _jsonable(result.diagnostics),
    }
    forbidden = (
        "board",
        "canvas",
        "residual",
        "placement",
        "neighbour",
        "ssim",
        "nlm",
        "target",
        "permutation",
    )

    def check_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if any(marker in str(key).lower() for marker in forbidden):
                    raise E20ContractError(f"core payload contains forbidden key {key}")
                check_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_keys(item)

    check_keys(payload)
    return payload


def _validate_permutation(permutation: np.ndarray) -> np.ndarray:
    value = np.asarray(permutation)
    if value.shape != (e12.NFRAG,) or value.dtype.kind not in "iu":
        raise E20ContractError("permutation geometry/dtype drifted")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(e12.NFRAG)):
        raise E20ContractError("permutation is not a bijection")
    return value


def _seam_is_true(seam: Sequence[int], permutation: np.ndarray) -> bool:
    values = _tuple_rows([seam], width=4, label="accepted cross seam")[0]
    first, second, dy, dx = values
    if (dy, dx) not in {(0, 1), (1, 0)}:
        raise E20ContractError("accepted cross seam is not canonical right/down")
    if not (0 <= first < e12.NFRAG and 0 <= second < e12.NFRAG and first != second):
        raise E20ContractError("accepted cross seam tile ID is invalid")
    value = _validate_permutation(permutation)
    first_row, first_col = divmod(int(value[first]), 24)
    second_row, second_col = divmod(int(value[second]), 24)
    return (second_row - first_row, second_col - first_col) == (dy, dx)


def _component_truth_shifts(
    graph: e18_core.GraphData, permutation: np.ndarray
) -> dict[int, tuple[int, int] | None]:
    value = _validate_permutation(permutation)
    output: dict[int, tuple[int, int] | None] = {}
    for component in graph.components:
        offsets = {
            (
                int(value[tile] // 24) - int(local_row),
                int(value[tile] % 24) - int(local_col),
            )
            for tile, local_row, local_col in component.entries
        }
        output[component.component_id] = next(iter(offsets)) if len(offsets) == 1 else None
    return output


def measure_selected(
    cluster: pose.PoseCluster,
    graph: e18_core.GraphData,
    permutation: np.ndarray,
) -> dict[str, Any]:
    value = _validate_permutation(permutation)
    entries = _tuple_rows(
        cluster.relative_entries, width=3, label="selected relative entries"
    )
    if not entries:
        raise E20ContractError("selected pose cluster has no tiles")
    offsets = [
        (
            int(value[tile] // 24) - row,
            int(value[tile] % 24) - col,
        )
        for tile, row, col in entries
    ]
    counts = Counter(offsets)
    modal_count = max(counts.values())
    modal_offset = min(offset for offset, count in counts.items() if count == modal_count)
    rigid_tiles = len(entries)

    truth_shifts = _component_truth_shifts(graph, value)
    relations = _tuple_rows(
        cluster.accepted_relations, width=4, label="selected accepted relations"
    )
    if len(set(relations)) != len(relations):
        raise E20ContractError("accepted relations are duplicated")
    true_relations = 0
    for u, v, dr, dc in relations:
        if not (0 <= u < v < len(graph.components)):
            raise E20ContractError("accepted relation component IDs are invalid")
        left = truth_shifts[u]
        right = truth_shifts[v]
        true_relations += int(
            left is not None
            and right is not None
            and (right[0] - left[0], right[1] - left[1]) == (dr, dc)
        )

    seams = _tuple_rows(
        cluster.accepted_cross_seams, width=4, label="selected accepted seams"
    )
    if len(set(seams)) != len(seams):
        raise E20ContractError("accepted cross seams are duplicated")
    true_seams = sum(_seam_is_true(seam, value) for seam in seams)
    relation_count = len(relations)
    seam_count = len(seams)
    return {
        "rigid_tiles": int(rigid_tiles),
        "rigid_coverage": float(rigid_tiles / e12.NFRAG),
        "modal_truth_offset": list(modal_offset),
        "exact_pose_tiles": int(modal_count),
        "exact_pose_coverage": float(modal_count / e12.NFRAG),
        "exact_relative_pose_precision": float(modal_count / rigid_tiles),
        "accepted_relations": int(relation_count),
        "true_accepted_relations": int(true_relations),
        "accepted_relation_precision": float(
            true_relations / relation_count if relation_count else 0.0
        ),
        "accepted_cross_seams": int(seam_count),
        "true_accepted_cross_seams": int(true_seams),
        "accepted_cross_seam_precision": float(
            true_seams / seam_count if seam_count else 0.0
        ),
        "component_cycle_rank_ratio": _finite(
            cluster.component_cycle_rank_ratio,
            label="selected component cycle rank ratio",
            minimum=0.0,
            maximum=float("inf"),
        ),
        "legal_origin_count": _integer(
            cluster.legal_origin_count, label="selected legal origin count"
        ),
    }


def _validate_core_geometry(
    payload: Mapping[str, Any],
    *,
    graph: e18_core.GraphData,
    right: np.ndarray,
    down: np.ndarray,
) -> None:
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise E20ContractError("selected core cluster is missing")
    translations = _tuple_rows(
        selected.get("translations"), width=3, label="component translations"
    )
    component_ids_raw = selected.get("component_ids")
    if not isinstance(component_ids_raw, (list, tuple)):
        raise E20ContractError("selected component IDs are malformed")
    component_ids = tuple(
        _integer(value, label="selected component ID") for value in component_ids_raw
    )
    if (
        not translations
        or translations != tuple(sorted(translations))
        or component_ids != tuple(sorted(component_ids))
        or tuple(value[0] for value in translations) != component_ids
        or len(set(component_ids)) != len(component_ids)
        or any(value not in graph.nontrivial for value in component_ids)
    ):
        raise E20ContractError("selected component translation key drifted")
    translation_map = {cid: (row, col) for cid, row, col in translations}
    expected_entries = tuple(
        sorted(
            (
                int(tile),
                int(local_row + translation_map[cid][0]),
                int(local_col + translation_map[cid][1]),
            )
            for cid in component_ids
            for tile, local_row, local_col in graph.components[cid].entries
        )
    )
    entries = _tuple_rows(
        selected.get("relative_entries"), width=3, label="relative entries"
    )
    if entries != expected_entries or len({(r, c) for _t, r, c in entries}) != len(entries):
        raise E20ContractError("selected relative entries drifted or overlap")
    rows = [row for _tile, row, _col in entries]
    cols = [col for _tile, _row, col in entries]
    bbox = (min(rows), max(rows), min(cols), max(cols))
    if min(rows) != 0 or min(cols) != 0:
        raise E20ContractError("selected cluster is not min-coordinate normalized")
    stored_bbox = tuple(
        _integer(value, label="bbox value") for value in selected.get("bbox", [])
    )
    height = bbox[1] - bbox[0] + 1
    width = bbox[3] - bbox[2] + 1
    expected_bounds = (-bbox[0], 23 - bbox[1], -bbox[2], 23 - bbox[3])
    bounds = tuple(
        _integer(value, label="legal origin bound")
        for value in selected.get("legal_origin_bounds", [])
    )
    origin_count = (25 - height) * (25 - width)
    if (
        stored_bbox != bbox
        or not 1 <= height <= 24
        or not 1 <= width <= 24
        or _integer(selected.get("bbox_height"), label="bbox height") != height
        or _integer(selected.get("bbox_width"), label="bbox width") != width
        or bounds != expected_bounds
        or _integer(selected.get("legal_origin_count"), label="origin count")
        != origin_count
        or origin_count < 1
    ):
        raise E20ContractError("selected bbox/legal-origin algebra drifted")
    rigid_tiles = _integer(selected.get("rigid_tiles"), label="rigid tiles")
    if rigid_tiles != len(entries) or not math.isclose(
        _finite(
            selected.get("rigid_coverage"),
            label="rigid coverage",
            minimum=0.0,
            maximum=1.0,
        ),
        rigid_tiles / e12.NFRAG,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise E20ContractError("selected rigid coverage drifted")

    accepted_ids_raw = selected.get("accepted_hypothesis_ids")
    tree_ids_raw = selected.get("tree_hypothesis_ids")
    cycle_ids_raw = selected.get("cycle_hypothesis_ids")
    if not all(isinstance(value, (list, tuple)) for value in (accepted_ids_raw, tree_ids_raw, cycle_ids_raw)):
        raise E20ContractError("selected hypothesis IDs are malformed")
    accepted_ids = tuple(_integer(value, label="accepted hypothesis ID") for value in accepted_ids_raw)
    tree_ids = tuple(_integer(value, label="tree hypothesis ID") for value in tree_ids_raw)
    cycle_ids = tuple(_integer(value, label="cycle hypothesis ID") for value in cycle_ids_raw)
    if (
        accepted_ids != tuple(sorted(accepted_ids))
        or tree_ids != tuple(sorted(tree_ids))
        or cycle_ids != tuple(sorted(cycle_ids))
        or len(set(accepted_ids)) != len(accepted_ids)
        or len(set(tree_ids)) != len(tree_ids)
        or len(set(cycle_ids)) != len(cycle_ids)
        or set(tree_ids) & set(cycle_ids)
        or set(accepted_ids) != set(tree_ids) | set(cycle_ids)
        or len(tree_ids) != len(component_ids) - 1
    ):
        raise E20ContractError("selected tree/cycle hypothesis algebra drifted")
    relations = _tuple_rows(
        selected.get("accepted_relations"), width=4, label="accepted relations"
    )
    contacts = _tuple_rows(
        selected.get("component_contacts"), width=2, label="component contacts"
    )
    hypotheses = pose.add_triangle_support(pose.build_pose_hypotheses(graph))
    by_hypothesis = {
        hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses
    }
    if any(value not in by_hypothesis for value in accepted_ids):
        raise E20ContractError("selected hypothesis ID is outside the frozen graph")
    accepted_hypotheses = [by_hypothesis[value] for value in accepted_ids]
    if any(not hypothesis.eligible for hypothesis in accepted_hypotheses):
        raise E20ContractError("weak hypothesis leaked into selected evidence")
    expected_relations = tuple(
        hypothesis.relation for hypothesis in accepted_hypotheses
    )
    expected_contacts = tuple(
        sorted({(hypothesis.u, hypothesis.v) for hypothesis in accepted_hypotheses})
    )
    if (
        len(relations) != len(accepted_ids)
        or relations != expected_relations
        or contacts != expected_contacts
    ):
        raise E20ContractError("selected relation/contact algebra drifted")
    for u, v, dr, dc in relations:
        if u not in translation_map or v not in translation_map:
            raise E20ContractError("selected relation leaves its pose cluster")
        expected_delta = (
            translation_map[v][0] - translation_map[u][0],
            translation_map[v][1] - translation_map[u][1],
        )
        if expected_delta != (dr, dc):
            raise E20ContractError("selected relation contradicts translations")
    cycle_rank = max(0, len(contacts) - len(component_ids) + 1)
    cycle_ratio = float(cycle_rank / max(1, len(component_ids) - 1))
    if (
        _integer(selected.get("component_cycle_rank"), label="cycle rank") != cycle_rank
        or not math.isclose(
            _finite(
                selected.get("component_cycle_rank_ratio"),
                label="cycle ratio",
                minimum=0.0,
                maximum=float("inf"),
            ),
            cycle_ratio,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise E20ContractError("selected cycle algebra drifted")

    seams = _tuple_rows(
        selected.get("accepted_cross_seams"), width=4, label="accepted seams"
    )
    expected_seam_scores: dict[tuple[int, int, int, int], float] = {}
    for hypothesis in accepted_hypotheses:
        for seam, score in hypothesis.seam_scores:
            expected_seam_scores[seam] = max(
                expected_seam_scores.get(seam, 0.0), float(score)
            )
    if seams != tuple(sorted(expected_seam_scores)):
        raise E20ContractError("selected physical seams are not unique/canonical")
    occupied = {(row, col): tile for tile, row, col in entries}
    tile_position = {tile: (row, col) for tile, row, col in entries}
    for first, second, dy, dx in seams:
        if (dy, dx) not in {(0, 1), (1, 0)}:
            raise E20ContractError("selected seam direction is not canonical")
        if first not in tile_position or second not in tile_position:
            raise E20ContractError("selected seam tile is outside selected cluster")
        first_position = tile_position[first]
        second_position = tile_position[second]
        if (
            second_position[0] - first_position[0],
            second_position[1] - first_position[1],
        ) != (dy, dx):
            raise E20ContractError("selected seam is not a physical contact")
    e18_core._dense(right, label="right")
    e18_core._dense(down, label="down")
    neural = float(sum(expected_seam_scores.values()))
    if not math.isclose(
        _finite(
            selected.get("cross_neural_sum"),
            label="cross neural sum",
            minimum=0.0,
            maximum=float("inf"),
        ),
        neural,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise E20ContractError("selected cross neural sum drifted")
    minimum_tile = min(tile for tile, _row, _col in entries)
    if _integer(selected.get("minimum_tile"), label="minimum tile") != minimum_tile:
        raise E20ContractError("selected minimum tile drifted")


def _validate_metric_payload(
    metrics: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    if metrics != expected:
        raise E20ContractError("E20 label-derived metrics drifted")


def evaluate_structure(
    scene: e12.RawScene,
    result: pose.TrianglePoseResult,
    *,
    clean_score_cache_sha256: str,
    graph: e18_core.GraphData,
) -> dict[str, Any]:
    core_payload = _core_payload(result)
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "clean_score_cache_sha256": str(clean_score_cache_sha256),
        "arm": "E20_triangle_potential_DSU",
        "core": core_payload,
        "core_sha256": e12.canonical_digest(core_payload),
        "metrics": measure_selected(result.selected, graph, scene.permutation),
    }


def _validate_success_row(
    row: Mapping[str, Any],
    *,
    scene: e12.RawScene,
    cache_sha256: str,
    right: np.ndarray,
    down: np.ndarray,
    expected_result: pose.TrianglePoseResult | None = None,
) -> None:
    expected_keys = {
        "image",
        "validation_name",
        "clean_score_cache_sha256",
        "arm",
        "core",
        "core_sha256",
        "metrics",
    }
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise E20ContractError("E20 row fields drifted")
    if (
        _integer(row.get("image"), label="row image") != int(scene.image_id)
        or row.get("validation_name") != str(scene.validation_name)
        or row.get("clean_score_cache_sha256") != cache_sha256
        or not _is_sha256(row.get("clean_score_cache_sha256"))
        or row.get("arm") != "E20_triangle_potential_DSU"
    ):
        raise E20ContractError(f"E20 row provenance drifted for image {scene.image_id}")
    result = (
        expected_result
        if expected_result is not None
        else pose.run_triangle_potential_dsu(right, down)
    )
    expected_core = _core_payload(result)
    if row.get("core") != expected_core:
        raise E20ContractError(f"E20 core replay drifted for image {scene.image_id}")
    if row.get("core_sha256") != e12.canonical_digest(expected_core):
        raise E20ContractError(f"E20 core payload hash drifted for image {scene.image_id}")
    graph = e18_core.build_graph_data(right, down)
    _validate_core_geometry(expected_core, graph=graph, right=right, down=down)
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise E20ContractError(f"E20 metrics are missing for image {scene.image_id}")
    expected_metrics = measure_selected(result.selected, graph, scene.permutation)
    _validate_metric_payload(metrics, expected=expected_metrics)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E20ContractError("E20 summary requires exactly eight rows")
    images = [int(row.get("image", -1)) for row in rows]
    if len(set(images)) != len(images) or tuple(sorted(images)) != e12.CALIBRATION_IDS:
        raise E20ContractError("E20 row image IDs drifted")
    metrics = [row["metrics"] for row in rows]

    def mean(key: str) -> float:
        return float(np.mean([float(value[key]) for value in metrics]))

    def worst(key: str) -> float:
        return float(min(float(value[key]) for value in metrics))

    return {
        "images": len(rows),
        "completed_scenes": len(rows),
        "legal_origin_scenes": int(
            sum(int(value["legal_origin_count"]) >= 1 for value in metrics)
        ),
        "mean_rigid_coverage": mean("rigid_coverage"),
        "worst_rigid_coverage": worst("rigid_coverage"),
        "mean_exact_pose_coverage": mean("exact_pose_coverage"),
        "worst_exact_pose_coverage": worst("exact_pose_coverage"),
        "mean_exact_relative_pose_precision": mean(
            "exact_relative_pose_precision"
        ),
        "worst_exact_relative_pose_precision": worst(
            "exact_relative_pose_precision"
        ),
        "mean_accepted_relation_precision": mean(
            "accepted_relation_precision"
        ),
        "worst_accepted_relation_precision": worst(
            "accepted_relation_precision"
        ),
        "mean_accepted_cross_seam_precision": mean(
            "accepted_cross_seam_precision"
        ),
        "worst_accepted_cross_seam_precision": worst(
            "accepted_cross_seam_precision"
        ),
        "mean_component_cycle_rank_ratio": mean(
            "component_cycle_rank_ratio"
        ),
        "mean_rigid_tiles": mean("rigid_tiles"),
        "mean_exact_pose_tiles": mean("exact_pose_tiles"),
        "total_accepted_relations": int(
            sum(int(value["accepted_relations"]) for value in metrics)
        ),
        "total_accepted_cross_seams": int(
            sum(int(value["accepted_cross_seams"]) for value in metrics)
        ),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "completed_scenes": int(summary["completed_scenes"]),
        "legal_origin_scenes": int(summary["legal_origin_scenes"]),
        "mean_rigid_coverage": float(summary["mean_rigid_coverage"]),
        "worst_rigid_coverage": float(summary["worst_rigid_coverage"]),
        "mean_exact_pose_coverage": float(summary["mean_exact_pose_coverage"]),
        "worst_exact_pose_coverage": float(summary["worst_exact_pose_coverage"]),
        "mean_exact_relative_pose_precision": float(
            summary["mean_exact_relative_pose_precision"]
        ),
        "worst_exact_relative_pose_precision": float(
            summary["worst_exact_relative_pose_precision"]
        ),
        "mean_accepted_relation_precision": float(
            summary["mean_accepted_relation_precision"]
        ),
        "worst_accepted_relation_precision": float(
            summary["worst_accepted_relation_precision"]
        ),
        "mean_accepted_cross_seam_precision": float(
            summary["mean_accepted_cross_seam_precision"]
        ),
        "worst_accepted_cross_seam_precision": float(
            summary["worst_accepted_cross_seam_precision"]
        ),
        "mean_component_cycle_rank_ratio": float(
            summary["mean_component_cycle_rank_ratio"]
        ),
    }
    checks = {
        "completed_scenes": observed["completed_scenes"]
        == int(DECISION_RULE["completed_scenes"]),
        "legal_origin_scenes": observed["legal_origin_scenes"]
        == int(DECISION_RULE["legal_origin_scenes"]),
        "mean_rigid_coverage": observed["mean_rigid_coverage"]
        >= float(DECISION_RULE["mean_rigid_coverage_min"]),
        "worst_rigid_coverage": observed["worst_rigid_coverage"]
        >= float(DECISION_RULE["worst_rigid_coverage_min"]),
        "mean_exact_pose_coverage": observed["mean_exact_pose_coverage"]
        >= float(DECISION_RULE["mean_exact_pose_coverage_min"]),
        "worst_exact_pose_coverage": observed["worst_exact_pose_coverage"]
        >= float(DECISION_RULE["worst_exact_pose_coverage_min"]),
        "mean_exact_relative_pose_precision": observed[
            "mean_exact_relative_pose_precision"
        ]
        >= float(DECISION_RULE["mean_exact_relative_pose_precision_min"]),
        "worst_exact_relative_pose_precision": observed[
            "worst_exact_relative_pose_precision"
        ]
        >= float(DECISION_RULE["worst_exact_relative_pose_precision_min"]),
        "mean_accepted_relation_precision": observed[
            "mean_accepted_relation_precision"
        ]
        >= float(DECISION_RULE["mean_accepted_relation_precision_min"]),
        "worst_accepted_relation_precision": observed[
            "worst_accepted_relation_precision"
        ]
        >= float(DECISION_RULE["worst_accepted_relation_precision_min"]),
        "mean_accepted_cross_seam_precision": observed[
            "mean_accepted_cross_seam_precision"
        ]
        >= float(DECISION_RULE["mean_accepted_cross_seam_precision_min"]),
        "worst_accepted_cross_seam_precision": observed[
            "worst_accepted_cross_seam_precision"
        ]
        >= float(DECISION_RULE["worst_accepted_cross_seam_precision_min"]),
        "mean_component_cycle_rank_ratio": observed[
            "mean_component_cycle_rank_ratio"
        ]
        >= float(DECISION_RULE["mean_component_cycle_rank_ratio_min"]),
    }
    passed = all(checks.values())
    return {
        "status": (
            "go_E21_one_cluster_absolute_origin_residual"
            if passed
            else "kill_top8_triangle_potential_route"
        ),
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_score_structure_oracle_not_deployable",
    }


def _validate_complete_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_digest: str,
    e12_report: Mapping[str, Any],
    scenes: Sequence[e12.RawScene],
    clean_records: Mapping[int, Mapping[str, Any]],
) -> None:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != E20_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E20_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E20ContractError("existing E20 complete report contract drifted")
    _finite(
        report.get("runtime_seconds"),
        label="existing E20 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("completed_images") != list(
        e12.CALIBRATION_IDS
    ):
        raise E20ContractError("existing E20 rows/completion IDs drifted")
    by_image = {
        int(row.get("image", -1)): row for row in rows if isinstance(row, Mapping)
    }
    if (
        len(rows) != len(e12.CALIBRATION_IDS)
        or len(by_image) != len(rows)
        or tuple(sorted(by_image)) != e12.CALIBRATION_IDS
    ):
        raise E20ContractError("existing E20 rows are incomplete or duplicated")
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    for image in e12.CALIBRATION_IDS:
        scene = scene_by_image[image]
        cache = e14._load_cc_cache(scene, e12_report, clean_records[image])
        right, down = e12.dense_from_graph(cache.cc_candidates, cache.cc_scores)
        _validate_success_row(
            by_image[image],
            scene=scene,
            cache_sha256=cache.sha256,
            right=right,
            down=down,
        )
    computed_summary = summarize(rows)
    computed_decision = decision(computed_summary)
    if report.get("summary") != computed_summary:
        raise E20ContractError("existing E20 summary drifted")
    if report.get("decision") != computed_decision:
        raise E20ContractError("existing E20 decision drifted")
    if report.get("stage") != computed_decision["status"]:
        raise E20ContractError("existing E20 terminal stage drifted")


def run_gate(paths: E20Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E20 report")
    if report_path.suffix.lower() != ".json":
        raise E20ContractError("E20 report must be a .json file")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e19_report_path = _require_e_drive(paths.e19_report, label="E19 report")
    calibration_path = paths.calibration_report.resolve()
    if report_path in {e12_report_path, e19_report_path, calibration_path}:
        raise E20ContractError("E20 report must not overwrite an input")
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(
        clean_cache_dir
    ):
        raise E20ContractError("E20 report must not be written inside an input cache")

    e19_report = _verify_e19_cap_kill(e19_report_path)
    try:
        e12_report, _calibration, scenes = e17._load_verified_structure_inputs(
            raw_cache_dir,
            calibration_path,
            e12_report_path,
        )
        clean_records = e14._clean_cache_records(e12_report)
    except (e17.E17ContractError, e14.E14ContractError) as exc:
        raise E20ContractError(str(exc)) from exc
    contract = {
        "protocol_sha256": e12.canonical_digest(E20_PROTOCOL),
        "e19_report": {
            "path": str(e19_report_path),
            "sha256": EXPECTED_E19_REPORT_SHA256,
            "run_contract_sha256": str(e19_report["run_contract_sha256"]),
        },
        "e12_report": {
            "path": str(e12_report_path),
            "sha256": EXPECTED_E12_REPORT_SHA256,
        },
        "calibration_report": {
            "path": str(calibration_path),
            "sha256": e12.CALIBRATION_REPORT_SHA256,
        },
        "raw_cache_dir": str(raw_cache_dir),
        "report": str(report_path),
        "scene_provenance_digest": str(e12_report["scene_provenance_digest"]),
        "clean_score_caches": {
            str(image): {
                "path": str(Path(str(record["path"])).resolve()),
                "sha256": str(record["sha256"]),
            }
            for image, record in sorted(clean_records.items())
        },
        "source_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }
    contract_digest = e12.canonical_digest(contract)
    if report_path.is_file():
        existing = _load_json(report_path, label="existing E20 report")
        if existing.get("run_contract_sha256") != contract_digest:
            raise E20ContractError("existing E20 report belongs to different bytes")
        if existing.get("run_contract") != contract:
            raise E20ContractError("existing E20 report contract payload drifted")
        if existing.get("status") == "complete":
            _validate_complete_report(
                existing,
                contract=contract,
                contract_digest=contract_digest,
                e12_report=e12_report,
                scenes=scenes,
                clean_records=clean_records,
            )
            return existing

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "triangle_pose_structure",
        "protocol": E20_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E20_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rows": [],
        "completed_images": [],
        "decision": {"status": "not_run"},
    }
    _atomic_write_json(report_path, output)
    try:
        for scene in scenes:
            cache = e14._load_cc_cache(
                scene, e12_report, clean_records[scene.image_id]
            )
            right, down = e12.dense_from_graph(
                cache.cc_candidates, cache.cc_scores
            )
            result = pose.run_triangle_potential_dsu(right, down)
            graph = e18_core.build_graph_data(right, down)
            row = evaluate_structure(
                scene,
                result,
                clean_score_cache_sha256=cache.sha256,
                graph=graph,
            )
            _validate_success_row(
                row,
                scene=scene,
                cache_sha256=cache.sha256,
                right=right,
                down=down,
                expected_result=result,
            )
            output["rows"].append(row)
            output["completed_images"].append(int(scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
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
        description="Run fixed CPU-only E20 triangle-potential viability gate."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e19-report", type=Path, default=DEFAULT_E19_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gate(
        E20Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e19_report=args.e19_report,
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
