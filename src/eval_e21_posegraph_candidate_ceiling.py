"""Frozen E21 raw-candidate oracle ceiling for a learned pose-graph verifier.

The production candidate core receives only the two raw Rank96 dense score
matrices.  Ground-truth permutation data is consulted strictly after that core
returns, and only to label whole-component purity and exact signed component
relations.  The evaluator unions every oracle-true relation, validates the
relative geometry independently, and reports the largest exact connected
cluster.  It never constructs an absolute 24x24 solution.
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

import e21_posegraph_candidate_oracle as pose
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
import eval_e20_triangle_potential_viability as e20_eval


class E21ContractError(RuntimeError):
    """The frozen E21 protocol, byte lineage, or oracle algebra drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e21-raw-cc96-anchor-candidate-ceiling-report-v1"
EXPERIMENT = "e21_raw_cc96_anchor_top8_pose_candidate_ceiling_v1"

EXPECTED_E12_REPORT_SHA256 = e14.EXPECTED_E12_REPORT_SHA256
EXPECTED_E20_REPORT_SHA256 = (
    "4538e35825bdfae86aa7bda252d7a7a5aa2b8e933ffc6deaab74ebade8f557be"
)
EXPECTED_E20_RUN_CONTRACT_SHA256 = (
    "5473fddb78c24923f277fd4ab8ae3753b87b14c453405d5ad35297314b70abe5"
)
EXPECTED_E20_PROTOCOL_SHA256 = (
    "78c4f44a5c0be496b3dbe789779340e49d24318a9d2a2f7502bc18c4360fc4d5"
)
EXPECTED_E20_STAGE = "kill_top8_triangle_potential_route"
EXPECTED_RUNTIME_PROVENANCE = dict(e20_eval.EXPECTED_RUNTIME_PROVENANCE)

MAX_HYPOTHESES = 6000
DECISION_RULE: dict[str, float | int] = {
    "completed_scenes": 8,
    "max_hypotheses_each": MAX_HYPOTHESES,
    "true_relation_scenes": 8,
    "legal_origin_scenes": 8,
    "mean_exact_connected_coverage_min": 0.30,
    "worst_exact_connected_coverage_min": 0.20,
}

E21_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e21-raw-cc96-anchor-top8-candidate-ceiling-v1",
    "role": "label_only_raw_rank96_candidate_ceiling_no_board",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "authorization": {
        "e20_report_sha256": EXPECTED_E20_REPORT_SHA256,
        "e20_run_contract_sha256": EXPECTED_E20_RUN_CONTRACT_SHA256,
        "e20_protocol_sha256": EXPECTED_E20_PROTOCOL_SHA256,
        "required_status": "complete",
        "required_stage": EXPECTED_E20_STAGE,
    },
    "input": {
        "source": "exact_byte_pinned_E12_raw_Rank96_scenes",
        "candidate_ids": "raw_only",
        "scores": "raw_ranker_only",
        "dense_conversion": "frozen_E12_CPU_float32",
        "clean_score_input": False,
    },
    "components": {
        "builder": "corrected_exact_buddies",
        "max_edges": 96,
        "min_margin": 0.0,
        "partition_includes_singletons": True,
        "normalised_deterministic": True,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
    },
    "claims": {
        "emitters": "tiles_in_nontrivial_CC96_components_only",
        "directions": ["U", "D", "L", "R"],
        "positive_dense_top_k": 8,
        "rank": ["score_desc", "tile_id_asc"],
        "rank_before_component_filter": True,
        "target": "any_different_component_including_singleton",
        "iterative_growth": False,
    },
    "hypotheses": {
        "key": "u_lt_v_plus_exact_signed_offset",
        "all_pair_offsets_retained": True,
        "physical_seams": "canonical_deduplicated",
        "reciprocal_same_physical_seam": "metadata_not_second_relation",
        "triangle_filter": False,
    },
    "oracle": {
        "labels_available_to_core": False,
        "component_purity": (
            "whole_component_one_exact_truth_coordinate_minus_local_coordinate"
        ),
        "relation_truth": (
            "both_components_wholly_pure_and_signed_translation_delta_exact"
        ),
        "union": "all_true_hypotheses_once_in_canonical_relation_order",
        "geometry": "exact_signed_translation_no_collision_bbox_each_at_most_24",
        "consistent_connected_relation": "cycle_evidence",
        "pure_components_without_true_relations": "eligible_singleton_clusters",
    },
    "output": {
        "selection_rank": [
            "exact_connected_tiles_desc",
            "accepted_relations_desc",
            "cycle_rank_desc",
            "minimum_tile_asc",
            "canonical_translations_asc",
        ],
        "cluster_normalisation": "subtract_minimum_occupied_row_and_column",
        "legal_origins": "analytic_after_selection",
        "absolute_origin_selection": False,
        "absolute_board": False,
    },
    "decision": dict(DECISION_RULE),
    "routing": {
        "pass": "open_separately_frozen_E22_factor_graph_relation_verifier_pilot",
        "fail": "close_exact_raw_CC96_anchor_top8_pool_before_training",
    },
    "excluded": [
        "clean_score_input",
        "learned_relation_logits",
        "board",
        "residual_completion",
        "placement",
        "neighbour",
        "SSIM",
        "NLM",
        "absolute_origin_choice",
        "iterative_oracle_growth",
        "pool_topk_component_sweep",
        "rotation",
        "reflection",
        "GPU",
        "diffusion",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_E20_REPORT = Path(
    "E:/pazzle_work/triangle_pose_e20/cc192_triangle_potential_viability_v1.json"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/posegraph_e21/cc96_top8_anchor_candidate_ceiling_v1.json"
)


@dataclass(frozen=True)
class E21Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e20_report: Path
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
        raise E21ContractError(f"{label} is not an integer")
    return int(value)


def _finite(
    value: object, *, label: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise E21ContractError(f"{label} is not numeric")
    observed = float(value)
    if not math.isfinite(observed):
        raise E21ContractError(f"{label} is not finite")
    if not minimum <= observed <= maximum:
        raise E21ContractError(f"{label} is outside [{minimum}, {maximum}]")
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
        raise E21ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E21ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E21ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E21 report")
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
        raise E21ContractError(
            f"E21 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e21_posegraph_candidate_oracle.py": source
        / "e21_posegraph_candidate_oracle.py",
        "eval_buddies_ssim_budget.py": source / "eval_buddies_ssim_budget.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_e20_triangle_potential_viability.py": source
        / "eval_e20_triangle_potential_viability.py",
        "eval_e21_posegraph_candidate_ceiling.py": Path(__file__).resolve(),
        "eval_seeded_qap.py": source / "eval_seeded_qap.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise E21ContractError("E21 source file is missing: " + ", ".join(missing))
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _verify_e20_kill(path: Path) -> Mapping[str, Any]:
    resolved = _require_e_drive(path, label="E20 report")
    if not resolved.is_file():
        raise E21ContractError(f"E20 report is missing: {resolved}")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E20_REPORT_SHA256:
        raise E21ContractError(
            "E20 report SHA256 mismatch: "
            f"expected {EXPECTED_E20_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E20 report")
    contract = report.get("run_contract")
    rows = report.get("rows")
    if (
        _integer(report.get("schema_version"), label="E20 schema version")
        != e20_eval.SCHEMA_VERSION
        or report.get("schema") != e20_eval.REPORT_SCHEMA
        or report.get("experiment") != e20_eval.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("stage") != EXPECTED_E20_STAGE
        or report.get("protocol") != e20_eval.E20_PROTOCOL
        or report.get("protocol_sha256") != EXPECTED_E20_PROTOCOL_SHA256
        or report.get("run_contract_sha256")
        != EXPECTED_E20_RUN_CONTRACT_SHA256
        or report.get("completed_images") != list(e12.CALIBRATION_IDS)
        or not isinstance(contract, Mapping)
        or not isinstance(rows, list)
        or len(rows) != len(e12.CALIBRATION_IDS)
    ):
        raise E21ContractError("E20 authorization contract drifted")
    _finite(
        report.get("runtime_seconds"),
        label="E20 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    try:
        expected_summary = e20_eval.summarize(rows)
        expected_decision = e20_eval.decision(expected_summary)
    except Exception as exc:
        raise E21ContractError(f"E20 terminal payload is malformed: {exc}") from exc
    if (
        report.get("summary") != expected_summary
        or report.get("decision") != expected_decision
        or expected_decision.get("passed") is not False
        or expected_decision.get("status") != EXPECTED_E20_STAGE
    ):
        raise E21ContractError("E20 KILL decision drifted")
    frozen_sources = contract.get("source_provenance")
    frozen_runtime = contract.get("runtime_provenance")
    if not isinstance(frozen_sources, Mapping):
        raise E21ContractError("E20 source provenance is malformed")
    source = Path(__file__).resolve().parent
    for name, expected in frozen_sources.items():
        if not isinstance(name, str) or not _is_sha256(expected):
            raise E21ContractError("E20 source provenance entry is malformed")
        observed = e12.sha256_file(source / name)
        if observed != expected:
            raise E21ContractError(
                f"source shared with E20 drifted for {name}: expected {expected}, got {observed}"
            )
    if frozen_runtime != _runtime_provenance():
        raise E21ContractError("E20-to-E21 runtime provenance drifted")
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
            raise E21ContractError("core payload contains a non-finite float")
        return value
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    raise E21ContractError(f"core payload contains unsupported type {type(value)}")


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
        "clean_score",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            # Claim diagnostics such as ``singleton_target_claims`` are legal
            # label-free inventory.  Reject semantic label/pixel payload keys,
            # not the ordinary word "target" inside candidate accounting.
            if lowered in forbidden_exact:
                raise E21ContractError(f"core payload contains forbidden key {key}")
            _check_forbidden_payload_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_forbidden_payload_keys(item)


def _strict_vector(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (e12.NFRAG,) or array.dtype.kind not in "iu":
        raise E21ContractError(f"{label} must be an integer ({e12.NFRAG},) vector")
    return np.ascontiguousarray(array.astype(np.int64, copy=False))


def _validate_permutation(value: object) -> np.ndarray:
    permutation = _strict_vector(value, label="permutation")
    if not np.array_equal(
        np.sort(permutation), np.arange(e12.NFRAG, dtype=np.int64)
    ):
        raise E21ContractError("permutation is not an input-tile to cell bijection")
    return permutation


def _component_entries(component: pose.RigidComponent) -> tuple[tuple[int, int, int], ...]:
    raw = getattr(component, "entries", None)
    if not isinstance(raw, (tuple, list)) or not raw:
        raise E21ContractError("component entries are missing")
    entries: list[tuple[int, int, int]] = []
    for entry in raw:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise E21ContractError("component contains a malformed entry")
        entries.append(
            tuple(_integer(item, label="component entry") for item in entry)
        )
    return tuple(entries)


def validate_candidate_pool(result: pose.CandidatePoolResult) -> None:
    if not isinstance(result, pose.CandidatePoolResult):
        raise E21ContractError("candidate core returned the wrong result type")
    components = result.components
    if not isinstance(components, tuple) or not components:
        raise E21ContractError("candidate component partition is malformed")
    owner = _strict_vector(result.owner, label="component owner")
    local_rows = _strict_vector(result.local_rows, label="component local rows")
    local_cols = _strict_vector(result.local_cols, label="component local cols")
    seen_tiles: set[int] = set()
    observed_order: list[tuple[int, int, tuple[tuple[int, int, int], ...]]] = []
    expected_nontrivial: list[int] = []
    for expected_id, component in enumerate(components):
        component_id = _integer(
            getattr(component, "component_id", None), label="component ID"
        )
        if component_id != expected_id:
            raise E21ContractError("component IDs are not dense deterministic IDs")
        entries = _component_entries(component)
        if entries != tuple(sorted(entries)):
            raise E21ContractError("component entries are not canonical")
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
            raise E21ContractError("component geometry is invalid or not normalized")
        if seen_tiles.intersection(tiles):
            raise E21ContractError("component partition duplicates a tile")
        seen_tiles.update(tiles)
        for tile, row, col in entries:
            if (
                int(owner[tile]) != component_id
                or int(local_rows[tile]) != row
                or int(local_cols[tile]) != col
            ):
                raise E21ContractError("owner/local coordinate arrays drifted")
        observed_order.append((-len(entries), min(tiles), entries))
        if len(entries) >= 2:
            expected_nontrivial.append(component_id)
    if seen_tiles != set(range(e12.NFRAG)):
        raise E21ContractError("component partition does not cover all 576 tiles")
    if observed_order != sorted(observed_order):
        raise E21ContractError("component ordering drifted")
    nontrivial = result.nontrivial_component_ids
    if not isinstance(nontrivial, frozenset) or frozenset(expected_nontrivial) != nontrivial:
        raise E21ContractError("nontrivial component IDs drifted")

    claims = result.claims
    if not isinstance(claims, tuple):
        raise E21ContractError("candidate claims are malformed")
    claim_identities: set[tuple[int, int, int, int]] = set()
    directions = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    for expected_id, claim in enumerate(claims):
        claim_id = _integer(getattr(claim, "claim_id", None), label="claim ID")
        anchor = _integer(getattr(claim, "anchor", None), label="claim anchor")
        target = _integer(getattr(claim, "target", None), label="claim target")
        dy = _integer(getattr(claim, "dy", None), label="claim dy")
        dx = _integer(getattr(claim, "dx", None), label="claim dx")
        anchor_component = _integer(
            getattr(claim, "anchor_component", None), label="claim anchor component"
        )
        target_component = _integer(
            getattr(claim, "target_component", None), label="claim target component"
        )
        score = _finite(
            getattr(claim, "score", None),
            label="claim score",
            minimum=0.0,
            maximum=float("inf"),
        )
        identity = (anchor, target, dy, dx)
        if (
            claim_id != expected_id
            or score <= 0.0
            or not 0 <= anchor < e12.NFRAG
            or not 0 <= target < e12.NFRAG
            or anchor == target
            or (dy, dx) not in directions
            or anchor_component != int(owner[anchor])
            or target_component != int(owner[target])
            or anchor_component not in expected_nontrivial
            or anchor_component == target_component
            or identity in claim_identities
        ):
            raise E21ContractError("candidate claim algebra drifted")
        claim_identities.add(identity)

    hypotheses = result.hypotheses
    if not isinstance(hypotheses, tuple):
        raise E21ContractError("pose hypotheses are malformed")
    relations: list[tuple[int, int, int, int]] = []
    for expected_id, hypothesis in enumerate(hypotheses):
        hypothesis_id = _integer(
            getattr(hypothesis, "hypothesis_id", None), label="hypothesis ID"
        )
        relation = getattr(hypothesis, "relation", None)
        if not isinstance(relation, (tuple, list)) or len(relation) != 4:
            raise E21ContractError("hypothesis relation is malformed")
        u, v, dr, dc = tuple(
            _integer(item, label="hypothesis relation") for item in relation
        )
        if (
            hypothesis_id != expected_id
            or not 0 <= u < v < len(components)
            or u == v
        ):
            raise E21ContractError("hypothesis key drifted")
        seam_scores = getattr(hypothesis, "seam_scores", None)
        if not isinstance(seam_scores, tuple) or not seam_scores:
            raise E21ContractError("hypothesis seam evidence is malformed")
        seen_seams: set[tuple[int, int, int, int]] = set()
        for seam, score in seam_scores:
            if not isinstance(seam, (tuple, list)) or len(seam) != 4:
                raise E21ContractError("physical seam is malformed")
            first, second, seam_dy, seam_dx = tuple(
                _integer(item, label="physical seam") for item in seam
            )
            canonical = (first, second, seam_dy, seam_dx)
            if (
                not 0 <= first < e12.NFRAG
                or not 0 <= second < e12.NFRAG
                or first == second
                or (seam_dy, seam_dx) not in {(0, 1), (1, 0)}
                or canonical in seen_seams
                or _finite(
                    score,
                    label="physical seam score",
                    minimum=0.0,
                    maximum=float("inf"),
                )
                <= 0.0
            ):
                raise E21ContractError("physical seam evidence drifted")
            seen_seams.add(canonical)
        relations.append((u, v, dr, dc))
    if relations != sorted(relations) or len(set(relations)) != len(relations):
        raise E21ContractError("hypotheses are not unique canonical relations")
    diagnostics = _jsonable(result.diagnostics)
    _check_forbidden_payload_keys(diagnostics)


def _core_payload(result: pose.CandidatePoolResult) -> dict[str, Any]:
    validate_candidate_pool(result)
    components = _jsonable(result.components)
    claims = _jsonable(result.claims)
    hypotheses = _jsonable(result.hypotheses)
    diagnostics = _jsonable(result.diagnostics)
    _check_forbidden_payload_keys(diagnostics)
    return {
        "component_count": len(result.components),
        "nontrivial_component_ids": _jsonable(result.nontrivial_component_ids),
        "claim_count": len(result.claims),
        "hypothesis_count": len(result.hypotheses),
        "components_sha256": e12.canonical_digest(components),
        "owner_sha256": e12.array_sha256(np.asarray(result.owner)),
        "local_rows_sha256": e12.array_sha256(np.asarray(result.local_rows)),
        "local_cols_sha256": e12.array_sha256(np.asarray(result.local_cols)),
        "claims_sha256": e12.canonical_digest(claims),
        "hypotheses_sha256": e12.canonical_digest(hypotheses),
        "diagnostics": diagnostics,
    }


def component_truth_shifts(
    result: pose.CandidatePoolResult, permutation: object
) -> dict[int, tuple[int, int] | None]:
    validate_candidate_pool(result)
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


def true_pose_hypotheses(
    result: pose.CandidatePoolResult,
    shifts: Mapping[int, tuple[int, int] | None],
) -> tuple[pose.PoseHypothesis, ...]:
    output: list[pose.PoseHypothesis] = []
    for hypothesis in result.hypotheses:
        u, v, dr, dc = hypothesis.relation
        left = shifts.get(int(u))
        right = shifts.get(int(v))
        if (
            left is not None
            and right is not None
            and (right[0] - left[0], right[1] - left[1])
            == (int(dr), int(dc))
        ):
            output.append(hypothesis)
    relations = [tuple(map(int, value.relation)) for value in output]
    if relations != sorted(relations) or len(set(relations)) != len(relations):
        raise E21ContractError("oracle-true relations are not canonical and unique")
    return tuple(output)


class _PotentialDSU:
    """Independent signed-translation DSU with merge-time geometry checks."""

    def __init__(
        self,
        components: Sequence[pose.RigidComponent],
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
            raise E21ContractError("relation touches an impure component")
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
                raise E21ContractError("DSU member/root algebra drifted")
            for tile, row, col in _component_entries(self.components[component_id]):
                position = (row + shift[0], col + shift[1])
                if position in positions and positions[position] != tile:
                    raise E21ContractError("oracle union creates a tile collision")
                positions[position] = tile
        if not positions:
            raise E21ContractError("oracle DSU root is empty")
        rows = [row for row, _col in positions]
        cols = [col for _row, col in positions]
        if max(rows) - min(rows) + 1 > 24 or max(cols) - min(cols) + 1 > 24:
            raise E21ContractError("oracle union exceeds the upright 24x24 span")

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
                raise E21ContractError("oracle cycle contradicts signed translations")
            return False

        # T(root_v) - T(root_u), derived independently from T(v)-T(u).
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
        raise E21ContractError("cannot create an empty oracle cluster")
    root, _ = dsu.find(ids[0])
    raw_translations: dict[int, tuple[int, int]] = {}
    raw_entries: list[tuple[int, int, int]] = []
    for component_id in ids:
        observed_root, translation = dsu.find(component_id)
        if observed_root != root:
            raise E21ContractError("oracle cluster is disconnected")
        raw_translations[component_id] = translation
        for tile, row, col in _component_entries(dsu.components[component_id]):
            raw_entries.append((tile, row + translation[0], col + translation[1]))
    if len({tile for tile, _row, _col in raw_entries}) != len(raw_entries):
        raise E21ContractError("oracle cluster duplicates a tile")
    if len({(row, col) for _tile, row, col in raw_entries}) != len(raw_entries):
        raise E21ContractError("oracle cluster contains a collision")
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
        raise E21ContractError("oracle cluster normalization/span drifted")
    accepted = tuple(sorted(tuple(map(int, relation)) for relation in relations))
    if len(set(accepted)) != len(accepted):
        raise E21ContractError("oracle cluster duplicates a relation")
    if any(u not in ids or v not in ids for u, v, _dr, _dc in accepted):
        raise E21ContractError("oracle relation leaves its connected cluster")
    translation_map = {cid: (row, col) for cid, row, col in translations}
    for u, v, dr, dc in accepted:
        observed = (
            translation_map[v][0] - translation_map[u][0],
            translation_map[v][1] - translation_map[u][1],
        )
        if observed != (dr, dc):
            raise E21ContractError("accepted relation contradicts cluster translations")
    cycle_rank = len(accepted) - len(ids) + 1
    if cycle_rank < 0:
        raise E21ContractError("oracle relation graph is not connected")
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
        minimum_tile=min(tile for tile, _row, _col in entries),
        bbox=bbox,
        bbox_height=height,
        bbox_width=width,
        legal_origin_bounds=(0, 23 - bbox[1], 0, 23 - bbox[3]),
        legal_origin_count=(25 - height) * (25 - width),
    )


def select_oracle_cluster(clusters: Sequence[OracleCluster]) -> OracleCluster:
    if not clusters:
        raise E21ContractError("no exact pure component cluster exists")
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
    result: pose.CandidatePoolResult, permutation: object
) -> tuple[
    dict[int, tuple[int, int] | None],
    tuple[pose.PoseHypothesis, ...],
    tuple[OracleCluster, ...],
    OracleCluster,
]:
    validate_candidate_pool(result)
    shifts = component_truth_shifts(result, permutation)
    pure_ids = tuple(sorted(cid for cid, shift in shifts.items() if shift is not None))
    if not pure_ids:
        raise E21ContractError("candidate partition contains no exact pure component")
    true_hypotheses = true_pose_hypotheses(result, shifts)
    dsu = _PotentialDSU(result.components, pure_ids)
    for hypothesis in true_hypotheses:
        u, v, dr, dc = tuple(map(int, hypothesis.relation))
        dsu.union(u, v, dr, dc)

    relations_by_root: dict[int, list[tuple[int, int, int, int]]] = {}
    for hypothesis in true_hypotheses:
        relation = tuple(map(int, hypothesis.relation))
        root_u, _ = dsu.find(relation[0])
        root_v, _ = dsu.find(relation[1])
        if root_u != root_v:
            raise E21ContractError("true relation endpoints remained disconnected")
        relations_by_root.setdefault(root_u, []).append(relation)
    clusters: list[OracleCluster] = []
    roots = sorted({dsu.find(component_id)[0] for component_id in pure_ids})
    for root in roots:
        ids = tuple(sorted(dsu.members[root]))
        clusters.append(_make_cluster(dsu, ids, relations_by_root.get(root, ())))
    clusters.sort(key=lambda cluster: (cluster.minimum_tile, cluster.translations))
    selected = select_oracle_cluster(clusters)
    return shifts, true_hypotheses, tuple(clusters), selected


def _cluster_payload(cluster: OracleCluster) -> dict[str, Any]:
    return _jsonable(asdict(cluster))


def evaluate_scene(
    scene: e12.RawScene,
    result: pose.CandidatePoolResult,
    *,
    right: np.ndarray,
    down: np.ndarray,
) -> dict[str, Any]:
    core = _core_payload(result)
    shifts, true_hypotheses, clusters, selected = build_oracle_ceiling(
        result, scene.permutation
    )
    pure_ids = tuple(sorted(cid for cid, shift in shifts.items() if shift is not None))
    pure_tiles = sum(len(_component_entries(result.components[cid])) for cid in pure_ids)
    true_relations = tuple(
        tuple(map(int, hypothesis.relation)) for hypothesis in true_hypotheses
    )
    oracle = {
        "pure_component_ids": _jsonable(pure_ids),
        "true_hypothesis_ids": [
            int(hypothesis.hypothesis_id) for hypothesis in true_hypotheses
        ],
        "true_relations": _jsonable(true_relations),
        "cluster_count": len(clusters),
        "selected": _cluster_payload(selected),
    }
    metrics = {
        "component_count": len(result.components),
        "nontrivial_component_count": len(result.nontrivial_component_ids),
        "pure_component_count": len(pure_ids),
        "pure_component_tiles": int(pure_tiles),
        "hypothesis_count": len(result.hypotheses),
        "true_hypotheses": len(true_hypotheses),
        "selected_components": len(selected.component_ids),
        "selected_accepted_relations": selected.accepted_relation_count,
        "selected_cycle_rank": selected.cycle_rank,
        "selected_exact_connected_tiles": selected.exact_connected_tiles,
        "selected_exact_connected_coverage": selected.exact_connected_coverage,
        "legal_origin_count": selected.legal_origin_count,
    }
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "raw_cache_sha256": str(scene.cache_sha256),
        "candidate_ids_sha256": e12.array_sha256(scene.candidate_ids),
        "raw_scores_sha256": e12.array_sha256(scene.base_scores),
        "dense_right_sha256": e12.array_sha256(right),
        "dense_down_sha256": e12.array_sha256(down),
        "arm": "E21_raw_CC96_anchor_top8_candidate_ceiling",
        "core": core,
        "core_sha256": e12.canonical_digest(core),
        "oracle": oracle,
        "oracle_sha256": e12.canonical_digest(oracle),
        "metrics": metrics,
    }


def _dense_raw_scene(scene: e12.RawScene) -> tuple[np.ndarray, np.ndarray]:
    try:
        right, down = e12.dense_from_graph(
            scene.candidate_ids,
            np.ascontiguousarray(scene.base_scores, dtype=np.float32),
        )
    except Exception as exc:
        raise E21ContractError(
            f"raw dense conversion failed for image {scene.image_id}: {exc}"
        ) from exc
    for label, value in (("right", right), ("down", down)):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (e12.NFRAG, e12.NFRAG)
            or value.dtype != np.float32
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
            or bool((value < 0.0).any())
            or bool((np.diag(value) != 0.0).any())
        ):
            raise E21ContractError(f"raw dense {label} matrix drifted")
    return right, down


def _validate_success_row(
    row: Mapping[str, Any],
    *,
    scene: e12.RawScene,
    right: np.ndarray,
    down: np.ndarray,
    expected_result: pose.CandidatePoolResult | None = None,
) -> None:
    expected_keys = {
        "image",
        "validation_name",
        "raw_cache_sha256",
        "candidate_ids_sha256",
        "raw_scores_sha256",
        "dense_right_sha256",
        "dense_down_sha256",
        "arm",
        "core",
        "core_sha256",
        "oracle",
        "oracle_sha256",
        "metrics",
    }
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise E21ContractError("E21 row fields drifted")
    for label, observed, expected in (
        ("raw cache", row.get("raw_cache_sha256"), scene.cache_sha256),
        (
            "candidate IDs",
            row.get("candidate_ids_sha256"),
            e12.array_sha256(scene.candidate_ids),
        ),
        ("raw scores", row.get("raw_scores_sha256"), e12.array_sha256(scene.base_scores)),
        ("dense right", row.get("dense_right_sha256"), e12.array_sha256(right)),
        ("dense down", row.get("dense_down_sha256"), e12.array_sha256(down)),
    ):
        if observed != expected or not _is_sha256(observed):
            raise E21ContractError(f"E21 {label} lineage drifted")
    if (
        _integer(row.get("image"), label="row image") != int(scene.image_id)
        or row.get("validation_name") != str(scene.validation_name)
        or row.get("arm") != "E21_raw_CC96_anchor_top8_candidate_ceiling"
    ):
        raise E21ContractError(f"E21 row provenance drifted for image {scene.image_id}")
    result = (
        expected_result
        if expected_result is not None
        else pose.run_posegraph_candidate_oracle(right, down)
    )
    expected = evaluate_scene(scene, result, right=right, down=down)
    if row != expected:
        raise E21ContractError(f"E21 row replay drifted for image {scene.image_id}")
    if row.get("core_sha256") != e12.canonical_digest(row["core"]):
        raise E21ContractError("E21 core hash drifted")
    if row.get("oracle_sha256") != e12.canonical_digest(row["oracle"]):
        raise E21ContractError("E21 oracle hash drifted")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E21ContractError("E21 summary requires exactly eight rows")
    images = [_integer(row.get("image"), label="summary row image") for row in rows]
    if tuple(sorted(images)) != e12.CALIBRATION_IDS or len(set(images)) != len(images):
        raise E21ContractError("E21 summary image IDs drifted")
    metrics: list[Mapping[str, Any]] = []
    for row in rows:
        value = row.get("metrics")
        if not isinstance(value, Mapping):
            raise E21ContractError("E21 row metrics are malformed")
        metrics.append(value)
    hypotheses = [
        _integer(value.get("hypothesis_count"), label="hypothesis count")
        for value in metrics
    ]
    true_counts = [
        _integer(value.get("true_hypotheses"), label="true hypothesis count")
        for value in metrics
    ]
    legal_counts = [
        _integer(value.get("legal_origin_count"), label="legal origin count")
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
    return {
        "images": len(rows),
        "completed_scenes": len(rows),
        "hypotheses_within_cap_scenes": int(
            sum(value <= MAX_HYPOTHESES for value in hypotheses)
        ),
        "max_hypothesis_count": max(hypotheses),
        "true_relation_scenes": int(sum(value >= 1 for value in true_counts)),
        "legal_origin_scenes": int(sum(value >= 1 for value in legal_counts)),
        "mean_exact_connected_tiles": float(np.mean(tiles)),
        "mean_exact_connected_coverage": float(np.mean(coverage)),
        "worst_exact_connected_coverage": float(min(coverage)),
        "total_hypotheses": int(sum(hypotheses)),
        "total_true_hypotheses": int(sum(true_counts)),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "completed_scenes": _integer(
            summary.get("completed_scenes"), label="completed scenes"
        ),
        "hypotheses_within_cap_scenes": _integer(
            summary.get("hypotheses_within_cap_scenes"),
            label="hypotheses-within-cap scenes",
        ),
        "max_hypothesis_count": _integer(
            summary.get("max_hypothesis_count"), label="maximum hypothesis count"
        ),
        "true_relation_scenes": _integer(
            summary.get("true_relation_scenes"), label="true relation scenes"
        ),
        "legal_origin_scenes": _integer(
            summary.get("legal_origin_scenes"), label="legal origin scenes"
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
    }
    checks = {
        "completed_scenes": observed["completed_scenes"]
        == int(DECISION_RULE["completed_scenes"]),
        "hypotheses_within_cap_each": observed["hypotheses_within_cap_scenes"]
        == int(DECISION_RULE["completed_scenes"])
        and observed["max_hypothesis_count"]
        <= int(DECISION_RULE["max_hypotheses_each"]),
        "true_relation_scenes": observed["true_relation_scenes"]
        == int(DECISION_RULE["true_relation_scenes"]),
        "legal_origin_scenes": observed["legal_origin_scenes"]
        == int(DECISION_RULE["legal_origin_scenes"]),
        "mean_exact_connected_coverage": observed[
            "mean_exact_connected_coverage"
        ]
        >= float(DECISION_RULE["mean_exact_connected_coverage_min"]),
        "worst_exact_connected_coverage": observed[
            "worst_exact_connected_coverage"
        ]
        >= float(DECISION_RULE["worst_exact_connected_coverage_min"]),
    }
    passed = all(checks.values())
    return {
        "status": (
            "go_E22_factor_graph_relation_verifier_pilot"
            if passed
            else "kill_raw_CC96_anchor_top8_candidate_pool"
        ),
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "raw_candidate_oracle_ceiling_not_deployable",
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
        raise E21ContractError("existing E21 complete report fields drifted")
    if (
        _integer(report.get("schema_version"), label="E21 schema version")
        != SCHEMA_VERSION
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != E21_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E21_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E21ContractError("existing E21 complete report contract drifted")
    _finite(
        report.get("runtime_seconds"),
        label="existing E21 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("completed_images") != list(
        e12.CALIBRATION_IDS
    ):
        raise E21ContractError("existing E21 rows/completion IDs drifted")
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
        raise E21ContractError("existing E21 rows are incomplete or duplicated")
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    for image in e12.CALIBRATION_IDS:
        scene = scene_by_image[image]
        right, down = _dense_raw_scene(scene)
        _validate_success_row(
            by_image[image], scene=scene, right=right, down=down
        )
    expected_summary = summarize(rows)
    expected_decision = decision(expected_summary)
    if report.get("summary") != expected_summary:
        raise E21ContractError("existing E21 summary drifted")
    if report.get("decision") != expected_decision:
        raise E21ContractError("existing E21 decision drifted")
    if report.get("stage") != expected_decision["status"]:
        raise E21ContractError("existing E21 terminal stage drifted")


def _load_verified_raw_inputs(
    paths: E21Paths,
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
        raise E21ContractError(str(exc)) from exc


def run_gate(paths: E21Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E21 report")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e20_report_path = _require_e_drive(paths.e20_report, label="E20 report")
    calibration_path = paths.calibration_report.resolve()
    if report_path.suffix.lower() != ".json":
        raise E21ContractError("E21 report must be a .json file")
    if report_path in {e12_report_path, e20_report_path, calibration_path}:
        raise E21ContractError("E21 report must not overwrite an input")
    if report_path.is_relative_to(raw_cache_dir):
        raise E21ContractError("E21 report must not be written inside the raw cache")

    e20_report = _verify_e20_kill(e20_report_path)
    e12_report, calibration, scenes = _load_verified_raw_inputs(paths)
    if tuple(int(scene.image_id) for scene in scenes) != e12.CALIBRATION_IDS:
        raise E21ContractError("E21 inputs are not exact E12 scenes 10..17")
    scene_records = [e12.scene_provenance(scene) for scene in scenes]
    contract = {
        "protocol_sha256": e12.canonical_digest(E21_PROTOCOL),
        "e20_report": {
            "path": str(e20_report_path),
            "sha256": EXPECTED_E20_REPORT_SHA256,
            "run_contract_sha256": str(e20_report["run_contract_sha256"]),
            "stage": str(e20_report["stage"]),
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
        "report": str(report_path),
        "source_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }
    # Keep the verified calibration object live in the lineage without placing
    # its potentially broad payload in the report.
    if not isinstance(calibration, Mapping):
        raise E21ContractError("verified calibration payload is malformed")
    contract_digest = e12.canonical_digest(contract)
    if report_path.is_file():
        existing = _load_json(report_path, label="existing E21 report")
        if existing.get("run_contract_sha256") != contract_digest:
            raise E21ContractError("existing E21 report belongs to different bytes")
        if existing.get("run_contract") != contract:
            raise E21ContractError("existing E21 report contract payload drifted")
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
        "stage": "raw_candidate_oracle_ceiling",
        "protocol": E21_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E21_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rows": [],
        "completed_images": [],
        "decision": {"status": "not_run"},
    }
    _atomic_write_json(report_path, output)
    try:
        for scene in scenes:
            right, down = _dense_raw_scene(scene)
            result = pose.run_posegraph_candidate_oracle(right, down)
            row = evaluate_scene(scene, result, right=right, down=down)
            _validate_success_row(
                row,
                scene=scene,
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
        description="Run fixed CPU-only E21 raw candidate oracle ceiling."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e20-report", type=Path, default=DEFAULT_E20_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gate(
        E21Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e20_report=args.e20_report,
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
