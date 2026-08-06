"""E17 fail-fast viability gate for rigid CC192 single-edge islands.

The diagnostic uses the byte-pinned E12 clean-score graph and labels only to
measure directional claim precision and whole-component geometric purity.  It
constructs no candidate board and calls neither a solver nor a restorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import skimage

import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
from solve_buddies import _candidate_edges, build_buddies_components


class E17ContractError(RuntimeError):
    """The frozen E17 protocol, input bytes, or runtime drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e17-cc192-rigid-viability-report-v1"
EXPERIMENT = "e17_cc192_single_edge_rigid_island_viability_v1"
EXPECTED_RUNTIME_PROVENANCE = dict(e14.EXPECTED_RUNTIME_PROVENANCE)
FULL_PREFIX = 192
BASE_PREFIX = 96

DECISION_RULE: dict[str, float | int] = {
    "selected_claims_each": 192,
    "mean_full_prefix_precision_min": 0.95,
    "mean_incremental96_precision_min": 0.90,
    "worst_incremental96_precision_min": 0.80,
    "mean_exact_pure_rigid_tile_coverage_min": 0.35,
    "worst_exact_pure_rigid_tile_coverage_min": 0.25,
    "mean_largest_exact_pure_component_size_min": 8.0,
}

E17_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e17-cc192-rigid-island-viability-v1",
    "role": "target_derived_clean_score_structure_gate_no_candidate_board",
    "input_validation": (
        "byte_pinned_E12_reports_scenes_checkpoints_and_clean_score_caches_"
        "without_RR_metric_rows"
    ),
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "graph": {
        "source": "exact_byte_pinned_E12_clean_candidates_and_scores",
        "selector": "solve_buddies._candidate_edges",
        "builder": "solve_buddies.build_buddies_components",
        "full_prefix": FULL_PREFIX,
        "base_prefix": BASE_PREFIX,
        "incremental_indices": [BASE_PREFIX, FULL_PREFIX - 1],
        "min_margin": 0.0,
    },
    "purity": {
        "components": "nontrivial_CC192_builder_components_only",
        "definition": "all_tiles_share_one_truth_coordinate_minus_local_coordinate_translation",
        "modal_trim": False,
        "oracle_edge_removal": False,
    },
    "geometry": {
        "grid": 24,
        "rotation": False,
        "reflection": False,
        "integer_translation_only": True,
    },
    "decision": dict(DECISION_RULE),
    "excluded": [
        "candidate_board",
        "solver",
        "NLM",
        "SSIM",
        "budget_sweep",
        "threshold_sweep",
        "rotation",
        "reflection",
        "GPU",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_REPORT = Path(
    "E:/pazzle_work/single_edge_frame_e17/cc192_rigid_viability_v1.json"
)


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E17ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E17ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E17ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E17 report")
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
    }
    if observed != EXPECTED_RUNTIME_PROVENANCE:
        raise E17ContractError(
            f"E17 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "eval_e17_cc192_rigid_viability.py": Path(__file__).resolve(),
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _validate_permutation(permutation: np.ndarray) -> np.ndarray:
    value = np.asarray(permutation)
    if value.shape != (e12.NFRAG,) or value.dtype.kind not in "iu":
        raise E17ContractError("permutation geometry/dtype drifted")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(e12.NFRAG)):
        raise E17ContractError("permutation is not a bijection")
    return value


def _validate_structure_calibration_payload(payload: Mapping[str, Any]) -> None:
    """Validate only immutable data/provenance fields, never calibration metrics."""

    if int(payload.get("schema_version", -1)) != 1:
        raise E17ContractError("calibration report schema_version drifted")
    if payload.get("experiment") != "raw_buddies_solve_ssim_budget":
        raise E17ContractError("calibration report experiment drifted")
    if payload.get("phase") != "calibration" or payload.get("status") != "frozen":
        raise E17ContractError("calibration report is not frozen")
    if payload.get("calibration_ids") != list(e12.CALIBRATION_IDS):
        raise E17ContractError("calibration IDs are not exactly 10..17")
    if payload.get("confirmation_ids_reserved") != [18, 19, 20, 21]:
        raise E17ContractError("calibration confirmation reservation drifted")
    if payload.get("contract") != e12.CALIBRATION_CONTRACT:
        raise E17ContractError("calibration replay contract drifted")
    if int(payload.get("selected_budget", -1)) != e12.MAX_EDGES:
        raise E17ContractError("calibration did not freeze budget 96")
    if payload.get("scene_provenance_digest") != e12.SCENE_PROVENANCE_DIGEST:
        raise E17ContractError("calibration scene provenance digest drifted")
    provenance = payload.get("scene_provenance")
    if (
        not isinstance(provenance, list)
        or e12.canonical_digest(provenance) != e12.SCENE_PROVENANCE_DIGEST
    ):
        raise E17ContractError("calibration scene provenance payload drifted")
    if len(provenance) != len(e12.CALIBRATION_IDS) or not all(
        isinstance(row, Mapping) for row in provenance
    ):
        raise E17ContractError("calibration scene provenance rows are malformed")
    if [int(row.get("image", -1)) for row in provenance] != list(
        e12.CALIBRATION_IDS
    ):
        raise E17ContractError("calibration scene provenance IDs drifted")
    if [str(row.get("validation_name", "")) for row in provenance] != list(
        e12.CALIBRATION_NAMES
    ):
        raise E17ContractError("calibration validation names drifted")


def _load_structure_calibration_report(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise E17ContractError(f"calibration report is missing: {resolved}")
    digest = e12.sha256_file(resolved)
    if digest != e12.CALIBRATION_REPORT_SHA256:
        raise E17ContractError(
            "calibration report SHA256 mismatch: "
            f"expected {e12.CALIBRATION_REPORT_SHA256}, got {digest}"
        )
    payload = _load_json(resolved, label="calibration report")
    _validate_structure_calibration_payload(payload)
    return payload


def _load_verified_structure_inputs(
    raw_cache_dir: Path,
    calibration_report: Path,
    e12_report_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[e12.RawScene]]:
    """Replay byte provenance needed by E17 without touching RR metric rows."""

    raw_dir = _require_e_drive(raw_cache_dir, label="raw score cache")
    e12_path = _require_e_drive(e12_report_path, label="E12 report")
    digest = e12.sha256_file(e12_path)
    if digest != e14.EXPECTED_E12_REPORT_SHA256:
        raise E17ContractError(
            "E12 report SHA256 mismatch: "
            f"expected {e14.EXPECTED_E12_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(e12_path, label="E12 report")
    if (
        report.get("schema") != e12.REPORT_SCHEMA
        or report.get("experiment") != e12.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != e12.ORACLE_PROTOCOL
        or report.get("protocol_sha256")
        != e12.canonical_digest(e12.ORACLE_PROTOCOL)
    ):
        raise E17ContractError("E12 report protocol/status drifted")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise E17ContractError("E12 report inputs are malformed")
    calibration_path = calibration_report.resolve()
    if Path(str(inputs.get("cache_dir", ""))).resolve() != raw_dir:
        raise E17ContractError("requested raw score cache differs from E12")
    if Path(str(inputs.get("calibration_report", ""))).resolve() != calibration_path:
        raise E17ContractError("requested calibration report differs from E12")

    calibration = _load_structure_calibration_report(calibration_path)
    if report.get("code_provenance") != e12.code_provenance():
        raise E17ContractError("source code used by E12 has drifted")
    if report.get("scoring_code_provenance") != e12.scoring_code_provenance():
        raise E17ContractError("E12 clean score-cache provenance has drifted")
    checkpoint_records = report.get("checkpoints")
    if not isinstance(checkpoint_records, Mapping):
        raise E17ContractError("E12 checkpoint provenance is malformed")
    try:
        e14._verify_checkpoint_records(checkpoint_records)
        scenes = e12.load_raw_scenes(raw_dir, e12.CALIBRATION_IDS)
        observed = e12.validate_scene_replay(scenes, calibration)
    except (e12.OracleContractError, e14.E14ContractError) as exc:
        raise E17ContractError(str(exc)) from exc
    if (
        report.get("scene_provenance") != observed
        or report.get("scene_provenance_digest") != e12.canonical_digest(observed)
    ):
        raise E17ContractError("E12 scene provenance differs from replayed bytes")
    return report, calibration, scenes


def _validate_component(
    component: Mapping[int, tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    if not isinstance(component, Mapping):
        raise E17ContractError("CC192 builder component is not a mapping")
    normalized: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for raw_tile, raw_coord in component.items():
        if isinstance(raw_tile, (bool, np.bool_)) or not isinstance(
            raw_tile, (int, np.integer)
        ):
            raise E17ContractError("CC192 component tile ID is not an integer")
        tile = int(raw_tile)
        if tile < 0 or tile >= e12.NFRAG or tile in normalized:
            raise E17ContractError("CC192 component tile ID is invalid")
        if (
            not isinstance(raw_coord, (tuple, list))
            or len(raw_coord) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in raw_coord
            )
        ):
            raise E17ContractError("CC192 component coordinate is not an integer pair")
        coord = (int(raw_coord[0]), int(raw_coord[1]))
        if coord in occupied:
            raise E17ContractError("CC192 component overlaps a local coordinate")
        normalized[tile] = coord
        occupied.add(coord)
    if normalized:
        rows = [coord[0] for coord in normalized.values()]
        cols = [coord[1] for coord in normalized.values()]
        if max(rows) - min(rows) >= 24 or max(cols) - min(cols) >= 24:
            raise E17ContractError("CC192 component exceeds the 24x24 frame span")
    return normalized


def _validate_selected_edges(
    edges: Sequence[Sequence[float | int]],
) -> list[tuple[float, float, int, int, int, int]]:
    if len(edges) != FULL_PREFIX:
        raise E17ContractError(
            f"CC192 selector returned {len(edges)} claims instead of {FULL_PREFIX}"
        )
    normalized: list[tuple[float, float, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 6:
            raise E17ContractError("CC192 selector returned a malformed claim")
        score = float(edge[0])
        margin = float(edge[1])
        raw_anchor, raw_target, raw_dy, raw_dx = edge[2:]
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (raw_anchor, raw_target, raw_dy, raw_dx)
        ):
            raise E17ContractError("CC192 claim geometry is not integral")
        anchor, target = int(raw_anchor), int(raw_target)
        dy, dx = int(raw_dy), int(raw_dx)
        if not math.isfinite(score) or not math.isfinite(margin) or margin < 0.0:
            raise E17ContractError("CC192 claim score or margin is invalid")
        if (
            anchor < 0
            or anchor >= e12.NFRAG
            or target < 0
            or target >= e12.NFRAG
            or anchor == target
        ):
            raise E17ContractError("CC192 claim tile IDs are invalid")
        if (dy, dx) not in {(0, 1), (1, 0)}:
            raise E17ContractError("CC192 claim direction drifted")
        key = (anchor, target, dy, dx)
        if key in seen:
            raise E17ContractError("CC192 selector returned a duplicate claim")
        seen.add(key)
        normalized.append((score, margin, anchor, target, dy, dx))
    return normalized


def _edge_is_true(
    edge: Sequence[float | int], permutation: np.ndarray
) -> bool:
    _score, _margin, anchor, target, dy, dx = edge
    anchor_row, anchor_col = divmod(int(permutation[int(anchor)]), 24)
    target_row, target_col = divmod(int(permutation[int(target)]), 24)
    return (target_row - anchor_row, target_col - anchor_col) == (
        int(dy),
        int(dx),
    )


def component_is_exactly_pure(
    component: Mapping[int, tuple[int, int]], permutation: np.ndarray
) -> bool:
    """Return true only when the entire rigid island matches one truth shift."""

    value = _validate_permutation(permutation)
    normalized = _validate_component(component)
    if len(normalized) < 2:
        return False
    offsets = {
        (
            int(value[int(tile)] // 24) - int(local_row),
            int(value[int(tile)] % 24) - int(local_col),
        )
        for tile, (local_row, local_col) in normalized.items()
    }
    return len(offsets) == 1


def measure_structure(
    right: np.ndarray,
    down: np.ndarray,
    permutation: np.ndarray,
) -> dict[str, Any]:
    r = np.ascontiguousarray(right, dtype=np.float32)
    d = np.ascontiguousarray(down, dtype=np.float32)
    if r.shape != (e12.NFRAG, e12.NFRAG) or d.shape != r.shape:
        raise E17ContractError("dense score geometry drifted")
    if not np.isfinite(r).all() or not np.isfinite(d).all():
        raise E17ContractError("dense scores contain non-finite values")
    if bool((r < 0.0).any()) or bool((d < 0.0).any()):
        raise E17ContractError("dense scores violate the nonnegative probability contract")
    if bool((np.diag(r) != 0.0).any()) or bool((np.diag(d) != 0.0).any()):
        raise E17ContractError("dense score diagonals must be exactly zero")
    value = _validate_permutation(permutation)
    edges = _validate_selected_edges(
        _candidate_edges(r, d, max_edges=FULL_PREFIX, min_margin=0.0)
    )
    full_true = sum(_edge_is_true(edge, value) for edge in edges)
    incremental = edges[BASE_PREFIX:FULL_PREFIX]
    incremental_true = sum(_edge_is_true(edge, value) for edge in incremental)
    components = build_buddies_components(
        r, d, max_edges=FULL_PREFIX, min_margin=0.0
    )
    if not isinstance(components, Sequence) or not components:
        raise E17ContractError("CC192 builder returned no component sequence")
    normalized_components = [_validate_component(component) for component in components]
    if any(len(component) < 2 for component in normalized_components):
        raise E17ContractError("CC192 builder returned a singleton component")
    component_tiles = {
        tile for component in normalized_components for tile in component
    }
    if len(component_tiles) != sum(
        len(component) for component in normalized_components
    ):
        raise E17ContractError("CC192 components overlap in tile identity")
    pure_sizes = sorted(
        (
            len(component)
            for component in normalized_components
            if component_is_exactly_pure(component, value)
        ),
        reverse=True,
    )
    pure_tiles = int(sum(pure_sizes))
    return {
        "selected_claims": len(edges),
        "true_full_prefix_claims": int(full_true),
        "full_prefix_precision": float(full_true / FULL_PREFIX),
        "incremental_claims": len(incremental),
        "true_incremental_claims": int(incremental_true),
        "incremental96_precision": float(incremental_true / len(incremental)),
        "component_count": len(normalized_components),
        "component_tiles": len(component_tiles),
        "component_coverage": float(len(component_tiles) / e12.NFRAG),
        "exact_pure_component_count": len(pure_sizes),
        "exact_pure_rigid_tiles": pure_tiles,
        "exact_pure_rigid_tile_coverage": float(pure_tiles / e12.NFRAG),
        "largest_exact_pure_component_size": int(max(pure_sizes, default=0)),
        "exact_pure_component_sizes": pure_sizes,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E17ContractError("E17 summary requires exactly eight rows")
    images = tuple(sorted(int(row["image"]) for row in rows))
    if images != e12.CALIBRATION_IDS:
        raise E17ContractError("E17 row image IDs drifted")
    return {
        "images": len(rows),
        "selected_claims_each": sorted({int(row["selected_claims"]) for row in rows}),
        "incremental_claims_each": sorted(
            {int(row["incremental_claims"]) for row in rows}
        ),
        "mean_full_prefix_precision": float(
            np.mean([float(row["full_prefix_precision"]) for row in rows])
        ),
        "mean_incremental96_precision": float(
            np.mean([float(row["incremental96_precision"]) for row in rows])
        ),
        "worst_incremental96_precision": float(
            min(float(row["incremental96_precision"]) for row in rows)
        ),
        "mean_exact_pure_rigid_tile_coverage": float(
            np.mean(
                [float(row["exact_pure_rigid_tile_coverage"]) for row in rows]
            )
        ),
        "worst_exact_pure_rigid_tile_coverage": float(
            min(float(row["exact_pure_rigid_tile_coverage"]) for row in rows)
        ),
        "mean_largest_exact_pure_component_size": float(
            np.mean(
                [int(row["largest_exact_pure_component_size"]) for row in rows]
            )
        ),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "selected_claims_each": list(summary["selected_claims_each"]),
        "mean_full_prefix_precision": float(summary["mean_full_prefix_precision"]),
        "mean_incremental96_precision": float(summary["mean_incremental96_precision"]),
        "worst_incremental96_precision": float(summary["worst_incremental96_precision"]),
        "mean_exact_pure_rigid_tile_coverage": float(
            summary["mean_exact_pure_rigid_tile_coverage"]
        ),
        "worst_exact_pure_rigid_tile_coverage": float(
            summary["worst_exact_pure_rigid_tile_coverage"]
        ),
        "mean_largest_exact_pure_component_size": float(
            summary["mean_largest_exact_pure_component_size"]
        ),
    }
    checks = {
        "selected_claims_each": observed["selected_claims_each"]
        == [int(DECISION_RULE["selected_claims_each"])],
        "mean_full_prefix_precision": observed["mean_full_prefix_precision"]
        >= float(DECISION_RULE["mean_full_prefix_precision_min"]),
        "mean_incremental96_precision": observed["mean_incremental96_precision"]
        >= float(DECISION_RULE["mean_incremental96_precision_min"]),
        "worst_incremental96_precision": observed["worst_incremental96_precision"]
        >= float(DECISION_RULE["worst_incremental96_precision_min"]),
        "mean_exact_pure_rigid_tile_coverage": observed[
            "mean_exact_pure_rigid_tile_coverage"
        ]
        >= float(DECISION_RULE["mean_exact_pure_rigid_tile_coverage_min"]),
        "worst_exact_pure_rigid_tile_coverage": observed[
            "worst_exact_pure_rigid_tile_coverage"
        ]
        >= float(DECISION_RULE["worst_exact_pure_rigid_tile_coverage_min"]),
        "mean_largest_exact_pure_component_size": observed[
            "mean_largest_exact_pure_component_size"
        ]
        >= float(DECISION_RULE["mean_largest_exact_pure_component_size_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "go_E18_absolute_frame_beam" if passed else "kill_CC192_rigid_frame_route",
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_score_structure_oracle_not_deployable",
    }


def _validate_result_row(
    row: Mapping[str, Any],
    *,
    expected_image: int,
    expected_validation_name: str,
    expected_cache_sha256: str,
) -> None:
    if not isinstance(row, Mapping):
        raise E17ContractError("E17 result row is not an object")
    if int(row.get("image", -1)) != expected_image:
        raise E17ContractError("E17 result row image drifted")
    if str(row.get("validation_name", "")) != expected_validation_name:
        raise E17ContractError(f"E17 validation name drifted for image {expected_image}")
    if str(row.get("clean_score_cache_sha256", "")) != expected_cache_sha256:
        raise E17ContractError(f"E17 cache SHA256 drifted for image {expected_image}")
    if any(
        marker in str(key).lower()
        for key in row
        for marker in ("ssim", "board", "canvas", "nlm", "placement", "neighbour")
    ):
        raise E17ContractError("E17 structure row contains a forbidden board metric")
    try:
        selected = int(row["selected_claims"])
        full_true = int(row["true_full_prefix_claims"])
        incremental = int(row["incremental_claims"])
        incremental_true = int(row["true_incremental_claims"])
        component_count = int(row["component_count"])
        component_tiles = int(row["component_tiles"])
        pure_count = int(row["exact_pure_component_count"])
        pure_tiles = int(row["exact_pure_rigid_tiles"])
        largest = int(row["largest_exact_pure_component_size"])
        pure_sizes_raw = row["exact_pure_component_sizes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise E17ContractError("E17 result row fields are malformed") from exc
    if (
        selected != FULL_PREFIX
        or incremental != FULL_PREFIX - BASE_PREFIX
        or full_true < 0
        or full_true > selected
        or incremental_true < 0
        or incremental_true > incremental
        or full_true - incremental_true < 0
        or full_true - incremental_true > BASE_PREFIX
    ):
        raise E17ContractError("E17 claim counts are inconsistent")
    if float(row.get("full_prefix_precision", float("nan"))) != full_true / selected:
        raise E17ContractError("E17 full-prefix precision is inconsistent")
    if (
        float(row.get("incremental96_precision", float("nan")))
        != incremental_true / incremental
    ):
        raise E17ContractError("E17 incremental precision is inconsistent")
    if (
        component_count < 0
        or component_tiles < 0
        or component_tiles > e12.NFRAG
        or pure_count < 0
        or pure_count > component_count
        or pure_tiles < 0
        or pure_tiles > component_tiles
        or largest < 0
    ):
        raise E17ContractError("E17 component counts are inconsistent")
    if float(row.get("component_coverage", float("nan"))) != component_tiles / e12.NFRAG:
        raise E17ContractError("E17 component coverage is inconsistent")
    if (
        float(row.get("exact_pure_rigid_tile_coverage", float("nan")))
        != pure_tiles / e12.NFRAG
    ):
        raise E17ContractError("E17 exact-pure coverage is inconsistent")
    if not isinstance(pure_sizes_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in pure_sizes_raw
    ):
        raise E17ContractError("E17 exact-pure component sizes are malformed")
    pure_sizes = list(pure_sizes_raw)
    if pure_sizes != sorted(pure_sizes, reverse=True):
        raise E17ContractError("E17 exact-pure component sizes are not sorted")
    if (
        len(pure_sizes) != pure_count
        or sum(pure_sizes) != pure_tiles
        or max(pure_sizes, default=0) != largest
    ):
        raise E17ContractError("E17 exact-pure component summaries are inconsistent")


def _validate_complete_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_digest: str,
    scenes: Sequence[e12.RawScene],
) -> None:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "complete"
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("protocol") != E17_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E17_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E17ContractError("existing E17 complete report contract drifted")
    if "rr_reproducibility" in report:
        raise E17ContractError("existing E17 report contains forbidden RR metrics")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise E17ContractError("existing E17 complete report rows are missing")
    if report.get("completed_images") != list(e12.CALIBRATION_IDS):
        raise E17ContractError("existing E17 completed image list drifted")
    by_image = {
        int(row.get("image", -1)): row for row in rows if isinstance(row, Mapping)
    }
    if tuple(sorted(by_image)) != e12.CALIBRATION_IDS or len(by_image) != len(rows):
        raise E17ContractError("existing E17 row IDs are incomplete or duplicated")
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    if tuple(sorted(scene_by_image)) != e12.CALIBRATION_IDS:
        raise E17ContractError("existing E17 validation scenes drifted")
    cache_records = contract.get("clean_score_caches")
    if not isinstance(cache_records, Mapping):
        raise E17ContractError("existing E17 cache contract is malformed")
    for image in e12.CALIBRATION_IDS:
        record = cache_records.get(str(image))
        if not isinstance(record, Mapping):
            raise E17ContractError(f"existing E17 cache contract misses image {image}")
        _validate_result_row(
            by_image[image],
            expected_image=image,
            expected_validation_name=str(scene_by_image[image].validation_name),
            expected_cache_sha256=str(record.get("sha256", "")),
        )
    computed_summary = summarize(rows)
    computed_decision = decision(computed_summary)
    if report.get("summary") != computed_summary:
        raise E17ContractError("existing E17 summary does not match its rows")
    if report.get("decision") != computed_decision:
        raise E17ContractError("existing E17 decision does not match its summary")
    if report.get("stage") != computed_decision["status"]:
        raise E17ContractError("existing E17 terminal stage drifted")
    runtime = report.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)) or runtime < 0:
        raise E17ContractError("existing E17 runtime is invalid")


def run_gate(
    raw_cache_dir: Path,
    calibration_report: Path,
    e12_report_path: Path,
    report: Path,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(report, label="E17 report")
    if report_path.suffix.lower() != ".json":
        raise E17ContractError("E17 report must be a .json file")
    raw_dir = _require_e_drive(raw_cache_dir, label="raw score cache")
    e12_path = _require_e_drive(e12_report_path, label="E12 report")
    calibration_path = calibration_report.resolve()
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path in {e12_path, calibration_path}:
        raise E17ContractError("E17 report must not overwrite an input")
    if report_path.is_relative_to(raw_dir) or report_path.is_relative_to(clean_cache_dir):
        raise E17ContractError("E17 report must not be written inside an input cache")

    e12_report, _calibration, scenes = _load_verified_structure_inputs(
        raw_dir,
        calibration_path,
        e12_path,
    )
    clean_records = e14._clean_cache_records(e12_report)
    contract = {
        "protocol_sha256": e12.canonical_digest(E17_PROTOCOL),
        "report": str(report_path),
        "e12_report": {
            "path": str(e12_path),
            "sha256": e14.EXPECTED_E12_REPORT_SHA256,
        },
        "calibration_report": {
            "path": str(calibration_path),
            "sha256": e12.CALIBRATION_REPORT_SHA256,
        },
        "raw_cache_dir": str(raw_dir),
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
        existing = _load_json(report_path, label="existing E17 report")
        if existing.get("run_contract_sha256") != contract_digest:
            raise E17ContractError("existing E17 report belongs to different bytes")
        if existing.get("run_contract") != contract:
            raise E17ContractError("existing E17 report contract payload drifted")
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
        "stage": "structure",
        "protocol": E17_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E17_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rows": [],
        "completed_images": [],
    }
    _atomic_write_json(report_path, output)
    try:
        for scene in scenes:
            cache = e14._load_cc_cache(
                scene, e12_report, clean_records[scene.image_id]
            )
            right, down = e12.dense_from_graph(cache.cc_candidates, cache.cc_scores)
            row = {
                "image": int(scene.image_id),
                "validation_name": str(scene.validation_name),
                "clean_score_cache_sha256": str(cache.sha256),
                **measure_structure(right, down, scene.permutation),
            }
            _validate_result_row(
                row,
                expected_image=int(scene.image_id),
                expected_validation_name=str(scene.validation_name),
                expected_cache_sha256=str(cache.sha256),
            )
            output["rows"].append(row)
            output["completed_images"].append(int(scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
        summary = summarize(output["rows"])
        result = decision(summary)
        output["summary"] = summary
        output["decision"] = result
        output["status"] = "complete"
        output["stage"] = result["status"]
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
        description="Run fixed E17 CC192 rigid-island viability gate."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gate(
        args.raw_cache_dir,
        args.calibration_report,
        args.e12_report,
        args.report,
    )
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "stage": result["stage"],
                "decision": result["decision"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
