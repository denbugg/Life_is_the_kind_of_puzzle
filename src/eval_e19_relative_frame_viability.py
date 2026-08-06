"""Frozen E19 structure-only viability gate for the relative CC192 beam.

The evaluator authenticates the exact E18 complexity KILL, replays the
byte-pinned E12 clean-score inputs, and measures only the first label-free
relative layout.  It never constructs an absolute board or runs residual
completion, assembly, SSIM, restoration, or NLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import skimage

import e18_absolute_frame_beam as e18_core
import e19_relative_frame_beam as relative
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
import eval_e17_cc192_rigid_viability as e17
import eval_e18_absolute_frame_oracle as e18_eval


class E19ContractError(RuntimeError):
    """The frozen E19 protocol, input bytes, or runtime drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e19-cc192-relative-frame-viability-report-v1"
EXPERIMENT = "e19_cc192_symbolic_origin_quotient_viability_v1"

EXPECTED_E12_REPORT_SHA256 = e14.EXPECTED_E12_REPORT_SHA256
EXPECTED_E17_REPORT_SHA256 = e18_eval.EXPECTED_E17_REPORT_SHA256
EXPECTED_E18_REPORT_SHA256 = (
    "d321fee199b6459d017f4ce9febc20469684aa6c2d7adda61eb6cc7f5c20dcf8"
)
EXPECTED_E18_RUN_CONTRACT_SHA256 = (
    "a32fabab9dcf67e213b75240df93bb8efb8e9bb8d4bc08dadec4d5685c266830"
)
EXPECTED_E18_PROTOCOL_SHA256 = (
    "a1e4efb6af77d58ae495b32b6d50eb2ed7be7c2b59f6231b45d665a64335ee84"
)
EXPECTED_E18_ERROR = (
    "AbsoluteFrameError: absolute beam reached the frozen cumulative proposal cap"
)
EXPECTED_RUNTIME_PROVENANCE = dict(e18_eval.EXPECTED_RUNTIME_PROVENANCE)
E18_SHARED_SOURCE_NAMES = (
    "e15_frame_consensus.py",
    "e18_absolute_frame_beam.py",
    "eval_clean_score_oracle.py",
    "eval_e14_cc192_discovery.py",
    "eval_e17_cc192_rigid_viability.py",
    "eval_e18_absolute_frame_oracle.py",
    "rank96_lab_selector.py",
    "solve_buddies.py",
)

DECISION_RULE: dict[str, float | int] = {
    "expansion_cap_hit_scenes_max": 0,
    "one_initial_zero_root_scenes": 8,
    "legal_origin_scenes": 8,
    "mean_rigid_coverage_min": 0.35,
    "worst_rigid_coverage_min": 0.25,
    "mean_accepted_cross_seam_precision_min": 0.85,
    "worst_accepted_cross_seam_precision_min": 0.70,
    "mean_component_cycle_rank_ratio_min": 0.05,
}

E19_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e19-cc192-symbolic-origin-quotient-v1",
    "role": "target_derived_clean_score_structure_and_complexity_gate_no_board",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "authorization": {
        "e18_report_sha256": EXPECTED_E18_REPORT_SHA256,
        "e18_run_contract_sha256": EXPECTED_E18_RUN_CONTRACT_SHA256,
        "e18_protocol_sha256": EXPECTED_E18_PROTOCOL_SHA256,
        "required_status": "failed",
        "required_stage": "decoder",
        "required_error": EXPECTED_E18_ERROR,
        "candidate_rows": 0,
    },
    "inputs": {
        "e12_report_sha256": EXPECTED_E12_REPORT_SHA256,
        "e17_report_sha256_embedded_in_e18": EXPECTED_E17_REPORT_SHA256,
        "candidate": "existing_E12_CC_clean_candidates_and_scores",
    },
    "single_changed_variable": (
        "quotient_global_translation_with_largest_CC192_root_fixed_at_zero"
    ),
    "geometry": {
        "grid": relative.GRID,
        "tile_size": 20,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
        "root_translation": [0, 0],
        "initial_states": 1,
        "coordinates": "signed_relative_never_clipped_to_absolute_frame",
        "placement": "whole_rigid_component_integer_translation",
        "hard_non_overlap": True,
        "merged_bbox_height_width_max": relative.GRID,
        "legal_origins": "derive_inclusive_rectangle_and_count_after_ranking_only",
        "absolute_board": False,
    },
    "components": {
        "builder": "exact_E18_CC192_components",
        "max_edges": relative.COMPONENT_MAX_EDGES,
        "min_margin": relative.MIN_MARGIN,
        "root": "largest_then_minimum_tile_then_entries",
        "purity_trim": False,
        "singletons_in_beam": False,
    },
    "bridges": {
        "source": "unchanged_E18_positive_dense_top8_U_D_L_R_per_frontier",
        "top_k": relative.CANDIDATE_TOP_K,
        "single_bridge_allowed": True,
        "candidate_dedupe": "component_id_plus_relative_shift",
        "collect_all_physical_contacts": True,
    },
    "search": {
        "beam_width": relative.BEAM_WIDTH,
        "evaluated_translations_per_state": relative.PROPOSALS_PER_STATE,
        "pre_geometry_translation_rank": [
            "distinct_supporting_claim_count_desc",
            "supporting_claim_score_sum_desc",
            "maximum_supporting_claim_score_desc",
            "component_id_asc",
            "shift_row_asc",
            "shift_col_asc",
        ],
        "attachment_rounds": relative.MAX_ATTACHMENTS,
        "relative_layouts_global_per_scene": relative.RELATIVE_LAYOUTS,
        "proposal_evaluation_cap_per_scene": relative.EXPANSION_CAP,
        "cap_counter": (
            "distinct_relative_state_key_component_id_relative_shift_before_geometry"
        ),
        "cap_reaching": "immediate_complete_KILL_no_truncated_metrics",
        "state_rank": [
            "component_cycle_rank",
            "satisfied_distinct_dense_top8_bridge_claims",
            "rigid_tiles",
            "unique_component_contacts",
            "unique_physical_cross_seams",
            "frozen_cross_neural_sum",
            "corrupted_depth1_Lab_exact_neural_tie_only",
        ],
        "component_cycle_rank_ratio": (
            "cycle_rank/max(1,placed_components_minus_1)"
        ),
        "dedupe": "exact_relative_component_translations_with_root_zero",
    },
    "measurement": {
        "layout": "first_layout_under_frozen_label_free_rank_only",
        "rigid_coverage": "rigid_tiles_divided_by_576",
        "accepted_cross_seam_precision": "true_divided_by_count_else_zero",
        "labels": "evaluator_only_after_search_returns",
    },
    "decision": dict(DECISION_RULE),
    "routing": {
        "pass": "open_separately_frozen_E20_absolute_origin_and_residual",
        "fail": "close_exact_dense_top8_single_edge_beam_without_resweep",
    },
    "excluded": [
        "absolute_board",
        "absolute_origin_inside_search",
        "residual_completion",
        "candidate_solve_metric",
        "placement",
        "neighbour",
        "SSIM",
        "NLM",
        "labels_inside_search",
        "modal_purity_trim",
        "bridge_topk_beam_or_cap_sweep",
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
DEFAULT_E18_REPORT = Path(
    "E:/pazzle_work/absolute_frame_e18/cc192_absolute_frame_beam_v1.json"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/relative_frame_e19/cc192_origin_quotient_viability_v1.json"
)


@dataclass(frozen=True)
class E19Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e18_report: Path
    report: Path


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E19ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E19ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E19ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E19 report")
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
        raise E19ContractError(
            f"E19 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e15_frame_consensus.py": source / "e15_frame_consensus.py",
        "e18_absolute_frame_beam.py": source / "e18_absolute_frame_beam.py",
        "e19_relative_frame_beam.py": source / "e19_relative_frame_beam.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_e17_cc192_rigid_viability.py": source
        / "eval_e17_cc192_rigid_viability.py",
        "eval_e18_absolute_frame_oracle.py": source
        / "eval_e18_absolute_frame_oracle.py",
        "eval_e19_relative_frame_viability.py": Path(__file__).resolve(),
        "rank96_lab_selector.py": source / "rank96_lab_selector.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _verify_e18_cap_kill(path: Path) -> Mapping[str, Any]:
    resolved = _require_e_drive(path, label="E18 report")
    if not resolved.is_file():
        raise E19ContractError(f"E18 report is missing: {resolved}")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E18_REPORT_SHA256:
        raise E19ContractError(
            "E18 report SHA256 mismatch: "
            f"expected {EXPECTED_E18_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E18 report")
    rows = report.get("rows")
    decisions = report.get("decisions")
    contract = report.get("run_contract")
    rr = rows.get("RR96") if isinstance(rows, Mapping) else None
    candidate = rows.get("candidate") if isinstance(rows, Mapping) else None
    if (
        report.get("schema_version") != e18_eval.SCHEMA_VERSION
        or report.get("schema") != e18_eval.REPORT_SCHEMA
        or report.get("experiment") != e18_eval.EXPERIMENT
        or report.get("status") != "failed"
        or report.get("stage") != "decoder"
        or report.get("error") != EXPECTED_E18_ERROR
        or report.get("protocol") != e18_eval.E18_PROTOCOL
        or report.get("protocol_sha256") != EXPECTED_E18_PROTOCOL_SHA256
        or report.get("run_contract_sha256")
        != EXPECTED_E18_RUN_CONTRACT_SHA256
        or report.get("completed_decoder_images") != []
        or report.get("completed_nlm_images") != []
        or not isinstance(rr, list)
        or len(rr) != len(e12.CALIBRATION_IDS)
        or [int(row.get("image", -1)) for row in rr if isinstance(row, Mapping)]
        != list(e12.CALIBRATION_IDS)
        or candidate != []
        or decisions
        != {
            "decoder": {"status": "not_run"},
            "end_to_end": {"status": "not_run"},
        }
        or not isinstance(contract, Mapping)
        or contract.get("protocol_sha256") != EXPECTED_E18_PROTOCOL_SHA256
    ):
        raise E19ContractError("E18 fixed-cap KILL contract drifted")
    e12_record = contract.get("e12_report")
    e17_record = contract.get("e17_report")
    frozen_sources = contract.get("source_provenance")
    if (
        not isinstance(e12_record, Mapping)
        or e12_record.get("sha256") != EXPECTED_E12_REPORT_SHA256
        or not isinstance(e17_record, Mapping)
        or e17_record.get("sha256") != EXPECTED_E17_REPORT_SHA256
        or not isinstance(frozen_sources, Mapping)
    ):
        raise E19ContractError("E18 embedded E12/E17 provenance drifted")
    source_dir = Path(__file__).resolve().parent
    for name in E18_SHARED_SOURCE_NAMES:
        observed = e12.sha256_file(source_dir / name)
        if frozen_sources.get(name) != observed:
            raise E19ContractError(
                f"shared E18-to-E19 source drifted for {name}: "
                f"expected {frozen_sources.get(name)}, got {observed}"
            )
    runtime = report.get("runtime_seconds")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise E19ContractError("E18 failed-report runtime is invalid")
    return report


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise E19ContractError(f"{label} is not an integer")
    return int(value)


def _finite(
    value: object, *, label: str, minimum: float, maximum: float
) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise E19ContractError(f"{label} is not numeric") from exc
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(observed):
        raise E19ContractError(f"{label} is not finite")
    if not minimum <= observed <= maximum:
        raise E19ContractError(f"{label} is outside [{minimum}, {maximum}]")
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
        raise E19ContractError(f"{label} is not a sequence")
    output: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise E19ContractError(f"{label} contains a malformed row")
        output.append(
            tuple(_integer(item, label=f"{label} value") for item in row)
        )
    return tuple(output)


def _validate_permutation(permutation: np.ndarray) -> np.ndarray:
    value = np.asarray(permutation)
    if value.shape != (relative.NUM_TILES,) or value.dtype.kind not in "iu":
        raise E19ContractError("permutation geometry/dtype drifted")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(relative.NUM_TILES)):
        raise E19ContractError("permutation is not a bijection")
    return value


def _seam_is_true(seam: Sequence[int], permutation: np.ndarray) -> bool:
    if not isinstance(seam, (list, tuple)) or len(seam) != 4:
        raise E19ContractError("accepted cross seam is malformed")
    first, second, dy, dx = (
        _integer(value, label="accepted cross seam value") for value in seam
    )
    if (dy, dx) not in {(0, 1), (1, 0)}:
        raise E19ContractError("accepted cross seam is not canonical right/down")
    if not (
        0 <= first < relative.NUM_TILES
        and 0 <= second < relative.NUM_TILES
        and first != second
    ):
        raise E19ContractError("accepted cross seam tile ID is invalid")
    value = _validate_permutation(permutation)
    first_row, first_col = divmod(int(value[first]), relative.GRID)
    second_row, second_col = divmod(int(value[second]), relative.GRID)
    return (second_row - first_row, second_col - first_col) == (dy, dx)


def accepted_cross_seam_precision(
    seams: Sequence[Sequence[int]], permutation: np.ndarray
) -> tuple[int, int, float]:
    identities = _tuple_rows(seams, width=4, label="accepted cross seams")
    if len(set(identities)) != len(identities):
        raise E19ContractError("accepted cross seams are duplicated")
    true_count = sum(_seam_is_true(seam, permutation) for seam in identities)
    count = len(identities)
    return int(true_count), count, float(true_count / count if count else 0.0)


def _diagnostics_payload(
    diagnostics: relative.RelativeBeamDiagnostics,
) -> dict[str, Any]:
    return {
        "cc192_component_count": int(diagnostics.cc192_component_count),
        "cc192_nontrivial_components": int(
            diagnostics.cc192_nontrivial_components
        ),
        "cc192_nontrivial_tiles": int(diagnostics.cc192_nontrivial_tiles),
        "root_component_id": int(diagnostics.root_component_id),
        "root_component_size": int(diagnostics.root_component_size),
        "initial_states": int(diagnostics.initial_states),
        "bridge_claims": int(diagnostics.bridge_claims),
        "rounds": int(diagnostics.rounds),
        "proposal_evaluations": int(diagnostics.proposal_evaluations),
        "cap_hit": bool(diagnostics.cap_hit),
        "layouts_retained": int(diagnostics.layouts_retained),
    }


def _layout_payload(
    layout: relative.RelativeLayout, permutation: np.ndarray
) -> dict[str, Any]:
    true_seams, seam_count, seam_precision = accepted_cross_seam_precision(
        layout.cross_seams, permutation
    )
    if seam_count != len(layout.cross_seams):
        raise E19ContractError("accepted cross seam count drifted")
    return {
        "translations": [list(value) for value in layout.translations],
        "relative_entries": [list(value) for value in layout.relative_entries],
        "satisfied_bridge_claims": list(layout.satisfied_bridge_claims),
        "component_contacts": [list(value) for value in layout.component_contacts],
        "accepted_cross_seams": [list(value) for value in layout.cross_seams],
        "true_accepted_cross_seams": int(true_seams),
        "accepted_cross_seam_precision": float(seam_precision),
        "cross_neural_sum": float(layout.cross_neural_sum),
        "cross_lab_sum": float(layout.cross_lab_sum),
        "rigid_tiles": int(layout.rigid_tiles),
        "rigid_coverage": float(layout.rigid_coverage),
        "component_cycle_rank": int(layout.component_cycle_rank),
        "component_cycle_rank_ratio": float(
            layout.component_cycle_rank_ratio
        ),
        "bbox": list(layout.bbox),
        "bbox_height": int(layout.bbox_height),
        "bbox_width": int(layout.bbox_width),
        "legal_origin_bounds": list(layout.legal_origin_bounds),
        "legal_origin_count": int(layout.legal_origin_count),
    }


def evaluate_structure(
    scene: e12.RawScene,
    result: relative.RelativeBeamResult,
    *,
    clean_score_cache_sha256: str,
) -> dict[str, Any]:
    if not result.layouts:
        raise E19ContractError("relative beam returned no ranked layout")
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "clean_score_cache_sha256": str(clean_score_cache_sha256),
        "arm": "E19_relative_frame_quotient",
        "diagnostics": _diagnostics_payload(result.diagnostics),
        "best_layout": _layout_payload(result.layouts[0], scene.permutation),
    }


def _expected_cross_geometry(
    entries: Sequence[tuple[int, int, int]], graph: relative.GraphData
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int, int, int]]]:
    occupied = {(row, col): tile for tile, row, col in entries}
    contacts: set[tuple[int, int]] = set()
    seams: set[tuple[int, int, int, int]] = set()
    for (row, col), tile in occupied.items():
        for dy, dx in relative.DELTAS:
            neighbour = occupied.get((row + dy, col + dx))
            if neighbour is None:
                continue
            first_component = int(graph.owner[tile])
            second_component = int(graph.owner[neighbour])
            if first_component == second_component:
                continue
            contacts.add(tuple(sorted((first_component, second_component))))
            seams.add(
                e18_core.e15._physical_seam_identity(
                    tile, neighbour, dy, dx
                )
            )
    return frozenset(contacts), frozenset(seams)


def _validate_success_row(
    row: Mapping[str, Any],
    *,
    scene: e12.RawScene,
    cache_sha256: str,
    right: np.ndarray,
    down: np.ndarray,
    graph: relative.GraphData,
) -> None:
    expected_row_keys = {
        "image",
        "validation_name",
        "clean_score_cache_sha256",
        "arm",
        "diagnostics",
        "best_layout",
    }
    if not isinstance(row, Mapping) or set(row) != expected_row_keys:
        raise E19ContractError("E19 structure row fields drifted")
    image = int(scene.image_id)
    if (
        _integer(row.get("image"), label="row image") != image
        or row.get("validation_name") != str(scene.validation_name)
        or row.get("clean_score_cache_sha256") != cache_sha256
        or not _is_sha256(row.get("clean_score_cache_sha256"))
        or row.get("arm") != "E19_relative_frame_quotient"
    ):
        raise E19ContractError(f"E19 row provenance drifted for image {image}")

    diagnostics = row.get("diagnostics")
    expected_diagnostic_keys = {
        "cc192_component_count",
        "cc192_nontrivial_components",
        "cc192_nontrivial_tiles",
        "root_component_id",
        "root_component_size",
        "initial_states",
        "bridge_claims",
        "rounds",
        "proposal_evaluations",
        "cap_hit",
        "layouts_retained",
    }
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != expected_diagnostic_keys:
        raise E19ContractError(f"E19 diagnostics drifted for image {image}")
    expected_diagnostics = {
        "cc192_component_count": len(graph.components),
        "cc192_nontrivial_components": len(graph.nontrivial),
        "cc192_nontrivial_tiles": sum(
            component.size
            for component in graph.components
            if component.size >= 2
        ),
        "root_component_id": 0,
        "root_component_size": graph.components[0].size,
        "initial_states": 1,
        "bridge_claims": len(graph.claims),
    }
    for key, expected in expected_diagnostics.items():
        if _integer(diagnostics.get(key), label=f"image {image} {key}") != expected:
            raise E19ContractError(f"E19 {key} drifted for image {image}")
    rounds = _integer(diagnostics.get("rounds"), label="relative rounds")
    evaluations = _integer(
        diagnostics.get("proposal_evaluations"), label="proposal evaluations"
    )
    retained = _integer(
        diagnostics.get("layouts_retained"), label="layouts retained"
    )
    if (
        not 0 <= rounds <= relative.MAX_ATTACHMENTS
        or not 0 <= evaluations < relative.EXPANSION_CAP
        or diagnostics.get("cap_hit") is not False
        or not 1 <= retained <= relative.RELATIVE_LAYOUTS
    ):
        raise E19ContractError(f"E19 search diagnostics drifted for image {image}")

    layout = row.get("best_layout")
    expected_layout_keys = {
        "translations",
        "relative_entries",
        "satisfied_bridge_claims",
        "component_contacts",
        "accepted_cross_seams",
        "true_accepted_cross_seams",
        "accepted_cross_seam_precision",
        "cross_neural_sum",
        "cross_lab_sum",
        "rigid_tiles",
        "rigid_coverage",
        "component_cycle_rank",
        "component_cycle_rank_ratio",
        "bbox",
        "bbox_height",
        "bbox_width",
        "legal_origin_bounds",
        "legal_origin_count",
    }
    if not isinstance(layout, Mapping) or set(layout) != expected_layout_keys:
        raise E19ContractError(f"E19 best layout fields drifted for image {image}")
    translations = _tuple_rows(
        layout.get("translations"), width=3, label="relative translations"
    )
    if (
        not translations
        or translations != tuple(sorted(translations))
        or len({value[0] for value in translations}) != len(translations)
        or translations[0] != (0, 0, 0)
        or any(value[0] not in graph.nontrivial for value in translations)
    ):
        raise E19ContractError(f"E19 quotient key drifted for image {image}")
    translation_map = {
        component_id: (shift_row, shift_col)
        for component_id, shift_row, shift_col in translations
    }
    expected_entries = tuple(
        sorted(
            (
                int(tile),
                int(local_row + translation_map[component_id][0]),
                int(local_col + translation_map[component_id][1]),
            )
            for component_id in translation_map
            for tile, local_row, local_col in graph.components[
                component_id
            ].entries
        )
    )
    entries = _tuple_rows(
        layout.get("relative_entries"), width=3, label="relative entries"
    )
    if entries != expected_entries or entries != tuple(sorted(entries)):
        raise E19ContractError(f"E19 rigid entries drifted for image {image}")
    if (
        len({tile for tile, _row, _col in entries}) != len(entries)
        or len({(row, col) for _tile, row, col in entries}) != len(entries)
    ):
        raise E19ContractError(f"E19 relative entries overlap for image {image}")
    rows = [row_value for _tile, row_value, _col in entries]
    cols = [col_value for _tile, _row, col_value in entries]
    expected_bbox = (min(rows), max(rows), min(cols), max(cols))
    bbox = tuple(
        _integer(value, label="bbox value")
        for value in (layout.get("bbox") or [])
    )
    if len(bbox) != 4 or bbox != expected_bbox:
        raise E19ContractError(f"E19 bbox drifted for image {image}")
    height = bbox[1] - bbox[0] + 1
    width = bbox[3] - bbox[2] + 1
    if (
        not 1 <= height <= relative.GRID
        or not 1 <= width <= relative.GRID
        or _integer(layout.get("bbox_height"), label="bbox height") != height
        or _integer(layout.get("bbox_width"), label="bbox width") != width
    ):
        raise E19ContractError(f"E19 bbox span drifted for image {image}")
    expected_origin_bounds = (
        -bbox[0],
        relative.GRID - 1 - bbox[1],
        -bbox[2],
        relative.GRID - 1 - bbox[3],
    )
    origin_bounds = tuple(
        _integer(value, label="legal-origin bound")
        for value in (layout.get("legal_origin_bounds") or [])
    )
    origin_count = (relative.GRID + 1 - height) * (
        relative.GRID + 1 - width
    )
    if (
        len(origin_bounds) != 4
        or origin_bounds != expected_origin_bounds
        or _integer(
            layout.get("legal_origin_count"), label="legal-origin count"
        )
        != origin_count
        or origin_count < 1
    ):
        raise E19ContractError(f"E19 legal-origin algebra drifted for image {image}")

    rigid_tiles = _integer(layout.get("rigid_tiles"), label="rigid tiles")
    if rigid_tiles != len(entries):
        raise E19ContractError(f"E19 rigid tile count drifted for image {image}")
    rigid_coverage = _finite(
        layout.get("rigid_coverage"),
        label="rigid coverage",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(
        rigid_coverage,
        rigid_tiles / relative.NUM_TILES,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise E19ContractError(f"E19 rigid coverage drifted for image {image}")

    claims_raw = layout.get("satisfied_bridge_claims")
    if not isinstance(claims_raw, (list, tuple)):
        raise E19ContractError(f"E19 satisfied claims are malformed for image {image}")
    claims = tuple(
        _integer(value, label="satisfied claim ID") for value in claims_raw
    )
    expected_claims: set[int] = set()
    for component_id in translation_map:
        expected_claims.update(
            relative._satisfied_new_claims(component_id, translation_map, graph)
        )
    if claims != tuple(sorted(expected_claims)):
        raise E19ContractError(f"E19 satisfied claims drifted for image {image}")

    contacts = _tuple_rows(
        layout.get("component_contacts"), width=2, label="component contacts"
    )
    seams = _tuple_rows(
        layout.get("accepted_cross_seams"), width=4, label="accepted cross seams"
    )
    expected_contacts, expected_seams = _expected_cross_geometry(entries, graph)
    if contacts != tuple(sorted(expected_contacts)) or seams != tuple(
        sorted(expected_seams)
    ):
        raise E19ContractError(f"E19 cross geometry drifted for image {image}")
    if len(contacts) < len(translations) - 1:
        raise E19ContractError(f"E19 component graph disconnected for image {image}")
    cycle_rank = max(0, len(contacts) - len(translations) + 1)
    if _integer(
        layout.get("component_cycle_rank"), label="component cycle rank"
    ) != cycle_rank:
        raise E19ContractError(f"E19 component cycle rank drifted for image {image}")
    expected_cycle_ratio = float(cycle_rank / max(1, len(translations) - 1))
    cycle_ratio = _finite(
        layout.get("component_cycle_rank_ratio"),
        label="component cycle ratio",
        minimum=0.0,
        maximum=float("inf"),
    )
    if not math.isclose(
        cycle_ratio, expected_cycle_ratio, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise E19ContractError(f"E19 cycle ratio drifted for image {image}")

    r = e18_core._dense(right, label="right")
    d = e18_core._dense(down, label="down")
    neural = float(sum(relative._seam_value(seam, r, d) for seam in seams))
    lab_right, lab_down = e18_core.e15._lab_pair_matrices(scene.tiles_uint8)
    lab = float(
        sum(relative._seam_value(seam, lab_right, lab_down) for seam in seams)
    )
    if not math.isclose(
        _finite(
            layout.get("cross_neural_sum"),
            label="cross neural sum",
            minimum=0.0,
            maximum=float("inf"),
        ),
        neural,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        _finite(
            layout.get("cross_lab_sum"),
            label="cross Lab sum",
            minimum=float("-inf"),
            maximum=float("inf"),
        ),
        lab,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise E19ContractError(f"E19 cross evidence drifted for image {image}")

    true_seams, seam_count, seam_precision = accepted_cross_seam_precision(
        seams, scene.permutation
    )
    if (
        _integer(
            layout.get("true_accepted_cross_seams"),
            label="true accepted seams",
        )
        != true_seams
        or seam_count != len(seams)
        or not math.isclose(
            _finite(
                layout.get("accepted_cross_seam_precision"),
                label="accepted cross seam precision",
                minimum=0.0,
                maximum=1.0,
            ),
            seam_precision,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise E19ContractError(f"E19 seam truth drifted for image {image}")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E19ContractError("E19 summary requires exactly eight rows")
    images = [int(row.get("image", -1)) for row in rows]
    if len(set(images)) != len(images) or tuple(sorted(images)) != e12.CALIBRATION_IDS:
        raise E19ContractError("E19 summary image IDs drifted")
    layouts = [row["best_layout"] for row in rows]
    diagnostics = [row["diagnostics"] for row in rows]
    return {
        "images": len(rows),
        "expansion_cap_hit_scenes": int(
            sum(bool(value["cap_hit"]) for value in diagnostics)
        ),
        "one_initial_zero_root_scenes": int(
            sum(
                int(value["initial_states"]) == 1
                and tuple(layout["translations"][0]) == (0, 0, 0)
                for value, layout in zip(diagnostics, layouts)
            )
        ),
        "legal_origin_scenes": int(
            sum(int(layout["legal_origin_count"]) >= 1 for layout in layouts)
        ),
        "mean_rigid_coverage": float(
            np.mean([float(layout["rigid_coverage"]) for layout in layouts])
        ),
        "worst_rigid_coverage": float(
            min(float(layout["rigid_coverage"]) for layout in layouts)
        ),
        "mean_accepted_cross_seam_precision": float(
            np.mean(
                [
                    float(layout["accepted_cross_seam_precision"])
                    for layout in layouts
                ]
            )
        ),
        "worst_accepted_cross_seam_precision": float(
            min(
                float(layout["accepted_cross_seam_precision"])
                for layout in layouts
            )
        ),
        "mean_component_cycle_rank_ratio": float(
            np.mean(
                [
                    float(layout["component_cycle_rank_ratio"])
                    for layout in layouts
                ]
            )
        ),
        "mean_proposal_evaluations": float(
            np.mean([int(value["proposal_evaluations"]) for value in diagnostics])
        ),
        "max_proposal_evaluations": int(
            max(int(value["proposal_evaluations"]) for value in diagnostics)
        ),
    }


def decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "expansion_cap_hit_scenes": int(summary["expansion_cap_hit_scenes"]),
        "one_initial_zero_root_scenes": int(
            summary["one_initial_zero_root_scenes"]
        ),
        "legal_origin_scenes": int(summary["legal_origin_scenes"]),
        "mean_rigid_coverage": float(summary["mean_rigid_coverage"]),
        "worst_rigid_coverage": float(summary["worst_rigid_coverage"]),
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
        "expansion_cap_hit_scenes": observed["expansion_cap_hit_scenes"]
        <= int(DECISION_RULE["expansion_cap_hit_scenes_max"]),
        "one_initial_zero_root_scenes": observed["one_initial_zero_root_scenes"]
        == int(DECISION_RULE["one_initial_zero_root_scenes"]),
        "legal_origin_scenes": observed["legal_origin_scenes"]
        == int(DECISION_RULE["legal_origin_scenes"]),
        "mean_rigid_coverage": observed["mean_rigid_coverage"]
        >= float(DECISION_RULE["mean_rigid_coverage_min"]),
        "worst_rigid_coverage": observed["worst_rigid_coverage"]
        >= float(DECISION_RULE["worst_rigid_coverage_min"]),
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
            "go_E20_absolute_origin_residual"
            if passed
            else "kill_dense_top8_single_edge_beam"
        ),
        "passed": passed,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_score_structure_oracle_not_deployable",
    }


def _cap_failure_payload(
    *,
    scene: e12.RawScene,
    cache_sha256: str,
    error: relative.RelativeFrameCapError,
) -> dict[str, Any]:
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "clean_score_cache_sha256": str(cache_sha256),
        "proposal_evaluations": int(error.proposal_evaluations),
        "rounds": int(error.rounds),
        "initial_states": int(error.initial_states),
        "cap_hit": bool(error.cap_hit),
        "error": f"{type(error).__name__}: {error}",
    }


def cap_decision(cap_failure: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "expansion_cap_hit_scenes": 1,
        "image": int(cap_failure["image"]),
        "proposal_evaluations": int(cap_failure["proposal_evaluations"]),
        "rounds": int(cap_failure["rounds"]),
    }
    return {
        "status": "kill_relative_cap",
        "passed": False,
        "thresholds": dict(DECISION_RULE),
        "observed": observed,
        "checks": {
            "expansion_cap_hit_scenes": False,
            "one_initial_zero_root_scenes": "not_run",
            "legal_origin_scenes": "not_run",
            "mean_rigid_coverage": "not_run",
            "worst_rigid_coverage": "not_run",
            "mean_accepted_cross_seam_precision": "not_run",
            "worst_accepted_cross_seam_precision": "not_run",
            "mean_component_cycle_rank_ratio": "not_run",
        },
        "scope": "complexity_KILL_no_truncated_layout_metrics",
    }


def _validate_cap_failure(
    value: Mapping[str, Any],
    *,
    e12_report: Mapping[str, Any],
    scenes: Sequence[e12.RawScene],
    clean_records: Mapping[int, Mapping[str, Any]],
) -> None:
    expected_keys = {
        "image",
        "validation_name",
        "clean_score_cache_sha256",
        "proposal_evaluations",
        "rounds",
        "initial_states",
        "cap_hit",
        "error",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise E19ContractError("E19 cap failure fields drifted")
    image = _integer(value.get("image"), label="cap failure image")
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    if image not in scene_by_image:
        raise E19ContractError("E19 cap failure image drifted")
    scene = scene_by_image[image]
    record = clean_records.get(image)
    if not isinstance(record, Mapping):
        raise E19ContractError("E19 cap failure cache record is missing")
    try:
        cache = e14._load_cc_cache(scene, e12_report, record)
    except e14.E14ContractError as exc:
        raise E19ContractError(str(exc)) from exc
    if (
        value.get("validation_name") != str(scene.validation_name)
        or value.get("clean_score_cache_sha256") != str(cache.sha256)
        or str(cache.sha256) != str(record.get("sha256", ""))
        or _integer(
            value.get("proposal_evaluations"), label="cap proposal evaluations"
        )
        != relative.EXPANSION_CAP
        or not 0
        <= _integer(value.get("rounds"), label="cap rounds")
        < relative.MAX_ATTACHMENTS
        or _integer(value.get("initial_states"), label="cap initial states") != 1
        or value.get("cap_hit") is not True
        or value.get("error")
        != (
            "RelativeFrameCapError: relative beam reached the frozen cumulative "
            "proposal cap"
        )
    ):
        raise E19ContractError("E19 cap failure payload drifted")


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
        or report.get("protocol") != E19_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E19_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E19ContractError("existing E19 complete report contract drifted")
    _finite(
        report.get("runtime_seconds"),
        label="existing E19 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    rows = report.get("rows")
    completed = report.get("completed_images")
    if not isinstance(rows, list) or not isinstance(completed, list):
        raise E19ContractError("existing E19 row groups are malformed")
    if report.get("stage") == "kill_relative_cap":
        cap_failure = report.get("cap_failure")
        if (
            rows != []
            or completed != []
            or "summary" in report
            or not isinstance(cap_failure, Mapping)
        ):
            raise E19ContractError("existing E19 cap-KILL stage drifted")
        _validate_cap_failure(
            cap_failure,
            e12_report=e12_report,
            scenes=scenes,
            clean_records=clean_records,
        )
        if report.get("decision") != cap_decision(cap_failure):
            raise E19ContractError("existing E19 cap decision drifted")
        return
    if "cap_failure" in report:
        raise E19ContractError("existing E19 successful route contains cap failure")
    if completed != list(e12.CALIBRATION_IDS):
        raise E19ContractError("existing E19 completed image IDs drifted")
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E19ContractError("existing E19 rows are incomplete")
    by_image = {
        int(row.get("image", -1)): row for row in rows if isinstance(row, Mapping)
    }
    if tuple(sorted(by_image)) != e12.CALIBRATION_IDS or len(by_image) != len(rows):
        raise E19ContractError("existing E19 rows are duplicated or misidentified")
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    for image in e12.CALIBRATION_IDS:
        scene = scene_by_image[image]
        cache = e14._load_cc_cache(scene, e12_report, clean_records[image])
        right, down = e12.dense_from_graph(cache.cc_candidates, cache.cc_scores)
        graph = relative.build_graph_data(right, down)
        _validate_success_row(
            by_image[image],
            scene=scene,
            cache_sha256=str(cache.sha256),
            right=right,
            down=down,
            graph=graph,
        )
    computed_summary = summarize(rows)
    computed_decision = decision(computed_summary)
    if report.get("summary") != computed_summary:
        raise E19ContractError("existing E19 summary drifted")
    if report.get("decision") != computed_decision:
        raise E19ContractError("existing E19 decision drifted")
    if report.get("stage") != computed_decision["status"]:
        raise E19ContractError("existing E19 terminal stage drifted")


def run_gate(paths: E19Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E19 report")
    if report_path.suffix.lower() != ".json":
        raise E19ContractError("E19 report must be a .json file")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e18_report_path = _require_e_drive(paths.e18_report, label="E18 report")
    calibration_path = paths.calibration_report.resolve()
    if report_path in {e12_report_path, e18_report_path, calibration_path}:
        raise E19ContractError("E19 report must not overwrite an input")
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(
        clean_cache_dir
    ):
        raise E19ContractError("E19 report must not be written inside an input cache")

    e18_report = _verify_e18_cap_kill(e18_report_path)
    try:
        e12_report, _calibration, scenes = e17._load_verified_structure_inputs(
            raw_cache_dir,
            calibration_path,
            e12_report_path,
        )
        clean_records = e14._clean_cache_records(e12_report)
    except (e17.E17ContractError, e14.E14ContractError) as exc:
        raise E19ContractError(str(exc)) from exc
    contract = {
        "protocol_sha256": e12.canonical_digest(E19_PROTOCOL),
        "e18_report": {
            "path": str(e18_report_path),
            "sha256": EXPECTED_E18_REPORT_SHA256,
            "run_contract_sha256": str(e18_report["run_contract_sha256"]),
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
        existing = _load_json(report_path, label="existing E19 report")
        if existing.get("run_contract_sha256") != contract_digest:
            raise E19ContractError("existing E19 report belongs to different bytes")
        if existing.get("run_contract") != contract:
            raise E19ContractError("existing E19 report contract payload drifted")
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
        "stage": "relative_structure",
        "protocol": E19_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E19_PROTOCOL),
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
            graph = relative.build_graph_data(right, down)
            try:
                result = relative.run_relative_frame(
                    right, down, scene.tiles_uint8
                )
            except relative.RelativeFrameCapError as exc:
                cap_failure = _cap_failure_payload(
                    scene=scene, cache_sha256=cache.sha256, error=exc
                )
                output["rows"] = []
                output["completed_images"] = []
                output["cap_failure"] = cap_failure
                output["decision"] = cap_decision(cap_failure)
                _validate_cap_failure(
                    cap_failure,
                    e12_report=e12_report,
                    scenes=scenes,
                    clean_records=clean_records,
                )
                output["status"] = "complete"
                output["stage"] = "kill_relative_cap"
                output["runtime_seconds"] = float(time.perf_counter() - started)
                _atomic_write_json(report_path, output)
                return output
            row = evaluate_structure(
                scene,
                result,
                clean_score_cache_sha256=cache.sha256,
            )
            _validate_success_row(
                row,
                scene=scene,
                cache_sha256=cache.sha256,
                right=right,
                down=down,
                graph=graph,
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
        description="Run fixed CPU-only E19 relative-frame viability gate."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e18-report", type=Path, default=DEFAULT_E18_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gate(
        E19Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e18_report=args.e18_report,
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
