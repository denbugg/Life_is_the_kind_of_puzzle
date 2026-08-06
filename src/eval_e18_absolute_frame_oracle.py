"""Staged CPU-only E18 CC192 absolute-frame clean-oracle discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import skimage
from skimage.metrics import structural_similarity as sk_ssim

import e18_absolute_frame_beam as frame
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
import eval_e17_cc192_rigid_viability as e17
from imgio import assemble
from placement_metrics import neighbour_accuracy, placement_accuracy


class E18ContractError(RuntimeError):
    """The frozen E18 protocol, code, or an input byte drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e18-cc192-absolute-frame-clean-oracle-report-v1"
EXPERIMENT = "e18_cc192_absolute_frame_sparse_path_beam_v1"
EXPECTED_E12_REPORT_SHA256 = e14.EXPECTED_E12_REPORT_SHA256
EXPECTED_E17_REPORT_SHA256 = (
    "09fc4fed8e222a1de917f9781a1ec94d4b428b6dad06aa289dfd2a9f0fbbde92"
)
EXPECTED_RR_MEAN_SOLVE_SSIM = e14.EXPECTED_RR_MEAN_SOLVE_SSIM
EXPECTED_RR_MEAN_FINAL_SSIM = e14.EXPECTED_RR_MEAN_FINAL_SSIM
EXPECTED_RUNTIME_PROVENANCE = {
    **dict(e14.EXPECTED_RUNTIME_PROVENANCE),
    "scipy": "1.16.2",
}

DECODER_RULE: dict[str, float | int] = {
    "expansion_cap_hit_scenes_max": 0,
    "strict_bijection_scenes": 8,
    "mean_rigid_coverage_min": 0.35,
    "mean_accepted_cross_seam_precision_min": 0.85,
    "worst_accepted_cross_seam_precision_min": 0.70,
    "mean_component_cycle_rank_ratio_min": 0.05,
    "mean_placement_min": 0.02,
    "mean_neighbour_min": 0.20,
    "candidate_minus_rr96_mean_solve_ssim_min": 0.005,
}
END_TO_END_RULE: dict[str, float | int] = {
    "candidate_minus_rr96_mean_solve_ssim_min": 0.010,
    "candidate_minus_rr96_mean_final_ssim_min": 0.015,
    "candidate_minus_rr96_final_wins_min": 6,
    "candidate_minus_rr96_worst_final_delta_min": -0.020,
}

E18_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e18-cc192-absolute-frame-sparse-path-beam-v1",
    "role": "opened_E12_changed_decoder_clean_oracle_not_deployable",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "inputs": {
        "e12_report_sha256": EXPECTED_E12_REPORT_SHA256,
        "e17_report_sha256": EXPECTED_E17_REPORT_SHA256,
        "e17_required_stage": "go_E18_absolute_frame_beam",
        "RR96": "exact_pinned_E12_replay",
        "candidate": "existing_E12_CC_clean_candidates_and_scores",
    },
    "geometry": {
        "grid": frame.GRID,
        "tile_size": 20,
        "orientation": "upright_only",
        "rotation": False,
        "reflection": False,
        "component_transform": "integer_translation_only",
        "hard_non_overlap": True,
        "absolute_shift_canonicalisation": False,
    },
    "components": {
        "builder": "solve_buddies.build_buddies_components_exact",
        "max_edges": frame.COMPONENT_MAX_EDGES,
        "min_margin": frame.MIN_MARGIN,
        "rigid": "every_component_admitted_to_partial_core",
        "root": "largest_then_minimum_tile_then_entries",
        "purity_trim": False,
        "singletons_in_beam": False,
    },
    "bridges": {
        "source": "positive_dense_top8_U_D_L_R_per_frontier",
        "top_k": frame.CANDIDATE_TOP_K,
        "sort": "score_desc_then_tile_id",
        "single_bridge_allowed": True,
        "candidate_dedupe": "component_id_plus_absolute_shift",
        "collect_all_physical_contacts": True,
    },
    "search": {
        "root_origins": "every_legal_origin_processed_through_first_proposal_layer",
        "beam_width": frame.BEAM_WIDTH,
        "evaluated_translations_per_state": frame.PROPOSALS_PER_STATE,
        "pre_geometry_translation_rank": [
            "distinct_supporting_claim_count_desc",
            "supporting_claim_score_sum_desc",
            "maximum_supporting_claim_score_desc",
            "component_id_asc",
            "shift_row_asc",
            "shift_col_asc",
        ],
        "attachment_rounds": frame.MAX_ATTACHMENTS,
        "absolute_layouts_global_per_scene": frame.ABSOLUTE_LAYOUTS,
        "proposal_evaluation_cap_per_scene": frame.EXPANSION_CAP,
        "cap_counter": "distinct_state_candidate_before_geometry_acceptance",
        "cap_reaching": "hard_failure_not_truncated_success",
        "state_rank": [
            "component_cycle_rank",
            "satisfied_distinct_dense_top8_bridge_claims",
            "rigid_tiles",
            "unique_component_contacts",
            "unique_physical_cross_seams",
            "frozen_cross_neural_sum",
            "corrupted_depth1_Lab_exact_neural_tie_only",
        ],
        "component_cycle_rank_ratio": "cycle_rank/max(1,placed_components_minus_1)",
        "dedupe": "exact_absolute_component_translations",
        "exact_tie_origin_diversity": True,
        "score_floor_terminal_only": frame.SCORE_FLOOR,
    },
    "residual": {
        "placed_rigid_core_locked": True,
        "unplaced_tiles_individual": True,
        "multi_contact_min_neighbours": frame.MIN_MULTI_CONTACTS,
        "mutual_best_cell_tile_only": True,
        "hungarian_rounds": frame.HUNGARIAN_ROUNDS,
        "identity_bonus": frame.IDENTITY_BONUS,
        "repair_passes": frame.REPAIR_PASSES,
    },
    "assembly": "original_corrupted_upright_tiles_only",
    "report_artifacts": {
        "candidate_board": "strict_flat_position_to_tile_vector",
        "candidate_canvases": "hashes_only_reconstruct_from_pinned_scene_and_board",
    },
    "restoration": {
        "name": "opencv_fast_nlm_colored",
        "h": 10,
        "scope": "candidate_once_only_after_decoder_gate",
        "RR96": "reuse_pinned_E12_final_metrics",
    },
    "staged_rules": {
        "decoder": dict(DECODER_RULE),
        "end_to_end": dict(END_TO_END_RULE),
    },
    "excluded": [
        "labels_inside_decoder",
        "component_purity_trim",
        "budget_or_threshold_sweep",
        "contact_bonus",
        "null_or_border_prior",
        "rotation_or_reflection",
        "colour_fit",
        "GPU",
        "diffusion",
        "clean_pixels_in_candidate_canvas",
        "swap_or_repair_pass",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_E17_REPORT = Path(
    "E:/pazzle_work/single_edge_frame_e17/cc192_rigid_viability_v1.json"
)
DEFAULT_REPORT = Path(
    "E:/pazzle_work/absolute_frame_e18/cc192_absolute_frame_beam_v1.json"
)


@dataclass(frozen=True)
class E18Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e17_report: Path
    report: Path


@dataclass(frozen=True)
class _State:
    scene: e12.RawScene
    right: np.ndarray
    down: np.ndarray
    clean_cache_sha256: str


def _require_e_drive(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "E:":
        raise E18ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E18ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E18ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E18 report")
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
        raise E18ContractError(
            f"E18 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e15_frame_consensus.py": source / "e15_frame_consensus.py",
        "e18_absolute_frame_beam.py": source / "e18_absolute_frame_beam.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_e17_cc192_rigid_viability.py": source
        / "eval_e17_cc192_rigid_viability.py",
        "eval_e18_absolute_frame_oracle.py": Path(__file__).resolve(),
        "imgio.py": source / "imgio.py",
        "placement_metrics.py": source / "placement_metrics.py",
        "rank96_lab_selector.py": source / "rank96_lab_selector.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _verify_e17_report(path: Path) -> Mapping[str, Any]:
    resolved = _require_e_drive(path, label="E17 report")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E17_REPORT_SHA256:
        raise E18ContractError(
            f"E17 report SHA256 mismatch: expected {EXPECTED_E17_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E17 report")
    decision = report.get("decision")
    checks = decision.get("checks") if isinstance(decision, Mapping) else None
    if (
        report.get("schema") != e17.REPORT_SCHEMA
        or report.get("experiment") != e17.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("stage") != "go_E18_absolute_frame_beam"
        or report.get("protocol") != e17.E17_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(e17.E17_PROTOCOL)
        or not isinstance(decision, Mapping)
        or decision.get("passed") is not True
        or decision.get("status") != "go_E18_absolute_frame_beam"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise E18ContractError("E17 report protocol/pass status drifted")
    return report


def _seam_is_true(seam: Sequence[int], permutation: np.ndarray) -> bool:
    if len(seam) != 4:
        raise E18ContractError("accepted cross seam is malformed")
    first, second, dy, dx = map(int, seam)
    if (dy, dx) not in {(0, 1), (1, 0)}:
        raise E18ContractError("accepted cross seam is not canonical right/down")
    if not (0 <= first < frame.NUM_TILES and 0 <= second < frame.NUM_TILES):
        raise E18ContractError("accepted cross seam tile ID is out of range")
    value = np.asarray(permutation)
    if value.shape != (frame.NUM_TILES,) or value.dtype.kind not in "iu":
        raise E18ContractError("permutation geometry/dtype drifted")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(frame.NUM_TILES)):
        raise E18ContractError("permutation is not a bijection")
    first_row, first_col = divmod(int(value[first]), frame.GRID)
    second_row, second_col = divmod(int(value[second]), frame.GRID)
    return (second_row - first_row, second_col - first_col) == (dy, dx)


def accepted_cross_seam_precision(
    seams: Sequence[Sequence[int]], permutation: np.ndarray
) -> tuple[int, int, float]:
    identities = [tuple(map(int, seam)) for seam in seams]
    if len(set(identities)) != len(identities):
        raise E18ContractError("accepted cross seams are duplicated")
    true_count = sum(_seam_is_true(seam, permutation) for seam in identities)
    return true_count, len(identities), float(true_count / max(1, len(identities)))


def evaluate_solve_only(
    scene: e12.RawScene, result: frame.SolveResult
) -> dict[str, Any]:
    board = e12._strict_board(np.asarray(result.board))
    truth = np.argsort(np.asarray(scene.permutation, dtype=np.int64))
    placement = placement_accuracy(board, truth)[0]
    neighbour, right_accuracy, down_accuracy = neighbour_accuracy(board, truth)
    solved = np.ascontiguousarray(assemble(scene.tiles_uint8, board), dtype=np.uint8)
    solve_ssim = float(
        sk_ssim(scene.target_uint8, solved, channel_axis=2, data_range=255)
    )
    diagnostics = asdict(result.diagnostics)
    seams = diagnostics["accepted_cross_seams"]
    true_seams, seam_count, seam_precision = accepted_cross_seam_precision(
        seams, scene.permutation
    )
    diagnostics["true_accepted_cross_seams"] = true_seams
    diagnostics["accepted_cross_seam_precision"] = seam_precision
    if seam_count != int(diagnostics["unique_physical_cross_seams"]):
        raise E18ContractError("accepted cross seam count drifted")
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right_accuracy),
        "down": float(down_accuracy),
        "solve_only_ssim": solve_ssim,
        "board_sha256": e12.array_sha256(board.astype(np.int64, copy=False)),
        "solved_corrupted_canvas_sha256": e12.array_sha256(solved),
        "diagnostics": diagnostics,
        "_board": board,
        "_solved": solved,
    }


def _validate_eight_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E18ContractError(f"{label} requires exactly eight rows")
    images = [int(row.get("image", -1)) for row in rows]
    if len(set(images)) != len(images) or tuple(sorted(images)) != e12.CALIBRATION_IDS:
        raise E18ContractError(f"{label} image IDs drifted")


def _finite_float(
    value: object, *, label: str, minimum: float, maximum: float
) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise E18ContractError(f"{label} is not numeric") from exc
    if not np.isfinite(observed) or not minimum <= observed <= maximum:
        raise E18ContractError(f"{label} is outside [{minimum}, {maximum}]")
    return observed


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_decoder_row_integrity(row: Mapping[str, Any]) -> None:
    image = int(row.get("image", -1))
    if row.get("arm") != "E18_absolute_frame_beam":
        raise E18ContractError(f"candidate arm drifted for image {image}")
    for key in ("placement", "neighbour", "right", "down"):
        _finite_float(row.get(key), label=f"image {image} {key}", minimum=0.0, maximum=1.0)
    _finite_float(
        row.get("solve_only_ssim"),
        label=f"image {image} solve-only SSIM",
        minimum=-1.0,
        maximum=1.0,
    )
    _finite_float(
        row.get("solver_seconds"),
        label=f"image {image} solver seconds",
        minimum=0.0,
        maximum=float("inf"),
    )
    for key in (
        "board_sha256",
        "solved_corrupted_canvas_sha256",
        "clean_score_cache_sha256",
    ):
        if not _is_sha256(row.get(key)):
            raise E18ContractError(f"image {image} {key} is not a SHA256 digest")
    try:
        board = e12._strict_board(np.asarray(row.get("board"), dtype=np.int64))
    except (TypeError, ValueError, e12.OracleContractError) as exc:
        raise E18ContractError(f"image {image} stored board is invalid") from exc
    if e12.array_sha256(board.astype(np.int64, copy=False)) != row.get(
        "board_sha256"
    ):
        raise E18ContractError(f"image {image} stored board hash drifted")

    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise E18ContractError(f"image {image} decoder diagnostics are malformed")
    try:
        placed_components = int(diagnostics["rigid_components_placed"])
        contacts = int(diagnostics["unique_component_contacts"])
        cycle_rank = int(diagnostics["component_cycle_rank"])
        rigid_tiles = int(diagnostics["rigid_tiles_placed"])
        seam_count = int(diagnostics["unique_physical_cross_seams"])
        true_seams = int(diagnostics["true_accepted_cross_seams"])
        translations = diagnostics["translations"]
        root_origin = tuple(map(int, diagnostics["root_origin"]))
        seams = diagnostics["accepted_cross_seams"]
    except (KeyError, TypeError, ValueError) as exc:
        raise E18ContractError(f"image {image} decoder diagnostics are incomplete") from exc
    if not isinstance(translations, (list, tuple)) or len(translations) != placed_components:
        raise E18ContractError(f"image {image} translation count drifted")
    try:
        translation_tuples = tuple(tuple(map(int, value)) for value in translations)
    except (TypeError, ValueError) as exc:
        raise E18ContractError(f"image {image} translations are malformed") from exc
    if (
        any(len(value) != 3 for value in translation_tuples)
        or len({value[0] for value in translation_tuples}) != len(translation_tuples)
        or tuple(sorted(translation_tuples)) != translation_tuples
    ):
        raise E18ContractError(f"image {image} translations are not canonical")
    translation_map = {value[0]: (value[1], value[2]) for value in translation_tuples}
    if translation_map.get(0) != root_origin:
        raise E18ContractError(f"image {image} root origin/translation drifted")
    if placed_components < 1 or contacts < placed_components - 1:
        raise E18ContractError("partial component graph is not connected")
    expected_cycle = max(0, contacts - placed_components + 1)
    if cycle_rank != expected_cycle:
        raise E18ContractError(f"image {image} component cycle rank drifted")
    expected_cycle_ratio = float(cycle_rank / max(1, placed_components - 1))
    if not np.isclose(
        float(diagnostics["component_cycle_rank_ratio"]),
        expected_cycle_ratio,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise E18ContractError(f"image {image} component cycle ratio drifted")
    if not 1 <= rigid_tiles <= frame.NUM_TILES:
        raise E18ContractError(f"image {image} rigid tile count drifted")
    if not np.isclose(
        float(diagnostics["rigid_coverage"]),
        rigid_tiles / frame.NUM_TILES,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise E18ContractError(f"image {image} rigid coverage drifted")
    if not isinstance(seams, (list, tuple)):
        raise E18ContractError(f"image {image} accepted cross seams are malformed")
    try:
        seam_tuples = tuple(tuple(map(int, value)) for value in seams)
    except (TypeError, ValueError) as exc:
        raise E18ContractError(f"image {image} accepted cross seams are malformed") from exc
    if (
        any(
            len(value) != 4
            or value[2:] not in {(0, 1), (1, 0)}
            or not 0 <= value[0] < frame.NUM_TILES
            or not 0 <= value[1] < frame.NUM_TILES
            for value in seam_tuples
        )
        or len(set(seam_tuples)) != len(seam_tuples)
        or len(seam_tuples) != seam_count
        or not 0 <= true_seams <= seam_count
    ):
        raise E18ContractError(f"image {image} accepted cross seam evidence drifted")
    expected_precision = float(true_seams / max(1, seam_count))
    if not np.isclose(
        float(diagnostics["accepted_cross_seam_precision"]),
        expected_precision,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise E18ContractError(f"image {image} accepted cross seam precision drifted")
    layouts = int(diagnostics["absolute_layouts_retained"])
    if not 1 <= layouts <= frame.ABSOLUTE_LAYOUTS:
        raise E18ContractError("absolute layout budget drifted")
    if int(diagnostics["root_origins_evaluated"]) < 1:
        raise E18ContractError("no legal root origin was evaluated")
    if int(diagnostics["hungarian_rounds"]) != frame.HUNGARIAN_ROUNDS:
        raise E18ContractError("residual Hungarian rounds drifted")


def summarize_decoder(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_eight_rows(rows, label="decoder stage")
    for row in rows:
        _validate_decoder_row_integrity(row)
    diagnostics = [row.get("diagnostics") for row in rows]
    if not all(isinstance(value, Mapping) for value in diagnostics):
        raise E18ContractError("decoder diagnostics are malformed")
    mean_solve = float(np.mean([float(row["solve_only_ssim"]) for row in rows]))
    precisions = [
        float(value["accepted_cross_seam_precision"]) for value in diagnostics
    ]
    return {
        "images": len(rows),
        "strict_bijection_scenes": len(rows),
        "expansion_cap_hit_scenes": int(
            sum(bool(value["expansion_cap_hit"]) for value in diagnostics)
        ),
        "mean_rigid_coverage": float(
            np.mean([float(value["rigid_coverage"]) for value in diagnostics])
        ),
        "mean_accepted_cross_seam_precision": float(np.mean(precisions)),
        "worst_accepted_cross_seam_precision": float(min(precisions)),
        "mean_component_cycle_rank_ratio": float(
            np.mean(
                [float(value["component_cycle_rank_ratio"]) for value in diagnostics]
            )
        ),
        "mean_placement": float(np.mean([float(row["placement"]) for row in rows])),
        "mean_neighbour": float(np.mean([float(row["neighbour"]) for row in rows])),
        "mean_solve_only_ssim": mean_solve,
        "candidate_minus_rr96_mean_solve_ssim": float(
            mean_solve - EXPECTED_RR_MEAN_SOLVE_SSIM
        ),
        "mean_proposal_evaluations": float(
            np.mean([int(value["proposal_evaluations"]) for value in diagnostics])
        ),
        "mean_rigid_components": float(
            np.mean([int(value["rigid_components_placed"]) for value in diagnostics])
        ),
    }


def decoder_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "expansion_cap_hit_scenes": int(summary["expansion_cap_hit_scenes"]),
        "strict_bijection_scenes": int(summary["strict_bijection_scenes"]),
        "mean_rigid_coverage": float(summary["mean_rigid_coverage"]),
        "mean_accepted_cross_seam_precision": float(
            summary["mean_accepted_cross_seam_precision"]
        ),
        "worst_accepted_cross_seam_precision": float(
            summary["worst_accepted_cross_seam_precision"]
        ),
        "mean_component_cycle_rank_ratio": float(
            summary["mean_component_cycle_rank_ratio"]
        ),
        "mean_placement": float(summary["mean_placement"]),
        "mean_neighbour": float(summary["mean_neighbour"]),
        "candidate_minus_rr96_mean_solve_ssim": float(
            summary["candidate_minus_rr96_mean_solve_ssim"]
        ),
    }
    checks = {
        "expansion_cap_hit_scenes": observed["expansion_cap_hit_scenes"]
        <= int(DECODER_RULE["expansion_cap_hit_scenes_max"]),
        "strict_bijection_scenes": observed["strict_bijection_scenes"]
        == int(DECODER_RULE["strict_bijection_scenes"]),
        "mean_rigid_coverage": observed["mean_rigid_coverage"]
        >= float(DECODER_RULE["mean_rigid_coverage_min"]),
        "mean_accepted_cross_seam_precision": observed[
            "mean_accepted_cross_seam_precision"
        ]
        >= float(DECODER_RULE["mean_accepted_cross_seam_precision_min"]),
        "worst_accepted_cross_seam_precision": observed[
            "worst_accepted_cross_seam_precision"
        ]
        >= float(DECODER_RULE["worst_accepted_cross_seam_precision_min"]),
        "mean_component_cycle_rank_ratio": observed[
            "mean_component_cycle_rank_ratio"
        ]
        >= float(DECODER_RULE["mean_component_cycle_rank_ratio_min"]),
        "mean_placement": observed["mean_placement"]
        >= float(DECODER_RULE["mean_placement_min"]),
        "mean_neighbour": observed["mean_neighbour"]
        >= float(DECODER_RULE["mean_neighbour_min"]),
        "candidate_minus_rr96_mean_solve_ssim": observed[
            "candidate_minus_rr96_mean_solve_ssim"
        ]
        >= float(DECODER_RULE["candidate_minus_rr96_mean_solve_ssim_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "go_nlm" if passed else "kill_decoder",
        "passed": passed,
        "thresholds": dict(DECODER_RULE),
        "observed": observed,
        "checks": checks,
    }


def end_to_end_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    solve = summary["metrics"]["solve_only_ssim"]
    final = summary["metrics"]["final_ssim"]
    observed = {
        "mean_solve_ssim_delta": float(solve["mean_delta"]),
        "mean_final_ssim_delta": float(final["mean_delta"]),
        "final_wins": int(final["wins"]),
        "worst_final_delta": float(final["worst_delta"]),
    }
    checks = {
        "mean_solve_ssim_delta": observed["mean_solve_ssim_delta"]
        >= float(END_TO_END_RULE["candidate_minus_rr96_mean_solve_ssim_min"]),
        "mean_final_ssim_delta": observed["mean_final_ssim_delta"]
        >= float(END_TO_END_RULE["candidate_minus_rr96_mean_final_ssim_min"]),
        "final_wins": observed["final_wins"]
        >= int(END_TO_END_RULE["candidate_minus_rr96_final_wins_min"]),
        "worst_final_delta": observed["worst_final_delta"]
        >= float(END_TO_END_RULE["candidate_minus_rr96_worst_final_delta_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "go_raw_adaptation_confirmation" if passed else "kill_end_to_end",
        "passed": passed,
        "thresholds": dict(END_TO_END_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_oracle_changed_decoder_only_not_deployable",
    }


def _serialisable_decoder_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _validate_complete_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_digest: str,
    rr_rows: Mapping[int, Mapping[str, Any]],
    scenes: Sequence[e12.RawScene],
) -> None:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("schema") != REPORT_SCHEMA
        or report.get("experiment") != EXPERIMENT
        or report.get("status") != "complete"
        or report.get("protocol") != E18_PROTOCOL
        or report.get("protocol_sha256") != e12.canonical_digest(E18_PROTOCOL)
        or report.get("run_contract") != contract
        or report.get("run_contract_sha256") != contract_digest
    ):
        raise E18ContractError("existing E18 complete report contract drifted")
    _finite_float(
        report.get("runtime_seconds"),
        label="existing E18 runtime",
        minimum=0.0,
        maximum=float("inf"),
    )
    rows = report.get("rows")
    if not isinstance(rows, Mapping):
        raise E18ContractError("existing E18 rows are missing")
    rr = rows.get("RR96")
    candidate = rows.get("candidate")
    if not isinstance(rr, list) or not isinstance(candidate, list):
        raise E18ContractError("existing E18 row groups are malformed")
    _validate_eight_rows(rr, label="existing RR96")
    _validate_eight_rows(candidate, label="existing candidate")
    if report.get("completed_decoder_images") != list(e12.CALIBRATION_IDS):
        raise E18ContractError("existing E18 decoder completion IDs drifted")
    rr_by_image = {int(row["image"]): row for row in rr}
    candidate_by_image = {int(row["image"]): row for row in candidate}
    scene_by_image = {int(scene.image_id): scene for scene in scenes}
    if tuple(sorted(scene_by_image)) != e12.CALIBRATION_IDS:
        raise E18ContractError("existing E18 validation scenes drifted")
    clean_caches = contract.get("clean_score_caches")
    if not isinstance(clean_caches, Mapping):
        raise E18ContractError("existing E18 clean-cache contract is malformed")
    for image in e12.CALIBRATION_IDS:
        for key in (
            *e14.BOARD_METRICS,
            "objective",
            "board_sha256",
            "solved_corrupted_canvas_sha256",
            "restored_canvas_sha256",
        ):
            if rr_by_image[image].get(key) != rr_rows[image].get(key):
                raise E18ContractError(
                    f"existing E18 RR96 {key} drifted for image {image}"
                )
        cache_record = clean_caches.get(str(image))
        if (
            not isinstance(cache_record, Mapping)
            or candidate_by_image[image].get("clean_score_cache_sha256")
            != cache_record.get("sha256")
        ):
            raise E18ContractError(
                f"existing E18 clean-cache digest drifted for image {image}"
            )
        candidate_row = candidate_by_image[image]
        diagnostics = candidate_row["diagnostics"]
        true_seams, seam_count, seam_precision = accepted_cross_seam_precision(
            diagnostics["accepted_cross_seams"],
            scene_by_image[image].permutation,
        )
        if (
            true_seams != int(diagnostics["true_accepted_cross_seams"])
            or seam_count != int(diagnostics["unique_physical_cross_seams"])
            or not np.isclose(
                seam_precision,
                float(diagnostics["accepted_cross_seam_precision"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise E18ContractError(
                f"existing E18 seam truth drifted for image {image}"
            )
    computed_decoder = summarize_decoder(candidate)
    computed_decoder_decision = decoder_decision(computed_decoder)
    if report.get("decoder_summary") != computed_decoder:
        raise E18ContractError("existing E18 decoder summary drifted")
    decisions = report.get("decisions")
    if not isinstance(decisions, Mapping) or decisions.get(
        "decoder"
    ) != computed_decoder_decision:
        raise E18ContractError("existing E18 decoder decision drifted")
    if not computed_decoder_decision["passed"]:
        if (
            report.get("stage") != "kill_decoder"
            or report.get("completed_nlm_images") != []
            or "comparison" in report
            or decisions.get("end_to_end") != {"status": "not_run"}
            or any(
                "final_ssim" in row or "restored_canvas_sha256" in row
                for row in candidate
            )
        ):
            raise E18ContractError("existing E18 decoder-kill stage drifted")
        return
    if report.get("completed_nlm_images") != list(e12.CALIBRATION_IDS):
        raise E18ContractError("existing E18 NLM completion IDs drifted")
    if not all(
        "final_ssim" in row and _is_sha256(row.get("restored_canvas_sha256"))
        for row in candidate
    ):
        raise E18ContractError("existing E18 final metrics are incomplete")
    for row in candidate:
        _finite_float(
            row["final_ssim"],
            label=f"image {int(row['image'])} final SSIM",
            minimum=-1.0,
            maximum=1.0,
        )
    comparison = dict(e14.paired_summary(candidate, rr_rows))
    comparison["candidate_arm"] = "E18_absolute_frame_beam"
    computed_end = end_to_end_decision(comparison)
    if report.get("comparison") != comparison:
        raise E18ContractError("existing E18 comparison drifted")
    if decisions.get("end_to_end") != computed_end:
        raise E18ContractError("existing E18 end-to-end decision drifted")
    if report.get("stage") != computed_end["status"]:
        raise E18ContractError("existing E18 terminal stage drifted")


def run_discovery(paths: E18Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E18 report")
    if report_path.suffix.lower() != ".json":
        raise E18ContractError("E18 report must be a .json file")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e17_report_path = _require_e_drive(paths.e17_report, label="E17 report")
    calibration_path = paths.calibration_report.resolve()
    if report_path in {e12_report_path, e17_report_path, calibration_path}:
        raise E18ContractError("E18 report must not overwrite an input")
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(
        clean_cache_dir
    ):
        raise E18ContractError("E18 report must not be written inside an input cache")

    e12_report, _calibration, scenes = e14.load_verified_e12_inputs(
        e14.E14Paths(
            raw_cache_dir=raw_cache_dir,
            calibration_report=calibration_path,
            e12_report=e12_report_path,
            report=e14.DEFAULT_REPORT,
        )
    )
    e17_report = _verify_e17_report(e17_report_path)
    rr_rows = e14._e12_rr_rows(e12_report)
    rr_verification = e14.verify_rr_means(rr_rows)
    clean_records = e14._clean_cache_records(e12_report)
    contract = {
        "protocol_sha256": e12.canonical_digest(E18_PROTOCOL),
        "e12_report": {
            "path": str(e12_report_path),
            "sha256": EXPECTED_E12_REPORT_SHA256,
        },
        "e17_report": {
            "path": str(e17_report_path),
            "sha256": EXPECTED_E17_REPORT_SHA256,
            "run_contract_sha256": str(e17_report["run_contract_sha256"]),
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
        existing = _load_json(report_path, label="existing E18 report")
        if existing.get("run_contract_sha256") != contract_digest:
            raise E18ContractError("existing E18 report belongs to different bytes")
        if existing.get("run_contract") != contract:
            raise E18ContractError("existing E18 report contract payload drifted")
        if existing.get("status") == "complete":
            _validate_complete_report(
                existing,
                contract=contract,
                contract_digest=contract_digest,
                rr_rows=rr_rows,
                scenes=scenes,
            )
            return existing

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "decoder",
        "protocol": E18_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E18_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rr_reproducibility": rr_verification,
        "rows": {"RR96": [], "candidate": []},
        "completed_decoder_images": [],
        "completed_nlm_images": [],
        "decisions": {
            "decoder": {"status": "not_run"},
            "end_to_end": {"status": "not_run"},
        },
    }
    _atomic_write_json(report_path, output)
    states: list[_State] = []
    decoder_rows: list[dict[str, Any]] = []
    try:
        for scene in scenes:
            before = rr_rows[scene.image_id]
            _board, _objective, rr_seconds = e14._replay_rr96(scene, before)
            output["rows"]["RR96"].append(
                {
                    "image": int(scene.image_id),
                    "validation_name": str(scene.validation_name),
                    **{key: before[key] for key in e14.BOARD_METRICS},
                    "objective": float(before["objective"]),
                    "board_sha256": str(before["board_sha256"]),
                    "solved_corrupted_canvas_sha256": str(
                        before["solved_corrupted_canvas_sha256"]
                    ),
                    "restored_canvas_sha256": str(before["restored_canvas_sha256"]),
                    "solver_replay_seconds": rr_seconds,
                    "source": "exact_E12_RR_record_no_second_NLM_call",
                }
            )
            clean_cache = e14._load_cc_cache(
                scene, e12_report, clean_records[scene.image_id]
            )
            right, down = e12.dense_from_graph(
                clean_cache.cc_candidates, clean_cache.cc_scores
            )
            states.append(
                _State(
                    scene=scene,
                    right=right,
                    down=down,
                    clean_cache_sha256=clean_cache.sha256,
                )
            )

        for state in states:
            solve_started = time.perf_counter()
            result = frame.solve_absolute_frame(
                state.right, state.down, state.scene.tiles_uint8
            )
            solver_seconds = time.perf_counter() - solve_started
            evaluated = evaluate_solve_only(state.scene, result)
            decoder_rows.append(evaluated)
            output["rows"]["candidate"].append(
                {
                    "image": int(state.scene.image_id),
                    "validation_name": str(state.scene.validation_name),
                    "arm": "E18_absolute_frame_beam",
                    **_serialisable_decoder_row(evaluated),
                    "board": evaluated["_board"].astype(np.int64, copy=False).tolist(),
                    "solver_seconds": float(solver_seconds),
                    "clean_score_cache_sha256": state.clean_cache_sha256,
                }
            )
            output["completed_decoder_images"].append(int(state.scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)

        decoder_summary = summarize_decoder(output["rows"]["candidate"])
        decoder_gate = decoder_decision(decoder_summary)
        output["decoder_summary"] = decoder_summary
        output["decisions"]["decoder"] = decoder_gate
        if not decoder_gate["passed"]:
            output["status"] = "complete"
            output["stage"] = "kill_decoder"
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
            return output

        output["stage"] = "nlm_and_end_to_end"
        _atomic_write_json(report_path, output)
        candidate_by_image = {
            int(row["image"]): row for row in output["rows"]["candidate"]
        }
        for state, evaluated in zip(states, decoder_rows):
            restored = np.asarray(e12.fixed_nlm(evaluated["_solved"].copy()))
            if restored.shape != (e12.IMG, e12.IMG, 3) or restored.dtype != np.uint8:
                raise E18ContractError("fixed NLM returned invalid geometry/dtype")
            final_ssim = float(
                sk_ssim(
                    state.scene.target_uint8,
                    restored,
                    channel_axis=2,
                    data_range=255,
                )
            )
            row = candidate_by_image[state.scene.image_id]
            row["final_ssim"] = final_ssim
            row["restored_canvas_sha256"] = e12.array_sha256(restored)
            output["completed_nlm_images"].append(int(state.scene.image_id))
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)

        comparison = dict(e14.paired_summary(output["rows"]["candidate"], rr_rows))
        comparison["candidate_arm"] = "E18_absolute_frame_beam"
        end_to_end = end_to_end_decision(comparison)
        output["comparison"] = comparison
        output["decisions"]["end_to_end"] = end_to_end
        output["status"] = "complete"
        output["stage"] = end_to_end["status"]
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
        description="Run fixed CPU-only E18 absolute-frame clean oracle."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e17-report", type=Path, default=DEFAULT_E17_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_discovery(
        E18Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e17_report=args.e17_report,
            report=args.report,
        )
    )
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "stage": result["stage"],
                "decoder": result["decisions"]["decoder"]["status"],
                "end_to_end": result["decisions"]["end_to_end"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
