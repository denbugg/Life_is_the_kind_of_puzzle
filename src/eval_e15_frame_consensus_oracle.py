"""Staged CPU-only E15 frame-consensus clean-oracle discovery.

The evaluator is fail-closed.  It first verifies the byte-pinned E12/E14
provenance and measures the predeclared CC96/two-vote structure.  It creates a
candidate board only if that stage passes, and calls NLM only if the decoder
stage also passes.  There is one candidate configuration and no sweep.
"""
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

import e15_frame_consensus as frame
import eval_clean_score_oracle as e12
import eval_e14_cc192_discovery as e14
from imgio import assemble
from placement_metrics import neighbour_accuracy, placement_accuracy


class E15ContractError(RuntimeError):
    """The frozen E15 protocol, code, or an input byte drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-e15-frame-consensus-clean-oracle-report-v1"
EXPERIMENT = "e15_cc96_cc192_two_vote_frame_consensus_v1"
EXPECTED_E12_REPORT_SHA256 = (
    "16ceecfea99e006a1126b17d7d58fb5d188ec694c6a5097310dfe021bd2f901a"
)
EXPECTED_E14_REPORT_SHA256 = (
    "eb6c6c00aeaa827a6179d48af6fd17f0a203dbb0881dd35aeecdf5853b9b06eb"
)
EXPECTED_RR_MEAN_SOLVE_SSIM = 0.094607964147414
EXPECTED_RR_MEAN_FINAL_SSIM = 0.15930445310452002

STRUCTURAL_RULE: dict[str, float | int] = {
    "selected_cc96_claims_each": 96,
    "mean_cc96_edge_precision_min": 0.98,
    "mean_cc96_component_coverage_min": 0.25,
    "mean_two_vote_hypothesis_precision_min": 0.98,
    "worst_two_vote_hypothesis_precision_min": 0.90,
    "mean_relation_supported_tile_coverage_min": 0.15,
}
DECODER_RULE: dict[str, float | int | bool] = {
    "expansion_cap_hit_allowed": False,
    "strict_bijection_scenes": 8,
    "mean_rigid_coverage_min": 0.20,
    "all_non_seed_attachments_two_seam": True,
    "mean_placement_min": 0.02,
    "mean_neighbour_min": 0.20,
}
END_TO_END_RULE: dict[str, float | int] = {
    "candidate_minus_rr96_mean_solve_ssim_min": 0.010,
    "candidate_minus_rr96_mean_final_ssim_min": 0.015,
    "candidate_minus_rr96_final_wins_min": 6,
    "candidate_minus_rr96_worst_final_delta_min": -0.020,
}
EXPECTED_RUNTIME_PROVENANCE = {
    **dict(e14.EXPECTED_RUNTIME_PROVENANCE),
    "scipy": "1.16.2",
}

E15_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-e15-cc96-cc192-two-vote-frame-consensus-v1",
    "role": "opened_E12_changed_decoder_clean_oracle_not_deployable",
    "calibration_ids": list(e12.CALIBRATION_IDS),
    "inputs": {
        "e12_report_sha256": EXPECTED_E12_REPORT_SHA256,
        "e14_report_sha256": EXPECTED_E14_REPORT_SHA256,
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
    },
    "seed": {
        "builder": "solve_buddies.build_buddies_components_exact",
        "max_edges": frame.SEED_MAX_EDGES,
        "min_margin": 0.0,
        "geometry": "rigid_CC96",
    },
    "translation_votes": {
        "selector": "solve_buddies._candidate_edges_exact",
        "max_edges": frame.VOTE_MAX_EDGES,
        "min_margin": 0.0,
        "distinct_physical_seams_required": frame.MIN_DISTINCT_SEAMS,
        "single_claim_attachment": False,
    },
    "search": {
        "beam_width": frame.BEAM_WIDTH,
        "proposals_per_state": frame.PROPOSALS_PER_STATE,
        "relative_layouts": frame.RELATIVE_LAYOUTS,
        "absolute_layouts_per_scene": frame.ABSOLUTE_LAYOUTS,
        "expansion_cap_per_scene": frame.EXPANSION_CAP,
        "score_floor": frame.SCORE_FLOOR,
        "first_component_origin": "temporary_gauge_then_legal_absolute_origins",
        "lab": "exact_lexicographic_tie_break_only",
        "null_weight": frame.NULL_WEIGHT,
        "null_or_border_model": False,
    },
    "residual": {
        "multi_contact_min_neighbours": frame.MIN_MULTI_CONTACTS,
        "mutual_best_cell_tile_only": True,
        "hungarian_rounds": frame.HUNGARIAN_ROUNDS,
        "rigid_core_locked": True,
        "identity_bonus": 0.0,
        "repair_passes": frame.REPAIR_PASSES,
    },
    "assembly": "original_corrupted_upright_tiles_only",
    "restoration": {
        "name": "opencv_fast_nlm_colored",
        "h": 10,
        "scope": "candidate_once_only_after_decoder_gate",
        "RR96": "reuse_pinned_E12_final_metrics",
    },
    "staged_rules": {
        "structure": dict(STRUCTURAL_RULE),
        "decoder": dict(DECODER_RULE),
        "end_to_end": dict(END_TO_END_RULE),
    },
    "excluded": [
        "budget_sweep",
        "threshold_sweep",
        "GPU",
        "denoiser_training",
        "hard_or_soft_null_prior",
        "rotation_or_reflection",
        "swap_or_repair_pass",
    ],
    "runtime_provenance": EXPECTED_RUNTIME_PROVENANCE,
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CACHE_DIR = e14.DEFAULT_RAW_CACHE_DIR
DEFAULT_CALIBRATION_REPORT = e14.DEFAULT_CALIBRATION_REPORT
DEFAULT_E12_REPORT = e14.DEFAULT_E12_REPORT
DEFAULT_E14_REPORT = e14.DEFAULT_REPORT
DEFAULT_REPORT = Path(
    "E:/pazzle_work/frame_consensus_e15/frame_consensus_clean_oracle_v1.json"
)


@dataclass(frozen=True)
class E15Paths:
    raw_cache_dir: Path
    calibration_report: Path
    e12_report: Path
    e14_report: Path
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
        raise E15ContractError(f"{label} must stay on E:, got {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E15ContractError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise E15ContractError(f"{label} root is not an object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = _require_e_drive(path, label="E15 report")
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
        "scipy": str(scipy.__version__),
        "torch": str(e12.torch.__version__),
        "execution": "CPU_only",
    }
    if observed != EXPECTED_RUNTIME_PROVENANCE:
        raise E15ContractError(
            f"E15 runtime drifted: expected {EXPECTED_RUNTIME_PROVENANCE}, got {observed}"
        )
    return observed


def _source_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "e15_frame_consensus.py": source / "e15_frame_consensus.py",
        "eval_e15_frame_consensus_oracle.py": Path(__file__).resolve(),
        "eval_e14_cc192_discovery.py": source / "eval_e14_cc192_discovery.py",
        "eval_clean_score_oracle.py": source / "eval_clean_score_oracle.py",
        "imgio.py": source / "imgio.py",
        "placement_metrics.py": source / "placement_metrics.py",
        "rank96_lab_selector.py": source / "rank96_lab_selector.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    return {name: e12.sha256_file(path) for name, path in sorted(paths.items())}


def _verify_e14_report(path: Path) -> Mapping[str, Any]:
    resolved = _require_e_drive(path, label="E14 report")
    digest = e12.sha256_file(resolved)
    if digest != EXPECTED_E14_REPORT_SHA256:
        raise E15ContractError(
            f"E14 report SHA256 mismatch: expected {EXPECTED_E14_REPORT_SHA256}, got {digest}"
        )
    report = _load_json(resolved, label="E14 report")
    if (
        report.get("schema") != e14.REPORT_SCHEMA
        or report.get("experiment") != e14.EXPERIMENT
        or report.get("status") != "complete"
        or report.get("stage") != "kill_cc192"
        or report.get("protocol") != e14.E14_PROTOCOL
    ):
        raise E15ContractError("E14 report protocol/status drifted")
    return report


def _claim_is_true(claim: frame.SeamClaim, permutation: np.ndarray) -> bool:
    anchor_row, anchor_col = divmod(int(permutation[claim.anchor]), frame.GRID)
    target_row, target_col = divmod(int(permutation[claim.target]), frame.GRID)
    return (target_row - anchor_row, target_col - anchor_col) == (
        claim.dy,
        claim.dx,
    )


def measure_structure(
    components: Sequence[frame.Component],
    claims96: Sequence[frame.SeamClaim],
    hypotheses: Sequence[frame.TranslationHypothesis],
    permutation: np.ndarray,
) -> dict[str, Any]:
    value = np.asarray(permutation)
    if value.shape != (frame.NUM_TILES,) or value.dtype.kind not in "iu":
        raise E15ContractError("permutation geometry/dtype drifted")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(frame.NUM_TILES)):
        raise E15ContractError("permutation is not a bijection")
    true_claims = sum(_claim_is_true(claim, value) for claim in claims96)
    nontrivial = [component for component in components if component.size >= 2]
    seed_tiles = {tile for component in nontrivial for tile in component.tiles}
    hypothesis_truth = [
        all(_claim_is_true(claim, value) for claim in hypothesis.claims)
        for hypothesis in hypotheses
    ]
    relation_components = {
        component_id
        for hypothesis in hypotheses
        for component_id in (
            hypothesis.left_component,
            hypothesis.right_component,
        )
    }
    by_id = {component.component_id: component for component in components}
    relation_tiles = {
        tile
        for component_id in relation_components
        for tile in by_id[component_id].tiles
    }
    return {
        "selected_cc96_claims": int(len(claims96)),
        "true_cc96_claims": int(true_claims),
        "cc96_edge_precision": float(true_claims / max(1, len(claims96))),
        "cc96_component_count": int(len(nontrivial)),
        "cc96_component_tiles": int(len(seed_tiles)),
        "cc96_component_coverage": float(len(seed_tiles) / frame.NUM_TILES),
        "two_vote_hypotheses": int(len(hypotheses)),
        "true_two_vote_hypotheses": int(sum(hypothesis_truth)),
        "two_vote_hypothesis_precision": float(
            sum(hypothesis_truth) / max(1, len(hypotheses))
        ),
        "relation_supported_tiles": int(len(relation_tiles)),
        "relation_supported_tile_coverage": float(
            len(relation_tiles) / frame.NUM_TILES
        ),
    }


def summarize_structure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E15ContractError("structure stage requires exactly eight rows")
    return {
        "images": len(rows),
        "selected_cc96_claims_each": sorted(
            {int(row["selected_cc96_claims"]) for row in rows}
        ),
        "mean_cc96_edge_precision": float(
            np.mean([float(row["cc96_edge_precision"]) for row in rows])
        ),
        "worst_cc96_edge_precision": float(
            min(float(row["cc96_edge_precision"]) for row in rows)
        ),
        "mean_cc96_component_coverage": float(
            np.mean([float(row["cc96_component_coverage"]) for row in rows])
        ),
        "mean_two_vote_hypothesis_precision": float(
            np.mean([float(row["two_vote_hypothesis_precision"]) for row in rows])
        ),
        "worst_two_vote_hypothesis_precision": float(
            min(float(row["two_vote_hypothesis_precision"]) for row in rows)
        ),
        "mean_relation_supported_tile_coverage": float(
            np.mean(
                [float(row["relation_supported_tile_coverage"]) for row in rows]
            )
        ),
        "mean_two_vote_hypotheses": float(
            np.mean([int(row["two_vote_hypotheses"]) for row in rows])
        ),
    }


def structural_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "selected_cc96_claims_each": list(summary["selected_cc96_claims_each"]),
        "mean_cc96_edge_precision": float(summary["mean_cc96_edge_precision"]),
        "mean_cc96_component_coverage": float(
            summary["mean_cc96_component_coverage"]
        ),
        "mean_two_vote_hypothesis_precision": float(
            summary["mean_two_vote_hypothesis_precision"]
        ),
        "worst_two_vote_hypothesis_precision": float(
            summary["worst_two_vote_hypothesis_precision"]
        ),
        "mean_relation_supported_tile_coverage": float(
            summary["mean_relation_supported_tile_coverage"]
        ),
    }
    checks = {
        "selected_cc96_claims_each": observed["selected_cc96_claims_each"]
        == [int(STRUCTURAL_RULE["selected_cc96_claims_each"])],
        "mean_cc96_edge_precision": observed["mean_cc96_edge_precision"]
        >= float(STRUCTURAL_RULE["mean_cc96_edge_precision_min"]),
        "mean_cc96_component_coverage": observed["mean_cc96_component_coverage"]
        >= float(STRUCTURAL_RULE["mean_cc96_component_coverage_min"]),
        "mean_two_vote_hypothesis_precision": observed[
            "mean_two_vote_hypothesis_precision"
        ]
        >= float(STRUCTURAL_RULE["mean_two_vote_hypothesis_precision_min"]),
        "worst_two_vote_hypothesis_precision": observed[
            "worst_two_vote_hypothesis_precision"
        ]
        >= float(STRUCTURAL_RULE["worst_two_vote_hypothesis_precision_min"]),
        "mean_relation_supported_tile_coverage": observed[
            "mean_relation_supported_tile_coverage"
        ]
        >= float(STRUCTURAL_RULE["mean_relation_supported_tile_coverage_min"]),
    }
    passed = all(checks.values())
    return {
        "status": "go_decoder" if passed else "kill_structure",
        "passed": passed,
        "thresholds": dict(STRUCTURAL_RULE),
        "observed": observed,
        "checks": checks,
    }


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
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right_accuracy),
        "down": float(down_accuracy),
        "solve_only_ssim": solve_ssim,
        "board_sha256": e12.array_sha256(board.astype(np.int64, copy=False)),
        "solved_corrupted_canvas_sha256": e12.array_sha256(solved),
        "diagnostics": asdict(result.diagnostics),
        "_board": board,
        "_solved": solved,
    }


def summarize_decoder(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(e12.CALIBRATION_IDS):
        raise E15ContractError("decoder stage requires exactly eight rows")
    diagnostics = [row["diagnostics"] for row in rows]
    if not all(isinstance(value, Mapping) for value in diagnostics):
        raise E15ContractError("decoder diagnostics are malformed")
    supports = [
        int(support)
        for value in diagnostics
        for support in value["non_seed_attachment_supports"]
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
        "minimum_non_seed_attachment_support": int(min(supports, default=2)),
        "all_non_seed_attachments_two_seam": bool(
            all(support >= frame.MIN_DISTINCT_SEAMS for support in supports)
        ),
        "mean_placement": float(np.mean([float(row["placement"]) for row in rows])),
        "mean_neighbour": float(np.mean([float(row["neighbour"]) for row in rows])),
        "mean_solve_only_ssim": float(
            np.mean([float(row["solve_only_ssim"]) for row in rows])
        ),
        "mean_expansions": float(
            np.mean([int(value["expansions"]) for value in diagnostics])
        ),
    }


def decoder_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "expansion_cap_hit_scenes": int(summary["expansion_cap_hit_scenes"]),
        "strict_bijection_scenes": int(summary["strict_bijection_scenes"]),
        "mean_rigid_coverage": float(summary["mean_rigid_coverage"]),
        "all_non_seed_attachments_two_seam": bool(
            summary["all_non_seed_attachments_two_seam"]
        ),
        "mean_placement": float(summary["mean_placement"]),
        "mean_neighbour": float(summary["mean_neighbour"]),
    }
    checks = {
        "expansion_cap_not_hit": observed["expansion_cap_hit_scenes"] == 0,
        "strict_bijection_scenes": observed["strict_bijection_scenes"]
        == int(DECODER_RULE["strict_bijection_scenes"]),
        "mean_rigid_coverage": observed["mean_rigid_coverage"]
        >= float(DECODER_RULE["mean_rigid_coverage_min"]),
        "all_non_seed_attachments_two_seam": observed[
            "all_non_seed_attachments_two_seam"
        ]
        == bool(DECODER_RULE["all_non_seed_attachments_two_seam"]),
        "mean_placement": observed["mean_placement"]
        >= float(DECODER_RULE["mean_placement_min"]),
        "mean_neighbour": observed["mean_neighbour"]
        >= float(DECODER_RULE["mean_neighbour_min"]),
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
        "status": "go_changed_decoder_oracle" if passed else "kill_end_to_end",
        "passed": passed,
        "thresholds": dict(END_TO_END_RULE),
        "observed": observed,
        "checks": checks,
        "scope": "clean_oracle_changed_decoder_only_not_deployable",
    }


def _serialisable_decoder_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def run_discovery(paths: E15Paths) -> Mapping[str, Any]:
    started = time.perf_counter()
    report_path = _require_e_drive(paths.report, label="E15 report")
    if report_path.suffix.lower() != ".json":
        raise E15ContractError("E15 report must be a .json file")
    raw_cache_dir = _require_e_drive(paths.raw_cache_dir, label="raw score cache")
    e12_report_path = _require_e_drive(paths.e12_report, label="E12 report")
    e14_report_path = _require_e_drive(paths.e14_report, label="E14 report")
    if report_path in {e12_report_path, e14_report_path, paths.calibration_report.resolve()}:
        raise E15ContractError("E15 report must not overwrite an input")
    clean_cache_dir = (DEFAULT_E12_REPORT.parent / "score_cache").resolve()
    if report_path.is_relative_to(raw_cache_dir) or report_path.is_relative_to(
        clean_cache_dir
    ):
        raise E15ContractError(
            "E15 report must not be written inside an input cache directory"
        )

    e12_report, _calibration, scenes = e14.load_verified_e12_inputs(
        e14.E14Paths(
            raw_cache_dir=raw_cache_dir,
            calibration_report=paths.calibration_report.resolve(),
            e12_report=e12_report_path,
            report=e14_report_path,
        )
    )
    e14_report = _verify_e14_report(e14_report_path)
    rr_rows = e14._e12_rr_rows(e12_report)
    rr_verification = e14.verify_rr_means(rr_rows)
    clean_records = e14._clean_cache_records(e12_report)
    contract = {
        "protocol_sha256": e12.canonical_digest(E15_PROTOCOL),
        "e12_report": {
            "path": str(e12_report_path),
            "sha256": EXPECTED_E12_REPORT_SHA256,
        },
        "e14_report": {
            "path": str(e14_report_path),
            "sha256": EXPECTED_E14_REPORT_SHA256,
            "run_contract_sha256": str(e14_report["run_contract_sha256"]),
        },
        "calibration_report": {
            "path": str(paths.calibration_report.resolve()),
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
        "e15_code_provenance": _source_provenance(),
        "runtime_provenance": _runtime_provenance(),
    }
    contract_digest = e12.canonical_digest(contract)
    if report_path.is_file():
        existing = _load_json(report_path, label="existing E15 report")
        if (
            existing.get("status") == "complete"
            and existing.get("schema") == REPORT_SCHEMA
            and existing.get("experiment") == EXPERIMENT
            and existing.get("run_contract_sha256") == contract_digest
            and existing.get("protocol") == E15_PROTOCOL
        ):
            return existing
        if existing.get("run_contract_sha256") != contract_digest:
            raise E15ContractError("existing E15 report belongs to different bytes")

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "in_progress",
        "stage": "structure",
        "protocol": E15_PROTOCOL,
        "protocol_sha256": e12.canonical_digest(E15_PROTOCOL),
        "run_contract": contract,
        "run_contract_sha256": contract_digest,
        "rr_reproducibility": rr_verification,
        "rows": {"RR96": [], "structure": [], "candidate": []},
        "completed_structure_images": [],
        "completed_decoder_images": [],
        "completed_nlm_images": [],
        "decisions": {
            "structure": {"status": "not_run"},
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
            components, owner = frame.build_seed_components(right, down)
            claims96 = frame.selected_claims(
                right, down, max_edges=frame.SEED_MAX_EDGES
            )
            claims192 = frame.selected_claims(
                right, down, max_edges=frame.VOTE_MAX_EDGES
            )
            hypotheses = frame.build_translation_hypotheses(
                components, owner, claims192
            )
            row = {
                "image": int(scene.image_id),
                "validation_name": str(scene.validation_name),
                **measure_structure(
                    components, claims96, hypotheses, scene.permutation
                ),
            }
            output["rows"]["structure"].append(row)
            output["completed_structure_images"].append(int(scene.image_id))
            states.append(
                _State(
                    scene=scene,
                    right=right,
                    down=down,
                    clean_cache_sha256=clean_cache.sha256,
                )
            )
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)

        structure_summary = summarize_structure(output["rows"]["structure"])
        structure_gate = structural_decision(structure_summary)
        output["structural_summary"] = structure_summary
        output["decisions"]["structure"] = structure_gate
        if not structure_gate["passed"]:
            output["status"] = "complete"
            output["stage"] = "kill_structure"
            output["runtime_seconds"] = float(time.perf_counter() - started)
            _atomic_write_json(report_path, output)
            return output

        output["stage"] = "decoder"
        _atomic_write_json(report_path, output)
        for state in states:
            solve_started = time.perf_counter()
            result = frame.solve_frame_consensus(
                state.right, state.down, state.scene.tiles_uint8
            )
            solver_seconds = time.perf_counter() - solve_started
            evaluated = evaluate_solve_only(state.scene, result)
            decoder_rows.append(evaluated)
            serialised = {
                "image": int(state.scene.image_id),
                "validation_name": str(state.scene.validation_name),
                "arm": "E15_frame_consensus",
                **_serialisable_decoder_row(evaluated),
                "solver_seconds": float(solver_seconds),
                "clean_score_cache_sha256": state.clean_cache_sha256,
            }
            output["rows"]["candidate"].append(serialised)
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
                raise E15ContractError("fixed NLM returned invalid geometry/dtype")
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

        comparison = dict(
            e14.paired_summary(output["rows"]["candidate"], rr_rows)
        )
        comparison["candidate_arm"] = "E15_frame_consensus"
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
        description="Run fixed CPU-only E15 two-vote frame-consensus oracle."
    )
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--e12-report", type=Path, default=DEFAULT_E12_REPORT)
    parser.add_argument("--e14-report", type=Path, default=DEFAULT_E14_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_discovery(
        E15Paths(
            raw_cache_dir=args.raw_cache_dir,
            calibration_report=args.calibration_report,
            e12_report=args.e12_report,
            e14_report=args.e14_report,
            report=args.report,
        )
    )
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "stage": result["stage"],
                "structure": result["decisions"]["structure"]["status"],
                "decoder": result["decisions"]["decoder"]["status"],
                "end_to_end": result["decisions"]["end_to_end"]["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
