"""Frozen clean-score oracle for pre-denoising placement headroom.

This evaluator is deliberately a diagnostic ceiling, not a submission path.
It replays the already-open calibration scenes 10..17 and compares exactly
three predeclared arms:

``RR``
    Frozen raw affinity candidates and frozen raw ranker scores.
``RC``
    The same raw candidates, rescored by the frozen ranker on clean tiles.
``CC``
    Candidates mined by both frozen affinity encoders on clean tiles and then
    scored by the frozen ranker on those clean tiles.

Clean pixels may influence candidate mining/scoring only.  Every arm solves
with the rank-96 buddies contract, assembles the original corrupted upright
tiles, and applies the same OpenCV NLM h=10 restoration.  No rotation or
reflection path exists here.

Permutation contract:
    ``permutation[input_tile] = clean_row_major_cell``;
    ``board[cell] = input_tile``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from config import FS, GRID, IMG, NFRAG
from eval_buddies_ssim_budget import (
    RawScene,
    array_sha256,
    canonical_digest,
    load_raw_scenes,
    scene_provenance,
    sha256_file,
)
from eval_candidate_rank import load_ranker, score_full_graph
from eval_seeded_qap import dense_rd
from imgio import assemble, to_frags
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates


class OracleContractError(RuntimeError):
    """The frozen oracle contract or one of its byte identities drifted."""


SCHEMA_VERSION = 1
REPORT_SCHEMA = "pazzle-clean-score-oracle-report-v1"
SCORE_CACHE_SCHEMA = "pazzle-clean-score-oracle-score-cache-v1"
EXPERIMENT = "clean_score_oracle_calibration_v1"

CALIBRATION_IDS = tuple(range(10, 18))
CALIBRATION_NAMES = (
    "img_006710.png",
    "img_006711.png",
    "img_006712.png",
    "img_006713.png",
    "img_006714.png",
    "img_006715.png",
    "img_006716.png",
    "img_006717.png",
)
REPLAY_GROUP = (10, 12)
REPLAY_SEED = 1234
DATASET_SEED = 401234
CACHE_TAG = "k64"

CALIBRATION_REPORT_SHA256 = (
    "3b76d6bed59df13eb98af049c3a756151b4485c2e50b1da88ec50fb7a1dfe305"
)
SCENE_PROVENANCE_DIGEST = (
    "00cd2fdd9189d6453e7c1b215e4ee067b843bc51cdcd0122fa66fdc076779c98"
)
EXPECTED_RR_MEAN_SOLVE_SSIM = 0.094607964147414

CANDIDATE_K_PER_ENCODER = 64
CANDIDATE_STORAGE_WIDTH = 128
PAIR_BATCH = 4096
MAX_EDGES = 96
MIN_MARGIN = 0.0
REPAIR_PASSES = 0
NLM_H = 10
RUNTIME_SEED = 20_260_806
BOOTSTRAP_SEED = 20_260_812
BOOTSTRAP_SAMPLES = 20_000
ARMS = ("RR", "RC", "CC")

EXPECTED_CHECKPOINT_SHA256: dict[str, str] = {
    "ranker": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
    "affinity_primary": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
    "affinity_secondary": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
}

KILL_RULE: dict[str, float | int] = {
    "cc_minus_rr_mean_solve_ssim_min": 0.010,
    "cc_minus_rr_mean_final_ssim_min": 0.015,
    "cc_minus_rr_final_wins_min": 6,
    "cc_minus_rr_worst_final_delta_min": -0.020,
}

# Literal, immutable experiment contract.  CLI arguments may relocate files,
# but cannot alter any experimental choice below.
ORACLE_PROTOCOL: dict[str, Any] = {
    "schema": "pazzle-clean-score-oracle-v1",
    "role": "calibration_diagnostic_ceiling_not_production_or_submission",
    "grid": 24,
    "tile_size": 20,
    "image_size": 480,
    "num_tiles": 576,
    "orientation": "fixed",
    "orientation_detail": "upright_tiles_no_rotation_or_reflection_search",
    "calibration_ids": [10, 11, 12, 13, 14, 15, 16, 17],
    "calibration_names": [
        "img_006710.png",
        "img_006711.png",
        "img_006712.png",
        "img_006713.png",
        "img_006714.png",
        "img_006715.png",
        "img_006716.png",
        "img_006717.png",
    ],
    "replay": {
        "group": [10, 12],
        "seed": 1234,
        "dataset_seed": 401234,
        "cache_tag": "k64",
    },
    "corruption": {
        "scope": "independent_per_upright_tile",
        "order": [
            "contrast_about_per_tile_grayscale_mean",
            "brightness",
            "gaussian_noise",
            "clip_0_255",
            "reflect_gaussian_blur_3x3",
            "clip_and_uint8",
            "jpeg_reencode",
            "shuffle",
        ],
        "contrast_range": [0.70, 1.30],
        "grayscale_weights_rgb": [0.299, 0.587, 0.114],
        "brightness_range": [-30.0, 30.0],
        "gaussian_noise_sigma_range": [40.0, 55.0],
        "blur_kernel": [0.25, 0.50, 0.25],
        "blur_border": "reflect",
        "jpeg_quality_inclusive": [35, 50],
        "rotation": False,
        "reflection": False,
    },
    "checkpoints_sha256": {
        "ranker": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
        "affinity_primary": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
        "affinity_secondary": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
    },
    "candidate_graph": {
        "per_encoder_top_k": 64,
        "union": "ordered_deduplicated_primary_then_secondary",
        "max_candidates_per_row": 128,
    },
    "clean_tile_mapping": "imgio.to_frags(target_uint8)[permutation]",
    "arms": {
        "RR": {
            "candidates": "frozen_raw_cache",
            "scores": "frozen_raw_cache",
        },
        "RC": {
            "candidates": "frozen_raw_cache",
            "scores": "clean_tiles_frozen_candidate_seam_ranker",
        },
        "CC": {
            "candidates": "clean_tiles_frozen_dual_affinity_union",
            "scores": "clean_tiles_frozen_candidate_seam_ranker",
        },
    },
    "score_tensor_layout": "direction_anchor_candidate_4x576x128",
    "score_pair_batch": 4096,
    "score_cache_identity": "protocol+scene+clean_tiles+checkpoints+scoring_code",
    "dense_conversion": "eval_seeded_qap.dense_rd_cpu_float32",
    "solver": {
        "name": "solve_buddies.solve_buddies_from_scores",
        "max_edges": 96,
        "min_margin": 0.0,
        "repair_passes": 0,
    },
    "assembly": "original_corrupted_upright_tiles_only",
    "restoration": {
        "name": "opencv_fast_nlm_colored",
        "h": 10,
        "h_color": 10,
        "template_window": 7,
        "search_window": 21,
    },
    "rr_reproducibility": {
        "board_hashes": "calibration_v1.grid_per_image.96",
        "mean_solve_ssim": 0.094607964147414,
        "absolute_tolerance": 1.0e-12,
    },
    "kill_rule": {
        "cc_minus_rr_mean_solve_ssim_min": 0.010,
        "cc_minus_rr_mean_final_ssim_min": 0.015,
        "cc_minus_rr_final_wins_min": 6,
        "cc_minus_rr_worst_final_delta_min": -0.020,
    },
}

CALIBRATION_CONTRACT: dict[str, Any] = {
    "replay_group": "10:12",
    "replay_seed": 1234,
    "dataset_seed": 401234,
    "cache_tag": "k64",
    "budgets": [64, 96, 128, 192, 256, 384, 512, 768, 900],
    "baseline_budget": 512,
    "repair_passes": 0,
    "min_margin": 0.0,
    "orientation": "fixed",
    "score_source": "raw_cached_candidate_seam_ranker",
}

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path("E:/pazzle_work/edge_confidence/full_graph_cache")
DEFAULT_CALIBRATION_REPORT = WORKSPACE / "artifacts" / "buddies_budget" / "calibration_v1.json"
DEFAULT_RANKER_CHECKPOINT = WORKSPACE / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt"
DEFAULT_AFFINITY_PRIMARY = WORKSPACE / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt"
DEFAULT_AFFINITY_SECONDARY = WORKSPACE / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt"
DEFAULT_OUTPUT_DIR = Path("E:/pazzle_work/denoise_oracle")
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "clean_score_oracle_calibration_v1.json"

METRICS = (
    "edge_r1",
    "candidate_recall",
    "placement",
    "neighbour",
    "right",
    "down",
    "solve_only_ssim",
    "final_ssim",
    "objective",
)


@dataclass(frozen=True)
class OraclePaths:
    cache_dir: Path
    calibration_report: Path
    ranker_checkpoint: Path
    affinity_primary: Path
    affinity_secondary: Path
    output_dir: Path
    report: Path


@dataclass(frozen=True)
class CleanScoreCache:
    rc_scores: np.ndarray
    cc_candidates: np.ndarray
    cc_valid: np.ndarray
    cc_scores: np.ndarray
    path: Path
    sha256: str
    status: str


@dataclass(frozen=True)
class OracleModels:
    ranker: Any
    affinity_primary: Any
    affinity_secondary: Any
    device: torch.device


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _strict_board(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board)
    if value.shape != (NFRAG,) or value.dtype.kind not in "iu":
        raise OracleContractError(f"board must be an integer ({NFRAG},) array")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(NFRAG, dtype=np.int64)):
        raise OracleContractError("solver board is not a strict tile permutation")
    return np.ascontiguousarray(value)


def clean_tiles_input_order(target_uint8: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Return pristine tiles indexed by the original corrupted input-tile ids."""

    target = np.asarray(target_uint8)
    perm = np.asarray(permutation)
    if target.shape != (IMG, IMG, 3) or target.dtype != np.uint8:
        raise OracleContractError(f"target must be uint8 ({IMG},{IMG},3)")
    if perm.shape != (NFRAG,) or perm.dtype.kind not in "iu":
        raise OracleContractError(f"permutation must be integer ({NFRAG},)")
    perm = perm.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(perm), np.arange(NFRAG, dtype=np.int64)):
        raise OracleContractError("permutation is not input_tile->clean_cell bijection")
    clean = to_frags(target)[perm]
    if clean.shape != (NFRAG, FS, FS, 3):
        raise OracleContractError("imgio.to_frags returned unexpected clean tile geometry")
    return np.ascontiguousarray(clean, dtype=np.uint8)


def model_tiles(clean_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    clean = np.asarray(clean_uint8)
    if clean.shape != (NFRAG, FS, FS, 3) or clean.dtype != np.uint8:
        raise OracleContractError("clean model input has invalid shape or dtype")
    return (
        torch.from_numpy(clean)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .to(device)
        / 255.0
    )


def raw_common_valid_mask(base_scores: np.ndarray) -> np.ndarray:
    """Recover the one common candidate mask required by ``score_full_graph``."""

    scores = np.asarray(base_scores)
    if scores.ndim != 3 or scores.shape[0] != 4 or scores.shape[1] != NFRAG:
        raise OracleContractError("raw scores must be direction-major (4,576,K)")
    finite = np.isfinite(scores)
    if not all(np.array_equal(finite[0], finite[direction]) for direction in range(1, 4)):
        raise OracleContractError("raw candidate validity mask differs by direction")
    valid = np.ascontiguousarray(finite[0], dtype=np.bool_)
    if not bool(valid.any(axis=1).all()):
        raise OracleContractError("raw graph contains an anchor with no valid candidate")
    return valid


def validate_graph_arrays(
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    *,
    label: str,
) -> None:
    candidate_array = np.asarray(candidates)
    valid_array = np.asarray(valid)
    score_array = np.asarray(scores)
    expected_candidates = (NFRAG, CANDIDATE_STORAGE_WIDTH)
    expected_scores = (4, NFRAG, CANDIDATE_STORAGE_WIDTH)
    if candidate_array.shape != expected_candidates or candidate_array.dtype.kind not in "iu":
        raise OracleContractError(f"{label} candidates must be integer {expected_candidates}")
    if valid_array.shape != expected_candidates or valid_array.dtype != np.bool_:
        raise OracleContractError(f"{label} valid mask must be boolean {expected_candidates}")
    if score_array.shape != expected_scores or score_array.dtype != np.float32:
        raise OracleContractError(f"{label} scores must be float32 {expected_scores}")
    if np.any(candidate_array < 0) or np.any(candidate_array >= NFRAG):
        raise OracleContractError(f"{label} candidate id is outside 0..{NFRAG - 1}")
    if not bool(valid_array.any(axis=1).all()):
        raise OracleContractError(f"{label} contains an anchor with no valid candidate")
    expanded = np.broadcast_to(valid_array, score_array.shape)
    if not bool(np.isfinite(score_array[expanded]).all()):
        raise OracleContractError(f"{label} has a non-finite valid score")
    if bool(np.isfinite(score_array[~expanded]).any()):
        raise OracleContractError(f"{label} has a finite score in an invalid slot")
    anchors = np.arange(NFRAG, dtype=np.int64)[:, None]
    if bool(((candidate_array == anchors) & valid_array).any()):
        raise OracleContractError(f"{label} contains a valid self-candidate")
    for anchor in range(NFRAG):
        row = candidate_array[anchor, valid_array[anchor]]
        if np.unique(row).size != row.size:
            raise OracleContractError(f"{label} contains duplicate valid candidates")


def directed_graph_metrics(
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    permutation: np.ndarray,
) -> dict[str, float | int]:
    """Compute directed true-edge candidate recall and rank-1 accuracy."""

    validate_graph_arrays(candidates, valid, scores, label="metric graph")
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (NFRAG,) or not np.array_equal(
        np.sort(perm), np.arange(NFRAG, dtype=np.int64)
    ):
        raise OracleContractError("metric permutation is not a bijection")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[perm] = np.arange(NFRAG, dtype=np.int64)
    deltas = (-GRID, GRID, -1, 1)  # U, D, L, R
    recalled = 0
    rank1 = 0
    total = 0
    for anchor in range(NFRAG):
        cell = int(perm[anchor])
        row, col = divmod(cell, GRID)
        exists = (row > 0, row < GRID - 1, col > 0, col < GRID - 1)
        for direction, present in enumerate(exists):
            if not present:
                continue
            total += 1
            target = int(inverse[cell + deltas[direction]])
            row_valid = valid[anchor]
            row_candidates = candidates[anchor]
            recalled += int(bool(np.any(row_valid & (row_candidates == target))))
            masked = np.where(row_valid, scores[direction, anchor], -np.inf)
            rank1 += int(int(row_candidates[int(np.argmax(masked))]) == target)
    expected_edges = 4 * GRID * (GRID - 1)
    if total != expected_edges:
        raise AssertionError(f"directed edge count drifted: {total} != {expected_edges}")
    return {
        "directed_true_edges": int(total),
        "candidate_hits": int(recalled),
        "rank1_hits": int(rank1),
        "candidate_recall": float(recalled / total),
        "edge_r1": float(rank1 / total),
    }


def dense_from_graph(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the frozen CPU-float32 boundary before the discrete solver."""

    candidates_cpu = torch.from_numpy(np.ascontiguousarray(candidates, dtype=np.int64)).long()
    scores_cpu = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32)).float()
    right_t, down_t = dense_rd(candidates_cpu, scores_cpu)
    right = np.ascontiguousarray(right_t.detach().float().cpu().numpy(), dtype=np.float32)
    down = np.ascontiguousarray(down_t.detach().float().cpu().numpy(), dtype=np.float32)
    for label, matrix in (("right", right), ("down", down)):
        if matrix.shape != (NFRAG, NFRAG) or not np.isfinite(matrix).all():
            raise OracleContractError(f"dense {label} matrix is invalid")
        if bool((matrix < 0.0).any()) or bool((np.diag(matrix) != 0.0).any()):
            raise OracleContractError(f"dense {label} matrix violates probability contract")
    return right, down


def solve_dense(
    right: np.ndarray,
    down: np.ndarray,
    *,
    solver: Callable[..., tuple[np.ndarray, float]] = solve_buddies_from_scores,
) -> tuple[np.ndarray, float, float]:
    """Solve with exactly the predeclared rank-96/no-repair settings."""

    right = np.ascontiguousarray(right, dtype=np.float32)
    down = np.ascontiguousarray(down, dtype=np.float32)
    if right.shape != (NFRAG, NFRAG) or down.shape != right.shape:
        raise OracleContractError("solver inputs must be two dense 576x576 matrices")
    started = time.perf_counter()
    board, objective = solver(
        right,
        down,
        max_edges=MAX_EDGES,
        min_margin=MIN_MARGIN,
        repair_passes=REPAIR_PASSES,
    )
    elapsed = time.perf_counter() - started
    board = _strict_board(np.asarray(board))
    if not math.isfinite(float(objective)):
        raise OracleContractError("solver returned a non-finite objective")
    return board, float(objective), float(elapsed)


def fixed_nlm(image: np.ndarray) -> np.ndarray:
    """The exact production restoration tail, kept lazy for data-free tests."""

    import cv2
    from pipeline import nlm_restore

    cv2.setNumThreads(1)
    return nlm_restore(np.ascontiguousarray(image), h=NLM_H)


def board_metrics(
    scene: RawScene,
    board: np.ndarray,
    objective: float,
    *,
    restorer: Callable[[np.ndarray], np.ndarray] = fixed_nlm,
) -> dict[str, Any]:
    """Evaluate a board while guaranteeing clean pixels never enter its canvas."""

    board = _strict_board(board)
    corrupted_tiles = np.asarray(scene.tiles_uint8)
    target = np.asarray(scene.target_uint8)
    if corrupted_tiles.shape != (NFRAG, FS, FS, 3) or corrupted_tiles.dtype != np.uint8:
        raise OracleContractError("scene corrupted tiles have invalid geometry or dtype")
    if target.shape != (IMG, IMG, 3) or target.dtype != np.uint8:
        raise OracleContractError("scene target has invalid geometry or dtype")
    truth_board = np.argsort(np.asarray(scene.permutation, dtype=np.int64))
    placement = placement_accuracy(board, truth_board)[0]
    neighbour, right_accuracy, down_accuracy = neighbour_accuracy(board, truth_board)
    solved = np.ascontiguousarray(assemble(corrupted_tiles, board), dtype=np.uint8)
    restored = np.asarray(restorer(solved.copy()))
    if restored.shape != (IMG, IMG, 3) or restored.dtype != np.uint8:
        raise OracleContractError("fixed restorer must return uint8 480x480 RGB")
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right_accuracy),
        "down": float(down_accuracy),
        "solve_only_ssim": float(
            sk_ssim(target, solved, channel_axis=2, data_range=255)
        ),
        "final_ssim": float(
            sk_ssim(target, restored, channel_axis=2, data_range=255)
        ),
        "objective": float(objective),
        "board_sha256": array_sha256(board.astype(np.int64, copy=False)),
        "solved_corrupted_canvas_sha256": array_sha256(solved),
        "restored_canvas_sha256": array_sha256(restored),
    }


def evaluate_arm(
    scene: RawScene,
    arm: str,
    candidates: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    *,
    solver: Callable[..., tuple[np.ndarray, float]] = solve_buddies_from_scores,
    restorer: Callable[[np.ndarray], np.ndarray] = fixed_nlm,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise OracleContractError(f"unknown oracle arm {arm!r}")
    validate_graph_arrays(candidates, valid, scores, label=arm)
    graph = directed_graph_metrics(candidates, valid, scores, scene.permutation)
    right, down = dense_from_graph(candidates, scores)
    board, objective, solver_seconds = solve_dense(right, down, solver=solver)
    return {
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "arm": arm,
        **graph,
        **board_metrics(scene, board, objective, restorer=restorer),
        "solver_seconds": solver_seconds,
    }


def validate_calibration_payload(payload: Mapping[str, Any]) -> dict[int, str]:
    """Validate every immutable field needed to accept the RR replay."""

    if int(payload.get("schema_version", -1)) != 1:
        raise OracleContractError("calibration report schema_version drifted")
    if payload.get("experiment") != "raw_buddies_solve_ssim_budget":
        raise OracleContractError("calibration report experiment drifted")
    if payload.get("phase") != "calibration" or payload.get("status") != "frozen":
        raise OracleContractError("calibration report is not a frozen calibration")
    if payload.get("calibration_ids") != list(CALIBRATION_IDS):
        raise OracleContractError("calibration IDs are not exactly 10..17")
    if payload.get("confirmation_ids_reserved") != [18, 19, 20, 21]:
        raise OracleContractError("calibration confirmation reservation drifted")
    if payload.get("contract") != CALIBRATION_CONTRACT:
        raise OracleContractError("calibration replay/solver contract drifted")
    if int(payload.get("selected_budget", -1)) != MAX_EDGES:
        raise OracleContractError("calibration did not freeze budget 96")
    if payload.get("scene_provenance_digest") != SCENE_PROVENANCE_DIGEST:
        raise OracleContractError("calibration scene provenance digest drifted")
    provenance = payload.get("scene_provenance")
    if not isinstance(provenance, list) or canonical_digest(provenance) != SCENE_PROVENANCE_DIGEST:
        raise OracleContractError("calibration scene provenance payload does not match its digest")
    if [row.get("image") for row in provenance if isinstance(row, Mapping)] != list(CALIBRATION_IDS):
        raise OracleContractError("calibration scene provenance rows are incomplete or reordered")
    if [row.get("validation_name") for row in provenance if isinstance(row, Mapping)] != list(
        CALIBRATION_NAMES
    ):
        raise OracleContractError("calibration validation names drifted")
    selected_metrics = payload.get("selected_metrics")
    if not isinstance(selected_metrics, Mapping) or not math.isclose(
        float(selected_metrics.get("solve_only_ssim", float("nan"))),
        EXPECTED_RR_MEAN_SOLVE_SSIM,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise OracleContractError("calibration selected mean solve SSIM drifted")
    grid = payload.get("grid")
    if not isinstance(grid, Mapping) or not isinstance(grid.get("96"), Mapping) or not math.isclose(
        float(grid["96"].get("solve_only_ssim", float("nan"))),
        EXPECTED_RR_MEAN_SOLVE_SSIM,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise OracleContractError("calibration budget-96 grid mean drifted")
    per_budget = payload.get("grid_per_image")
    rows = per_budget.get("96") if isinstance(per_budget, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(CALIBRATION_IDS):
        raise OracleContractError("calibration budget-96 rows are incomplete")
    hashes: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise OracleContractError("calibration budget-96 row is not an object")
        image = int(row.get("image", -1))
        digest = row.get("board_sha256")
        if image in hashes or image not in CALIBRATION_IDS or not _is_sha256(digest):
            raise OracleContractError("calibration budget-96 board hashes are invalid")
        hashes[image] = str(digest)
    if tuple(sorted(hashes)) != CALIBRATION_IDS:
        raise OracleContractError("calibration budget-96 board IDs drifted")
    return hashes


def load_calibration_report(path: Path) -> Mapping[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != CALIBRATION_REPORT_SHA256:
        raise OracleContractError(
            "calibration report SHA256 mismatch: "
            f"expected {CALIBRATION_REPORT_SHA256}, got {digest}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise OracleContractError("calibration report root is not an object")
    validate_calibration_payload(payload)
    return payload


def validate_scene_replay(
    scenes: Sequence[RawScene], calibration_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Require the exact recorded path and every byte-derived scene fingerprint."""

    if tuple(scene.image_id for scene in scenes) != CALIBRATION_IDS:
        raise OracleContractError("loaded scenes are not ordered calibration IDs 10..17")
    recorded_rows = calibration_payload.get("scene_provenance")
    if not isinstance(recorded_rows, list):
        raise OracleContractError("calibration report has no scene provenance")
    recorded = {int(row["image"]): row for row in recorded_rows}
    observed: list[dict[str, Any]] = []
    for scene in scenes:
        current = scene_provenance(scene)
        if current != recorded.get(scene.image_id):
            raise OracleContractError(
                f"scene {scene.image_id} replay differs from calibration_v1 provenance"
            )
        observed.append(current)
    if canonical_digest(observed) != SCENE_PROVENANCE_DIGEST:
        raise OracleContractError("observed scene provenance digest drifted")
    return observed


def verify_rr_replay(
    rr_rows: Sequence[Mapping[str, Any]], calibration_payload: Mapping[str, Any]
) -> dict[str, Any]:
    expected_hashes = validate_calibration_payload(calibration_payload)
    if len(rr_rows) != len(CALIBRATION_IDS):
        raise OracleContractError("RR replay did not return exactly eight rows")
    observed: dict[int, Mapping[str, Any]] = {}
    for row in rr_rows:
        image = int(row.get("image", -1))
        if image in observed or image not in CALIBRATION_IDS:
            raise OracleContractError("RR replay image IDs are duplicated or invalid")
        observed[image] = row
    for image in CALIBRATION_IDS:
        digest = observed[image].get("board_sha256")
        if digest != expected_hashes[image]:
            raise OracleContractError(
                f"RR board hash mismatch for image {image}: "
                f"expected {expected_hashes[image]}, got {digest}"
            )
    mean_solve = float(
        np.mean([float(observed[image]["solve_only_ssim"]) for image in CALIBRATION_IDS])
    )
    if not math.isclose(
        mean_solve,
        EXPECTED_RR_MEAN_SOLVE_SSIM,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise OracleContractError(
            "RR mean solve SSIM mismatch: "
            f"expected {EXPECTED_RR_MEAN_SOLVE_SSIM:.15f}, got {mean_solve:.15f}"
        )
    return {
        "passed": True,
        "expected_mean_solve_ssim": EXPECTED_RR_MEAN_SOLVE_SSIM,
        "observed_mean_solve_ssim": mean_solve,
        "absolute_tolerance": 1.0e-12,
        "board_hashes": {str(image): expected_hashes[image] for image in CALIBRATION_IDS},
    }


def validate_ranker_graph(payload: Mapping[str, Any]) -> None:
    graph = payload.get("candidate_graph")
    if not isinstance(graph, Mapping):
        raise OracleContractError("ranker checkpoint has no candidate_graph contract")
    if int(graph.get("per_encoder_top_k", -1)) != CANDIDATE_K_PER_ENCODER:
        raise OracleContractError("ranker graph was not trained with per-encoder K=64")
    if graph.get("union") is not True or int(graph.get("max_candidates_per_row", -1)) != 128:
        raise OracleContractError("ranker graph is not the exact two-encoder K=64 union")
    encoders = graph.get("encoders")
    if (
        not isinstance(encoders, Sequence)
        or isinstance(encoders, (str, bytes))
        or len(encoders) != 2
    ):
        raise OracleContractError("ranker graph must record exactly two affinity encoders")
    hashes = []
    for encoder in encoders:
        if not isinstance(encoder, Mapping) or not isinstance(encoder.get("sha256"), str):
            raise OracleContractError("ranker graph encoder provenance is incomplete")
        hashes.append(str(encoder["sha256"]).lower())
    expected = [
        EXPECTED_CHECKPOINT_SHA256["affinity_primary"],
        EXPECTED_CHECKPOINT_SHA256["affinity_secondary"],
    ]
    if hashes != expected:
        raise OracleContractError("ranker graph affinity hashes or order drifted")


def checkpoint_provenance(paths: OraclePaths) -> dict[str, dict[str, Any]]:
    checkpoint_paths = {
        "ranker": paths.ranker_checkpoint.resolve(),
        "affinity_primary": paths.affinity_primary.resolve(),
        "affinity_secondary": paths.affinity_secondary.resolve(),
    }
    records: dict[str, dict[str, Any]] = {}
    for role, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != EXPECTED_CHECKPOINT_SHA256[role]:
            raise OracleContractError(
                f"{role} checkpoint SHA256 mismatch: "
                f"expected {EXPECTED_CHECKPOINT_SHA256[role]}, got {digest}"
            )
        records[role] = {
            "path": str(path),
            "sha256": digest,
            "size": int(path.stat().st_size),
        }
    return records


def _set_deterministic_cuda_runtime() -> torch.device:
    if not torch.cuda.is_available():
        raise OracleContractError("the frozen oracle requires CUDA for model scoring")
    random.seed(RUNTIME_SEED)
    np.random.seed(RUNTIME_SEED)
    torch.manual_seed(RUNTIME_SEED)
    torch.cuda.manual_seed_all(RUNTIME_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda")


def load_oracle_models(paths: OraclePaths) -> OracleModels:
    device = _set_deterministic_cuda_runtime()
    ranker, ranker_payload = load_ranker(str(paths.ranker_checkpoint.resolve()), device)
    validate_ranker_graph(ranker_payload)
    ranker_kwargs = ranker_payload.get("model_kwargs")
    if not isinstance(ranker_kwargs, Mapping) or int(ranker_kwargs.get("tile_size", -1)) != FS:
        raise OracleContractError("ranker checkpoint tile geometry drifted")
    primary, _, primary_kwargs = load_frozen_affinity(
        str(paths.affinity_primary.resolve()), device
    )
    secondary, _, secondary_kwargs = load_frozen_affinity(
        str(paths.affinity_secondary.resolve()), device
    )
    for role, kwargs in (
        ("affinity_primary", primary_kwargs),
        ("affinity_secondary", secondary_kwargs),
    ):
        if int(kwargs.get("tiles", -1)) != NFRAG or int(kwargs.get("tile_size", -1)) != FS:
            raise OracleContractError(f"{role} checkpoint puzzle geometry drifted")
    return OracleModels(ranker, primary, secondary, device)


def _cache_metadata(
    scene: RawScene,
    clean_tiles: np.ndarray,
    checkpoints: Mapping[str, Mapping[str, Any]],
    scoring_code: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": SCORE_CACHE_SCHEMA,
        "experiment": EXPERIMENT,
        "image": int(scene.image_id),
        "validation_name": str(scene.validation_name),
        "protocol_sha256": canonical_digest(ORACLE_PROTOCOL),
        "scene_provenance": scene_provenance(scene),
        "clean_tiles_input_order_sha256": array_sha256(clean_tiles),
        "checkpoints_sha256": {
            role: str(checkpoints[role]["sha256"])
            for role in ("ranker", "affinity_primary", "affinity_secondary")
        },
        "scoring_code_sha256": dict(scoring_code),
        "scoring_code_digest": canonical_digest(scoring_code),
    }


def _validate_cache_arrays(
    scene: RawScene,
    rc_scores: np.ndarray,
    cc_candidates: np.ndarray,
    cc_valid: np.ndarray,
    cc_scores: np.ndarray,
) -> None:
    raw_valid = raw_common_valid_mask(scene.base_scores)
    validate_graph_arrays(
        scene.candidate_ids,
        raw_valid,
        np.asarray(rc_scores),
        label="RC cache",
    )
    validate_graph_arrays(
        np.asarray(cc_candidates),
        np.asarray(cc_valid),
        np.asarray(cc_scores),
        label="CC cache",
    )


def _load_clean_score_cache(
    path: Path,
    expected_metadata: Mapping[str, Any],
    scene: RawScene,
) -> CleanScoreCache:
    with np.load(path, allow_pickle=False) as stored:
        required = {"metadata_json", "rc_scores", "cc_candidates", "cc_valid", "cc_scores"}
        missing = sorted(required - set(stored.files))
        if missing:
            raise OracleContractError(f"{path} is missing score-cache fields {missing}")
        metadata = json.loads(str(stored["metadata_json"].item()))
        if metadata != expected_metadata:
            raise OracleContractError(f"{path} score-cache metadata drifted")
        rc_scores = stored["rc_scores"]
        cc_candidates = stored["cc_candidates"]
        cc_valid = stored["cc_valid"]
        cc_scores = stored["cc_scores"]
        if rc_scores.dtype != np.float32 or cc_scores.dtype != np.float32:
            raise OracleContractError(f"{path} cached score dtype is not float32")
        if cc_candidates.dtype != np.int64:
            raise OracleContractError(f"{path} cached candidate dtype is not int64")
        if cc_valid.dtype != np.bool_:
            raise OracleContractError(f"{path} cached validity dtype is not bool")
        rc_scores = np.ascontiguousarray(rc_scores)
        cc_candidates = np.ascontiguousarray(cc_candidates)
        cc_valid = np.ascontiguousarray(cc_valid)
        cc_scores = np.ascontiguousarray(cc_scores)
    _validate_cache_arrays(scene, rc_scores, cc_candidates, cc_valid, cc_scores)
    return CleanScoreCache(
        rc_scores=rc_scores,
        cc_candidates=cc_candidates,
        cc_valid=cc_valid,
        cc_scores=cc_scores,
        path=path.resolve(),
        sha256=sha256_file(path),
        status="reused",
    )


def _write_score_cache(
    path: Path,
    metadata: Mapping[str, Any],
    scene: RawScene,
    rc_scores: np.ndarray,
    cc_candidates: np.ndarray,
    cc_valid: np.ndarray,
    cc_scores: np.ndarray,
) -> CleanScoreCache:
    _validate_cache_arrays(scene, rc_scores, cc_candidates, cc_valid, cc_scores)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
            ),
            rc_scores=np.ascontiguousarray(rc_scores, dtype=np.float32),
            cc_candidates=np.ascontiguousarray(cc_candidates, dtype=np.int64),
            cc_valid=np.ascontiguousarray(cc_valid, dtype=np.bool_),
            cc_scores=np.ascontiguousarray(cc_scores, dtype=np.float32),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return CleanScoreCache(
        rc_scores=np.ascontiguousarray(rc_scores, dtype=np.float32),
        cc_candidates=np.ascontiguousarray(cc_candidates, dtype=np.int64),
        cc_valid=np.ascontiguousarray(cc_valid, dtype=np.bool_),
        cc_scores=np.ascontiguousarray(cc_scores, dtype=np.float32),
        path=path,
        sha256=sha256_file(path),
        status="created",
    )


@torch.inference_mode()
def load_or_create_clean_score_cache(
    scene: RawScene,
    models: OracleModels,
    checkpoints: Mapping[str, Mapping[str, Any]],
    scoring_code: Mapping[str, str],
    output_dir: Path,
) -> CleanScoreCache:
    clean_uint8 = clean_tiles_input_order(scene.target_uint8, scene.permutation)
    metadata = _cache_metadata(scene, clean_uint8, checkpoints, scoring_code)
    path = output_dir.resolve() / "score_cache" / f"image_{scene.image_id:04d}_clean_score_v1.npz"
    if path.is_file():
        return _load_clean_score_cache(path, metadata, scene)
    clean = model_tiles(clean_uint8, models.device)
    raw_valid = raw_common_valid_mask(scene.base_scores)
    raw_candidates_t = torch.from_numpy(scene.candidate_ids).long().to(models.device)
    raw_valid_t = torch.from_numpy(raw_valid).bool().to(models.device)
    rc_scores = (
        score_full_graph(
            models.ranker,
            clean,
            raw_candidates_t,
            raw_valid_t,
            pair_batch=PAIR_BATCH,
            device=models.device,
        )
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    cc_candidates_b, cc_valid_b = mine_affinity_candidates(
        models.affinity_primary,
        clean.unsqueeze(0),
        candidate_k=CANDIDATE_K_PER_ENCODER,
        device=models.device,
        affinity_secondary=models.affinity_secondary,
    )
    cc_candidates_t = cc_candidates_b[0]
    cc_valid_t = cc_valid_b[0]
    cc_scores = (
        score_full_graph(
            models.ranker,
            clean,
            cc_candidates_t,
            cc_valid_t,
            pair_batch=PAIR_BATCH,
            device=models.device,
        )
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    cc_candidates = cc_candidates_t.detach().cpu().long().numpy().astype(np.int64, copy=False)
    cc_valid = cc_valid_t.detach().cpu().bool().numpy().astype(np.bool_, copy=False)
    del clean, raw_candidates_t, raw_valid_t, cc_candidates_b, cc_valid_b
    torch.cuda.empty_cache()
    return _write_score_cache(
        path,
        metadata,
        scene,
        rc_scores,
        cc_candidates,
        cc_valid,
        cc_scores,
    )


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _rows_by_calibration_image(rows, label="arm summary")
    return {
        "images": len(rows),
        **{
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        },
        "solver_seconds": float(np.sum([float(row["solver_seconds"]) for row in rows])),
    }


def paired_summary(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_arm: str,
    baseline_arm: str = "RR",
) -> dict[str, Any]:
    candidate = _rows_by_calibration_image(candidate_rows, label=candidate_arm)
    baseline = _rows_by_calibration_image(baseline_rows, label=baseline_arm)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(
        0, len(CALIBRATION_IDS), size=(BOOTSTRAP_SAMPLES, len(CALIBRATION_IDS))
    )
    metric_rows: dict[str, Any] = {}
    per_image: list[dict[str, Any]] = [
        {"image": image} for image in CALIBRATION_IDS
    ]
    for metric in METRICS:
        base_values = np.asarray(
            [float(baseline[image][metric]) for image in CALIBRATION_IDS], dtype=np.float64
        )
        candidate_values = np.asarray(
            [float(candidate[image][metric]) for image in CALIBRATION_IDS], dtype=np.float64
        )
        delta = candidate_values - base_values
        bootstrap = delta[resamples].mean(axis=1)
        metric_rows[metric] = {
            "baseline_mean": float(base_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wins": int(np.sum(delta > 0.0)),
            "ties": int(np.sum(delta == 0.0)),
            "losses": int(np.sum(delta < 0.0)),
            "worst_delta": float(delta.min()),
            "best_delta": float(delta.max()),
            "bootstrap_95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
        for index, value in enumerate(delta):
            per_image[index][metric] = float(value)
    return {
        "candidate_arm": candidate_arm,
        "baseline_arm": baseline_arm,
        "images": len(CALIBRATION_IDS),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES},
        "metrics": metric_rows,
        "per_image_delta": per_image,
    }


def _same_threshold_diagnostic(comparison: Mapping[str, Any]) -> dict[str, Any]:
    metrics = comparison.get("metrics")
    if not isinstance(metrics, Mapping):
        raise OracleContractError("comparison has no metric summaries")
    solve = metrics.get("solve_only_ssim")
    final = metrics.get("final_ssim")
    if not isinstance(solve, Mapping) or not isinstance(final, Mapping):
        raise OracleContractError("comparison lacks solve/final SSIM")
    checks = {
        "mean_solve_ssim": float(solve["mean_delta"])
        >= float(KILL_RULE["cc_minus_rr_mean_solve_ssim_min"]),
        "mean_final_ssim": float(final["mean_delta"])
        >= float(KILL_RULE["cc_minus_rr_mean_final_ssim_min"]),
        "final_wins": int(final["wins"])
        >= int(KILL_RULE["cc_minus_rr_final_wins_min"]),
        "worst_final_delta": float(final["worst_delta"])
        >= float(KILL_RULE["cc_minus_rr_worst_final_delta_min"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "thresholds": dict(KILL_RULE),
        "observed": {
            "mean_solve_ssim_delta": float(solve["mean_delta"]),
            "mean_final_ssim_delta": float(final["mean_delta"]),
            "final_wins": int(final["wins"]),
            "worst_final_delta": float(final["worst_delta"]),
        },
    }


def passes_kill_rule(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared primary decision rule to CC minus RR only."""

    diagnostic = _same_threshold_diagnostic(comparison)
    observed = diagnostic["observed"]
    return {
        "passed": bool(diagnostic["passed"]),
        "status": (
            "pass_headroom" if diagnostic["passed"] else "kill_denoise_scoring"
        ),
        "checks": diagnostic["checks"],
        "thresholds": diagnostic["thresholds"],
        "observed": {
            "cc_minus_rr_mean_solve_ssim": observed["mean_solve_ssim_delta"],
            "cc_minus_rr_mean_final_ssim": observed["mean_final_ssim_delta"],
            "cc_minus_rr_final_wins": observed["final_wins"],
            "cc_minus_rr_worst_final_delta": observed["worst_final_delta"],
        },
    }


def routing_decision(
    cc_comparison: Mapping[str, Any], rc_comparison: Mapping[str, Any]
) -> dict[str, Any]:
    cc = passes_kill_rule(cc_comparison)
    rc = _same_threshold_diagnostic(rc_comparison)
    if not cc["passed"]:
        return {
            **cc,
            "route": "stop_denoising_for_candidate_scoring",
            "reason": "CC failed at least one predeclared headroom threshold",
            "rc_diagnostic": {
                "passes_same_thresholds": bool(rc["passed"]),
                "suggested_future_input": None,
                "checks": rc["checks"],
                "thresholds": rc["thresholds"],
                "observed_rc_minus_rr": rc["observed"],
            },
        }
    if rc["passed"]:
        suggested = "raw_affinity_plus_denoised_ranker_first"
    else:
        suggested = "denoiser_feeds_both_affinity_encoders_and_ranker"
    return {
        **cc,
        "route": "pursue_learned_pre_denoiser",
        "reason": "CC passed every predeclared headroom threshold",
        "rc_diagnostic": {
            "passes_same_thresholds": bool(rc["passed"]),
            "suggested_future_input": suggested,
            "checks": rc["checks"],
            "thresholds": rc["thresholds"],
            "observed_rc_minus_rr": rc["observed"],
        },
    }


def _rows_by_calibration_image(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[int, Mapping[str, Any]]:
    if len(rows) != len(CALIBRATION_IDS):
        raise OracleContractError(f"{label} requires exactly eight scene rows")
    image_ids = [int(row.get("image", -1)) for row in rows]
    if len(set(image_ids)) != len(image_ids) or tuple(sorted(image_ids)) != CALIBRATION_IDS:
        raise OracleContractError(
            f"{label} requires exactly one row for each calibration image 10..17"
        )
    return {image: row for image, row in zip(image_ids, rows)}


def _source_provenance(names: Sequence[str]) -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {name: source / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing oracle provenance files: " + ", ".join(missing))
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def scoring_code_provenance() -> dict[str, str]:
    """Files whose bytes can alter clean candidates or ranker logits."""

    return _source_provenance(
        (
            "eval_clean_score_oracle.py",
            "eval_candidate_rank.py",
            "candidate_rank.py",
            "train_offset_pose.py",
            "macro_affinity.py",
            "config.py",
        )
    )


def code_provenance() -> dict[str, str]:
    return _source_provenance(
        (
            "eval_clean_score_oracle.py",
            "eval_buddies_ssim_budget.py",
            "eval_candidate_rank.py",
            "candidate_rank.py",
            "train_offset_pose.py",
            "macro_affinity.py",
            "eval_seeded_qap.py",
            "solve_buddies.py",
            "canvas_data.py",
            "distort.py",
            "imgio.py",
            "placement_metrics.py",
            "pipeline.py",
            "config.py",
        )
    )


def verify_unchanged_provenance(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, label: str
) -> None:
    if dict(observed) != dict(expected):
        raise OracleContractError(f"{label} changed during the oracle run")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_oracle(paths: OraclePaths) -> Mapping[str, Any]:
    started = time.perf_counter()
    frozen_code = code_provenance()
    frozen_scoring_code = scoring_code_provenance()
    calibration = load_calibration_report(paths.calibration_report)
    frozen_checkpoints = checkpoint_provenance(paths)
    scenes = load_raw_scenes(paths.cache_dir.resolve(), CALIBRATION_IDS)
    observed_provenance = validate_scene_replay(scenes, calibration)

    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    # Fail before model loading/GPU work unless the exact RR baseline replays.
    for scene in scenes:
        raw_valid = raw_common_valid_mask(scene.base_scores)
        rows["RR"].append(
            evaluate_arm(
                scene,
                "RR",
                scene.candidate_ids,
                raw_valid,
                np.ascontiguousarray(scene.base_scores, dtype=np.float32),
            )
        )
    rr_verification = verify_rr_replay(rows["RR"], calibration)
    rr_by_image = {int(row["image"]): row for row in rows["RR"]}

    verify_unchanged_provenance(
        frozen_code, code_provenance(), label="oracle source code"
    )
    verify_unchanged_provenance(
        frozen_checkpoints,
        checkpoint_provenance(paths),
        label="checkpoint files before model load",
    )
    models = load_oracle_models(paths)
    verify_unchanged_provenance(
        frozen_checkpoints,
        checkpoint_provenance(paths),
        label="checkpoint files across model load",
    )
    cache_records: list[dict[str, Any]] = []
    for scene in scenes:
        cached = load_or_create_clean_score_cache(
            scene,
            models,
            frozen_checkpoints,
            frozen_scoring_code,
            paths.output_dir,
        )
        raw_valid = raw_common_valid_mask(scene.base_scores)
        rows["RC"].append(
            evaluate_arm(
                scene,
                "RC",
                scene.candidate_ids,
                raw_valid,
                cached.rc_scores,
            )
        )
        rows["CC"].append(
            evaluate_arm(
                scene,
                "CC",
                cached.cc_candidates,
                cached.cc_valid,
                cached.cc_scores,
            )
        )
        cache_records.append(
            {
                "image": int(scene.image_id),
                "path": str(cached.path),
                "sha256": cached.sha256,
                "status": cached.status,
            }
        )
        print(
            json.dumps(
                {
                    "image": scene.image_id,
                    "RR_solve": rr_by_image[scene.image_id]["solve_only_ssim"],
                    "RC_solve": rows["RC"][-1]["solve_only_ssim"],
                    "CC_solve": rows["CC"][-1]["solve_only_ssim"],
                    "score_cache": cached.status,
                }
            ),
            flush=True,
        )

    rc_comparison = paired_summary(
        rows["RC"], rows["RR"], candidate_arm="RC", baseline_arm="RR"
    )
    cc_comparison = paired_summary(
        rows["CC"], rows["RR"], candidate_arm="CC", baseline_arm="RR"
    )
    decision = routing_decision(cc_comparison, rc_comparison)
    verify_unchanged_provenance(
        frozen_code, code_provenance(), label="oracle source code"
    )
    verify_unchanged_provenance(
        frozen_checkpoints,
        checkpoint_provenance(paths),
        label="checkpoint files across scoring",
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "complete",
        "protocol": ORACLE_PROTOCOL,
        "protocol_sha256": canonical_digest(ORACLE_PROTOCOL),
        "inputs": {
            "cache_dir": str(paths.cache_dir.resolve()),
            "calibration_report": str(paths.calibration_report.resolve()),
            "calibration_report_sha256": CALIBRATION_REPORT_SHA256,
        },
        "checkpoints": frozen_checkpoints,
        "scene_provenance": observed_provenance,
        "scene_provenance_digest": canonical_digest(observed_provenance),
        "rr_reproducibility": rr_verification,
        "score_caches": cache_records,
        "rows": rows,
        "summaries": {arm: summarize_arm(rows[arm]) for arm in ARMS},
        "comparisons": {
            "RC_minus_RR": rc_comparison,
            "CC_minus_RR": cc_comparison,
        },
        "decision": decision,
        "scoring_code_provenance": frozen_scoring_code,
        "scoring_code_provenance_digest": canonical_digest(frozen_scoring_code),
        "code_provenance": frozen_code,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    write_json_atomic(paths.report, report)
    print(
        json.dumps(
            {
                "report": str(paths.report.resolve()),
                "decision": decision["status"],
                "route": decision["route"],
                "cc_minus_rr_solve": cc_comparison["metrics"]["solve_only_ssim"][
                    "mean_delta"
                ],
                "cc_minus_rr_final": cc_comparison["metrics"]["final_ssim"][
                    "mean_delta"
                ],
            }
        ),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen clean-score oracle; all experiment choices are hard-coded."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument(
        "--ranker-checkpoint", type=Path, default=DEFAULT_RANKER_CHECKPOINT
    )
    parser.add_argument(
        "--affinity-primary", type=Path, default=DEFAULT_AFFINITY_PRIMARY
    )
    parser.add_argument(
        "--affinity-secondary", type=Path, default=DEFAULT_AFFINITY_SECONDARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = OraclePaths(
        cache_dir=args.cache_dir,
        calibration_report=args.calibration_report,
        ranker_checkpoint=args.ranker_checkpoint,
        affinity_primary=args.affinity_primary,
        affinity_secondary=args.affinity_secondary,
        output_dir=args.output_dir,
        report=args.report,
    )
    run_oracle(paths)


if __name__ == "__main__":
    main()
