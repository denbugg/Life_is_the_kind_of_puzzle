"""Target-free E14 score fusion and sparse global puzzle solver.

This module is deliberately self-contained for Kaggle packaging.  The score
construction matches the verified E14 experiment: a frozen 0.2 contribution
from raw-tile MGC+SSD scores followed by the unchanged E11 relaxation/Hungarian
solver.  Targets and image-quality metrics are never accepted as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags
from scipy.special import log_softmax

GRID, TILE, N = 24, 20, 576
ALPHA = 0.2
POSITION_WEIGHT = 0.11
TOP_K_EDGES = 12
SINKHORN_STEPS = 14
EPS = 1e-12

DUMMY_DIFFS = np.asarray(
    [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 1], [0, 1, 0],
     [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    dtype=np.float32,
)


def _mahalanobis_gradient_cost(source_boundary, source_inner, target_boundary,
                               *, batch_size=24):
    source_boundary = np.asarray(source_boundary, np.float32)
    source_inner = np.asarray(source_inner, np.float32)
    target_boundary = np.asarray(target_boundary, np.float32)
    gradients = source_boundary - source_inner
    means = gradients.mean(axis=1)
    dummy = np.broadcast_to(DUMMY_DIFFS, (N, *DUMMY_DIFFS.shape))
    samples = np.concatenate((gradients, dummy), axis=1).astype(np.float64)
    centered = samples - samples.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered, optimize=True) / (
        samples.shape[1] - 1
    )
    precisions = np.linalg.inv(covariance).astype(np.float32)
    costs = np.empty((N, N), np.float32)
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        residual = (
            target_boundary[None, :, :, :]
            - source_boundary[start:stop, None, :, :]
            - means[start:stop, None, None, :]
        )
        costs[start:stop] = np.einsum(
            "btkc,bcd,btkd->bt", residual, precisions[start:stop], residual,
            optimize=True,
        )
    return costs


def _ssd_cost(source_boundary, target_boundary, *, batch_size=24):
    source_boundary = np.asarray(source_boundary, np.float32)
    target_boundary = np.asarray(target_boundary, np.float32)
    costs = np.empty((N, N), np.float32)
    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        residual = source_boundary[start:stop, None] - target_boundary[None]
        costs[start:stop] = np.einsum(
            "btkc,btkc->bt", residual, residual, optimize=True
        )
    return costs


def _row_robust_dissimilarity(cost):
    cost = np.asarray(cost, np.float32)
    off_diagonal = cost[~np.eye(N, dtype=bool)].reshape(N, N - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    scaled = (cost - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(scaled, np.inf)
    return scaled


def _dissimilarity_logp(dissimilarity):
    dissimilarity = np.asarray(dissimilarity, np.float32)
    off_diagonal = dissimilarity[~np.eye(N, dtype=bool)].reshape(N, N - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    z = -(dissimilarity - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(z, -1e4)
    return log_softmax(z, axis=1).astype(np.float32)


def classical_mgc_ssd_scores(tiles):
    """Return right/down log-probabilities from inference-visible raw tiles."""
    tiles = np.asarray(tiles)
    if tiles.shape != (N, TILE, TILE, 3):
        raise ValueError(f"expected {(N, TILE, TILE, 3)} tiles, got {tiles.shape}")
    pixel = tiles.astype(np.float32)
    left, left_inner = pixel[:, :, 0, :], pixel[:, :, 1, :]
    right, right_inner = pixel[:, :, -1, :], pixel[:, :, -2, :]
    top, top_inner = pixel[:, 0, :, :], pixel[:, 1, :, :]
    bottom, bottom_inner = pixel[:, -1, :, :], pixel[:, -2, :, :]
    right_mgc = _mahalanobis_gradient_cost(right, right_inner, left)
    right_mgc += _mahalanobis_gradient_cost(left, left_inner, right).T
    down_mgc = _mahalanobis_gradient_cost(bottom, bottom_inner, top)
    down_mgc += _mahalanobis_gradient_cost(top, top_inner, bottom).T
    right_dissimilarity = 0.5 * (
        _row_robust_dissimilarity(right_mgc)
        + _row_robust_dissimilarity(_ssd_cost(right, left))
    )
    down_dissimilarity = 0.5 * (
        _row_robust_dissimilarity(down_mgc)
        + _row_robust_dissimilarity(_ssd_cost(bottom, top))
    )
    return (_dissimilarity_logp(right_dissimilarity),
            _dissimilarity_logp(down_dissimilarity))


def fuse_scores(learned, classical, *, alpha=ALPHA):
    """Apply the frozen E14 fusion formula and mask self-neighbours."""
    if alpha != ALPHA:
        raise ValueError(f"E14 locks alpha={ALPHA}, got {alpha}")
    fused = (1.0 - alpha) * np.asarray(learned) + alpha * np.asarray(classical)
    fused = np.asarray(fused, np.float32)
    np.fill_diagonal(fused, -1e4)
    return fused


def fused_directional_scores(raw_tiles, learned_right, learned_down,
                             *, learned_are_logp=True):
    """Build the two E14 directional matrices without target access."""
    learned_right = np.asarray(learned_right)
    learned_down = np.asarray(learned_down)
    if learned_right.shape != (N, N) or learned_down.shape != (N, N):
        raise ValueError("learned directional matrices must be 576x576")
    if not learned_are_logp:
        learned_right = log_softmax(learned_right, axis=1)
        learned_down = log_softmax(learned_down, axis=1)
    classical_right, classical_down = classical_mgc_ssd_scores(raw_tiles)
    return (fuse_scores(learned_right, classical_right),
            fuse_scores(learned_down, classical_down))


@dataclass(frozen=True)
class RelaxationPhase:
    temperature: float
    edge_weight: float
    inertia: float
    hard_mix: float
    iterations: int
    freeze_fraction: float


PHASES = (
    RelaxationPhase(0.45, 1.50, 0.10, 0.55, 4, 0.00),
    RelaxationPhase(0.28, 3.00, 0.08, 0.70, 5, 0.03),
    RelaxationPhase(0.16, 6.00, 0.06, 0.85, 6, 0.08),
    RelaxationPhase(0.09, 10.0, 0.04, 0.94, 20, 0.15),
)


def objective(layout, right, down, weighted_pos):
    board = np.asarray(layout, np.int32).reshape(GRID, GRID)
    positions = np.arange(N, dtype=np.int32).reshape(GRID, GRID)
    score = float(weighted_pos[board, positions].sum())
    score += float(right[board[:, :-1], board[:, 1:]].sum())
    score += float(down[board[:-1], board[1:]].sum())
    return score


def _row_normalize_sparse(matrix):
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(totals, dtype=np.float64)
    np.divide(1.0, totals, out=inverse, where=totals > 0)
    return (diags(inverse) @ matrix).tocsr()


def _topk_compatibility(scores, top_k):
    scores = np.asarray(scores, np.float64)
    if scores.shape != (N, N):
        raise ValueError(f"expected {(N, N)} directional scores, got {scores.shape}")
    rank_scores = scores.copy()
    np.fill_diagonal(rank_scores, -np.inf)
    k = min(max(1, int(top_k)), N - 1)
    row_relative = rank_scores - np.max(rank_scores, axis=1, keepdims=True)
    column_relative = rank_scores - np.max(rank_scores, axis=0, keepdims=True)
    joint_scores = row_relative + column_relative
    columns = np.argpartition(joint_scores, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(N, dtype=np.int32), k)
    columns_flat = columns.reshape(-1)
    selected = joint_scores[rows, columns_flat].reshape(N, k)
    selected = np.exp((selected - selected.max(axis=1, keepdims=True)) / 0.75)
    selected /= np.maximum(selected.sum(axis=1, keepdims=True), EPS)
    outgoing = csr_matrix(
        (selected.reshape(-1), (rows, columns_flat)), shape=(N, N)
    )
    incoming = _row_normalize_sparse(outgoing.transpose().tocsr())
    return outgoing, incoming


def _masked_sinkhorn(logits, temperature, locked_position):
    beliefs = np.zeros((N, N), dtype=np.float64)
    locked_tiles = np.flatnonzero(locked_position >= 0)
    if locked_tiles.size:
        beliefs[locked_tiles, locked_position[locked_tiles]] = 1.0
    free_tiles = np.flatnonzero(locked_position < 0)
    occupied = locked_position[locked_tiles]
    free_positions = np.setdiff1d(
        np.arange(N, dtype=np.int32), occupied, assume_unique=False
    )
    if not free_tiles.size:
        return beliefs
    block = logits[np.ix_(free_tiles, free_positions)] / temperature
    block -= block.max(axis=1, keepdims=True)
    block = np.exp(np.clip(block, -60.0, 0.0))
    for _ in range(SINKHORN_STEPS):
        block /= np.maximum(block.sum(axis=1, keepdims=True), EPS)
        block /= np.maximum(block.sum(axis=0, keepdims=True), EPS)
    block /= np.maximum(block.sum(axis=1, keepdims=True), EPS)
    beliefs[np.ix_(free_tiles, free_positions)] = block
    return beliefs


def _directional_support(beliefs, right_out, right_in, down_out, down_in):
    board = beliefs.reshape(N, GRID, GRID)
    support = np.zeros_like(board)
    support[:, :, :-1] += (right_out @ board[:, :, 1:].reshape(N, -1)).reshape(
        N, GRID, GRID - 1
    )
    support[:, :, 1:] += (right_in @ board[:, :, :-1].reshape(N, -1)).reshape(
        N, GRID, GRID - 1
    )
    support[:, :-1, :] += (down_out @ board[:, 1:, :].reshape(N, -1)).reshape(
        N, GRID - 1, GRID
    )
    support[:, 1:, :] += (down_in @ board[:, :-1, :].reshape(N, -1)).reshape(
        N, GRID - 1, GRID
    )
    degree = np.full((GRID, GRID), 4.0, dtype=np.float64)
    degree[0, :] -= 1.0
    degree[-1, :] -= 1.0
    degree[:, 0] -= 1.0
    degree[:, -1] -= 1.0
    support /= degree[None, :, :]
    return support.reshape(N, N)


def _assignment(logits, locked_position):
    constrained = np.asarray(logits, np.float64).copy()
    locked_tiles = np.flatnonzero(locked_position >= 0)
    if locked_tiles.size:
        occupied = locked_position[locked_tiles]
        constrained[locked_tiles, :] = -1e12
        constrained[:, occupied] = -1e12
        constrained[locked_tiles, occupied] = 1e12
    tiles, positions = linear_sum_assignment(-constrained)
    position_of_tile = np.empty(N, dtype=np.int32)
    position_of_tile[tiles] = positions
    return tiles.astype(np.int32), position_of_tile


def _freeze_confident(logits, locked_position, fraction):
    target = int(round(fraction * N))
    already = int(np.count_nonzero(locked_position >= 0))
    if target <= already:
        return
    _, assigned = _assignment(logits, locked_position)
    free_tiles = np.flatnonzero(locked_position < 0)
    occupied = set(locked_position[locked_position >= 0].tolist())
    free_positions = np.asarray([p for p in range(N) if p not in occupied], np.int32)
    block = logits[np.ix_(free_tiles, free_positions)].copy()
    assigned_columns = np.searchsorted(free_positions, assigned[free_tiles])
    chosen = block[np.arange(len(free_tiles)), assigned_columns]
    block[np.arange(len(free_tiles)), assigned_columns] = -np.inf
    margins = chosen - block.max(axis=1)
    order = np.argsort(-margins, kind="stable")
    for local_index in order[:target - already]:
        tile = int(free_tiles[local_index])
        locked_position[tile] = int(assigned[tile])


def _hard_beliefs(logits, locked_position):
    _, position_of_tile = _assignment(logits, locked_position)
    hard = np.zeros((N, N), dtype=np.float64)
    hard[np.arange(N), position_of_tile] = 1.0
    return hard


def _layout_from_position_of_tile(position_of_tile):
    layout = np.empty(N, dtype=np.int32)
    layout[position_of_tile] = np.arange(N, dtype=np.int32)
    return layout


def solve_layout(right, down, pos, seed):
    """Return a valid tile-at-position permutation via global relaxation."""
    right = np.asarray(right, np.float64)
    down = np.asarray(down, np.float64)
    unary = np.asarray(pos, np.float64)
    if unary.shape != (N, N):
        raise ValueError(f"expected {(N, N)} position scores, got {unary.shape}")
    right_out, right_in = _topk_compatibility(right, TOP_K_EDGES)
    down_out, down_in = _topk_compatibility(down, TOP_K_EDGES)
    weighted_pos = POSITION_WEIGHT * unary
    unary = unary - unary.max(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    tie_break = rng.uniform(-1e-7, 1e-7, size=(N, N))
    locked_position = np.full(N, -1, dtype=np.int32)
    logits = unary + tie_break
    _, initial_position_of_tile = _assignment(logits, locked_position)
    best_layout = _layout_from_position_of_tile(initial_position_of_tile)
    best_objective = objective(best_layout, right, down, weighted_pos)
    soft = _masked_sinkhorn(logits, PHASES[0].temperature, locked_position)
    beliefs = 0.45 * soft + 0.55 * _hard_beliefs(logits, locked_position)
    for phase in PHASES:
        for _ in range(phase.iterations):
            support = _directional_support(
                beliefs, right_out, right_in, down_out, down_in
            )
            logits = (
                POSITION_WEIGHT * unary
                + phase.edge_weight * support
                + phase.inertia * np.log(np.maximum(beliefs, EPS))
                + tie_break
            )
            _, position_of_tile = _assignment(logits, locked_position)
            candidate_layout = _layout_from_position_of_tile(position_of_tile)
            candidate_objective = objective(candidate_layout, right, down, weighted_pos)
            if candidate_objective > best_objective:
                best_objective = candidate_objective
                best_layout = candidate_layout
            soft = _masked_sinkhorn(logits, phase.temperature, locked_position)
            hard = _hard_beliefs(logits, locked_position)
            beliefs = (1.0 - phase.hard_mix) * soft + phase.hard_mix * hard
        _freeze_confident(logits, locked_position, phase.freeze_fraction)
        soft = _masked_sinkhorn(logits, phase.temperature, locked_position)
        hard = _hard_beliefs(logits, locked_position)
        beliefs = (1.0 - phase.hard_mix) * soft + phase.hard_mix * hard
    support = _directional_support(beliefs, right_out, right_in, down_out, down_in)
    final_logits = POSITION_WEIGHT * unary + PHASES[-1].edge_weight * support
    _, position_of_tile = _assignment(final_logits, locked_position)
    candidate_layout = _layout_from_position_of_tile(position_of_tile)
    if objective(candidate_layout, right, down, weighted_pos) > best_objective:
        best_layout = candidate_layout
    return best_layout


def is_valid_layout(layout):
    layout = np.asarray(layout)
    return layout.shape == (N,) and np.array_equal(np.sort(layout), np.arange(N))
