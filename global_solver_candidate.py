"""Sparse relaxation-labeling solver for 24x24 directional jigsaws.

The solver treats every tile-to-position decision as a probability and passes
support through the cached right/down compatibility graph. It deliberately
does not inspect pixels, targets, truth layouts, or SSIM. A sequence of
increasingly sharp phases freezes only the most confident assignments; a final
Hungarian solve converts the doubly-stochastic belief matrix to a permutation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags

GRID = 24
N = GRID * GRID
POSITION_WEIGHT = 0.11


@dataclass(frozen=True)
class RelaxationPhase:
    temperature: float
    edge_weight: float
    inertia: float
    hard_mix: float
    iterations: int
    freeze_fraction: float


# Predeclared before metric evaluation. Early iterations preserve the global
# position prior; later phases give increasing weight to mutually supported
# directional neighborhoods and progressively sharpen the assignment.
PHASES = (
    RelaxationPhase(0.45, 1.50, 0.10, 0.55, 4, 0.00),
    RelaxationPhase(0.28, 3.00, 0.08, 0.70, 5, 0.03),
    RelaxationPhase(0.16, 6.00, 0.06, 0.85, 6, 0.08),
    RelaxationPhase(0.09, 10.0, 0.04, 0.94, 20, 0.15),
)
TOP_K_EDGES = 12
SINKHORN_STEPS = 14
EPS = 1e-12


def objective(layout, right, down, weighted_pos):
    """Return the cached-score objective used by the former SA solver."""
    board = np.asarray(layout, np.int32).reshape(GRID, GRID)
    positions = np.arange(N, dtype=np.int32).reshape(GRID, GRID)
    score = float(weighted_pos[board, positions].sum())
    score += float(right[board[:, :-1], board[:, 1:]].sum())
    score += float(down[board[:-1], board[1:]].sum())
    return score


def _row_normalize_sparse(matrix: csr_matrix) -> csr_matrix:
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(totals, dtype=np.float64)
    np.divide(1.0, totals, out=inverse, where=totals > 0)
    return (diags(inverse) @ matrix).tocsr()


def _topk_compatibility(scores: np.ndarray, top_k: int) -> tuple[csr_matrix, csr_matrix]:
    """Build normalized outgoing/incoming support graphs from dense log scores."""
    scores = np.asarray(scores, np.float64)
    if scores.shape != (N, N):
        raise ValueError(f"expected {(N, N)} directional scores, got {scores.shape}")
    rank_scores = scores.copy()
    np.fill_diagonal(rank_scores, -np.inf)
    k = min(max(1, int(top_k)), N - 1)
    # A useful global edge must be plausible both for its source row and its
    # destination column. This suppresses the popular-neighbour hubs that make
    # one-way top-k relaxation collapse onto mutually inconsistent labels.
    row_relative = rank_scores - np.max(rank_scores, axis=1, keepdims=True)
    column_relative = rank_scores - np.max(rank_scores, axis=0, keepdims=True)
    joint_scores = row_relative + column_relative
    columns = np.argpartition(joint_scores, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(N, dtype=np.int32), k)
    columns_flat = columns.reshape(-1)
    selected = joint_scores[rows, columns_flat].reshape(N, k)
    # Tempering keeps plausible alternatives alive instead of letting a noisy
    # top-1 edge monopolize support. Row maxima are subtracted for stability.
    selected = np.exp((selected - selected.max(axis=1, keepdims=True)) / 0.75)
    selected /= np.maximum(selected.sum(axis=1, keepdims=True), EPS)
    outgoing = csr_matrix(
        (selected.reshape(-1), (rows, columns_flat)), shape=(N, N)
    )
    incoming = _row_normalize_sparse(outgoing.transpose().tocsr())
    return outgoing, incoming


def _masked_sinkhorn(
    logits: np.ndarray,
    temperature: float,
    locked_position: np.ndarray,
) -> np.ndarray:
    """Project logits to a doubly-stochastic belief matrix with fixed pairs."""
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


def _directional_support(
    beliefs: np.ndarray,
    right_out: csr_matrix,
    right_in: csr_matrix,
    down_out: csr_matrix,
    down_in: csr_matrix,
) -> np.ndarray:
    """Propagate sparse compatibility through all four grid directions."""
    board = beliefs.reshape(N, GRID, GRID)
    support = np.zeros_like(board)
    # Candidate tile t at (r,c) is supported by likely compatible tiles at its
    # four adjacent grid positions. Sparse matrices make this O(N^2*k).
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


def _assignment(logits: np.ndarray, locked_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _freeze_confident(
    logits: np.ndarray,
    locked_position: np.ndarray,
    fraction: float,
) -> None:
    """Freeze a conservative prefix of assignment-margin confident pairs."""
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
    for local_index in order[: target - already]:
        tile = int(free_tiles[local_index])
        locked_position[tile] = int(assigned[tile])


def _hard_beliefs(logits: np.ndarray, locked_position: np.ndarray) -> np.ndarray:
    """Return the globally optimal one-to-one labels for the current scores."""
    _, position_of_tile = _assignment(logits, locked_position)
    hard = np.zeros((N, N), dtype=np.float64)
    hard[np.arange(N), position_of_tile] = 1.0
    return hard


def _layout_from_position_of_tile(position_of_tile: np.ndarray) -> np.ndarray:
    layout = np.empty(N, dtype=np.int32)
    layout[position_of_tile] = np.arange(N, dtype=np.int32)
    return layout


def solve_layout(right, down, pos, seed):
    """Return tile-at-position permutation via global relaxation labeling."""
    right = np.asarray(right, np.float64)
    down = np.asarray(down, np.float64)
    unary = np.asarray(pos, np.float64)
    if unary.shape != (N, N):
        raise ValueError(f"expected {(N, N)} position scores, got {unary.shape}")

    right_out, right_in = _topk_compatibility(right, TOP_K_EDGES)
    down_out, down_in = _topk_compatibility(down, TOP_K_EDGES)
    weighted_pos = POSITION_WEIGHT * unary
    unary = unary - unary.max(axis=1, keepdims=True)
    # Seed is used only for deterministic tie-breaking, not stochastic search.
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
            candidate_objective = objective(
                candidate_layout, right, down, weighted_pos
            )
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

    # One final support pass lets newly frozen confident labels influence the
    # remaining ambiguous tiles before exact one-to-one finalization.
    support = _directional_support(beliefs, right_out, right_in, down_out, down_in)
    final_logits = POSITION_WEIGHT * unary + PHASES[-1].edge_weight * support
    _, position_of_tile = _assignment(final_logits, locked_position)
    candidate_layout = _layout_from_position_of_tile(position_of_tile)
    if objective(candidate_layout, right, down, weighted_pos) > best_objective:
        best_layout = candidate_layout
    return best_layout
