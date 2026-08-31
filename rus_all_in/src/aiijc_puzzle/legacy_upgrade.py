"""Runnable target-free layout baselines recovered from the historical repository.

The original repository contains several stronger-looking pipelines whose model
weights were never committed.  This module keeps only the parts that can be
executed from a contest input image alone:

* the E14 MGC + one-pixel SSD directional score;
* rank-late fusion of several inference-visible analytic views;
* the E11/E14 sparse relaxation and Hungarian decoder;
* the ORBIT ``best buddies`` component decoder;
* the frozen coloured NLM output tail.

Targets are accepted only by the experiment runner, never by this module.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags
from scipy.special import log_softmax

from aiijc_puzzle.candidate_supply import analytic_views, classical_costs
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    GRID_SIZE,
    IMAGE_SIZE,
    TILE_COUNT,
    assemble_tiles,
    split_tiles,
)

EPS = 1e-12
TOP_K_EDGES = 12
SINKHORN_STEPS = 14


@dataclass(frozen=True)
class RelaxationPhase:
    """One frozen E11/E14 relaxation phase."""

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


@dataclass(frozen=True)
class LayoutResult:
    """A complete target-free board and basic solver diagnostics."""

    layout: np.ndarray
    objective: float
    solver: str
    runtime_seconds: float


@dataclass(frozen=True)
class Prediction:
    """End-to-end RGB prediction returned by :func:`predict`."""

    raw: np.ndarray
    restored: np.ndarray
    layout: np.ndarray
    score_seconds: float
    solve_seconds: float
    nlm_seconds: float
    objective: float
    score_variant: str
    solver: str


def constant_prediction(
    image: np.ndarray,
    *,
    statistic: str = "median",
    per_channel: bool = True,
) -> np.ndarray:
    """Return a target-free constant frame estimated from the dirty input.

    Shuffling preserves the global pixel population.  A constant output removes
    all false high-frequency structure, which the contest's local SSIM penalises
    more severely than loss of texture on many natural photographs.
    """

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 480x480 RGB input, got {value.dtype} {value.shape}")
    flat = value.reshape(-1, 3)
    if statistic == "median":
        color = np.median(flat, axis=0)
    elif statistic == "mean":
        color = flat.mean(axis=0)
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    if not per_channel:
        color = np.full(3, float(np.mean(color)))
    color_u8 = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return np.broadcast_to(color_u8, value.shape).copy()


def low_frequency_prediction(image: np.ndarray, *, sigma: float = 100.0) -> np.ndarray:
    """Suppress shuffled tile texture with a deterministic broad Gaussian tail."""

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 480x480 RGB input, got {value.dtype} {value.shape}")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be finite and positive, got {sigma}")
    return np.asarray(cv2.GaussianBlur(value, (0, 0), sigma), dtype=np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    """Encode one strict RGB prediction with deterministic Pillow settings."""

    value = np.asarray(image)
    if value.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or value.dtype != np.uint8:
        raise ValueError(f"expected uint8 480x480 RGB prediction, got {value.dtype} {value.shape}")
    stream = io.BytesIO()
    Image.fromarray(value, mode="RGB").save(
        stream,
        format="PNG",
        optimize=False,
        compress_level=6,
    )
    return stream.getvalue()


def atomic_write_png(path: Path, image: np.ndarray) -> str:
    """Atomically write a prediction and return its file SHA-256."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = png_bytes(image)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def deterministic_submission_zip(
    output_dir: Path,
    filenames: list[str],
    output_zip: Path,
) -> str:
    """Create a root-only deterministic ZIP from an exact ordered filename list."""

    if not filenames or len(filenames) != len(set(filenames)):
        raise ValueError("submission filenames must be a non-empty unique list")
    if filenames != sorted(filenames):
        raise ValueError("submission filenames must be sorted")
    if any(Path(name).name != name or Path(name).suffix.lower() != ".png" for name in filenames):
        raise ValueError("submission entries must be PNG basenames")
    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_zip.name}.", dir=output_zip.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in filenames:
                path = output_dir / name
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                        raise ValueError(f"invalid submission PNG {path}")
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        os.replace(temporary, output_zip)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output_zip.read_bytes()).hexdigest()


def _validate_square(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"expected {(TILE_COUNT, TILE_COUNT)} matrix, got {value.shape}")
    return value


def cost_to_logp(cost: np.ndarray) -> np.ndarray:
    """Convert a directional dissimilarity into stable row log-probabilities."""

    value = _validate_square(cost).copy()
    diagonal = np.eye(TILE_COUNT, dtype=bool)
    off_diagonal = value[~diagonal].reshape(TILE_COUNT, TILE_COUNT - 1)
    median = np.median(off_diagonal, axis=1, keepdims=True)
    mad = np.median(np.abs(off_diagonal - median), axis=1, keepdims=True)
    logits = -(value - median) / np.maximum(mad, 1e-6)
    np.fill_diagonal(logits, -1e4)
    return log_softmax(logits, axis=1).astype(np.float32)


def directional_scores(
    tiles: np.ndarray,
    *,
    views: tuple[str, ...] = ("raw", "bilateral"),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-view and fused target-free E14 directional score matrices.

    Fusion happens after each view has been converted to row log-probabilities.
    This avoids allowing one view's arbitrary cost scale to dominate another.
    """

    if not views:
        raise ValueError("at least one analytic view is required")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, transformed in analytic_views(tiles, views).items():
        right_cost, down_cost = classical_costs(transformed)
        result[name] = (cost_to_logp(right_cost), cost_to_logp(down_cost))
    if len(result) > 1:
        right = np.mean([pair[0] for pair in result.values()], axis=0, dtype=np.float64)
        down = np.mean([pair[1] for pair in result.values()], axis=0, dtype=np.float64)
        result["mean"] = (right.astype(np.float32), down.astype(np.float32))
    return result


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - np.median(values)) / (
        1.4826 * np.median(np.abs(values - np.median(values))) + 1e-6
    )


def border_position_scores(right: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Construct a weak inference-visible frame unary from missing-edge evidence.

    A tile is plausible on a border when the corresponding incoming/outgoing
    relation has no convincing partner.  The unary deliberately distinguishes
    only the four frame sides; it does not infer a clean target or use filenames.
    """

    right = _validate_square(right).astype(np.float64)
    down = _validate_square(down).astype(np.float64)
    masked_right = right.copy()
    masked_down = down.copy()
    np.fill_diagonal(masked_right, -np.inf)
    np.fill_diagonal(masked_down, -np.inf)
    # Negated best compatibility: weakly matchable sides are border candidates.
    left = _zscore(-np.max(masked_right, axis=0))
    right_edge = _zscore(-np.max(masked_right, axis=1))
    top = _zscore(-np.max(masked_down, axis=0))
    bottom = _zscore(-np.max(masked_down, axis=1))
    cells = np.arange(TILE_COUNT)
    rows, columns = divmod(cells, GRID_SIZE)
    unary = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
    unary[:, columns == 0] += left[:, None]
    unary[:, columns == GRID_SIZE - 1] += right_edge[:, None]
    unary[:, rows == 0] += top[:, None]
    unary[:, rows == GRID_SIZE - 1] += bottom[:, None]
    # Interior positions remain neutral.  Row scaling makes this compatible
    # with the historical E14 position term.
    unary -= unary.mean(axis=1, keepdims=True)
    unary /= unary.std(axis=1, keepdims=True) + 1e-6
    return unary.astype(np.float32)


def layout_objective(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    position: np.ndarray | None = None,
    *,
    position_weight: float = 0.0,
) -> float:
    """Evaluate a complete board under directional and optional unary scores."""

    board = validate_layout(layout).reshape(GRID_SIZE, GRID_SIZE)
    score = float(right[board[:, :-1], board[:, 1:]].sum(dtype=np.float64))
    score += float(down[board[:-1], board[1:]].sum(dtype=np.float64))
    if position is not None and position_weight:
        score += position_weight * float(
            position[board.reshape(-1), np.arange(TILE_COUNT)].sum(dtype=np.float64)
        )
    return score


def _row_normalize_sparse(matrix: csr_matrix) -> csr_matrix:
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros_like(totals, dtype=np.float64)
    np.divide(1.0, totals, out=inverse, where=totals > 0)
    return (diags(inverse) @ matrix).tocsr()


def _topk_compatibility(scores: np.ndarray, top_k: int) -> tuple[csr_matrix, csr_matrix]:
    values = _validate_square(scores).astype(np.float64)
    np.fill_diagonal(values, -np.inf)
    k = min(max(1, int(top_k)), TILE_COUNT - 1)
    row_relative = values - np.max(values, axis=1, keepdims=True)
    column_relative = values - np.max(values, axis=0, keepdims=True)
    joint = row_relative + column_relative
    columns = np.argpartition(joint, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(TILE_COUNT, dtype=np.int32), k)
    flat_columns = columns.reshape(-1)
    selected = joint[rows, flat_columns].reshape(TILE_COUNT, k)
    selected = np.exp((selected - selected.max(axis=1, keepdims=True)) / 0.75)
    selected /= np.maximum(selected.sum(axis=1, keepdims=True), EPS)
    outgoing = csr_matrix((selected.reshape(-1), (rows, flat_columns)), shape=values.shape)
    return outgoing, _row_normalize_sparse(outgoing.transpose().tocsr())


def _masked_sinkhorn(
    logits: np.ndarray,
    temperature: float,
    locked_position: np.ndarray,
) -> np.ndarray:
    beliefs = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
    locked_tiles = np.flatnonzero(locked_position >= 0)
    if locked_tiles.size:
        beliefs[locked_tiles, locked_position[locked_tiles]] = 1.0
    free_tiles = np.flatnonzero(locked_position < 0)
    occupied = locked_position[locked_tiles]
    free_positions = np.setdiff1d(np.arange(TILE_COUNT, dtype=np.int32), occupied)
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
    board = beliefs.reshape(TILE_COUNT, GRID_SIZE, GRID_SIZE)
    support = np.zeros_like(board)
    support[:, :, :-1] += (right_out @ board[:, :, 1:].reshape(TILE_COUNT, -1)).reshape(
        TILE_COUNT, GRID_SIZE, GRID_SIZE - 1
    )
    support[:, :, 1:] += (right_in @ board[:, :, :-1].reshape(TILE_COUNT, -1)).reshape(
        TILE_COUNT, GRID_SIZE, GRID_SIZE - 1
    )
    support[:, :-1, :] += (down_out @ board[:, 1:, :].reshape(TILE_COUNT, -1)).reshape(
        TILE_COUNT, GRID_SIZE - 1, GRID_SIZE
    )
    support[:, 1:, :] += (down_in @ board[:, :-1, :].reshape(TILE_COUNT, -1)).reshape(
        TILE_COUNT, GRID_SIZE - 1, GRID_SIZE
    )
    degree = np.full((GRID_SIZE, GRID_SIZE), 4.0, dtype=np.float64)
    degree[[0, -1], :] -= 1.0
    degree[:, [0, -1]] -= 1.0
    support /= degree[None]
    return support.reshape(TILE_COUNT, TILE_COUNT)


def _assignment(logits: np.ndarray, locked_position: np.ndarray) -> np.ndarray:
    constrained = np.asarray(logits, dtype=np.float64).copy()
    locked_tiles = np.flatnonzero(locked_position >= 0)
    if locked_tiles.size:
        occupied = locked_position[locked_tiles]
        constrained[locked_tiles] = -1e12
        constrained[:, occupied] = -1e12
        constrained[locked_tiles, occupied] = 1e12
    tiles, positions = linear_sum_assignment(-constrained)
    position_of_tile = np.empty(TILE_COUNT, dtype=np.int32)
    position_of_tile[tiles] = positions
    return position_of_tile


def _hard_beliefs(logits: np.ndarray, locked_position: np.ndarray) -> np.ndarray:
    position_of_tile = _assignment(logits, locked_position)
    result = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
    result[np.arange(TILE_COUNT), position_of_tile] = 1.0
    return result


def _layout_from_positions(position_of_tile: np.ndarray) -> np.ndarray:
    layout = np.empty(TILE_COUNT, dtype=np.int32)
    layout[position_of_tile] = np.arange(TILE_COUNT, dtype=np.int32)
    return layout


def _freeze_confident(
    logits: np.ndarray,
    locked_position: np.ndarray,
    fraction: float,
) -> None:
    target = int(round(fraction * TILE_COUNT))
    already = int(np.count_nonzero(locked_position >= 0))
    if target <= already:
        return
    assigned = _assignment(logits, locked_position)
    free_tiles = np.flatnonzero(locked_position < 0)
    occupied = set(locked_position[locked_position >= 0].tolist())
    free_positions = np.asarray([p for p in range(TILE_COUNT) if p not in occupied])
    block = logits[np.ix_(free_tiles, free_positions)].copy()
    assigned_columns = np.searchsorted(free_positions, assigned[free_tiles])
    chosen = block[np.arange(len(free_tiles)), assigned_columns]
    block[np.arange(len(free_tiles)), assigned_columns] = -np.inf
    margins = chosen - block.max(axis=1)
    order = np.argsort(-margins, kind="stable")
    for local_index in order[: target - already]:
        tile = int(free_tiles[local_index])
        locked_position[tile] = int(assigned[tile])


def solve_relaxation(
    right: np.ndarray,
    down: np.ndarray,
    *,
    position: np.ndarray | None = None,
    position_weight: float = 0.11,
    seed: int = 20260829,
) -> LayoutResult:
    """Run the historical E11/E14 sparse relaxation without learned artifacts."""

    started = perf_counter()
    right = _validate_square(right).astype(np.float64)
    down = _validate_square(down).astype(np.float64)
    if position is None:
        position = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
        position_weight = 0.0
    else:
        position = _validate_square(position).astype(np.float64)
    right_out, right_in = _topk_compatibility(right, TOP_K_EDGES)
    down_out, down_in = _topk_compatibility(down, TOP_K_EDGES)
    unary = position - position.max(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    tie_break = rng.uniform(-1e-7, 1e-7, size=(TILE_COUNT, TILE_COUNT))
    locked = np.full(TILE_COUNT, -1, dtype=np.int32)
    logits = position_weight * unary + tie_break
    initial = _assignment(logits, locked)
    best_layout = _layout_from_positions(initial)
    best_objective = layout_objective(
        best_layout, right, down, position, position_weight=position_weight
    )
    soft = _masked_sinkhorn(logits, PHASES[0].temperature, locked)
    beliefs = 0.45 * soft + 0.55 * _hard_beliefs(logits, locked)
    for phase in PHASES:
        for _ in range(phase.iterations):
            support = _directional_support(beliefs, right_out, right_in, down_out, down_in)
            logits = (
                position_weight * unary
                + phase.edge_weight * support
                + phase.inertia * np.log(np.maximum(beliefs, EPS))
                + tie_break
            )
            candidate = _layout_from_positions(_assignment(logits, locked))
            candidate_objective = layout_objective(
                candidate, right, down, position, position_weight=position_weight
            )
            if candidate_objective > best_objective:
                best_layout, best_objective = candidate, candidate_objective
            soft = _masked_sinkhorn(logits, phase.temperature, locked)
            hard = _hard_beliefs(logits, locked)
            beliefs = (1.0 - phase.hard_mix) * soft + phase.hard_mix * hard
        _freeze_confident(logits, locked, phase.freeze_fraction)
        soft = _masked_sinkhorn(logits, phase.temperature, locked)
        hard = _hard_beliefs(logits, locked)
        beliefs = (1.0 - phase.hard_mix) * soft + phase.hard_mix * hard
    support = _directional_support(beliefs, right_out, right_in, down_out, down_in)
    final_logits = position_weight * unary + PHASES[-1].edge_weight * support
    final_layout = _layout_from_positions(_assignment(final_logits, locked))
    final_objective = layout_objective(
        final_layout, right, down, position, position_weight=position_weight
    )
    if final_objective > best_objective:
        best_layout, best_objective = final_layout, final_objective
    return LayoutResult(
        validate_layout(best_layout),
        float(best_objective),
        "e14_relaxation",
        perf_counter() - started,
    )


class _ComponentBuilder:
    def __init__(self) -> None:
        self.frag_component: dict[int, int] = {}
        self.components: list[dict[int, tuple[int, int]]] = []

    @staticmethod
    def _span_ok(component: dict[int, tuple[int, int]]) -> bool:
        rows, columns = zip(*component.values(), strict=True)
        return max(rows) - min(rows) < GRID_SIZE and max(columns) - min(columns) < GRID_SIZE

    def add_edge(self, source: int, target: int, delta: tuple[int, int]) -> bool:
        source_component = self.frag_component.get(source)
        target_component = self.frag_component.get(target)
        if source_component is None and target_component is None:
            component_id = len(self.components)
            self.components.append({source: (0, 0), target: delta})
            self.frag_component[source] = component_id
            self.frag_component[target] = component_id
            return True
        if source_component is not None and target_component is None:
            component = self.components[source_component]
            source_position = component[source]
            position = (source_position[0] + delta[0], source_position[1] + delta[1])
            if position in component.values():
                return False
            component[target] = position
            if not self._span_ok(component):
                del component[target]
                return False
            self.frag_component[target] = source_component
            return True
        if source_component is None and target_component is not None:
            component = self.components[target_component]
            target_position = component[target]
            position = (target_position[0] - delta[0], target_position[1] - delta[1])
            if position in component.values():
                return False
            component[source] = position
            if not self._span_ok(component):
                del component[source]
                return False
            self.frag_component[source] = target_component
            return True
        assert source_component is not None and target_component is not None
        if source_component == target_component:
            source_position = self.components[source_component][source]
            target_position = self.components[source_component][target]
            return (
                target_position[0] - source_position[0],
                target_position[1] - source_position[1],
            ) == delta
        left = self.components[source_component]
        other = self.components[target_component]
        source_position, target_position = left[source], other[target]
        shift = (
            source_position[0] + delta[0] - target_position[0],
            source_position[1] + delta[1] - target_position[1],
        )
        moved = {tile: (r + shift[0], c + shift[1]) for tile, (r, c) in other.items()}
        if set(left.values()) & set(moved.values()):
            return False
        merged = {**left, **moved}
        if not self._span_ok(merged):
            return False
        self.components[source_component] = merged
        self.components[target_component] = {}
        for tile in moved:
            self.frag_component[tile] = source_component
        return True


def best_buddy_components(
    right: np.ndarray,
    down: np.ndarray,
    *,
    max_edges: int = 96,
) -> list[dict[int, tuple[int, int]]]:
    """Recover conflict-safe ORBIT components from mutual top-one edges."""

    candidates: list[tuple[float, int, int, tuple[int, int]]] = []
    for scores, delta in ((right, (0, 1)), (down, (1, 0))):
        matrix = _validate_square(scores).copy()
        np.fill_diagonal(matrix, -np.inf)
        row_best = np.argmax(matrix, axis=1)
        column_best = np.argmax(matrix, axis=0)
        for source, target in enumerate(row_best):
            if column_best[target] == source:
                candidates.append((float(matrix[source, target]), source, int(target), delta))
    candidates.sort(reverse=True)
    builder = _ComponentBuilder()
    for _, source, target, delta in candidates[:max_edges]:
        builder.add_edge(source, target, delta)
    return [component for component in builder.components if component]


def _component_shift_score(
    component: dict[int, tuple[int, int]],
    board: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    row_shift: int,
    column_shift: int,
) -> float:
    score = 0.0
    for tile, (row, column) in component.items():
        row += row_shift
        column += column_shift
        if column > 0 and board[row, column - 1] >= 0:
            score += right[board[row, column - 1], tile]
        if column < GRID_SIZE - 1 and board[row, column + 1] >= 0:
            score += right[tile, board[row, column + 1]]
        if row > 0 and board[row - 1, column] >= 0:
            score += down[board[row - 1, column], tile]
        if row < GRID_SIZE - 1 and board[row + 1, column] >= 0:
            score += down[tile, board[row + 1, column]]
    return float(score)


def _pack_components(
    components: list[dict[int, tuple[int, int]]],
    right: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, set[int]]:
    board = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int32)
    used: set[int] = set()
    for raw_component in sorted(components, key=len, reverse=True):
        min_row = min(row for row, _ in raw_component.values())
        min_column = min(column for _, column in raw_component.values())
        component = {
            tile: (row - min_row, column - min_column)
            for tile, (row, column) in raw_component.items()
        }
        max_row = max(row for row, _ in component.values())
        max_column = max(column for _, column in component.values())
        best_shift: tuple[int, int] | None = None
        best_score = -np.inf
        for row_shift in range(GRID_SIZE - max_row):
            for column_shift in range(GRID_SIZE - max_column):
                coordinates = [
                    (row + row_shift, column + column_shift) for row, column in component.values()
                ]
                if any(board[row, column] >= 0 for row, column in coordinates):
                    continue
                score = _component_shift_score(
                    component, board, right, down, row_shift, column_shift
                )
                if best_shift is None or score > best_score:
                    best_shift, best_score = (row_shift, column_shift), score
        if best_shift is None:
            continue
        for tile, (row, column) in component.items():
            board[row + best_shift[0], column + best_shift[1]] = tile
            used.add(tile)
    return board, used


def _fill_board(
    board: np.ndarray,
    used: set[int],
    right: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    remaining = set(range(TILE_COUNT)) - used
    while remaining:
        empty = np.argwhere(board < 0)
        contacts = []
        for row, column in empty:
            contacts.append(
                int(column > 0 and board[row, column - 1] >= 0)
                + int(column < GRID_SIZE - 1 and board[row, column + 1] >= 0)
                + int(row > 0 and board[row - 1, column] >= 0)
                + int(row < GRID_SIZE - 1 and board[row + 1, column] >= 0)
            )
        row, column = empty[int(np.argmax(contacts))]
        best_tile, best_score = -1, -np.inf
        for tile in remaining:
            score = 0.0
            if column > 0 and board[row, column - 1] >= 0:
                score += right[board[row, column - 1], tile]
            if column < GRID_SIZE - 1 and board[row, column + 1] >= 0:
                score += right[tile, board[row, column + 1]]
            if row > 0 and board[row - 1, column] >= 0:
                score += down[board[row - 1, column], tile]
            if row < GRID_SIZE - 1 and board[row + 1, column] >= 0:
                score += down[tile, board[row + 1, column]]
            if score > best_score:
                best_tile, best_score = tile, score
        board[row, column] = best_tile
        remaining.remove(best_tile)
    return board.reshape(-1)


def solve_buddies(
    right: np.ndarray,
    down: np.ndarray,
    *,
    max_edges: int = 96,
) -> LayoutResult:
    """Run the frozen rank-96 ORBIT component geometry without model weights."""

    started = perf_counter()
    components = best_buddy_components(right, down, max_edges=max_edges)
    board, used = _pack_components(components, right, down)
    layout = validate_layout(_fill_board(board, used, right, down))
    return LayoutResult(
        layout,
        layout_objective(layout, right, down),
        f"buddies_{max_edges}",
        perf_counter() - started,
    )


def validate_layout(layout: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous tile-at-position permutation."""

    value = np.asarray(layout, dtype=np.int32)
    if value.shape != (TILE_COUNT,):
        raise ValueError(f"expected layout shape {(TILE_COUNT,)}, got {value.shape}")
    if not np.array_equal(np.sort(value), np.arange(TILE_COUNT)):
        raise ValueError("layout is not a complete tile permutation")
    return np.ascontiguousarray(value)


def layout_digest(layout: np.ndarray) -> str:
    """Return a portable digest for a predicted permutation."""

    return hashlib.sha256(validate_layout(layout).astype("<i4").tobytes()).hexdigest()


def predict(
    image: np.ndarray,
    *,
    score_variant: str = "mean",
    solver: str = "relax_border",
    seed: int = 20260829,
    nlm_h: int = 9,
) -> Prediction:
    """Produce one target-free end-to-end prediction from a corrupted board."""

    tiles = split_tiles(image)
    score_started = perf_counter()
    scores = directional_scores(tiles)
    score_seconds = perf_counter() - score_started
    if score_variant not in scores:
        raise ValueError(f"unknown score variant {score_variant!r}; available={tuple(scores)}")
    right, down = scores[score_variant]
    if solver == "relax":
        layout_result = solve_relaxation(right, down, seed=seed)
    elif solver == "relax_border":
        layout_result = solve_relaxation(
            right,
            down,
            position=border_position_scores(right, down),
            seed=seed,
        )
    elif solver == "buddies96":
        layout_result = solve_buddies(right, down, max_edges=96)
    elif solver == "buddies256":
        layout_result = solve_buddies(right, down, max_edges=256)
    else:
        raise ValueError(f"unknown solver {solver!r}")
    raw = assemble_tiles(tiles[layout_result.layout])
    restored = apply_nlm_color(raw, h=nlm_h)
    return Prediction(
        raw=raw,
        restored=restored.image,
        layout=layout_result.layout,
        score_seconds=score_seconds,
        solve_seconds=layout_result.runtime_seconds,
        nlm_seconds=restored.seconds,
        objective=layout_result.objective,
        score_variant=score_variant,
        solver=solver,
    )
