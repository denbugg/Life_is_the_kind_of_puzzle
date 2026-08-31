"""Tile-preserving decoders with a weak train-population spatial unary.

Every decoder in this module returns a strict permutation of all 576 dirty
tiles.  The population atlas is learned from manifest-train targets and is used
only as an absolute-position unary; it never renders or substitutes pixels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from aiijc_puzzle.legacy_upgrade import (
    LayoutResult,
    best_buddy_components,
    directional_scores,
    layout_objective,
    validate_layout,
)
from aiijc_puzzle.novel_analog_layout import (
    analog_position_cost,
    robust_row_scale,
    tile_semantic_features,
)
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import GRID_SIZE, TILE_COUNT, assemble_tiles, split_tiles

ATLAS_WEIGHTS = (0.01, 0.03, 0.06, 0.12)
PRODUCTION_ATLAS_WEIGHT = 0.03
PRODUCTION_EDGE_BUDGET = 96
PRODUCTION_NLM_H = 10


def population_position_scores(tiles: np.ndarray, generic_tile_template: np.ndarray) -> np.ndarray:
    """Return a row-robust tile-to-position score from a train-only atlas."""

    query = tile_semantic_features(np.asarray(tiles)).copy()
    template = np.asarray(generic_tile_template, dtype=np.float32).copy()
    if template.shape != query.shape:
        raise ValueError(f"query/template feature mismatch: {query.shape}, {template.shape}")
    cost = analog_position_cost(query, template)
    scores = -robust_row_scale(cost)
    scores -= scores.mean(axis=1, keepdims=True)
    scores /= np.maximum(scores.std(axis=1, keepdims=True), 1e-5)
    return scores.astype(np.float32)


def _component_shift_score(
    component: dict[int, tuple[int, int]],
    board: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    position: np.ndarray,
    position_weight: float,
    row_shift: int,
    column_shift: int,
) -> float:
    score = 0.0
    for tile, (row, column) in component.items():
        row += row_shift
        column += column_shift
        score += position_weight * float(position[tile, row * GRID_SIZE + column])
        if column > 0 and board[row, column - 1] >= 0:
            score += float(right[board[row, column - 1], tile])
        if column < GRID_SIZE - 1 and board[row, column + 1] >= 0:
            score += float(right[tile, board[row, column + 1]])
        if row > 0 and board[row - 1, column] >= 0:
            score += float(down[board[row - 1, column], tile])
        if row < GRID_SIZE - 1 and board[row + 1, column] >= 0:
            score += float(down[tile, board[row + 1, column]])
    return score


def _pack_components_with_unary(
    components: list[dict[int, tuple[int, int]]],
    right: np.ndarray,
    down: np.ndarray,
    position: np.ndarray,
    position_weight: float,
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
                    (row + row_shift, column + column_shift)
                    for row, column in component.values()
                ]
                if any(board[row, column] >= 0 for row, column in coordinates):
                    continue
                score = _component_shift_score(
                    component,
                    board,
                    right,
                    down,
                    position,
                    position_weight,
                    row_shift,
                    column_shift,
                )
                if best_shift is None or score > best_score:
                    best_shift, best_score = (row_shift, column_shift), score
        if best_shift is None:
            continue
        for tile, (row, column) in component.items():
            board[row + best_shift[0], column + best_shift[1]] = tile
            used.add(tile)
    return board, used


def _fill_board_with_unary(
    board: np.ndarray,
    used: set[int],
    right: np.ndarray,
    down: np.ndarray,
    position: np.ndarray,
    position_weight: float,
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
        flat_position = int(row * GRID_SIZE + column)
        best_tile, best_score = -1, -np.inf
        for tile in remaining:
            score = position_weight * float(position[tile, flat_position])
            if column > 0 and board[row, column - 1] >= 0:
                score += float(right[board[row, column - 1], tile])
            if column < GRID_SIZE - 1 and board[row, column + 1] >= 0:
                score += float(right[tile, board[row, column + 1]])
            if row > 0 and board[row - 1, column] >= 0:
                score += float(down[board[row - 1, column], tile])
            if row < GRID_SIZE - 1 and board[row + 1, column] >= 0:
                score += float(down[tile, board[row + 1, column]])
            if score > best_score:
                best_tile, best_score = tile, score
        board[row, column] = best_tile
        remaining.remove(best_tile)
    return board.reshape(-1)


def solve_buddies_with_position(
    right: np.ndarray,
    down: np.ndarray,
    position: np.ndarray,
    *,
    position_weight: float,
    max_edges: int = 96,
) -> LayoutResult:
    """Pack bilateral best-buddy components using a weak absolute unary."""

    if not np.isfinite(position_weight) or position_weight < 0:
        raise ValueError("position_weight must be finite and non-negative")
    position_array = np.asarray(position, dtype=np.float32)
    if position_array.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(f"expected {(TILE_COUNT, TILE_COUNT)} position scores")
    started = perf_counter()
    components = best_buddy_components(right, down, max_edges=max_edges)
    board, used = _pack_components_with_unary(
        components,
        right,
        down,
        position_array,
        position_weight,
    )
    layout = validate_layout(
        _fill_board_with_unary(
            board,
            used,
            right,
            down,
            position_array,
            position_weight,
        )
    )
    objective = layout_objective(
        layout,
        right,
        down,
        position_array,
        position_weight=position_weight,
    )
    return LayoutResult(
        layout=layout,
        objective=objective,
        solver=f"buddies_{max_edges}_population_w{position_weight:g}",
        runtime_seconds=perf_counter() - started,
    )


def _tile_hashes(tiles: np.ndarray) -> list[str]:
    return sorted(
        hashlib.sha256(np.ascontiguousarray(tile).tobytes()).hexdigest() for tile in tiles
    )


@dataclass(frozen=True)
class PermutationAudit:
    """Machine-readable proof that raw output is an exact 576-tile reassembly."""

    grid_rows: int
    grid_columns: int
    tile_count: int
    unique_tile_indices: int
    missing_tile_indices: tuple[int, ...]
    duplicate_tile_indices: tuple[int, ...]
    exact_reassembly_from_declared_layout: bool
    input_output_tile_multiset_equal: bool
    raw_input_pixels_preserved: bool
    restoration_applied_after_audit: bool

    @property
    def passed(self) -> bool:
        return (
            self.grid_rows == GRID_SIZE
            and self.grid_columns == GRID_SIZE
            and self.tile_count == TILE_COUNT
            and self.unique_tile_indices == TILE_COUNT
            and not self.missing_tile_indices
            and not self.duplicate_tile_indices
            and self.exact_reassembly_from_declared_layout
            and self.input_output_tile_multiset_equal
            and self.raw_input_pixels_preserved
            and self.restoration_applied_after_audit
        )

    def as_dict(self) -> dict[str, object]:
        return {**self.__dict__, "passed": self.passed}


@dataclass(frozen=True)
class CompliantPrediction:
    """Production inference result with pre-restoration compliance evidence."""

    layout: np.ndarray
    raw: np.ndarray
    restored: np.ndarray
    audit: PermutationAudit
    score_seconds: float
    solve_seconds: float
    restoration_seconds: float
    atlas_weight: float
    edge_budget: int
    nlm_h: int


def audit_raw_permutation(
    input_image: np.ndarray,
    raw_output: np.ndarray,
    layout: np.ndarray,
    *,
    restoration_applied_after_audit: bool,
) -> PermutationAudit:
    """Audit bijection, exact rendering and tile-pixel multiset preservation."""

    layout_array = np.asarray(layout, dtype=np.int32)
    counts = np.bincount(
        layout_array[(layout_array >= 0) & (layout_array < TILE_COUNT)], minlength=TILE_COUNT
    )
    missing = tuple(int(index) for index in np.flatnonzero(counts == 0))
    duplicates = tuple(int(index) for index in np.flatnonzero(counts > 1))
    unique = int(np.count_nonzero(counts))
    input_tiles = split_tiles(input_image)
    output_tiles = split_tiles(raw_output)
    valid_shape = layout_array.shape == (TILE_COUNT,)
    expected = (
        assemble_tiles(input_tiles[layout_array])
        if valid_shape and not missing and not duplicates
        else None
    )
    exact = expected is not None and np.array_equal(expected, raw_output)
    multiset_equal = _tile_hashes(input_tiles) == _tile_hashes(output_tiles)
    return PermutationAudit(
        grid_rows=GRID_SIZE,
        grid_columns=GRID_SIZE,
        tile_count=int(len(layout_array)),
        unique_tile_indices=unique,
        missing_tile_indices=missing,
        duplicate_tile_indices=duplicates,
        exact_reassembly_from_declared_layout=bool(exact),
        input_output_tile_multiset_equal=bool(multiset_equal),
        raw_input_pixels_preserved=bool(multiset_equal and exact),
        restoration_applied_after_audit=bool(restoration_applied_after_audit),
    )


def predict_compliant_atlas(
    input_image: np.ndarray,
    generic_tile_template: np.ndarray,
    *,
    atlas_weight: float = PRODUCTION_ATLAS_WEIGHT,
    edge_budget: int = PRODUCTION_EDGE_BUDGET,
    nlm_h: int = PRODUCTION_NLM_H,
) -> CompliantPrediction:
    """Run the selected target-free tile layout, audit it, then restore quality."""

    tiles = split_tiles(input_image)
    score_started = perf_counter()
    right, down = directional_scores(tiles, views=("bilateral",))["bilateral"]
    position = population_position_scores(tiles, generic_tile_template)
    score_seconds = perf_counter() - score_started
    solved = solve_buddies_with_position(
        right,
        down,
        position,
        position_weight=atlas_weight,
        max_edges=edge_budget,
    )
    raw = assemble_tiles(tiles[solved.layout])
    audit = audit_raw_permutation(
        input_image,
        raw,
        solved.layout,
        restoration_applied_after_audit=True,
    )
    if not audit.passed:
        raise RuntimeError(f"production permutation audit failed: {audit.as_dict()}")
    restored = apply_nlm_color(raw, h=nlm_h)
    return CompliantPrediction(
        layout=solved.layout,
        raw=raw,
        restored=restored.image,
        audit=audit,
        score_seconds=score_seconds,
        solve_seconds=solved.runtime_seconds,
        restoration_seconds=restored.seconds,
        atlas_weight=float(atlas_weight),
        edge_budget=int(edge_budget),
        nlm_h=int(nlm_h),
    )
