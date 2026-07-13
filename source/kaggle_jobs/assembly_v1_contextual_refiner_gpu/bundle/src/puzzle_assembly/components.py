"""Reciprocal/loop component growth and deterministic grid completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, validate_permutation
from .solvers import placement_unary, swap_refine


@dataclass(frozen=True)
class ProposedEdge:
    first: int
    second: int
    dx: int
    dy: int
    cost: float
    margin: float
    reciprocal: bool
    in_loop: bool


@dataclass(frozen=True)
class ComponentSolveResult:
    position_to_slot: np.ndarray
    accepted_edges: int
    proposed_edges: int
    component_sizes: tuple[int, ...]
    placed_component_tiles: int
    unresolved_tiles_before_assignment: int
    consensus_added_tiles: int


@dataclass(frozen=True)
class LPSolveResult:
    position_to_slot: np.ndarray
    proposed_edges: int
    component_sizes: tuple[int, ...]
    placed_component_tiles: int
    unresolved_tiles_before_assignment: int
    lp_failures: int


@dataclass(frozen=True)
class SuccessiveLPSolveResult:
    position_to_slot: np.ndarray
    iterations: int
    advanced_sides: int
    active_edges: int
    consistent_edges: int
    component_sizes: tuple[int, ...]
    placed_component_tiles: int
    unresolved_tiles_before_assignment: int


def _best_and_margin(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = np.argmin(matrix, axis=1).astype(np.int32)
    rows = np.arange(TILE_COUNT)
    best_cost = matrix[rows, best].astype(np.float32)
    two = np.partition(matrix, 1, axis=1)[:, :2]
    two.sort(axis=1)
    margin = (two[:, 1] - two[:, 0]).astype(np.float32)
    return best, best_cost, margin


def propose_reciprocal_edges(
    compatibility: CompatibilityMatrices,
    *,
    require_reciprocal: bool = True,
    include_verified_loops: bool = True,
    only_verified_loops: bool = False,
    min_margin: float = 0.0,
) -> list[ProposedEdge]:
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative")
    if only_verified_loops and not include_verified_loops:
        raise ValueError("only_verified_loops requires include_verified_loops")
    right_best, right_cost, right_margin = _best_and_margin(compatibility.right)
    down_best, down_cost, down_margin = _best_and_margin(compatibility.down)
    right_in = np.argmin(compatibility.right, axis=0)
    down_in = np.argmin(compatibility.down, axis=0)
    right_recip = right_in[right_best] == np.arange(TILE_COUNT)
    down_recip = down_in[down_best] == np.arange(TILE_COUNT)

    loop_edges: set[tuple[int, int, int, int]] = set()
    if include_verified_loops:
        for first in range(TILE_COUNT):
            right = int(right_best[first])
            down = int(down_best[first])
            corner_from_right = int(down_best[right])
            corner_from_down = int(right_best[down])
            if corner_from_right != corner_from_down:
                continue
            corner = corner_from_right
            if len({first, right, down, corner}) != 4:
                continue
            if not (
                right_recip[first]
                and down_recip[first]
                and down_recip[right]
                and right_recip[down]
            ):
                continue
            loop_edges.update(
                {
                    (first, right, 1, 0),
                    (first, down, 0, 1),
                    (right, corner, 0, 1),
                    (down, corner, 1, 0),
                }
            )

    proposals: list[ProposedEdge] = []
    for direction, best, costs, margins, reciprocal in (
        ((1, 0), right_best, right_cost, right_margin, right_recip),
        ((0, 1), down_best, down_cost, down_margin, down_recip),
    ):
        dx, dy = direction
        for first in range(TILE_COUNT):
            if require_reciprocal and not bool(reciprocal[first]):
                continue
            # A zero-margin proposal is an arbitrary tie, which is especially
            # common on true outer boundaries.  Treating it as geometry can
            # merge otherwise correct components through a non-existent seam.
            if float(margins[first]) <= min_margin:
                continue
            second = int(best[first])
            in_loop = (first, second, dx, dy) in loop_edges
            if only_verified_loops and not in_loop:
                continue
            proposals.append(
                ProposedEdge(
                    first=first,
                    second=second,
                    dx=dx,
                    dy=dy,
                    cost=float(costs[first]),
                    margin=float(margins[first]),
                    reciprocal=bool(reciprocal[first]),
                    in_loop=in_loop,
                )
            )
    proposals.sort(
        key=lambda edge: (
            0 if edge.in_loop else 1,
            0 if edge.reciprocal else 1,
            edge.cost,
            -edge.margin,
            edge.first,
            edge.second,
            edge.dy,
            edge.dx,
        )
    )
    return proposals


def propose_mutual_topk_edges(
    compatibility: CompatibilityMatrices, *, top_k: int
) -> list[ProposedEdge]:
    """Propose row/column mutual top-k edges for geometric conflict filtering."""
    if not 1 <= top_k < TILE_COUNT:
        raise ValueError("top_k must be in [1, 575]")
    proposals = []
    for (dx, dy), matrix in (
        ((1, 0), compatibility.right),
        ((0, 1), compatibility.down),
    ):
        row_order = np.argsort(matrix, axis=1, kind="stable")[:, :top_k]
        column_order = np.argsort(matrix, axis=0, kind="stable")[:top_k]
        column_rank = np.full((TILE_COUNT, TILE_COUNT), top_k, dtype=np.int16)
        for rank in range(top_k):
            column_rank[column_order[rank], np.arange(TILE_COUNT)] = rank
        for first in range(TILE_COUNT):
            for row_rank, second in enumerate(row_order[first].tolist()):
                if second == first:
                    continue
                reverse_rank = int(column_rank[first, second])
                if reverse_rank >= top_k:
                    continue
                normalized = (row_rank + reverse_rank) / float(max(2 * top_k - 2, 1))
                proposals.append(
                    ProposedEdge(
                        first=first,
                        second=int(second),
                        dx=dx,
                        dy=dy,
                        cost=float(normalized),
                        margin=float(1.0 - normalized),
                        reciprocal=row_rank == 0 and reverse_rank == 0,
                        in_loop=False,
                    )
                )
    proposals.sort(
        key=lambda edge: (
            edge.cost,
            0 if edge.reciprocal else 1,
            edge.first,
            edge.second,
            edge.dy,
            edge.dx,
        )
    )
    return proposals


def _directional_rank_costs(
    matrix: np.ndarray, *, reciprocal_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return outgoing/incoming orders and a smooth bidirectional rank cost.

    The matrix stores an oriented relation (right-to-left or bottom-to-top).
    Row ranks therefore describe the requested outgoing side, while column
    ranks describe the matching incoming side.  Combining both ranks gives us
    four-side evidence without materialising separate left/up matrices.
    """
    if not 0.0 <= reciprocal_weight <= 1.0:
        raise ValueError("reciprocal_weight must be in [0, 1]")
    row_order = np.argsort(matrix, axis=1, kind="stable")
    column_order = np.argsort(matrix, axis=0, kind="stable")
    row_rank = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.int16)
    column_rank = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.int16)
    ranks = np.arange(TILE_COUNT, dtype=np.int16)
    row_rank[np.arange(TILE_COUNT)[:, None], row_order] = ranks[None, :]
    column_rank[column_order, np.arange(TILE_COUNT)[None, :]] = ranks[:, None]

    # log1p retains useful resolution among the first few candidates without
    # letting a merely mediocre reciprocal rank dominate an excellent row
    # rank.  Costs stay in [0, 1].
    normalizer = float(np.log1p(TILE_COUNT - 1))
    row_cost = np.log1p(row_rank.astype(np.float32)) / normalizer
    column_cost = np.log1p(column_rank.astype(np.float32)) / normalizer
    cost = (
        (1.0 - reciprocal_weight) * row_cost
        + reciprocal_weight * column_cost
    ).astype(np.float32)
    return row_order, column_order, row_rank, cost


def propose_soft_cycle_edges(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 16,
    keep_per_tile: int = 2,
    reciprocal_weight: float = 0.35,
    loop_weight: float = 1.0,
) -> list[ProposedEdge]:
    """Rank top-k edges by soft 2x2 cycle consistency.

    A horizontal proposal ``a -> b`` is supported when a plausible pair below
    (or above) it closes a 2x2 square.  Vertical proposals analogously seek a
    square on either side.  Unlike the old verified-loop path, every leg may be
    rank 2..k, so correct non-top-1 neighbours are not discarded immediately.

    The proposal list is still filtered by the coordinate-consistent component
    merger.  This function only supplies a better ordering and a bounded number
    of alternatives per outgoing side.
    """
    if not 2 <= top_k < TILE_COUNT:
        raise ValueError("top_k must be in [2, 575]")
    if not 1 <= keep_per_tile <= top_k:
        raise ValueError("keep_per_tile must be in [1, top_k]")
    if loop_weight < 0:
        raise ValueError("loop_weight must be non-negative")

    (
        right_out_order,
        right_in_order,
        right_row_rank,
        right_cost,
    ) = _directional_rank_costs(
        compatibility.right, reciprocal_weight=reciprocal_weight
    )
    (
        down_out_order,
        down_in_order,
        down_row_rank,
        down_cost,
    ) = _directional_rank_costs(
        compatibility.down, reciprocal_weight=reciprocal_weight
    )
    right_out = right_out_order[:, :top_k]
    right_in = right_in_order[:top_k, :].T
    down_out = down_out_order[:, :top_k]
    down_in = down_in_order[:top_k, :].T

    def _valid_square_mask(
        first: int,
        second: int,
        third: np.ndarray,
        fourth: np.ndarray,
        closure_rank: np.ndarray,
    ) -> np.ndarray:
        third_grid = third[:, None]
        fourth_grid = fourth[None, :]
        return (
            (closure_rank < top_k)
            & (third_grid != first)
            & (third_grid != second)
            & (fourth_grid != first)
            & (fourth_grid != second)
            & (third_grid != fourth_grid)
        )

    def _closed_support(
        support: np.ndarray, mask: np.ndarray
    ) -> tuple[float, int]:
        count = int(np.count_nonzero(mask))
        if count:
            return float(np.min(support[mask])), count
        # Keep an explicit fallback so boundary/no-loop edges still receive a
        # deterministic rank, but make them strictly less attractive than a
        # genuine top-k square closure.
        return float(np.min(support) + 0.5), 0

    def horizontal_loop(first: int, second: int) -> tuple[float, int]:
        best = float("inf")
        loop_count = 0
        # Square below the proposed edge.
        lower_first = down_out[first]
        lower_second = down_out[second]
        closure = right_row_rank[np.ix_(lower_first, lower_second)]
        support = (
            down_cost[first, lower_first][:, None]
            + down_cost[second, lower_second][None, :]
            + right_cost[np.ix_(lower_first, lower_second)]
        ) / 3.0
        value, count = _closed_support(
            support,
            _valid_square_mask(
                first, second, lower_first, lower_second, closure
            ),
        )
        best = min(best, value)
        loop_count += count

        # Square above the proposed edge.  Column orders represent candidate
        # predecessors, i.e. the physical up side.
        upper_first = down_in[first]
        upper_second = down_in[second]
        closure = right_row_rank[np.ix_(upper_first, upper_second)]
        support = (
            down_cost[upper_first, first][:, None]
            + down_cost[upper_second, second][None, :]
            + right_cost[np.ix_(upper_first, upper_second)]
        ) / 3.0
        value, count = _closed_support(
            support,
            _valid_square_mask(
                first, second, upper_first, upper_second, closure
            ),
        )
        best = min(best, value)
        loop_count += count
        return best, loop_count

    def vertical_loop(first: int, second: int) -> tuple[float, int]:
        best = float("inf")
        loop_count = 0
        # Square to the right of the proposed edge.
        right_first = right_out[first]
        right_second = right_out[second]
        closure = down_row_rank[np.ix_(right_first, right_second)]
        support = (
            right_cost[first, right_first][:, None]
            + right_cost[second, right_second][None, :]
            + down_cost[np.ix_(right_first, right_second)]
        ) / 3.0
        value, count = _closed_support(
            support,
            _valid_square_mask(
                first, second, right_first, right_second, closure
            ),
        )
        best = min(best, value)
        loop_count += count

        # Square to the left; incoming horizontal candidates are the physical
        # left-side alternatives.
        left_first = right_in[first]
        left_second = right_in[second]
        closure = down_row_rank[np.ix_(left_first, left_second)]
        support = (
            right_cost[left_first, first][:, None]
            + right_cost[left_second, second][None, :]
            + down_cost[np.ix_(left_first, left_second)]
        ) / 3.0
        value, count = _closed_support(
            support,
            _valid_square_mask(
                first, second, left_first, left_second, closure
            ),
        )
        best = min(best, value)
        loop_count += count
        return best, loop_count

    proposals: list[ProposedEdge] = []
    for (dx, dy), candidates, base_cost, loop_score in (
        ((1, 0), right_out, right_cost, horizontal_loop),
        ((0, 1), down_out, down_cost, vertical_loop),
    ):
        for first in range(TILE_COUNT):
            ranked = []
            for second in candidates[first].tolist():
                if second == first:
                    continue
                support, loop_count = loop_score(first, int(second))
                # Independent closures are stronger evidence than one
                # accidental square.  Keep the bonus bounded so the actual
                # edge and loop costs remain dominant.
                loop_bonus = 0.05 * min(float(np.log2(loop_count + 1)), 3.0)
                total = float(
                    base_cost[first, second] + loop_weight * support - loop_bonus
                )
                ranked.append(
                    (
                        total,
                        0 if loop_count >= 2 else (1 if loop_count == 1 else 2),
                        int(second),
                        support,
                        loop_count,
                    )
                )
            ranked.sort()
            if not ranked:
                continue
            selected = ranked[:keep_per_tile]
            runner_up = ranked[min(keep_per_tile, len(ranked) - 1)][0]
            for total, _, second, _, loop_count in selected:
                proposals.append(
                    ProposedEdge(
                        first=first,
                        second=second,
                        dx=dx,
                        dy=dy,
                        cost=total,
                        margin=max(0.0, float(runner_up - total)),
                        reciprocal=False,
                        in_loop=loop_count > 0,
                    )
                )
    proposals.sort(
        key=lambda edge: (
            0 if edge.in_loop else 1,
            edge.cost,
            -edge.margin,
            edge.first,
            edge.second,
            edge.dy,
            edge.dx,
        )
    )
    return proposals


def select_confident_edges(
    proposals: list[ProposedEdge], *, keep_fraction: float = 1.0
) -> list[ProposedEdge]:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if keep_fraction == 1.0 or not proposals:
        return list(proposals)
    keep = max(1, int(np.ceil(len(proposals) * keep_fraction)))
    ranked = sorted(
        proposals,
        key=lambda edge: (
            0 if edge.in_loop else 1,
            -edge.margin,
            edge.cost,
            edge.first,
            edge.second,
            edge.dy,
            edge.dx,
        ),
    )
    selected = set(ranked[:keep])
    return [edge for edge in proposals if edge in selected]


class _RelativeComponents:
    def __init__(self) -> None:
        self.component_of = np.arange(TILE_COUNT, dtype=np.int32)
        self.members: dict[int, dict[int, tuple[int, int]]] = {
            index: {index: (0, 0)} for index in range(TILE_COUNT)
        }

    def merge(self, edge: ProposedEdge) -> bool:
        first_root = int(self.component_of[edge.first])
        second_root = int(self.component_of[edge.second])
        first_members = self.members[first_root]
        second_members = self.members[second_root]
        first_xy = first_members[edge.first]
        desired_second = (first_xy[0] + edge.dx, first_xy[1] + edge.dy)
        if first_root == second_root:
            return second_members[edge.second] == desired_second

        current_second = second_members[edge.second]
        shift = (
            desired_second[0] - current_second[0],
            desired_second[1] - current_second[1],
        )
        shifted = {
            tile: (xy[0] + shift[0], xy[1] + shift[1])
            for tile, xy in second_members.items()
        }
        first_positions = set(first_members.values())
        if first_positions & set(shifted.values()):
            return False
        all_positions = list(first_members.values()) + list(shifted.values())
        xs = [xy[0] for xy in all_positions]
        ys = [xy[1] for xy in all_positions]
        if max(xs) - min(xs) >= GRID or max(ys) - min(ys) >= GRID:
            return False

        if len(first_members) < len(second_members):
            # Keep the larger dictionary and transform the old first component
            # into the second component's coordinate frame.
            reverse_shift = (-shift[0], -shift[1])
            transformed_first = {
                tile: (xy[0] + reverse_shift[0], xy[1] + reverse_shift[1])
                for tile, xy in first_members.items()
            }
            transformed_first.update(second_members)
            keep_root, remove_root = second_root, first_root
            self.members[keep_root] = transformed_first
        else:
            first_members.update(shifted)
            keep_root, remove_root = first_root, second_root
        for tile in self.members[keep_root]:
            self.component_of[tile] = keep_root
        del self.members[remove_root]
        return True

    def ordered_components(self) -> list[dict[int, tuple[int, int]]]:
        return sorted(
            (dict(members) for members in self.members.values()),
            key=lambda members: (-len(members), min(members)),
        )


def grow_components(proposals: list[ProposedEdge]) -> tuple[list[dict[int, tuple[int, int]]], int]:
    components = _RelativeComponents()
    accepted = 0
    for edge in proposals:
        if components.merge(edge):
            accepted += 1
    return components.ordered_components(), accepted


def grow_two_side_consensus(
    components: list[dict[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 8,
    max_additions: int = 256,
) -> tuple[list[dict[int, tuple[int, int]]], int]:
    """Grow reliable components using candidates supported by two placed sides."""
    if top_k <= 0 or max_additions < 0:
        raise ValueError("top_k must be positive and max_additions non-negative")
    working = [dict(component) for component in components]
    singleton_tiles = {
        next(iter(component)) for component in working if len(component) == 1
    }
    added = 0
    for members in working:
        if len(members) < 4 or not singleton_tiles:
            continue
        while singleton_tiles and added < max_additions:
            coordinate_to_tile = {coordinate: tile for tile, coordinate in members.items()}
            boundary: set[tuple[int, int]] = set()
            for x, y in coordinate_to_tile:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    candidate_coordinate = (x + dx, y + dy)
                    if candidate_coordinate not in coordinate_to_tile:
                        boundary.add(candidate_coordinate)
            candidates = np.asarray(sorted(singleton_tiles), dtype=np.int32)
            proposals = []
            for x, y in sorted(boundary, key=lambda value: (value[1], value[0])):
                neighbours: list[np.ndarray] = []
                left = coordinate_to_tile.get((x - 1, y))
                right = coordinate_to_tile.get((x + 1, y))
                up = coordinate_to_tile.get((x, y - 1))
                down = coordinate_to_tile.get((x, y + 1))
                if left is not None:
                    neighbours.append(compatibility.right[left, candidates])
                if right is not None:
                    neighbours.append(compatibility.right[candidates, right])
                if up is not None:
                    neighbours.append(compatibility.down[up, candidates])
                if down is not None:
                    neighbours.append(compatibility.down[candidates, down])
                if len(neighbours) < 2:
                    continue
                rank_rows = []
                support = np.zeros(len(candidates), dtype=np.int32)
                for costs in neighbours:
                    order = np.argsort(costs, kind="stable")
                    ranks = np.empty(len(candidates), dtype=np.int32)
                    ranks[order] = np.arange(len(candidates), dtype=np.int32)
                    rank_rows.append(ranks.astype(np.float32) / max(len(candidates) - 1, 1))
                    support += ranks < min(top_k, len(candidates))
                mean_rank = np.mean(np.stack(rank_rows), axis=0)
                eligible = np.flatnonzero(support >= 2)
                if len(eligible) == 0:
                    continue
                eligible = eligible[np.lexsort((candidates[eligible], mean_rank[eligible], -support[eligible]))]
                selected_index = int(eligible[0])
                selected_tile = int(candidates[selected_index])
                selected_score = float(mean_rank[selected_index])
                second_score = (
                    float(mean_rank[int(eligible[1])]) if len(eligible) > 1 else 1.0
                )
                xs = [coordinate[0] for coordinate in coordinate_to_tile]
                ys = [coordinate[1] for coordinate in coordinate_to_tile]
                if max([*xs, x]) - min([*xs, x]) >= GRID:
                    continue
                if max([*ys, y]) - min([*ys, y]) >= GRID:
                    continue
                proposals.append(
                    (
                        -int(support[selected_index]),
                        selected_score,
                        -(second_score - selected_score),
                        y,
                        x,
                        selected_tile,
                    )
                )
            if not proposals:
                break
            proposals.sort()
            _, score, _, y, x, tile = proposals[0]
            if score > 0.15:
                break
            members[tile] = (x, y)
            singleton_tiles.remove(tile)
            added += 1
    consumed = {tile for component in working if len(component) > 1 for tile in component}
    result = [
        component
        for component in working
        if len(component) > 1 or next(iter(component)) not in consumed
    ]
    return sorted(result, key=lambda members: (-len(members), min(members))), added


def grow_component_translation_consensus(
    components: list[dict[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 8,
    min_support: int = 2,
    max_merges: int = TILE_COUNT,
    reciprocal_weight: float = 0.35,
) -> tuple[list[dict[int, tuple[int, int]]], int]:
    """Merge components only when multiple edges imply one translation.

    A single noisy top-k edge is never enough.  Candidate edges between two
    components vote for the relative integer translation of the entire second
    component.  Two or more agreeing contacts provide genuine left/right/up/
    down consensus and can also attach a singleton through two known sides.
    """
    if not 1 <= top_k < TILE_COUNT:
        raise ValueError("top_k must be in [1, 575]")
    if min_support < 2 or max_merges < 0:
        raise ValueError("min_support must be >=2 and max_merges non-negative")
    working = [dict(component) for component in components]
    direction_data = []
    for dx, dy, matrix in (
        (1, 0, compatibility.right),
        (0, 1, compatibility.down),
    ):
        outgoing, _, _, costs = _directional_rank_costs(
            matrix, reciprocal_weight=reciprocal_weight
        )
        direction_data.append((dx, dy, outgoing[:, :top_k], costs))

    merges = 0
    while merges < max_merges:
        component_of = np.empty(TILE_COUNT, dtype=np.int32)
        for component_index, members in enumerate(working):
            for tile in members:
                component_of[tile] = component_index
        groups: dict[
            tuple[int, int, int, int],
            dict[tuple[int, int, int, int], float],
        ] = {}
        for dx, dy, candidates, costs in direction_data:
            for first in range(TILE_COUNT):
                first_component = int(component_of[first])
                first_xy = working[first_component][first]
                for second in candidates[first].tolist():
                    second = int(second)
                    second_component = int(component_of[second])
                    if first_component == second_component:
                        continue
                    second_xy = working[second_component][second]
                    if first_component < second_component:
                        base_component = first_component
                        moving_component = second_component
                        shift = (
                            first_xy[0] + dx - second_xy[0],
                            first_xy[1] + dy - second_xy[1],
                        )
                    else:
                        base_component = second_component
                        moving_component = first_component
                        # second(base) = first(moving) + delta
                        shift = (
                            second_xy[0] - dx - first_xy[0],
                            second_xy[1] - dy - first_xy[1],
                        )
                    key = (
                        base_component,
                        moving_component,
                        int(shift[0]),
                        int(shift[1]),
                    )
                    evidence_key = (first, second, dx, dy)
                    groups.setdefault(key, {})[evidence_key] = float(
                        costs[first, second]
                    )

        candidates_to_merge = []
        for (base_index, moving_index, shift_x, shift_y), evidence in groups.items():
            if len(evidence) < min_support:
                continue
            base = working[base_index]
            moving = working[moving_index]
            shifted = {
                tile: (xy[0] + shift_x, xy[1] + shift_y)
                for tile, xy in moving.items()
            }
            if set(base.values()) & set(shifted.values()):
                continue
            all_positions = [*base.values(), *shifted.values()]
            xs = [xy[0] for xy in all_positions]
            ys = [xy[1] for xy in all_positions]
            if max(xs) - min(xs) >= GRID or max(ys) - min(ys) >= GRID:
                continue
            evidence_costs = sorted(evidence.values())
            # Extra agreeing contacts help, but weak tail votes should not
            # overpower the two strongest independent contacts.
            consensus_cost = float(np.mean(evidence_costs[:min_support]))
            candidates_to_merge.append(
                (
                    -len(evidence),
                    consensus_cost,
                    -(len(base) + len(moving)),
                    base_index,
                    moving_index,
                    shift_x,
                    shift_y,
                    shifted,
                )
            )
        if not candidates_to_merge:
            break
        candidates_to_merge.sort(key=lambda item: item[:7])
        (
            _,
            _,
            _,
            base_index,
            moving_index,
            _,
            _,
            shifted,
        ) = candidates_to_merge[0]
        working[base_index].update(shifted)
        del working[moving_index]
        merges += 1

    return sorted(working, key=lambda members: (-len(members), min(members))), merges


def _graph_components(proposals: list[ProposedEdge]) -> list[tuple[list[int], list[ProposedEdge]]]:
    adjacency: list[list[int]] = [[] for _ in range(TILE_COUNT)]
    for edge_index, edge in enumerate(proposals):
        adjacency[edge.first].append(edge_index)
        adjacency[edge.second].append(edge_index)
    visited = np.zeros(TILE_COUNT, dtype=bool)
    result = []
    for start in range(TILE_COUNT):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        tiles = []
        edge_indices: set[int] = set()
        while stack:
            tile = stack.pop()
            tiles.append(tile)
            for edge_index in adjacency[tile]:
                edge_indices.add(edge_index)
                edge = proposals[edge_index]
                neighbour = edge.second if edge.first == tile else edge.first
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        result.append((sorted(tiles), [proposals[index] for index in sorted(edge_indices)]))
    return sorted(result, key=lambda item: (-len(item[0]), item[0][0]))


def _lp_axis(
    tiles: list[int],
    edges: list[ProposedEdge],
    *,
    axis: str,
) -> np.ndarray | None:
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    count = len(tiles)
    if count == 1:
        return np.zeros(1, dtype=np.float64)
    local = {tile: index for index, tile in enumerate(tiles)}
    margins = np.asarray([edge.margin for edge in edges], dtype=np.float64)
    if len(margins) > 1:
        order = np.argsort(np.argsort(margins, kind="stable"), kind="stable")
        confidence = order / float(len(margins) - 1)
    else:
        confidence = np.ones(len(margins), dtype=np.float64)
    weights = 1.0 + 4.0 * confidence
    weights *= np.asarray([4.0 if edge.in_loop else 1.0 for edge in edges])
    variable_count = count + len(edges)
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[count:] = weights
    a_ub = np.zeros((2 * len(edges), variable_count), dtype=np.float64)
    b_ub = np.empty(2 * len(edges), dtype=np.float64)
    for edge_index, edge in enumerate(edges):
        first = local[edge.first]
        second = local[edge.second]
        delta = float(edge.dx if axis == "x" else edge.dy)
        residual = count + edge_index
        a_ub[2 * edge_index, second] = 1.0
        a_ub[2 * edge_index, first] = -1.0
        a_ub[2 * edge_index, residual] = -1.0
        b_ub[2 * edge_index] = delta
        a_ub[2 * edge_index + 1, second] = -1.0
        a_ub[2 * edge_index + 1, first] = 1.0
        a_ub[2 * edge_index + 1, residual] = -1.0
        b_ub[2 * edge_index + 1] = -delta
    a_eq = np.zeros((1, variable_count), dtype=np.float64)
    a_eq[0, 0] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.zeros(1, dtype=np.float64),
        bounds=[(-GRID + 1, GRID - 1)] * count + [(0.0, None)] * len(edges),
        method="highs",
        options={"time_limit": 10.0},
    )
    if not result.success:
        return None
    return np.asarray(result.x[:count], dtype=np.float64)


def _snap_component(
    tiles: list[int], x: np.ndarray, y: np.ndarray
) -> dict[int, tuple[int, int]]:
    count = len(tiles)
    x = x - float(np.min(x))
    y = y - float(np.min(y))
    width = min(GRID, max(1, int(np.ceil(float(np.max(x)))) + 1))
    height = min(GRID, max(1, int(np.ceil(float(np.max(y)))) + 1))
    while width * height < count:
        if width <= height and width < GRID:
            width += 1
        elif height < GRID:
            height += 1
        elif width < GRID:
            width += 1
        else:
            raise RuntimeError("component cannot fit the grid")
    if np.max(x) > 0 and width > 1:
        x = x * (width - 1) / float(np.max(x))
    if np.max(y) > 0 and height > 1:
        y = y * (height - 1) / float(np.max(y))
    candidates = np.asarray(
        [(column, row) for row in range(height) for column in range(width)],
        dtype=np.float64,
    )
    costs = (
        (x[:, None] - candidates[None, :, 0]) ** 2
        + (y[:, None] - candidates[None, :, 1]) ** 2
    )
    costs += 1e-10 * np.arange(len(candidates), dtype=np.float64)[None, :]
    tile_rows, candidate_columns = linear_sum_assignment(costs)
    return {
        tiles[tile_row]: (
            int(candidates[candidate_column, 0]),
            int(candidates[candidate_column, 1]),
        )
        for tile_row, candidate_column in zip(
            tile_rows.tolist(), candidate_columns.tolist(), strict=True
        )
    }


def weighted_l1_components(
    proposals: list[ProposedEdge],
) -> tuple[list[dict[int, tuple[int, int]]], int]:
    components = []
    failures = 0
    for tiles, edges in _graph_components(proposals):
        if len(tiles) == 1:
            components.append({tiles[0]: (0, 0)})
            continue
        x = _lp_axis(tiles, edges, axis="x")
        y = _lp_axis(tiles, edges, axis="y")
        if x is None or y is None:
            failures += 1
            components.extend({tile: (0, 0)} for tile in tiles)
            continue
        components.append(_snap_component(tiles, x, y))
    return sorted(components, key=lambda members: (-len(members), min(members))), failures


def _successive_lp_axis(
    edges: list[ProposedEdge],
    weights: np.ndarray,
    *,
    axis: str,
) -> np.ndarray | None:
    """Sparse weighted-L1 coordinate solve over all active side hypotheses."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    edge_count = len(edges)
    variable_count = TILE_COUNT + edge_count
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[TILE_COUNT:] = np.asarray(weights, dtype=np.float64)
    rows = []
    columns = []
    data = []
    upper = np.empty(2 * edge_count, dtype=np.float64)
    for edge_index, edge in enumerate(edges):
        delta = float(edge.dx if axis == "x" else edge.dy)
        residual = TILE_COUNT + edge_index
        row = 2 * edge_index
        rows.extend((row, row, row))
        columns.extend((edge.second, edge.first, residual))
        data.extend((1.0, -1.0, -1.0))
        upper[row] = delta
        row += 1
        rows.extend((row, row, row))
        columns.extend((edge.second, edge.first, residual))
        data.extend((-1.0, 1.0, -1.0))
        upper[row] = -delta
    inequalities = coo_matrix(
        (data, (rows, columns)),
        shape=(2 * edge_count, variable_count),
        dtype=np.float64,
    ).tocsr()
    equality = coo_matrix(
        ([1.0], ([0], [0])), shape=(1, variable_count), dtype=np.float64
    ).tocsr()
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=upper,
        A_eq=equality,
        b_eq=np.zeros(1, dtype=np.float64),
        bounds=[(-GRID + 1, GRID - 1)] * TILE_COUNT
        + [(0.0, None)] * edge_count,
        method="highs",
        options={"time_limit": 20.0},
    )
    if not result.success:
        return None
    return np.asarray(result.x[:TILE_COUNT], dtype=np.float64)


def successive_topk_lp_solver(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 16,
    max_iterations: int = 8,
    residual_tolerance: float = 0.25,
    reciprocal_weight: float = 0.35,
    boundary_weight: float = 0.2,
    snap_global: bool = False,
) -> SuccessiveLPSolveResult:
    """Successively reject inconsistent side candidates with a global L1 LP.

    Four side streams are active: right, left, down and up.  Whenever the
    global coordinate fit cannot satisfy a side hypothesis, that side advances
    to its next-ranked candidate instead of permanently freezing top-1.
    """
    if not 2 <= top_k < TILE_COUNT:
        raise ValueError("top_k must be in [2, 575]")
    if max_iterations <= 0 or residual_tolerance < 0:
        raise ValueError("invalid successive LP iteration parameters")
    right_out, right_in, _, right_cost = _directional_rank_costs(
        compatibility.right, reciprocal_weight=reciprocal_weight
    )
    down_out, down_in, _, down_cost = _directional_rank_costs(
        compatibility.down, reciprocal_weight=reciprocal_weight
    )
    side_candidates = [
        right_out[:, :top_k],
        right_in[:top_k, :].T,
        down_out[:, :top_k],
        down_in[:top_k, :].T,
    ]
    side_rank = np.zeros((4, TILE_COUNT), dtype=np.int16)
    advanced = 0
    coordinates_x = coordinates_y = None
    active_edges: list[ProposedEdge] = []
    active_meta: list[tuple[int, int]] = []
    active_costs = np.empty(0, dtype=np.float64)
    iterations_run = 0

    for iteration in range(max_iterations):
        iterations_run = iteration + 1
        active_edges = []
        active_meta = []
        costs = []
        for side in range(4):
            for tile in range(TILE_COUNT):
                rank = int(side_rank[side, tile])
                candidate = int(side_candidates[side][tile, rank])
                if side == 0:  # right successor
                    first, second, dx, dy = tile, candidate, 1, 0
                    cost = float(right_cost[first, second])
                elif side == 1:  # left predecessor
                    first, second, dx, dy = candidate, tile, 1, 0
                    cost = float(right_cost[first, second])
                elif side == 2:  # down successor
                    first, second, dx, dy = tile, candidate, 0, 1
                    cost = float(down_cost[first, second])
                else:  # up predecessor
                    first, second, dx, dy = candidate, tile, 0, 1
                    cost = float(down_cost[first, second])
                active_edges.append(
                    ProposedEdge(
                        first=first,
                        second=second,
                        dx=dx,
                        dy=dy,
                        cost=cost,
                        margin=max(0.0, 1.0 - cost),
                        reciprocal=False,
                        in_loop=False,
                    )
                )
                active_meta.append((side, tile))
                costs.append(cost)
        active_costs = np.asarray(costs, dtype=np.float64)
        # Confident edges receive more weight, while every side still has a
        # non-zero vote and can be rejected by a conflicting global cycle.
        weights = 1.0 + 4.0 * (1.0 - np.clip(active_costs, 0.0, 1.0))
        coordinates_x = _successive_lp_axis(active_edges, weights, axis="x")
        coordinates_y = _successive_lp_axis(active_edges, weights, axis="y")
        if coordinates_x is None or coordinates_y is None:
            break
        residuals = np.asarray(
            [
                abs(
                    coordinates_x[edge.second]
                    - coordinates_x[edge.first]
                    - edge.dx
                )
                + abs(
                    coordinates_y[edge.second]
                    - coordinates_y[edge.first]
                    - edge.dy
                )
                for edge in active_edges
            ],
            dtype=np.float64,
        )
        changed = 0
        for edge_index in np.flatnonzero(residuals > residual_tolerance).tolist():
            side, tile = active_meta[edge_index]
            if side_rank[side, tile] + 1 < top_k:
                side_rank[side, tile] += 1
                changed += 1
        advanced += changed
        if changed == 0:
            break

    if coordinates_x is None or coordinates_y is None:
        fallback = reciprocal_component_solver(
            compatibility, include_verified_loops=True, refine=False
        )
        return SuccessiveLPSolveResult(
            position_to_slot=fallback.position_to_slot,
            iterations=iterations_run,
            advanced_sides=advanced,
            active_edges=len(active_edges),
            consistent_edges=0,
            component_sizes=fallback.component_sizes,
            placed_component_tiles=fallback.placed_component_tiles,
            unresolved_tiles_before_assignment=fallback.unresolved_tiles_before_assignment,
        )

    residuals = np.asarray(
        [
            abs(coordinates_x[e.second] - coordinates_x[e.first] - e.dx)
            + abs(coordinates_y[e.second] - coordinates_y[e.first] - e.dy)
            for e in active_edges
        ],
        dtype=np.float64,
    )
    consistent_indices = np.flatnonzero(residuals <= residual_tolerance)
    consistent = [active_edges[index] for index in consistent_indices.tolist()]
    # Deduplicate left/right and up/down copies before hard component growth.
    deduplicated: dict[tuple[int, int, int, int], ProposedEdge] = {}
    for edge in consistent:
        key = (edge.first, edge.second, edge.dx, edge.dy)
        previous = deduplicated.get(key)
        if previous is None or edge.cost < previous.cost:
            deduplicated[key] = edge
    consistent = sorted(
        deduplicated.values(),
        key=lambda edge: (edge.cost, edge.first, edge.second, edge.dy, edge.dx),
    )
    components, _ = grow_components(consistent)

    if snap_global:
        snapped = _snap_component(
            list(range(TILE_COUNT)), coordinates_x, coordinates_y
        )
        grid = np.full((GRID, GRID), -1, dtype=np.int32)
        for tile, (column, row) in snapped.items():
            grid[row, column] = tile
        position_to_slot = validate_permutation(
            grid.ravel(), name="successive_lp_snapped_position_to_slot"
        )
        placed_tiles = TILE_COUNT
        unresolved = 0
    else:
        grid, placed_tiles = _place_components(
            components, compatibility, boundary_weight=boundary_weight
        )
        position_to_slot, unresolved = _complete_with_hungarian(
            grid, compatibility, boundary_weight=boundary_weight
        )
    return SuccessiveLPSolveResult(
        position_to_slot=position_to_slot,
        iterations=iterations_run,
        advanced_sides=advanced,
        active_edges=len(active_edges),
        consistent_edges=len(consistent),
        component_sizes=tuple(
            sorted((len(component) for component in components), reverse=True)
        ),
        placed_component_tiles=placed_tiles,
        unresolved_tiles_before_assignment=unresolved,
    )


def _translation_score(
    members: dict[int, tuple[int, int]],
    tx: int,
    ty: int,
    grid: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    boundary_weight: float,
) -> tuple[float, int]:
    member_tiles = set(members)
    seams = []
    unary_values = []
    for tile, (x, y) in members.items():
        column, row = x + tx, y + ty
        position = row * GRID + column
        unary_values.append(float(unary[position, tile]))
        for dc, dr, direction in ((-1, 0, "left"), (1, 0, "right"), (0, -1, "up"), (0, 1, "down")):
            nc, nr = column + dc, row + dr
            if not (0 <= nc < GRID and 0 <= nr < GRID):
                continue
            neighbour = int(grid[nr, nc])
            if neighbour < 0 or neighbour in member_tiles:
                continue
            if direction == "left":
                seams.append(float(compatibility.right[neighbour, tile]))
            elif direction == "right":
                seams.append(float(compatibility.right[tile, neighbour]))
            elif direction == "up":
                seams.append(float(compatibility.down[neighbour, tile]))
            else:
                seams.append(float(compatibility.down[tile, neighbour]))
    seam_cost = float(np.mean(seams)) if seams else (0.0 if np.all(grid < 0) else 0.75)
    return seam_cost + boundary_weight * float(np.mean(unary_values)), len(seams)


def _place_components(
    components: list[dict[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float,
    placement_costs: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    grid = np.full((GRID, GRID), -1, dtype=np.int32)
    unary = (
        placement_unary(compatibility)
        if placement_costs is None
        else np.asarray(placement_costs, dtype=np.float32)
    )
    if unary.shape != (TILE_COUNT, TILE_COUNT) or not np.all(np.isfinite(unary)):
        raise ValueError("placement_costs must be a finite 576x576 array")
    placed_tiles = 0
    for members in components:
        if len(members) < 2:
            continue
        member_items = list(members.items())
        xs = [xy[0] for _, xy in member_items]
        ys = [xy[1] for _, xy in member_items]
        translations = []
        for ty in range(-min(ys), GRID - max(ys)):
            for tx in range(-min(xs), GRID - max(xs)):
                positions = [(y + ty, x + tx) for _, (x, y) in member_items]
                if any(grid[row, column] >= 0 for row, column in positions):
                    continue
                score, contacts = _translation_score(
                    members, tx, ty, grid, compatibility, unary, boundary_weight
                )
                translations.append((score, -contacts, ty, tx, positions))
        if not translations:
            continue
        translations.sort(key=lambda item: item[:4])
        _, _, _, _, positions = translations[0]
        for (tile, _), (row, column) in zip(member_items, positions, strict=True):
            grid[row, column] = tile
        placed_tiles += len(members)
    return grid, placed_tiles


def _component_translations(
    members: dict[int, tuple[int, int]],
    grid: np.ndarray,
    compatibility: CompatibilityMatrices,
    unary: np.ndarray,
    boundary_weight: float,
) -> list[tuple[float, int, int, int, list[tuple[int, int]]]]:
    member_items = list(members.items())
    xs = [xy[0] for _, xy in member_items]
    ys = [xy[1] for _, xy in member_items]
    translations = []
    for ty in range(-min(ys), GRID - max(ys)):
        for tx in range(-min(xs), GRID - max(xs)):
            positions = [(y + ty, x + tx) for _, (x, y) in member_items]
            if any(grid[row, column] >= 0 for row, column in positions):
                continue
            score, contacts = _translation_score(
                members, tx, ty, grid, compatibility, unary, boundary_weight
            )
            translations.append((score, -contacts, ty, tx, positions))
    translations.sort(key=lambda item: item[:4])
    return translations


def _place_components_beam(
    components: list[dict[int, tuple[int, int]]],
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float,
    beam_width: int,
    beam_components: int,
    translations_per_state: int = 8,
    placement_costs: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    if beam_width <= 1 or beam_components <= 0 or translations_per_state <= 0:
        raise ValueError("beam placement requires width > 1 and positive limits")
    unary = (
        placement_unary(compatibility)
        if placement_costs is None
        else np.asarray(placement_costs, dtype=np.float32)
    )
    if unary.shape != (TILE_COUNT, TILE_COUNT) or not np.all(np.isfinite(unary)):
        raise ValueError("placement_costs must be a finite 576x576 array")
    states: list[tuple[float, np.ndarray, int]] = [
        (0.0, np.full((GRID, GRID), -1, dtype=np.int32), 0)
    ]
    non_singletons = [component for component in components if len(component) >= 2]
    for component_index, members in enumerate(non_singletons):
        candidates = []
        limit = translations_per_state if component_index < beam_components else 1
        active_states = states if component_index < beam_components else states[:1]
        member_items = list(members.items())
        for cumulative, grid, placed in active_states:
            translations = _component_translations(
                members, grid, compatibility, unary, boundary_weight
            )[:limit]
            if not translations:
                candidates.append((cumulative + 1.0, grid.copy(), placed))
                continue
            for score, _, _, _, positions in translations:
                next_grid = grid.copy()
                for (tile, _), (row, column) in zip(
                    member_items, positions, strict=True
                ):
                    next_grid[row, column] = tile
                candidates.append((cumulative + score, next_grid, placed + len(members)))
        if not candidates:
            continue
        candidates.sort(
            key=lambda state: (
                state[0],
                -state[2],
                tuple(state[1].ravel().tolist()),
            )
        )
        states = candidates[:beam_width] if component_index < beam_components else candidates[:1]
    return states[0][1], states[0][2]


def _complete_with_hungarian(
    grid: np.ndarray,
    compatibility: CompatibilityMatrices,
    *,
    boundary_weight: float,
    placement_costs: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    used = set(int(value) for value in grid.ravel() if value >= 0)
    remaining_tiles = np.asarray(
        [tile for tile in range(TILE_COUNT) if tile not in used], dtype=np.int32
    )
    remaining_positions = np.flatnonzero(grid.ravel() < 0).astype(np.int32)
    if len(remaining_tiles) != len(remaining_positions):
        raise RuntimeError("tile/cell count mismatch before assignment")
    unresolved = len(remaining_tiles)
    if unresolved == 0:
        return grid.ravel().copy(), 0
    unary = (
        placement_unary(compatibility)
        if placement_costs is None
        else np.asarray(placement_costs, dtype=np.float32)
    )
    if unary.shape != (TILE_COUNT, TILE_COUNT) or not np.all(np.isfinite(unary)):
        raise ValueError("placement_costs must be a finite 576x576 array")
    costs = np.empty((unresolved, unresolved), dtype=np.float64)
    for tile_index, tile in enumerate(remaining_tiles.tolist()):
        for position_index, position in enumerate(remaining_positions.tolist()):
            row, column = divmod(position, GRID)
            seams = []
            if column > 0 and grid[row, column - 1] >= 0:
                seams.append(float(compatibility.right[int(grid[row, column - 1]), tile]))
            if column + 1 < GRID and grid[row, column + 1] >= 0:
                seams.append(float(compatibility.right[tile, int(grid[row, column + 1])]))
            if row > 0 and grid[row - 1, column] >= 0:
                seams.append(float(compatibility.down[int(grid[row - 1, column]), tile]))
            if row + 1 < GRID and grid[row + 1, column] >= 0:
                seams.append(float(compatibility.down[tile, int(grid[row + 1, column])]))
            seam_cost = float(np.mean(seams)) if seams else 0.5
            costs[tile_index, position_index] = (
                seam_cost
                + boundary_weight * float(unary[position, tile])
                + 1e-10 * (tile * TILE_COUNT + position)
            )
    tile_rows, position_columns = linear_sum_assignment(costs)
    for tile_row, position_column in zip(tile_rows.tolist(), position_columns.tolist(), strict=True):
        position = int(remaining_positions[position_column])
        row, column = divmod(position, GRID)
        grid[row, column] = int(remaining_tiles[tile_row])
    return validate_permutation(grid.ravel(), name="component_position_to_slot"), unresolved


def reciprocal_component_solver(
    compatibility: CompatibilityMatrices,
    *,
    include_verified_loops: bool = True,
    only_verified_loops: bool = False,
    boundary_weight: float = 0.2,
    refine: bool = True,
    refine_weak_cells: int = 32,
    refine_max_swaps: int = 8,
    min_margin: float = 0.0,
    proposal_keep_fraction: float = 1.0,
    consensus: bool = False,
    consensus_top_k: int = 8,
    consensus_max_additions: int = 256,
    consensus_compatibility: CompatibilityMatrices | None = None,
    placement_costs: np.ndarray | None = None,
    placement_beam_width: int = 1,
    placement_beam_components: int = 0,
) -> ComponentSolveResult:
    proposals = propose_reciprocal_edges(
        compatibility,
        require_reciprocal=True,
        include_verified_loops=include_verified_loops,
        only_verified_loops=only_verified_loops,
        min_margin=min_margin,
    )
    proposals = select_confident_edges(
        proposals, keep_fraction=proposal_keep_fraction
    )
    components, accepted = grow_components(proposals)
    consensus_added = 0
    if consensus:
        components, consensus_added = grow_two_side_consensus(
            components,
            compatibility if consensus_compatibility is None else consensus_compatibility,
            top_k=consensus_top_k,
            max_additions=consensus_max_additions,
        )
    if placement_beam_width > 1 and placement_beam_components > 0:
        grid, placed_tiles = _place_components_beam(
            components,
            compatibility,
            boundary_weight=boundary_weight,
            beam_width=placement_beam_width,
            beam_components=placement_beam_components,
            placement_costs=placement_costs,
        )
    else:
        grid, placed_tiles = _place_components(
            components,
            compatibility,
            boundary_weight=boundary_weight,
            placement_costs=placement_costs,
        )
    position_to_slot, unresolved = _complete_with_hungarian(
        grid,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    if refine:
        position_to_slot = swap_refine(
            position_to_slot,
            compatibility,
            boundary_weight=boundary_weight,
            weak_cells=refine_weak_cells,
            max_swaps=refine_max_swaps,
        )
    return ComponentSolveResult(
        position_to_slot=position_to_slot,
        accepted_edges=accepted,
        proposed_edges=len(proposals),
        component_sizes=tuple(sorted((len(component) for component in components), reverse=True)),
        placed_component_tiles=placed_tiles,
        unresolved_tiles_before_assignment=unresolved,
        consensus_added_tiles=consensus_added,
    )


def mutual_topk_component_solver(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int,
    boundary_weight: float = 0.2,
    placement_costs: np.ndarray | None = None,
) -> ComponentSolveResult:
    proposals = propose_mutual_topk_edges(compatibility, top_k=top_k)
    components, accepted = grow_components(proposals)
    grid, placed_tiles = _place_components(
        components,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    position_to_slot, unresolved = _complete_with_hungarian(
        grid,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    return ComponentSolveResult(
        position_to_slot=position_to_slot,
        accepted_edges=accepted,
        proposed_edges=len(proposals),
        component_sizes=tuple(
            sorted((len(component) for component in components), reverse=True)
        ),
        placed_component_tiles=placed_tiles,
        unresolved_tiles_before_assignment=unresolved,
        consensus_added_tiles=0,
    )


def soft_cycle_component_solver(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 16,
    keep_per_tile: int = 2,
    reciprocal_weight: float = 0.35,
    loop_weight: float = 1.0,
    proposal_keep_fraction: float = 1.0,
    boundary_weight: float = 0.2,
    placement_costs: np.ndarray | None = None,
) -> ComponentSolveResult:
    """Assemble components from top-k edges re-ranked by soft square cycles."""
    proposals = propose_soft_cycle_edges(
        compatibility,
        top_k=top_k,
        keep_per_tile=keep_per_tile,
        reciprocal_weight=reciprocal_weight,
        loop_weight=loop_weight,
    )
    proposals = select_confident_edges(
        proposals, keep_fraction=proposal_keep_fraction
    )
    components, accepted = grow_components(proposals)
    grid, placed_tiles = _place_components(
        components,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    position_to_slot, unresolved = _complete_with_hungarian(
        grid,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    return ComponentSolveResult(
        position_to_slot=position_to_slot,
        accepted_edges=accepted,
        proposed_edges=len(proposals),
        component_sizes=tuple(
            sorted((len(component) for component in components), reverse=True)
        ),
        placed_component_tiles=placed_tiles,
        unresolved_tiles_before_assignment=unresolved,
        consensus_added_tiles=0,
    )


def translation_consensus_component_solver(
    compatibility: CompatibilityMatrices,
    *,
    seed_keep_fraction: float = 0.5,
    top_k: int = 8,
    min_support: int = 2,
    max_merges: int = TILE_COUNT,
    reciprocal_weight: float = 0.35,
    boundary_weight: float = 0.2,
    placement_costs: np.ndarray | None = None,
) -> ComponentSolveResult:
    """Seed with precise reciprocal edges, then merge on multi-edge votes."""
    proposals = propose_reciprocal_edges(
        compatibility,
        require_reciprocal=True,
        include_verified_loops=True,
    )
    proposals = select_confident_edges(
        proposals, keep_fraction=seed_keep_fraction
    )
    components, accepted = grow_components(proposals)
    components, consensus_merges = grow_component_translation_consensus(
        components,
        compatibility,
        top_k=top_k,
        min_support=min_support,
        max_merges=max_merges,
        reciprocal_weight=reciprocal_weight,
    )
    grid, placed_tiles = _place_components(
        components,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    position_to_slot, unresolved = _complete_with_hungarian(
        grid,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    return ComponentSolveResult(
        position_to_slot=position_to_slot,
        accepted_edges=accepted + consensus_merges,
        proposed_edges=len(proposals),
        component_sizes=tuple(
            sorted((len(component) for component in components), reverse=True)
        ),
        placed_component_tiles=placed_tiles,
        unresolved_tiles_before_assignment=unresolved,
        consensus_added_tiles=consensus_merges,
    )


def weighted_l1_component_solver(
    compatibility: CompatibilityMatrices,
    *,
    include_verified_loops: bool = True,
    only_verified_loops: bool = False,
    proposal_keep_fraction: float = 1.0,
    boundary_weight: float = 0.2,
    refine: bool = False,
    refine_weak_cells: int = 32,
    refine_max_swaps: int = 8,
    placement_costs: np.ndarray | None = None,
) -> LPSolveResult:
    proposals = propose_reciprocal_edges(
        compatibility,
        include_verified_loops=include_verified_loops,
        only_verified_loops=only_verified_loops,
    )
    proposals = select_confident_edges(
        proposals, keep_fraction=proposal_keep_fraction
    )
    components, failures = weighted_l1_components(proposals)
    grid, placed_tiles = _place_components(
        components,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    position_to_slot, unresolved = _complete_with_hungarian(
        grid,
        compatibility,
        boundary_weight=boundary_weight,
        placement_costs=placement_costs,
    )
    if refine:
        position_to_slot = swap_refine(
            position_to_slot,
            compatibility,
            boundary_weight=boundary_weight,
            weak_cells=refine_weak_cells,
            max_swaps=refine_max_swaps,
        )
    return LPSolveResult(
        position_to_slot=position_to_slot,
        proposed_edges=len(proposals),
        component_sizes=tuple(sorted((len(component) for component in components), reverse=True)),
        placed_component_tiles=placed_tiles,
        unresolved_tiles_before_assignment=unresolved,
        lp_failures=failures,
    )
