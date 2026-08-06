"""Deterministic fixed-orientation two-side plaquette assembly.

The core consumes sparse *directional* candidate rows.  It deliberately does
not consume the symmetrized ``R``/``D`` matrices used by the buddies solver:
the four corner witnesses and their reverse rows are the evidence which makes
an atomic 2x2 proposal safer than an individual edge.

The public permutation convention used by the diagnostics is
``permutation[input_tile] = clean_row_major_cell``.  A returned placement is
the inverse convention, ``placement[cell] = input_tile``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
NUM_DIRECTIONS = 4
INVERSE_DIRECTION = (DOWN, UP, RIGHT, LEFT)
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Walking clockwise around the four possible elbows.  The first arm followed
# by the second arm always closes the same fixed-orientation unit square.
CORNER_DIRECTIONS = (
    (RIGHT, DOWN),
    (DOWN, LEFT),
    (LEFT, UP),
    (UP, RIGHT),
)
CORNER_BIT = {
    (0, 0): 1 << 0,  # top-left
    (0, 1): 1 << 1,  # top-right
    (1, 0): 1 << 2,  # bottom-left
    (1, 1): 1 << 3,  # bottom-right
}


def _validate_permutation(permutation: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(permutation, dtype=np.int64)
    if value.shape != (count,):
        raise ValueError(f"permutation must have shape ({count},), got {value.shape}")
    if not np.array_equal(np.sort(value), np.arange(count, dtype=np.int64)):
        raise ValueError("permutation must contain every clean cell exactly once")
    return value


@dataclass(frozen=True)
class DirectionalTopK:
    """Stable top-k rows plus O(1) reciprocal evidence lookups."""

    ids: np.ndarray
    scores: np.ndarray
    logp: np.ndarray
    rank_lookup: np.ndarray
    logp_lookup: np.ndarray
    missing_logp: float

    @property
    def count(self) -> int:
        return int(self.ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.ids.shape[2])

    @classmethod
    def from_candidate_rows(
        cls,
        candidate_ids: np.ndarray,
        directional_scores: np.ndarray,
        *,
        top_k: int,
        missing_logp: float = -20.0,
    ) -> "DirectionalTopK":
        """Build stable rows from ``[N,K]`` or ``[N,4,K]`` candidate ids.

        Duplicate ids are reduced by maximum score.  Log probabilities are
        normalized over the complete finite deduplicated row before truncation.
        Ties are resolved by the lower target id, never by ``argpartition`` or
        container iteration order.
        """

        raw_ids = np.asarray(candidate_ids, dtype=np.int64)
        raw_scores = np.asarray(directional_scores, dtype=np.float64)
        if raw_scores.ndim != 3 or raw_scores.shape[1] != NUM_DIRECTIONS:
            raise ValueError("directional_scores must have shape (N,4,K)")
        count, _, width = raw_scores.shape
        if raw_ids.ndim == 2:
            if raw_ids.shape != (count, width):
                raise ValueError("2-D candidate_ids must have shape (N,K)")
            raw_ids = np.broadcast_to(raw_ids[:, None, :], raw_scores.shape)
        elif raw_ids.shape != raw_scores.shape:
            raise ValueError("3-D candidate_ids must match directional_scores")
        if not 1 <= int(top_k) < count:
            raise ValueError(f"top_k must lie in [1,{count - 1}]")

        ids = np.full((count, NUM_DIRECTIONS, top_k), -1, dtype=np.int64)
        scores = np.full(ids.shape, -np.inf, dtype=np.float64)
        logp = np.full(ids.shape, missing_logp, dtype=np.float64)
        rank_lookup = np.zeros((NUM_DIRECTIONS, count, count), dtype=np.int16)
        logp_lookup = np.full(
            (NUM_DIRECTIONS, count, count), missing_logp, dtype=np.float64
        )

        for anchor in range(count):
            for direction in range(NUM_DIRECTIONS):
                best: dict[int, float] = {}
                for target, score in zip(
                    raw_ids[anchor, direction], raw_scores[anchor, direction]
                ):
                    target_i = int(target)
                    score_f = float(score)
                    if not isfinite(score_f):
                        continue
                    if not 0 <= target_i < count:
                        raise ValueError("finite candidate id lies outside the tile bag")
                    if target_i == anchor:
                        continue
                    previous = best.get(target_i)
                    if previous is None or score_f > previous:
                        best[target_i] = score_f
                ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
                if not ordered:
                    continue
                all_values = np.asarray([item[1] for item in ordered], dtype=np.float64)
                maximum = float(all_values.max())
                log_normalizer = maximum + float(
                    np.log(np.exp(all_values - maximum).sum())
                )
                for rank, (target, score) in enumerate(ordered[:top_k], start=1):
                    local = rank - 1
                    value = float(score - log_normalizer)
                    ids[anchor, direction, local] = target
                    scores[anchor, direction, local] = score
                    logp[anchor, direction, local] = value
                    rank_lookup[direction, anchor, target] = rank
                    logp_lookup[direction, anchor, target] = value
        return cls(ids, scores, logp, rank_lookup, logp_lookup, float(missing_logp))

    def targets(self, anchor: int, direction: int) -> np.ndarray:
        row = self.ids[int(anchor), int(direction)]
        return row[row >= 0]

    def rank(self, anchor: int, direction: int, target: int) -> int:
        return int(self.rank_lookup[int(direction), int(anchor), int(target)])

    def directed_logp(self, anchor: int, direction: int, target: int) -> float:
        return float(self.logp_lookup[int(direction), int(anchor), int(target)])

    def edge_evidence(self, anchor: int, direction: int, target: int) -> float:
        """Mean forward/reverse row log-probability for one physical claim."""

        reverse = INVERSE_DIRECTION[int(direction)]
        forward_value = self.directed_logp(anchor, direction, target)
        reverse_value = self.directed_logp(target, reverse, anchor)
        return 0.5 * (forward_value + reverse_value)


@dataclass(frozen=True)
class Plaquette:
    """One canonical fixed-orientation ``(TL,TR,BL,BR)`` proposal."""

    tiles: tuple[int, int, int, int]
    corner_mask: int
    min_edge: float
    mean_edge: float
    reciprocal_rank_sum: int
    witness_rank_sum: int

    @property
    def tile_set(self) -> frozenset[int]:
        return frozenset(self.tiles)

    @property
    def corner_count(self) -> int:
        return int(self.corner_mask).bit_count()

    @property
    def has_opposite_witnesses(self) -> bool:
        mask = int(self.corner_mask)
        return (mask & 0b1001) == 0b1001 or (mask & 0b0110) == 0b0110

    @property
    def tier_a(self) -> bool:
        return self.has_opposite_witnesses or self.corner_count >= 3

    @property
    def coordinates(self) -> dict[int, tuple[int, int]]:
        tl, tr, bl, br = self.tiles
        return {tl: (0, 0), tr: (0, 1), bl: (1, 0), br: (1, 1)}

    @property
    def edges(self) -> tuple[tuple[int, int, int], ...]:
        tl, tr, bl, br = self.tiles
        return (
            (tl, RIGHT, tr),
            (tl, DOWN, bl),
            (tr, DOWN, br),
            (bl, RIGHT, br),
        )


def plaquette_sort_key(value: Plaquette) -> tuple[object, ...]:
    """Total deterministic order, strongest proposal first."""

    return (
        -int(value.has_opposite_witnesses),
        -value.corner_count,
        -float(value.min_edge),
        -float(value.mean_edge),
        int(value.reciprocal_rank_sum),
        int(value.witness_rank_sum),
        *value.tiles,
    )


def _normalise_witness(
    anchor: int,
    first: int,
    second: int,
    diagonal: int,
    first_direction: int,
    second_direction: int,
) -> tuple[tuple[int, int, int, int], int]:
    d1 = DELTAS[first_direction]
    d2 = DELTAS[second_direction]
    raw = {
        (0, 0): int(anchor),
        d1: int(first),
        d2: int(second),
        (d1[0] + d2[0], d1[1] + d2[1]): int(diagonal),
    }
    minimum_row = min(row for row, _ in raw)
    minimum_col = min(col for _, col in raw)
    normalised = {
        (row - minimum_row, col - minimum_col): tile
        for (row, col), tile in raw.items()
    }
    expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if set(normalised) != expected:
        raise AssertionError("perpendicular unit directions did not form a 2x2 square")
    tiles = (
        normalised[(0, 0)],
        normalised[(0, 1)],
        normalised[(1, 0)],
        normalised[(1, 1)],
    )
    elbow = (-minimum_row, -minimum_col)
    return tiles, CORNER_BIT[elbow]


def enumerate_plaquettes(
    graph: DirectionalTopK,
    *,
    max_per_elbow: int = 64,
) -> list[Plaquette]:
    """Enumerate bounded top-k L closures and aggregate corner witnesses.

    For an elbow ``a`` and perpendicular arms ``d1,d2``, the diagonal must
    occur in both ``TopK(b,d2)`` and ``TopK(c,d1)``.  This is a four-directed-
    seam configuration, not a vote by independent edges.
    """

    if max_per_elbow < 1:
        raise ValueError("max_per_elbow must be positive")
    aggregate: dict[tuple[int, int, int, int], list[int]] = {}
    for anchor in range(graph.count):
        for first_direction, second_direction in CORNER_DIRECTIONS:
            closures: list[tuple[tuple[object, ...], tuple[int, int, int, int], int, int]] = []
            for first in graph.targets(anchor, first_direction):
                first_i = int(first)
                diagonal_from_first = graph.targets(first_i, second_direction)
                if not len(diagonal_from_first):
                    continue
                for second in graph.targets(anchor, second_direction):
                    second_i = int(second)
                    if second_i == first_i:
                        continue
                    second_targets = set(
                        map(int, graph.targets(second_i, first_direction).tolist())
                    )
                    for diagonal in diagonal_from_first:
                        diagonal_i = int(diagonal)
                        if diagonal_i not in second_targets:
                            continue
                        if len({anchor, first_i, second_i, diagonal_i}) != 4:
                            continue
                        ranks = (
                            graph.rank(anchor, first_direction, first_i),
                            graph.rank(anchor, second_direction, second_i),
                            graph.rank(first_i, second_direction, diagonal_i),
                            graph.rank(second_i, first_direction, diagonal_i),
                        )
                        if min(ranks) <= 0:
                            raise AssertionError("closure contains an absent directed edge")
                        directed_logp = (
                            graph.directed_logp(anchor, first_direction, first_i),
                            graph.directed_logp(anchor, second_direction, second_i),
                            graph.directed_logp(first_i, second_direction, diagonal_i),
                            graph.directed_logp(second_i, first_direction, diagonal_i),
                        )
                        tiles, bit = _normalise_witness(
                            anchor,
                            first_i,
                            second_i,
                            diagonal_i,
                            first_direction,
                            second_direction,
                        )
                        rank_sum = int(sum(ranks))
                        key = (
                            rank_sum,
                            -float(min(directed_logp)),
                            -float(np.mean(directed_logp)),
                            *tiles,
                        )
                        closures.append((key, tiles, bit, rank_sum))
            closures.sort(key=lambda item: item[0])
            for _, tiles, bit, rank_sum in closures[:max_per_elbow]:
                current = aggregate.get(tiles)
                if current is None:
                    aggregate[tiles] = [bit, rank_sum]
                else:
                    current[0] |= bit
                    current[1] = min(current[1], rank_sum)

    motifs: list[Plaquette] = []
    missing_rank = graph.top_k + 1
    for tiles, (corner_mask, witness_rank_sum) in aggregate.items():
        tl, tr, bl, br = tiles
        edges = (
            (tl, RIGHT, tr),
            (tl, DOWN, bl),
            (tr, DOWN, br),
            (bl, RIGHT, br),
        )
        evidence = [graph.edge_evidence(a, direction, b) for a, direction, b in edges]
        reciprocal_rank_sum = 0
        for a, direction, b in edges:
            direct_rank = graph.rank(a, direction, b) or missing_rank
            reverse_rank = graph.rank(b, INVERSE_DIRECTION[direction], a) or missing_rank
            reciprocal_rank_sum += direct_rank + reverse_rank
        motifs.append(
            Plaquette(
                tiles=tiles,
                corner_mask=int(corner_mask),
                min_edge=float(min(evidence)),
                mean_edge=float(np.mean(evidence)),
                reciprocal_rank_sum=int(reciprocal_rank_sum),
                witness_rank_sum=int(witness_rank_sum),
            )
        )
    motifs.sort(key=plaquette_sort_key)
    return motifs


@dataclass(frozen=True)
class CommitResult:
    accepted: bool
    reason: str
    merged_roots: int = 0
    distinct_cross_seams: int = 0


@dataclass(frozen=True)
class _PreparedMerge:
    roots: tuple[int, ...]
    shifts: dict[int, tuple[int, int]]
    combined: dict[tuple[int, int], int]


class PotentialDSU:
    """Weighted DSU whose integer potentials are tile coordinates."""

    def __init__(self, count: int, grid_side: int) -> None:
        if count < 1 or grid_side < 1:
            raise ValueError("count and grid_side must be positive")
        self.count = int(count)
        self.grid_side = int(grid_side)
        self.parent = np.arange(count, dtype=np.int64)
        self.size = np.ones(count, dtype=np.int64)
        # potential[node] = coordinate(node) - coordinate(parent[node]).
        self.potential = np.zeros((count, 2), dtype=np.int64)
        self.version = np.zeros(count, dtype=np.int64)
        self.positions: dict[int, dict[int, tuple[int, int]]] = {
            tile: {tile: (0, 0)} for tile in range(count)
        }
        self.occupancy: dict[int, dict[tuple[int, int], int]] = {
            tile: {(0, 0): tile} for tile in range(count)
        }
        self.certified_edges: dict[int, set[tuple[int, int, int]]] = {
            tile: set() for tile in range(count)
        }

    def find(self, node: int) -> tuple[int, tuple[int, int]]:
        node_i = int(node)
        parent = int(self.parent[node_i])
        if parent == node_i:
            return node_i, (0, 0)
        root, parent_offset = self.find(parent)
        self.potential[node_i, 0] += parent_offset[0]
        self.potential[node_i, 1] += parent_offset[1]
        self.parent[node_i] = root
        return root, (
            int(self.potential[node_i, 0]),
            int(self.potential[node_i, 1]),
        )

    def roots(self) -> list[int]:
        return sorted(self.positions)

    def component_positions(self, node_or_root: int) -> dict[int, tuple[int, int]]:
        root, _ = self.find(int(node_or_root))
        return dict(self.positions[root])

    def signature(self) -> tuple[tuple[int, tuple[tuple[int, int, int], ...]], ...]:
        """Canonical state signature used by deterministic contract tests."""

        output = []
        for root in self.roots():
            positions = self.positions[root]
            minimum_row = min(row for row, _ in positions.values())
            minimum_col = min(col for _, col in positions.values())
            canonical = tuple(
                sorted(
                    (
                        tile,
                        row - minimum_row,
                        col - minimum_col,
                    )
                    for tile, (row, col) in positions.items()
                )
            )
            output.append((len(canonical), canonical))
        return tuple(sorted(output))

    def _prepare(self, motif: Plaquette) -> tuple[_PreparedMerge | None, str]:
        shifts: dict[int, tuple[int, int]] = {}
        for tile, desired in motif.coordinates.items():
            root, current = self.find(tile)
            shift = (desired[0] - current[0], desired[1] - current[1])
            previous = shifts.get(root)
            if previous is not None and previous != shift:
                return None, "geometry"
            shifts[root] = shift

        combined: dict[tuple[int, int], int] = {}
        for root in sorted(shifts):
            shift = shifts[root]
            for tile, coordinate in self.positions[root].items():
                moved = (coordinate[0] + shift[0], coordinate[1] + shift[1])
                occupied = combined.get(moved)
                if occupied is not None and occupied != tile:
                    return None, "collision"
                combined[moved] = tile
        rows = [row for row, _ in combined]
        cols = [col for _, col in combined]
        if (
            max(rows) - min(rows) >= self.grid_side
            or max(cols) - min(cols) >= self.grid_side
        ):
            return None, "span"
        return _PreparedMerge(tuple(sorted(shifts)), shifts, combined), "ok"

    def _root_support(
        self, motif: Plaquette
    ) -> tuple[dict[int, int], dict[tuple[int, int], int], int]:
        root_for_tile = {tile: self.find(tile)[0] for tile in motif.tiles}
        overlap: dict[int, int] = Counter(root_for_tile.values())
        pair_support: dict[tuple[int, int], int] = Counter()
        cross = 0
        for source, _, target in motif.edges:
            first, second = root_for_tile[source], root_for_tile[target]
            if first == second:
                continue
            cross += 1
            pair_support[tuple(sorted((first, second)))] += 1
        return dict(overlap), dict(pair_support), int(cross)

    def _commit(self, prepared: _PreparedMerge, motif: Plaquette) -> int:
        roots = prepared.roots
        base = min(roots, key=lambda root: (-int(self.size[root]), root))
        base_shift = prepared.shifts[base]
        merged_positions = {
            tile: (coordinate[0] - base_shift[0], coordinate[1] - base_shift[1])
            for coordinate, tile in prepared.combined.items()
        }
        merged_edges: set[tuple[int, int, int]] = set()
        total_size = 0
        for root in roots:
            total_size += int(self.size[root])
            merged_edges.update(self.certified_edges[root])
            if root == base:
                continue
            shift = prepared.shifts[root]
            self.parent[root] = base
            self.potential[root] = (
                shift[0] - base_shift[0],
                shift[1] - base_shift[1],
            )
        merged_edges.update(motif.edges)
        self.size[base] = total_size
        self.positions[base] = merged_positions
        self.occupancy[base] = {
            coordinate: tile for tile, coordinate in merged_positions.items()
        }
        self.certified_edges[base] = merged_edges
        self.version[base] += 1
        for root in roots:
            if root == base:
                continue
            del self.positions[root]
            del self.occupancy[root]
            del self.certified_edges[root]
        return len(roots) - 1

    def try_seed(self, motif: Plaquette, *, minimum_edge: float) -> CommitResult:
        if not motif.tier_a:
            return CommitResult(False, "not_tier_a")
        if motif.min_edge < minimum_edge:
            return CommitResult(False, "below_threshold")
        roots = [self.find(tile)[0] for tile in motif.tiles]
        if len(set(roots)) != 4 or any(int(self.size[root]) != 1 for root in roots):
            return CommitResult(False, "not_fresh")
        prepared, reason = self._prepare(motif)
        if prepared is None:
            return CommitResult(False, reason)
        merged = self._commit(prepared, motif)
        return CommitResult(True, "seed", merged, 4)

    def try_growth(self, motif: Plaquette, *, minimum_edge: float) -> CommitResult:
        if motif.min_edge < minimum_edge:
            return CommitResult(False, "below_threshold")
        overlap, pair_support, cross = self._root_support(motif)
        if len(overlap) == 1:
            prepared, reason = self._prepare(motif)
            if prepared is None:
                return CommitResult(False, reason)
            return CommitResult(False, "already_consistent")
        if cross < 2:
            return CommitResult(False, "one_seam")

        # Expansion: at least two motif coordinates are already fixed by the
        # same established component.  Block merge: two nontrivial roots have
        # two distinct physical seams between them.  A reverse row is never a
        # second item here; motif.edges contains each physical seam once.
        expansion = any(
            count >= 2 and int(self.size[root]) >= 2
            for root, count in overlap.items()
        )
        block_merge = any(
            support >= 2
            and int(self.size[first]) >= 2
            and int(self.size[second]) >= 2
            for (first, second), support in pair_support.items()
        )
        if not (expansion or block_merge):
            return CommitResult(False, "insufficient_overlap")
        prepared, reason = self._prepare(motif)
        if prepared is None:
            return CommitResult(False, reason, distinct_cross_seams=cross)
        merged = self._commit(prepared, motif)
        return CommitResult(True, "growth", merged, cross)


@dataclass
class GrowthResult:
    dsu: PotentialDSU
    seed_motifs: list[Plaquette]
    growth_motifs: list[Plaquette]
    rejection_counts: dict[str, int]
    rounds: int


def grow_plaquettes(
    count: int,
    grid_side: int,
    motifs: Sequence[Plaquette],
    *,
    minimum_edge: float = -np.inf,
    growth_min_corners: int = 2,
    max_rounds: int | None = None,
) -> GrowthResult:
    """Create strict Tier-A seeds, then apply two-side motif growth."""

    if count != grid_side * grid_side:
        raise ValueError("fixed rectangular MVP expects count == grid_side**2")
    if not 1 <= growth_min_corners <= 4:
        raise ValueError("growth_min_corners must lie in [1,4]")
    ordered = sorted(motifs, key=plaquette_sort_key)
    dsu = PotentialDSU(count, grid_side)
    seed_motifs: list[Plaquette] = []
    growth_motifs: list[Plaquette] = []
    accepted_keys: set[tuple[int, int, int, int]] = set()
    rejection_counts: Counter[str] = Counter()

    for motif in ordered:
        if not motif.tier_a:
            continue
        result = dsu.try_seed(motif, minimum_edge=minimum_edge)
        if result.accepted:
            seed_motifs.append(motif)
            accepted_keys.add(motif.tiles)
        elif result.reason not in ("not_fresh",):
            rejection_counts[result.reason] += 1

    candidates = [
        motif
        for motif in ordered
        if motif.tiles not in accepted_keys
        and motif.corner_count >= growth_min_corners
        and motif.min_edge >= minimum_edge
    ]
    limit = int(max_rounds if max_rounds is not None else count)
    rounds = 0
    last_reason: dict[tuple[int, int, int, int], str] = {}
    for rounds in range(1, limit + 1):
        changed = False
        for motif in candidates:
            if motif.tiles in accepted_keys:
                continue
            result = dsu.try_growth(motif, minimum_edge=minimum_edge)
            if result.accepted:
                growth_motifs.append(motif)
                accepted_keys.add(motif.tiles)
                changed = True
            else:
                last_reason[motif.tiles] = result.reason
        if not changed:
            break
    for reason in last_reason.values():
        if reason not in ("already_consistent",):
            rejection_counts[reason] += 1
    return GrowthResult(
        dsu=dsu,
        seed_motifs=seed_motifs,
        growth_motifs=growth_motifs,
        rejection_counts=dict(sorted(rejection_counts.items())),
        rounds=rounds,
    )


def _normalise_component(
    component: dict[int, tuple[int, int]],
) -> tuple[dict[int, tuple[int, int]], int, int]:
    minimum_row = min(row for row, _ in component.values())
    minimum_col = min(col for _, col in component.values())
    normalised = {
        tile: (row - minimum_row, col - minimum_col)
        for tile, (row, col) in component.items()
    }
    height = max(row for row, _ in normalised.values()) + 1
    width = max(col for _, col in normalised.values()) + 1
    return normalised, height, width


def _contact_evidence(
    board: np.ndarray,
    placed: dict[int, tuple[int, int]],
    graph: DirectionalTopK,
) -> tuple[int, float]:
    values: list[float] = []
    side = board.shape[0]
    for tile, (row, col) in placed.items():
        if col > 0 and board[row, col - 1] >= 0:
            values.append(graph.edge_evidence(int(board[row, col - 1]), RIGHT, tile))
        if col + 1 < side and board[row, col + 1] >= 0:
            values.append(graph.edge_evidence(tile, RIGHT, int(board[row, col + 1])))
        if row > 0 and board[row - 1, col] >= 0:
            values.append(graph.edge_evidence(int(board[row - 1, col]), DOWN, tile))
        if row + 1 < side and board[row + 1, col] >= 0:
            values.append(graph.edge_evidence(tile, DOWN, int(board[row + 1, col])))
    return len(values), float(np.mean(values)) if values else float("-inf")


def _unary_cell_score(
    board: np.ndarray,
    row: int,
    col: int,
    tile: int,
    graph: DirectionalTopK,
) -> float:
    side = board.shape[0]
    value = 0.0
    if col > 0 and board[row, col - 1] >= 0:
        value += graph.edge_evidence(int(board[row, col - 1]), RIGHT, tile)
    if col + 1 < side and board[row, col + 1] >= 0:
        value += graph.edge_evidence(tile, RIGHT, int(board[row, col + 1]))
    if row > 0 and board[row - 1, col] >= 0:
        value += graph.edge_evidence(int(board[row - 1, col]), DOWN, tile)
    if row + 1 < side and board[row + 1, col] >= 0:
        value += graph.edge_evidence(tile, DOWN, int(board[row + 1, col]))
    return float(value)


@dataclass(frozen=True)
class PackingResult:
    board: np.ndarray
    placement: np.ndarray
    rigid_components_placed: int
    rigid_tiles_placed: int
    hungarian_tiles: int


def pack_components(
    dsu: PotentialDSU,
    graph: DirectionalTopK,
) -> PackingResult:
    """Pack only two-contact rigid blocks, then complete by Hungarian unary.

    Residual components are intentionally dissolved into their tiles.  This
    prevents a false low-coverage block from becoming an uncorrectable rigid
    gene, which was the principal failure mode of the component beam branch.
    """

    if graph.count != dsu.count:
        raise ValueError("graph and DSU tile counts differ")
    side = dsu.grid_side
    board = np.full((side, side), -1, dtype=np.int64)
    components = []
    for root in dsu.roots():
        component = dsu.positions[root]
        normalised, height, width = _normalise_component(component)
        components.append(
            (
                root,
                normalised,
                height,
                width,
                len(dsu.certified_edges[root]),
            )
        )
    components.sort(key=lambda item: (-len(item[1]), -item[4], item[0]))
    anchor_root, anchor, height, width, _ = components[0]
    if height > side or width > side:
        raise AssertionError("DSU component exceeds the fixed frame")
    shift_row = (side - height) // 2
    shift_col = (side - width) // 2
    for tile, (row, col) in anchor.items():
        board[row + shift_row, col + shift_col] = tile
    used = set(anchor)
    placed_roots = {anchor_root}
    rigid_tiles = len(anchor)

    remaining = {item[0]: item for item in components[1:]}
    while remaining:
        proposals: list[tuple[tuple[object, ...], int, dict[int, tuple[int, int]]]] = []
        for root in sorted(remaining):
            _, component, component_height, component_width, _ = remaining[root]
            for sy in range(side - component_height + 1):
                for sx in range(side - component_width + 1):
                    placed = {
                        tile: (row + sy, col + sx)
                        for tile, (row, col) in component.items()
                    }
                    if any(board[row, col] >= 0 for row, col in placed.values()):
                        continue
                    contacts, mean_evidence = _contact_evidence(board, placed, graph)
                    if contacts < 2:
                        continue
                    key = (
                        -contacts,
                        -mean_evidence,
                        -len(component),
                        root,
                        sy,
                        sx,
                    )
                    proposals.append((key, root, placed))
        if not proposals:
            break
        _, root, placed = min(proposals, key=lambda item: item[0])
        for tile, (row, col) in placed.items():
            board[row, col] = tile
            used.add(tile)
        rigid_tiles += len(placed)
        placed_roots.add(root)
        del remaining[root]

    empty_cells = [tuple(map(int, value)) for value in np.argwhere(board < 0)]
    unused_tiles = sorted(set(range(dsu.count)) - used)
    if len(empty_cells) != len(unused_tiles):
        raise AssertionError("packing lost tile/cell cardinality")
    hungarian_count = len(unused_tiles)
    if unused_tiles:
        unary = np.empty((len(empty_cells), len(unused_tiles)), dtype=np.float64)
        for cell_index, (row, col) in enumerate(empty_cells):
            for tile_index, tile in enumerate(unused_tiles):
                unary[cell_index, tile_index] = _unary_cell_score(
                    board, row, col, tile, graph
                )
        # Both axes are lexicographically sorted.  SciPy's deterministic dense
        # implementation therefore also provides a stable all-ties result.
        rows, cols = linear_sum_assignment(-unary)
        for row_index, col_index in zip(rows.tolist(), cols.tolist()):
            row, col = empty_cells[row_index]
            board[row, col] = unused_tiles[col_index]

    flat = board.reshape(-1)
    if np.any(flat < 0):
        raise AssertionError("packing left a hole")
    if not np.array_equal(np.sort(flat), np.arange(dsu.count, dtype=np.int64)):
        raise AssertionError("packing did not return a strict tile permutation")
    return PackingResult(
        board=board,
        placement=flat.copy(),
        rigid_components_placed=len(placed_roots),
        rigid_tiles_placed=rigid_tiles,
        hungarian_tiles=hungarian_count,
    )


def true_plaquette_keys(
    permutation: np.ndarray, grid_side: int
) -> set[tuple[int, int, int, int]]:
    count = grid_side * grid_side
    value = _validate_permutation(permutation, count)
    inverse = np.empty(count, dtype=np.int64)
    inverse[value] = np.arange(count, dtype=np.int64)
    keys = set()
    for row in range(grid_side - 1):
        for col in range(grid_side - 1):
            cell = row * grid_side + col
            keys.add(
                (
                    int(inverse[cell]),
                    int(inverse[cell + 1]),
                    int(inverse[cell + grid_side]),
                    int(inverse[cell + grid_side + 1]),
                )
            )
    return keys


def _edge_is_exact(
    edge: tuple[int, int, int], permutation: np.ndarray, grid_side: int
) -> bool:
    source, direction, target = edge
    source_cell = int(permutation[source])
    target_cell = int(permutation[target])
    source_row, source_col = divmod(source_cell, grid_side)
    target_row, target_col = divmod(target_cell, grid_side)
    dr, dc = DELTAS[direction]
    return (target_row, target_col) == (source_row + dr, source_col + dc)


def component_purity_metrics(
    dsu: PotentialDSU, permutation: np.ndarray
) -> dict[str, float]:
    value = _validate_permutation(permutation, dsu.count)
    pure_coverage = 0
    aligned = 0
    largest_pure = 0
    nontrivial_coverage = 0
    for root in dsu.roots():
        component = dsu.positions[root]
        offsets: Counter[tuple[int, int]] = Counter()
        for tile, (row, col) in component.items():
            clean_row, clean_col = divmod(int(value[tile]), dsu.grid_side)
            offsets[(clean_row - row, clean_col - col)] += 1
        correct = max(offsets.values())
        aligned += correct
        if len(component) >= 2:
            nontrivial_coverage += len(component)
            if correct == len(component):
                pure_coverage += len(component)
                largest_pure = max(largest_pure, len(component))
    return {
        "pure_nontrivial_tile_coverage": pure_coverage / dsu.count,
        "nontrivial_tile_coverage": nontrivial_coverage / dsu.count,
        "translation_aligned_tile_accuracy": aligned / dsu.count,
        "largest_pure_component": float(largest_pure),
        "component_count": float(len(dsu.roots())),
        "largest_component": float(
            max((len(dsu.positions[root]) for root in dsu.roots()), default=1)
        ),
    }


def early_gate_metrics(
    motifs: Sequence[Plaquette],
    growth: GrowthResult,
    permutation: np.ndarray,
) -> dict[str, float]:
    """Label-only diagnostics; no label participates in construction."""

    dsu = growth.dsu
    value = _validate_permutation(permutation, dsu.count)
    truth = true_plaquette_keys(value, dsu.grid_side)
    proposal = {motif.tiles for motif in motifs}
    seeds = growth.seed_motifs
    exact_seed_count = sum(motif.tiles in truth for motif in seeds)
    seed_tiles = {tile for motif in seeds for tile in motif.tiles}
    certified: set[tuple[int, int, int]] = set()
    for root in dsu.roots():
        certified.update(dsu.certified_edges[root])
    exact_edges = sum(_edge_is_exact(edge, value, dsu.grid_side) for edge in certified)
    metrics = {
        "enumerated_motifs": float(len(motifs)),
        "tier_a_motifs": float(sum(motif.tier_a for motif in motifs)),
        "true_plaquette_proposal_recall": len(proposal & truth) / max(1, len(truth)),
        "accepted_seed_motifs": float(len(seeds)),
        "accepted_growth_motifs": float(len(growth.growth_motifs)),
        "exact_seed_motif_precision": exact_seed_count / max(1, len(seeds)),
        "seed_tile_coverage": len(seed_tiles) / dsu.count,
        "certified_edges": float(len(certified)),
        "certified_edge_precision": exact_edges / max(1, len(certified)),
        "growth_rounds": float(growth.rounds),
    }
    metrics.update(component_purity_metrics(dsu, value))
    return metrics


def fixed_gate_checks(
    metrics: dict[str, float],
    *,
    motif_precision: float = 0.95,
    seed_coverage: float = 0.15,
    grown_pure_coverage: float = 0.25,
    edge_precision: float = 0.97,
    largest_pure: float = 12.0,
) -> dict[str, bool]:
    return {
        "motif_precision": metrics["exact_seed_motif_precision"] >= motif_precision,
        "seed_coverage": metrics["seed_tile_coverage"] >= seed_coverage,
        "grown_pure_coverage": metrics["pure_nontrivial_tile_coverage"]
        >= grown_pure_coverage,
        "edge_precision": metrics["certified_edge_precision"] >= edge_precision,
        "largest_pure": metrics["largest_pure_component"] >= largest_pure,
    }


def make_synthetic_plaquette(
    tiles: Iterable[int],
    *,
    corner_mask: int = 0b1111,
    evidence: float = 0.0,
) -> Plaquette:
    """Small explicit constructor for smoke/contract tests."""

    values = tuple(map(int, tiles))
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("synthetic plaquette needs four distinct tile ids")
    return Plaquette(
        tiles=values,
        corner_mask=int(corner_mask),
        min_edge=float(evidence),
        mean_edge=float(evidence),
        reciprocal_rank_sum=8,
        witness_rank_sum=4,
    )


__all__ = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "DirectionalTopK",
    "Plaquette",
    "PotentialDSU",
    "GrowthResult",
    "PackingResult",
    "enumerate_plaquettes",
    "grow_plaquettes",
    "pack_components",
    "early_gate_metrics",
    "fixed_gate_checks",
    "true_plaquette_keys",
    "make_synthetic_plaquette",
]
