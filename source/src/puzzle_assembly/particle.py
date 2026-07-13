"""Deterministic multi-contact particle-beam assembly.

The solver in this module grows several partial layouts at once.  It differs
from a row-major beam in two important ways:

* the next empty cell is selected from the current frontier by the margin
  between its best and second-best unused tiles;
* every already occupied side of that cell contributes an oriented seam cost.

The beam is deterministic (there is no random resampling), but otherwise acts
like a small sequential Monte Carlo population: hypotheses branch, duplicate
hashes are removed, hash buckets retain some diversity, and weak particles are
pruned.  Initial components are translation-free and are tried at several
anchors, so callers do not need to provide a ground-truth absolute anchor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, validate_permutation


_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class ParticleBeamResult:
    """Result and inexpensive diagnostics from :func:`particle_beam_solver`."""

    position_to_slot: np.ndarray
    objective: float
    source: str
    anchors_started: int
    beam_steps: int
    completed_hypotheses: int


@dataclass
class _Particle:
    grid: np.ndarray
    unused: np.ndarray
    seam_cost: float
    seam_count: int
    unary_cost: float
    placed: int
    state_hash: int
    anchor_group: int


def _splitmix64(value: int) -> int:
    """Return a stable 64-bit mixing of one integer."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _tile_token(position: int, tile: int) -> int:
    return _splitmix64(position * TILE_COUNT + tile + 1)


def _calibrated_direction(
    matrix: np.ndarray,
    *,
    reciprocal_weight: float,
    raw_weight: float,
) -> np.ndarray:
    """Combine raw score, outgoing rank and reciprocal incoming rank.

    Rows are outgoing sides and columns are incoming sides.  A low column rank
    therefore means that the proposed target also regards the query as a good
    match for that incoming side.  Log ranks keep useful resolution near rank
    one while preventing one bad reciprocal rank from overwhelming all other
    contacts.
    """
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError("compatibility directions must be 576x576")
    finite = np.isfinite(values)
    if not np.all(np.any(finite, axis=1)):
        raise ValueError("every compatibility row must have a finite value")

    sortable = np.where(finite, values, np.inf)
    row_order = np.argsort(sortable, axis=1, kind="stable")
    column_order = np.argsort(sortable, axis=0, kind="stable")
    row_rank = np.empty_like(row_order, dtype=np.int32)
    column_rank = np.empty_like(column_order, dtype=np.int32)
    ranks = np.arange(TILE_COUNT, dtype=np.int32)
    row_rank[np.arange(TILE_COUNT)[:, None], row_order] = ranks[None, :]
    column_rank[column_order, np.arange(TILE_COUNT)[None, :]] = ranks[:, None]

    rank_scale = float(np.log1p(TILE_COUNT - 1))
    outgoing = np.log1p(row_rank.astype(np.float32)) / rank_scale
    incoming = np.log1p(column_rank.astype(np.float32)) / rank_scale
    rank_cost = (
        (1.0 - reciprocal_weight) * outgoing
        + reciprocal_weight * incoming
    )

    # Row-wise robust scaling retains a little magnitude information without
    # allowing score families with a large physical scale to dominate ranks.
    finite_values = np.where(finite, values, np.nan)
    row_min = np.nanmin(finite_values, axis=1, keepdims=True)
    row_q90 = np.nanpercentile(finite_values, 90.0, axis=1, keepdims=True)
    raw_scale = np.maximum(row_q90 - row_min, 1e-6)
    raw_cost = np.clip((values - row_min) / raw_scale, 0.0, 1.0)
    calibrated = (1.0 - raw_weight) * rank_cost + raw_weight * raw_cost
    calibrated = np.where(finite, calibrated, 2.0).astype(np.float32)
    np.fill_diagonal(calibrated, 2.0)
    return calibrated


def _placement_unary(compatibility: CompatibilityMatrices) -> np.ndarray:
    """Return a rank-calibrated outer-side cost for every position/tile."""

    def unit_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(TILE_COUNT, dtype=np.int32)
        ranks[order] = np.arange(TILE_COUNT, dtype=np.int32)
        return ranks.astype(np.float32) / float(TILE_COUNT - 1)

    right_values = np.where(
        np.isfinite(compatibility.right), compatibility.right, np.inf
    )
    down_values = np.where(
        np.isfinite(compatibility.down), compatibility.down, np.inf
    )
    outside = np.stack(
        [
            unit_ranks(np.min(right_values, axis=0)),
            unit_ranks(np.min(right_values, axis=1)),
            unit_ranks(np.min(down_values, axis=0)),
            unit_ranks(np.min(down_values, axis=1)),
        ],
        axis=1,
    )
    unary = np.empty((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    for position in range(TILE_COUNT):
        row, column = divmod(position, GRID)
        outer = np.asarray(
            [column == 0, column == GRID - 1, row == 0, row == GRID - 1],
            dtype=bool,
        )
        side_cost = np.where(outer[None, :], 1.0 - outside, outside)
        unary[position] = side_cost.mean(axis=1)
    return unary


def _normalise_seed_layouts(
    seed_layouts: Sequence[np.ndarray] | np.ndarray | None,
) -> list[np.ndarray]:
    if seed_layouts is None:
        return []
    if isinstance(seed_layouts, np.ndarray):
        values = np.asarray(seed_layouts)
        if values.shape == (TILE_COUNT,):
            candidates = [values]
        elif values.ndim == 2 and values.shape[1] == TILE_COUNT:
            candidates = [values[index] for index in range(len(values))]
        else:
            raise ValueError("seed_layouts must be 576 or Nx576")
    else:
        candidates = list(seed_layouts)
    return [
        validate_permutation(candidate, name=f"seed_layout_{index}").copy()
        for index, candidate in enumerate(candidates)
    ]


def _normalise_component(
    component: Mapping[int, tuple[int, int]],
    *,
    name: str,
) -> dict[int, tuple[int, int]]:
    if not isinstance(component, Mapping) or not component:
        raise ValueError(f"{name} must be a non-empty tile -> (x, y) mapping")
    parsed: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for raw_tile, raw_coordinate in component.items():
        tile = int(raw_tile)
        if tile < 0 or tile >= TILE_COUNT or tile != raw_tile:
            raise ValueError(f"{name} contains an invalid tile id")
        if len(raw_coordinate) != 2:
            raise ValueError(f"{name} coordinates must be (x, y)")
        x, y = int(raw_coordinate[0]), int(raw_coordinate[1])
        if x != raw_coordinate[0] or y != raw_coordinate[1]:
            raise ValueError(f"{name} coordinates must be integral")
        if (x, y) in occupied:
            raise ValueError(f"{name} assigns two tiles to {(x, y)}")
        parsed[tile] = (x, y)
        occupied.add((x, y))
    min_x = min(x for x, _ in parsed.values())
    min_y = min(y for _, y in parsed.values())
    normalised = {tile: (x - min_x, y - min_y) for tile, (x, y) in parsed.items()}
    max_x = max(x for x, _ in normalised.values())
    max_y = max(y for _, y in normalised.values())
    if max_x >= GRID or max_y >= GRID:
        raise ValueError(f"{name} does not fit inside the {GRID}x{GRID} board")
    return normalised


def _normalise_seed_components(
    seed_components: Sequence[Mapping[int, tuple[int, int]]] | None,
) -> list[dict[int, tuple[int, int]]]:
    if seed_components is None:
        return []
    return [
        _normalise_component(component, name=f"seed_component_{index}")
        for index, component in enumerate(seed_components)
    ]


def _layout_component(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    *,
    size: int,
) -> dict[int, tuple[int, int]]:
    """Extract one confident connected patch from a complete seed layout."""
    grid = layout.reshape(GRID, GRID)
    edges: list[tuple[float, int, int, int, int]] = []
    for row in range(GRID):
        for column in range(GRID - 1):
            first = int(grid[row, column])
            second = int(grid[row, column + 1])
            edges.append((float(right[first, second]), row, column, row, column + 1))
    for row in range(GRID - 1):
        for column in range(GRID):
            first = int(grid[row, column])
            second = int(grid[row + 1, column])
            edges.append((float(down[first, second]), row, column, row + 1, column))
    edges.sort()
    _, first_row, first_column, second_row, second_column = edges[0]
    selected = {(first_row, first_column), (second_row, second_column)}
    target_size = min(max(size, 2), TILE_COUNT)
    while len(selected) < target_size:
        frontier_edges = []
        for row, column in selected:
            tile = int(grid[row, column])
            for dr, dc, matrix, forward in (
                (0, -1, right, False),
                (0, 1, right, True),
                (-1, 0, down, False),
                (1, 0, down, True),
            ):
                nr, nc = row + dr, column + dc
                if not (0 <= nr < GRID and 0 <= nc < GRID):
                    continue
                if (nr, nc) in selected:
                    continue
                neighbour = int(grid[nr, nc])
                cost = matrix[tile, neighbour] if forward else matrix[neighbour, tile]
                frontier_edges.append((float(cost), nr, nc))
        if not frontier_edges:
            break
        _, row, column = min(frontier_edges)
        selected.add((row, column))
    min_row = min(row for row, _ in selected)
    min_column = min(column for _, column in selected)
    return {
        int(grid[row, column]): (column - min_column, row - min_row)
        for row, column in sorted(selected)
    }


def _data_seed_components(
    right: np.ndarray,
    down: np.ndarray,
    *,
    count: int,
) -> list[dict[int, tuple[int, int]]]:
    """Build diverse two-tile seeds from confident directed edges."""
    proposals: list[tuple[float, float, int, int, int, int]] = []
    for dx, dy, matrix in ((1, 0, right), (0, 1, down)):
        for first in range(TILE_COUNT):
            order = np.argsort(matrix[first], kind="stable")[:2]
            second = int(order[0])
            margin = float(matrix[first, order[1]] - matrix[first, second])
            proposals.append(
                (
                    float(matrix[first, second]) - 0.20 * margin,
                    -margin,
                    first,
                    second,
                    dx,
                    dy,
                )
            )
    proposals.sort()
    result = []
    tile_usage = np.zeros(TILE_COUNT, dtype=np.int16)
    seen: set[tuple[int, int, int, int]] = set()
    for _, _, first, second, dx, dy in proposals:
        key = (first, second, dx, dy)
        if key in seen or first == second:
            continue
        # The first pass favours seeds that do not all contain the same easy
        # texture tile.  A later relaxed pass below fills any remaining slots.
        if tile_usage[first] >= 2 or tile_usage[second] >= 2:
            continue
        result.append({first: (0, 0), second: (dx, dy)})
        tile_usage[first] += 1
        tile_usage[second] += 1
        seen.add(key)
        if len(result) >= count:
            return result
    for _, _, first, second, dx, dy in proposals:
        key = (first, second, dx, dy)
        if key in seen or first == second:
            continue
        result.append({first: (0, 0), second: (dx, dy)})
        seen.add(key)
        if len(result) >= count:
            break
    return result


def _component_offsets(
    component: Mapping[int, tuple[int, int]],
    unary: np.ndarray,
    *,
    hypotheses: int,
) -> list[tuple[int, int]]:
    """Choose low-unary and spatially diverse translations for a component."""
    max_x = max(x for x, _ in component.values())
    max_y = max(y for _, y in component.values())
    candidates: list[tuple[float, int, int]] = []
    member_items = sorted(component.items())
    for offset_y in range(GRID - max_y):
        for offset_x in range(GRID - max_x):
            cost = np.mean(
                [
                    unary[(y + offset_y) * GRID + x + offset_x, tile]
                    for tile, (x, y) in member_items
                ],
                dtype=np.float64,
            )
            candidates.append((float(cost), offset_y, offset_x))
    candidates.sort()
    selected: list[tuple[int, int]] = []
    if candidates:
        selected.append((candidates[0][2], candidates[0][1]))

    desired_centres = (
        (0.50, 0.50),
        (0.22, 0.22),
        (0.22, 0.78),
        (0.78, 0.22),
        (0.78, 0.78),
        (0.10, 0.50),
        (0.90, 0.50),
        (0.50, 0.10),
        (0.50, 0.90),
    )
    component_center_x = 0.5 * max_x
    component_center_y = 0.5 * max_y
    unary_range = max(candidates[-1][0] - candidates[0][0], 1e-6)
    for fraction_y, fraction_x in desired_centres:
        target_y = fraction_y * (GRID - 1)
        target_x = fraction_x * (GRID - 1)
        choice = min(
            candidates,
            key=lambda candidate: (
                (candidate[1] + component_center_y - target_y) ** 2
                + (candidate[2] + component_center_x - target_x) ** 2
                + 0.05 * (candidate[0] - candidates[0][0]) / unary_range,
                candidate[0],
                candidate[1],
                candidate[2],
            ),
        )
        offset = (choice[2], choice[1])
        if offset not in selected:
            selected.append(offset)
        if len(selected) >= hypotheses:
            break
    return selected[:hypotheses]


def _initial_particle(
    component: Mapping[int, tuple[int, int]],
    offset: tuple[int, int],
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    *,
    anchor_group: int,
) -> _Particle:
    grid = np.full(TILE_COUNT, -1, dtype=np.int32)
    unused = np.ones(TILE_COUNT, dtype=bool)
    coordinate_to_tile: dict[tuple[int, int], int] = {}
    state_hash = 0
    offset_x, offset_y = offset
    unary_cost = 0.0
    for tile, (x, y) in sorted(component.items()):
        row, column = y + offset_y, x + offset_x
        position = row * GRID + column
        if grid[position] >= 0 or not unused[tile]:
            raise RuntimeError("invalid component while creating a particle")
        grid[position] = tile
        unused[tile] = False
        coordinate_to_tile[(row, column)] = tile
        state_hash ^= _tile_token(position, tile)
        unary_cost += float(unary[position, tile])
    seam_cost = 0.0
    seam_count = 0
    for (row, column), tile in coordinate_to_tile.items():
        right_tile = coordinate_to_tile.get((row, column + 1))
        if right_tile is not None:
            seam_cost += float(right[tile, right_tile])
            seam_count += 1
        down_tile = coordinate_to_tile.get((row + 1, column))
        if down_tile is not None:
            seam_cost += float(down[tile, down_tile])
            seam_count += 1
    return _Particle(
        grid=grid,
        unused=unused,
        seam_cost=seam_cost,
        seam_count=seam_count,
        unary_cost=unary_cost,
        placed=len(component),
        state_hash=state_hash,
        anchor_group=anchor_group,
    )


def _priority(particle: _Particle, boundary_weight: float) -> tuple[float, int, int]:
    # A caller-supplied singleton is a valid translation hypothesis, but it
    # has no evidence yet and must not look artificially better than a strong
    # two-tile seed merely because its seam sum is zero.
    seam_mean = (
        particle.seam_cost / float(particle.seam_count)
        if particle.seam_count
        else 1.0
    )
    unary_mean = particle.unary_cost / float(max(particle.placed, 1))
    return (
        seam_mean + boundary_weight * unary_mean,
        -particle.placed,
        particle.state_hash,
    )


def _frontier(grid: np.ndarray, *, limit: int) -> np.ndarray:
    occupied = grid.reshape(GRID, GRID) >= 0
    neighbours = np.zeros((GRID, GRID), dtype=np.int8)
    neighbours[:, 1:] += occupied[:, :-1]
    neighbours[:, :-1] += occupied[:, 1:]
    neighbours[1:, :] += occupied[:-1, :]
    neighbours[:-1, :] += occupied[1:, :]
    positions = np.flatnonzero((~occupied & (neighbours > 0)).ravel()).astype(np.int32)
    if not len(positions):
        return positions
    counts = neighbours.ravel()[positions]
    order = np.lexsort((positions, -counts))
    if limit > 0:
        order = order[:limit]
    return positions[order]


def _local_candidate_costs(
    position: int,
    grid: np.ndarray,
    candidates: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    *,
    boundary_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    row, column = divmod(position, GRID)
    seam_sum = np.zeros(len(candidates), dtype=np.float32)
    neighbour_count = 0
    if column > 0 and grid[position - 1] >= 0:
        seam_sum += right[int(grid[position - 1]), candidates]
        neighbour_count += 1
    if column + 1 < GRID and grid[position + 1] >= 0:
        seam_sum += right[candidates, int(grid[position + 1])]
        neighbour_count += 1
    if row > 0 and grid[position - GRID] >= 0:
        seam_sum += down[int(grid[position - GRID]), candidates]
        neighbour_count += 1
    if row + 1 < GRID and grid[position + GRID] >= 0:
        seam_sum += down[candidates, int(grid[position + GRID])]
        neighbour_count += 1
    if neighbour_count <= 0:
        raise RuntimeError("frontier cell has no occupied neighbour")
    ranking = seam_sum / float(neighbour_count)
    ranking = ranking + boundary_weight * unary[position, candidates]
    return ranking.astype(np.float32, copy=False), seam_sum, neighbour_count


def _choose_frontier_position(
    particle: _Particle,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    *,
    boundary_weight: float,
    frontier_limit: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    candidates = np.flatnonzero(particle.unused).astype(np.int32)
    frontier = _frontier(particle.grid, limit=frontier_limit)
    if not len(frontier):
        raise RuntimeError("incomplete particle has an empty frontier")
    best_choice = None
    for position in frontier.tolist():
        ranking, seam_sum, neighbour_count = _local_candidate_costs(
            position,
            particle.grid,
            candidates,
            right,
            down,
            unary,
            boundary_weight=boundary_weight,
        )
        if len(candidates) >= 2:
            first_two = np.partition(ranking, 1)[:2]
            first_two.sort()
            margin = float(first_two[1] - first_two[0])
        else:
            margin = 0.0
        best = float(np.min(ranking))
        confidence = margin * np.sqrt(float(neighbour_count))
        key = (-confidence, -neighbour_count, best, position)
        if best_choice is None or key < best_choice[0]:
            best_choice = (
                key,
                position,
                ranking,
                seam_sum,
                neighbour_count,
            )
    assert best_choice is not None
    _, position, ranking, seam_sum, neighbour_count = best_choice
    return position, candidates, ranking, seam_sum, neighbour_count


def _expand_particle(
    particle: _Particle,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    *,
    top_k: int,
    boundary_weight: float,
    frontier_limit: int,
) -> list[_Particle]:
    if particle.placed >= TILE_COUNT:
        return [particle]
    position, candidates, ranking, seam_sum, neighbour_count = _choose_frontier_position(
        particle,
        right,
        down,
        unary,
        boundary_weight=boundary_weight,
        frontier_limit=frontier_limit,
    )
    take = min(top_k, len(candidates))
    if take < len(candidates):
        selected_indices = np.argpartition(ranking, take - 1)[:take]
    else:
        selected_indices = np.arange(len(candidates))
    selected_indices = selected_indices[
        np.lexsort((candidates[selected_indices], ranking[selected_indices]))
    ]
    children = []
    for candidate_index in selected_indices.tolist():
        tile = int(candidates[candidate_index])
        grid = particle.grid.copy()
        unused = particle.unused.copy()
        grid[position] = tile
        unused[tile] = False
        children.append(
            _Particle(
                grid=grid,
                unused=unused,
                seam_cost=particle.seam_cost + float(seam_sum[candidate_index]),
                seam_count=particle.seam_count + neighbour_count,
                unary_cost=particle.unary_cost + float(unary[position, tile]),
                placed=particle.placed + 1,
                state_hash=particle.state_hash ^ _tile_token(position, tile),
                anchor_group=particle.anchor_group,
            )
        )
    return children


def _prune(
    candidates: Sequence[_Particle],
    *,
    width: int,
    boundary_weight: float,
    diversity_buckets: int,
    preserve_anchors: bool,
) -> list[_Particle]:
    ordered = sorted(candidates, key=lambda item: _priority(item, boundary_weight))
    unique = []
    seen_hashes: set[int] = set()
    for particle in ordered:
        if particle.state_hash in seen_hashes:
            continue
        seen_hashes.add(particle.state_hash)
        unique.append(particle)

    selected: list[_Particle] = []
    selected_hashes: set[int] = set()
    if preserve_anchors and width >= 2:
        best_by_anchor: dict[int, _Particle] = {}
        for particle in unique:
            best_by_anchor.setdefault(particle.anchor_group, particle)
        anchor_best = sorted(
            best_by_anchor.values(), key=lambda item: _priority(item, boundary_weight)
        )
        for particle in anchor_best[: max(2, width // 2)]:
            selected.append(particle)
            selected_hashes.add(particle.state_hash)

    if diversity_buckets > 1 and len(selected) < width:
        used_buckets = {particle.state_hash % diversity_buckets for particle in selected}
        for particle in unique:
            bucket = particle.state_hash % diversity_buckets
            if particle.state_hash in selected_hashes or bucket in used_buckets:
                continue
            selected.append(particle)
            selected_hashes.add(particle.state_hash)
            used_buckets.add(bucket)
            if len(selected) >= width:
                break

    if len(selected) < width:
        for particle in unique:
            if particle.state_hash in selected_hashes:
                continue
            selected.append(particle)
            selected_hashes.add(particle.state_hash)
            if len(selected) >= width:
                break
    selected.sort(key=lambda item: _priority(item, boundary_weight))
    return selected[:width]


def _complete_objective(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    unary: np.ndarray,
    *,
    boundary_weight: float,
) -> float:
    grid = layout.reshape(GRID, GRID)
    seam = float(
        right[grid[:, :-1], grid[:, 1:]].sum(dtype=np.float64)
        + down[grid[:-1, :], grid[1:, :]].sum(dtype=np.float64)
    )
    seam_count = 2 * GRID * (GRID - 1)
    placement = float(
        unary[np.arange(TILE_COUNT, dtype=np.int32), layout].mean(dtype=np.float64)
    )
    return seam / float(seam_count) + boundary_weight * placement


def particle_beam_solver(
    compatibility: CompatibilityMatrices,
    *,
    seed_layouts: Sequence[np.ndarray] | np.ndarray | None = None,
    seed_components: Sequence[Mapping[int, tuple[int, int]]] | None = None,
    particles: int = 8,
    top_k: int = 3,
    anchor_hypotheses: int = 4,
    seed_component_size: int = 8,
    frontier_limit: int = 24,
    reciprocal_weight: float = 0.35,
    raw_weight: float = 0.20,
    boundary_weight: float = 0.08,
    diversity_buckets: int = 16,
    anchor_survival_steps: int = 24,
) -> ParticleBeamResult:
    """Assemble a 24x24 puzzle with a deterministic particle beam.

    Parameters
    ----------
    compatibility:
        Oriented right and down compatibility matrices.
    seed_layouts:
        Optional complete ``position_to_slot`` permutations.  A confident
        connected patch is extracted from each one; complete seeds are also
        retained as final fallback hypotheses.
    seed_components:
        Optional translation-free mappings ``tile -> (x, y)``.  Each component
        is evaluated at several absolute translations.
    particles / top_k:
        Beam width and branches per particle.  The defaults deliberately keep
        a 576-tile solve CPU-feasible.

    Returns
    -------
    ParticleBeamResult
        ``position_to_slot`` is always a validated one-to-one permutation.
    """
    if particles < 2:
        raise ValueError("particles must be at least 2")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if anchor_hypotheses < 2:
        raise ValueError("anchor_hypotheses must be at least 2")
    if seed_component_size < 2 or seed_component_size > TILE_COUNT:
        raise ValueError("seed_component_size must be in [2, 576]")
    if frontier_limit < 0:
        raise ValueError("frontier_limit must be non-negative")
    if not 0.0 <= reciprocal_weight <= 1.0:
        raise ValueError("reciprocal_weight must be in [0, 1]")
    if not 0.0 <= raw_weight <= 1.0:
        raise ValueError("raw_weight must be in [0, 1]")
    if boundary_weight < 0.0:
        raise ValueError("boundary_weight must be non-negative")
    if diversity_buckets <= 0 or anchor_survival_steps < 0:
        raise ValueError("invalid diversity or anchor-survival setting")

    layouts = _normalise_seed_layouts(seed_layouts)
    components = _normalise_seed_components(seed_components)
    right = _calibrated_direction(
        compatibility.right,
        reciprocal_weight=reciprocal_weight,
        raw_weight=raw_weight,
    )
    down = _calibrated_direction(
        compatibility.down,
        reciprocal_weight=reciprocal_weight,
        raw_weight=raw_weight,
    )
    unary = _placement_unary(compatibility)

    components.extend(
        _layout_component(
            layout,
            right,
            down,
            size=seed_component_size,
        )
        for layout in layouts
    )
    # Always retain data-derived alternatives so a bad optional seed cannot
    # monopolise the entire beam.
    data_seed_count = max(2, (particles + anchor_hypotheses - 1) // anchor_hypotheses)
    components.extend(
        _data_seed_components(right, down, count=data_seed_count)
    )
    if not components:
        raise RuntimeError("could not construct an initial component")

    initial_by_component: list[list[_Particle]] = []
    anchor_group = 0
    for component in components:
        offsets = _component_offsets(
            component,
            unary,
            hypotheses=anchor_hypotheses,
        )
        group = []
        for offset in offsets:
            group.append(
                _initial_particle(
                    component,
                    offset,
                    right,
                    down,
                    unary,
                    anchor_group=anchor_group,
                )
            )
            anchor_group += 1
        if group:
            initial_by_component.append(group)

    # Round-robin first: at least one hypothesis from as many independent
    # components as possible.  Remaining capacity goes to alternative anchors.
    beam: list[_Particle] = []
    depth = 0
    while len(beam) < particles:
        added = False
        for group in initial_by_component:
            if depth < len(group):
                beam.append(group[depth])
                added = True
                if len(beam) >= particles:
                    break
        if not added:
            break
        depth += 1
    beam = _prune(
        beam,
        width=particles,
        boundary_weight=boundary_weight,
        diversity_buckets=diversity_buckets,
        preserve_anchors=True,
    )
    anchors_started = len(beam)

    steps = 0
    while any(particle.placed < TILE_COUNT for particle in beam):
        expanded = []
        for particle in beam:
            expanded.extend(
                _expand_particle(
                    particle,
                    right,
                    down,
                    unary,
                    top_k=top_k,
                    boundary_weight=boundary_weight,
                    frontier_limit=frontier_limit,
                )
            )
        steps += 1
        beam = _prune(
            expanded,
            width=particles,
            boundary_weight=boundary_weight,
            diversity_buckets=diversity_buckets,
            preserve_anchors=steps <= anchor_survival_steps,
        )
        if not beam:
            raise RuntimeError("particle beam became empty")
        if steps > TILE_COUNT:
            raise RuntimeError("particle beam did not converge in 576 steps")

    completed = [particle for particle in beam if particle.placed == TILE_COUNT]
    if not completed:
        raise RuntimeError("particle beam produced no complete hypothesis")
    best_particle = min(completed, key=lambda item: _priority(item, boundary_weight))
    best_layout = validate_permutation(
        best_particle.grid.copy(), name="particle_position_to_slot"
    )
    best_objective = _complete_objective(
        best_layout,
        right,
        down,
        unary,
        boundary_weight=boundary_weight,
    )
    source = "particle"
    for index, layout in enumerate(layouts):
        objective = _complete_objective(
            layout,
            right,
            down,
            unary,
            boundary_weight=boundary_weight,
        )
        if objective < best_objective:
            best_layout = layout.copy()
            best_objective = objective
            source = f"seed_layout_{index}"

    return ParticleBeamResult(
        position_to_slot=validate_permutation(
            best_layout, name="particle_position_to_slot"
        ),
        objective=float(best_objective),
        source=source,
        anchors_started=anchors_started,
        beam_steps=steps,
        completed_hypotheses=len(completed),
    )


__all__ = ["ParticleBeamResult", "particle_beam_solver"]
