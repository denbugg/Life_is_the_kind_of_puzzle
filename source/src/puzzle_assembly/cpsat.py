"""Optional top-k directional CP-SAT grid embedding.

The puzzle stores only two oriented compatibility matrices: ``right[a, b]``
scores tile ``b`` immediately to the right of tile ``a``, and ``down[a, b]``
scores tile ``b`` immediately below tile ``a``.  Their rows and columns cover
all four physical sides (right/left and bottom/top respectively), so a grid
model does not need separate left/up matrices.

OR-Tools is deliberately imported inside :func:`topk_cpsat_grid_solver`.  This
keeps the rest of the assembly package usable in offline runtimes where the
optional dependency is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, validate_permutation


Direction = Literal["right", "down"]
EdgeKey = tuple[Direction, int, int]


@dataclass(frozen=True)
class DirectionalCandidate:
    """One low-cost oriented neighbour candidate and its integer reward."""

    direction: Direction
    first: int
    second: int
    cost: float
    outgoing_rank: int
    incoming_rank: int | None
    reward: int

    @property
    def key(self) -> EdgeKey:
        return (self.direction, self.first, self.second)

    @property
    def reciprocal(self) -> bool:
        return self.incoming_rank is not None


@dataclass(frozen=True)
class CandidateSquare:
    """A candidate 2x2 closure represented by its four oriented seams."""

    top_left: int
    top_right: int
    bottom_left: int
    bottom_right: int
    edge_keys: tuple[EdgeKey, EdgeKey, EdgeKey, EdgeKey]
    mean_reward: int


@dataclass(frozen=True)
class CPSATGridResult:
    """Validated position-to-slot layout plus serialisable solver diagnostics."""

    position_to_slot: np.ndarray
    diagnostics: dict[str, Any]


def _validate_candidate_parameters(top_k: int, reward_scale: int) -> None:
    if not 1 <= top_k < TILE_COUNT:
        raise ValueError(f"top_k must be in [1, {TILE_COUNT - 1}]")
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")


def _robust_row_quality(costs: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Map finite costs to [0, 1] quality without trusting their raw scale.

    Compatibility heads can have very different units and may contain extreme
    values.  Median/MAD plus an IQR fallback produces a monotone, bounded score
    while preserving useful distance information among the top candidates.
    """

    finite_costs = np.asarray(costs, dtype=np.float64)
    finite_costs = finite_costs[np.isfinite(finite_costs)]
    if finite_costs.size == 0:
        return np.zeros(len(selected), dtype=np.float64)
    center = float(np.median(finite_costs))
    mad = float(np.median(np.abs(finite_costs - center)))
    q25, q75 = np.quantile(finite_costs, (0.25, 0.75))
    scale = max(
        1.4826 * mad,
        float(q75 - q25) / 1.349,
        abs(center) * 1e-9,
        1e-12,
    )
    z = np.clip((center - np.asarray(selected, dtype=np.float64)) / scale, -12.0, 12.0)
    # Stable logistic.  The clipped input also avoids overflow on corrupt heads.
    return 1.0 / (1.0 + np.exp(-z))


def _extract_directional_candidates(
    matrix: np.ndarray,
    *,
    direction: Direction,
    top_k: int,
    reward_scale: int,
) -> tuple[DirectionalCandidate, ...]:
    """Extract deterministic finite top-k candidates from one cost matrix.

    This helper has no OR-Tools dependency and intentionally accepts arbitrary
    square matrices, which makes candidate conversion testable on tiny inputs.
    Incoming column rank supplies reciprocal evidence for the opposite physical
    side of the candidate tile.
    """

    if direction not in {"right", "down"}:
        raise ValueError("direction must be 'right' or 'down'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("directional compatibility must be a square matrix")
    tile_count = int(values.shape[0])
    if tile_count < 2 or top_k >= tile_count:
        raise ValueError("top_k must be smaller than the matrix dimension")

    finite = np.isfinite(values)
    finite = finite.copy()
    np.fill_diagonal(finite, False)
    sortable = np.where(finite, values, np.inf)
    row_order = np.argsort(sortable, axis=1, kind="stable")
    column_order = np.argsort(sortable, axis=0, kind="stable")
    column_rank = np.empty((tile_count, tile_count), dtype=np.int32)
    ranks = np.arange(tile_count, dtype=np.int32)
    column_rank[column_order, np.arange(tile_count)[None, :]] = ranks[:, None]

    candidates: list[DirectionalCandidate] = []
    for first in range(tile_count):
        valid_seconds = row_order[first, finite[first, row_order[first]]]
        take = min(top_k, int(valid_seconds.size))
        if take == 0:
            continue
        seconds = valid_seconds[:take]
        selected_costs = values[first, seconds]
        robust_quality = _robust_row_quality(values[first, finite[first]], selected_costs)
        # Rank is scale-free and prevents a flat/constant cost head from
        # assigning identical rewards to an arbitrary top-k tie.
        rank_quality = (take - np.arange(take, dtype=np.float64)) / float(take)
        quality = 0.70 * robust_quality + 0.30 * rank_quality
        rewards = np.maximum(1, np.rint(reward_scale * quality).astype(np.int64))
        for outgoing_rank, (second, cost, reward) in enumerate(
            zip(seconds.tolist(), selected_costs.tolist(), rewards.tolist())
        ):
            incoming = int(column_rank[first, second])
            incoming_rank = incoming if incoming < top_k and finite[first, second] else None
            candidates.append(
                DirectionalCandidate(
                    direction=direction,
                    first=first,
                    second=int(second),
                    cost=float(cost),
                    outgoing_rank=outgoing_rank,
                    incoming_rank=incoming_rank,
                    reward=int(reward),
                )
            )
    return tuple(candidates)


def extract_topk_candidates(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 4,
    reward_scale: int = 1_000,
) -> tuple[DirectionalCandidate, ...]:
    """Return robust integer-weighted right/down candidates, without OR-Tools."""

    _validate_candidate_parameters(top_k, reward_scale)
    right = np.asarray(compatibility.right)
    down = np.asarray(compatibility.down)
    expected = (TILE_COUNT, TILE_COUNT)
    if right.shape != expected or down.shape != expected:
        raise ValueError(f"compatibility matrices must both have shape {expected}")
    return _extract_directional_candidates(
        right, direction="right", top_k=top_k, reward_scale=reward_scale
    ) + _extract_directional_candidates(
        down, direction="down", top_k=top_k, reward_scale=reward_scale
    )


def _extract_candidate_squares(
    candidates: tuple[DirectionalCandidate, ...] | list[DirectionalCandidate],
    *,
    limit: int,
) -> tuple[CandidateSquare, ...]:
    """Find strongest top-k 2x2 closures, capped before model construction."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return ()

    by_key = {candidate.key: candidate for candidate in candidates}
    right: dict[int, dict[int, DirectionalCandidate]] = {}
    down: dict[int, dict[int, DirectionalCandidate]] = {}
    for candidate in candidates:
        target = right if candidate.direction == "right" else down
        target.setdefault(candidate.first, {})[candidate.second] = candidate

    found: dict[tuple[int, int, int, int], CandidateSquare] = {}
    for top_left in sorted(set(right).intersection(down)):
        for top_right in sorted(right[top_left]):
            below_top_right = down.get(top_right)
            if not below_top_right:
                continue
            for bottom_left in sorted(down[top_left]):
                right_of_bottom_left = right.get(bottom_left)
                if not right_of_bottom_left:
                    continue
                # Iterating the smaller target map bounds the common-neighbour
                # lookup for larger top-k settings.
                if len(below_top_right) <= len(right_of_bottom_left):
                    possible_bottom_right = below_top_right
                    other = right_of_bottom_left
                else:
                    possible_bottom_right = right_of_bottom_left
                    other = below_top_right
                for bottom_right in possible_bottom_right:
                    if bottom_right not in other:
                        continue
                    if len({top_left, top_right, bottom_left, bottom_right}) != 4:
                        continue
                    keys: tuple[EdgeKey, EdgeKey, EdgeKey, EdgeKey] = (
                        ("right", top_left, top_right),
                        ("down", top_left, bottom_left),
                        ("down", top_right, bottom_right),
                        ("right", bottom_left, bottom_right),
                    )
                    edge_rewards = [by_key[key].reward for key in keys]
                    square = CandidateSquare(
                        top_left=top_left,
                        top_right=top_right,
                        bottom_left=bottom_left,
                        bottom_right=bottom_right,
                        edge_keys=keys,
                        mean_reward=max(1, int(round(float(np.mean(edge_rewards))))),
                    )
                    found[(top_left, top_right, bottom_left, bottom_right)] = square

    ordered = sorted(
        found.values(),
        key=lambda square: (
            -square.mean_reward,
            square.top_left,
            square.top_right,
            square.bottom_left,
            square.bottom_right,
        ),
    )
    return tuple(ordered[:limit])


def _load_cp_model() -> Any:
    """Import the optional solver only when the CP-SAT path is requested."""

    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "topk_cpsat_grid_solver requires the optional 'ortools' package. "
            "Install OR-Tools in the runtime before calling this solver; for an "
            "offline Kaggle kernel, attach a compatible wheel or dataset first."
        ) from exc
    return cp_model


def _reified_equality(model: Any, left: Any, right: Any, offset: int, name: str) -> Any:
    literal = model.NewBoolVar(name)
    model.Add(left == right + offset).OnlyEnforceIf(literal)
    model.Add(left != right + offset).OnlyEnforceIf(literal.Not())
    return literal


def _exact_adjacency_literal(
    model: Any,
    *,
    candidate: DirectionalCandidate,
    rows: list[Any],
    columns: list[Any],
) -> Any:
    """Create a Boolean equivalent to one exact oriented grid adjacency."""

    first = candidate.first
    second = candidate.second
    prefix = f"{candidate.direction}_{first}_{second}"
    if candidate.direction == "right":
        aligned = _reified_equality(
            model, rows[second], rows[first], 0, f"{prefix}_same_row"
        )
        advanced = _reified_equality(
            model, columns[second], columns[first], 1, f"{prefix}_next_col"
        )
    else:
        aligned = _reified_equality(
            model, columns[second], columns[first], 0, f"{prefix}_same_col"
        )
        advanced = _reified_equality(
            model, rows[second], rows[first], 1, f"{prefix}_next_row"
        )
    adjacency = model.NewBoolVar(f"{prefix}_adjacent")
    # adjacency <=> (aligned AND advanced), not merely one-way implication.
    model.AddBoolAnd([aligned, advanced]).OnlyEnforceIf(adjacency)
    model.AddBoolOr([aligned.Not(), advanced.Not()]).OnlyEnforceIf(adjacency.Not())
    return adjacency


def topk_cpsat_grid_solver(
    compatibility: CompatibilityMatrices,
    *,
    top_k: int = 4,
    max_time_seconds: float = 60.0,
    workers: int = 1,
    seed: int = 0,
    reward_scale: int = 1_000,
    reciprocal_bonus: float = 0.20,
    square_bonus: float = 0.25,
    max_square_terms: int = 2_048,
    initial_position_to_slot: np.ndarray | None = None,
) -> CPSATGridResult:
    """Solve a weighted top-k directional embedding on the fixed 24x24 grid.

    Lower compatibility costs become larger bounded integer rewards.  Every
    candidate Boolean is exactly equivalent to its oriented adjacency, while
    reciprocal column-rank and closed 2x2 evidence add bounded bonus terms.
    ``workers=1`` is recommended for repeatable runs; a fixed seed is still set
    when parallel workers are explicitly requested.
    """

    _validate_candidate_parameters(top_k, reward_scale)
    if not np.isfinite(max_time_seconds) or max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be finite and positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be in [0, 2147483647]")
    if reciprocal_bonus < 0 or not np.isfinite(reciprocal_bonus):
        raise ValueError("reciprocal_bonus must be finite and non-negative")
    if square_bonus < 0 or not np.isfinite(square_bonus):
        raise ValueError("square_bonus must be finite and non-negative")
    if max_square_terms < 0:
        raise ValueError("max_square_terms must be non-negative")

    hint_layout: np.ndarray | None = None
    if initial_position_to_slot is not None:
        hint_layout = validate_permutation(
            initial_position_to_slot, name="initial_position_to_slot"
        ).copy()

    # Candidate preparation is pure NumPy and happens before importing OR-Tools,
    # making malformed inputs fail consistently even across optional runtimes.
    candidates = extract_topk_candidates(
        compatibility, top_k=top_k, reward_scale=reward_scale
    )
    squares = (
        _extract_candidate_squares(candidates, limit=max_square_terms)
        if square_bonus > 0 and max_square_terms > 0
        else ()
    )
    cp_model = _load_cp_model()

    model = cp_model.CpModel()
    positions = [
        model.NewIntVar(0, TILE_COUNT - 1, f"position_{tile}")
        for tile in range(TILE_COUNT)
    ]
    rows = [model.NewIntVar(0, GRID - 1, f"row_{tile}") for tile in range(TILE_COUNT)]
    columns = [
        model.NewIntVar(0, GRID - 1, f"column_{tile}") for tile in range(TILE_COUNT)
    ]
    model.AddAllDifferent(positions)
    for tile in range(TILE_COUNT):
        model.Add(positions[tile] == GRID * rows[tile] + columns[tile])

    edge_literals: dict[EdgeKey, Any] = {}
    objective_terms: list[Any] = []
    reciprocal_candidates = 0
    total_edge_reward = 0
    total_reciprocal_bonus = 0
    for candidate in candidates:
        literal = _exact_adjacency_literal(
            model, candidate=candidate, rows=rows, columns=columns
        )
        edge_literals[candidate.key] = literal
        coefficient = candidate.reward
        if candidate.incoming_rank is not None:
            reciprocal_candidates += 1
            if reciprocal_bonus > 0:
                reciprocal_strength = (top_k - candidate.incoming_rank) / float(top_k)
                bonus = int(round(reward_scale * reciprocal_bonus * reciprocal_strength))
                coefficient += max(0, bonus)
                total_reciprocal_bonus += max(0, bonus)
        objective_terms.append(coefficient * literal)
        total_edge_reward += candidate.reward

    square_literals: list[Any] = []
    total_square_bonus = 0
    for index, square in enumerate(squares):
        edges = [edge_literals[key] for key in square.edge_keys]
        literal = model.NewBoolVar(f"square_{index}")
        model.AddBoolAnd(edges).OnlyEnforceIf(literal)
        model.AddBoolOr([literal] + [edge.Not() for edge in edges])
        coefficient = max(1, int(round(square_bonus * square.mean_reward)))
        objective_terms.append(coefficient * literal)
        square_literals.append(literal)
        total_square_bonus += coefficient

    if objective_terms:
        model.Maximize(sum(objective_terms))

    hint_tile_to_position: np.ndarray | None = None
    if hint_layout is not None:
        hint_tile_to_position = np.empty(TILE_COUNT, dtype=np.int32)
        hint_tile_to_position[hint_layout] = np.arange(TILE_COUNT, dtype=np.int32)
        for tile, position in enumerate(hint_tile_to_position.tolist()):
            row, column = divmod(position, GRID)
            model.AddHint(positions[tile], position)
            model.AddHint(rows[tile], row)
            model.AddHint(columns[tile], column)
        for candidate in candidates:
            first_position = int(hint_tile_to_position[candidate.first])
            second_position = int(hint_tile_to_position[candidate.second])
            first_row, first_column = divmod(first_position, GRID)
            second_row, second_column = divmod(second_position, GRID)
            if candidate.direction == "right":
                active = first_row == second_row and second_column == first_column + 1
            else:
                active = first_column == second_column and second_row == first_row + 1
            model.AddHint(edge_literals[candidate.key], int(active))
        for square, literal in zip(squares, square_literals):
            active = all(
                (
                    int(hint_tile_to_position[second])
                    == int(hint_tile_to_position[first]) + (1 if direction == "right" else GRID)
                    and (
                        direction == "down"
                        or int(hint_tile_to_position[first]) % GRID < GRID - 1
                    )
                )
                for direction, first, second in square.edge_keys
            )
            model.AddHint(literal, int(active))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.randomize_search = False
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status not in {cp_model.FEASIBLE, cp_model.OPTIMAL}:
        raise RuntimeError(
            "CP-SAT did not return a feasible grid within the configured limit "
            f"(status={status_name}, seconds={max_time_seconds}, workers={workers})."
        )

    tile_to_position = np.fromiter(
        (solver.Value(variable) for variable in positions),
        dtype=np.int32,
        count=TILE_COUNT,
    )
    position_to_slot = np.empty(TILE_COUNT, dtype=np.int32)
    position_to_slot[tile_to_position] = np.arange(TILE_COUNT, dtype=np.int32)
    position_to_slot = validate_permutation(
        position_to_slot, name="cpsat_position_to_slot"
    )

    selected_edges = sum(int(solver.Value(literal)) for literal in edge_literals.values())
    selected_reciprocal = sum(
        int(solver.Value(edge_literals[candidate.key]))
        for candidate in candidates
        if candidate.reciprocal
    )
    selected_squares = sum(int(solver.Value(literal)) for literal in square_literals)
    objective = float(solver.ObjectiveValue())
    bound = float(solver.BestObjectiveBound())
    diagnostics: dict[str, Any] = {
        "status": status_name,
        "optimal": status == cp_model.OPTIMAL,
        "deterministic_parallelism": workers == 1,
        "seed": int(seed),
        "workers": int(workers),
        "time_limit_seconds": float(max_time_seconds),
        "wall_time_seconds": float(solver.WallTime()),
        "branches": int(solver.NumBranches()),
        "conflicts": int(solver.NumConflicts()),
        "objective": objective,
        "best_objective_bound": bound,
        "relative_objective_gap": max(0.0, bound - objective) / max(abs(objective), 1.0),
        "top_k": int(top_k),
        "candidate_edges": len(candidates),
        "right_candidates": sum(candidate.direction == "right" for candidate in candidates),
        "down_candidates": sum(candidate.direction == "down" for candidate in candidates),
        "reciprocal_candidates": reciprocal_candidates,
        "candidate_squares": len(squares),
        "selected_candidate_edges": selected_edges,
        "selected_reciprocal_edges": selected_reciprocal,
        "selected_squares": selected_squares,
        "total_base_reward": total_edge_reward,
        "total_reciprocal_bonus": total_reciprocal_bonus,
        "total_square_bonus": total_square_bonus,
        "used_initial_hint": hint_layout is not None,
    }
    return CPSATGridResult(position_to_slot=position_to_slot, diagnostics=diagnostics)


# Short aliases make the optional solver discoverable without imposing one
# naming convention on evaluation scripts.
cpsat_grid_solver = topk_cpsat_grid_solver
topk_directional_cpsat_solver = topk_cpsat_grid_solver


__all__ = [
    "CPSATGridResult",
    "CandidateSquare",
    "DirectionalCandidate",
    "cpsat_grid_solver",
    "extract_topk_candidates",
    "topk_cpsat_grid_solver",
    "topk_directional_cpsat_solver",
]
