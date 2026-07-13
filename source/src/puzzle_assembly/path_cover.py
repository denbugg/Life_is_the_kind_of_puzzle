"""Exact fixed-length directed path covers for axis-first puzzle assembly.

This module deliberately solves a narrower problem than the existing grid QAP:
given a directed cost graph on ``N = path_count * path_length`` tiles, select
exactly ``path_count`` vertex-disjoint paths of exactly ``path_length`` tiles.
The formulation is useful for recovering the 24 image rows (or columns) before
attempting the orthogonal ordering problem.

OR-Tools is an optional dependency and is imported only when
:func:`solve_candidate_path_cover` is called.  Candidate extraction and result
validation remain dependency-free so protocol construction and correctness
tests can run in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Collection, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DirectedCandidate:
    """One finite directed edge retained from a cost-matrix row."""

    source: int
    destination: int
    cost: float
    outgoing_rank: int

    @property
    def edge(self) -> tuple[int, int]:
        return (self.source, self.destination)


@dataclass(frozen=True)
class PathCoverResult:
    """A validated exact path cover and serialisable solver diagnostics."""

    paths: tuple[tuple[int, ...], ...]
    accepted_candidate: bool
    used_reference_fallback: bool
    fallback_reason: str | None
    diagnostics: dict[str, Any]


class PathCoverInfeasibleError(RuntimeError):
    """Raised when the retained candidate graph has no requested exact cover."""


def _validate_shape(node_count: int, path_count: int, path_length: int) -> None:
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
        raise TypeError("node_count must be an integer")
    if isinstance(path_count, bool) or not isinstance(path_count, (int, np.integer)):
        raise TypeError("path_count must be an integer")
    if isinstance(path_length, bool) or not isinstance(path_length, (int, np.integer)):
        raise TypeError("path_length must be an integer")
    if node_count <= 0 or path_count <= 0 or path_length <= 0:
        raise ValueError("node_count, path_count, and path_length must be positive")
    if path_count * path_length != node_count:
        raise ValueError("node_count must equal path_count * path_length")


def extract_topk_directed_candidates(
    cost_matrix: np.ndarray,
    *,
    top_k: int,
) -> tuple[DirectedCandidate, ...]:
    """Extract deterministic finite non-self top-k candidates per source.

    Ties are resolved by ascending destination node.  Rows may retain fewer
    than ``top_k`` edges when the remaining entries are non-finite; feasibility
    is intentionally left to the exact-cover solver.
    """

    try:
        costs = np.asarray(cost_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("cost_matrix must be convertible to float64") from exc
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("cost_matrix must be square")
    node_count = int(costs.shape[0])
    if node_count < 2:
        raise ValueError("cost_matrix must contain at least two nodes")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise TypeError("top_k must be an integer")
    if not 1 <= int(top_k) < node_count:
        raise ValueError("top_k must lie in [1, node_count - 1]")

    candidates: list[DirectedCandidate] = []
    destinations = np.arange(node_count, dtype=np.int64)
    for source in range(node_count):
        finite = np.isfinite(costs[source])
        finite[source] = False
        allowed = destinations[finite]
        if not len(allowed):
            continue
        # lexsort uses the last key first: cost is primary, node id breaks ties.
        order = np.lexsort((allowed, costs[source, allowed]))
        selected = allowed[order[: int(top_k)]]
        for rank, destination in enumerate(selected.tolist()):
            candidates.append(
                DirectedCandidate(
                    source=source,
                    destination=int(destination),
                    cost=float(costs[source, destination]),
                    outgoing_rank=rank,
                )
            )
    return tuple(candidates)


def extract_union_directed_candidates(
    cost_matrix: np.ndarray,
    *,
    outgoing_top_k: int,
    incoming_top_k: int,
    rescue_edges: Iterable[tuple[int, int]] = (),
    rescue_cost: float | None = None,
) -> tuple[DirectedCandidate, ...]:
    """Return outgoing/incoming top-k union plus optional low-priority arcs.

    ``rescue_edges`` are intended for a frozen input-only reference cover.  A
    missing rescue cost is placed strictly above every regular union cost, so
    rescue arcs guarantee feasibility without pretending to be strong image
    evidence.  Existing union edges retain their measured cost.
    """

    try:
        costs = np.asarray(cost_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("cost_matrix must be convertible to float64") from exc
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("cost_matrix must be square")
    node_count = int(costs.shape[0])
    if node_count < 2:
        raise ValueError("cost_matrix must contain at least two nodes")
    for name, value in (
        ("outgoing_top_k", outgoing_top_k),
        ("incoming_top_k", incoming_top_k),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if not 1 <= int(value) < node_count:
            raise ValueError(f"{name} must lie in [1, node_count - 1]")

    outgoing = extract_topk_directed_candidates(
        costs, top_k=int(outgoing_top_k)
    )
    by_edge: dict[tuple[int, int], DirectedCandidate] = {
        candidate.edge: candidate for candidate in outgoing
    }
    nodes = np.arange(node_count, dtype=np.int64)
    for destination in range(node_count):
        finite = np.isfinite(costs[:, destination])
        finite[destination] = False
        sources = nodes[finite]
        if not len(sources):
            continue
        order = np.lexsort((sources, costs[sources, destination]))
        for incoming_rank, source in enumerate(
            sources[order[: int(incoming_top_k)]].tolist()
        ):
            edge = (int(source), destination)
            if edge in by_edge:
                continue
            by_edge[edge] = DirectedCandidate(
                source=int(source),
                destination=destination,
                cost=float(costs[source, destination]),
                outgoing_rank=int(outgoing_top_k) + incoming_rank,
            )

    regular_costs = np.asarray(
        [candidate.cost for candidate in by_edge.values()], dtype=np.float64
    )
    if rescue_cost is None:
        if not len(regular_costs):
            default_rescue_cost = 1.0
        else:
            span = float(regular_costs.max() - regular_costs.min())
            default_rescue_cost = float(
                regular_costs.max() + max(span, abs(float(regular_costs.max())) * 1e-6, 1.0)
            )
    else:
        if not np.isfinite(rescue_cost):
            raise ValueError("rescue_cost must be finite")
        default_rescue_cost = float(rescue_cost)

    try:
        rescue = tuple(rescue_edges)
    except TypeError as exc:
        raise TypeError("rescue_edges must be iterable") from exc
    for raw_edge in rescue:
        if not isinstance(raw_edge, (tuple, list)) or len(raw_edge) != 2:
            raise TypeError("every rescue edge must be a (source, destination) pair")
        source, destination = raw_edge
        if isinstance(source, bool) or isinstance(destination, bool):
            raise TypeError("rescue edge endpoints must be integers")
        if not isinstance(source, (int, np.integer)) or not isinstance(
            destination, (int, np.integer)
        ):
            raise TypeError("rescue edge endpoints must be integers")
        source = int(source)
        destination = int(destination)
        if not 0 <= source < node_count or not 0 <= destination < node_count:
            raise ValueError("rescue edge endpoint is outside the node range")
        if source == destination:
            raise ValueError("self-edge rescue candidates are forbidden")
        edge = (source, destination)
        if edge not in by_edge:
            by_edge[edge] = DirectedCandidate(
                source=source,
                destination=destination,
                cost=default_rescue_cost,
                outgoing_rank=int(outgoing_top_k) + int(incoming_top_k),
            )
    return tuple(
        sorted(
            by_edge.values(),
            key=lambda candidate: (
                candidate.source,
                candidate.destination,
                candidate.cost,
                candidate.outgoing_rank,
            ),
        )
    )


def path_cover_edges(
    paths: Iterable[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Return directed consecutive edges in canonical path order."""

    normalized = tuple(tuple(int(node) for node in path) for path in paths)
    return tuple(
        (path[index], path[index + 1])
        for path in normalized
        for index in range(len(path) - 1)
    )


def validate_exact_path_cover(
    paths: Iterable[Sequence[int]],
    *,
    node_count: int,
    path_count: int,
    path_length: int,
    allowed_edges: Collection[tuple[int, int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate, canonicalise, and return an exact fixed-length path cover.

    Path labels are immaterial, so the returned paths are sorted
    lexicographically.  The orientation inside each path is preserved.
    """

    _validate_shape(node_count, path_count, path_length)
    try:
        raw_paths = tuple(paths)
    except TypeError as exc:
        raise TypeError("paths must be iterable") from exc
    if len(raw_paths) != path_count:
        raise ValueError(f"expected {path_count} paths, got {len(raw_paths)}")

    normalized: list[tuple[int, ...]] = []
    for path_index, path in enumerate(raw_paths):
        if isinstance(path, (str, bytes)):
            raise TypeError("each path must be a sequence of integer nodes")
        try:
            raw_nodes = tuple(path)
        except TypeError as exc:
            raise TypeError(f"path {path_index} is not iterable") from exc
        if len(raw_nodes) != path_length:
            raise ValueError(
                f"path {path_index} has length {len(raw_nodes)}, expected {path_length}"
            )
        nodes: list[int] = []
        for node in raw_nodes:
            if isinstance(node, bool) or not isinstance(node, (int, np.integer)):
                raise TypeError("path nodes must be integers")
            value = int(node)
            if not 0 <= value < node_count:
                raise ValueError(f"path node {value} is outside [0, {node_count})")
            nodes.append(value)
        normalized.append(tuple(nodes))

    flattened = tuple(node for path in normalized for node in path)
    if len(set(flattened)) != node_count:
        raise ValueError("paths must use every node exactly once")
    if set(flattened) != set(range(node_count)):
        raise ValueError("paths do not cover exactly nodes 0..node_count-1")

    if allowed_edges is not None:
        allowed = set(allowed_edges)
        for edge in path_cover_edges(normalized):
            if edge not in allowed:
                raise ValueError(f"path cover uses unavailable edge {edge}")
    return tuple(sorted(normalized))


def exhaustive_path_cover_reference(
    candidates: Iterable[DirectedCandidate],
    *,
    node_count: int,
    path_count: int,
    path_length: int,
    maximum_nodes: int = 9,
) -> PathCoverResult:
    """Pure exhaustive reference solver for tiny correctness fixtures only.

    This is intentionally guarded against accidental production use.  It is
    useful when OR-Tools is absent and provides an independent specification of
    the exact-cover objective exercised by unit tests.
    """

    _validate_shape(node_count, path_count, path_length)
    if maximum_nodes <= 0:
        raise ValueError("maximum_nodes must be positive")
    if node_count > maximum_nodes:
        raise ValueError(
            f"exhaustive reference is limited to {maximum_nodes} nodes, got {node_count}"
        )
    canonical = _validate_candidates(candidates, node_count=node_count)
    costs = {candidate.edge: candidate.cost for candidate in canonical}

    best_paths: tuple[tuple[int, ...], ...] | None = None
    best_cost = np.inf
    feasible_covers = 0
    # Canonical path sorting removes the path-label symmetry.  The remaining
    # enumeration is deliberately literal and independent of the CP-SAT model.
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for ordering in permutations(range(node_count)):
        paths = tuple(
            tuple(ordering[offset : offset + path_length])
            for offset in range(0, node_count, path_length)
        )
        paths = tuple(sorted(paths))
        if paths in seen:
            continue
        seen.add(paths)
        edges = path_cover_edges(paths)
        if any(edge not in costs for edge in edges):
            continue
        feasible_covers += 1
        objective = float(sum(costs[edge] for edge in edges))
        key = (objective, paths)
        best_key = (best_cost, best_paths if best_paths is not None else ())
        if best_paths is None or key < best_key:
            best_cost = objective
            best_paths = paths

    if best_paths is None:
        raise PathCoverInfeasibleError("candidate graph has no requested exact path cover")
    validated = validate_exact_path_cover(
        best_paths,
        node_count=node_count,
        path_count=path_count,
        path_length=path_length,
        allowed_edges=set(costs),
    )
    return PathCoverResult(
        paths=validated,
        accepted_candidate=True,
        used_reference_fallback=False,
        fallback_reason=None,
        diagnostics={
            "solver": "pure_exhaustive_reference",
            "status": "OPTIMAL",
            "optimal": True,
            "node_count": node_count,
            "path_count": path_count,
            "path_length": path_length,
            "candidate_count": len(canonical),
            "selected_edge_count": node_count - path_count,
            "objective_cost": best_cost,
            "feasible_cover_count": feasible_covers,
        },
    )


def _validate_candidates(
    candidates: Iterable[DirectedCandidate],
    *,
    node_count: int,
) -> tuple[DirectedCandidate, ...]:
    try:
        values = tuple(candidates)
    except TypeError as exc:
        raise TypeError("candidates must be iterable") from exc
    by_edge: dict[tuple[int, int], DirectedCandidate] = {}
    for candidate in values:
        if not isinstance(candidate, DirectedCandidate):
            raise TypeError("every candidate must be a DirectedCandidate")
        if isinstance(candidate.source, bool) or isinstance(candidate.destination, bool):
            raise TypeError("candidate endpoints must be integer nodes")
        if not isinstance(candidate.source, (int, np.integer)) or not isinstance(
            candidate.destination, (int, np.integer)
        ):
            raise TypeError("candidate endpoints must be integer nodes")
        source = int(candidate.source)
        destination = int(candidate.destination)
        if not 0 <= source < node_count or not 0 <= destination < node_count:
            raise ValueError("candidate endpoint is outside the node range")
        if source == destination:
            raise ValueError("self-edge candidates are forbidden")
        if not np.isfinite(candidate.cost):
            raise ValueError("candidate costs must be finite")
        if isinstance(candidate.outgoing_rank, bool) or not isinstance(
            candidate.outgoing_rank, (int, np.integer)
        ):
            raise TypeError("candidate outgoing_rank must be an integer")
        if int(candidate.outgoing_rank) < 0:
            raise ValueError("candidate outgoing_rank must be non-negative")
        edge = (source, destination)
        if edge in by_edge:
            raise ValueError(f"duplicate directed candidate {edge}")
        by_edge[edge] = DirectedCandidate(
            source=source,
            destination=destination,
            cost=float(candidate.cost),
            outgoing_rank=int(candidate.outgoing_rank),
        )
    return tuple(
        sorted(
            by_edge.values(),
            key=lambda candidate: (
                candidate.source,
                candidate.destination,
                candidate.cost,
                candidate.outgoing_rank,
            ),
        )
    )


def _load_cp_model() -> Any:
    """Import the optional CP-SAT dependency only at solver call time."""

    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "solve_candidate_path_cover requires the optional 'ortools' package"
        ) from exc
    return cp_model


def _integer_penalties(
    candidates: tuple[DirectedCandidate, ...],
    *,
    cost_scale: int,
) -> tuple[tuple[int, ...], float, float]:
    if isinstance(cost_scale, bool) or not isinstance(cost_scale, (int, np.integer)):
        raise TypeError("cost_scale must be an integer")
    if not 1 <= int(cost_scale) <= 1_000_000_000:
        raise ValueError("cost_scale must lie in [1, 1_000_000_000]")
    if not candidates:
        return (), 0.0, 0.0
    costs = np.asarray([candidate.cost for candidate in candidates], dtype=np.float64)
    offset = float(costs.min())
    span = float(costs.max() - offset)
    if span <= np.finfo(np.float64).eps * max(1.0, abs(offset)):
        return tuple(0 for _ in candidates), offset, 0.0
    normalized = np.clip((costs - offset) / span, 0.0, 1.0)
    penalties = np.rint(normalized * int(cost_scale)).astype(np.int64)
    return tuple(int(value) for value in penalties.tolist()), offset, span


def solve_candidate_path_cover(
    candidates: Iterable[DirectedCandidate],
    *,
    node_count: int,
    path_count: int,
    path_length: int,
    time_limit_seconds: float = 30.0,
    random_seed: int = 0,
    cost_scale: int = 1_000_000,
    require_optimal: bool = True,
    reference_paths: Iterable[Sequence[int]] | None = None,
    require_strict_reference_improvement: bool = True,
    reference_improvement_feasibility: bool = False,
) -> PathCoverResult:
    """Solve an exact fixed-length directed path cover with CP-SAT.

    Every node receives one depth.  Exactly ``path_count`` nodes occupy each
    depth; selected edges advance by exactly one depth; nonfinal nodes have one
    selected successor and nonstart nodes one selected predecessor.  Therefore
    every node belongs to one requested path and directed subtours are
    impossible.

    The default optimization mode is single-threaded with fixed search.  A
    wall-time cutoff that returns only ``FEASIBLE`` can still depend on machine
    speed, so scientific optimization runs should retain
    ``require_optimal=True``.

    With ``reference_improvement_feasibility=True``, a validated input-only
    reference is mandatory.  The model adds the exact integer constraint
    ``candidate_primary < reference_primary`` and removes the objective,
    turning the solve into pure satisfaction.  In that mode the existing
    ``time_limit_seconds`` value is deliberately interpreted as CP-SAT
    *deterministic-time units*, not wall seconds.  Any returned solution is a
    complete satisfying cover; ``UNKNOWN``/timeout falls back to the reference,
    and raw floating cost is checked again before acceptance.
    """

    _validate_shape(node_count, path_count, path_length)
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0.0:
        raise ValueError("time_limit_seconds must be positive and finite")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)):
        raise TypeError("random_seed must be an integer")
    if not 0 <= int(random_seed) < 2**31:
        raise ValueError("random_seed must lie in [0, 2**31)")
    if not isinstance(require_optimal, bool):
        raise TypeError("require_optimal must be a bool")
    if not isinstance(require_strict_reference_improvement, bool):
        raise TypeError("require_strict_reference_improvement must be a bool")
    if not isinstance(reference_improvement_feasibility, bool):
        raise TypeError("reference_improvement_feasibility must be a bool")
    if reference_improvement_feasibility and reference_paths is None:
        raise ValueError(
            "reference_improvement_feasibility requires reference_paths"
        )
    if reference_improvement_feasibility and not require_strict_reference_improvement:
        raise ValueError(
            "reference_improvement_feasibility requires strict reference improvement"
        )

    canonical = _validate_candidates(candidates, node_count=node_count)
    selected_edge_count = node_count - path_count
    if len(canonical) < selected_edge_count:
        raise PathCoverInfeasibleError(
            "candidate graph has fewer edges than any requested exact cover"
        )
    sources = {candidate.source for candidate in canonical}
    destinations = {candidate.destination for candidate in canonical}
    if node_count - len(sources) > path_count:
        raise PathCoverInfeasibleError(
            "too many nodes lack outgoing candidates to fit the final depth"
        )
    if node_count - len(destinations) > path_count:
        raise PathCoverInfeasibleError(
            "too many nodes lack incoming candidates to fit the start depth"
        )

    edge_cost = {candidate.edge: candidate.cost for candidate in canonical}
    reference: tuple[tuple[int, ...], ...] | None = None
    reference_edges: set[tuple[int, int]] = set()
    reference_cost: float | None = None
    if reference_paths is not None:
        reference = validate_exact_path_cover(
            reference_paths,
            node_count=node_count,
            path_count=path_count,
            path_length=path_length,
            allowed_edges=set(edge_cost),
        )
        reference_edges = set(path_cover_edges(reference))
        reference_cost = float(sum(edge_cost[edge] for edge in reference_edges))

    penalties, cost_offset, cost_span = _integer_penalties(
        canonical, cost_scale=cost_scale
    )
    penalty_by_edge = {
        candidate.edge: penalty
        for candidate, penalty in zip(canonical, penalties)
    }
    reference_primary_integer: int | None = None
    if reference is not None:
        reference_primary_integer = int(
            sum(penalty_by_edge[edge] for edge in reference_edges)
        )
    if reference_improvement_feasibility and reference_primary_integer == 0:
        return _reference_fallback_result(
            reference,
            reason="no_strict_integer_improvement_possible",
            reference_cost=float(reference_cost),
            diagnostics={
                "solver": "pre_solve_integer_bound",
                "status": "INFEASIBLE_BY_BOUND",
                "optimal": None,
                "node_count": node_count,
                "path_count": path_count,
                "path_length": path_length,
                "candidate_count": len(canonical),
                "reference_objective_cost": reference_cost,
                "reference_primary_integer_objective": reference_primary_integer,
                "candidate_primary_integer_objective": None,
                "best_objective_bound": None,
                "deterministic_time": 0.0,
                "deterministic_time_limit": float(time_limit_seconds),
                "time_limit_kind": "cp_sat_deterministic_time",
            },
        )
    cp_model = _load_cp_model()
    model = cp_model.CpModel()

    at_depth = [
        [model.NewBoolVar(f"node_{node}_depth_{depth}") for depth in range(path_length)]
        for node in range(node_count)
    ]
    depths = [model.NewIntVar(0, path_length - 1, f"depth_{node}") for node in range(node_count)]
    for node in range(node_count):
        model.AddExactlyOne(at_depth[node])
        model.Add(
            depths[node]
            == sum(depth * at_depth[node][depth] for depth in range(path_length))
        )
    for depth in range(path_length):
        model.Add(
            sum(at_depth[node][depth] for node in range(node_count)) == path_count
        )

    edge_variables: list[Any] = []
    outgoing: list[list[Any]] = [[] for _ in range(node_count)]
    incoming: list[list[Any]] = [[] for _ in range(node_count)]
    for index, candidate in enumerate(canonical):
        variable = model.NewBoolVar(
            f"edge_{candidate.source}_{candidate.destination}_{index}"
        )
        edge_variables.append(variable)
        outgoing[candidate.source].append(variable)
        incoming[candidate.destination].append(variable)
        model.Add(depths[candidate.destination] == depths[candidate.source] + 1).OnlyEnforceIf(
            variable
        )

    if reference is not None:
        reference_depth: dict[int, int] = {
            node: depth for path in reference for depth, node in enumerate(path)
        }
        for node in range(node_count):
            depth = reference_depth[node]
            model.AddHint(depths[node], depth)
            for candidate_depth in range(path_length):
                model.AddHint(
                    at_depth[node][candidate_depth],
                    int(candidate_depth == depth),
                )
        for candidate, variable in zip(canonical, edge_variables):
            model.AddHint(variable, int(candidate.edge in reference_edges))

    for node in range(node_count):
        model.Add(sum(outgoing[node]) == 1 - at_depth[node][path_length - 1])
        model.Add(sum(incoming[node]) == 1 - at_depth[node][0])

    # The affine-normalized primary objective dominates a deterministic edge-id
    # tie objective.  This cannot change the selected primary integer optimum.
    tie_bound = selected_edge_count * max(len(canonical), 1)
    dominance = tie_bound + 1
    maximum_objective = int(cost_scale) * dominance * max(selected_edge_count, 1)
    if maximum_objective >= 2**60:
        raise ValueError("problem/cost_scale combination risks CP-SAT integer overflow")
    primary = sum(
        penalty * variable for penalty, variable in zip(penalties, edge_variables)
    )
    tie = sum(
        (index + 1) * variable for index, variable in enumerate(edge_variables)
    )
    if reference_improvement_feasibility:
        assert reference_primary_integer is not None
        model.Add(primary <= reference_primary_integer - 1)
    else:
        model.Minimize(primary * dominance + tie)
    if edge_variables:
        model.AddDecisionStrategy(
            edge_variables, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
        )

    solver = cp_model.CpSolver()
    if reference_improvement_feasibility:
        solver.parameters.max_deterministic_time = float(time_limit_seconds)
    else:
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    status = solver.Solve(model)
    response = solver.ResponseProto()
    deterministic_time = float(getattr(response, "deterministic_time", 0.0))
    time_limit_kind = (
        "cp_sat_deterministic_time"
        if reference_improvement_feasibility
        else "wall_time_seconds"
    )
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        label = solver.StatusName(status)
        if reference is not None:
            return _reference_fallback_result(
                reference,
                reason=f"solver_status_{label}",
                reference_cost=float(reference_cost),
                diagnostics={
                    "solver": "ortools_cp_sat",
                    "status": label,
                    "optimal": False,
                    "node_count": node_count,
                    "path_count": path_count,
                    "path_length": path_length,
                    "candidate_count": len(canonical),
                    "reference_objective_cost": reference_cost,
                    "reference_primary_integer_objective": reference_primary_integer,
                    "candidate_primary_integer_objective": None,
                    "best_objective_bound": (
                        None
                        if reference_improvement_feasibility
                        else float(solver.BestObjectiveBound())
                    ),
                    "wall_time_seconds": float(solver.WallTime()),
                    "deterministic_time": deterministic_time,
                    "deterministic_time_limit": (
                        float(time_limit_seconds)
                        if reference_improvement_feasibility
                        else None
                    ),
                    "time_limit_kind": time_limit_kind,
                    "reference_improvement_feasibility": (
                        reference_improvement_feasibility
                    ),
                },
            )
        raise PathCoverInfeasibleError(
            f"CP-SAT found no exact path cover; status={label}"
        )
    if (
        not reference_improvement_feasibility
        and require_optimal
        and status != cp_model.OPTIMAL
    ):
        if reference is not None:
            return _reference_fallback_result(
                reference,
                reason="candidate_not_proven_optimal",
                reference_cost=float(reference_cost),
                diagnostics={
                    "solver": "ortools_cp_sat",
                    "status": solver.StatusName(status),
                    "optimal": False,
                    "node_count": node_count,
                    "path_count": path_count,
                    "path_length": path_length,
                    "candidate_count": len(canonical),
                    "reference_objective_cost": reference_cost,
                    "reference_primary_integer_objective": reference_primary_integer,
                    "candidate_primary_integer_objective": None,
                    "best_objective_bound": float(solver.BestObjectiveBound()),
                    "wall_time_seconds": float(solver.WallTime()),
                    "deterministic_time": deterministic_time,
                    "deterministic_time_limit": None,
                    "time_limit_kind": time_limit_kind,
                    "reference_improvement_feasibility": False,
                },
            )
        raise RuntimeError(
            "CP-SAT reached the time limit with a feasible but unproven path cover"
        )

    try:
        selected = [
            candidate
            for candidate, variable in zip(canonical, edge_variables)
            if solver.Value(variable)
        ]
        selected_by_source = {
            candidate.source: candidate.destination for candidate in selected
        }
        selected_edges = {candidate.edge for candidate in selected}
        node_depths = tuple(int(solver.Value(depth)) for depth in depths)
        starts = sorted(node for node, depth in enumerate(node_depths) if depth == 0)
        paths: list[tuple[int, ...]] = []
        for start in starts:
            path = [start]
            while len(path) < path_length:
                current = path[-1]
                if current not in selected_by_source:
                    raise RuntimeError(
                        "CP-SAT result ended a path before the final depth"
                    )
                path.append(selected_by_source[current])
            paths.append(tuple(path))
        validated = validate_exact_path_cover(
            paths,
            node_count=node_count,
            path_count=path_count,
            path_length=path_length,
            allowed_edges={candidate.edge for candidate in canonical},
        )
        if selected_edges != set(path_cover_edges(validated)):
            raise RuntimeError(
                "selected CP-SAT edges disagree with reconstructed paths"
            )
        if any(
            node_depths[destination] != node_depths[source] + 1
            for source, destination in selected_edges
        ):
            raise RuntimeError(
                "selected CP-SAT edge does not advance exactly one depth"
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if reference is not None:
            return _reference_fallback_result(
                reference,
                reason=f"invalid_candidate_structure:{type(exc).__name__}",
                reference_cost=float(reference_cost),
                diagnostics={
                    "solver": "ortools_cp_sat",
                    "status": solver.StatusName(status),
                    "optimal": status == cp_model.OPTIMAL,
                    "node_count": node_count,
                    "path_count": path_count,
                    "path_length": path_length,
                    "candidate_count": len(canonical),
                    "reference_objective_cost": reference_cost,
                    "reference_primary_integer_objective": reference_primary_integer,
                    "candidate_primary_integer_objective": None,
                    "best_objective_bound": (
                        None
                        if reference_improvement_feasibility
                        else float(solver.BestObjectiveBound())
                    ),
                    "wall_time_seconds": float(solver.WallTime()),
                    "deterministic_time": deterministic_time,
                    "deterministic_time_limit": (
                        float(time_limit_seconds)
                        if reference_improvement_feasibility
                        else None
                    ),
                    "time_limit_kind": time_limit_kind,
                    "reference_improvement_feasibility": (
                        reference_improvement_feasibility
                    ),
                    "candidate_structure_error": str(exc),
                },
            )
        raise

    status_name = solver.StatusName(status)
    raw_cost = float(
        sum(candidate.cost for candidate in selected)
    )
    primary_integer = int(
        sum(
            penalty
            for penalty, variable in zip(penalties, edge_variables)
            if solver.Value(variable)
        )
    )
    tie_integer = int(
        sum(
            index + 1
            for index, variable in enumerate(edge_variables)
            if solver.Value(variable)
        )
    )
    best_objective_bound = (
        None
        if reference_improvement_feasibility
        else float(solver.BestObjectiveBound())
    )
    common_diagnostics = {
        "solver": "ortools_cp_sat",
        "status": status_name,
        "optimal": status == cp_model.OPTIMAL,
        "node_count": node_count,
        "path_count": path_count,
        "path_length": path_length,
        "candidate_count": len(canonical),
        "selected_edge_count": len(selected),
        "candidate_objective_cost": raw_cost,
        "reference_objective_cost": reference_cost,
        "primary_integer_objective": primary_integer,
        "candidate_primary_integer_objective": primary_integer,
        "reference_primary_integer_objective": reference_primary_integer,
        "tie_integer_objective": tie_integer,
        "best_objective_bound": best_objective_bound,
        "cost_offset": cost_offset,
        "cost_span": cost_span,
        "cost_scale": int(cost_scale),
        "depth_counts": [node_depths.count(depth) for depth in range(path_length)],
        "selected_edges": [list(edge) for edge in sorted(selected_edges)],
        "wall_time_seconds": float(solver.WallTime()),
        "deterministic_time": deterministic_time,
        "deterministic_time_limit": (
            float(time_limit_seconds)
            if reference_improvement_feasibility
            else None
        ),
        "time_limit_kind": (
            "cp_sat_deterministic_time"
            if reference_improvement_feasibility
            else "wall_time_seconds"
        ),
        "reference_improvement_feasibility": reference_improvement_feasibility,
        "branches": int(solver.NumBranches()),
        "conflicts": int(solver.NumConflicts()),
        "single_threaded": True,
        "random_seed": int(random_seed),
    }
    if (
        reference_improvement_feasibility
        and not primary_integer < int(reference_primary_integer)
    ):
        return _reference_fallback_result(
            reference,
            reason="integer_improvement_constraint_not_satisfied",
            reference_cost=float(reference_cost),
            diagnostics=common_diagnostics,
        )
    if (
        reference is not None
        and (
            require_strict_reference_improvement
            or reference_improvement_feasibility
        )
        and not raw_cost < float(reference_cost) - 1e-12
    ):
        return _reference_fallback_result(
            reference,
            reason="no_strict_raw_cost_improvement",
            reference_cost=float(reference_cost),
            diagnostics=common_diagnostics,
        )
    common_diagnostics["objective_cost"] = raw_cost
    common_diagnostics["accepted_candidate"] = True
    common_diagnostics["used_reference_fallback"] = False
    return PathCoverResult(
        paths=validated,
        accepted_candidate=True,
        used_reference_fallback=False,
        fallback_reason=None,
        diagnostics=common_diagnostics,
    )


def _reference_fallback_result(
    reference: tuple[tuple[int, ...], ...],
    *,
    reason: str,
    reference_cost: float,
    diagnostics: dict[str, Any],
) -> PathCoverResult:
    values = dict(diagnostics)
    values.update(
        {
            "objective_cost": reference_cost,
            "accepted_candidate": False,
            "used_reference_fallback": True,
            "fallback_reason": reason,
        }
    )
    return PathCoverResult(
        paths=reference,
        accepted_candidate=False,
        used_reference_fallback=True,
        fallback_reason=reason,
        diagnostics=values,
    )


def solve_path_cover(
    cost_matrix: np.ndarray,
    *,
    path_count: int,
    path_length: int,
    outgoing_top_k: int,
    incoming_top_k: int,
    rescue_edges: Iterable[tuple[int, int]] = (),
    rescue_cost: float | None = None,
    reference_paths: Iterable[Sequence[int]] | None = None,
    time_limit_seconds: float = 30.0,
    random_seed: int = 0,
    cost_scale: int = 1_000_000,
    require_optimal: bool = True,
    require_strict_reference_improvement: bool = True,
    reference_improvement_feasibility: bool = False,
) -> PathCoverResult:
    """Extract finite top-k candidates and solve the requested exact cover."""

    costs = np.asarray(cost_matrix)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("cost_matrix must be square")
    node_count = int(costs.shape[0])
    _validate_shape(node_count, path_count, path_length)
    rescue = tuple(rescue_edges)
    candidates = extract_union_directed_candidates(
        costs,
        outgoing_top_k=outgoing_top_k,
        incoming_top_k=incoming_top_k,
        rescue_edges=rescue,
        rescue_cost=rescue_cost,
    )
    result = solve_candidate_path_cover(
        candidates,
        node_count=node_count,
        path_count=path_count,
        path_length=path_length,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        cost_scale=cost_scale,
        require_optimal=require_optimal,
        reference_paths=reference_paths,
        require_strict_reference_improvement=require_strict_reference_improvement,
        reference_improvement_feasibility=reference_improvement_feasibility,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "outgoing_top_k": int(outgoing_top_k),
            "incoming_top_k": int(incoming_top_k),
            "rescue_edge_count": len(set(rescue)),
            "rescue_cost": rescue_cost,
        }
    )
    return PathCoverResult(
        paths=result.paths,
        accepted_candidate=result.accepted_candidate,
        used_reference_fallback=result.used_reference_fallback,
        fallback_reason=result.fallback_reason,
        diagnostics=diagnostics,
    )


__all__ = [
    "DirectedCandidate",
    "PathCoverInfeasibleError",
    "PathCoverResult",
    "exhaustive_path_cover_reference",
    "extract_topk_directed_candidates",
    "extract_union_directed_candidates",
    "path_cover_edges",
    "solve_candidate_path_cover",
    "solve_path_cover",
    "validate_exact_path_cover",
]
