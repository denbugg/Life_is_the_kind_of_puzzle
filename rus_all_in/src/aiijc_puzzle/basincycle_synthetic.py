"""Synthetic-only mechanism probe for the BasinCycle design.

This module deliberately contains no image loader, learned model, organizer
artifact path, or absolute-position score.  It asks one narrow question: can a
target-free directional edge field propose useful *closed* edits of an existing
strict permutation while retaining an explicit identity action?

The implementation is an executable specification, not a competition solver.
It uses exact evidence deltas to isolate cycle generation from the future
learned value head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CycleMove:
    """One atomic closed cycle of raster positions."""

    positions: tuple[int, ...]
    evidence_delta: float


@dataclass(frozen=True)
class RefinementTrace:
    """All legal prefixes produced by deterministic cycle refinement."""

    layouts: tuple[np.ndarray, ...]
    moves: tuple[CycleMove, ...]
    energies: tuple[float, ...]

    @property
    def output(self) -> np.ndarray:
        return self.layouts[-1]

    @property
    def chose_keep(self) -> bool:
        return not self.moves


@dataclass(frozen=True)
class SyntheticCase:
    """A planted directional-edge instance and a legally corrupted layout."""

    seed: int
    truth: np.ndarray
    control: np.ndarray
    right_scores: np.ndarray
    down_scores: np.ndarray
    corruption_cycles: tuple[tuple[int, ...], ...]


def is_strict_permutation(layout: np.ndarray) -> bool:
    """Return whether ``layout`` contains every integer in ``[0, N)`` once."""

    values = np.asarray(layout)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        return False
    return np.array_equal(np.sort(values), np.arange(values.size))


def _validate_grid(layout: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.asarray(layout)
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2")
    if values.shape != (grid_size * grid_size,):
        raise ValueError("layout length does not match grid_size")
    if not is_strict_permutation(values):
        raise ValueError("layout must be a strict permutation")
    return values.astype(np.int64, copy=False)


def _validate_scores(
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    tile_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    right = np.asarray(right_scores, dtype=np.float64)
    down = np.asarray(down_scores, dtype=np.float64)
    expected = (tile_count, tile_count)
    if right.shape != expected or down.shape != expected:
        raise ValueError(f"directional scores must both have shape {expected}")
    if not np.all(np.isfinite(right)) or not np.all(np.isfinite(down)):
        raise ValueError("directional scores must be finite")
    return right, down


def apply_cycle(layout: np.ndarray, positions: tuple[int, ...] | list[int]) -> np.ndarray:
    """Apply one closed cycle, preserving every tile identity exactly once.

    For ``(p0, ..., pL)`` the tile at each next position moves into the
    preceding position.  A repeated position would not be a permutation action
    under this notation and therefore fails closed.
    """

    values = np.asarray(layout)
    if not is_strict_permutation(values):
        raise ValueError("layout must be a strict permutation")
    cycle = tuple(int(position) for position in positions)
    if len(cycle) < 2:
        raise ValueError("a cycle must contain at least two positions")
    if len(cycle) != len(set(cycle)):
        raise ValueError("cycle positions must be distinct")
    if min(cycle) < 0 or max(cycle) >= values.size:
        raise ValueError("cycle position is out of range")

    output = values.copy()
    destination = np.asarray(cycle, dtype=np.int64)
    source = np.roll(destination, -1)
    output[destination] = values[source]
    if not is_strict_permutation(output):  # Defensive executable invariant.
        raise AssertionError("closed cycle violated the strict-permutation invariant")
    return output


def canonical_cycle(positions: tuple[int, ...]) -> tuple[int, ...]:
    """Canonicalise cyclic rotations without merging opposite orientations."""

    if len(positions) < 2 or len(set(positions)) != len(positions):
        raise ValueError("canonical cycle requires at least two distinct positions")
    rotations = tuple(positions[offset:] + positions[:offset] for offset in range(len(positions)))
    return min(rotations)


def evidence_energy(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
) -> float:
    """Sum the two directional scores realised by a board layout."""

    values = _validate_grid(layout, grid_size)
    right, down = _validate_scores(right_scores, down_scores, tile_count=values.size)
    board = values.reshape(grid_size, grid_size)
    horizontal = right[board[:, :-1], board[:, 1:]].sum()
    vertical = down[board[:-1, :], board[1:, :]].sum()
    return float(horizontal + vertical)


def incident_evidence(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
) -> np.ndarray:
    """Mean realised-contact evidence at each slot, used only to seed search."""

    values = _validate_grid(layout, grid_size)
    right, down = _validate_scores(right_scores, down_scores, tile_count=values.size)
    board = values.reshape(grid_size, grid_size)
    totals = np.zeros((grid_size, grid_size), dtype=np.float64)
    degree = np.zeros((grid_size, grid_size), dtype=np.float64)

    horizontal = right[board[:, :-1], board[:, 1:]]
    totals[:, :-1] += horizontal
    totals[:, 1:] += horizontal
    degree[:, :-1] += 1.0
    degree[:, 1:] += 1.0

    vertical = down[board[:-1, :], board[1:, :]]
    totals[:-1, :] += vertical
    totals[1:, :] += vertical
    degree[:-1, :] += 1.0
    degree[1:, :] += 1.0
    return (totals / degree).reshape(-1)


def candidate_positions(
    layout: np.ndarray,
    position: int,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
    top_k: int,
    candidate_cap: int,
    protected_positions: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Map evidence-suggested tiles for one slot back to their current slots.

    Candidate ranking is based only on visible neighbours and directional edge
    scores.  Stable ties use current raster position rather than tile identity,
    retaining equivariance to arbitrary tile relabeling.
    """

    values = _validate_grid(layout, grid_size)
    right, down = _validate_scores(right_scores, down_scores, tile_count=values.size)
    valid_position = (
        not isinstance(position, bool)
        and isinstance(position, int)
        and 0 <= position < values.size
    )
    if not valid_position:
        raise ValueError("position is out of range")
    if top_k <= 0 or candidate_cap <= 0:
        raise ValueError("top_k and candidate_cap must be positive")

    row, col = divmod(position, grid_size)
    board = values.reshape(grid_size, grid_size)
    tile_scores = np.zeros(values.size, dtype=np.float64)
    if col > 0:
        tile_scores += right[board[row, col - 1], :]
    if col + 1 < grid_size:
        tile_scores += right[:, board[row, col + 1]]
    if row > 0:
        tile_scores += down[board[row - 1, col], :]
    if row + 1 < grid_size:
        tile_scores += down[:, board[row + 1, col]]

    # Score candidate *positions* through their current tile.  This avoids a
    # tile-ID tie break and makes relabeling a true symmetry of the mechanism.
    positions = np.arange(values.size, dtype=np.int64)
    position_scores = tile_scores[values]
    order = np.lexsort((positions, -position_scores))
    available = [
        int(candidate)
        for candidate in order[: min(top_k, values.size)]
        if candidate != position and candidate not in protected_positions
    ]
    return tuple(available[:candidate_cap])


def propose_cycles(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
    top_k: int,
    candidate_cap: int,
    seed_count: int,
    max_cycle_length: int,
    protected_positions: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], ...]:
    """Build a deterministic dynamic closure of short legal cycles."""

    values = _validate_grid(layout, grid_size)
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if not 2 <= max_cycle_length <= 8:
        raise ValueError("max_cycle_length must be in [2, 8]")
    if any(position < 0 or position >= values.size for position in protected_positions):
        raise ValueError("protected position is out of range")

    conflicts = incident_evidence(
        values,
        right_scores,
        down_scores,
        grid_size=grid_size,
    )
    raster = np.arange(values.size, dtype=np.int64)
    seed_order = np.lexsort((raster, conflicts))
    seeds = [
        int(position)
        for position in seed_order
        if position not in protected_positions
    ][:seed_count]

    proposals: set[tuple[int, ...]] = set()

    def expand(prefix: tuple[int, ...]) -> None:
        if len(prefix) >= 2:
            proposals.add(canonical_cycle(prefix))
        if len(prefix) == max_cycle_length:
            return
        next_positions = candidate_positions(
            values,
            prefix[-1],
            right_scores,
            down_scores,
            grid_size=grid_size,
            top_k=top_k,
            candidate_cap=candidate_cap,
            protected_positions=protected_positions,
        )
        for candidate in next_positions:
            if candidate not in prefix:
                expand(prefix + (candidate,))

    for seed in seeds:
        expand((seed,))
    return tuple(sorted(proposals, key=lambda cycle: (len(cycle), cycle)))


def select_cycle(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
    proposals: tuple[tuple[int, ...], ...],
    minimum_delta: float,
) -> CycleMove | None:
    """Return the best strictly improving cycle, or the explicit KEEP action."""

    baseline = evidence_energy(
        layout,
        right_scores,
        down_scores,
        grid_size=grid_size,
    )
    scored: list[CycleMove] = []
    for positions in proposals:
        candidate = apply_cycle(layout, positions)
        delta = evidence_energy(
            candidate,
            right_scores,
            down_scores,
            grid_size=grid_size,
        ) - baseline
        if delta > minimum_delta:
            scored.append(CycleMove(positions=positions, evidence_delta=float(delta)))
    if not scored:
        return None
    return min(
        scored,
        key=lambda move: (-move.evidence_delta, len(move.positions), move.positions),
    )


def refine_layout(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
    top_k: int,
    candidate_cap: int,
    seed_count: int,
    max_cycle_length: int,
    max_steps: int,
    minimum_delta: float = 1e-9,
    protected_positions: frozenset[int] = frozenset(),
) -> RefinementTrace:
    """Run deterministic strict-cycle ascent with an absorbing KEEP state."""

    current = _validate_grid(layout, grid_size).copy()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    layouts = [current.copy()]
    energies = [
        evidence_energy(current, right_scores, down_scores, grid_size=grid_size)
    ]
    moves: list[CycleMove] = []

    for _ in range(max_steps):
        proposals = propose_cycles(
            current,
            right_scores,
            down_scores,
            grid_size=grid_size,
            top_k=top_k,
            candidate_cap=candidate_cap,
            seed_count=seed_count,
            max_cycle_length=max_cycle_length,
            protected_positions=protected_positions,
        )
        move = select_cycle(
            current,
            right_scores,
            down_scores,
            grid_size=grid_size,
            proposals=proposals,
            minimum_delta=minimum_delta,
        )
        if move is None:
            break
        current = apply_cycle(current, move.positions)
        energy = evidence_energy(current, right_scores, down_scores, grid_size=grid_size)
        if energy <= energies[-1] + minimum_delta:
            raise AssertionError("accepted BasinCycle action did not improve evidence")
        if any(current[position] != layouts[0][position] for position in protected_positions):
            raise AssertionError("a protected slot changed")
        moves.append(move)
        layouts.append(current.copy())
        energies.append(energy)

    return RefinementTrace(
        layouts=tuple(layouts),
        moves=tuple(moves),
        energies=tuple(energies),
    )


def exact_count(layout: np.ndarray, truth: np.ndarray) -> int:
    """Count tiles at their planted synthetic positions."""

    values = np.asarray(layout)
    reference = np.asarray(truth)
    if values.shape != reference.shape:
        raise ValueError("layout and truth shapes differ")
    return int(np.count_nonzero(values == reference))


def true_pair_count(layout: np.ndarray, truth: np.ndarray, *, grid_size: int) -> int:
    """Count realised right/down pairs belonging to the planted truth board."""

    values = _validate_grid(layout, grid_size)
    reference = _validate_grid(truth, grid_size)
    truth_board = reference.reshape(grid_size, grid_size)
    truth_right = set(zip(truth_board[:, :-1].flat, truth_board[:, 1:].flat, strict=True))
    truth_down = set(zip(truth_board[:-1, :].flat, truth_board[1:, :].flat, strict=True))
    board = values.reshape(grid_size, grid_size)
    observed_right = zip(board[:, :-1].flat, board[:, 1:].flat, strict=True)
    observed_down = zip(board[:-1, :].flat, board[1:, :].flat, strict=True)
    return sum(pair in truth_right for pair in observed_right) + sum(
        pair in truth_down for pair in observed_down
    )


def relabel_instance(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    relabel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an arbitrary old-ID -> new-ID bijection to a complete instance."""

    values = np.asarray(layout)
    mapping = np.asarray(relabel)
    if not is_strict_permutation(values) or not is_strict_permutation(mapping):
        raise ValueError("layout and relabel mapping must be strict permutations")
    if values.shape != mapping.shape:
        raise ValueError("relabel mapping size differs from layout")
    right, down = _validate_scores(right_scores, down_scores, tile_count=values.size)
    relabeled_right = np.empty_like(right)
    relabeled_down = np.empty_like(down)
    relabeled_right[np.ix_(mapping, mapping)] = right
    relabeled_down[np.ix_(mapping, mapping)] = down
    return mapping[values], relabeled_right, relabeled_down


def transpose_instance(
    layout: np.ndarray,
    right_scores: np.ndarray,
    down_scores: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transpose board geometry and consistently exchange right/down evidence."""

    values = _validate_grid(layout, grid_size)
    right, down = _validate_scores(right_scores, down_scores, tile_count=values.size)
    return values.reshape(grid_size, grid_size).T.reshape(-1), down.copy(), right.copy()


def make_synthetic_case(
    *,
    grid_size: int,
    seed: int,
    true_edge_score: float,
    false_edge_sigma: float,
    true_edge_noise_sigma: float,
    distractor_probability: float,
    distractor_boost: float,
    corruption_cycle_count: int,
    corruption_cycle_length: int,
) -> SyntheticCase:
    """Create one fixed planted-edge case without reading any external data."""

    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if false_edge_sigma < 0 or true_edge_noise_sigma < 0:
        raise ValueError("noise scales must be nonnegative")
    if not 0 <= distractor_probability <= 1:
        raise ValueError("distractor_probability must be in [0, 1]")
    if corruption_cycle_count <= 0 or corruption_cycle_length < 2:
        raise ValueError("corruption cycle parameters must be positive")

    tile_count = grid_size * grid_size
    rng = np.random.default_rng(seed)
    truth = np.arange(tile_count, dtype=np.int64)
    right = rng.normal(0.0, false_edge_sigma, size=(tile_count, tile_count))
    down = rng.normal(0.0, false_edge_sigma, size=(tile_count, tile_count))
    distractor_right = rng.random((tile_count, tile_count)) < distractor_probability
    distractor_down = rng.random((tile_count, tile_count)) < distractor_probability
    right += distractor_right * distractor_boost
    down += distractor_down * distractor_boost
    np.fill_diagonal(right, -abs(true_edge_score))
    np.fill_diagonal(down, -abs(true_edge_score))

    board = truth.reshape(grid_size, grid_size)
    right_truth_left = board[:, :-1].reshape(-1)
    right_truth_right = board[:, 1:].reshape(-1)
    down_truth_top = board[:-1, :].reshape(-1)
    down_truth_bottom = board[1:, :].reshape(-1)
    right[right_truth_left, right_truth_right] = true_edge_score + rng.normal(
        0.0,
        true_edge_noise_sigma,
        size=right_truth_left.size,
    )
    down[down_truth_top, down_truth_bottom] = true_edge_score + rng.normal(
        0.0,
        true_edge_noise_sigma,
        size=down_truth_top.size,
    )

    # Same-parity cells are never side-adjacent.  This isolates whether the
    # action closure can invert multiple legal cycles, rather than testing a
    # pathological overlap generator in the mechanism-only gate.
    parity = seed % 2
    independent_positions = np.array(
        [
            row * grid_size + col
            for row in range(grid_size)
            for col in range(grid_size)
            if (row + col) % 2 == parity
        ],
        dtype=np.int64,
    )
    required = corruption_cycle_count * corruption_cycle_length
    if required > independent_positions.size:
        raise ValueError("requested disjoint corruption cycles exceed parity-cell capacity")
    rng.shuffle(independent_positions)
    selected = independent_positions[:required]
    cycles = tuple(
        tuple(
            int(position)
            for position in selected[
                offset : offset + corruption_cycle_length
            ]
        )
        for offset in range(0, required, corruption_cycle_length)
    )
    control = truth.copy()
    for cycle in cycles:
        control = apply_cycle(control, cycle)
    return SyntheticCase(
        seed=seed,
        truth=truth,
        control=control,
        right_scores=right,
        down_scores=down,
        corruption_cycles=cycles,
    )


def evaluate_synthetic_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one already-frozen synthetic mechanism configuration."""

    panel = config["panel"]
    search = config["search"]
    gates = config["gates"]
    case_rows: list[dict[str, Any]] = []
    all_prefixes_legal = True
    truth_keep_count = 0

    for case_index in range(int(panel["case_count"])):
        seed = int(panel["seed_start"]) + case_index
        case = make_synthetic_case(
            grid_size=int(panel["grid_size"]),
            seed=seed,
            true_edge_score=float(panel["true_edge_score"]),
            false_edge_sigma=float(panel["false_edge_sigma"]),
            true_edge_noise_sigma=float(panel["true_edge_noise_sigma"]),
            distractor_probability=float(panel["distractor_probability"]),
            distractor_boost=float(panel["distractor_boost"]),
            corruption_cycle_count=int(panel["corruption_cycle_count"]),
            corruption_cycle_length=int(panel["corruption_cycle_length"]),
        )
        keyword = {
            "grid_size": int(panel["grid_size"]),
            "top_k": int(search["top_k"]),
            "candidate_cap": int(search["candidate_cap"]),
            "seed_count": int(search["seed_count"]),
            "max_cycle_length": int(search["max_cycle_length"]),
            "max_steps": int(search["max_steps"]),
            "minimum_delta": float(search["minimum_delta"]),
        }
        control_trace = refine_layout(
            case.control,
            case.right_scores,
            case.down_scores,
            **keyword,
        )
        truth_trace = refine_layout(
            case.truth,
            case.right_scores,
            case.down_scores,
            **keyword,
        )
        legal = all(is_strict_permutation(prefix) for prefix in control_trace.layouts)
        legal = legal and all(is_strict_permutation(prefix) for prefix in truth_trace.layouts)
        all_prefixes_legal = all_prefixes_legal and legal
        truth_keep_count += int(truth_trace.chose_keep)

        initial_pairs = true_pair_count(
            case.control,
            case.truth,
            grid_size=int(panel["grid_size"]),
        )
        final_pairs = true_pair_count(
            control_trace.output,
            case.truth,
            grid_size=int(panel["grid_size"]),
        )
        initial_exact = exact_count(case.control, case.truth)
        final_exact = exact_count(control_trace.output, case.truth)
        case_rows.append(
            {
                "case_index": case_index,
                "seed": seed,
                "legal": legal,
                "move_count": len(control_trace.moves),
                "truth_move_count": len(truth_trace.moves),
                "pair_before": initial_pairs,
                "pair_after": final_pairs,
                "pair_delta": final_pairs - initial_pairs,
                "exact_before": initial_exact,
                "exact_after": final_exact,
                "exact_delta": final_exact - initial_exact,
                "evidence_delta": control_trace.energies[-1] - control_trace.energies[0],
            }
        )

    pair_deltas = np.asarray([row["pair_delta"] for row in case_rows], dtype=np.float64)
    exact_deltas = np.asarray([row["exact_delta"] for row in case_rows], dtype=np.float64)
    truth_keep_rate = truth_keep_count / len(case_rows)
    negative_pair_rate = float(np.mean(pair_deltas < 0))
    summary = {
        "all_prefixes_strict": bool(all_prefixes_legal),
        "truth_keep_rate": truth_keep_rate,
        "mean_pair_delta": float(pair_deltas.mean()),
        "median_pair_delta": float(np.median(pair_deltas)),
        "min_pair_delta": float(pair_deltas.min()),
        "mean_exact_delta": float(exact_deltas.mean()),
        "median_exact_delta": float(np.median(exact_deltas)),
        "min_exact_delta": float(exact_deltas.min()),
        "negative_pair_rate": negative_pair_rate,
    }
    checks = {
        "strict_permutation": summary["all_prefixes_strict"],
        "truth_keep_rate": truth_keep_rate >= float(gates["minimum_truth_keep_rate"]),
        "mean_pair_delta": summary["mean_pair_delta"] >= float(gates["minimum_mean_pair_delta"]),
        "mean_exact_delta": summary["mean_exact_delta"] >= float(gates["minimum_mean_exact_delta"]),
        "negative_pair_rate": negative_pair_rate <= float(gates["maximum_negative_pair_rate"]),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "synthetic-mechanism-only",
        "summary": summary,
        "checks": checks,
        "cases": case_rows,
    }
