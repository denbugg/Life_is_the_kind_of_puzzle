"""Directional quadratic-assignment relaxation for the 24x24 puzzle.

The relaxed assignment matrix is tile x position: ``X[tile, position]``.
For horizontal position edges ``p -> q`` and vertical edges ``p -> q`` the
quadratic cost is respectively::

    sum(right[a, b] * X[a, p] * X[b, q])
    sum(down[a, b] * X[a, p] * X[b, q])

Thus the two stored directional matrices cover all four physical sides.  A
tile contributes through a row of ``right``/``down`` at its right/bottom side
and through a column at its left/top side.

This is a FAQ-style non-convex Frank-Wolfe heuristic, not an exact QAP solver.
The Birkhoff iterate is represented as a uniform component plus a sparse
convex combination of permutation matrices.  That representation keeps the
gradient at O(k*N^2 + N*E), where k is the number of active permutations and
E is the number of grid edges, instead of performing a dense O(N^3) matrix
product at every iteration.  The Hungarian linear oracle remains cubic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .compatibility import CompatibilityMatrices
from .geometry import GRID, TILE_COUNT, validate_permutation


@dataclass(frozen=True)
class DirectionalQAPResult:
    """Best integral layout and diagnostics from ``directional_qap``."""

    position_to_slot: np.ndarray
    objective: float
    relaxed_objective: float
    restart: int
    iterations: int
    converged: bool
    history: tuple[float, ...]


@dataclass
class _BirkhoffMixture:
    """Uniform matrix plus a weighted list of tile-at-position layouts."""

    uniform_weight: float
    layouts: list[np.ndarray]
    weights: list[float]

    def normalize(self, *, prune_tolerance: float = 1e-13) -> None:
        merged_layouts: list[np.ndarray] = []
        merged_weights: list[float] = []
        keys: dict[bytes, int] = {}
        for layout, weight in zip(self.layouts, self.weights):
            if weight <= prune_tolerance:
                continue
            key = np.asarray(layout, dtype=np.int32).tobytes()
            previous = keys.get(key)
            if previous is None:
                keys[key] = len(merged_layouts)
                merged_layouts.append(np.asarray(layout, dtype=np.int32).copy())
                merged_weights.append(float(weight))
            else:
                merged_weights[previous] += float(weight)
        self.layouts = merged_layouts
        self.weights = merged_weights
        self.uniform_weight = max(0.0, float(self.uniform_weight))
        total = self.uniform_weight + float(sum(self.weights))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("invalid Birkhoff mixture weights")
        self.uniform_weight /= total
        self.weights = [weight / total for weight in self.weights]

    def step_towards(self, layout: np.ndarray, alpha: float) -> None:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Frank-Wolfe step must lie in [0, 1]")
        keep = 1.0 - alpha
        self.uniform_weight *= keep
        self.weights = [keep * weight for weight in self.weights]
        if alpha > 0.0:
            self.layouts.append(np.asarray(layout, dtype=np.int32).copy())
            self.weights.append(alpha)
        self.normalize()

    def permutation_part(self, size: int) -> csr_matrix:
        if not self.layouts:
            return csr_matrix((size, size), dtype=np.float64)
        positions = np.arange(size, dtype=np.int32)
        rows = np.concatenate(self.layouts)
        columns = np.tile(positions, len(self.layouts))
        data = np.concatenate(
            [np.full(size, weight, dtype=np.float64) for weight in self.weights]
        )
        matrix = csr_matrix((data, (rows, columns)), shape=(size, size))
        matrix.sum_duplicates()
        return matrix


def _grid_adjacencies(grid: int) -> tuple[csr_matrix, csr_matrix]:
    size = grid * grid
    positions = np.arange(size, dtype=np.int32).reshape(grid, grid)
    horizontal_sources = positions[:, :-1].ravel()
    horizontal_targets = positions[:, 1:].ravel()
    vertical_sources = positions[:-1, :].ravel()
    vertical_targets = positions[1:, :].ravel()
    horizontal = csr_matrix(
        (
            np.ones(len(horizontal_sources), dtype=np.float64),
            (horizontal_sources, horizontal_targets),
        ),
        shape=(size, size),
    )
    vertical = csr_matrix(
        (
            np.ones(len(vertical_sources), dtype=np.float64),
            (vertical_sources, vertical_targets),
        ),
        shape=(size, size),
    )
    return horizontal, vertical


def _finite_directional_cost(
    values: np.ndarray,
    *,
    name: str,
    diagonal_cost: float | None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (TILE_COUNT, TILE_COUNT):
        raise ValueError(
            f"{name} must have shape {(TILE_COUNT, TILE_COUNT)}, got {values.shape}"
        )
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    if not np.all(np.isfinite(values[off_diagonal])):
        raise ValueError(f"{name} has non-finite off-diagonal costs")
    if diagonal_cost is None:
        # The source matrices normally mask self matches with +inf.  A large
        # finite fill avoids making fractional copies of one tile artificially
        # attractive while leaving every integral permutation unchanged.
        resolved_diagonal = float(np.max(values[off_diagonal]))
    else:
        resolved_diagonal = float(diagonal_cost)
    if not np.isfinite(resolved_diagonal):
        raise ValueError("diagonal_cost must be finite or None")
    values = values.copy()
    np.fill_diagonal(values, resolved_diagonal)
    return values


def _layout_quadratic(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    horizontal: csr_matrix,
    vertical: csr_matrix,
) -> float:
    horizontal_sources, horizontal_targets = horizontal.nonzero()
    vertical_sources, vertical_targets = vertical.nonzero()
    value = right[
        layout[horizontal_sources], layout[horizontal_targets]
    ].sum(dtype=np.float64)
    value += down[
        layout[vertical_sources], layout[vertical_targets]
    ].sum(dtype=np.float64)
    return float(value)


def _uniform_gradient(
    right: np.ndarray,
    down: np.ndarray,
    horizontal: csr_matrix,
    vertical: csr_matrix,
) -> np.ndarray:
    """Gradient at the uniform doubly-stochastic matrix in O(N^2)."""
    size = right.shape[0]
    horizontal_out = np.asarray(horizontal.sum(axis=1)).ravel()
    horizontal_in = np.asarray(horizontal.sum(axis=0)).ravel()
    vertical_out = np.asarray(vertical.sum(axis=1)).ravel()
    vertical_in = np.asarray(vertical.sum(axis=0)).ravel()
    return (
        right.mean(axis=1)[:, None] * horizontal_out[None, :]
        + right.mean(axis=0)[:, None] * horizontal_in[None, :]
        + down.mean(axis=1)[:, None] * vertical_out[None, :]
        + down.mean(axis=0)[:, None] * vertical_in[None, :]
    ).reshape(size, size)


def _relaxed_gradient(
    state: _BirkhoffMixture,
    right: np.ndarray,
    down: np.ndarray,
    horizontal: csr_matrix,
    vertical: csr_matrix,
    uniform_gradient: np.ndarray,
) -> tuple[np.ndarray, csr_matrix]:
    """Return the exact quadratic gradient for the sparse mixture."""
    size = right.shape[0]
    assignment = state.permutation_part(size)
    gradient = state.uniform_weight * uniform_gradient
    if assignment.nnz == 0:
        return gradient.copy(), assignment

    # For Q_C,A(X) = tr(C.T @ X @ A @ X.T),
    # grad Q = C @ X @ A.T + C.T @ X @ A.
    # Sparse @ dense and sparse-grid @ dense products avoid dense C @ X.
    assignment_transpose = assignment.transpose().tocsr()
    right_x = np.asarray(assignment_transpose @ right.T).T
    right_transpose_x = np.asarray(assignment_transpose @ right).T
    down_x = np.asarray(assignment_transpose @ down.T).T
    down_transpose_x = np.asarray(assignment_transpose @ down).T
    gradient = gradient + np.asarray(horizontal @ right_x.T).T
    gradient += np.asarray(horizontal.T @ right_transpose_x.T).T
    gradient += np.asarray(vertical @ down_x.T).T
    gradient += np.asarray(vertical.T @ down_transpose_x.T).T
    return gradient, assignment


def _state_inner(
    state: _BirkhoffMixture,
    assignment: csr_matrix,
    values: np.ndarray,
) -> float:
    size = values.shape[0]
    result = state.uniform_weight * float(values.sum(dtype=np.float64)) / float(size)
    if assignment.nnz:
        result += float(assignment.multiply(values).sum())
    return result


def _layout_inner(layout: np.ndarray, values: np.ndarray) -> float:
    positions = np.arange(len(layout), dtype=np.int32)
    return float(values[layout, positions].sum(dtype=np.float64))


def _linear_oracle(cost: np.ndarray) -> np.ndarray:
    """Minimize <cost, P> and return P as position_to_slot."""
    tile_indices, position_indices = linear_sum_assignment(cost)
    layout = np.empty(cost.shape[0], dtype=np.int32)
    layout[position_indices] = tile_indices.astype(np.int32, copy=False)
    return layout


def _project_assignment(state: _BirkhoffMixture, size: int) -> np.ndarray:
    """Nearest permutation in Frobenius norm (maximum overlap)."""
    assignment = state.permutation_part(size).toarray()
    if state.uniform_weight:
        assignment += state.uniform_weight / float(size)
    tile_indices, position_indices = linear_sum_assignment(-assignment)
    layout = np.empty(size, dtype=np.int32)
    layout[position_indices] = tile_indices.astype(np.int32, copy=False)
    return layout


def _sinkhorn(logits: np.ndarray, iterations: int) -> np.ndarray:
    """Stable square log-domain Sinkhorn normalization."""
    if iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    log_assignment = np.asarray(logits, dtype=np.float64).copy()
    if log_assignment.ndim != 2 or log_assignment.shape[0] != log_assignment.shape[1]:
        raise ValueError("Sinkhorn logits must be square")
    if not np.all(np.isfinite(log_assignment)):
        raise ValueError("Sinkhorn logits must be finite")
    for _ in range(iterations):
        log_assignment -= logsumexp(log_assignment, axis=1, keepdims=True)
        log_assignment -= logsumexp(log_assignment, axis=0, keepdims=True)
    assignment = np.exp(log_assignment)
    # One cheap real-domain polish makes the projected starts reproducible even
    # when log-domain convergence was deliberately given a small iteration cap.
    assignment /= np.maximum(assignment.sum(axis=1, keepdims=True), 1e-300)
    assignment /= np.maximum(assignment.sum(axis=0, keepdims=True), 1e-300)
    return assignment


def _noisy_start_layout(
    rng: np.random.Generator,
    *,
    initial: np.ndarray | None,
    noise_scale: float,
    initial_logit_bias: float,
    sinkhorn_iterations: int,
) -> np.ndarray:
    logits = rng.normal(0.0, noise_scale, size=(TILE_COUNT, TILE_COUNT))
    if initial is not None and initial_logit_bias:
        positions = np.arange(TILE_COUNT, dtype=np.int32)
        logits[initial, positions] += initial_logit_bias
    assignment = _sinkhorn(logits, sinkhorn_iterations)
    tile_indices, position_indices = linear_sum_assignment(-assignment)
    layout = np.empty(TILE_COUNT, dtype=np.int32)
    layout[position_indices] = tile_indices.astype(np.int32, copy=False)
    return layout


def _make_start(
    restart: int,
    rng: np.random.Generator,
    *,
    initial: np.ndarray | None,
    initial_weight: float,
    noisy_components: int,
    noise_scale: float,
    initial_logit_bias: float,
    sinkhorn_iterations: int,
) -> _BirkhoffMixture:
    if restart == 0:
        if initial is None:
            return _BirkhoffMixture(1.0, [], [])
        state = _BirkhoffMixture(
            1.0 - initial_weight,
            [initial.copy()] if initial_weight else [],
            [initial_weight] if initial_weight else [],
        )
        state.normalize()
        return state

    layouts = [
        _noisy_start_layout(
            rng,
            initial=initial,
            noise_scale=noise_scale,
            initial_logit_bias=initial_logit_bias,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        for _ in range(noisy_components)
    ]
    if initial is None:
        state = _BirkhoffMixture(
            0.0,
            layouts,
            [1.0 / float(noisy_components)] * noisy_components,
        )
    else:
        residual = 1.0 - initial_weight
        state = _BirkhoffMixture(
            0.0,
            [initial.copy(), *layouts] if initial_weight else layouts,
            (
                [initial_weight]
                + [residual / float(noisy_components)] * noisy_components
                if initial_weight
                else [1.0 / float(noisy_components)] * noisy_components
            ),
        )
    state.normalize()
    return state


def _bounded_quadratic_step(
    linear: float,
    quadratic: float,
    *,
    tolerance: float,
) -> tuple[float, float]:
    """Tolerance-exact minimization of a bounded scalar quadratic."""
    candidates = [0.0, 1.0]
    scale = max(1.0, abs(linear), abs(quadratic))
    if quadratic > tolerance * scale:
        stationary = -linear / (2.0 * quadratic)
        if 0.0 < stationary < 1.0 and np.isfinite(stationary):
            candidates.append(float(stationary))
    best_alpha = 0.0
    best_delta = 0.0
    for alpha in candidates[1:]:
        delta = float(alpha * linear + alpha * alpha * quadratic)
        # Prefer no move under numerical ties; otherwise prefer the smaller
        # step, which preserves more of the current relaxation.
        if delta < best_delta - tolerance * scale or (
            delta < best_delta
            and abs(delta - best_delta) <= tolerance * scale
            and alpha < best_alpha
        ):
            best_alpha = alpha
            best_delta = delta
    return best_alpha, best_delta


def _integral_objective(
    layout: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    horizontal: csr_matrix,
    vertical: csr_matrix,
    unary_tile_position: np.ndarray,
    boundary_weight: float,
) -> float:
    return _layout_quadratic(layout, right, down, horizontal, vertical) + float(
        boundary_weight * _layout_inner(layout, unary_tile_position)
    )


def directional_qap(
    compatibility: CompatibilityMatrices,
    *,
    initial: np.ndarray | None = None,
    iterations: int = 20,
    restarts: int = 1,
    seed: int = 0,
    boundary_weight: float = 0.0,
    diagonal_cost: float | None = None,
    initial_weight: float = 0.75,
    noisy_components: int = 3,
    noise_scale: float = 1.0,
    initial_logit_bias: float = 2.0,
    sinkhorn_iterations: int = 20,
    tolerance: float = 1e-9,
    refine_swaps: int = 8,
    refine_weak_cells: int = 32,
) -> DirectionalQAPResult:
    """Solve the directional 24x24 QAP with a bounded FAQ heuristic.

    Parameters
    ----------
    compatibility:
        Directed right and down tile-to-tile seam costs.  Their masked
        diagonals are replaced by ``diagonal_cost`` only for the continuous
        relaxation; diagonals never contribute to an integral permutation.
        ``None`` uses the largest finite off-diagonal cost, discouraging a
        fractional tile from cheaply occupying adjacent positions.
    initial:
        Optional ``position_to_slot`` permutation.  The first start blends it
        with the Birkhoff barycenter according to ``initial_weight``.
    restarts:
        Start zero is deterministic (barycenter or the initial blend).
        Additional starts are sparse mixtures of Sinkhorn-normalized,
        noisy-Hungarian permutations.  Fixed ``seed`` makes them reproducible.
    refine_swaps:
        Number of deterministic ``swap_refine`` moves after projection.  Set
        to zero to return the unrefined projected FAQ candidate.

    Notes
    -----
    The objective is non-convex, so neither the relaxed nor projected answer
    is globally optimal.  The sparse-mixture implementation deliberately
    approximates a dense noisy Sinkhorn start by a small convex combination of
    projected permutations; this keeps 576-tile iterations practical.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    if boundary_weight < 0.0:
        raise ValueError("boundary_weight must be non-negative")
    if not 0.0 <= initial_weight <= 1.0:
        raise ValueError("initial_weight must lie in [0, 1]")
    if noisy_components <= 0:
        raise ValueError("noisy_components must be positive")
    if noise_scale < 0.0 or not np.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite and non-negative")
    if not np.isfinite(initial_logit_bias):
        raise ValueError("initial_logit_bias must be finite")
    if sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if refine_swaps < 0 or (refine_swaps > 0 and refine_weak_cells < 2):
        raise ValueError("invalid swap-refinement settings")

    initial_layout = None
    if initial is not None:
        initial_layout = validate_permutation(initial, name="initial_position_to_slot").copy()

    # Keep these imports local so this standalone solver may safely be
    # re-exported by ``solvers.py`` without creating an import cycle.
    from .solvers import placement_unary, swap_refine

    right = _finite_directional_cost(
        compatibility.right, name=f"{compatibility.name}.right", diagonal_cost=diagonal_cost
    )
    down = _finite_directional_cost(
        compatibility.down, name=f"{compatibility.name}.down", diagonal_cost=diagonal_cost
    )
    horizontal, vertical = _grid_adjacencies(GRID)
    # Boundary evidence should ignore self matches.  Rebuild a clean view so
    # unusual NaN/-inf input diagonals cannot poison a zero-weight unary term;
    # ordinary CompatibilityMatrices use +inf here already.
    boundary_right = np.asarray(compatibility.right, dtype=np.float32).copy()
    boundary_down = np.asarray(compatibility.down, dtype=np.float32).copy()
    np.fill_diagonal(boundary_right, np.inf)
    np.fill_diagonal(boundary_down, np.inf)
    boundary_compatibility = CompatibilityMatrices(
        f"{compatibility.name}_qap_boundary", boundary_right, boundary_down
    )
    unary_tile_position = placement_unary(boundary_compatibility).T.astype(
        np.float64, copy=False
    )
    uniform_gradient = _uniform_gradient(right, down, horizontal, vertical)
    rng = np.random.default_rng(seed)

    best_layout: np.ndarray | None = None
    best_objective = np.inf
    best_relaxed_objective = np.inf
    best_restart = -1
    best_iterations = 0
    best_converged = False
    best_history: tuple[float, ...] = ()

    for restart in range(restarts):
        state = _make_start(
            restart,
            rng,
            initial=initial_layout,
            initial_weight=initial_weight,
            noisy_components=noisy_components,
            noise_scale=noise_scale,
            initial_logit_bias=initial_logit_bias,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        history: list[float] = []
        converged = False
        completed_iterations = 0

        for iteration in range(iterations):
            quadratic_gradient, assignment = _relaxed_gradient(
                state,
                right,
                down,
                horizontal,
                vertical,
                uniform_gradient,
            )
            quadratic_inner = _state_inner(state, assignment, quadratic_gradient)
            quadratic_objective = 0.5 * quadratic_inner
            unary_objective = _state_inner(state, assignment, unary_tile_position)
            relaxed_objective = quadratic_objective + boundary_weight * unary_objective
            history.append(float(relaxed_objective))

            total_gradient = quadratic_gradient + boundary_weight * unary_tile_position
            oracle_layout = _linear_oracle(total_gradient)
            current_gradient_inner = _state_inner(state, assignment, total_gradient)
            oracle_gradient_inner = _layout_inner(oracle_layout, total_gradient)
            linear = oracle_gradient_inner - current_gradient_inner
            objective_scale = max(1.0, abs(relaxed_objective))
            if linear >= -tolerance * objective_scale:
                converged = True
                break

            oracle_quadratic = _layout_quadratic(
                oracle_layout, right, down, horizontal, vertical
            )
            oracle_quadratic_gradient_inner = _layout_inner(
                oracle_layout, quadratic_gradient
            )
            direction_quadratic = (
                oracle_quadratic
                + quadratic_objective
                - oracle_quadratic_gradient_inner
            )
            alpha, delta = _bounded_quadratic_step(
                linear, direction_quadratic, tolerance=tolerance
            )
            if alpha <= tolerance or delta >= -tolerance * objective_scale:
                converged = True
                break
            state.step_towards(oracle_layout, alpha)
            completed_iterations = iteration + 1

        final_gradient, final_assignment = _relaxed_gradient(
            state,
            right,
            down,
            horizontal,
            vertical,
            uniform_gradient,
        )
        final_relaxed = 0.5 * _state_inner(
            state, final_assignment, final_gradient
        ) + boundary_weight * _state_inner(
            state, final_assignment, unary_tile_position
        )
        if not history or abs(final_relaxed - history[-1]) > tolerance:
            history.append(float(final_relaxed))

        candidates = [_project_assignment(state, TILE_COUNT), *state.layouts]
        if initial_layout is not None:
            candidates.append(initial_layout)
        candidate = min(
            candidates,
            key=lambda layout: (
                _integral_objective(
                    layout,
                    right,
                    down,
                    horizontal,
                    vertical,
                    unary_tile_position,
                    boundary_weight,
                ),
                tuple(np.asarray(layout, dtype=np.int32).tolist()),
            ),
        ).copy()
        if refine_swaps:
            candidate = swap_refine(
                candidate,
                boundary_compatibility,
                boundary_weight=boundary_weight,
                weak_cells=refine_weak_cells,
                max_swaps=refine_swaps,
            )
        candidate = validate_permutation(candidate, name="qap_position_to_slot")
        candidate_objective = _integral_objective(
            candidate,
            right,
            down,
            horizontal,
            vertical,
            unary_tile_position,
            boundary_weight,
        )
        candidate_key = (candidate_objective, tuple(candidate.tolist()))
        best_key = (
            best_objective,
            tuple(best_layout.tolist()) if best_layout is not None else (),
        )
        if best_layout is None or candidate_key < best_key:
            best_layout = candidate.copy()
            best_objective = float(candidate_objective)
            best_relaxed_objective = float(final_relaxed)
            best_restart = restart
            best_iterations = completed_iterations
            best_converged = converged
            best_history = tuple(history)

    assert best_layout is not None
    return DirectionalQAPResult(
        position_to_slot=best_layout,
        objective=best_objective,
        relaxed_objective=best_relaxed_objective,
        restart=best_restart,
        iterations=best_iterations,
        converged=best_converged,
        history=best_history,
    )


def directional_qap_solver(
    compatibility: CompatibilityMatrices,
    **kwargs: object,
) -> np.ndarray:
    """Layout-only convenience wrapper around :func:`directional_qap`."""
    return directional_qap(compatibility, **kwargs).position_to_slot


__all__ = [
    "DirectionalQAPResult",
    "directional_qap",
    "directional_qap_solver",
]
