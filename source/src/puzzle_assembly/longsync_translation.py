"""Sparse LongSync-4 corruption scores for 2-D translation constraints.

The input is an undirected simple graph.  Every edge is stored once as
``(i, j)`` with ``i < j`` and its displacement is the vector from node ``i``
to node ``j``.  Traversing that edge in the opposite direction negates the
vector.  For each measured edge, this module enumerates every *simple*
three-edge alternate path between its endpoints; the measured edge and one
such path form a simple 4-cycle.

The update is the weighted RMS update in equations (6)--(7) of Li, Shi and
Lerman, CVPR 2024, specialized to the additive translation group.  Explicit
path enumeration is intentional here: it keeps the sparse-graph semantics
exact and excludes matrix-power walks with repeated nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SimpleLength3Paths = tuple[tuple[tuple[int, int, int, int], ...], ...]


@dataclass(frozen=True)
class LongSyncTranslationResult:
    """Per-edge LongSync-4 estimates, aligned with the input edge order.

    Edges with no simple 4-cycle support cannot be scored by LongSync-4.  They
    are explicitly marked by ``supported == False`` and ``support_counts == 0``;
    their neutral corruption and reliability weight are respectively ``0`` and
    ``1``.  Consumers must use the support mask rather than treating those
    neutral values as positive evidence.
    """

    corruption: np.ndarray
    weights: np.ndarray
    support_counts: np.ndarray
    supported: np.ndarray
    alternate_paths: SimpleLength3Paths
    corruption_history: np.ndarray
    beta_history: np.ndarray

    @property
    def edge_corruption(self) -> np.ndarray:
        """Alias making the per-edge alignment explicit."""

        return self.corruption

    @property
    def edge_weights(self) -> np.ndarray:
        """Alias making the per-edge alignment explicit."""

        return self.weights


def _validate_node_count(n_nodes: int) -> int:
    if not isinstance(n_nodes, (int, np.integer)) or isinstance(n_nodes, bool):
        raise TypeError("n_nodes must be a positive integer")
    value = int(n_nodes)
    if value <= 0:
        raise ValueError("n_nodes must be a positive integer")
    return value


def _validate_edges(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    edge_array = np.asarray(edges)
    if edge_array.ndim != 2 or edge_array.shape[1:] != (2,):
        raise ValueError("edges must have shape (n_edges, 2)")
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise TypeError("edge endpoints must be integers")
    canonical = edge_array.astype(np.int64, copy=True)
    if len(canonical) == 0:
        return canonical
    if int(canonical.min()) < 0 or int(canonical.max()) >= n_nodes:
        raise ValueError("edge endpoint is outside [0, n_nodes)")
    if np.any(canonical[:, 0] >= canonical[:, 1]):
        raise ValueError(
            "every edge must be canonical i < j with displacement directed from i to j"
        )
    if len(np.unique(canonical, axis=0)) != len(canonical):
        raise ValueError("duplicate undirected edge pair")
    return canonical


def _validate_graph(
    n_nodes: int, edges: np.ndarray, displacements: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    node_count = _validate_node_count(n_nodes)
    canonical = _validate_edges(node_count, edges)
    displacement = np.asarray(displacements, dtype=np.float64)
    if displacement.shape != (len(canonical), 2):
        raise ValueError(f"displacements must have shape {(len(canonical), 2)}")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("displacements must be finite")
    return node_count, canonical, displacement.copy()


def enumerate_simple_length3_paths(
    n_nodes: int, edges: np.ndarray
) -> SimpleLength3Paths:
    """Enumerate simple three-edge alternate paths for every measured edge.

    The outer tuple follows input edge order.  Each inner path is represented
    as ``(i, a, b, j)`` for target edge ``(i, j)``.  Paths and adjacency lists
    are lexicographically ordered, making the result deterministic independent
    of the input edge ordering.
    """

    node_count = _validate_node_count(n_nodes)
    canonical = _validate_edges(node_count, edges)
    adjacency: list[set[int]] = [set() for _ in range(node_count)]
    for first, second in canonical.tolist():
        adjacency[first].add(second)
        adjacency[second].add(first)

    all_paths: list[tuple[tuple[int, int, int, int], ...]] = []
    for first, last in canonical.tolist():
        paths: list[tuple[int, int, int, int]] = []
        for middle_first in sorted(adjacency[first]):
            if middle_first == last:
                continue
            for middle_second in sorted(adjacency[middle_first]):
                if middle_second in {first, middle_first, last}:
                    continue
                if last in adjacency[middle_second]:
                    paths.append((first, middle_first, middle_second, last))
        all_paths.append(tuple(paths))
    return tuple(all_paths)


def _edge_index_and_sign(
    source: int,
    destination: int,
    edge_lookup: dict[tuple[int, int], int],
) -> tuple[int, int]:
    """Return canonical edge index and traversal sign.

    This is the single implementation point for the inverse convention:
    traversing ``j -> i`` uses the negative of the stored ``i -> j`` vector.
    """

    if source < destination:
        return edge_lookup[(source, destination)], 1
    return edge_lookup[(destination, source)], -1


def _cycle_distances_and_edge_indices(
    edges: np.ndarray,
    displacements: np.ndarray,
    paths: SimpleLength3Paths,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    edge_lookup = {
        (int(first), int(second)): index
        for index, (first, second) in enumerate(edges.tolist())
    }
    all_distances: list[np.ndarray] = []
    all_path_edges: list[np.ndarray] = []
    maximum_float = np.longdouble(np.finfo(np.float64).max)

    for target_index, target_paths in enumerate(paths):
        distances = np.empty(len(target_paths), dtype=np.float64)
        path_edges = np.empty((len(target_paths), 3), dtype=np.int64)
        target = displacements[target_index].astype(np.longdouble)
        for path_index, path in enumerate(target_paths):
            path_sum = np.zeros(2, dtype=np.longdouble)
            for step, (source, destination) in enumerate(zip(path[:-1], path[1:])):
                edge_index, sign = _edge_index_and_sign(
                    source, destination, edge_lookup
                )
                path_edges[path_index, step] = edge_index
                path_sum += np.longdouble(sign) * displacements[edge_index]
            residual = path_sum - target
            distance = np.hypot(residual[0], residual[1])
            # Float64 is the public numeric contract.  Clamping an extreme but
            # finite long-double residual preserves finite deterministic output.
            distances[path_index] = float(min(distance, maximum_float))
        all_distances.append(distances)
        all_path_edges.append(path_edges)
    return tuple(all_distances), tuple(all_path_edges)


def _weighted_rms(
    distances: np.ndarray,
    path_edges: np.ndarray,
    previous_corruption: np.ndarray | None,
    previous_beta: float | None,
) -> float:
    if len(distances) == 0:
        return 0.0

    scale = float(np.max(distances))
    if scale == 0.0:
        return 0.0
    if previous_corruption is None or previous_beta is None:
        normalized_weights = np.ones(len(distances), dtype=np.longdouble)
    else:
        # Equation (7): each cycle weight excludes the target edge and is the
        # product of its three alternate-path edge weights.  Subtracting the
        # smallest path penalty is an exact common rescaling and prevents every
        # path weight from underflowing simultaneously.
        penalties = np.sum(
            previous_corruption[path_edges].astype(np.longdouble), axis=1
        )
        relative_penalties = penalties - np.min(penalties)
        exponents = -np.longdouble(previous_beta) * relative_penalties
        normalized_weights = np.exp(exponents)

    scaled_distances = distances.astype(np.longdouble) / np.longdouble(scale)
    mean_square = np.sum(
        normalized_weights * scaled_distances * scaled_distances
    ) / np.sum(normalized_weights)
    value = np.longdouble(scale) * np.sqrt(max(mean_square, np.longdouble(0.0)))
    return float(min(value, np.longdouble(np.finfo(np.float64).max)))


def longsync4_translation(
    n_nodes: int,
    edges: np.ndarray,
    displacements: np.ndarray,
    *,
    iterations: int = 10,
) -> LongSyncTranslationResult:
    """Estimate translation-edge corruption using simple 4-cycles.

    Iteration ``t`` uses ``beta_t = min(2**t, 20)`` and follows equations
    (6)--(7): cycle inconsistency is aggregated with a weighted RMS, while a
    cycle's weight is the product of the reliability weights on the three
    alternate-path edges.  Ten iterations match the paper's experimental
    schedule.
    """

    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool):
        raise TypeError("iterations must be a positive integer")
    iteration_count = int(iterations)
    if iteration_count <= 0:
        raise ValueError("iterations must be a positive integer")

    node_count, canonical, displacement = _validate_graph(
        n_nodes, edges, displacements
    )
    paths = enumerate_simple_length3_paths(node_count, canonical)
    support_counts = np.fromiter(
        (len(edge_paths) for edge_paths in paths),
        dtype=np.int64,
        count=len(canonical),
    )
    supported = support_counts > 0
    cycle_distances, path_edge_indices = _cycle_distances_and_edge_indices(
        canonical, displacement, paths
    )

    beta_history = np.asarray(
        [min(2.0 ** min(iteration, 5), 20.0) for iteration in range(iteration_count)],
        dtype=np.float64,
    )
    history = np.zeros((iteration_count, len(canonical)), dtype=np.float64)
    previous_corruption: np.ndarray | None = None
    previous_beta: float | None = None
    for iteration, beta in enumerate(beta_history.tolist()):
        for edge_index in range(len(canonical)):
            history[iteration, edge_index] = _weighted_rms(
                cycle_distances[edge_index],
                path_edge_indices[edge_index],
                previous_corruption,
                previous_beta,
            )
        previous_corruption = history[iteration]
        previous_beta = beta

    corruption = history[-1].copy()
    # Unsupported entries already have neutral zero corruption.  The direct
    # exponential may legitimately underflow to zero for overwhelming evidence;
    # zero remains a deterministic finite reliability weight.
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        weights = np.exp(-beta_history[-1] * corruption)
    weights[~supported] = 1.0

    return LongSyncTranslationResult(
        corruption=corruption,
        weights=weights,
        support_counts=support_counts,
        supported=supported,
        alternate_paths=paths,
        corruption_history=history,
        beta_history=beta_history,
    )


__all__ = [
    "LongSyncTranslationResult",
    "SimpleLength3Paths",
    "enumerate_simple_length3_paths",
    "longsync4_translation",
]
