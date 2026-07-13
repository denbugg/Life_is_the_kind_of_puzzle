"""Leakage-safe 24x24 tile assembly primitives."""

from .compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    build_edge_filter_score_bank,
    fuse_ranked_scores,
)
from .components import ComponentSolveResult, reciprocal_component_solver
from .geometry import GRID, TILE, TILE_COUNT, inverse_permutation, validate_permutation
from .metrics import layout_metrics, retrieval_metrics
from .solvers import beam_row_major, greedy_row_major, identity_layout, random_layout

__all__ = [
    "GRID",
    "TILE",
    "TILE_COUNT",
    "CompatibilityMatrices",
    "ComponentSolveResult",
    "beam_row_major",
    "build_classical_score_bank",
    "build_edge_filter_score_bank",
    "fuse_ranked_scores",
    "greedy_row_major",
    "identity_layout",
    "inverse_permutation",
    "layout_metrics",
    "random_layout",
    "reciprocal_component_solver",
    "retrieval_metrics",
    "validate_permutation",
]
