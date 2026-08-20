"""E15 multiplex relaxation over frozen raw-E14 and guarded-restored graphs."""
from __future__ import annotations

import numpy as np

import kaggle_e14_solver as e14


RAW_SUPPORT_WEIGHT = 0.70
GUARDED_SUPPORT_WEIGHT = 0.30
DISAGREEMENT_PENALTY = 0.15


def _multiplex_support(beliefs, raw_graph, guarded_graph):
    raw_support = e14._directional_support(beliefs, *raw_graph)
    guarded_support = e14._directional_support(beliefs, *guarded_graph)
    return (
        RAW_SUPPORT_WEIGHT * raw_support
        + GUARDED_SUPPORT_WEIGHT * guarded_support
        - DISAGREEMENT_PENALTY * np.abs(raw_support - guarded_support)
    )


def solve_layout(raw_right, raw_down, guarded_right, guarded_down, pos, seed):
    """Run unchanged E14 relaxation except for the declared multiplex support."""
    raw_right = np.asarray(raw_right, np.float64)
    raw_down = np.asarray(raw_down, np.float64)
    guarded_right = np.asarray(guarded_right, np.float64)
    guarded_down = np.asarray(guarded_down, np.float64)
    unary = np.asarray(pos, np.float64)
    expected = (e14.N, e14.N)
    for name, matrix in (
        ("raw_right", raw_right), ("raw_down", raw_down),
        ("guarded_right", guarded_right), ("guarded_down", guarded_down),
        ("pos", unary),
    ):
        if matrix.shape != expected:
            raise ValueError(f"{name}: expected {expected}, got {matrix.shape}")

    raw_right_graph = e14._topk_compatibility(raw_right, e14.TOP_K_EDGES)
    raw_down_graph = e14._topk_compatibility(raw_down, e14.TOP_K_EDGES)
    guarded_right_graph = e14._topk_compatibility(guarded_right, e14.TOP_K_EDGES)
    guarded_down_graph = e14._topk_compatibility(guarded_down, e14.TOP_K_EDGES)
    raw_graph = (*raw_right_graph, *raw_down_graph)
    guarded_graph = (*guarded_right_graph, *guarded_down_graph)

    weighted_pos = e14.POSITION_WEIGHT * unary
    unary = unary - unary.max(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    tie_break = rng.uniform(-1e-7, 1e-7, size=expected)
    locked_position = np.full(e14.N, -1, dtype=np.int32)
    logits = unary + tie_break
    _, initial_position_of_tile = e14._assignment(logits, locked_position)
    best_layout = e14._layout_from_position_of_tile(initial_position_of_tile)
    best_objective = e14.objective(best_layout, raw_right, raw_down, weighted_pos)
    soft = e14._masked_sinkhorn(logits, e14.PHASES[0].temperature, locked_position)
    beliefs = 0.45 * soft + 0.55 * e14._hard_beliefs(logits, locked_position)

    for phase in e14.PHASES:
        for _ in range(phase.iterations):
            support = _multiplex_support(beliefs, raw_graph, guarded_graph)
            logits = (
                e14.POSITION_WEIGHT * unary
                + phase.edge_weight * support
                + phase.inertia * np.log(np.maximum(beliefs, e14.EPS))
                + tie_break
            )
            _, position_of_tile = e14._assignment(logits, locked_position)
            candidate_layout = e14._layout_from_position_of_tile(position_of_tile)
            candidate_objective = e14.objective(
                candidate_layout, raw_right, raw_down, weighted_pos
            )
            if candidate_objective > best_objective:
                best_objective = candidate_objective
                best_layout = candidate_layout
            soft = e14._masked_sinkhorn(logits, phase.temperature, locked_position)
            hard = e14._hard_beliefs(logits, locked_position)
            beliefs = (1.0 - phase.hard_mix) * soft + phase.hard_mix * hard
        e14._freeze_confident(logits, locked_position, phase.freeze_fraction)
        soft = e14._masked_sinkhorn(logits, phase.temperature, locked_position)
        hard = e14._hard_beliefs(logits, locked_position)
        beliefs = (1.0 - phase.hard_mix) * soft + phase.hard_mix * hard

    support = _multiplex_support(beliefs, raw_graph, guarded_graph)
    final_logits = e14.POSITION_WEIGHT * unary + e14.PHASES[-1].edge_weight * support
    _, position_of_tile = e14._assignment(final_logits, locked_position)
    candidate_layout = e14._layout_from_position_of_tile(position_of_tile)
    if e14.objective(candidate_layout, raw_right, raw_down, weighted_pos) > best_objective:
        best_layout = candidate_layout
    return best_layout
