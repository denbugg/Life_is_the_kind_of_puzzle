from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_fullres_union_voter import (
    FULLRES_DENOISER_SHA256,
    accept_focal_proposals,
    compose_fullres_union_focal_arm,
    load_fullres_denoiser,
    supported_absent_edges,
)
from aiijc_puzzle.taska_pair_pipeline import ARM_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "fullres_boundary_denoiser.pt"
)


def test_fixed_support_and_focal_gates_are_exact() -> None:
    old = (RawTailEdge(0, 1, "right"),)
    three = RawTailEdge(1, 2, "right")
    four = RawTailEdge(2, 3, "down")
    only_two = RawTailEdge(3, 0, "down")
    scorers = (
        {old[0], three, four, only_two},
        {three, four, only_two},
        {three, four},
        {four},
    )
    proposed, support = supported_absent_edges(old, scorers)
    assert proposed == (three, four)
    assert support == (3, 4)
    accepted, logits = accept_focal_proposals(proposed, np.array([-1e-6, 0.0]))
    assert accepted == (four,)
    assert np.array_equal(logits, np.array([0.0], dtype=np.float32))


def test_composition_is_strict_and_uses_only_original_costs() -> None:
    grid = 2
    count = grid * grid
    right = np.full((count, count), 5.0, dtype=np.float64)
    down = np.full((count, count), 5.0, dtype=np.float64)
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    right[0, 1] = 0.1
    right[2, 3] = 0.1
    down[0, 2] = 0.1
    down[1, 3] = 0.1
    current = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(0, 2, "down"),
    )
    new = (RawTailEdge(2, 3, "right"), RawTailEdge(1, 3, "down"))
    base = np.arange(count, dtype=np.int32)
    layouts = {name: base for name in ARM_NAMES}
    result = compose_fullres_union_focal_arm(
        cost_right=right,
        cost_down=down,
        current_edges=current,
        current_focal_logits=np.ones(len(current), dtype=np.float32),
        accepted_new_edges=new,
        accepted_new_logits=np.ones(len(new), dtype=np.float32),
        four_layouts=layouts,
        grid=grid,
    )
    assert np.array_equal(np.sort(result.layout), np.arange(count))
    assert result.union_edges == current + new
    assert result.diagnostics["raw_dense_cost_matrices_unchanged"] is True
    assert result.diagnostics["restored_pixels_matcher_only"] is True


def test_composition_rejects_overlap_between_current_and_new() -> None:
    edge = RawTailEdge(0, 1, "right")
    costs = np.zeros((4, 4), dtype=np.float64)
    layouts = {name: np.arange(4, dtype=np.int32) for name in ARM_NAMES}
    with pytest.raises(ValueError, match="absent"):
        compose_fullres_union_focal_arm(
            cost_right=costs,
            cost_down=costs,
            current_edges=(edge,),
            current_focal_logits=np.array([1.0]),
            accepted_new_edges=(edge,),
            accepted_new_logits=np.array([1.0]),
            four_layouts=layouts,
            grid=2,
        )


def test_sha_locked_fullres_checkpoint_loads_on_cpu() -> None:
    model = load_fullres_denoiser(CHECKPOINT, device="cpu")
    assert model.checkpoint_sha256 == FULLRES_DENOISER_SHA256
    assert next(model.parameters()).device.type == "cpu"
