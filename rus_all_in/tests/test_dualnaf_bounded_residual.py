from __future__ import annotations

import numpy as np
import pytest

from aiijc_puzzle.dualnaf_bounded_residual import (
    ARM_ALPHAS,
    blend_same_index_tiles,
    choose_winner,
    paired_bootstrap_ci,
)


def constant_tiles(value: int) -> np.ndarray:
    return np.full((576, 20, 20, 3), value, dtype=np.uint8)


def test_zero_alpha_preserves_original_tiles_exactly() -> None:
    original = constant_tiles(17)
    rendered = constant_tiles(240)
    blended = blend_same_index_tiles(original, rendered, 0.0)
    assert np.array_equal(blended, original)
    assert not np.shares_memory(blended, original)


def test_convex_blend_uses_only_same_tile_index() -> None:
    original = constant_tiles(8)
    rendered = constant_tiles(8)
    rendered[73] = 200
    blended = blend_same_index_tiles(original, rendered, 0.25)
    assert np.all(blended[73] == 56)
    assert np.all(blended[np.arange(576) != 73] == 8)


def test_blend_rejects_nonpreregistered_strength_range() -> None:
    original = constant_tiles(8)
    with pytest.raises(ValueError, match=r"\[0, 0.5\]"):
        blend_same_index_tiles(original, original, 0.75)


def test_winner_selection_uses_only_candidates_and_low_alpha_tie_break() -> None:
    means = dict.fromkeys(ARM_ALPHAS, 0.2)
    means["baseline_alpha_0"] = 0.9
    means["dualnaf_residual_alpha_0_25"] = 0.3
    means["dualnaf_residual_alpha_0_375"] = 0.3
    assert choose_winner(means) == "dualnaf_residual_alpha_0_25"


def test_paired_bootstrap_interval_is_deterministic_and_positive() -> None:
    values = np.linspace(0.001, 0.01, 24)
    first = paired_bootstrap_ci(values)
    second = paired_bootstrap_ci(values)
    assert first == second
    assert first[0] > 0
