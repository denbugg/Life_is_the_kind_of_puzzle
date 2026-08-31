from __future__ import annotations

import numpy as np

from aiijc_puzzle.pixel_tails import (
    GRID_SIZE,
    IMAGE_SIZE,
    TILE_COUNT,
    TILE_SIZE,
    apply_nlm_color,
    assemble_tiles,
    gray_cell_mask,
    no_new_gray_guard,
    normalized_grid_descriptors,
    oracle_cell_fallback,
    recover_layout,
    split_tiles,
    summarize_variant_rows,
)


def _synthetic_target(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tiles = rng.integers(0, 256, (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    # A small smooth component makes the test closer to natural image tiles while
    # each tile remains uniquely recoverable.
    tiles = ((tiles.astype(np.uint16) + np.roll(tiles, 1, axis=1)) // 2).astype(np.uint8)
    return assemble_tiles(tiles)


def test_split_and_assemble_round_trip() -> None:
    target = _synthetic_target()
    assert np.array_equal(assemble_tiles(split_tiles(target)), target)


def test_recover_layout_finds_affine_corrupted_permutation() -> None:
    rng = np.random.default_rng(11)
    target = _synthetic_target()
    target_tiles = split_tiles(target)
    input_to_slot = rng.permutation(TILE_COUNT)
    shuffled = target_tiles[input_to_slot].astype(np.float32)
    scale = rng.uniform(0.8, 1.2, (TILE_COUNT, 1, 1, 1))
    offset = rng.uniform(-15, 15, (TILE_COUNT, 1, 1, 1))
    noise = rng.normal(0, 2, shuffled.shape)
    shuffled = np.clip(shuffled * scale + offset + noise, 0, 255).astype(np.uint8)
    input_image = assemble_tiles(shuffled)

    recovery = recover_layout(input_image, target, descriptor_bins=5)

    assert np.array_equal(recovery.input_to_slot, input_to_slot)
    assert np.array_equal(recovery.slot_to_input, np.argsort(input_to_slot))
    assert recovery.diagnostics()["mean_confidence"] > 0.99


def test_descriptor_bins_must_divide_tile_size() -> None:
    with np.testing.assert_raises_regex(ValueError, "positive divisor"):
        normalized_grid_descriptors(split_tiles(_synthetic_target()), bins=6)


def test_no_new_gray_guard_reverts_only_new_gray_cells() -> None:
    rng = np.random.default_rng(3)
    raw_tiles = rng.integers(0, 256, (TILE_COUNT, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    filtered_tiles = raw_tiles.copy()
    filtered_tiles[5] = 127
    raw = assemble_tiles(raw_tiles)
    filtered = assemble_tiles(filtered_tiles)

    guarded, reverted = no_new_gray_guard(raw, filtered)

    assert reverted == 1
    assert not gray_cell_mask(raw)[5]
    assert gray_cell_mask(filtered)[5]
    assert np.array_equal(split_tiles(guarded)[5], raw_tiles[5])
    assert np.array_equal(split_tiles(guarded)[6], filtered_tiles[6])


def test_oracle_cell_fallback_uses_target_but_not_unrelated_cells() -> None:
    target = _synthetic_target()
    target_tiles = split_tiles(target)
    raw_tiles = target_tiles.copy()
    filtered_tiles = target_tiles.copy()
    raw_tiles[1] = 0
    filtered_tiles[0] = 0
    raw = assemble_tiles(raw_tiles)
    filtered = assemble_tiles(filtered_tiles)

    selected, reverted = oracle_cell_fallback(raw, filtered, target)
    selected_tiles = split_tiles(selected)

    assert reverted == 1
    assert np.array_equal(selected_tiles[0], target_tiles[0])
    assert np.array_equal(selected_tiles[1], target_tiles[1])


def test_summarize_variant_rows() -> None:
    rows = [
        {
            "variants": {
                "raw": {"ssim": 0.2, "runtime_seconds": 0.0, "deployable": True},
                "better": {"ssim": 0.3, "runtime_seconds": 0.1, "deployable": True},
            }
        },
        {
            "variants": {
                "raw": {"ssim": 0.4, "runtime_seconds": 0.0, "deployable": True},
                "better": {"ssim": 0.5, "runtime_seconds": 0.2, "deployable": True},
            }
        },
    ]

    summary = summarize_variant_rows(rows)

    assert summary["better"]["mean_ssim"] == 0.4
    assert np.isclose(summary["better"]["mean_gain_vs_raw"], 0.1)
    assert np.isclose(summary["better"]["mean_gain_ci95_low"], 0.1)
    assert np.isclose(summary["better"]["mean_gain_ci95_high"], 0.1)
    assert summary["better"]["wins_vs_raw"] == 2
    assert np.isclose(summary["better"]["mean_runtime_seconds"], 0.15)


def test_constants_describe_contest_canvas() -> None:
    assert GRID_SIZE * TILE_SIZE == IMAGE_SIZE


def test_apply_nlm_color_is_lightweight_public_tail() -> None:
    image = _synthetic_target()

    result = apply_nlm_color(image)

    assert result.image.shape == image.shape
    assert result.image.dtype == np.uint8
    assert result.seconds >= 0
    assert result.deployable

    with np.testing.assert_raises_regex(ValueError, "must be positive"):
        apply_nlm_color(image, h=0)
