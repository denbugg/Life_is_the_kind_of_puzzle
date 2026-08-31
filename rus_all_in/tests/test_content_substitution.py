from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from aiijc_puzzle.content_substitution import (
    build_assignments,
    evaluate_variants,
    extract_tiles,
    pairwise_tile_rmse,
    recover_dirty_tile_alignment,
    render_tiles,
    select_target_paths,
)


def _four_tile_image() -> np.ndarray:
    tiles = np.stack([np.full((20, 20, 3), value, dtype=np.uint8) for value in (0, 12, 40, 220)])
    return render_tiles(tiles, grid_size=2)


def test_extract_and_render_tiles_round_trip() -> None:
    image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)

    tiles = extract_tiles(image, grid_size=2, tile_size=20)

    assert tiles.shape == (4, 20, 20, 3)
    assert np.array_equal(render_tiles(tiles, grid_size=2), image)


def test_pairwise_tile_rmse_matches_direct_computation() -> None:
    tiles = extract_tiles(_four_tile_image(), grid_size=2, tile_size=20)

    actual = pairwise_tile_rmse(tiles)
    direct = np.empty((4, 4), dtype=np.float64)
    for row in range(4):
        for column in range(4):
            delta = tiles[row].astype(np.float64) - tiles[column].astype(np.float64)
            direct[row, column] = np.sqrt(np.mean(np.square(delta)))

    assert actual.dtype == np.float64
    assert np.allclose(actual, direct, atol=1e-5)
    assert np.array_equal(np.diag(actual), np.zeros(4, dtype=np.float64))


def test_all_non_identity_assignments_forbid_diagonal_and_hungarian_is_optimal() -> None:
    tiles = extract_tiles(_four_tile_image(), grid_size=2, tile_size=20)
    costs = pairwise_tile_rmse(tiles)

    assignments = build_assignments(
        costs,
        seed=420,
        board_key="board.png",
        nearest_ks=(2, 3),
    )

    indices = np.arange(4)
    assert np.array_equal(assignments["identity"], indices)
    for name, assignment in assignments.items():
        if name != "identity":
            assert np.all(assignment != indices), name
    assert len(np.unique(assignments["bijective_derangement"])) == 4

    derangements = [
        permutation
        for permutation in itertools.permutations(range(4))
        if all(index != value for index, value in enumerate(permutation))
    ]
    optimal_cost = min(
        sum(costs[index, value] for index, value in enumerate(permutation))
        for permutation in derangements
    )
    actual_cost = float(np.sum(costs[indices, assignments["bijective_derangement"]]))
    assert actual_cost == optimal_cost

    repeated = build_assignments(
        costs,
        seed=420,
        board_key="board.png",
        nearest_ks=(2, 3),
    )
    assert all(np.array_equal(assignments[name], repeated[name]) for name in assignments)


def test_evaluate_variants_reports_ssim_placement_duplicates_and_rmse() -> None:
    target = _four_tile_image()
    tiles = extract_tiles(target, grid_size=2, tile_size=20)
    costs = pairwise_tile_rmse(tiles)
    assignments = build_assignments(
        costs,
        seed=7,
        board_key="tiny.png",
        nearest_ks=(2,),
    )

    results = evaluate_variants(target, costs, assignments, grid_size=2)

    identity = results["identity"].metrics
    assert identity["ssim"] == 1.0
    assert identity["exact_placement_count"] == 4
    assert identity["exact_placement_fraction"] == 1.0
    assert identity["duplicate_use_count"] == 0
    assert identity["selected_rmse_quantiles"]["q50"] == 0.0

    bijective = results["bijective_derangement"].metrics
    assert bijective["exact_placement_count"] == 0
    assert bijective["duplicate_use_count"] == 0
    assert 0.0 <= bijective["ssim"] < 1.0

    nearest = results["nearest_other"].metrics
    assert nearest["exact_placement_count"] == 0
    assert nearest["duplicate_use_count"] > 0
    assert nearest["selected_rmse_mean"] > 0.0


def test_recover_dirty_alignment_finds_affine_corrupted_permutation() -> None:
    rng = np.random.default_rng(11)
    target_tiles = rng.integers(0, 256, (4, 20, 20, 3), dtype=np.uint8)
    target_tiles = (
        (target_tiles.astype(np.uint16) + np.roll(target_tiles, 1, axis=1)) // 2
    ).astype(np.uint8)
    input_to_target = np.asarray([2, 0, 3, 1])
    dirty_tiles = target_tiles[input_to_target].astype(np.float64)
    dirty_tiles = np.clip(
        dirty_tiles * rng.uniform(0.85, 1.15, (4, 1, 1, 1))
        + rng.uniform(-10, 10, (4, 1, 1, 1))
        + rng.normal(0, 1, dirty_tiles.shape),
        0,
        255,
    ).astype(np.uint8)

    alignment = recover_dirty_tile_alignment(
        render_tiles(dirty_tiles, grid_size=2),
        render_tiles(target_tiles, grid_size=2),
        grid_size=2,
        tile_size=20,
    )

    assert np.array_equal(alignment.target_to_input, np.argsort(input_to_target))
    assert alignment.metrics["descriptor_correlation_mean"] > 0.99
    assert alignment.metrics["permutation_labels_available"] is False
    assert alignment.aligned_tiles.shape == target_tiles.shape


def test_select_target_paths_is_stable_and_confined_to_pool(tmp_path: Path) -> None:
    for index in range(12):
        (tmp_path / f"img_{index:06d}.png").touch()

    first = select_target_paths(tmp_path, count=4, seed=123, pool_start=3, pool_stop=10)
    second = select_target_paths(tmp_path, count=4, seed=123, pool_start=3, pool_stop=10)

    assert first == second
    assert len(first) == 4
    assert all(3 <= int(path.stem.removeprefix("img_")) < 10 for path in first)


def test_cli_writes_per_board_and_aggregate_json(tmp_path: Path) -> None:
    targets_dir = tmp_path / "targets"
    inputs_dir = tmp_path / "inputs"
    output_dir = tmp_path / "output"
    targets_dir.mkdir()
    inputs_dir.mkdir()
    Image.fromarray(_four_tile_image(), mode="RGB").save(targets_dir / "img_000001.png")
    Image.fromarray(_four_tile_image(), mode="RGB").save(inputs_dir / "img_000001.png")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_content_substitution.py",
            "--targets-dir",
            str(targets_dir),
            "--inputs-dir",
            str(inputs_dir),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
            "--grid-size",
            "2",
            "--tile-size",
            "20",
            "--nearest-k",
            "2",
            "--save-images",
            "--no-apply-nlm-h9",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    per_board = json.loads((output_dir / "per_board.json").read_text())
    aggregate = json.loads((output_dir / "aggregate.json").read_text())
    assert per_board[0]["board"] == "img_000001.png"
    assert per_board[0]["clean_oracle_variants"]["identity"]["ssim"] == 1.0
    assert per_board[0]["raw_dirty_proxy_variants"]["identity"]["ssim"] == 1.0
    assert aggregate["configuration"]["target_source"].startswith("clean train targets")
    assert aggregate["clean_oracle_variants"]["bijective_derangement"]["board_count"] == 1
    assert aggregate["raw_dirty_proxy_variants"]["bijective_derangement"]["board_count"] == 1
    assert aggregate["dirty_alignment"]["board_count"] == 1
    assert (output_dir / "images/img_000001/clean_oracle/bijective_derangement.png").is_file()
    assert (output_dir / "images/img_000001/raw_dirty_proxy/identity.png").is_file()
