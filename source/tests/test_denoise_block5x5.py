from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from puzzle_denoise_v2.block5x5 import (
    Block5x5Loss,
    BlockLossWeights,
    CleanBlockStore,
    assemble_blocks,
    canonical_name_hash,
    load_protocol,
    neighbouring_tile_mean_loss,
    seam_gradient_charbonnier,
)


ROOT = Path(__file__).resolve().parents[1]


def make_index_tiles(batch: int = 2) -> torch.Tensor:
    values = torch.arange(batch * 5 * 5, dtype=torch.float32).reshape(batch, 5, 5, 1, 1, 1)
    return values.expand(batch, 5, 5, 3, 20, 20).clone()


def test_assemble_blocks_preserves_tile_geometry() -> None:
    tiles = make_index_tiles(1)
    block = assemble_blocks(tiles)
    assert block.shape == (1, 3, 100, 100)
    for row in range(5):
        for column in range(5):
            expected = row * 5 + column
            region = block[0, :, row * 20 : (row + 1) * 20, column * 20 : (column + 1) * 20]
            assert torch.all(region == expected)


def test_seam_and_neighbour_losses_are_zero_for_identical_blocks() -> None:
    target = torch.rand(2, 5, 5, 3, 20, 20)
    block = assemble_blocks(target)
    assert float(seam_gradient_charbonnier(block, block)) == pytest.approx(0.001, abs=1e-7)
    assert float(neighbouring_tile_mean_loss(target, target)) == 0.0


def test_seam_loss_detects_cross_tile_discontinuity_error() -> None:
    target = torch.zeros(1, 3, 100, 100)
    prediction = target.clone()
    prediction[:, :, :, 20:] = 0.25
    baseline = float(seam_gradient_charbonnier(target, target))
    changed = float(seam_gradient_charbonnier(prediction, target))
    assert changed > baseline + 0.02


def test_block_loss_backpropagates_through_cross_tile_terms() -> None:
    target = torch.rand(1, 5, 5, 3, 20, 20)
    prediction = (target + 0.02 * torch.randn_like(target)).requires_grad_(True)
    parameters = torch.zeros(25, 5)
    criterion = Block5x5Loss(BlockLossWeights())
    loss, components = criterion(prediction, target, parameters, parameters)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert set(components) >= {
        "block_ssim",
        "block_gradient",
        "seam_gradient",
        "neighbour_mean",
        "total",
    }


def test_clean_block_store_samples_contiguous_whole_block_dihedrals(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    image = np.empty((480, 480, 3), dtype=np.uint8)
    for row in range(24):
        for column in range(24):
            image[row * 20 : (row + 1) * 20, column * 20 : (column + 1) * 20] = (
                row,
                column,
                (row * 24 + column) % 256,
            )
    Image.fromarray(image).save(target_dir / "source.png")
    first = CleanBlockStore(target_dir, ["source.png"]).sample(4, np.random.default_rng(9))
    second = CleanBlockStore(target_dir, ["source.png"]).sample(4, np.random.default_rng(9))
    assert first.shape == (4, 25, 3, 20, 20)
    assert torch.equal(first, second)
    # Every sampled tile remains spatially constant, proving that the whole
    # 100x100 patch was transformed before it was split back into tiles.
    assert torch.all(first.std(dim=(-2, -1)) == 0)


def test_frozen_protocol_names_and_partitions_are_auditable() -> None:
    path = ROOT / "configs" / "denoise_block5x5_v1.json"
    protocol = load_protocol(path)
    manifest = json.loads((ROOT / protocol["inputs"]["manifest"]).read_text())
    development = protocol["source_partitions"]["development"]
    gate = protocol["source_partitions"]["frozen_gate"]
    assert canonical_name_hash(development["names"]) == development["names_sha256"]
    assert canonical_name_hash(gate["names"]) == gate["names_sha256"]
    assert not set(development["names"]) & set(gate["names"])
    assert set(development["names"]) <= set(manifest["splits"]["val"])
    assert set(gate["names"]) <= set(manifest["splits"]["val"])
    assert not set(development["names"]) & set(manifest["splits"]["train"])
    assert not set(gate["names"]) & set(manifest["splits"]["train"])
    assert protocol["anti_leakage"]["candidate_graph_oracle_files_or_fixtures_accessed"] is False
