from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.border_pointer_sorter import (
    AbsolutePointerDecoder,
    BorderPointerSorter,
    FullResolutionPerimeterEncoder,
    border_pointer_loss,
)


def _small_sorter() -> BorderPointerSorter:
    return BorderPointerSorter(
        feature_width=8,
        feature_blocks=1,
        dimension=16,
        heads=4,
        board_layers=1,
        pointer_layers=2,
        max_grid=4,
    )


def test_perimeter_is_clockwise_unique_and_full_resolution() -> None:
    field = torch.arange(20 * 20).reshape(1, 1, 20, 20).float()
    perimeter = FullResolutionPerimeterEncoder.ordered_perimeter(field)
    assert perimeter.shape == (1, 1, 76)
    assert len(torch.unique(perimeter)) == 76
    assert perimeter[0, 0, :20].tolist() == list(range(20))
    assert perimeter[0, 0, 20:39].tolist() == list(range(39, 400, 20))

    encoder = FullResolutionPerimeterEncoder(width=8, blocks=1)
    tile, sides = encoder(torch.rand(2, 3, 3, 20, 20))
    assert tile.shape == (2, 3, 8)
    assert sides.shape == (2, 3, 4, 8)


def test_pointer_left_and_up_evidence_changes_conditional_logits() -> None:
    torch.manual_seed(1)
    decoder = AbsolutePointerDecoder(dimension=8, max_grid=2, layers=1).eval()
    memory = torch.randn(1, 4, 8)
    layout = torch.tensor([[0, 1, 2, 3]])
    zeros = torch.zeros(1, 4, 4)
    baseline = decoder.teacher_forced(
        memory,
        layout,
        grid=2,
        right_logits=zeros,
        down_logits=zeros,
    )

    right = zeros.clone()
    right[0, 0, 1] = 3.0
    with_right = decoder.teacher_forced(
        memory,
        layout,
        grid=2,
        right_logits=right,
        down_logits=zeros,
    )
    expected = torch.nn.functional.softplus(decoder.left_edge_log_weight) * 3.0
    assert torch.allclose(with_right[0, 1, 1] - baseline[0, 1, 1], expected)

    down = zeros.clone()
    down[0, 0, 2] = 4.0
    with_down = decoder.teacher_forced(
        memory,
        layout,
        grid=2,
        right_logits=zeros,
        down_logits=down,
    )
    expected = torch.nn.functional.softplus(decoder.up_edge_log_weight) * 4.0
    assert torch.allclose(with_down[0, 2, 2] - baseline[0, 2, 2], expected)


def test_sorter_is_input_permutation_equivariant_and_decodes_strictly() -> None:
    torch.manual_seed(2)
    model = _small_sorter().eval()
    tiles = torch.rand(1, 16, 3, 20, 20)
    permutation = torch.randperm(16)

    memory = model.encode(tiles)
    permuted_memory = model.encode(tiles[:, permutation])
    assert torch.allclose(permuted_memory, memory[:, permutation], atol=2e-5, rtol=2e-5)

    layout = model.decode(tiles, grid=4)[0]
    permuted_layout = model.decode(tiles[:, permutation], grid=4)[0]
    assert torch.equal(permutation[permuted_layout], layout)
    assert np.array_equal(np.sort(layout.numpy()), np.arange(16))
    beam_layout = model.decode_beam(tiles, grid=4, width=2)[0]
    assert np.array_equal(np.sort(beam_layout.numpy()), np.arange(16))


def test_loss_backpropagates_through_strict_teacher_layout() -> None:
    torch.manual_seed(3)
    model = _small_sorter()
    tiles = torch.rand(1, 16, 3, 20, 20)
    target = torch.randperm(16).unsqueeze(0)
    output = model(tiles, teacher_layout=target, grid=4)
    loss, diagnostics = border_pointer_loss(output, target, grid=4)
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["pointer_nll"] > 0
    assert any(parameter.grad is not None for parameter in model.perimeter.parameters())
