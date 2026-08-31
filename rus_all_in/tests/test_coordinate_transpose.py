from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.absolute_coordinate_sorter import (
    AbsoluteCoordinateSorter,
    decode_coordinate_logits,
)
from aiijc_puzzle.coordinate_transpose import (
    collect_transpose_coordinate_views,
    fuse_transpose_coordinate_views,
    map_transposed_axis_logits,
    symmetric_axis_consistency_loss,
    transpose_positions,
    transpose_tile_view,
)
from aiijc_puzzle.socket_matcher import SocketMatcher


def _model() -> AbsoluteCoordinateSorter:
    backbone = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    )
    return AbsoluteCoordinateSorter(
        backbone,
        grid=3,
        head_dimension=8,
        heads=2,
        set_layers=1,
        sinkhorn_iterations=4,
    )


def test_transpose_position_and_axis_mapping_are_exact() -> None:
    positions = torch.arange(9).reshape(1, 9)
    expected = torch.tensor([[0, 3, 6, 1, 4, 7, 2, 5, 8]])
    transposed = transpose_positions(positions, grid=3)
    assert torch.equal(transposed, expected)
    assert torch.equal(transpose_positions(transposed, grid=3), positions)

    transposed_rows = torch.arange(27, dtype=torch.float32).reshape(1, 9, 3)
    transposed_columns = transposed_rows + 100.0
    mapped_rows, mapped_columns = map_transposed_axis_logits(
        transposed_rows,
        transposed_columns,
    )
    assert torch.equal(mapped_rows, transposed_columns)
    assert torch.equal(mapped_columns, transposed_rows)


def test_transpose_view_is_an_involution_without_tile_index_reordering() -> None:
    tiles = torch.arange(9 * 3 * 4 * 4).reshape(1, 9, 3, 4, 4)
    transposed = transpose_tile_view(tiles)
    assert torch.equal(transposed[:, :, :, 1, 2], tiles[:, :, :, 2, 1])
    assert torch.equal(transpose_tile_view(transposed), tiles)


def test_transpose_tta_is_input_permutation_equivariant_and_state_dict_safe() -> None:
    torch.manual_seed(20260909)
    model = _model().eval()
    tiles = torch.rand(1, 9, 3, 20, 20)
    permutation = torch.tensor([5, 0, 8, 2, 7, 3, 1, 6, 4])
    state_keys = tuple(model.state_dict())
    with torch.no_grad():
        reference_views = collect_transpose_coordinate_views(model, tiles)
        shuffled_views = collect_transpose_coordinate_views(model, tiles[:, permutation])
        reference = fuse_transpose_coordinate_views(
            reference_views,
            grid=3,
            mode="symmetric",
        )
        shuffled = fuse_transpose_coordinate_views(
            shuffled_views,
            grid=3,
            mode="symmetric",
        )
    for field in ("row_logits", "column_logits", "slot_logits"):
        torch.testing.assert_close(
            getattr(shuffled, field),
            getattr(reference, field)[:, permutation],
            atol=4e-5,
            rtol=4e-5,
        )
    # The helper introduces neither parameters nor an input-index embedding.
    assert tuple(model.state_dict()) == state_keys
    assert all("input_index" not in name for name, _ in model.named_parameters())


def test_transpose_consistency_has_finite_head_gradients() -> None:
    torch.manual_seed(20260910)
    model = _model().train()
    views = collect_transpose_coordinate_views(model, torch.rand(1, 9, 3, 20, 20))
    loss = symmetric_axis_consistency_loss(views)
    assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_fused_transpose_logits_decode_to_one_strict_original_tile_per_slot() -> None:
    torch.manual_seed(20260911)
    model = _model().eval()
    original_tile_ids = np.arange(9, dtype=np.int32)
    with torch.no_grad():
        views = collect_transpose_coordinate_views(
            model,
            torch.rand(1, 9, 3, 20, 20),
        )
        fused = fuse_transpose_coordinate_views(
            views,
            grid=3,
            mode="row-teacher",
        )
    layout = decode_coordinate_logits(fused.slot_logits)
    assert np.array_equal(np.sort(layout), original_tile_ids)
    # Decoding selects indices from the untouched upright tile array only.
    assert np.array_equal(np.sort(original_tile_ids[layout]), original_tile_ids)
