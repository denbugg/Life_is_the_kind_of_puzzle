from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.socket_matcher import SocketMatcher, partial_log_optimal_transport
from aiijc_puzzle.socket_permutation_flow import (
    SocketPermutationFlow,
    build_socket_topk_graph,
    extract_frozen_socket_evidence,
    hungarian_layout,
    interpolate_permutations,
    permutation_flow_loss,
    tile_positions,
)


def _assignments(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(41)
    right = torch.randn(1, count, count)
    down = torch.randn(1, count, count)
    right[:, torch.arange(count), torch.arange(count)] = -1e4
    down[:, torch.arange(count), torch.arange(count)] = -1e4
    grid = int(np.sqrt(count))
    return (
        partial_log_optimal_transport(right, torch.tensor(0.0), unmatched=grid, iterations=20),
        partial_log_optimal_transport(down, torch.tensor(0.0), unmatched=grid, iterations=20),
    )


def test_interpolation_preserves_strict_permutation_and_endpoints() -> None:
    start = torch.tensor([[7, 2, 5, 1, 0, 6, 3, 8, 4]])
    target = torch.arange(9).reshape(1, 9)
    generator = torch.Generator().manual_seed(43)
    beginning = interpolate_permutations(start, target, 0.0, generator=generator)
    middle = interpolate_permutations(start, target, 0.5, generator=generator)
    end = interpolate_permutations(start, target, 1.0, generator=generator)
    assert torch.equal(beginning, start)
    assert torch.equal(end, target)
    assert torch.equal(middle.sort(1).values, target)
    assert int((middle == target).sum()) >= int((start == target).sum())


def test_graph_and_flow_are_equivariant_to_tile_relabelling() -> None:
    grid = 3
    count = grid * grid
    right, down = _assignments(count)
    graph = build_socket_topk_graph(right, down, top_k=3)
    tile_features = torch.randn(1, count, 12)
    current = torch.tensor([[7, 2, 5, 1, 0, 6, 3, 8, 4]])
    model = SocketPermutationFlow(
        tile_feature_dimension=12,
        dimension=16,
        layers=2,
        coordinate_bands=2,
        time_bands=2,
        sinkhorn_iterations=12,
    ).eval()
    with torch.no_grad():
        model.flow_head.weight.normal_(0, 0.1)
        model.flow_head.bias.normal_(0, 0.1)
        reference = model(tile_features, graph, current, 0.3, grid=grid)

    permutation = torch.tensor([5, 0, 8, 2, 7, 3, 1, 6, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(count)
    relabelled_layout = inverse[current]
    extended = torch.cat((permutation, torch.tensor([count])))
    relabelled_right = right[:, extended][:, :, extended]
    relabelled_down = down[:, extended][:, :, extended]
    relabelled_graph = build_socket_topk_graph(relabelled_right, relabelled_down, top_k=3)
    with torch.no_grad():
        relabelled = model(
            tile_features[:, permutation],
            relabelled_graph,
            relabelled_layout,
            0.3,
            grid=grid,
        )
    assert torch.allclose(
        relabelled.proposed_coordinates,
        reference.proposed_coordinates[:, permutation],
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(
        relabelled.slot_log_assignment,
        reference.slot_log_assignment[:, permutation],
        atol=2e-5,
        rtol=2e-5,
    )


def test_zero_flow_is_identity_hungarian_and_exact_loss_has_gradient() -> None:
    grid = 3
    count = grid * grid
    right, down = _assignments(count)
    graph = build_socket_topk_graph(right, down, top_k=3)
    model = SocketPermutationFlow(
        tile_feature_dimension=10,
        dimension=16,
        layers=2,
        coordinate_bands=2,
        time_bands=2,
        sinkhorn_iterations=12,
    )
    current = torch.tensor([[7, 2, 5, 1, 0, 6, 3, 8, 4]])
    target = torch.arange(count).reshape(1, count)
    output = model(torch.randn(1, count, 10), graph, current, 0.0, grid=grid)
    assert torch.equal(hungarian_layout(output), current)
    loss, diagnostics = permutation_flow_loss(output, target, grid=grid)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.flow_head.weight.grad is not None
    assert float(model.flow_head.weight.grad.abs().sum()) > 1e-5
    assert diagnostics["assignment_nll"] > 0
    assert torch.equal(tile_positions(target, grid=grid), target)


def test_frozen_extractor_reproduces_matcher_transport_and_feature_shape() -> None:
    torch.manual_seed(47)
    matcher = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=4,
    ).eval()
    tiles = torch.rand(1, 9, 3, 20, 20)
    with torch.no_grad():
        expected = matcher(tiles, grid=3)
        evidence = extract_frozen_socket_evidence(matcher, tiles, grid=3, top_k=3)
    assert evidence.tile_features.shape == (1, 9, 5 * matcher.dimension)
    assert evidence.graph.indices.shape == (1, 9, 4, 3)
    assert evidence.graph.log_scores.shape == evidence.graph.indices.shape
    assert torch.allclose(evidence.right_log_assignment, expected.right_log_assignment)
    assert torch.allclose(evidence.down_log_assignment, expected.down_log_assignment)
    assert not evidence.tile_features.requires_grad
