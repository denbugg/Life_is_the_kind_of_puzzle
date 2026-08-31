from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.sparse_bordergraph_qap import (
    SparseBorderGraphQAP,
    decode_hungarian,
    layout_to_probability,
    log_sinkhorn,
    qap_training_loss,
    sparse_quadratic_energy,
    sparse_quadratic_message,
)


def _perfect_graph(grid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    targets: list[int] = []
    directions: list[int] = []
    for slot in range(grid * grid):
        if slot % grid < grid - 1:
            sources.append(slot)
            targets.append(slot + 1)
            directions.append(0)
        if slot < grid * (grid - 1):
            sources.append(slot)
            targets.append(slot + grid)
            directions.append(1)
    return tuple(torch.tensor(value, dtype=torch.long) for value in (sources, targets, directions))


def test_sparse_energy_and_message_prefer_the_perfect_grid() -> None:
    grid = 3
    count = grid * grid
    sources, targets, directions = _perfect_graph(grid)
    weights = torch.ones(len(sources))
    truth = layout_to_probability(np.arange(count), grid=grid, device=torch.device("cpu"))
    wrong = layout_to_probability(
        np.roll(np.arange(count), 1),
        grid=grid,
        device=torch.device("cpu"),
    )
    truth_energy = sparse_quadratic_energy(
        truth,
        sources,
        targets,
        directions,
        weights,
        grid=grid,
    )
    wrong_energy = sparse_quadratic_energy(
        wrong,
        sources,
        targets,
        directions,
        weights,
        grid=grid,
    )
    assert float(truth_energy) == 12.0
    assert float(truth_energy) > float(wrong_energy)
    message = sparse_quadratic_message(
        truth,
        sources,
        targets,
        directions,
        weights,
        grid=grid,
        normalizer=1.0,
    )
    assert message.shape == (count, count)
    assert torch.all(message.diag() > 0)


def test_sinkhorn_and_hungarian_are_bijective() -> None:
    logits = torch.eye(4) * 8.0
    probability = log_sinkhorn(logits, iterations=12, temperature=0.5)
    assert torch.allclose(probability.sum(0), torch.ones(4), atol=1e-4)
    assert torch.allclose(probability.sum(1), torch.ones(4), atol=1e-4)
    assert np.array_equal(decode_hungarian(probability), np.arange(4))


def test_zero_unary_qap_preserves_strict_baseline_and_backpropagates() -> None:
    grid = 3
    count = grid * grid
    sources, targets, directions = _perfect_graph(grid)
    edge_features = torch.zeros((len(sources), 5))
    edge_features[:, 0] = 1.0
    tile_features = torch.randn(count, 7)
    layout = np.random.default_rng(7).permutation(count).astype(np.int32)
    model = SparseBorderGraphQAP(
        7,
        5,
        hidden_dimension=16,
        edge_hidden_dimension=8,
        max_grid=3,
        unrolled_steps=2,
        sinkhorn_iterations=8,
        baseline_anchor=8.0,
    )
    zero_modules = (
        model.tile_encoder,
        model.row_embedding,
        model.column_embedding,
        model.coordinate_encoder,
    )
    for module in zero_modules:
        for parameter in module.parameters():
            torch.nn.init.zeros_(parameter)
    output = model(
        tile_features,
        edge_features,
        sources,
        targets,
        directions,
        layout,
        grid=grid,
    )
    assert np.array_equal(decode_hungarian(output.final_logits), layout)
    loss, diagnostics = qap_training_loss(
        output,
        layout,
        layout,
        sources,
        targets,
        directions,
        grid=grid,
    )
    assert torch.isfinite(loss)
    assert diagnostics["edge_positive_fraction"] > 0
    loss.backward()
    assert model.edge_head.weight.grad is not None


def test_input_relabelling_equivariance_with_frozen_baseline() -> None:
    grid = 2
    count = 4
    sources, targets, directions = _perfect_graph(grid)
    features = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    edge_features = torch.ones((len(sources), 2))
    baseline = np.array([0, 1, 2, 3], dtype=np.int32)
    model = SparseBorderGraphQAP(
        3,
        2,
        hidden_dimension=8,
        edge_hidden_dimension=8,
        max_grid=2,
        unrolled_steps=1,
        baseline_anchor=8.0,
    )
    model.eval()
    first = model(
        features,
        edge_features,
        sources,
        targets,
        directions,
        baseline,
        grid=grid,
    )
    first_layout = decode_hungarian(first.final_logits)

    old_to_new = np.array([2, 0, 3, 1], dtype=np.int64)
    new_to_old = np.argsort(old_to_new)
    relabelled_features = features[torch.from_numpy(new_to_old)]
    relabelled_sources = torch.from_numpy(old_to_new)[sources]
    relabelled_targets = torch.from_numpy(old_to_new)[targets]
    relabelled_baseline = old_to_new[baseline]
    second = model(
        relabelled_features,
        edge_features,
        relabelled_sources,
        relabelled_targets,
        directions,
        relabelled_baseline,
        grid=grid,
    )
    second_layout = decode_hungarian(second.final_logits)
    assert np.array_equal(new_to_old[second_layout], first_layout)
