from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.component_shift_head import (
    ComponentDescriptor,
    ComponentShiftHead,
    component_descriptors_from_decoder,
    component_shift_loss,
    component_shift_unary,
    dominant_component_shift_targets,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments


def _impure_grid3_components() -> tuple[ComponentDescriptor, ...]:
    first = ComponentDescriptor(
        tiles=(0, 1, 3),
        relative_rows=(0, 0, 1),
        relative_columns=(0, 1, 1),
        confidence=1.5,
    )
    singles = tuple(
        ComponentDescriptor((tile,), (0,), (0,), 0.0)
        for tile in (2, 4, 5, 6, 7, 8)
    )
    return (first, *singles)


def test_component_head_is_member_order_invariant_and_masks_infeasible_shifts() -> None:
    torch.manual_seed(131)
    components = _impure_grid3_components()
    reversed_first = ComponentDescriptor(
        tiles=components[0].tiles[::-1],
        relative_rows=components[0].relative_rows[::-1],
        relative_columns=components[0].relative_columns[::-1],
        confidence=components[0].confidence,
    )
    reordered_members = (reversed_first, *components[1:])
    head = ComponentShiftHead(6, grid=3, hidden_dimension=12).eval()
    tokens = torch.randn(9, 6)
    with torch.no_grad():
        first = head(tokens, components)
        second = head(tokens, reordered_members)
    torch.testing.assert_close(first.row_logits, second.row_logits)
    torch.testing.assert_close(first.column_logits, second.column_logits)
    assert first.feasible_row_shifts[0] == 2
    assert first.feasible_column_shifts[0] == 2
    assert first.row_logits[0, 2] == -1e4
    assert first.column_logits[0, 2] == -1e4


def test_dominant_target_retains_impure_component_and_loss_backpropagates() -> None:
    torch.manual_seed(137)
    components = _impure_grid3_components()
    targets = dominant_component_shift_targets(components, np.arange(9), grid=3)
    assert targets[0].support == 2
    assert targets[0].purity == 2 / 3
    assert (targets[0].target_row_shift, targets[0].target_column_shift) == (0, 0)

    head = ComponentShiftHead(5, grid=3, hidden_dimension=10)
    output = head(torch.randn(9, 5), components)
    loss, diagnostics = component_shift_loss(output, targets)
    assert torch.isfinite(loss)
    assert diagnostics["component_count"] == len(components)
    assert diagnostics["mean_training_weight"] > 0
    assert diagnostics["pure_component_tile_fraction"] < 1
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_component_logits_convert_exactly_to_decoder_unary_contract() -> None:
    torch.manual_seed(139)
    grid = 3
    count = grid * grid
    components = _impure_grid3_components()
    head = ComponentShiftHead(4, grid=grid, hidden_dimension=8).eval()
    output = head(torch.randn(count, 4), components)
    unary = component_shift_unary(output, components, grid=grid)
    assert unary.shape == (count, count)
    assert torch.isfinite(unary).all()

    component = components[0]
    row_log_probability = output.row_logits[0, :2].log_softmax(0)
    column_log_probability = output.column_logits[0, :2].log_softmax(0)
    for row_shift in range(2):
        for column_shift in range(2):
            observed = sum(
                unary[
                    tile,
                    (relative_row + row_shift) * grid
                    + relative_column
                    + column_shift,
                ]
                for tile, relative_row, relative_column in zip(
                    component.tiles,
                    component.relative_rows,
                    component.relative_columns,
                    strict=True,
                )
            )
            expected = row_log_probability[row_shift] + column_log_probability[column_shift]
            torch.testing.assert_close(observed, expected)

    generator = np.random.default_rng(141)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right[count, count] = down[count, count] = -1e4
    decoded = decode_socket_assignments(
        right,
        down,
        grid=grid,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=6,
            swap_edge_budget_per_axis=6,
            max_swap_steps=0,
            component_shift_unary_weight=0.1,
        ),
        component_shift_unary=unary,
    )
    assert np.array_equal(np.sort(decoded.layout), np.arange(count))
    assert decoded.diagnostics.component_shift_unary_used


def test_descriptor_adapter_uses_frozen_decoder_component_partition() -> None:
    grid = 3
    count = grid * grid
    generator = np.random.default_rng(147)
    right = generator.normal(size=(count + 1, count + 1))
    down = generator.normal(size=(count + 1, count + 1))
    right[count, count] = down[count, count] = -1e4
    component_build = rebuild_decoder_components(
        right,
        down,
        grid=grid,
        edge_budget_per_axis=6,
    )
    descriptors = component_descriptors_from_decoder(component_build, grid=grid)
    observed = sorted(tile for component in descriptors for tile in component.tiles)
    assert observed == list(range(count))
    assert all(np.isfinite(component.confidence) for component in descriptors)
    output = ComponentShiftHead(4, grid=grid, hidden_dimension=8)(
        torch.randn(count, 4),
        descriptors,
    )
    assert output.row_logits.shape == (len(descriptors), grid)


def test_tiny_4x4_component_shift_capacity_smoke() -> None:
    torch.manual_seed(149)
    grid = 4
    count = grid * grid
    components = tuple(
        ComponentDescriptor(
            tiles=tuple(range(row * grid, (row + 1) * grid)),
            relative_rows=(0,) * grid,
            relative_columns=tuple(range(grid)),
            confidence=3.0,
        )
        for row in range(grid)
    )
    coordinates = torch.arange(count)
    rows = (coordinates // grid).float() / (grid - 1)
    columns = (coordinates % grid).float() / (grid - 1)
    tokens = torch.stack((rows, columns, rows.square(), columns.square()), dim=1)
    targets = dominant_component_shift_targets(components, np.arange(count), grid=grid)
    head = ComponentShiftHead(4, grid=grid, hidden_dimension=16)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)

    initial = None
    diagnostics: dict[str, float] = {}
    for _ in range(60):
        output = head(tokens, components)
        loss, diagnostics = component_shift_loss(output, targets)
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert initial is not None
    assert float(loss.detach()) < 0.1 * initial
    assert diagnostics["component_shift_argmax_accuracy"] == 1.0
