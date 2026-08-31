from __future__ import annotations

import math

import torch

from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SCORE_STATISTIC_NAMES,
    SocketMatcher,
    SocketOutput,
    partial_log_optimal_transport,
    socket_matching_loss,
    socket_score_statistics,
    socket_targets,
)


def test_partial_transport_respects_exact_dustbin_capacity() -> None:
    scores = torch.randn(2, 16, 16)
    transport = partial_log_optimal_transport(
        scores,
        torch.tensor(0.0),
        unmatched=4,
        iterations=100,
    ).exp()
    expected_real = torch.full((2, 16), 1.0 / 20.0)
    expected_bin = torch.full((2,), 4.0 / 20.0)
    assert torch.allclose(transport.sum(2)[:, :16], expected_real, atol=2e-5)
    assert torch.allclose(transport.sum(1)[:, :16], expected_real, atol=2e-5)
    assert torch.allclose(transport.sum(2)[:, 16], expected_bin, atol=2e-5)
    assert torch.allclose(transport.sum(1)[:, 16], expected_bin, atol=2e-5)
    assert torch.all(transport[:, 16, 16] < 1e-8)


def test_socket_targets_encode_interior_and_border_sides() -> None:
    layout = torch.arange(16).reshape(1, 16)
    targets = socket_targets(layout, grid=4)
    assert targets["right_out"][0].tolist() == [
        1,
        2,
        3,
        16,
        5,
        6,
        7,
        16,
        9,
        10,
        11,
        16,
        13,
        14,
        15,
        16,
    ]
    assert targets["right_in"][0, [0, 4, 8, 12]].tolist() == [16] * 4
    assert targets["right_in"][0, [1, 5, 9, 13]].tolist() == [0, 4, 8, 12]
    assert targets["down_out"][0, :12].tolist() == list(range(4, 16))
    assert targets["down_out"][0, 12:].tolist() == [16] * 4


def test_socket_targets_are_index_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(7)
    layout = torch.randperm(16, generator=generator).reshape(1, 16)
    targets = socket_targets(layout, grid=4)
    for position in range(16):
        tile = int(layout[0, position])
        expected_right = int(layout[0, position + 1]) if position % 4 != 3 else 16
        expected_down = int(layout[0, position + 4]) if position < 12 else 16
        assert int(targets["right_out"][0, tile]) == expected_right
        assert int(targets["down_out"][0, tile]) == expected_down


def test_socket_model_shapes_loss_and_gradient() -> None:
    torch.manual_seed(3)
    model = SocketMatcher(
        dimension=16,
        heads=4,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=4,
    )
    tiles = torch.rand(1, 16, 3, 20, 20)
    layout = torch.randperm(16).reshape(1, 16)
    output = model(tiles, grid=4)
    assert output.right_raw.shape == (1, 16, 16)
    assert output.down_raw.shape == (1, 16, 16)
    assert output.right_log_assignment.shape == (1, 17, 17)
    assert output.down_log_assignment.shape == (1, 17, 17)
    loss, diagnostics = socket_matching_loss(output, layout, grid=4)
    assert torch.isfinite(loss)
    assert diagnostics["right_supervised"] == 32.0
    assert diagnostics["down_supervised"] == 32.0
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    border_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.border_heads.parameters()
        if parameter.grad is not None
    )
    assert border_gradient > 1e-6
    assert math.isfinite(diagnostics["loss"])


def test_score_statistics_are_permutation_equivariant_and_differentiable() -> None:
    torch.manual_seed(29)
    count = 7
    scores = torch.randn(2, count, count, requires_grad=True)
    outgoing = socket_score_statistics(scores, outgoing=True)
    incoming = socket_score_statistics(scores, outgoing=False)
    assert outgoing.shape == (2, count, len(SCORE_STATISTIC_NAMES))
    assert incoming.shape == outgoing.shape
    assert torch.isfinite(outgoing).all() and torch.isfinite(incoming).all()

    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    permuted = scores.detach()[:, permutation][:, :, permutation]
    permuted_outgoing = socket_score_statistics(permuted, outgoing=True)
    permuted_incoming = socket_score_statistics(permuted, outgoing=False)
    assert torch.allclose(permuted_outgoing, outgoing.detach()[:, permutation], atol=5e-6)
    assert torch.allclose(permuted_incoming, incoming.detach()[:, permutation], atol=5e-6)

    socket_weight = torch.linspace(-1.0, 1.0, count).reshape(1, count, 1)
    statistic_weight = torch.arange(1, len(SCORE_STATISTIC_NAMES) + 1).reshape(1, 1, -1)
    objective = ((outgoing + 0.7 * incoming) * socket_weight * statistic_weight).sum()
    objective.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 1e-4


def test_v3_is_explicit_and_v2_warmstart_is_initially_behaviour_preserving() -> None:
    torch.manual_seed(31)
    model_v2 = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    ).eval()
    assert model_v2.border_head_version == BORDER_HEAD_EMBEDDING_V2
    assert not any(name.startswith("border_distribution_heads.") for name in model_v2.state_dict())

    model_v3 = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
        border_head_version=BORDER_HEAD_SCORE_STATS_V3,
    ).eval()
    incompatible = model_v3.load_state_dict(model_v2.state_dict(), strict=False)
    expected_missing = {
        f"border_distribution_heads.{side}.weight"
        for side in ("right", "left", "bottom", "top")
    }
    assert set(incompatible.missing_keys) == expected_missing
    assert not incompatible.unexpected_keys
    assert all(
        torch.count_nonzero(head.weight) == 0
        for head in model_v3.border_distribution_heads.values()
    )

    tiles = torch.rand(1, 9, 3, 20, 20)
    with torch.no_grad():
        output_v2 = model_v2(tiles, grid=3)
        output_v3 = model_v3(tiles, grid=3)
    for field in SocketOutput.__dataclass_fields__:
        assert torch.equal(getattr(output_v2, field), getattr(output_v3, field))


def test_v3_distribution_border_path_is_equivariant_and_receives_gradient() -> None:
    torch.manual_seed(37)
    model = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
        border_head_version=BORDER_HEAD_SCORE_STATS_V3,
    )
    with torch.no_grad():
        weights = torch.tensor([[0.7, -0.4, 0.2, 0.3, -0.1, 0.5]])
        for head in model.border_distribution_heads.values():
            head.weight.copy_(weights)

    tiles = torch.rand(1, 9, 3, 20, 20)
    permutation = torch.tensor([5, 0, 8, 2, 7, 3, 1, 6, 4])
    model.eval()
    with torch.no_grad():
        reference = model(tiles, grid=3)
        shuffled = model(tiles[:, permutation], grid=3)
    for field in (
        "right_out_border_logits",
        "left_in_border_logits",
        "bottom_out_border_logits",
        "top_in_border_logits",
    ):
        assert torch.allclose(
            getattr(shuffled, field),
            getattr(reference, field)[:, permutation],
            atol=2e-5,
            rtol=2e-5,
        )

    model.train()
    layout = torch.randperm(9).reshape(1, 9)
    loss, _ = socket_matching_loss(model(tiles, grid=3), layout, grid=3, border_weight=1.0)
    loss.backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.border_distribution_heads.parameters()
        if parameter.grad is not None
    )
    assert gradient > 1e-5


def test_trusted_mask_removes_untrusted_edges_but_keeps_border_targets() -> None:
    torch.manual_seed(5)
    model = SocketMatcher(
        dimension=16,
        heads=4,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    )
    tiles = torch.rand(1, 16, 3, 20, 20)
    layout = torch.arange(16).reshape(1, 16)
    trusted = torch.zeros_like(layout, dtype=torch.bool)
    trusted[:, :8] = True
    loss, diagnostics = socket_matching_loss(
        model(tiles, grid=4),
        layout,
        grid=4,
        trusted_position=trusted,
    )
    assert torch.isfinite(loss)
    assert 0 < diagnostics["right_supervised"] < 32
    assert 0 < diagnostics["down_supervised"] < 32


def test_raw_rank_auxiliary_is_bidirectional_and_masks_untrusted_tiles() -> None:
    torch.manual_seed(11)
    count = 16
    right_raw = torch.randn(1, count, count, requires_grad=True)
    down_raw = torch.randn(1, count, count, requires_grad=True)
    output = SocketOutput(
        right_raw=right_raw,
        down_raw=down_raw,
        right_log_assignment=torch.zeros(1, count + 1, count + 1),
        down_log_assignment=torch.zeros(1, count + 1, count + 1),
        right_out_border_logits=torch.zeros(1, count),
        left_in_border_logits=torch.zeros(1, count),
        bottom_out_border_logits=torch.zeros(1, count),
        top_in_border_logits=torch.zeros(1, count),
    )
    layout = torch.arange(count).reshape(1, count)
    trusted = torch.zeros_like(layout, dtype=torch.bool)
    trusted[:, :8] = True

    loss, diagnostics = socket_matching_loss(
        output,
        layout,
        grid=4,
        trusted_position=trusted,
        border_weight=0.0,
        raw_rank_weight=1.0,
    )
    loss.backward()

    # Six horizontal and four vertical trusted interior edges are each
    # supervised once from their row and once from their column.
    assert diagnostics["right_raw_rank_supervised"] == 12.0
    assert diagnostics["down_raw_rank_supervised"] == 8.0
    assert diagnostics["raw_rank_nll"] > 0
    assert right_raw.grad is not None and float(right_raw.grad.abs().sum()) > 0
    assert down_raw.grad is not None and float(down_raw.grad.abs().sum()) > 0
    # Untrusted queries and candidates are entirely disconnected from the
    # auxiliary, preventing uncertain recovered labels becoming negatives.
    assert torch.count_nonzero(right_raw.grad[:, 8:, :]) == 0
    assert torch.count_nonzero(right_raw.grad[:, :, 8:]) == 0
    assert torch.count_nonzero(down_raw.grad[:, 8:, :]) == 0
    assert torch.count_nonzero(down_raw.grad[:, :, 8:]) == 0


def test_raw_rank_auxiliary_default_weight_preserves_total_loss() -> None:
    torch.manual_seed(19)
    model = SocketMatcher(
        dimension=16,
        heads=4,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    )
    output = model(torch.rand(1, 16, 3, 20, 20), grid=4)
    layout = torch.randperm(16).reshape(1, 16)
    default_loss, default_diagnostics = socket_matching_loss(output, layout, grid=4)
    zero_loss, zero_diagnostics = socket_matching_loss(
        output,
        layout,
        grid=4,
        raw_rank_weight=0.0,
    )
    assert torch.equal(default_loss, zero_loss)
    assert default_diagnostics == zero_diagnostics
    assert default_diagnostics["raw_rank_weight"] == 0.0


def test_raw_rank_auxiliary_rejects_invalid_weight() -> None:
    model = SocketMatcher(dimension=16, heads=4)
    output = model(torch.rand(1, 4, 3, 20, 20), grid=2)
    layout = torch.arange(4).reshape(1, 4)
    for value in (-0.1, float("nan"), float("inf")):
        try:
            socket_matching_loss(output, layout, grid=2, raw_rank_weight=value)
        except ValueError as error:
            assert "raw_rank_weight" in str(error)
        else:
            raise AssertionError(f"invalid raw rank weight {value} was accepted")
