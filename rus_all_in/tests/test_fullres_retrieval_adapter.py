from __future__ import annotations

import pytest
import torch

from aiijc_puzzle.fullres_retrieval_adapter import (
    FullResolutionRetrievalAdapter,
    retrieval_adapter_contract,
    retrieval_adapter_loss,
)
from aiijc_puzzle.socket_matcher import SocketMatcher


def _frozen_socket() -> SocketMatcher:
    model = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=2,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_zero_initialised_adapter_is_exact_identity_and_has_no_downsampling() -> None:
    model = FullResolutionRetrievalAdapter().eval()
    value = torch.rand(5, 3, 20, 20)
    observed_shapes: list[tuple[int, int]] = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, output: observed_shapes.append(
                tuple(output.shape[-2:])
            )
        )
        for block in model.body
    ]

    with torch.inference_mode():
        output = model(value)
    for hook in hooks:
        hook.remove()

    torch.testing.assert_close(output, value, rtol=0.0, atol=0.0)
    assert observed_shapes == [(20, 20)] * 8
    assert retrieval_adapter_contract(model)["spatial_downsampling"] is False


def test_exact_socket_loss_backpropagates_only_into_adapter() -> None:
    torch.manual_seed(0)
    adapter = FullResolutionRetrievalAdapter().train()
    socket = _frozen_socket()
    dirty = torch.rand(4, 3, 20, 20)
    clean = torch.rand(4, 3, 20, 20)
    layout = torch.arange(4).unsqueeze(0)

    result = retrieval_adapter_loss(
        adapter,
        socket,
        dirty,
        clean,
        layout,
        grid=2,
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in socket.parameters())


def test_loss_fails_closed_if_socket_is_not_frozen_eval() -> None:
    adapter = FullResolutionRetrievalAdapter()
    socket = _frozen_socket().train()
    value = torch.rand(4, 3, 20, 20)

    with pytest.raises(ValueError, match="eval mode"):
        retrieval_adapter_loss(
            adapter,
            socket,
            value,
            value,
            torch.arange(4).unsqueeze(0),
            grid=2,
        )
