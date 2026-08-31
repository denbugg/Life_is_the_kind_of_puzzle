from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import aiijc_puzzle.socket_pasha_topk as topk_module
from aiijc_puzzle.socket_pasha_topk import (
    MASKED_PRIORITY,
    decode_socket_with_pasha_topk_priority,
    fuse_socket_pasha_topk_rank_percentiles,
    rerank_socket_topk_with_pasha,
    select_socket_topk_candidates,
)


class _CapturePairModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        self.calls.append(pairs.detach().cpu().clone())
        weights = torch.linspace(0.0, 1.0, pairs.shape[-1], device=pairs.device)
        return (pairs[:, 0, 0] * weights).sum(dim=1)


def _scores(count: int) -> np.ndarray:
    rows = np.arange(count, dtype=np.float32)[:, None]
    columns = np.arange(count, dtype=np.float32)[None, :]
    value = 10.0 * rows + columns
    value[np.arange(count), np.arange(count)] = 1_000_000.0
    return value


def _tiles(count: int) -> np.ndarray:
    tiles = np.zeros((count, 20, 20, 3), dtype=np.uint8)
    for tile in range(count):
        tiles[tile, :, :, 0] = tile * 10
        tiles[tile, 1, :, 0] += np.arange(20, dtype=np.uint8)
        tiles[tile, :, 2, 1] = np.arange(20, dtype=np.uint8)
    return tiles


def test_socket_topk_masks_self_and_caps_production_width() -> None:
    scores = _scores(5)
    selected = select_socket_topk_candidates(scores, top_k=3)
    assert selected.shape == (5, 3)
    assert not np.any(selected == np.arange(5)[:, None])
    assert selected[0].tolist() == [4, 3, 2]
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        select_socket_topk_candidates(np.zeros((40, 40), dtype=np.float32), top_k=33)


def test_fixed_topk_rank_fusion_never_imputes_unscored_pasha_values() -> None:
    socket = _scores(4)
    candidates = np.asarray([[3, 2], [3, 2], [3, 1], [2, 1]], dtype=np.int32)
    pasha = np.asarray([[0.0, 1.0], [2.0, 1.0], [0.0, 1.0], [2.0, 1.0]])
    priority = fuse_socket_pasha_topk_rank_percentiles(socket, candidates, pasha)

    assert priority[0, 3] == pytest.approx(0.5)
    assert priority[0, 2] == pytest.approx(0.5)
    assert priority[1, 3] == pytest.approx(1.0)
    assert priority[1, 2] == pytest.approx(0.0)
    admitted = np.zeros_like(priority, dtype=bool)
    admitted[np.arange(4)[:, None], candidates] = True
    assert np.all(priority[~admitted] == MASKED_PRIORITY)
    assert np.all(np.diag(priority) == MASKED_PRIORITY)


def test_topk_rank_fusion_is_tile_permutation_equivariant_without_ties() -> None:
    generator = np.random.default_rng(123)
    socket = generator.normal(size=(7, 7)).astype(np.float32)
    pasha_full = generator.normal(size=(7, 7)).astype(np.float32)
    candidates = select_socket_topk_candidates(socket, top_k=3)
    rows = np.arange(7)[:, None]
    original = fuse_socket_pasha_topk_rank_percentiles(
        socket,
        candidates,
        pasha_full[rows, candidates],
    )

    permutation = np.asarray([4, 0, 6, 2, 5, 1, 3])
    permuted_socket = socket[np.ix_(permutation, permutation)]
    permuted_pasha = pasha_full[np.ix_(permutation, permutation)]
    permuted_candidates = select_socket_topk_candidates(permuted_socket, top_k=3)
    observed = fuse_socket_pasha_topk_rank_percentiles(
        permuted_socket,
        permuted_candidates,
        permuted_pasha[rows, permuted_candidates],
    )
    assert np.array_equal(observed, original[np.ix_(permutation, permutation)])


def test_sparse_scorer_evaluates_only_selected_pairs_and_transposes_vertical() -> None:
    count = 4
    top_k = 2
    tiles = _tiles(count)
    right = _scores(count)
    down = np.flip(_scores(count), axis=1).copy()
    model = _CapturePairModel().eval()
    result = rerank_socket_topk_with_pasha(
        model,
        tiles,
        right,
        down,
        device=torch.device("cpu"),
        top_k=top_k,
        batch_size=count * top_k,
    )

    assert result.pair_evaluations == 2 * count * top_k
    assert len(model.calls) == 2
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float() / 255.0
    sources = torch.arange(count).repeat_interleave(top_k)
    right_targets = torch.from_numpy(result.right_candidates.reshape(-1).astype(np.int64))
    down_targets = torch.from_numpy(result.down_candidates.reshape(-1).astype(np.int64))
    expected_right = torch.cat((tensor[sources], tensor[right_targets]), dim=-1)
    transposed = tensor.transpose(-1, -2)
    expected_down = torch.cat((transposed[sources], transposed[down_targets]), dim=-1)
    assert torch.equal(model.calls[0], expected_right)
    assert torch.equal(model.calls[1], expected_down)
    assert np.all(np.diag(result.right_priority) == MASKED_PRIORITY)


def test_decoder_integration_uses_priority_without_replacing_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 4
    right = np.zeros((count + 1, count + 1), dtype=np.float32)
    down = np.zeros_like(right)
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_decode(
        right_assignment: object,
        down_assignment: object,
        **kwargs: object,
    ) -> object:
        captured.update(
            right=right_assignment,
            down=down_assignment,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(topk_module, "decode_socket_assignments", fake_decode)
    result = decode_socket_with_pasha_topk_priority(
        _CapturePairModel().eval(),
        _tiles(count),
        right,
        down,
        device=torch.device("cpu"),
        grid=2,
        top_k=2,
    )
    assert result.decoder is sentinel
    assert captured["right"] is right and captured["down"] is down
    priority = captured["component_edge_priority"]
    assert isinstance(priority, dict)
    assert set(priority) == {"right", "down"}
    assert np.all(np.diag(priority["right"]) == MASKED_PRIORITY)
