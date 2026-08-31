"""Bidirectional listwise reranker for the immutable raw32/twin32 edge union.

Frozen d64 Socket context and the frozen full-resolution twin matcher provide
target-free edge features.  The learned head is permutation equivariant over
outgoing rows, incoming columns and board axes.  Its zero-initialised residual
preserves raw d64 ordering on every union row.  Outside-union edges are always
forbidden before partial optimal transport and exact one-to-one projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.fullres_twin_side_matcher import OPPOSITE_SIDE, TwinSideOutput
from aiijc_puzzle.socket_decoder import hard_partial_axis_matching
from aiijc_puzzle.socket_matcher import (
    SocketOutput,
    partial_log_optimal_transport,
)

SCALAR_FEATURE_NAMES = (
    "raw_score_z",
    "twin_score_z",
    "raw_outgoing_rank_inv",
    "raw_incoming_rank_inv",
    "twin_outgoing_rank_inv",
    "twin_incoming_rank_inv",
    "raw_row_competitor_margin",
    "raw_column_competitor_margin",
    "twin_row_competitor_margin",
    "twin_column_competitor_margin",
    "in_raw_top32",
    "in_twin_top32",
    "in_frozen_raw_hard_projection",
    "raw_reciprocal_top1",
    "twin_reciprocal_top1",
    "sequence_dot_mean",
    "sequence_dot_standard_deviation",
    "sequence_dot_minimum",
    "sequence_dot_maximum",
    "source_tangent_variation",
    "target_tangent_variation",
    "outgoing_border_z",
    "incoming_border_z",
    "direction_down",
)
TOKEN_INTERACTION_NAMES = tuple(
    f"{operation}_token_{dimension}"
    for operation in ("source", "target", "absolute_difference", "product")
    for dimension in range(64)
)
FEATURE_NAMES = SCALAR_FEATURE_NAMES + TOKEN_INTERACTION_NAMES


@dataclass(frozen=True)
class RawTwinUnionBoard:
    """One target-free board candidate tensor and immutable edge identities."""

    values: torch.Tensor
    raw_scores: torch.Tensor
    axis: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    rows: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]
    grid: int


@dataclass(frozen=True)
class UnionRerankerOutput:
    """Candidate score and bounded residual vectors."""

    scores: torch.Tensor
    residual: torch.Tensor


def _numpy_matrix(value: torch.Tensor, *, count: int, name: str) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (count, count) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {(count, count)}")
    return np.asarray(array, dtype=np.float32)


def _standardise_matrix(value: np.ndarray) -> np.ndarray:
    count = len(value)
    selected = value[~np.eye(count, dtype=bool)]
    return np.ascontiguousarray(
        (value - float(selected.mean())) / max(float(selected.std()), 1e-6),
        dtype=np.float32,
    )


def _ranks_and_margins(value: np.ndarray) -> tuple[np.ndarray, ...]:
    count = len(value)
    masked = np.asarray(value, dtype=np.float32).copy()
    np.fill_diagonal(masked, -np.inf)
    row_order = np.argsort(-masked, axis=1, kind="stable")
    column_order = np.argsort(-masked, axis=0, kind="stable")
    row_rank = np.empty((count, count), dtype=np.int32)
    column_rank = np.empty((count, count), dtype=np.int32)
    ranks = np.arange(count, dtype=np.int32)
    row_rank[np.arange(count)[:, None], row_order] = ranks[None, :]
    column_rank[column_order, np.arange(count)[None, :]] = ranks[:, None]
    scale = max(float(value[~np.eye(count, dtype=bool)].std()), 1e-6)
    row_best = masked[np.arange(count), row_order[:, 0]]
    row_second = masked[np.arange(count), row_order[:, 1]]
    row_competitor = np.broadcast_to(row_best[:, None], masked.shape).copy()
    row_competitor[np.arange(count), row_order[:, 0]] = row_second
    column_best = masked[column_order[0], np.arange(count)]
    column_second = masked[column_order[1], np.arange(count)]
    column_competitor = np.broadcast_to(column_best[None, :], masked.shape).copy()
    column_competitor[column_order[0], np.arange(count)] = column_second
    return (
        row_rank,
        column_rank,
        np.asarray((value - row_competitor) / scale, dtype=np.float32),
        np.asarray((value - column_competitor) / scale, dtype=np.float32),
    )


def _topk(value: np.ndarray, *, k: int) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float32).copy()
    if not 1 <= k < len(scores):
        raise ValueError("topk must be in [1, count - 1]")
    np.fill_diagonal(scores, -np.inf)
    return np.ascontiguousarray(
        np.argsort(-scores, axis=1, kind="stable")[:, :k],
        dtype=np.int32,
    )


def _border_vectors(
    output: SocketOutput,
    *,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == 0:
        outgoing = output.right_out_border_logits
        incoming = output.left_in_border_logits
    elif direction == 1:
        outgoing = output.bottom_out_border_logits
        incoming = output.top_in_border_logits
    else:
        raise ValueError("direction must be zero/right or one/down")
    first = outgoing.detach().float().cpu().numpy().reshape(-1)
    second = incoming.detach().float().cpu().numpy().reshape(-1)
    first = (first - first.mean()) / max(float(first.std()), 1e-6)
    second = (second - second.mean()) / max(float(second.std()), 1e-6)
    return np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)


def _sequence_statistics(
    sides: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    direction: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        source_index = source[start:stop]
        target_index = target[start:stop]
        selected_direction = direction[start:stop]
        source_tokens = sides[source_index, selected_direction]
        opposite = torch.as_tensor(
            OPPOSITE_SIDE,
            device=direction.device,
            dtype=torch.long,
        )[selected_direction]
        target_tokens = sides[target_index, opposite]
        dot = (source_tokens * target_tokens).sum(dim=2)
        source_tangent = torch.diff(source_tokens, dim=1).square().mean(dim=(1, 2))
        target_tangent = torch.diff(target_tokens, dim=1).square().mean(dim=(1, 2))
        rows.append(
            torch.stack(
                (
                    dot.mean(1),
                    dot.std(1, unbiased=False),
                    dot.amin(1),
                    dot.amax(1),
                    source_tangent,
                    target_tangent,
                ),
                dim=1,
            )
        )
    return torch.cat(rows, dim=0)


@torch.no_grad()
def prepare_raw_twin_union_board(
    tile_tokens: torch.Tensor,
    socket_output: SocketOutput,
    twin_output: TwinSideOutput,
    *,
    grid: int,
    topk: int = 32,
) -> RawTwinUnionBoard:
    """Build raw32 union twin32 features without targets or shuffled indices."""

    count = grid * grid
    if tile_tokens.ndim == 3 and tile_tokens.shape[0] == 1:
        tile_tokens = tile_tokens[0]
    if tile_tokens.shape != (count, 64):
        raise ValueError("tile_tokens must have shape grid**2 x 64")
    if twin_output.sides.shape != (1, count, 4, 20, 48):
        raise ValueError("twin side tensor violates the frozen 48-D contract")
    k = min(topk, count - 1)
    raw_matrices = (
        _numpy_matrix(socket_output.right_raw, count=count, name="right_raw"),
        _numpy_matrix(socket_output.down_raw, count=count, name="down_raw"),
    )
    twin_matrices = (
        _numpy_matrix(twin_output.scores[:, 1], count=count, name="twin_right"),
        _numpy_matrix(twin_output.scores[:, 3], count=count, name="twin_down"),
    )
    scalar_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    axis_rows: list[np.ndarray] = []
    candidate_rows: list[tuple[np.ndarray, ...]] = []
    raw_value_rows: list[np.ndarray] = []
    for axis, (raw, twin) in enumerate(zip(raw_matrices, twin_matrices, strict=True)):
        raw_z = _standardise_matrix(raw)
        twin_z = _standardise_matrix(twin)
        raw_rank, raw_column_rank, raw_row_margin, raw_column_margin = _ranks_and_margins(raw)
        twin_rank, twin_column_rank, twin_row_margin, twin_column_margin = _ranks_and_margins(twin)
        raw_top = _topk(raw, k=k)
        twin_top = _topk(twin, k=k)
        raw_assignment = (
            socket_output.right_log_assignment if axis == 0 else socket_output.down_log_assignment
        )
        raw_matching = hard_partial_axis_matching(
            raw_assignment,
            grid=grid,
            axis="right" if axis == 0 else "down",
        )
        raw_hard_by_source = {edge.source: edge.target for edge in raw_matching.edges}
        outgoing_border, incoming_border = _border_vectors(socket_output, direction=axis)
        per_axis_rows: list[np.ndarray] = []
        for source in range(count):
            additions = np.concatenate(
                (
                    raw_top[source],
                    twin_top[source],
                    np.asarray(
                        [raw_hard_by_source[source]] if source in raw_hard_by_source else [],
                        dtype=np.int32,
                    ),
                )
            )
            union = np.unique(additions).astype(np.int32)
            order = np.lexsort((union, -raw[source, union]))
            per_axis_rows.append(np.ascontiguousarray(union[order], dtype=np.int32))
        candidate_rows.append(tuple(per_axis_rows))
        source = np.concatenate(
            [np.full(len(row), index, dtype=np.int32) for index, row in enumerate(per_axis_rows)]
        )
        target = np.concatenate(per_axis_rows)
        raw_membership = np.concatenate(
            [np.isin(row, raw_top[index]) for index, row in enumerate(per_axis_rows)]
        ).astype(np.float32)
        twin_membership = np.concatenate(
            [np.isin(row, twin_top[index]) for index, row in enumerate(per_axis_rows)]
        ).astype(np.float32)
        hard_membership = np.asarray(
            [
                raw_hard_by_source.get(int(anchor), -1) == int(candidate)
                for anchor, candidate in zip(source, target, strict=True)
            ],
            dtype=np.float32,
        )
        scalar = np.stack(
            (
                raw_z[source, target],
                twin_z[source, target],
                1.0 / (1.0 + raw_rank[source, target]),
                1.0 / (1.0 + raw_column_rank[source, target]),
                1.0 / (1.0 + twin_rank[source, target]),
                1.0 / (1.0 + twin_column_rank[source, target]),
                raw_row_margin[source, target],
                raw_column_margin[source, target],
                twin_row_margin[source, target],
                twin_column_margin[source, target],
                raw_membership,
                twin_membership,
                hard_membership,
                ((raw_rank[source, target] == 0) & (raw_column_rank[source, target] == 0)),
                ((twin_rank[source, target] == 0) & (twin_column_rank[source, target] == 0)),
                np.zeros(len(source), dtype=np.float32),
                np.zeros(len(source), dtype=np.float32),
                np.zeros(len(source), dtype=np.float32),
                np.zeros(len(source), dtype=np.float32),
                np.zeros(len(source), dtype=np.float32),
                np.zeros(len(source), dtype=np.float32),
                outgoing_border[source],
                incoming_border[target],
                np.full(len(source), axis, dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float32)
        scalar_rows.append(scalar)
        source_rows.append(source)
        target_rows.append(target)
        axis_rows.append(np.full(len(source), axis, dtype=np.int64))
        raw_value_rows.append(raw[source, target])
    device = tile_tokens.device
    source = torch.from_numpy(np.concatenate(source_rows)).to(device=device, dtype=torch.long)
    target = torch.from_numpy(np.concatenate(target_rows)).to(device=device, dtype=torch.long)
    axis = torch.from_numpy(np.concatenate(axis_rows)).to(device=device, dtype=torch.long)
    physical_direction = torch.where(axis == 0, 1, 3)
    sequence = _sequence_statistics(
        twin_output.sides[0],
        source,
        target,
        physical_direction,
    )
    scalar = torch.from_numpy(np.concatenate(scalar_rows)).to(
        device=device,
        dtype=tile_tokens.dtype,
    )
    scalar[:, 15:21] = sequence.to(dtype=scalar.dtype)
    source_token = tile_tokens[source]
    target_token = tile_tokens[target]
    token = torch.cat(
        (
            source_token,
            target_token,
            torch.abs(source_token - target_token),
            source_token * target_token,
        ),
        dim=1,
    )
    values = torch.cat((scalar, token), dim=1)
    raw_scores = torch.from_numpy(np.concatenate(raw_value_rows)).to(
        device=device,
        dtype=tile_tokens.dtype,
    )
    if values.shape != (len(source), len(FEATURE_NAMES)):
        raise RuntimeError("raw/twin union feature dimension invariant failed")
    if not bool(torch.isfinite(values).all().item()) or not bool(
        torch.isfinite(raw_scores).all().item()
    ):
        raise RuntimeError("raw/twin union features contain non-finite values")
    return RawTwinUnionBoard(
        values=values,
        raw_scores=raw_scores,
        axis=axis,
        source=source,
        target=target,
        rows=(candidate_rows[0], candidate_rows[1]),
        grid=grid,
    )


def _group_summary(
    embedded: torch.Tensor,
    group: torch.Tensor,
    *,
    group_count: int,
) -> torch.Tensor:
    dimension = embedded.shape[1]
    total = embedded.new_zeros((group_count, dimension))
    total.index_add_(0, group, embedded)
    count = embedded.new_zeros(group_count)
    count.index_add_(0, group, torch.ones_like(group, dtype=embedded.dtype))
    mean = total / count.clamp_min(1.0).unsqueeze(1)
    maximum = embedded.new_full((group_count, dimension), -torch.inf)
    maximum.scatter_reduce_(
        0,
        group[:, None].expand(-1, dimension),
        embedded,
        reduce="amax",
        include_self=True,
    )
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean[group], maximum[group]), dim=1)


class RawTwinUnionReranker(nn.Module):
    """Row/column/axis-context DeepSets residual over frozen raw d64 scores."""

    def __init__(
        self,
        feature_dimension: int = len(FEATURE_NAMES),
        *,
        hidden_dimension: int = 64,
        residual_limit: float = 2.0,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden_dimension <= 0 or residual_limit <= 0:
            raise ValueError("feature/hidden dimensions and residual limit must be positive")
        self.feature_dimension = int(feature_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.residual_limit = float(residual_limit)
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Linear(feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(7 * hidden_dimension),
            nn.Linear(7 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, 1),
        )
        final = self.residual_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, board: RawTwinUnionBoard) -> UnionRerankerOutput:
        if board.values.ndim != 2 or board.values.shape[1] != self.feature_dimension:
            raise ValueError("board values violate the v2 feature contract")
        count = board.grid * board.grid
        embedded = self.edge_encoder(board.values)
        row_group = board.axis * count + board.source
        column_group = board.axis * count + board.target
        row = _group_summary(embedded, row_group, group_count=2 * count)
        column = _group_summary(embedded, column_group, group_count=2 * count)
        by_axis = _group_summary(embedded, board.axis, group_count=2)
        hidden = torch.cat((embedded, row, column, by_axis), dim=1)
        residual = self.residual_limit * torch.tanh(self.residual_head(hidden).squeeze(1))
        return UnionRerankerOutput(scores=board.raw_scores + residual, residual=residual)


def union_edge_labels(
    board: RawTwinUnionBoard,
    tile_at_position: torch.Tensor,
) -> torch.Tensor:
    """Return exact directed-neighbour labels for training/scoring only."""

    count = board.grid * board.grid
    layout = tile_at_position.long().reshape(-1)
    if layout.shape != (count,) or not torch.equal(
        layout.sort().values,
        torch.arange(count, device=layout.device),
    ):
        raise ValueError("tile_at_position must be one strict grid permutation")
    truth = torch.full((2, count), -1, device=layout.device, dtype=torch.long)
    position = torch.arange(count, device=layout.device)
    right_valid = position % board.grid != board.grid - 1
    down_valid = position < count - board.grid
    truth[0].scatter_(0, layout[right_valid], layout[position[right_valid] + 1])
    truth[1].scatter_(
        0,
        layout[down_valid],
        layout[position[down_valid] + board.grid],
    )
    return truth[board.axis, board.source] == board.target


def _segment_logsumexp(
    values: torch.Tensor,
    groups: torch.Tensor,
    *,
    group_count: int,
) -> torch.Tensor:
    maximum = values.new_full((group_count,), -torch.inf)
    maximum.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
    finite = torch.isfinite(maximum)
    shifted = torch.where(
        finite[groups],
        torch.exp(values - maximum[groups]),
        torch.zeros_like(values),
    )
    total = values.new_zeros(group_count)
    total.index_add_(0, groups, shifted)
    result = maximum + torch.log(total.clamp_min(torch.finfo(values.dtype).tiny))
    return torch.where(finite, result, torch.full_like(result, -torch.inf))


def bidirectional_union_loss(
    output: UnionRerankerOutput,
    board: RawTwinUnionBoard,
    labels: torch.Tensor,
    *,
    pairwise_weight: float = 0.15,
    residual_weight: float = 1e-3,
    hard_negatives_per_axis: int = 512,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Outgoing-row and incoming-column listwise CE plus hard-edge auxiliary."""

    if output.scores.ndim != 1 or labels.shape != output.scores.shape:
        raise ValueError("scores and labels must be aligned vectors")
    if labels.dtype != torch.bool:
        raise ValueError("labels must be boolean")
    if pairwise_weight < 0 or residual_weight < 0 or hard_negatives_per_axis <= 0:
        raise ValueError("loss weights must be non-negative and hard-negative count positive")
    count = board.grid * board.grid
    listwise_terms: list[torch.Tensor] = []
    supervised_groups: list[int] = []
    for group in (
        board.axis * count + board.source,
        board.axis * count + board.target,
    ):
        all_lse = _segment_logsumexp(output.scores, group, group_count=2 * count)
        positive_values = output.scores.masked_fill(~labels, -torch.inf)
        positive_lse = _segment_logsumexp(
            positive_values,
            group,
            group_count=2 * count,
        )
        valid = torch.isfinite(positive_lse)
        listwise_terms.append((all_lse[valid] - positive_lse[valid]).mean())
        supervised_groups.append(int(valid.sum()))
    row_listwise, column_listwise = listwise_terms
    pairwise_terms: list[torch.Tensor] = []
    for axis in (0, 1):
        selected = board.axis == axis
        positive = output.scores[selected & labels]
        negative = output.scores[selected & ~labels]
        if not len(positive) or not len(negative):
            raise ValueError("each axis requires supplied positive and false candidates")
        negative = negative.topk(min(hard_negatives_per_axis, len(negative))).values
        pairwise_terms.append(F.softplus(negative[:, None] - positive[None, :]).mean())
    pairwise = torch.stack(pairwise_terms).mean()
    residual_l2 = output.residual.square().mean()
    loss = (
        0.5 * (row_listwise + column_listwise)
        + pairwise_weight * pairwise
        + residual_weight * residual_l2
    )
    return loss, {
        "loss": float(loss.detach()),
        "row_listwise": float(row_listwise.detach()),
        "column_listwise": float(column_listwise.detach()),
        "hard_pairwise": float(pairwise.detach()),
        "residual_l2": float(residual_l2.detach()),
        "row_supervised_groups": supervised_groups[0],
        "column_supervised_groups": supervised_groups[1],
        "positive_edges": int(labels.sum()),
        "candidate_edges": len(labels),
    }


def candidate_score_matrices(
    board: RawTwinUnionBoard,
    scores: torch.Tensor,
    *,
    fill_value: float = -1e4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse union scores into two dense matrices; outside is forbidden."""

    count = board.grid * board.grid
    if scores.shape != (len(board.values),):
        raise ValueError("candidate scores must align with board rows")
    matrices = scores.new_full((2, count, count), fill_value)
    matrices[board.axis, board.source, board.target] = scores
    diagonal = torch.arange(count, device=scores.device)
    matrices[:, diagonal, diagonal] = fill_value
    return matrices[0].unsqueeze(0), matrices[1].unsqueeze(0)


def restricted_partial_ot(
    board: RawTwinUnionBoard,
    scores: torch.Tensor,
    socket_output: SocketOutput,
    *,
    iterations: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply frozen d64 border logits and partial OT to learned union scores."""

    right, down = candidate_score_matrices(board, scores)
    return (
        partial_log_optimal_transport(
            right,
            socket_output.right_out_border_logits,
            unmatched=board.grid,
            iterations=iterations,
            target_bin_score=socket_output.left_in_border_logits,
        ),
        partial_log_optimal_transport(
            down,
            socket_output.bottom_out_border_logits,
            unmatched=board.grid,
            iterations=iterations,
            target_bin_score=socket_output.top_in_border_logits,
        ),
    )


__all__ = [
    "FEATURE_NAMES",
    "SCALAR_FEATURE_NAMES",
    "RawTwinUnionBoard",
    "RawTwinUnionReranker",
    "UnionRerankerOutput",
    "bidirectional_union_loss",
    "candidate_score_matrices",
    "prepare_raw_twin_union_board",
    "restricted_partial_ot",
    "union_edge_labels",
]
