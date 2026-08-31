"""Learned context fusion for raw and full-resolution-restored relation supply.

The frozen component-relation model is useful but only sees the raw Socket
view.  The full-resolution denoiser supplies additional true neighbours, while
direct score averaging is harmful.  This module therefore builds target-free
features for the *union* candidate roster and learns two outputs:

* a bounded residual over the frozen component-relation score;
* a correctness confidence used only to order whole relation queries.

Exact synthetic targets are accepted by the loss/metrics outside this module;
candidate construction and feature extraction never receive them.  Restored
pixels remain a matcher-only view and are never legal output material.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.component_relation_reranker import (
    DIRECTION_TO_INDEX,
    DIRECTIONS,
    ComponentRelationCandidate,
    RelationCandidateLabel,
)
from aiijc_puzzle.component_shift_head import ComponentDescriptor
from aiijc_puzzle.socket_matcher import SocketOutput


@dataclass(frozen=True)
class FusionOutput:
    """Candidate ranking scores and independently learned correctness logits."""

    scores: torch.Tensor
    confidence_logits: torch.Tensor


@dataclass(frozen=True)
class _AxisEvidence:
    raw_z: np.ndarray
    ot_z: np.ndarray
    row_rank: np.ndarray
    column_rank: np.ndarray
    row_margin: np.ndarray
    column_margin: np.ndarray
    outgoing_border_z: np.ndarray
    incoming_border_z: np.ndarray


@dataclass(frozen=True)
class _DescriptorEvidence:
    score_z: np.ndarray
    row_rank: np.ndarray
    column_rank: np.ndarray
    row_margin: np.ndarray
    column_margin: np.ndarray


def _numpy(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == len(shape) + 1 and result.shape[0] == 1:
        result = result[0]
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}, got {result.shape}")
    return result


def _standardise(value: np.ndarray, *, ignore_diagonal: bool = True) -> np.ndarray:
    selected = value[~np.eye(len(value), dtype=bool)] if ignore_diagonal else value
    scale = max(float(selected.std()), 1e-6)
    return (value - float(selected.mean())) / scale


def _ranks_and_margins(value: np.ndarray) -> tuple[np.ndarray, ...]:
    count = len(value)
    masked = value.copy()
    masked[np.arange(count), np.arange(count)] = -np.inf
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
        (value - row_competitor) / scale,
        (value - column_competitor) / scale,
    )


def _axis_evidence(
    output: SocketOutput,
    *,
    axis: str,
    count: int,
) -> _AxisEvidence:
    if axis == "right":
        raw_value = output.right_raw
        ot_value = output.right_log_assignment
        outgoing_value = output.right_out_border_logits
        incoming_value = output.left_in_border_logits
    elif axis == "down":
        raw_value = output.down_raw
        ot_value = output.down_log_assignment
        outgoing_value = output.bottom_out_border_logits
        incoming_value = output.top_in_border_logits
    else:
        raise ValueError("axis must be right or down")
    raw = _numpy(raw_value, shape=(count, count), name=f"{axis}_raw")
    ot_full = _numpy(
        ot_value,
        shape=(count + 1, count + 1),
        name=f"{axis}_log_assignment",
    )
    outgoing = _numpy(outgoing_value, shape=(count,), name=f"{axis}_out_border")
    incoming = _numpy(incoming_value, shape=(count,), name=f"{axis}_in_border")
    row_rank, column_rank, row_margin, column_margin = _ranks_and_margins(raw)
    return _AxisEvidence(
        raw_z=_standardise(raw),
        ot_z=_standardise(ot_full[:count, :count]),
        row_rank=row_rank,
        column_rank=column_rank,
        row_margin=row_margin,
        column_margin=column_margin,
        outgoing_border_z=(outgoing - outgoing.mean()) / max(float(outgoing.std()), 1e-6),
        incoming_border_z=(incoming - incoming.mean()) / max(float(incoming.std()), 1e-6),
    )


def _descriptor_evidence(value: Any, *, count: int, name: str) -> _DescriptorEvidence:
    score = _numpy(value, shape=(count, count), name=name)
    row_rank, column_rank, row_margin, column_margin = _ranks_and_margins(score)
    return _DescriptorEvidence(
        score_z=_standardise(score),
        row_rank=row_rank,
        column_rank=column_rank,
        row_margin=row_margin,
        column_margin=column_margin,
    )


def _directed_indices(
    direction: str,
    source_tile: int,
    target_tile: int,
) -> tuple[str, bool, int, int]:
    if direction == "right":
        return "right", True, source_tile, target_tile
    if direction == "left":
        return "right", False, target_tile, source_tile
    if direction == "down":
        return "down", True, source_tile, target_tile
    if direction == "up":
        return "down", False, target_tile, source_tile
    raise ValueError(f"unknown direction {direction!r}")


def _oriented_socket_features(
    evidence: _AxisEvidence,
    *,
    forward: bool,
    outgoing: int,
    incoming: int,
) -> tuple[float, ...]:
    if forward:
        source_margin = evidence.row_margin[outgoing, incoming]
        target_margin = evidence.column_margin[outgoing, incoming]
        source_rank = evidence.row_rank[outgoing, incoming]
        target_rank = evidence.column_rank[outgoing, incoming]
        source_border = evidence.outgoing_border_z[outgoing]
        target_border = evidence.incoming_border_z[incoming]
    else:
        source_margin = evidence.column_margin[outgoing, incoming]
        target_margin = evidence.row_margin[outgoing, incoming]
        source_rank = evidence.column_rank[outgoing, incoming]
        target_rank = evidence.row_rank[outgoing, incoming]
        source_border = evidence.incoming_border_z[incoming]
        target_border = evidence.outgoing_border_z[outgoing]
    return (
        float(evidence.raw_z[outgoing, incoming]),
        float(evidence.ot_z[outgoing, incoming]),
        float(source_margin),
        float(target_margin),
        1.0 / (1.0 + int(source_rank)),
        1.0 / (1.0 + int(target_rank)),
        float(source_border),
        float(target_border),
    )


def _oriented_descriptor_features(
    evidence: _DescriptorEvidence,
    *,
    forward: bool,
    outgoing: int,
    incoming: int,
) -> tuple[float, ...]:
    if forward:
        source_margin = evidence.row_margin[outgoing, incoming]
        target_margin = evidence.column_margin[outgoing, incoming]
        source_rank = evidence.row_rank[outgoing, incoming]
        target_rank = evidence.column_rank[outgoing, incoming]
    else:
        source_margin = evidence.column_margin[outgoing, incoming]
        target_margin = evidence.row_margin[outgoing, incoming]
        source_rank = evidence.column_rank[outgoing, incoming]
        target_rank = evidence.row_rank[outgoing, incoming]
    return (
        float(evidence.score_z[outgoing, incoming]),
        float(source_margin),
        float(target_margin),
        1.0 / (1.0 + int(source_rank)),
        1.0 / (1.0 + int(target_rank)),
    )


def _cosine_rows(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norms, 1e-8)


def preserve_raw_union_candidates(
    raw_candidates: Sequence[ComponentRelationCandidate],
    expanded_candidates: Sequence[ComponentRelationCandidate],
    *,
    max_candidates_per_query: int,
) -> tuple[ComponentRelationCandidate, ...]:
    """Keep every raw relation and fill remaining query capacity from restored supply."""

    if max_candidates_per_query <= 0:
        raise ValueError("max_candidates_per_query must be positive")
    raw_by_query: dict[tuple[int, str], list[ComponentRelationCandidate]] = defaultdict(list)
    expanded_by_query: dict[tuple[int, str], list[ComponentRelationCandidate]] = defaultdict(list)
    for candidate in raw_candidates:
        raw_by_query[candidate.query_key].append(candidate)
    for candidate in expanded_candidates:
        expanded_by_query[candidate.query_key].append(candidate)
    result: list[ComponentRelationCandidate] = []
    for query in sorted(
        set(raw_by_query) | set(expanded_by_query),
        key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]]),
    ):
        raw = raw_by_query.get(query, [])
        if len(raw) > max_candidates_per_query:
            raise ValueError("union cap is smaller than the frozen raw roster")
        chosen = list(raw)
        keys = {candidate.relation_key for candidate in chosen}
        for candidate in expanded_by_query.get(query, []):
            if candidate.relation_key in keys:
                continue
            chosen.append(candidate)
            keys.add(candidate.relation_key)
            if len(chosen) == max_candidates_per_query:
                break
        result.extend(chosen)
    raw_keys = {candidate.relation_key for candidate in raw_candidates}
    union_keys = {candidate.relation_key for candidate in result}
    if not raw_keys <= union_keys:
        raise RuntimeError("raw candidate preservation invariant failed")
    return tuple(result)


_RAW_CONTACT_NAMES = (
    "raw_socket_z",
    "raw_ot_z",
    "raw_source_margin",
    "raw_target_margin",
    "raw_source_reciprocal_rank",
    "raw_target_reciprocal_rank",
    "raw_source_border_z",
    "raw_target_border_z",
)
_RESTORED_CONTACT_NAMES = tuple(name.replace("raw_", "restored_", 1) for name in _RAW_CONTACT_NAMES)
_DESCRIPTOR_CONTACT_NAMES = (
    "descriptor_z",
    "descriptor_source_margin",
    "descriptor_target_margin",
    "descriptor_source_reciprocal_rank",
    "descriptor_target_reciprocal_rank",
)
_CONTEXT_CONTACT_NAMES = (
    "raw_pair_context_cosine",
    "restored_pair_context_cosine",
    "source_view_context_cosine",
    "target_view_context_cosine",
)


def fusion_feature_names() -> tuple[str, ...]:
    """Return the immutable ordered feature contract."""

    candidate = (
        "raw_component_baseline",
        "frozen_relation_score",
        "frozen_relation_residual",
        "raw_roster_member",
        "log_proposal_count",
        "log_contact_count",
        "source_size_fraction",
        "target_size_fraction",
        "source_log_size",
        "target_log_size",
        "source_height_fraction",
        "source_width_fraction",
        "target_height_fraction",
        "target_width_fraction",
        "source_density",
        "target_density",
        "source_confidence",
        "target_confidence",
        "row_offset_fraction",
        "column_offset_fraction",
        *(f"direction_{direction}" for direction in DIRECTIONS),
    )
    contact_names = (
        _RAW_CONTACT_NAMES
        + _RESTORED_CONTACT_NAMES
        + _DESCRIPTOR_CONTACT_NAMES
        + _CONTEXT_CONTACT_NAMES
    )
    aggregated = tuple(
        f"contact_{stat}_{name}" for stat in ("mean", "max") for name in contact_names
    )
    query = (
        "query_log_candidate_count",
        "query_relation_z",
        "query_relation_percentile",
        "query_relation_gap_to_best",
        "query_raw_z",
        "query_raw_percentile",
        "query_raw_gap_to_best",
        "query_relation_raw_rank_agreement",
    )
    return tuple(candidate) + aggregated + query


def build_fusion_features(
    components: tuple[ComponentDescriptor, ...],
    candidates: tuple[ComponentRelationCandidate, ...],
    *,
    raw_candidate_keys: frozenset[tuple[int, str, int, int, int]],
    frozen_relation_scores: torch.Tensor | np.ndarray,
    raw_tile_tokens: torch.Tensor | np.ndarray,
    restored_tile_tokens: torch.Tensor | np.ndarray,
    restored_socket_output: SocketOutput,
    restored_descriptor_scores: Mapping[str, Any],
    grid: int,
) -> np.ndarray:
    """Build target-free candidate features for the learned union selector."""

    count = grid * grid
    if not candidates:
        raise ValueError("candidates must be non-empty")
    relation_scores = _numpy(
        frozen_relation_scores,
        shape=(len(candidates),),
        name="frozen_relation_scores",
    )
    raw_tokens = _numpy(raw_tile_tokens, shape=(count, 64), name="raw_tile_tokens")
    restored_tokens = _numpy(
        restored_tile_tokens,
        shape=(count, 64),
        name="restored_tile_tokens",
    )
    raw_unit = _cosine_rows(raw_tokens)
    restored_unit = _cosine_rows(restored_tokens)
    restored_axis = {
        axis: _axis_evidence(restored_socket_output, axis=axis, count=count)
        for axis in ("right", "down")
    }
    if set(restored_descriptor_scores) != {"right", "down"}:
        raise ValueError("restored_descriptor_scores must contain right and down")
    descriptor_axis = {
        axis: _descriptor_evidence(
            restored_descriptor_scores[axis],
            count=count,
            name=f"restored_descriptor_{axis}",
        )
        for axis in ("right", "down")
    }

    rows: list[list[float]] = []
    for index, candidate in enumerate(candidates):
        source = components[candidate.source_component]
        target = components[candidate.target_component]
        direction = [0.0] * len(DIRECTIONS)
        direction[DIRECTION_TO_INDEX[candidate.direction]] = 1.0
        base = [
            candidate.baseline_score,
            relation_scores[index],
            relation_scores[index] - candidate.baseline_score,
            float(candidate.relation_key in raw_candidate_keys),
            math.log1p(candidate.proposal_count),
            math.log1p(len(candidate.contacts)),
            source.size / count,
            target.size / count,
            math.log1p(source.size) / math.log1p(count),
            math.log1p(target.size) / math.log1p(count),
            source.height / grid,
            source.width / grid,
            target.height / grid,
            target.width / grid,
            source.size / max(source.height * source.width, 1),
            target.size / max(target.height * target.width, 1),
            math.tanh(source.confidence / 5.0),
            math.tanh(target.confidence / 5.0),
            candidate.target_row_offset / max(grid - 1, 1),
            candidate.target_column_offset / max(grid - 1, 1),
            *direction,
        ]
        contacts: list[tuple[float, ...]] = []
        for contact in candidate.contacts:
            axis, forward, outgoing, incoming = _directed_indices(
                candidate.direction,
                contact.source_tile,
                contact.target_tile,
            )
            restored = _oriented_socket_features(
                restored_axis[axis],
                forward=forward,
                outgoing=outgoing,
                incoming=incoming,
            )
            descriptor = _oriented_descriptor_features(
                descriptor_axis[axis],
                forward=forward,
                outgoing=outgoing,
                incoming=incoming,
            )
            context = (
                float(raw_unit[contact.source_tile] @ raw_unit[contact.target_tile]),
                float(
                    restored_unit[contact.source_tile]
                    @ restored_unit[contact.target_tile]
                ),
                float(raw_unit[contact.source_tile] @ restored_unit[contact.source_tile]),
                float(raw_unit[contact.target_tile] @ restored_unit[contact.target_tile]),
            )
            contacts.append(tuple(contact.features) + restored + descriptor + context)
        contact_array = np.asarray(contacts, dtype=np.float64)
        rows.append(base + contact_array.mean(0).tolist() + contact_array.max(0).tolist())

    features = np.asarray(rows, dtype=np.float32)
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.query_key].append(index)
    query_features = np.empty((len(candidates), 8), dtype=np.float32)
    raw_scores = np.asarray(
        [candidate.baseline_score for candidate in candidates],
        dtype=np.float64,
    )
    for indices in grouped.values():
        relation = relation_scores[indices]
        raw = raw_scores[indices]

        def query_values(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            scale = max(float(value.std()), 1e-6)
            z = (value - float(value.mean())) / scale
            order = np.argsort(-value, kind="stable")
            rank = np.empty(len(value), dtype=np.int32)
            rank[order] = np.arange(len(value), dtype=np.int32)
            percentile = 1.0 - rank / max(len(value) - 1, 1)
            gap = value - float(value[order[0]])
            return z, percentile, gap

        relation_z, relation_percentile, relation_gap = query_values(relation)
        raw_z, raw_percentile, raw_gap = query_values(raw)
        agreement = 1.0 - np.abs(relation_percentile - raw_percentile)
        query_features[indices] = np.column_stack(
            (
                np.full(len(indices), math.log1p(len(indices)), dtype=np.float64),
                relation_z,
                relation_percentile,
                relation_gap,
                raw_z,
                raw_percentile,
                raw_gap,
                agreement,
            )
        )
    features = np.concatenate((features, query_features), axis=1)
    expected = len(fusion_feature_names())
    if features.shape != (len(candidates), expected) or not np.isfinite(features).all():
        raise RuntimeError(
            f"fusion feature contract changed: {features.shape}, expected "
            f"({len(candidates)}, {expected})"
        )
    return np.ascontiguousarray(features)


class FullresRelationFusion(nn.Module):
    """Nonlinear selector/reranker with an exact frozen-relation step-zero path."""

    def __init__(
        self,
        feature_dimension: int,
        *,
        hidden_dimension: int = 96,
        residual_limit: float = 2.0,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or hidden_dimension <= 0 or residual_limit <= 0:
            raise ValueError("feature/hidden dimensions and residual limit must be positive")
        self.feature_dimension = feature_dimension
        self.residual_limit = float(residual_limit)
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Linear(feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
        )
        self.rank_head = nn.Linear(hidden_dimension // 2, 1)
        self.confidence_head = nn.Linear(hidden_dimension // 2, 1)
        nn.init.zeros_(self.rank_head.weight)
        nn.init.zeros_(self.rank_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        frozen_relation_scores: torch.Tensor,
    ) -> FusionOutput:
        if features.ndim != 2 or features.shape[1] != self.feature_dimension:
            raise ValueError("features violate the fusion feature dimension")
        if frozen_relation_scores.shape != (len(features),):
            raise ValueError("frozen_relation_scores must align with feature rows")
        if not torch.isfinite(features).all() or not torch.isfinite(
            frozen_relation_scores
        ).all():
            raise ValueError("fusion inputs must be finite")
        hidden = self.trunk(features)
        residual = self.residual_limit * torch.tanh(self.rank_head(hidden).squeeze(1))
        return FusionOutput(
            scores=frozen_relation_scores + residual,
            confidence_logits=self.confidence_head(hidden).squeeze(1),
        )


def fusion_training_loss(
    output: FusionOutput,
    candidates: tuple[ComponentRelationCandidate, ...],
    labels: tuple[RelationCandidateLabel, ...],
    *,
    confidence_weight: float = 0.15,
    residual_weight: float = 1e-3,
    frozen_relation_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-positive listwise ranking plus balanced candidate confidence BCE."""

    if len(output.scores) != len(candidates) or len(labels) != len(candidates):
        raise ValueError("fusion output, candidates and labels must align")
    if confidence_weight < 0 or residual_weight < 0:
        raise ValueError("loss weights must be non-negative")
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[candidate.query_key].append(index)
    listwise_terms: list[torch.Tensor] = []
    for indices in groups.values():
        positives = [index for index in indices if labels[index].positive]
        if not positives:
            continue
        all_index = torch.tensor(indices, device=output.scores.device, dtype=torch.long)
        positive_index = torch.tensor(
            positives,
            device=output.scores.device,
            dtype=torch.long,
        )
        listwise_terms.append(
            torch.logsumexp(output.scores[all_index], dim=0)
            - torch.logsumexp(output.scores[positive_index], dim=0)
        )
    if not listwise_terms:
        raise ValueError("candidate board contains no supplied positive relation")
    listwise = torch.stack(listwise_terms).mean()
    target = output.confidence_logits.new_tensor(
        [float(label.positive) for label in labels]
    )
    positives = max(float(target.sum()), 1.0)
    negatives = max(float(len(target) - target.sum()), 1.0)
    positive_weight = output.confidence_logits.new_tensor(
        min(negatives / positives, 50.0)
    )
    confidence = nn.functional.binary_cross_entropy_with_logits(
        output.confidence_logits,
        target,
        pos_weight=positive_weight,
    )
    residual = output.scores
    if frozen_relation_scores is not None:
        residual = output.scores - frozen_relation_scores
    residual_penalty = residual.square().mean()
    total = listwise + confidence_weight * confidence + residual_weight * residual_penalty
    return total, {
        "total": float(total.detach()),
        "listwise": float(listwise.detach()),
        "confidence_bce": float(confidence.detach()),
        "residual_l2": float(residual_penalty.detach()),
        "supervised_queries": float(len(listwise_terms)),
        "positive_candidates": positives,
        "candidate_count": float(len(candidates)),
    }


__all__ = [
    "FullresRelationFusion",
    "FusionOutput",
    "build_fusion_features",
    "fusion_feature_names",
    "fusion_training_loss",
    "preserve_raw_union_candidates",
]
