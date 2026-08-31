from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    RelationCandidateLabel,
    RelationContact,
)
from aiijc_puzzle.component_shift_head import ComponentDescriptor
from aiijc_puzzle.fullres_relation_fusion import (
    FullresRelationFusion,
    build_fusion_features,
    fusion_feature_names,
    fusion_training_loss,
    preserve_raw_union_candidates,
)
from aiijc_puzzle.socket_matcher import SocketOutput


def _candidate(
    source: int,
    target: int,
    *,
    baseline: float = 0.0,
    direction: str = "right",
) -> ComponentRelationCandidate:
    row_offset, column_offset = (1, 0) if direction == "down" else (0, 1)
    return ComponentRelationCandidate(
        source_component=source,
        target_component=target,
        direction=direction,
        target_row_offset=row_offset,
        target_column_offset=column_offset,
        contacts=(
            RelationContact(
                source_tile=source,
                target_tile=target,
                features=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            ),
        ),
        proposal_count=1,
        baseline_score=baseline,
    )


def _label(positive: bool) -> RelationCandidateLabel:
    return RelationCandidateLabel(
        correct_contacts=int(positive),
        contact_count=1,
        positive=positive,
        source_purity=1.0,
        target_purity=1.0,
        source_size=1,
        target_size=1,
    )


def _socket_output(count: int) -> SocketOutput:
    generator = torch.Generator().manual_seed(112)
    right = torch.randn(1, count, count, generator=generator)
    down = torch.randn(1, count, count, generator=generator)
    right_assignment = torch.randn(1, count + 1, count + 1, generator=generator)
    down_assignment = torch.randn(1, count + 1, count + 1, generator=generator)
    zeros = torch.zeros(1, count)
    return SocketOutput(
        right_raw=right,
        down_raw=down,
        right_log_assignment=right_assignment,
        down_log_assignment=down_assignment,
        right_out_border_logits=zeros,
        left_in_border_logits=zeros,
        bottom_out_border_logits=zeros,
        top_in_border_logits=zeros,
    )


def test_step_zero_is_exactly_the_frozen_relation_score() -> None:
    torch.manual_seed(4)
    model = FullresRelationFusion(11, hidden_dimension=16)
    features = torch.randn(29, 11)
    baseline = torch.randn(29)
    output = model(features, baseline)
    torch.testing.assert_close(output.scores, baseline, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output.confidence_logits,
        torch.zeros_like(output.confidence_logits),
        rtol=0.0,
        atol=0.0,
    )


def test_union_preserves_every_raw_candidate_before_restored_fill() -> None:
    raw = (_candidate(0, 1), _candidate(0, 2), _candidate(1, 3))
    expanded = (_candidate(0, 2), _candidate(0, 3), _candidate(1, 2))
    union = preserve_raw_union_candidates(raw, expanded, max_candidates_per_query=3)
    assert {candidate.relation_key for candidate in raw} <= {
        candidate.relation_key for candidate in union
    }
    assert len([candidate for candidate in union if candidate.query_key == (0, "right")]) == 3


def test_target_free_feature_builder_has_stable_finite_contract() -> None:
    grid = 2
    components = tuple(
        ComponentDescriptor((tile,), (0,), (0,), 0.0) for tile in range(grid * grid)
    )
    candidates = (_candidate(0, 1), _candidate(0, 2, direction="down"))
    relation = torch.tensor([0.4, -0.2])
    generator = np.random.default_rng(19)
    raw_tokens = generator.normal(size=(4, 64)).astype(np.float32)
    restored_tokens = generator.normal(size=(4, 64)).astype(np.float32)
    descriptor = {
        "right": generator.normal(size=(4, 4)).astype(np.float32),
        "down": generator.normal(size=(4, 4)).astype(np.float32),
    }
    first = build_fusion_features(
        components,
        candidates,
        raw_candidate_keys=frozenset({candidates[0].relation_key}),
        frozen_relation_scores=relation,
        raw_tile_tokens=raw_tokens,
        restored_tile_tokens=restored_tokens,
        restored_socket_output=_socket_output(4),
        restored_descriptor_scores=descriptor,
        grid=grid,
    )
    second = build_fusion_features(
        components,
        candidates,
        raw_candidate_keys=frozenset({candidates[0].relation_key}),
        frozen_relation_scores=relation,
        raw_tile_tokens=raw_tokens,
        restored_tile_tokens=restored_tokens,
        restored_socket_output=_socket_output(4),
        restored_descriptor_scores=descriptor,
        grid=grid,
    )
    assert first.shape == (2, len(fusion_feature_names()))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_four_by_four_capacity_smoke_learns_positive_candidate_per_query() -> None:
    torch.manual_seed(29)
    candidates: list[ComponentRelationCandidate] = []
    labels: list[RelationCandidateLabel] = []
    features: list[list[float]] = []
    for source in range(16):
        for option in range(4):
            target = (source + option + 1) % 16
            candidates.append(_candidate(source, target, baseline=0.0))
            positive = option == source % 4
            labels.append(_label(positive))
            features.append(
                [
                    2.0 if positive else -2.0,
                    float(option) / 4.0,
                    float(source % 4) / 4.0,
                    1.0,
                    -1.0,
                    0.5,
                ]
            )
    feature_tensor = torch.tensor(features)
    frozen = torch.zeros(len(candidates))
    candidate_tuple = tuple(candidates)
    label_tuple = tuple(labels)
    model = FullresRelationFusion(6, hidden_dimension=24)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-2)
    first_loss = None
    for _ in range(80):
        output = model(feature_tensor, frozen)
        loss, _ = fusion_training_loss(
            output,
            candidate_tuple,
            label_tuple,
            frozen_relation_scores=frozen,
        )
        first_loss = float(loss.detach()) if first_loss is None else first_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    final_output = model(feature_tensor, frozen)
    final_loss, _ = fusion_training_loss(
        final_output,
        candidate_tuple,
        label_tuple,
        frozen_relation_scores=frozen,
    )
    correct = 0
    for source in range(16):
        start = 4 * source
        predicted = int(final_output.scores[start : start + 4].argmax())
        correct += int(predicted == source % 4)
    assert correct == 16
    assert float(final_loss.detach()) < 0.2 * float(first_loss)
