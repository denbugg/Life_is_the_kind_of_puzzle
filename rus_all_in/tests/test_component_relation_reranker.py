from __future__ import annotations

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationCandidate,
    ComponentRelationReranker,
    ComponentTruthProfile,
    RelationCandidateLabel,
    RelationContact,
    aggregate_relation_observations,
    build_component_relation_candidates,
    component_relation_targets,
    extract_frozen_socket_context,
    relation_listwise_loss,
    relation_query_observations,
)
from aiijc_puzzle.component_shift_head import ComponentDescriptor
from aiijc_puzzle.socket_matcher import SocketMatcher, SocketOutput


def _singleton_components(grid: int) -> tuple[ComponentDescriptor, ...]:
    return tuple(
        ComponentDescriptor((tile,), (0,), (0,), 0.0) for tile in range(grid * grid)
    )


def _perfect_socket_output(grid: int) -> SocketOutput:
    count = grid * grid
    right = np.full((count, count), -5.0, dtype=np.float32)
    down = np.full((count, count), -5.0, dtype=np.float32)
    np.fill_diagonal(right, -1e4)
    np.fill_diagonal(down, -1e4)
    for position in range(count):
        if position % grid != grid - 1:
            right[position, position + 1] = 7.0
        if position < count - grid:
            down[position, position + grid] = 7.0
    right_assignment = np.full((count + 1, count + 1), -8.0, dtype=np.float32)
    down_assignment = np.full((count + 1, count + 1), -8.0, dtype=np.float32)
    right_assignment[:count, :count] = right
    down_assignment[:count, :count] = down
    right_assignment[count, count] = down_assignment[count, count] = -1e4

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).unsqueeze(0)

    zeros = torch.zeros(1, count)
    return SocketOutput(
        right_raw=tensor(right),
        down_raw=tensor(down),
        right_log_assignment=tensor(right_assignment),
        down_log_assignment=tensor(down_assignment),
        right_out_border_logits=zeros,
        left_in_border_logits=zeros,
        bottom_out_border_logits=zeros,
        top_in_border_logits=zeros,
    )


def _candidate(
    source: int,
    target: int,
    *,
    feature: float,
    baseline: float,
) -> ComponentRelationCandidate:
    contact_features = (feature, -feature, feature, feature, 1.0, 1.0, 0.0, 0.0)
    return ComponentRelationCandidate(
        source_component=source,
        target_component=target,
        direction="right",
        target_row_offset=0,
        target_column_offset=1,
        contacts=(RelationContact(source, target, contact_features),),
        proposal_count=1,
        baseline_score=baseline,
    )


def test_frozen_socket_context_matches_the_model_forward_exactly() -> None:
    torch.manual_seed(211)
    model = SocketMatcher(
        dimension=8,
        heads=2,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=3,
    ).eval()
    tiles = torch.rand(1, 9, 3, 20, 20)
    with torch.no_grad():
        context, extracted = extract_frozen_socket_context(model, tiles, grid=3)
        expected = model(tiles, grid=3)
    assert context.shape == (1, 9, 8)
    for field in SocketOutput.__dataclass_fields__:
        torch.testing.assert_close(getattr(extracted, field), getattr(expected, field))


def test_candidate_builder_recovers_real_pair_translations_without_labels() -> None:
    grid = 3
    components = _singleton_components(grid)
    candidates = build_component_relation_candidates(
        components,
        _perfect_socket_output(grid),
        grid=grid,
        proposal_topk=1,
        max_candidates_per_query=4,
    )
    assert candidates
    assert len({candidate.relation_key for candidate in candidates}) == len(candidates)
    labels, oracle, profiles = component_relation_targets(
        candidates,
        components,
        np.arange(grid * grid),
        grid=grid,
    )
    assert oracle
    assert all(profile.purity == 1.0 for profile in profiles)
    right_zero = next(
        candidate
        for candidate in candidates
        if candidate.source_component == 0 and candidate.direction == "right"
    )
    assert right_zero.target_component == 1
    assert (right_zero.target_row_offset, right_zero.target_column_offset) == (0, 1)
    assert labels[candidates.index(right_zero)].positive
    assert labels[candidates.index(right_zero)].correct_contacts == 1


def test_restored_union_can_expand_supply_without_changing_raw_baseline() -> None:
    grid = 3
    count = grid * grid
    output = _perfect_socket_output(grid)
    raw_only = build_component_relation_candidates(
        _singleton_components(grid),
        output,
        grid=grid,
        proposal_topk=1,
        max_candidates_per_query=8,
    )
    extra_right = np.full((count, count), -3.0, dtype=np.float32)
    extra_down = np.full((count, count), -3.0, dtype=np.float32)
    np.fill_diagonal(extra_right, -4.0)
    np.fill_diagonal(extra_down, -4.0)
    extra_right[0, 4] = 9.0
    expanded = build_component_relation_candidates(
        _singleton_components(grid),
        output,
        grid=grid,
        proposal_topk=1,
        max_candidates_per_query=8,
        additional_proposal_scores={"right": extra_right, "down": extra_down},
    )
    raw_lookup = {candidate.relation_key: candidate.baseline_score for candidate in raw_only}
    expanded_lookup = {candidate.relation_key: candidate.baseline_score for candidate in expanded}
    assert len(expanded) > len(raw_only)
    assert all(expanded_lookup[key] == score for key, score in raw_lookup.items())


def test_component_and_contact_set_encodings_are_order_invariant() -> None:
    torch.manual_seed(223)
    grid = 3
    first = ComponentDescriptor((0, 1), (0, 0), (0, 1), 1.0)
    second = ComponentDescriptor((2, 3), (0, 0), (0, 1), 0.8)
    rest = tuple(
        ComponentDescriptor((tile,), (0,), (0,), 0.0) for tile in range(4, grid * grid)
    )
    components = (first, second, *rest)
    reversed_components = (
        ComponentDescriptor((1, 0), (0, 0), (1, 0), 1.0),
        second,
        *rest,
    )
    contacts = (
        RelationContact(0, 2, (1.0, 0.2, 0.4, 0.1, 1.0, 0.5, 0.0, 0.0)),
        RelationContact(1, 3, (0.7, 0.1, 0.2, 0.3, 0.5, 1.0, 0.0, 0.0)),
    )
    candidate = ComponentRelationCandidate(0, 1, "down", 1, 0, contacts, 2, 1.0)
    reversed_candidate = ComponentRelationCandidate(
        0,
        1,
        "down",
        1,
        0,
        contacts[::-1],
        2,
        1.0,
    )
    model = ComponentRelationReranker(6, grid=grid, hidden_dimension=12).eval()
    tokens = torch.randn(grid * grid, 6)
    with torch.no_grad():
        expected = model(tokens, components, (candidate,))
        observed = model(tokens, reversed_components, (reversed_candidate,))
    torch.testing.assert_close(expected, observed)


def test_tiny_component_relation_capacity_and_metrics_smoke() -> None:
    torch.manual_seed(227)
    grid = 4
    components = _singleton_components(grid)
    candidates: list[ComponentRelationCandidate] = []
    labels: list[RelationCandidateLabel] = []
    oracle: set[tuple[int, str, int, int, int]] = set()
    for source in range(4):
        correct_target = 4 + source
        oracle.add((source, "right", correct_target, 0, 1))
        for candidate_index, target in enumerate(range(4, 8)):
            positive = target == correct_target
            candidates.append(
                _candidate(
                    source,
                    target,
                    feature=3.0 if positive else -3.0 - candidate_index,
                    baseline=-3.0 if positive else float(candidate_index),
                )
            )
            labels.append(
                RelationCandidateLabel(
                    correct_contacts=int(positive),
                    contact_count=1,
                    positive=positive,
                    source_purity=1.0,
                    target_purity=1.0,
                    source_size=1,
                    target_size=1,
                )
            )
    candidates_tuple = tuple(candidates)
    labels_tuple = tuple(labels)
    tokens = torch.randn(grid * grid, 5)
    model = ComponentRelationReranker(5, grid=grid, hidden_dimension=16)
    with torch.no_grad():
        zero_step = model(tokens, components, candidates_tuple)
    torch.testing.assert_close(
        zero_step,
        torch.tensor([candidate.baseline_score for candidate in candidates_tuple]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = None
    for _ in range(50):
        logits = model(tokens, components, candidates_tuple)
        loss, diagnostics = relation_listwise_loss(logits, candidates_tuple, labels_tuple)
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert initial is not None
    assert diagnostics["supervised_queries"] == 4
    assert float(loss.detach()) < 0.15 * initial

    profiles = tuple(ComponentTruthProfile(1, 1, 1.0) for _ in components)
    observations = relation_query_observations(
        logits,
        candidates_tuple,
        labels_tuple,
        frozenset(oracle),
        profiles,
        board_id="capacity",
    )
    metrics = aggregate_relation_observations(observations, high_confidence_caps=(4,))
    assert metrics["learned"]["r1"] == 1.0
    assert metrics["raw_socket_component_baseline"]["r1"] == 0.0
    assert metrics["learned"]["high_confidence"]["top4"]["correct_per_board"] == 4.0
