from __future__ import annotations

import inspect
from dataclasses import fields
from math import log

import numpy as np
import pytest
import torch

import e23_i21_residual_candidate_oracle as e23
import e26_relation_verifier as core
import train_e26_relation_verifier as trainer


def _diagnostics() -> e23.CandidatePoolDiagnostics:
    return e23.CandidatePoolDiagnostics(
        **{field.name: 0 for field in fields(e23.CandidatePoolDiagnostics)}
    )


def _result() -> e23.CandidatePoolResult:
    components = tuple(
        e23.RigidComponent(component_id=tile, entries=((tile, 0, 0),))
        for tile in range(core.NUM_TILES)
    )
    owner = np.arange(core.NUM_TILES, dtype=np.int64)
    local_rows = np.zeros(core.NUM_TILES, dtype=np.int64)
    local_cols = np.zeros(core.NUM_TILES, dtype=np.int64)
    pairs = (
        e23.SpatialPair(pair_id=0, a=0, b=1, nomination_count=1),
        e23.SpatialPair(pair_id=1, a=2, b=3, nomination_count=1),
    )
    claims = (
        # component 1 is right of component 0 -> canonical dc=+1
        e23.RCCE4Claim(0, 0, 0, 1, 0, 1, 0, 1, None, None),
        # component 0 is right of component 1 -> canonical dc=-1
        e23.RCCE4Claim(1, 0, 1, 0, 0, 1, 1, 0, None, None),
        # component 3 is below component 2 -> canonical dr=+1
        e23.RCCE4Claim(2, 1, 2, 3, 1, 0, 2, 3, None, None),
    )
    relations = (
        e23.RelationCandidate(0, 0, 1, 0, -1, (1,)),
        e23.RelationCandidate(1, 0, 1, 0, 1, (0,)),
        e23.RelationCandidate(2, 2, 3, 1, 0, (2,)),
    )
    hypotheses = tuple(
        e23.PoseHypothesis(index, relation.relation_id, relation.u, relation.v, relation.dr, relation.dc, relation.claim_ids)
        for index, relation in enumerate(relations)
    )
    return e23.CandidatePoolResult(
        components=components,
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial_component_ids=frozenset(),
        affinity_pairs=pairs,
        base_affinity_pairs=(),
        spatial_selected_ids=np.zeros((4, core.NUM_TILES, e23.SPATIAL_K), dtype=np.int64),
        spatial_pairs=pairs,
        claims=claims,
        relation_candidates=relations,
        hypotheses=hypotheses,
        rejections=(),
        diagnostics=_diagnostics(),
    )


@pytest.fixture(scope="module")
def extracted() -> tuple[e23.CandidatePoolResult, core.RelationQueryTable]:
    result = _result()
    candidate_ids = np.empty((core.NUM_TILES, core.RAW_WIDTH), dtype=np.int64)
    for source in range(core.NUM_TILES):
        candidate_ids[source] = (
            source + 1 + np.arange(core.RAW_WIDTH, dtype=np.int64)
        ) % core.NUM_TILES
    raw = np.zeros((4, core.NUM_TILES, core.RAW_WIDTH), dtype=np.float32)
    i21 = np.zeros((4, core.NUM_TILES, core.NUM_TILES), dtype=np.float32)
    context = np.zeros_like(i21)
    none = np.zeros((4, core.NUM_TILES), dtype=np.float32)
    # Correct directional signs strongly favor canonical relation (0,1,0,+1).
    context[core.RIGHT, 0, 1] = 6.0
    context[core.LEFT, 1, 0] = 6.0
    context[core.RIGHT, 1, 0] = -2.0
    context[core.LEFT, 0, 1] = -2.0
    i21[:] = context
    table = core.extract_relation_queries(
        result, candidate_ids, raw, i21, context, none
    )
    return result, table


def test_canonical_sign_all_offsets_and_explicit_none(extracted) -> None:
    _result_value, table = extracted
    assert tuple(table.query_offsets) == (0, 3, 5)
    assert [tuple(row) for row in table.relations[:3]] == [
        (0, 1, 0, -1),
        (0, 1, 0, 1),
        (0, 1, 0, 0),
    ]
    assert tuple(table.row_kind) == (
        core.ROW_OFFSET,
        core.ROW_OFFSET,
        core.ROW_NONE,
        core.ROW_OFFSET,
        core.ROW_NONE,
    )
    feature = core.FEATURE_INDEX["context_logprob_mean"]
    assert table.features[1, feature] > table.features[0, feature]


def test_labels_choose_true_offset_and_none(extracted) -> None:
    result, table = extracted
    permutation = np.arange(core.NUM_TILES, dtype=np.int64)
    labels = trainer.build_relation_labels(result, table, permutation)
    # Tile 1 is right of tile 0, while the only (2,3) candidate says "below".
    assert tuple(labels) == (0, 1, 0, 0, 1)


def test_extractor_has_no_label_capability(extracted) -> None:
    signature = inspect.signature(core.extract_relation_queries)
    forbidden = {"permutation", "truth", "labels", "source_group", "scene_id"}
    assert forbidden.isdisjoint(signature.parameters)
    assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    result, _table = extracted
    with pytest.raises(TypeError):
        core.extract_relation_queries(result, permutation=np.arange(core.NUM_TILES))
    assert len(core.FEATURE_NAMES) == 64
    assert not any("_id" in name or "truth" in name for name in core.FEATURE_NAMES)


def test_exact_softmax_and_sigmoid_temperature(extracted) -> None:
    _result_value, table = extracted
    row_logits = np.asarray((0.0, 0.0, 0.0, log(2.0), 0.0), dtype=np.float64)
    edge_logits = np.asarray((0.0, log(3.0)), dtype=np.float64)
    probabilities = core.calibrated_probabilities(
        table, row_logits, edge_logits, row_temperature=1.0, edge_temperature=1.0
    )
    assert probabilities.row_probabilities[3] == pytest.approx(2.0 / 3.0)
    assert probabilities.row_probabilities[4] == pytest.approx(1.0 / 3.0)
    assert probabilities.edge_probabilities[1] == pytest.approx(0.75)


def test_strict_threshold_rejects_and_cap_never_fills(extracted) -> None:
    _result_value, table = extracted
    probabilities = core.CalibratedRelationProbabilities(
        row_probabilities=np.asarray((0.1, 0.8, 0.1, 0.8, 0.2), dtype=np.float64),
        edge_probabilities=np.asarray((0.90, 0.89), dtype=np.float64),
        row_temperature=1.0,
        edge_temperature=1.0,
    )
    qualified, attempted, cap = core.select_qualified_relations(
        table, probabilities, component_count=core.NUM_TILES, edge_threshold=0.90
    )
    assert qualified == attempted == ()
    assert cap == 2 * (core.NUM_TILES - 1)


def _selection(hypothesis: e23.PoseHypothesis) -> core.SelectedRelation:
    return core.SelectedRelation(
        hypothesis_id=hypothesis.hypothesis_id,
        relation_id=hypothesis.relation_id,
        u=hypothesis.u,
        v=hypothesis.v,
        dr=hypothesis.dr,
        dc=hypothesis.dc,
        edge_probability=0.99,
        offset_probability=0.9,
        none_probability=0.1,
        probability_margin=0.8,
        support=len(hypothesis.claim_ids),
    )


def test_rejected_potential_dsu_operation_does_not_mutate(extracted) -> None:
    result, _table = extracted
    dsu = core._PotentialDSU(result)
    assert dsu.try_accept(_selection(result.hypotheses[1])) == (
        True,
        "tree",
        True,
        False,
    )
    before = dsu.state_signature()
    assert dsu.try_accept(_selection(result.hypotheses[0])) == (
        False,
        "conflict",
        False,
        False,
    )
    assert dsu.state_signature() == before


def test_hard_negative_subset_keeps_positive_none_and_best_negative(extracted) -> None:
    result, table = extracted
    labels = trainer.build_relation_labels(
        result, table, np.arange(core.NUM_TILES, dtype=np.int64)
    )
    subsets = trainer.hard_negative_query_subset(
        table,
        labels,
        np.asarray((0.9, 0.1, 0.0, 0.7, 0.2), dtype=np.float64),
        max_hard_negatives=1,
    )
    assert tuple(subsets[0]) == (0, 1, 2)
    assert tuple(subsets[1]) == (3, 4)


def test_mlp_outputs_listwise_rows_and_edge_existence(extracted) -> None:
    _result_value, table = extracted
    torch.manual_seed(1)
    model = trainer.RelationVerifierMLP()
    model.eval()
    output = model(torch.from_numpy(table.features.copy()), table.query_offsets)
    assert output.row_logits.shape == (table.rows,)
    assert output.edge_logits.shape == (table.queries,)
    assert output.embeddings.shape == (table.rows, trainer.EMBEDDING_WIDTH)
    assert torch.isfinite(output.row_logits).all()


def test_concatenation_resets_canonical_pair_order_per_scene(extracted) -> None:
    _result_value, table = extracted
    combined = core.concatenate_query_tables((table, table))
    assert combined.rows == 2 * table.rows
    assert tuple(combined.scene_offsets) == (0, table.rows, 2 * table.rows)
    assert combined.queries == 2 * table.queries


def test_streaming_train_epoch_backpropagates_one_scene(extracted) -> None:
    result, table = extracted
    labels = trainer.build_relation_labels(
        result, table, np.arange(core.NUM_TILES, dtype=np.int64)
    )
    torch.manual_seed(2)
    model = trainer.RelationVerifierMLP()
    before = model.row_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = trainer.train_relation_epoch(
        model,
        (trainer.TrainingScene(table, labels, "source-a"),),
        optimizer,
        device=torch.device("cpu"),
        max_hard_negatives=trainer.MAX_HARD_NEGATIVES,
    )
    assert torch.isfinite(loss.total)
    assert loss.queries == table.queries
    assert not torch.equal(before, model.row_head.weight.detach())
