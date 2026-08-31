from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from aiijc_puzzle.learned_membership_rank_delta_priority import (
    compose_learned_membership_rank_delta_priority,
)
from aiijc_puzzle.socket_decoder import (
    PartialAxisMatching,
    SocketEdge,
    prioritise_component_edges,
)


def _fixture() -> dict[str, Any]:
    # Two exact 3x3 hard partial matchings: six immutable identities per axis.
    return {
        "edge_source": np.asarray(
            [0, 1, 3, 4, 6, 7, 0, 1, 2, 3, 4, 5],
            dtype=np.int32,
        ),
        "edge_target": np.asarray(
            [1, 2, 4, 5, 7, 8, 3, 4, 5, 6, 7, 8],
            dtype=np.int32,
        ),
        "edge_axis": np.asarray([0] * 6 + [1] * 6, dtype=np.int8),
        "union_base_priority": np.asarray(
            [60, 50, 40, 30, 20, 10, 6, 5, 4, 3, 2, 1],
            dtype=np.float64,
        ),
        "learned_priority": np.asarray(
            [0, 9, 1, 8, 10, 2, 4, 0, 3, 1, 5, 2],
            dtype=np.float64,
        ),
        "rank_delta_priority": np.asarray(
            [10, 40, 20, 50, 60, 30, 1, 6, 2, 5, 3, 4],
            dtype=np.float64,
        ),
        "grid": 3,
        "edge_budget_per_axis": 2,
    }


def _identity_values(
    axis: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
) -> dict[tuple[int, int, int], Any]:
    return {
        (int(axis_value), int(source_value), int(target_value)): values[index].item()
        for index, (axis_value, source_value, target_value) in enumerate(
            zip(axis, source, target, strict=True)
        )
    }


def test_composition_preserves_roster_and_confidence_multiset_exactly() -> None:
    inputs = _fixture()
    original_source = inputs["edge_source"].copy()
    original_target = inputs["edge_target"].copy()
    original_axis = inputs["edge_axis"].copy()
    result = compose_learned_membership_rank_delta_priority(**inputs)

    np.testing.assert_array_equal(result.source, original_source)
    np.testing.assert_array_equal(result.target, original_target)
    np.testing.assert_array_equal(result.axis, original_axis)
    assert result.scores.shape == (12,)
    assert result.scores.dtype == np.float64
    assert result.scores.flags.c_contiguous
    assert result.learned_membership.dtype == np.bool_
    assert result.learned_membership.flags.c_contiguous

    base = inputs["union_base_priority"]
    for axis_index in (0, 1):
        selected = original_axis == axis_index
        assert np.count_nonzero(result.learned_membership[selected]) == 2
        np.testing.assert_array_equal(
            np.sort(result.scores[selected]),
            np.sort(base[selected]),
        )

    assert set(result.component_edge_priority) == {"right", "down"}
    for axis_index, name in ((0, "right"), (1, "down")):
        matrix = result.component_edge_priority[name]
        assert matrix.shape == (9, 9)
        assert matrix.dtype == np.float64
        assert matrix.flags.c_contiguous
        selected = original_axis == axis_index
        np.testing.assert_array_equal(
            matrix[original_source[selected], original_target[selected]],
            result.scores[selected],
        )
        roster_mask = np.zeros((9, 9), dtype=bool)
        roster_mask[original_source[selected], original_target[selected]] = True
        assert np.all(matrix[~roster_mask] == 0.0)

    diagnostics = result.diagnostics
    assert diagnostics.learned_membership_per_axis == (2, 2)
    assert diagnostics.decoder_membership_matches_learned_per_axis == (True, True)
    assert diagnostics.rank_delta_input_multiset_preserved_per_axis == (True, True)
    assert diagnostics.output_multiset_preserved_per_axis == (True, True)
    assert diagnostics.immutable_identity_count == 12
    assert result.report()["legality"] == {
        "new_hard_edges_introduced": False,
        "targets_labels_or_pixels_accepted": False,
        "original_union_confidence_multiset_preserved_per_axis": True,
        "layout_or_pixel_output_produced": False,
    }


def test_learned_membership_is_top_k_but_rank_delta_orders_members() -> None:
    inputs = _fixture()
    result = compose_learned_membership_rank_delta_priority(**inputs)

    # Right learned top-2 are local rows 4 and 1.  Rank-delta puts row 4 first.
    assert np.flatnonzero(result.learned_membership[:6]).tolist() == [1, 4]
    assert result.scores[4] == 60.0
    assert result.scores[1] == 50.0
    # The strongest non-member by rank-delta cannot cross the learned cutoff.
    assert result.scores[3] == 40.0

    # Down learned top-2 are local rows 4 and 0; rank-delta orders row 4 first.
    assert np.flatnonzero(result.learned_membership[6:]).tolist() == [0, 4]
    assert result.scores[10] == 6.0
    assert result.scores[6] == 5.0


def test_learned_membership_matches_the_real_decoder_tie_break() -> None:
    inputs = _fixture()
    result = compose_learned_membership_rank_delta_priority(**inputs)
    source = inputs["edge_source"]
    target = inputs["edge_target"]
    axis = inputs["edge_axis"]
    base = inputs["union_base_priority"]

    def matching(axis_index: int, name: str) -> PartialAxisMatching:
        indices = np.flatnonzero(axis == axis_index)
        delta = (0, 1) if name == "right" else (1, 0)
        edges = tuple(
            SocketEdge(
                source=int(source[index]),
                target=int(target[index]),
                delta_row=delta[0],
                delta_column=delta[1],
                confidence=float(base[index]),
                axis=name,
            )
            for index in indices
        )
        return PartialAxisMatching(edges=edges, outgoing_unmatched=(), incoming_unmatched=())

    selected = prioritise_component_edges(
        matching(0, "right"),
        matching(1, "down"),
        edge_budget_per_axis=2,
        tile_count=9,
        component_edge_priority=result.component_edge_priority,
    )
    selected_identities = {
        (0 if edge.axis == "right" else 1, edge.source, edge.target) for edge in selected
    }
    expected_identities = {
        (int(axis[index]), int(source[index]), int(target[index]))
        for index in np.flatnonzero(result.learned_membership)
    }
    assert selected_identities == expected_identities


def test_cutoff_tie_that_decoder_cannot_realise_fails_closed() -> None:
    inputs = _fixture()
    inputs["union_base_priority"][:6] = [6, 5, 5, 3, 2, 1]
    inputs["rank_delta_priority"][:6] = [5, 3, 2, 1, 6, 5]
    inputs["learned_priority"][:6] = [0, 0, 0, 0, 10, 9]

    with pytest.raises(ValueError, match="tie at the component cutoff"):
        compose_learned_membership_rank_delta_priority(**inputs)


def test_ties_are_identity_deterministic_and_independent_of_input_row_order() -> None:
    inputs = _fixture()
    inputs["learned_priority"] = np.ones(12, dtype=np.float64)
    inputs["rank_delta_priority"] = inputs["union_base_priority"].copy()
    first = compose_learned_membership_rank_delta_priority(**inputs)

    permutation = np.asarray([4, 8, 0, 11, 3, 7, 1, 10, 5, 9, 2, 6])
    shuffled = {
        key: value[permutation] if isinstance(value, np.ndarray) and value.shape == (12,) else value
        for key, value in inputs.items()
    }
    second = compose_learned_membership_rank_delta_priority(**shuffled)

    assert _identity_values(first.axis, first.source, first.target, first.scores) == (
        _identity_values(second.axis, second.source, second.target, second.scores)
    )
    assert _identity_values(
        first.axis,
        first.source,
        first.target,
        first.learned_membership,
    ) == _identity_values(
        second.axis,
        second.source,
        second.target,
        second.learned_membership,
    )
    assert np.flatnonzero(first.learned_membership[:6]).tolist() == [0, 1]
    assert np.flatnonzero(first.learned_membership[6:]).tolist() == [0, 1]


def test_output_does_not_alias_mutable_inputs() -> None:
    inputs = _fixture()
    result = compose_learned_membership_rank_delta_priority(**inputs)
    inputs["edge_source"][:] = 8
    inputs["edge_target"][:] = 0
    inputs["edge_axis"][:] = 1
    inputs["union_base_priority"][:] = -1.0

    np.testing.assert_array_equal(
        result.source,
        [0, 1, 3, 4, 6, 7, 0, 1, 2, 3, 4, 5],
    )
    np.testing.assert_array_equal(
        result.target,
        [1, 2, 4, 5, 7, 8, 3, 4, 5, 6, 7, 8],
    )
    np.testing.assert_array_equal(result.axis, [0] * 6 + [1] * 6)
    assert np.all(result.scores > 0.0)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("edge_source", np.arange(11, dtype=np.int32), "shape"),
        ("edge_target", np.arange(12, dtype=np.float64), "integers"),
        ("edge_axis", np.zeros((12, 1), dtype=np.int8), "shape"),
        ("union_base_priority", np.full(12, np.inf), "finite"),
        ("learned_priority", np.full(12, np.nan), "finite"),
        ("rank_delta_priority", np.full(12, -np.inf), "finite"),
        ("edge_source", np.full(12, 2**32, dtype=np.int64), "int32 range"),
    ],
)
def test_vector_shape_type_and_finiteness_are_validated(
    key: str,
    value: np.ndarray,
    message: str,
) -> None:
    inputs = _fixture()
    inputs[key] = value
    with pytest.raises(ValueError, match=message):
        compose_learned_membership_rank_delta_priority(**inputs)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda values: values["edge_source"].__setitem__(1, 0), "outgoing"),
        (lambda values: values["edge_target"].__setitem__(2, 1), "incoming"),
        (lambda values: values["edge_target"].__setitem__(0, 0), "self"),
        (lambda values: values["edge_axis"].__setitem__(0, 1), "exactly"),
        (lambda values: values["edge_target"].__setitem__(0, 9), "out-of-range"),
    ],
)
def test_malformed_or_duplicate_union_roster_is_rejected(
    mutate: Any,
    message: str,
) -> None:
    inputs = _fixture()
    mutate(inputs)
    with pytest.raises(ValueError, match=message):
        compose_learned_membership_rank_delta_priority(**inputs)


def test_duplicate_union_edge_is_rejected_explicitly() -> None:
    inputs = _fixture()
    inputs["edge_source"][1] = inputs["edge_source"][0]
    inputs["edge_target"][1] = inputs["edge_target"][0]
    with pytest.raises(ValueError, match="duplicate edges"):
        compose_learned_membership_rank_delta_priority(**inputs)


def test_rank_delta_must_preserve_union_confidence_multiset_per_axis() -> None:
    inputs = _fixture()
    inputs["rank_delta_priority"][0] += 0.25
    with pytest.raises(ValueError, match="multiset on axis 0"):
        compose_learned_membership_rank_delta_priority(**inputs)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("grid", True, "grid"),
        ("grid", 1, "grid"),
        ("edge_budget_per_axis", True, "integer"),
        ("edge_budget_per_axis", 0, r"\[1, 6\]"),
        ("edge_budget_per_axis", 7, r"\[1, 6\]"),
    ],
)
def test_scalar_contract_is_validated(key: str, value: object, message: str) -> None:
    inputs = _fixture()
    inputs[key] = value
    with pytest.raises(ValueError, match=message):
        compose_learned_membership_rank_delta_priority(**inputs)


def test_public_api_accepts_no_reference_layout_pixels_or_labels() -> None:
    parameters = set(inspect.signature(compose_learned_membership_rank_delta_priority).parameters)
    assert parameters == {
        "edge_source",
        "edge_target",
        "edge_axis",
        "union_base_priority",
        "learned_priority",
        "rank_delta_priority",
        "grid",
        "edge_budget_per_axis",
    }
    assert parameters.isdisjoint(
        {
            "labels",
            "pixels",
            "reference_tiles",
            "target_layout",
            "input_tile_to_position",
        }
    )
