"""Label-free E26 component-relation features and conservative decoder.

The module is intentionally a production boundary.  It accepts the exact
frozen E23 candidate result plus inference-time logits, retains every
geometry-valid offset and appends one explicit ``NONE`` row to every
canonical component-pair query.  Labels, permutations, scene/source IDs and
metrics are not accepted by the extractor and identifiers are metadata only.

Tiles are upright throughout: no rotation or reflection state exists here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Mapping, Sequence

import numpy as np

import e23_i21_residual_candidate_oracle as e23


GRID = e23.GRID
NUM_TILES = e23.NUM_TILES
NUM_DIRECTIONS = e23.NUM_DIRECTIONS
RAW_WIDTH = e23.CANDIDATE_WIDTH
UP, DOWN, LEFT, RIGHT = range(NUM_DIRECTIONS)
ROW_OFFSET = np.uint8(0)
ROW_NONE = np.uint8(1)
NONE_ID = -1
FEATURE_SCHEMA = "pazzle-e26-relation-verifier-features-v1"
DEFAULT_EDGE_THRESHOLD = 0.90
ATTEMPT_MULTIPLIER = 2


FEATURE_NAMES: tuple[str, ...] = (
    "is_none",
    "has_offset",
    "size_min_log1p",
    "size_max_log1p",
    "size_ratio",
    "height_min_scaled",
    "height_max_scaled",
    "width_min_scaled",
    "width_max_scaled",
    "density_min",
    "density_max",
    "merged_height_scaled",
    "merged_width_scaled",
    "merged_density",
    "offset_l1_scaled",
    "claim_count_log1p",
    "base_claim_count_log1p",
    "residual_claim_count_log1p",
    "base_claim_fraction",
    *tuple(
        f"{source}_{stat}"
        for source in ("context", "i21", "raw")
        for stat in (
            "logprob_min",
            "logprob_mean",
            "logprob_max",
            "logprob_logsumexp",
            "rank_min",
            "rank_mean",
            "rank_max",
            "rank_logsumexp",
            "margin_min",
            "margin_mean",
            "margin_max",
            "margin_logsumexp",
        )
    ),
    "query_offset_count_log1p",
    "query_support_sum_log1p",
    "query_best_context_mean",
    "query_second_context_mean",
    "query_context_gap",
    "offset_context_rank_percentile",
    "offset_context_gap_best",
    "offset_context_margin_other",
    "query_none_context_mean",
)
FEATURE_INDEX: Mapping[str, int] = {
    name: index for index, name in enumerate(FEATURE_NAMES)
}

if len(FEATURE_NAMES) != 64 or len(set(FEATURE_NAMES)) != 64:
    raise RuntimeError("E26 relation feature schema must contain exactly 64 names")
if any(
    forbidden in name
    for name in FEATURE_NAMES
    for forbidden in (
        "tile_id",
        "component_id",
        "relation_id",
        "hypothesis_id",
        "scene_id",
        "source_id",
        "permutation",
        "truth",
        "label",
        "metric",
    )
):
    raise RuntimeError("an identifier or supervised value leaked into E26 features")


class RelationVerifierError(ValueError):
    """A label-free input, probability, or geometry invariant failed closed."""


@dataclass(frozen=True, slots=True)
class RelationQueryTable:
    features: np.ndarray
    hypothesis_ids: np.ndarray
    relation_ids: np.ndarray
    relations: np.ndarray
    row_kind: np.ndarray
    support: np.ndarray
    query_offsets: np.ndarray
    scene_offsets: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def queries(self) -> int:
        return int(self.query_offsets.size - 1)

    @property
    def none_rows(self) -> np.ndarray:
        return self.query_offsets[1:] - 1


@dataclass(frozen=True, slots=True)
class CalibratedRelationProbabilities:
    row_probabilities: np.ndarray
    edge_probabilities: np.ndarray
    row_temperature: float
    edge_temperature: float


@dataclass(frozen=True, slots=True)
class SelectedRelation:
    hypothesis_id: int
    relation_id: int
    u: int
    v: int
    dr: int
    dc: int
    edge_probability: float
    offset_probability: float
    none_probability: float
    probability_margin: float
    support: int

    @property
    def relation(self) -> tuple[int, int, int, int]:
        return self.u, self.v, self.dr, self.dc


@dataclass(frozen=True, slots=True)
class DSUOutcome:
    selection: SelectedRelation
    accepted: bool
    reason: str
    tree_merge: bool
    cycle: bool


@dataclass(frozen=True, slots=True)
class RelationDecodeResult:
    qualified: tuple[SelectedRelation, ...]
    attempted: tuple[SelectedRelation, ...]
    outcomes: tuple[DSUOutcome, ...]
    components: tuple[dict[int, tuple[int, int]], ...]
    attempt_cap: int
    tree_merges: int
    cycle_acceptances: int
    rejection_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _Geometry:
    size: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    height: int
    width: int
    density: float


@dataclass(frozen=True, slots=True)
class _Evidence:
    logprob: np.ndarray
    rank: np.ndarray
    margin: np.ndarray
    none_logprob: np.ndarray | None
    none_rank: np.ndarray | None
    none_margin: np.ndarray | None


def _readonly(value: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _validate_table(table: RelationQueryTable) -> RelationQueryTable:
    if type(table) is not RelationQueryTable:
        raise RelationVerifierError("table must be an exact RelationQueryTable")
    features = table.features
    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or not features.flags.c_contiguous
        or not np.isfinite(features).all()
    ):
        raise RelationVerifierError("feature matrix contract failed")
    rows = int(features.shape[0])
    for value, dtype, shape, name in (
        (table.hypothesis_ids, np.int64, (rows,), "hypothesis_ids"),
        (table.relation_ids, np.int64, (rows,), "relation_ids"),
        (table.relations, np.int64, (rows, 4), "relations"),
        (table.row_kind, np.uint8, (rows,), "row_kind"),
        (table.support, np.int64, (rows,), "support"),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise RelationVerifierError(f"{name} contract failed")
    query_offsets = table.query_offsets
    if (
        not isinstance(query_offsets, np.ndarray)
        or query_offsets.dtype != np.int64
        or query_offsets.ndim != 1
        or query_offsets.size < 2
        or int(query_offsets[0]) != 0
        or int(query_offsets[-1]) != rows
        or bool((np.diff(query_offsets) < 2).any())
    ):
        raise RelationVerifierError("query offsets contract failed")
    scene_offsets = table.scene_offsets
    if (
        not isinstance(scene_offsets, np.ndarray)
        or scene_offsets.dtype != np.int64
        or scene_offsets.ndim != 1
        or scene_offsets.size < 2
        or int(scene_offsets[0]) != 0
        or int(scene_offsets[-1]) != rows
        or bool((np.diff(scene_offsets) <= 0).any())
        or not set(scene_offsets.tolist()).issubset(set(query_offsets.tolist()))
    ):
        raise RelationVerifierError("scene offsets contract failed")
    scene_boundaries = set(map(int, scene_offsets[:-1].tolist()))
    previous_pair: tuple[int, int] | None = None
    for start, stop in zip(query_offsets[:-1], query_offsets[1:]):
        first, last = int(start), int(stop)
        if first in scene_boundaries:
            previous_pair = None
        if table.row_kind[last - 1] != ROW_NONE or bool(
            (table.row_kind[first : last - 1] != ROW_OFFSET).any()
        ):
            raise RelationVerifierError("each query must end in exactly one NONE row")
        pair = tuple(map(int, table.relations[first, :2]))
        if pair[0] >= pair[1] or not bool(
            np.all(table.relations[first:last, :2] == table.relations[first, :2])
        ):
            raise RelationVerifierError("query pair is not canonical")
        if previous_pair is not None and pair <= previous_pair:
            raise RelationVerifierError("query order is not canonical")
        previous_pair = pair
        offsets = [tuple(map(int, row)) for row in table.relations[first : last - 1]]
        if offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
            raise RelationVerifierError("offset rows are not canonical and unique")
        if (
            int(table.hypothesis_ids[last - 1]) != NONE_ID
            or int(table.relation_ids[last - 1]) != NONE_ID
            or int(table.support[last - 1]) != 0
        ):
            raise RelationVerifierError("NONE metadata drifted")
    return table


def _geometry(component: e23.RigidComponent) -> _Geometry:
    rows = [int(row) for _tile, row, _col in component.entries]
    cols = [int(col) for _tile, _row, col in component.entries]
    if not rows:
        raise RelationVerifierError("E23 component is empty")
    height, width = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    return _Geometry(
        size=len(rows),
        min_row=min(rows),
        max_row=max(rows),
        min_col=min(cols),
        max_col=max(cols),
        height=height,
        width=width,
        density=float(len(rows) / (height * width)),
    )


def _validate_inputs(
    result: e23.CandidatePoolResult,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    i21_logits: np.ndarray,
    contextual_pair_logits: np.ndarray,
    contextual_none_logits: np.ndarray | None,
) -> tuple[tuple[_Geometry, ...], np.ndarray]:
    if type(result) is not e23.CandidatePoolResult:
        raise RelationVerifierError("result must be the exact frozen E23 result")
    if (
        not isinstance(candidate_ids, np.ndarray)
        or candidate_ids.shape != (NUM_TILES, RAW_WIDTH)
        or candidate_ids.dtype != np.int64
        or not candidate_ids.flags.c_contiguous
    ):
        raise RelationVerifierError("candidate_ids must be contiguous int64[576,128]")
    if (
        not isinstance(raw_logits, np.ndarray)
        or raw_logits.shape != (NUM_DIRECTIONS, NUM_TILES, RAW_WIDTH)
        or raw_logits.dtype != np.float32
        or not raw_logits.flags.c_contiguous
        or bool(np.isnan(raw_logits).any())
        or bool(np.isposinf(raw_logits).any())
    ):
        raise RelationVerifierError("raw_logits must be float32[4,576,128]")
    valid = np.isfinite(raw_logits)
    if not all(np.array_equal(valid[0], valid[d]) for d in range(1, 4)):
        raise RelationVerifierError("raw finite masks differ by direction")
    if not bool(valid[0].any(axis=1).all()) or not bool(
        np.isneginf(raw_logits[~valid]).all()
    ):
        raise RelationVerifierError("raw rows are empty or padding is not -inf")
    valid_ids = candidate_ids[valid[0]]
    if bool(((valid_ids < 0) | (valid_ids >= NUM_TILES)).any()):
        raise RelationVerifierError("raw candidate target is outside the tile bag")
    for name, value in (
        ("i21_logits", i21_logits),
        ("contextual_pair_logits", contextual_pair_logits),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (4, NUM_TILES, NUM_TILES)
            or value.dtype != np.float32
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise RelationVerifierError(f"{name} must be finite float32[4,576,576]")
    if not np.array_equal(contextual_pair_logits[LEFT], contextual_pair_logits[RIGHT].T):
        raise RelationVerifierError("contextual LEFT logits must equal RIGHT transpose")
    if not np.array_equal(contextual_pair_logits[UP], contextual_pair_logits[DOWN].T):
        raise RelationVerifierError("contextual UP logits must equal DOWN transpose")
    if contextual_none_logits is not None and (
        not isinstance(contextual_none_logits, np.ndarray)
        or contextual_none_logits.shape != (4, NUM_TILES)
        or contextual_none_logits.dtype != np.float32
        or not contextual_none_logits.flags.c_contiguous
        or not np.isfinite(contextual_none_logits).all()
    ):
        raise RelationVerifierError("contextual_none_logits must be float32[4,576]")

    components = result.components
    if (
        not components
        or tuple(component.component_id for component in components)
        != tuple(range(len(components)))
    ):
        raise RelationVerifierError("E23 components are not canonical")
    owner = np.asarray(result.owner)
    local_rows, local_cols = np.asarray(result.local_rows), np.asarray(result.local_cols)
    if any(
        value.shape != (NUM_TILES,) or value.dtype != np.int64
        for value in (owner, local_rows, local_cols)
    ):
        raise RelationVerifierError("E23 owner/local arrays must be int64[576]")
    seen: set[int] = set()
    geometries: list[_Geometry] = []
    for component in components:
        if type(component) is not e23.RigidComponent:
            raise RelationVerifierError("E23 component has the wrong type")
        occupied: set[tuple[int, int]] = set()
        for tile, row, col in component.entries:
            tile_i, row_i, col_i = int(tile), int(row), int(col)
            if tile_i in seen or not 0 <= tile_i < NUM_TILES or (row_i, col_i) in occupied:
                raise RelationVerifierError("E23 component partition is invalid")
            if (
                int(owner[tile_i]) != component.component_id
                or int(local_rows[tile_i]) != row_i
                or int(local_cols[tile_i]) != col_i
            ):
                raise RelationVerifierError("E23 owner/local binding drifted")
            seen.add(tile_i)
            occupied.add((row_i, col_i))
        one = _geometry(component)
        if one.height > GRID or one.width > GRID:
            raise RelationVerifierError("E23 component exceeds board span")
        geometries.append(one)
    if len(seen) != NUM_TILES:
        raise RelationVerifierError("E23 components do not partition 576 tiles")

    hypotheses = result.hypotheses
    if not hypotheses or tuple(h.hypothesis_id for h in hypotheses) != tuple(
        range(len(hypotheses))
    ):
        raise RelationVerifierError("E23 hypotheses are empty or non-contiguous")
    previous: tuple[int, int, int, int] | None = None
    for hypothesis in hypotheses:
        relation = hypothesis.relation
        if (
            not 0 <= hypothesis.u < hypothesis.v < len(components)
            or previous is not None
            and relation <= previous
            or not hypothesis.claim_ids
            or tuple(sorted(hypothesis.claim_ids)) != hypothesis.claim_ids
        ):
            raise RelationVerifierError("E23 hypotheses are not canonical")
        if not 0 <= hypothesis.relation_id < len(result.relation_candidates):
            raise RelationVerifierError("E23 hypothesis relation ID is invalid")
        source = result.relation_candidates[hypothesis.relation_id]
        if source.relation != relation or source.claim_ids != hypothesis.claim_ids:
            raise RelationVerifierError("E23 hypothesis binding drifted")
        for claim_id in hypothesis.claim_ids:
            if not 0 <= int(claim_id) < len(result.claims):
                raise RelationVerifierError("E23 hypothesis claim ID is invalid")
        previous = relation
    return tuple(geometries), valid


def _dense_evidence(
    pair_logits: np.ndarray, none_logits: np.ndarray | None
) -> _Evidence:
    values = pair_logits.astype(np.float64, copy=True)
    diagonal = np.arange(NUM_TILES)
    values[:, diagonal, diagonal] = -np.inf
    if none_logits is not None:
        full = np.concatenate((values, none_logits[..., None].astype(np.float64)), axis=2)
    else:
        full = values
    maximum = np.max(full, axis=2, keepdims=True)
    shifted = full - maximum
    denominator = np.log(np.exp(shifted).sum(axis=2, keepdims=True)) + maximum
    logprob_full = full - denominator
    # Ties receive an identical percentile (strictly-lower fraction), so a
    # numeric target ID can never leak through a score-rank feature.
    order = np.argsort(full, axis=2, kind="stable")
    ordered_values = np.take_along_axis(full, order, axis=2)
    positions = np.broadcast_to(np.arange(full.shape[2]), order.shape)
    changes = np.concatenate(
        (
            np.ones((*full.shape[:2], 1), dtype=np.bool_),
            ordered_values[:, :, 1:] != ordered_values[:, :, :-1],
        ),
        axis=2,
    )
    left_positions = np.maximum.accumulate(
        np.where(changes, positions, 0), axis=2
    )
    rank_index = np.empty_like(order)
    np.put_along_axis(
        rank_index,
        order,
        left_positions,
        axis=2,
    )
    eligible = full.shape[2] - 1  # the masked self-target is never eligible
    rank = np.clip((rank_index - 1) / max(1, eligible - 1), 0.0, 1.0)
    best = np.max(full, axis=2, keepdims=True)
    winner_count = np.sum(full == best, axis=2, keepdims=True)
    second = np.partition(full, -2, axis=2)[:, :, -2][:, :, None]
    best_other = np.where((full == best) & (winner_count == 1), second, best)
    margin = full - best_other
    targets = slice(0, NUM_TILES)
    none_logprob = (
        np.ascontiguousarray(logprob_full[:, :, -1], dtype=np.float32)
        if none_logits is not None
        else None
    )
    none_rank = (
        np.ascontiguousarray(rank[:, :, -1], dtype=np.float32)
        if none_logits is not None
        else None
    )
    none_margin = (
        np.ascontiguousarray(margin[:, :, -1], dtype=np.float32)
        if none_logits is not None
        else None
    )
    return _Evidence(
        logprob=np.ascontiguousarray(logprob_full[:, :, targets], dtype=np.float32),
        rank=np.ascontiguousarray(rank[:, :, targets], dtype=np.float32),
        margin=np.ascontiguousarray(margin[:, :, targets], dtype=np.float32),
        none_logprob=none_logprob,
        none_rank=none_rank,
        none_margin=none_margin,
    )


def _raw_evidence(raw_logits: np.ndarray, valid: np.ndarray) -> _Evidence:
    logprob = np.zeros_like(raw_logits, dtype=np.float32)
    rank = np.zeros_like(raw_logits, dtype=np.float32)
    margin = np.zeros_like(raw_logits, dtype=np.float32)
    for direction in range(NUM_DIRECTIONS):
        for source in range(NUM_TILES):
            mask = valid[direction, source]
            values = raw_logits[direction, source, mask].astype(np.float64)
            maximum = float(values.max())
            logprob[direction, source, mask] = (
                values - maximum - log(float(np.exp(values - maximum).sum()))
            ).astype(np.float32)
            order = np.argsort(values, kind="stable")
            ordered = values[order]
            changes = np.concatenate(
                (np.asarray((True,)), ordered[1:] != ordered[:-1])
            )
            left_positions = np.maximum.accumulate(
                np.where(changes, np.arange(values.size), 0)
            )
            ranks = np.empty(values.size, dtype=np.int64)
            ranks[order] = left_positions
            rank[direction, source, mask] = (
                ranks / max(1, values.size - 1)
            ).astype(np.float32)
            best = float(values.max())
            winners = values == best
            second = float(np.partition(values, -2)[-2]) if values.size > 1 else best
            best_other = np.where(winners & (int(winners.sum()) == 1), second, best)
            margin[direction, source, mask] = (values - best_other).astype(np.float32)
    return _Evidence(logprob, rank, margin, None, None, None)


def _claim_directions(claim: e23.RCCE4Claim) -> tuple[tuple[int, int, int], ...]:
    direction = RIGHT if int(claim.dx) == 1 else DOWN
    inverse = LEFT if direction == RIGHT else UP
    return (
        (direction, int(claim.first), int(claim.second)),
        (inverse, int(claim.second), int(claim.first)),
    )


def _put_evidence(row: np.ndarray, prefix: str, values: Sequence[tuple[float, float, float]]) -> None:
    if not values:
        return
    array = np.asarray(values, dtype=np.float64)
    logs, ranks, margins = array[:, 0], array[:, 1], array[:, 2]
    log_maximum = float(logs.max())
    rank_maximum = float(ranks.max())
    margin_maximum = float(margins.max())
    stats = {
        "logprob_min": float(logs.min()),
        "logprob_mean": float(logs.mean()),
        "logprob_max": log_maximum,
        "logprob_logsumexp": log_maximum + log(float(np.exp(logs - log_maximum).sum())),
        "rank_min": float(ranks.min()),
        "rank_mean": float(ranks.mean()),
        "rank_max": rank_maximum,
        "rank_logsumexp": rank_maximum + log(float(np.exp(ranks - rank_maximum).sum())),
        "margin_min": float(margins.min()),
        "margin_mean": float(margins.mean()),
        "margin_max": margin_maximum,
        "margin_logsumexp": margin_maximum + log(
            float(np.exp(margins - margin_maximum).sum())
        ),
    }
    for suffix, value in stats.items():
        row[FEATURE_INDEX[f"{prefix}_{suffix}"]] = np.float32(value)


def _put_geometry(
    row: np.ndarray, first: _Geometry, second: _Geometry, dr: int | None, dc: int | None
) -> None:
    small, large = min(first.size, second.size), max(first.size, second.size)
    values = {
        "size_min_log1p": np.log1p(small),
        "size_max_log1p": np.log1p(large),
        "size_ratio": small / large,
        "height_min_scaled": min(first.height, second.height) / GRID,
        "height_max_scaled": max(first.height, second.height) / GRID,
        "width_min_scaled": min(first.width, second.width) / GRID,
        "width_max_scaled": max(first.width, second.width) / GRID,
        "density_min": min(first.density, second.density),
        "density_max": max(first.density, second.density),
    }
    if dr is not None and dc is not None:
        min_row = min(first.min_row, second.min_row + dr)
        max_row = max(first.max_row, second.max_row + dr)
        min_col = min(first.min_col, second.min_col + dc)
        max_col = max(first.max_col, second.max_col + dc)
        height, width = max_row - min_row + 1, max_col - min_col + 1
        values.update(
            {
                "merged_height_scaled": height / GRID,
                "merged_width_scaled": width / GRID,
                "merged_density": (first.size + second.size) / (height * width),
                "offset_l1_scaled": (abs(dr) + abs(dc)) / (2 * GRID),
            }
        )
    for name, value in values.items():
        row[FEATURE_INDEX[name]] = np.float32(value)


def extract_relation_queries(
    result: e23.CandidatePoolResult,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    i21_logits: np.ndarray,
    contextual_pair_logits: np.ndarray,
    contextual_none_logits: np.ndarray | None = None,
) -> RelationQueryTable:
    """Extract exactly 64 label-free features for all offsets plus ``NONE``.

    ``contextual_pair_logits`` and ``contextual_none_logits`` are the NumPy
    outputs corresponding to E26 contextual model keys ``pair_logits`` and
    ``none_logits``.  The signature intentionally has no extensible kwargs.
    """

    geometries, valid = _validate_inputs(
        result,
        candidate_ids,
        raw_logits,
        i21_logits,
        contextual_pair_logits,
        contextual_none_logits,
    )
    raw = _raw_evidence(raw_logits, valid)
    i21 = _dense_evidence(i21_logits, None)
    context = _dense_evidence(contextual_pair_logits, contextual_none_logits)

    hypotheses = result.hypotheses
    group_starts = [0]
    for index in range(1, len(hypotheses)):
        if (hypotheses[index].u, hypotheses[index].v) != (
            hypotheses[index - 1].u,
            hypotheses[index - 1].v,
        ):
            group_starts.append(index)
    group_starts.append(len(hypotheses))
    groups = tuple(zip(group_starts[:-1], group_starts[1:]))
    output_rows = len(hypotheses) + len(groups)
    features = np.zeros((output_rows, len(FEATURE_NAMES)), dtype=np.float32)
    hypothesis_ids = np.full(output_rows, NONE_ID, dtype=np.int64)
    relation_ids = np.full(output_rows, NONE_ID, dtype=np.int64)
    relations = np.zeros((output_rows, 4), dtype=np.int64)
    row_kind = np.full(output_rows, ROW_OFFSET, dtype=np.uint8)
    support = np.zeros(output_rows, dtype=np.int64)
    query_offsets = np.empty(len(groups) + 1, dtype=np.int64)
    query_offsets[0] = 0
    cursor = 0

    for query_index, (start, stop) in enumerate(groups):
        offset_rows: list[int] = []
        context_means: list[float] = []
        query_none_values: list[float] = []
        for source_index in range(start, stop):
            hypothesis = hypotheses[source_index]
            out = cursor
            offset_rows.append(out)
            cursor += 1
            row = features[out]
            row[FEATURE_INDEX["has_offset"]] = 1.0
            u, v, dr, dc = map(
                int, (hypothesis.u, hypothesis.v, hypothesis.dr, hypothesis.dc)
            )
            hypothesis_ids[out] = hypothesis.hypothesis_id
            relation_ids[out] = hypothesis.relation_id
            relations[out] = (u, v, dr, dc)
            support[out] = len(hypothesis.claim_ids)
            _put_geometry(row, geometries[u], geometries[v], dr, dc)

            base_count = 0
            context_values: list[tuple[float, float, float]] = []
            i21_values: list[tuple[float, float, float]] = []
            raw_values: list[tuple[float, float, float]] = []
            for claim_id in hypothesis.claim_ids:
                claim = result.claims[int(claim_id)]
                base_count += int(int(claim.pair_id) < len(result.base_affinity_pairs))
                for direction, source, target in _claim_directions(claim):
                    context_values.append(
                        (
                            float(context.logprob[direction, source, target]),
                            float(context.rank[direction, source, target]),
                            float(context.margin[direction, source, target]),
                        )
                    )
                    i21_values.append(
                        (
                            float(i21.logprob[direction, source, target]),
                            float(i21.rank[direction, source, target]),
                            float(i21.margin[direction, source, target]),
                        )
                    )
                    if context.none_logprob is not None:
                        query_none_values.append(
                            float(context.none_logprob[direction, source])
                        )
                for observation in claim.observations:
                    direction, source, slot = (
                        int(observation.direction),
                        int(observation.source),
                        int(observation.slot),
                    )
                    if (
                        not bool(valid[direction, source, slot])
                        or int(candidate_ids[source, slot]) != int(observation.target)
                        or float(raw_logits[direction, source, slot])
                        != float(observation.logit)
                    ):
                        raise RelationVerifierError("raw claim observation binding drifted")
                    raw_values.append(
                        (
                            float(raw.logprob[direction, source, slot]),
                            float(raw.rank[direction, source, slot]),
                            float(raw.margin[direction, source, slot]),
                        )
                    )
            count = len(hypothesis.claim_ids)
            residual_count = count - base_count
            for name, value in (
                ("claim_count_log1p", np.log1p(count)),
                ("base_claim_count_log1p", np.log1p(base_count)),
                ("residual_claim_count_log1p", np.log1p(residual_count)),
                ("base_claim_fraction", base_count / count),
            ):
                row[FEATURE_INDEX[name]] = np.float32(value)
            _put_evidence(row, "context", context_values)
            _put_evidence(row, "i21", i21_values)
            _put_evidence(row, "raw", raw_values)
            context_means.append(float(row[FEATURE_INDEX["context_logprob_mean"]]))

        ranked = sorted(
            range(len(offset_rows)),
            key=lambda idx: (
                -context_means[idx],
                int(relations[offset_rows[idx], 2]),
                int(relations[offset_rows[idx], 3]),
            ),
        )
        best = context_means[ranked[0]]
        second = context_means[ranked[1]] if len(ranked) > 1 else best
        query_support = int(sum(support[row] for row in offset_rows))
        none_mean = float(np.mean(query_none_values)) if query_none_values else 0.0
        for rank_index, local_index in enumerate(ranked):
            row_index = offset_rows[local_index]
            row = features[row_index]
            own = context_means[local_index]
            other = max(
                (value for index, value in enumerate(context_means) if index != local_index),
                default=own,
            )
            for name, value in (
                ("query_offset_count_log1p", np.log1p(len(offset_rows))),
                ("query_support_sum_log1p", np.log1p(query_support)),
                ("query_best_context_mean", best),
                ("query_second_context_mean", second),
                ("query_context_gap", best - second),
                (
                    "offset_context_rank_percentile",
                    (len(ranked) - 1 - rank_index) / max(1, len(ranked) - 1),
                ),
                ("offset_context_gap_best", own - best),
                ("offset_context_margin_other", own - other),
                ("query_none_context_mean", none_mean),
            ):
                row[FEATURE_INDEX[name]] = np.float32(value)

        none_index = cursor
        cursor += 1
        row_kind[none_index] = ROW_NONE
        features[none_index, FEATURE_INDEX["is_none"]] = 1.0
        u, v = map(int, relations[offset_rows[0], :2])
        relations[none_index] = (u, v, 0, 0)
        _put_geometry(features[none_index], geometries[u], geometries[v], None, None)
        for name, value in (
            ("query_offset_count_log1p", np.log1p(len(offset_rows))),
            ("query_support_sum_log1p", np.log1p(query_support)),
            ("query_best_context_mean", best),
            ("query_second_context_mean", second),
            ("query_context_gap", best - second),
            ("query_none_context_mean", none_mean),
        ):
            features[none_index, FEATURE_INDEX[name]] = np.float32(value)
        if query_none_values:
            none_evidence: list[tuple[float, float, float]] = []
            for source_index in range(start, stop):
                hypothesis = hypotheses[source_index]
                for claim_id in hypothesis.claim_ids:
                    claim = result.claims[int(claim_id)]
                    for direction, source, _target in _claim_directions(claim):
                        if (
                            context.none_logprob is None
                            or context.none_rank is None
                            or context.none_margin is None
                        ):
                            raise AssertionError("context NONE statistics disappeared")
                        none_evidence.append(
                            (
                                float(context.none_logprob[direction, source]),
                                float(context.none_rank[direction, source]),
                                float(context.none_margin[direction, source]),
                            )
                        )
            _put_evidence(features[none_index], "context", none_evidence)
        query_offsets[query_index + 1] = cursor

    if cursor != output_rows or not np.isfinite(features).all():
        raise RelationVerifierError("E26 feature materialization drifted")
    table = RelationQueryTable(
        features=_readonly(features, np.float32),
        hypothesis_ids=_readonly(hypothesis_ids, np.int64),
        relation_ids=_readonly(relation_ids, np.int64),
        relations=_readonly(relations, np.int64),
        row_kind=_readonly(row_kind, np.uint8),
        support=_readonly(support, np.int64),
        query_offsets=_readonly(query_offsets, np.int64),
        scene_offsets=_readonly(np.asarray((0, output_rows)), np.int64),
    )
    return _validate_table(table)


def concatenate_query_tables(tables: Sequence[RelationQueryTable]) -> RelationQueryTable:
    values = tuple(_validate_table(table) for table in tables)
    if not values:
        raise RelationVerifierError("at least one query table is required")
    row_counts = np.asarray([table.rows for table in values], dtype=np.int64)
    row_bases = np.concatenate((np.asarray((0,), dtype=np.int64), np.cumsum(row_counts)))
    query_offsets = [0]
    for base, table in zip(row_bases[:-1], values):
        query_offsets.extend((table.query_offsets[1:] + int(base)).tolist())
    combined = RelationQueryTable(
        features=_readonly(np.concatenate([x.features for x in values]), np.float32),
        hypothesis_ids=_readonly(np.concatenate([x.hypothesis_ids for x in values]), np.int64),
        relation_ids=_readonly(np.concatenate([x.relation_ids for x in values]), np.int64),
        relations=_readonly(np.concatenate([x.relations for x in values]), np.int64),
        row_kind=_readonly(np.concatenate([x.row_kind for x in values]), np.uint8),
        support=_readonly(np.concatenate([x.support for x in values]), np.int64),
        query_offsets=_readonly(np.asarray(query_offsets), np.int64),
        scene_offsets=_readonly(row_bases, np.int64),
    )
    return _validate_table(combined)


def calibrated_probabilities(
    table: RelationQueryTable,
    row_logits: np.ndarray,
    edge_logits: np.ndarray,
    *,
    row_temperature: float = 1.0,
    edge_temperature: float = 1.0,
) -> CalibratedRelationProbabilities:
    value = _validate_table(table)
    rows = np.asarray(row_logits, dtype=np.float64)
    edges = np.asarray(edge_logits, dtype=np.float64)
    if rows.shape != (value.rows,) or not np.isfinite(rows).all():
        raise RelationVerifierError("row logits must be finite and row-aligned")
    if edges.shape != (value.queries,) or not np.isfinite(edges).all():
        raise RelationVerifierError("edge logits must be finite and query-aligned")
    if not 0.05 <= float(row_temperature) <= 20.0 or not 0.05 <= float(edge_temperature) <= 20.0:
        raise RelationVerifierError("calibration temperatures must lie in [0.05,20]")
    probabilities = np.empty(value.rows, dtype=np.float64)
    for start, stop in zip(value.query_offsets[:-1], value.query_offsets[1:]):
        first, last = int(start), int(stop)
        logits = rows[first:last] / float(row_temperature)
        shifted = logits - float(logits.max())
        one = np.exp(shifted)
        probabilities[first:last] = one / float(one.sum())
    scaled_edges = np.clip(edges / float(edge_temperature), -80.0, 80.0)
    edge_probabilities = 1.0 / (1.0 + np.exp(-scaled_edges))
    return CalibratedRelationProbabilities(
        row_probabilities=_readonly(probabilities, np.float64),
        edge_probabilities=_readonly(edge_probabilities, np.float64),
        row_temperature=float(row_temperature),
        edge_temperature=float(edge_temperature),
    )


def select_qualified_relations(
    table: RelationQueryTable,
    probabilities: CalibratedRelationProbabilities,
    *,
    component_count: int,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> tuple[tuple[SelectedRelation, ...], tuple[SelectedRelation, ...], int]:
    value = _validate_table(table)
    if probabilities.row_probabilities.shape != (value.rows,) or probabilities.edge_probabilities.shape != (value.queries,):
        raise RelationVerifierError("calibrated probabilities are not table-aligned")
    if not np.isfinite(probabilities.row_probabilities).all() or not np.isfinite(probabilities.edge_probabilities).all():
        raise RelationVerifierError("calibrated probabilities must be finite")
    if bool(
        ((probabilities.row_probabilities < 0.0) | (probabilities.row_probabilities > 1.0)).any()
    ) or bool(
        ((probabilities.edge_probabilities < 0.0) | (probabilities.edge_probabilities > 1.0)).any()
    ):
        raise RelationVerifierError("calibrated probabilities must lie in [0,1]")
    if any(
        not np.isclose(
            float(probabilities.row_probabilities[int(start) : int(stop)].sum()),
            1.0,
            rtol=1.0e-7,
            atol=1.0e-9,
        )
        for start, stop in zip(value.query_offsets[:-1], value.query_offsets[1:])
    ):
        raise RelationVerifierError("row probabilities must sum to one per query")
    if not 0.0 <= float(edge_threshold) < 1.0:
        raise RelationVerifierError("edge threshold must lie in [0,1)")
    if int(component_count) != component_count or component_count < 1:
        raise RelationVerifierError("component count must be positive")
    qualified: list[SelectedRelation] = []
    for query, (start, stop) in enumerate(zip(value.query_offsets[:-1], value.query_offsets[1:])):
        first, last = int(start), int(stop)
        none = last - 1
        best = min(
            range(first, none),
            key=lambda row: (
                -float(probabilities.row_probabilities[row]),
                int(value.relations[row, 2]),
                int(value.relations[row, 3]),
            ),
        )
        edge_probability = float(probabilities.edge_probabilities[query])
        offset_probability = float(probabilities.row_probabilities[best])
        none_probability = float(probabilities.row_probabilities[none])
        margin = offset_probability - none_probability
        if not edge_probability > edge_threshold or not margin > 0.0:
            continue
        u, v, dr, dc = map(int, value.relations[best])
        qualified.append(
            SelectedRelation(
                hypothesis_id=int(value.hypothesis_ids[best]),
                relation_id=int(value.relation_ids[best]),
                u=u,
                v=v,
                dr=dr,
                dc=dc,
                edge_probability=edge_probability,
                offset_probability=offset_probability,
                none_probability=none_probability,
                probability_margin=margin,
                support=int(value.support[best]),
            )
        )
    qualified.sort(
        key=lambda item: (
            -item.edge_probability,
            -item.probability_margin,
            item.u,
            item.v,
            item.dr,
            item.dc,
        )
    )
    upper_bound = max(0, ATTEMPT_MULTIPLIER * (int(component_count) - 1))
    attempted = tuple(qualified[:upper_bound])
    return tuple(qualified), attempted, upper_bound


class _PotentialDSU:
    """Signed-potential DSU; every rejection is representation immutable."""

    def __init__(self, result: e23.CandidatePoolResult) -> None:
        if type(result) is not e23.CandidatePoolResult:
            raise RelationVerifierError("DSU requires the exact E23 result")
        self.result = result
        self.parent = {c.component_id: c.component_id for c in result.components}
        self.size = {c.component_id: 1 for c in result.components}
        self.delta = {c.component_id: (0, 0) for c in result.components}
        self.entries: dict[int, dict[tuple[int, int], int]] = {}
        self.translations: dict[int, dict[int, tuple[int, int]]] = {}
        for component in result.components:
            entries = {(int(row), int(col)): int(tile) for tile, row, col in component.entries}
            if len(entries) != len(component.entries):
                raise RelationVerifierError("component collision reached DSU")
            self.entries[component.component_id] = entries
            self.translations[component.component_id] = {component.component_id: (0, 0)}

    def find(self, component: int) -> tuple[int, tuple[int, int]]:
        if component not in self.parent:
            raise RelationVerifierError("unknown component in relation")
        parent = self.parent[component]
        if parent == component:
            return component, (0, 0)
        root, parent_delta = self.find(parent)
        own = self.delta[component]
        return root, (own[0] + parent_delta[0], own[1] + parent_delta[1])

    def state_signature(self) -> tuple[object, ...]:
        return (
            tuple(sorted(self.parent.items())),
            tuple(sorted(self.size.items())),
            tuple(sorted(self.delta.items())),
            tuple((root, tuple(sorted(values.items()))) for root, values in sorted(self.entries.items())),
            tuple((root, tuple(sorted(values.items()))) for root, values in sorted(self.translations.items())),
        )

    def _contacts_valid(
        self, selection: SelectedRelation, translations: Mapping[int, tuple[int, int]]
    ) -> bool:
        hypothesis = self.result.hypotheses[selection.hypothesis_id]
        if hypothesis.relation_id != selection.relation_id or hypothesis.relation != selection.relation:
            raise RelationVerifierError("selection does not bind to its E23 hypothesis")
        for claim_id in hypothesis.claim_ids:
            claim = self.result.claims[int(claim_id)]
            first_component = int(self.result.owner[claim.first])
            second_component = int(self.result.owner[claim.second])
            if first_component not in translations or second_component not in translations:
                return False
            first = (
                translations[first_component][0] + int(self.result.local_rows[claim.first]),
                translations[first_component][1] + int(self.result.local_cols[claim.first]),
            )
            second = (
                translations[second_component][0] + int(self.result.local_rows[claim.second]),
                translations[second_component][1] + int(self.result.local_cols[claim.second]),
            )
            if (second[0] - first[0], second[1] - first[1]) != (claim.dy, claim.dx):
                return False
        return True

    def try_accept(self, selection: SelectedRelation) -> tuple[bool, str, bool, bool]:
        root_u, shift_u = self.find(selection.u)
        root_v, shift_v = self.find(selection.v)
        if root_u == root_v:
            observed = (shift_v[0] - shift_u[0], shift_v[1] - shift_u[1])
            if observed != (selection.dr, selection.dc):
                return False, "conflict", False, False
            if not self._contacts_valid(selection, self.translations[root_u]):
                return False, "contact", False, False
            return True, "cycle", False, True

        root_v_from_u = (
            selection.dr + shift_u[0] - shift_v[0],
            selection.dc + shift_u[1] - shift_v[1],
        )
        proposed_translations = dict(self.translations[root_u])
        proposed_translations.update(
            {
                component: (position[0] + root_v_from_u[0], position[1] + root_v_from_u[1])
                for component, position in self.translations[root_v].items()
            }
        )
        if not self._contacts_valid(selection, proposed_translations):
            return False, "contact", False, False
        shifted_v = {
            (position[0] + root_v_from_u[0], position[1] + root_v_from_u[1]): tile
            for position, tile in self.entries[root_v].items()
        }
        if set(self.entries[root_u]).intersection(shifted_v):
            return False, "collision", False, False
        proposed_entries = dict(self.entries[root_u])
        proposed_entries.update(shifted_v)
        rows, cols = zip(*proposed_entries)
        if max(rows) - min(rows) + 1 > GRID or max(cols) - min(cols) + 1 > GRID:
            return False, "span", False, False

        keep_u = (self.size[root_u], -root_u) >= (self.size[root_v], -root_v)
        if keep_u:
            self.parent[root_v] = root_u
            self.delta[root_v] = root_v_from_u
            self.size[root_u] += self.size[root_v]
            self.entries[root_u] = proposed_entries
            self.translations[root_u] = proposed_translations
            del self.entries[root_v]
            del self.translations[root_v]
        else:
            root_u_from_v = (-root_v_from_u[0], -root_v_from_u[1])
            shifted_u = {
                (position[0] + root_u_from_v[0], position[1] + root_u_from_v[1]): tile
                for position, tile in self.entries[root_u].items()
            }
            retained_entries = dict(self.entries[root_v])
            retained_entries.update(shifted_u)
            retained_translations = dict(self.translations[root_v])
            retained_translations.update(
                {
                    component: (position[0] + root_u_from_v[0], position[1] + root_u_from_v[1])
                    for component, position in self.translations[root_u].items()
                }
            )
            self.parent[root_u] = root_v
            self.delta[root_u] = root_u_from_v
            self.size[root_v] += self.size[root_u]
            self.entries[root_v] = retained_entries
            self.translations[root_v] = retained_translations
            del self.entries[root_u]
            del self.translations[root_u]
        return True, "tree", True, False

    def normalized_components(self) -> tuple[dict[int, tuple[int, int]], ...]:
        output: list[dict[int, tuple[int, int]]] = []
        for root in sorted(self.entries, key=lambda value: (-len(self.entries[value]), value)):
            entries = self.entries[root]
            minimum_row = min(row for row, _col in entries)
            minimum_col = min(col for _row, col in entries)
            output.append(
                dict(
                    sorted(
                        (tile, (row - minimum_row, col - minimum_col))
                        for (row, col), tile in entries.items()
                    )
                )
            )
        if sum(map(len, output)) != NUM_TILES:
            raise RelationVerifierError("decoder lost or duplicated tiles")
        return tuple(output)


def decode_relation_probabilities(
    result: e23.CandidatePoolResult,
    table: RelationQueryTable,
    probabilities: CalibratedRelationProbabilities,
    *,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> RelationDecodeResult:
    if type(result) is not e23.CandidatePoolResult:
        raise RelationVerifierError("decoder requires the exact E23 result")
    qualified, attempted, cap = select_qualified_relations(
        table,
        probabilities,
        component_count=len(result.components),
        edge_threshold=edge_threshold,
    )
    dsu = _PotentialDSU(result)
    outcomes: list[DSUOutcome] = []
    rejections: dict[str, int] = {}
    trees = cycles = 0
    for selection in attempted:
        before = dsu.state_signature()
        accepted, reason, tree, cycle = dsu.try_accept(selection)
        if not accepted and dsu.state_signature() != before:
            raise RelationVerifierError("rejected DSU proposal mutated state")
        outcomes.append(DSUOutcome(selection, accepted, reason, tree, cycle))
        trees += int(tree)
        cycles += int(cycle)
        if not accepted:
            rejections[reason] = rejections.get(reason, 0) + 1
    return RelationDecodeResult(
        qualified=qualified,
        attempted=attempted,
        outcomes=tuple(outcomes),
        components=dsu.normalized_components(),
        attempt_cap=cap,
        tree_merges=trees,
        cycle_acceptances=cycles,
        rejection_counts=dict(sorted(rejections.items())),
    )


__all__ = (
    "ATTEMPT_MULTIPLIER",
    "CalibratedRelationProbabilities",
    "DEFAULT_EDGE_THRESHOLD",
    "DSUOutcome",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "NONE_ID",
    "ROW_NONE",
    "ROW_OFFSET",
    "RelationDecodeResult",
    "RelationQueryTable",
    "RelationVerifierError",
    "SelectedRelation",
    "_PotentialDSU",
    "calibrated_probabilities",
    "concatenate_query_tables",
    "decode_relation_probabilities",
    "extract_relation_queries",
    "select_qualified_relations",
)
