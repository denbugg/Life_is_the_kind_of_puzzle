"""Frozen label-free E22 RCCE-4 candidate generator.

The core consumes only the exact production Rank96 affinity-union IDs and
direction-major raw logits.  It reconstructs raw CC96 geometry internally,
enumerates the four upright cardinal adjacencies for every canonical affinity
pair, and removes only relations that are impossible from component geometry.

No permutation, truth, image pixels, board, rotation, reflection, learned
shortlist, or oracle label is accepted here.  Labels are applied only by the
separate E22 ceiling evaluator after this function returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

import e21_posegraph_candidate_oracle as e21
from eval_seeded_qap import dense_rd


GRID = 24
NUM_TILES = GRID * GRID
CANDIDATE_WIDTH = 128
NUM_DIRECTIONS = 4
UP, DOWN, LEFT, RIGHT = range(NUM_DIRECTIONS)
DIRECTION_NAMES = ("U", "D", "L", "R")

COMPONENT_MAX_EDGES = 96
MIN_MARGIN = 0.0

MAX_DIRECTED_MEMBERSHIPS = NUM_TILES * CANDIDATE_WIDTH
MAX_UNORDERED_AFFINITY_PAIRS = MAX_DIRECTED_MEMBERSHIPS
MAX_LOGIT_OBSERVATIONS = NUM_DIRECTIONS * MAX_DIRECTED_MEMBERSHIPS
MAX_RCCE4_CLAIMS = 4 * MAX_UNORDERED_AFFINITY_PAIRS
MAX_GEOMETRY_HYPOTHESES = MAX_RCCE4_CLAIMS

# For canonical pair a < b, enumerate only upright adjacency.  The first tile
# is physically left/top and the second tile is right/bottom.  Endpoint order
# represents L/U counterparts; tile coordinates are never rotated or flipped.
RCCE4_CLAIM_ORDER = (
    ("a", "b", 0, 1),  # b is right of a
    ("b", "a", 0, 1),  # a is right of b
    ("a", "b", 1, 0),  # b is below a
    ("b", "a", 1, 0),  # a is below b
)

RigidComponent = e21.RigidComponent
ComponentEntry = e21.ComponentEntry
PhysicalSeam = tuple[int, int, int, int]
PoseRelation = tuple[int, int, int, int]


class RCCE4OracleError(ValueError):
    """The frozen E22 input or candidate-generator invariant failed closed."""


@dataclass(frozen=True, slots=True)
class AffinityPair:
    pair_id: int
    a: int
    b: int
    a_to_b_slot: int | None
    b_to_a_slot: int | None

    @property
    def identity(self) -> tuple[int, int]:
        return (self.a, self.b)

    @property
    def membership_count(self) -> int:
        return int(self.a_to_b_slot is not None) + int(self.b_to_a_slot is not None)

    @property
    def reciprocal(self) -> bool:
        return self.membership_count == 2


@dataclass(frozen=True, slots=True)
class LogitObservation:
    source: int
    target: int
    direction: int
    slot: int
    logit: float

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (self.source, self.target, self.direction, self.slot)

    @property
    def direction_name(self) -> str:
        return DIRECTION_NAMES[self.direction]


@dataclass(frozen=True, slots=True)
class RCCE4Claim:
    claim_id: int
    pair_id: int
    first: int
    second: int
    dy: int
    dx: int
    first_component: int
    second_component: int
    forward_observation: LogitObservation | None
    reverse_observation: LogitObservation | None

    @property
    def physical_seam(self) -> PhysicalSeam:
        return (self.first, self.second, self.dy, self.dx)

    @property
    def adjacency(self) -> tuple[int, int, str]:
        return (self.first, self.second, "R" if self.dx == 1 else "D")

    @property
    def observations(self) -> tuple[LogitObservation, ...]:
        return tuple(
            value
            for value in (self.forward_observation, self.reverse_observation)
            if value is not None
        )


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    relation_id: int
    u: int
    v: int
    dr: int
    dc: int
    claim_ids: tuple[int, ...]

    @property
    def relation(self) -> PoseRelation:
        return (self.u, self.v, self.dr, self.dc)


@dataclass(frozen=True, slots=True)
class PoseHypothesis:
    hypothesis_id: int
    relation_id: int
    u: int
    v: int
    dr: int
    dc: int
    claim_ids: tuple[int, ...]

    @property
    def relation(self) -> PoseRelation:
        return (self.u, self.v, self.dr, self.dc)


@dataclass(frozen=True, slots=True)
class GeometryRejection:
    relation_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class CandidatePoolDiagnostics:
    component_count: int
    nontrivial_components: int
    singleton_components: int
    total_tiles: int
    nontrivial_tiles: int
    singleton_tiles: int
    emitter_tiles: int
    directed_valid_memberships: int
    input_logit_observations: int
    unordered_affinity_pairs: int
    one_way_affinity_pairs: int
    reciprocal_affinity_pairs: int
    pre_component_filter_claims: int
    same_component_pairs: int
    same_component_claims_removed: int
    cross_component_pairs: int
    claims: int
    claim_logit_observations: int
    relation_candidates: int
    geometry_valid_hypotheses: int
    geometry_rejected_relations: int
    geometry_rejected_adjacency: int
    geometry_rejected_collision: int
    geometry_rejected_span: int
    component_pairs: int
    component_pairs_with_alternative_offsets: int


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CandidatePoolResult:
    components: tuple[RigidComponent, ...]
    owner: np.ndarray
    local_rows: np.ndarray
    local_cols: np.ndarray
    nontrivial_component_ids: frozenset[int]
    affinity_pairs: tuple[AffinityPair, ...]
    claims: tuple[RCCE4Claim, ...]
    relation_candidates: tuple[RelationCandidate, ...]
    hypotheses: tuple[PoseHypothesis, ...]
    geometry_rejections: tuple[GeometryRejection, ...]
    diagnostics: CandidatePoolDiagnostics


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    candidate_ids: np.ndarray
    raw_logits: np.ndarray
    valid: np.ndarray
    directed_memberships: int


def _validate_inputs(
    candidate_ids: np.ndarray, raw_logits: np.ndarray
) -> _ValidatedInputs:
    if not isinstance(candidate_ids, np.ndarray):
        raise RCCE4OracleError("candidate_ids must be a numpy array")
    if candidate_ids.shape != (NUM_TILES, CANDIDATE_WIDTH):
        raise RCCE4OracleError("candidate_ids must be exactly int64[576,128]")
    if candidate_ids.dtype != np.int64:
        raise RCCE4OracleError("candidate_ids must have exact int64 dtype")
    if not candidate_ids.flags.c_contiguous:
        raise RCCE4OracleError("candidate_ids must be C-contiguous")
    if not isinstance(raw_logits, np.ndarray):
        raise RCCE4OracleError("raw_logits must be a numpy array")
    if raw_logits.shape != (NUM_DIRECTIONS, NUM_TILES, CANDIDATE_WIDTH):
        raise RCCE4OracleError("raw_logits must be exactly float32[4,576,128]")
    if raw_logits.dtype != np.float32:
        raise RCCE4OracleError("raw_logits must have exact float32 dtype")
    if not raw_logits.flags.c_contiguous:
        raise RCCE4OracleError("raw_logits must be C-contiguous")

    candidates = np.array(candidate_ids, dtype=np.int64, order="C", copy=True)
    logits = np.array(raw_logits, dtype=np.float32, order="C", copy=True)
    if bool(np.isnan(logits).any()) or bool(np.isposinf(logits).any()):
        raise RCCE4OracleError("raw_logits may contain only finite values or -inf padding")
    finite = np.isfinite(logits)
    if not all(np.array_equal(finite[0], finite[index]) for index in range(1, 4)):
        raise RCCE4OracleError("raw_logits finite mask differs across U,D,L,R")
    valid = np.ascontiguousarray(finite[0], dtype=np.bool_)
    if not bool(valid.any(axis=1).all()):
        raise RCCE4OracleError("every affinity row must have a valid candidate")
    expanded_valid = np.broadcast_to(valid, logits.shape)
    if not bool(np.isfinite(logits[expanded_valid]).all()):
        raise RCCE4OracleError("valid raw logit is not finite")
    if not bool(np.isneginf(logits[~expanded_valid]).all()):
        raise RCCE4OracleError("invalid raw logit padding must be -inf")

    # Only finite affinity memberships have an ID contract.  Padding IDs are
    # semantically absent and may contain arbitrary int64 values in a cache;
    # sanitize the private copy before dense_rd's scatter operations.
    valid_candidates = candidates[valid]
    if bool((valid_candidates < 0).any()) or bool(
        (valid_candidates >= NUM_TILES).any()
    ):
        raise RCCE4OracleError("valid candidate ID lies outside 0..575")
    candidates[~valid] = 0

    anchors = np.arange(NUM_TILES, dtype=np.int64)[:, None]
    if bool(((candidates == anchors) & valid).any()):
        raise RCCE4OracleError("valid affinity membership contains a self candidate")
    for anchor in range(NUM_TILES):
        row = candidates[anchor, valid[anchor]]
        if np.unique(row).size != row.size:
            raise RCCE4OracleError("valid affinity row contains duplicate candidates")
    directed_memberships = int(valid.sum())
    if not NUM_TILES <= directed_memberships <= MAX_DIRECTED_MEMBERSHIPS:
        raise RCCE4OracleError("directed affinity membership bound failed")
    return _ValidatedInputs(candidates, logits, valid, directed_memberships)


def _derive_dense_scores_from_validated(
    values: _ValidatedInputs,
) -> tuple[np.ndarray, np.ndarray]:
    candidates_t = torch.from_numpy(values.candidate_ids).long()
    logits_t = torch.from_numpy(values.raw_logits).float()
    if candidates_t.device.type != "cpu" or logits_t.device.type != "cpu":
        raise RCCE4OracleError("RCCE-4 dense reconstruction must stay on CPU")
    try:
        with torch.inference_mode():
            right_t, down_t = dense_rd(candidates_t, logits_t)
    except Exception as exc:
        raise RCCE4OracleError(f"CPU dense reconstruction failed: {exc}") from exc
    right = np.ascontiguousarray(right_t.detach().float().cpu().numpy(), dtype=np.float32)
    down = np.ascontiguousarray(down_t.detach().float().cpu().numpy(), dtype=np.float32)
    for label, matrix in (("right", right), ("down", down)):
        if (
            matrix.shape != (NUM_TILES, NUM_TILES)
            or matrix.dtype != np.float32
            or not matrix.flags.c_contiguous
            or not np.isfinite(matrix).all()
            or bool((matrix < 0.0).any())
            or bool((np.diag(matrix) != 0.0).any())
        ):
            raise RCCE4OracleError(f"derived dense {label} matrix drifted")
    return right, down


def derive_dense_scores(
    candidate_ids: np.ndarray, raw_logits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the exact frozen Rank96 CPU-float32 right/down matrices."""

    return _derive_dense_scores_from_validated(
        _validate_inputs(candidate_ids, raw_logits)
    )


def _build_affinity_pairs(values: _ValidatedInputs) -> tuple[AffinityPair, ...]:
    slots: dict[tuple[int, int], list[int | None]] = {}
    for source in range(NUM_TILES):
        for slot_value in np.flatnonzero(values.valid[source]).tolist():
            slot = int(slot_value)
            target = int(values.candidate_ids[source, slot])
            a, b = (source, target) if source < target else (target, source)
            membership = slots.setdefault((a, b), [None, None])
            index = 0 if source == a else 1
            if membership[index] is not None:
                raise RCCE4OracleError("canonical affinity pair repeats a directed membership")
            membership[index] = slot
    pairs = tuple(
        AffinityPair(
            pair_id=pair_id,
            a=a,
            b=b,
            a_to_b_slot=value[0],
            b_to_a_slot=value[1],
        )
        for pair_id, ((a, b), value) in enumerate(sorted(slots.items()))
    )
    if not pairs or tuple(pair.pair_id for pair in pairs) != tuple(range(len(pairs))):
        raise RCCE4OracleError("canonical affinity pair IDs drifted")
    if any(pair.a >= pair.b or pair.membership_count not in (1, 2) for pair in pairs):
        raise RCCE4OracleError("canonical affinity pair algebra drifted")
    if len(pairs) > MAX_UNORDERED_AFFINITY_PAIRS:
        raise RCCE4OracleError("unordered affinity pair bound failed")
    if sum(pair.membership_count for pair in pairs) != values.directed_memberships:
        raise RCCE4OracleError("directed membership accounting drifted after pair OR")
    return pairs


def _observation(
    values: _ValidatedInputs,
    *,
    source: int,
    target: int,
    direction: int,
    slot: int | None,
) -> LogitObservation | None:
    if slot is None:
        return None
    if (
        not 0 <= slot < CANDIDATE_WIDTH
        or not bool(values.valid[source, slot])
        or int(values.candidate_ids[source, slot]) != target
    ):
        raise RCCE4OracleError("directed affinity slot metadata drifted")
    logit = float(values.raw_logits[direction, source, slot])
    if not isfinite(logit):
        raise RCCE4OracleError("claim raw-logit observation is not finite")
    return LogitObservation(source, target, direction, slot, logit)


def _claim_specs(
    pair: AffinityPair,
) -> tuple[
    tuple[int, int, int, int, int | None, int, int | None, int], ...
]:
    """Return first, second, dy, dx, forward slot/dir, reverse slot/dir."""

    slots = {"a": pair.a_to_b_slot, "b": pair.b_to_a_slot}
    tiles = {"a": pair.a, "b": pair.b}
    result = tuple(
        (
            tiles[first_name],
            tiles[second_name],
            dy,
            dx,
            slots[first_name],
            RIGHT if dx else DOWN,
            slots[second_name],
            LEFT if dx else UP,
        )
        for first_name, second_name, dy, dx in RCCE4_CLAIM_ORDER
    )
    if tuple((first, second, dy, dx) for first, second, dy, dx, *_ in result) != (
        (pair.a, pair.b, 0, 1),
        (pair.b, pair.a, 0, 1),
        (pair.a, pair.b, 1, 0),
        (pair.b, pair.a, 1, 0),
    ):
        raise RCCE4OracleError("frozen RCCE-4 claim order drifted")
    return result


def _build_claims(
    values: _ValidatedInputs,
    pairs: tuple[AffinityPair, ...],
    owner: np.ndarray,
) -> tuple[tuple[RCCE4Claim, ...], int, int]:
    claims: list[RCCE4Claim] = []
    same_component_pairs = 0
    cross_component_pairs = 0
    for pair in pairs:
        if int(owner[pair.a]) == int(owner[pair.b]):
            same_component_pairs += 1
            continue
        cross_component_pairs += 1
        for (
            first,
            second,
            dy,
            dx,
            forward_slot,
            forward_direction,
            reverse_slot,
            reverse_direction,
        ) in _claim_specs(pair):
            claim = RCCE4Claim(
                claim_id=len(claims),
                pair_id=pair.pair_id,
                first=first,
                second=second,
                dy=dy,
                dx=dx,
                first_component=int(owner[first]),
                second_component=int(owner[second]),
                forward_observation=_observation(
                    values,
                    source=first,
                    target=second,
                    direction=forward_direction,
                    slot=forward_slot,
                ),
                reverse_observation=_observation(
                    values,
                    source=second,
                    target=first,
                    direction=reverse_direction,
                    slot=reverse_slot,
                ),
            )
            if claim.first_component == claim.second_component:
                raise RCCE4OracleError("same-component claim escaped pair removal")
            if (claim.dy, claim.dx) not in ((0, 1), (1, 0)):
                raise RCCE4OracleError("RCCE-4 emitted a non-upright adjacency")
            if not claim.observations:
                raise RCCE4OracleError("RCCE-4 claim lost all directed metadata")
            claims.append(claim)
    if same_component_pairs + cross_component_pairs != len(pairs):
        raise RCCE4OracleError("component pair accounting drifted")
    if len(claims) != 4 * cross_component_pairs:
        raise RCCE4OracleError("RCCE-4 did not preserve four claims per cross pair")
    if len(claims) > MAX_RCCE4_CLAIMS:
        raise RCCE4OracleError("RCCE-4 claim bound failed")
    seams = tuple(claim.physical_seam for claim in claims)
    if len(seams) != len(set(seams)):
        raise RCCE4OracleError("RCCE-4 physical seam claims are not unique")
    return tuple(claims), same_component_pairs, cross_component_pairs


def _claim_relation(
    claim: RCCE4Claim,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> PoseRelation:
    raw_offset = (
        int(local_rows[claim.first]) + claim.dy - int(local_rows[claim.second]),
        int(local_cols[claim.first]) + claim.dx - int(local_cols[claim.second]),
    )
    if claim.first_component < claim.second_component:
        return (
            claim.first_component,
            claim.second_component,
            raw_offset[0],
            raw_offset[1],
        )
    return (
        claim.second_component,
        claim.first_component,
        -raw_offset[0],
        -raw_offset[1],
    )


def _build_relation_candidates(
    claims: tuple[RCCE4Claim, ...],
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> tuple[RelationCandidate, ...]:
    if tuple(claim.claim_id for claim in claims) != tuple(range(len(claims))):
        raise RCCE4OracleError("claim IDs are not contiguous")
    grouped: dict[PoseRelation, list[int]] = {}
    for claim in claims:
        grouped.setdefault(_claim_relation(claim, local_rows, local_cols), []).append(
            claim.claim_id
        )
    relations = tuple(
        RelationCandidate(
            relation_id=relation_id,
            u=relation[0],
            v=relation[1],
            dr=relation[2],
            dc=relation[3],
            claim_ids=tuple(grouped[relation]),
        )
        for relation_id, relation in enumerate(sorted(grouped))
    )
    if any(
        relation.u >= relation.v or not relation.claim_ids for relation in relations
    ):
        raise RCCE4OracleError("canonical relation inventory drifted")
    return relations


def _geometry_rejection_reason(
    relation: RelationCandidate,
    claims: tuple[RCCE4Claim, ...],
    components: tuple[RigidComponent, ...],
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> str | None:
    if not (0 <= relation.u < relation.v < len(components)):
        raise RCCE4OracleError("relation references an invalid component")
    shifts = {relation.u: (0, 0), relation.v: (relation.dr, relation.dc)}
    for claim_id in relation.claim_ids:
        if not 0 <= claim_id < len(claims):
            raise RCCE4OracleError("relation references an invalid claim")
        claim = claims[claim_id]
        if {claim.first_component, claim.second_component} != {
            relation.u,
            relation.v,
        }:
            raise RCCE4OracleError("relation claim leaves its component pair")
        first_shift = shifts[int(owner[claim.first])]
        second_shift = shifts[int(owner[claim.second])]
        first_position = (
            int(local_rows[claim.first]) + first_shift[0],
            int(local_cols[claim.first]) + first_shift[1],
        )
        second_position = (
            int(local_rows[claim.second]) + second_shift[0],
            int(local_cols[claim.second]) + second_shift[1],
        )
        if (
            second_position[0] - first_position[0],
            second_position[1] - first_position[1],
        ) != (claim.dy, claim.dx):
            return "adjacency"

    occupied: set[tuple[int, int]] = set()
    for component_id in (relation.u, relation.v):
        shift = shifts[component_id]
        for _tile, row, col in components[component_id].entries:
            position = (row + shift[0], col + shift[1])
            if position in occupied:
                return "collision"
            occupied.add(position)
    rows = [row for row, _col in occupied]
    cols = [col for _row, col in occupied]
    if max(rows) - min(rows) + 1 > GRID or max(cols) - min(cols) + 1 > GRID:
        return "span"
    return None


def filter_relation_geometry(
    relation_candidates: tuple[RelationCandidate, ...],
    claims: tuple[RCCE4Claim, ...],
    components: tuple[RigidComponent, ...],
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> tuple[tuple[PoseHypothesis, ...], tuple[GeometryRejection, ...]]:
    """Apply only asserted adjacency, collision, and 24x24 pair-span checks."""

    hypotheses: list[PoseHypothesis] = []
    rejections: list[GeometryRejection] = []
    for expected_id, relation in enumerate(relation_candidates):
        if relation.relation_id != expected_id:
            raise RCCE4OracleError("relation candidate IDs are not contiguous")
        reason = _geometry_rejection_reason(
            relation,
            claims,
            components,
            owner,
            local_rows,
            local_cols,
        )
        if reason is None:
            hypotheses.append(
                PoseHypothesis(
                    hypothesis_id=len(hypotheses),
                    relation_id=relation.relation_id,
                    u=relation.u,
                    v=relation.v,
                    dr=relation.dr,
                    dc=relation.dc,
                    claim_ids=relation.claim_ids,
                )
            )
        else:
            rejections.append(
                GeometryRejection(relation_id=relation.relation_id, reason=reason)
            )
    if len(hypotheses) + len(rejections) != len(relation_candidates):
        raise RCCE4OracleError("geometry filter accounting drifted")
    if len(hypotheses) > MAX_GEOMETRY_HYPOTHESES:
        raise RCCE4OracleError("geometry-valid hypothesis bound failed")
    return tuple(hypotheses), tuple(rejections)


def run_rcce4_candidate_oracle(
    candidate_ids: np.ndarray, raw_logits: np.ndarray
) -> CandidatePoolResult:
    """Build the complete frozen E22 label-free RCCE-4 candidate pool."""

    values = _validate_inputs(candidate_ids, raw_logits)
    right, down = _derive_dense_scores_from_validated(values)
    try:
        components, owner, local_rows, local_cols, nontrivial = e21.build_components(
            right, down
        )
    except Exception as exc:
        raise RCCE4OracleError(f"exact raw CC96 construction failed: {exc}") from exc

    pairs = _build_affinity_pairs(values)
    claims, same_component_pairs, cross_component_pairs = _build_claims(
        values, pairs, owner
    )
    relations = _build_relation_candidates(claims, local_rows, local_cols)
    hypotheses, rejections = filter_relation_geometry(
        relations,
        claims,
        components,
        owner,
        local_rows,
        local_cols,
    )

    nontrivial_tiles = sum(
        component.size
        for component in components
        if component.component_id in nontrivial
    )
    pair_offsets: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for relation in relations:
        pair_offsets.setdefault((relation.u, relation.v), set()).add(
            (relation.dr, relation.dc)
        )
    rejection_counts = {
        reason: sum(value.reason == reason for value in rejections)
        for reason in ("adjacency", "collision", "span")
    }
    diagnostics = CandidatePoolDiagnostics(
        component_count=len(components),
        nontrivial_components=len(nontrivial),
        singleton_components=len(components) - len(nontrivial),
        total_tiles=NUM_TILES,
        nontrivial_tiles=nontrivial_tiles,
        singleton_tiles=NUM_TILES - nontrivial_tiles,
        emitter_tiles=NUM_TILES,
        directed_valid_memberships=values.directed_memberships,
        input_logit_observations=NUM_DIRECTIONS * values.directed_memberships,
        unordered_affinity_pairs=len(pairs),
        one_way_affinity_pairs=sum(not pair.reciprocal for pair in pairs),
        reciprocal_affinity_pairs=sum(pair.reciprocal for pair in pairs),
        pre_component_filter_claims=4 * len(pairs),
        same_component_pairs=same_component_pairs,
        same_component_claims_removed=4 * same_component_pairs,
        cross_component_pairs=cross_component_pairs,
        claims=len(claims),
        claim_logit_observations=sum(len(claim.observations) for claim in claims),
        relation_candidates=len(relations),
        geometry_valid_hypotheses=len(hypotheses),
        geometry_rejected_relations=len(rejections),
        geometry_rejected_adjacency=rejection_counts["adjacency"],
        geometry_rejected_collision=rejection_counts["collision"],
        geometry_rejected_span=rejection_counts["span"],
        component_pairs=len(pair_offsets),
        component_pairs_with_alternative_offsets=sum(
            len(offsets) > 1 for offsets in pair_offsets.values()
        ),
    )
    if diagnostics.singleton_tiles != diagnostics.singleton_components:
        raise RCCE4OracleError("singleton component accounting drifted")
    if diagnostics.one_way_affinity_pairs + diagnostics.reciprocal_affinity_pairs != len(
        pairs
    ):
        raise RCCE4OracleError("affinity pair reciprocity accounting drifted")
    if diagnostics.pre_component_filter_claims != 4 * diagnostics.unordered_affinity_pairs:
        raise RCCE4OracleError("prefilter RCCE-4 claim accounting drifted")
    if diagnostics.claims != 4 * diagnostics.cross_component_pairs:
        raise RCCE4OracleError("retained RCCE-4 claim accounting drifted")
    expected_retained_observations = 4 * sum(
        pair.membership_count
        for pair in pairs
        if int(owner[pair.a]) != int(owner[pair.b])
    )
    if diagnostics.input_logit_observations != 4 * values.directed_memberships:
        raise RCCE4OracleError("input raw-logit observation accounting drifted")
    if diagnostics.claim_logit_observations != expected_retained_observations:
        raise RCCE4OracleError("retained raw-logit observation accounting drifted")
    if diagnostics.input_logit_observations > MAX_LOGIT_OBSERVATIONS:
        raise RCCE4OracleError("input logit observation bound failed")
    if diagnostics.claim_logit_observations > MAX_LOGIT_OBSERVATIONS:
        raise RCCE4OracleError("claim logit observation bound failed")
    if diagnostics.geometry_valid_hypotheses > MAX_GEOMETRY_HYPOTHESES:
        raise RCCE4OracleError("geometry hypothesis bound failed")
    return CandidatePoolResult(
        components=components,
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial_component_ids=nontrivial,
        affinity_pairs=pairs,
        claims=claims,
        relation_candidates=relations,
        hypotheses=hypotheses,
        geometry_rejections=rejections,
        diagnostics=diagnostics,
    )


__all__ = [
    "AffinityPair",
    "CANDIDATE_WIDTH",
    "COMPONENT_MAX_EDGES",
    "CandidatePoolDiagnostics",
    "CandidatePoolResult",
    "ComponentEntry",
    "DIRECTION_NAMES",
    "DOWN",
    "GeometryRejection",
    "GRID",
    "LEFT",
    "LogitObservation",
    "MAX_DIRECTED_MEMBERSHIPS",
    "MAX_GEOMETRY_HYPOTHESES",
    "MAX_LOGIT_OBSERVATIONS",
    "MAX_RCCE4_CLAIMS",
    "MAX_UNORDERED_AFFINITY_PAIRS",
    "MIN_MARGIN",
    "NUM_DIRECTIONS",
    "NUM_TILES",
    "PhysicalSeam",
    "PoseHypothesis",
    "PoseRelation",
    "RCCE4Claim",
    "RCCE4OracleError",
    "RCCE4_CLAIM_ORDER",
    "RIGHT",
    "RelationCandidate",
    "RigidComponent",
    "UP",
    "derive_dense_scores",
    "filter_relation_geometry",
    "run_rcce4_candidate_oracle",
]
