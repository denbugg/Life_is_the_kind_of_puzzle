"""Frozen label-free E23 I21-residual-K64 candidate generator.

The core preserves the complete E22 affinity/RCCE-4 pool and adds exactly 64
residual I21 nominations for every tile and literal U/D/L/R score row.  A
spatial direction nominates only an unordered pair: it never selects a
physical side.  Every new pair therefore receives the same four upright
RCCE-4 adjacency hypotheses as an E22 affinity pair.

No permutation, truth, labels, pixels, torch model, board, rotation,
reflection, score fusion, threshold, or post-union truncation is accepted by
this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import e22_rcce4_candidate_oracle as e22


GRID = e22.GRID
NUM_TILES = e22.NUM_TILES
CANDIDATE_WIDTH = e22.CANDIDATE_WIDTH
NUM_DIRECTIONS = e22.NUM_DIRECTIONS
UP, DOWN, LEFT, RIGHT = e22.UP, e22.DOWN, e22.LEFT, e22.RIGHT
DIRECTION_NAMES = e22.DIRECTION_NAMES
RCCE4_CLAIM_ORDER = e22.RCCE4_CLAIM_ORDER

SPATIAL_K = 64
SPATIAL_LOGIT_VALUES = NUM_DIRECTIONS * NUM_TILES * NUM_TILES
SPATIAL_SELECTIONS = NUM_DIRECTIONS * NUM_TILES * SPATIAL_K
MAX_DIRECTED_MEMBERSHIPS = NUM_TILES * CANDIDATE_WIDTH
MAX_BASE_AFFINITY_PAIRS = MAX_DIRECTED_MEMBERSHIPS
MAX_ALL_UNORDERED_PAIRS = NUM_TILES * (NUM_TILES - 1) // 2
MAX_SPATIAL_PAIRS_ABSOLUTE = min(SPATIAL_SELECTIONS, MAX_ALL_UNORDERED_PAIRS)
MAX_NEW_RCCE4_CLAIMS = 4 * SPATIAL_SELECTIONS
MAX_COMBINED_RCCE4_CLAIMS = 4 * MAX_ALL_UNORDERED_PAIRS
MAX_CROSS_COMPONENT_CLAIMS = MAX_COMBINED_RCCE4_CLAIMS
MAX_RELATION_CANDIDATES = MAX_COMBINED_RCCE4_CLAIMS
MAX_GEOMETRY_HYPOTHESES = MAX_COMBINED_RCCE4_CLAIMS

RigidComponent = e22.RigidComponent
ComponentEntry = e22.ComponentEntry
AffinityPair = e22.AffinityPair
RCCE4Claim = e22.RCCE4Claim
RelationCandidate = e22.RelationCandidate
PoseHypothesis = e22.PoseHypothesis
GeometryRejection = e22.GeometryRejection
PhysicalSeam = e22.PhysicalSeam
PoseRelation = e22.PoseRelation


class I21ResidualOracleError(ValueError):
    """The frozen E23 input or candidate-generator invariant failed closed."""


@dataclass(frozen=True, slots=True)
class SpatialPair:
    """A new unordered pair nominated only by residual spatial rows.

    ``nomination_count`` counts entries in ``spatial_selected_ids``.  Literal
    direction, source, and rank remain losslessly encoded by that compact
    array and deliberately do not select an RCCE-4 side.
    """

    pair_id: int
    a: int
    b: int
    nomination_count: int

    @property
    def identity(self) -> tuple[int, int]:
        return (self.a, self.b)


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
    spatial_logit_values: int
    spatial_selections: int
    spatial_pair_nominations: int
    base_affinity_pairs: int
    spatial_pairs: int
    unordered_affinity_pairs: int
    one_way_affinity_pairs: int
    reciprocal_affinity_pairs: int
    base_pre_component_filter_claims: int
    spatial_pre_component_filter_claims: int
    pre_component_filter_claims: int
    base_same_component_pairs: int
    spatial_same_component_pairs: int
    same_component_pairs: int
    same_component_claims_removed: int
    base_cross_component_pairs: int
    spatial_cross_component_pairs: int
    cross_component_pairs: int
    base_claims: int
    spatial_claims: int
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
    affinity_pairs: tuple[AffinityPair | SpatialPair, ...]
    base_affinity_pairs: tuple[AffinityPair, ...]
    spatial_selected_ids: np.ndarray
    spatial_pairs: tuple[SpatialPair, ...]
    claims: tuple[RCCE4Claim, ...]
    relation_candidates: tuple[RelationCandidate, ...]
    hypotheses: tuple[PoseHypothesis, ...]
    rejections: tuple[GeometryRejection, ...]
    diagnostics: CandidatePoolDiagnostics

    @property
    def geometry_rejections(self) -> tuple[GeometryRejection, ...]:
        """Compatibility alias for the unchanged E22 rejection inventory."""

        return self.rejections


def _validate_spatial_logits(spatial_logits: np.ndarray) -> np.ndarray:
    if not isinstance(spatial_logits, np.ndarray):
        raise I21ResidualOracleError("spatial_logits must be a numpy array")
    if spatial_logits.shape != (NUM_DIRECTIONS, NUM_TILES, NUM_TILES):
        raise I21ResidualOracleError(
            "spatial_logits must be exactly float32[4,576,576]"
        )
    if spatial_logits.dtype != np.float32:
        raise I21ResidualOracleError("spatial_logits must have exact float32 dtype")
    if not spatial_logits.flags.c_contiguous:
        raise I21ResidualOracleError("spatial_logits must be C-contiguous")
    if spatial_logits.size != SPATIAL_LOGIT_VALUES:
        raise I21ResidualOracleError("spatial logit value count drifted")
    if not bool(np.isfinite(spatial_logits).all()):
        raise I21ResidualOracleError("every spatial logit must be finite")
    return spatial_logits


def _base_targets_by_anchor(
    base_pairs: tuple[AffinityPair, ...],
) -> tuple[frozenset[int], ...]:
    targets: list[set[int]] = [set() for _ in range(NUM_TILES)]
    previous: tuple[int, int] | None = None
    for expected_id, pair in enumerate(base_pairs):
        if not isinstance(pair, AffinityPair):
            raise I21ResidualOracleError("base affinity pair has an invalid type")
        if pair.pair_id != expected_id or not 0 <= pair.a < pair.b < NUM_TILES:
            raise I21ResidualOracleError("base affinity pair IDs or endpoints drifted")
        if pair.membership_count not in (1, 2):
            raise I21ResidualOracleError("base affinity membership count drifted")
        if previous is not None and previous >= pair.identity:
            raise I21ResidualOracleError("base affinity pairs are not lexicographic")
        previous = pair.identity
        targets[pair.a].add(pair.b)
        targets[pair.b].add(pair.a)
    if len(base_pairs) > MAX_BASE_AFFINITY_PAIRS:
        raise I21ResidualOracleError("base affinity pair bound failed")
    directed_memberships = sum(pair.membership_count for pair in base_pairs)
    if directed_memberships > MAX_DIRECTED_MEMBERSHIPS:
        raise I21ResidualOracleError("base directed membership bound failed")
    return tuple(frozenset(row) for row in targets)


def _select_spatial_residuals(
    spatial_logits: np.ndarray,
    base_pairs: tuple[AffinityPair, ...],
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Select frozen score-desc/tile-ID-asc residual K64 nominations."""

    scores = _validate_spatial_logits(spatial_logits)
    excluded = _base_targets_by_anchor(base_pairs)
    tile_ids = np.arange(NUM_TILES, dtype=np.int64)
    selected = np.empty(
        (NUM_DIRECTIONS, NUM_TILES, SPATIAL_K), dtype=np.int64, order="C"
    )
    nomination_counts: dict[tuple[int, int], int] = {}

    for direction in range(NUM_DIRECTIONS):
        for source in range(NUM_TILES):
            eligible_mask = np.ones(NUM_TILES, dtype=np.bool_)
            eligible_mask[source] = False
            if excluded[source]:
                eligible_mask[np.fromiter(excluded[source], dtype=np.int64)] = False
            eligible = tile_ids[eligible_mask]
            if eligible.size < SPATIAL_K:
                raise I21ResidualOracleError(
                    "fewer than 64 residual spatial targets remain after exclusions"
                )
            row_scores = scores[direction, source, eligible]
            order = np.lexsort((eligible, -row_scores))
            chosen = eligible[order[:SPATIAL_K]]
            if chosen.size != SPATIAL_K or np.unique(chosen).size != SPATIAL_K:
                raise I21ResidualOracleError("residual spatial K64 selection drifted")
            selected[direction, source] = chosen
            for target_value in chosen.tolist():
                target = int(target_value)
                a, b = (source, target) if source < target else (target, source)
                pair = (a, b)
                nomination_counts[pair] = nomination_counts.get(pair, 0) + 1

    if selected.size != SPATIAL_SELECTIONS:
        raise I21ResidualOracleError("residual selection count drifted")
    if sum(nomination_counts.values()) != SPATIAL_SELECTIONS:
        raise I21ResidualOracleError("spatial pair nomination accounting drifted")
    base_identities = {pair.identity for pair in base_pairs}
    if base_identities.intersection(nomination_counts):
        raise I21ResidualOracleError("residual spatial pair intersects E22")
    for direction in range(NUM_DIRECTIONS):
        for source in range(NUM_TILES):
            row = selected[direction, source]
            if source in row or any(int(target) in excluded[source] for target in row):
                raise I21ResidualOracleError("residual selection exclusion drifted")
    selected.setflags(write=False)
    return selected, nomination_counts


def _build_spatial_pairs(
    nomination_counts: dict[tuple[int, int], int],
    base_pairs: tuple[AffinityPair, ...],
) -> tuple[SpatialPair, ...]:
    base_count = len(base_pairs)
    maximum_new = min(SPATIAL_SELECTIONS, MAX_ALL_UNORDERED_PAIRS - base_count)
    if not nomination_counts or len(nomination_counts) > maximum_new:
        raise I21ResidualOracleError("deduplicated spatial pair bound failed")
    base_identities = {pair.identity for pair in base_pairs}
    identities = tuple(sorted(nomination_counts))
    if base_identities.intersection(identities):
        raise I21ResidualOracleError("new spatial pairs are not disjoint from E22")
    pairs = tuple(
        SpatialPair(
            pair_id=base_count + index,
            a=identity[0],
            b=identity[1],
            nomination_count=int(nomination_counts[identity]),
        )
        for index, identity in enumerate(identities)
    )
    if any(
        not 0 <= pair.a < pair.b < NUM_TILES or not 1 <= pair.nomination_count <= 8
        for pair in pairs
    ):
        raise I21ResidualOracleError("spatial pair algebra drifted")
    if sum(pair.nomination_count for pair in pairs) != SPATIAL_SELECTIONS:
        raise I21ResidualOracleError("spatial pair nomination count drifted")
    if base_count + len(pairs) > MAX_ALL_UNORDERED_PAIRS:
        raise I21ResidualOracleError("combined unordered pair bound failed")
    return pairs


def _build_spatial_claims(
    spatial_pairs: tuple[SpatialPair, ...],
    owner: np.ndarray,
    *,
    first_claim_id: int,
) -> tuple[tuple[RCCE4Claim, ...], int, int]:
    claims: list[RCCE4Claim] = []
    same_component_pairs = 0
    cross_component_pairs = 0
    for pair in spatial_pairs:
        if int(owner[pair.a]) == int(owner[pair.b]):
            same_component_pairs += 1
            continue
        cross_component_pairs += 1
        tiles = {"a": pair.a, "b": pair.b}
        for first_name, second_name, dy, dx in RCCE4_CLAIM_ORDER:
            first = tiles[first_name]
            second = tiles[second_name]
            claims.append(
                RCCE4Claim(
                    claim_id=first_claim_id + len(claims),
                    pair_id=pair.pair_id,
                    first=first,
                    second=second,
                    dy=dy,
                    dx=dx,
                    first_component=int(owner[first]),
                    second_component=int(owner[second]),
                    forward_observation=None,
                    reverse_observation=None,
                )
            )
    if same_component_pairs + cross_component_pairs != len(spatial_pairs):
        raise I21ResidualOracleError("spatial component-pair accounting drifted")
    if len(claims) != 4 * cross_component_pairs:
        raise I21ResidualOracleError("spatial RCCE-4 did not retain four claims per cross pair")
    if 4 * len(spatial_pairs) > MAX_NEW_RCCE4_CLAIMS:
        raise I21ResidualOracleError("new literal RCCE-4 claim bound failed")
    if len(claims) > MAX_CROSS_COMPONENT_CLAIMS:
        raise I21ResidualOracleError("spatial cross-component claim bound failed")
    if any(
        claim.claim_id != first_claim_id + index
        for index, claim in enumerate(claims)
    ):
        raise I21ResidualOracleError("spatial claim IDs drifted")
    if any(
        (claim.dy, claim.dx) not in ((0, 1), (1, 0))
        or claim.first_component == claim.second_component
        or claim.observations
        for claim in claims
    ):
        raise I21ResidualOracleError("spatial RCCE-4 emitted invalid side metadata")
    return tuple(claims), same_component_pairs, cross_component_pairs


def filter_relation_geometry(
    relation_candidates: tuple[RelationCandidate, ...],
    claims: tuple[RCCE4Claim, ...],
    components: tuple[RigidComponent, ...],
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> tuple[tuple[PoseHypothesis, ...], tuple[GeometryRejection, ...]]:
    """Apply the unchanged E22 adjacency/collision/24x24-span filter."""

    if len(relation_candidates) > MAX_RELATION_CANDIDATES:
        raise I21ResidualOracleError("relation candidate bound failed")
    hypotheses: list[PoseHypothesis] = []
    rejections: list[GeometryRejection] = []
    for expected_id, relation in enumerate(relation_candidates):
        if relation.relation_id != expected_id:
            raise I21ResidualOracleError("relation candidate IDs are not contiguous")
        try:
            reason = e22._geometry_rejection_reason(
                relation, claims, components, owner, local_rows, local_cols
            )
        except Exception as exc:
            raise I21ResidualOracleError(
                f"unchanged E22 geometry filter failed: {exc}"
            ) from exc
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
            rejections.append(GeometryRejection(relation.relation_id, reason))
    if len(hypotheses) + len(rejections) != len(relation_candidates):
        raise I21ResidualOracleError("geometry filter accounting drifted")
    if len(hypotheses) > MAX_GEOMETRY_HYPOTHESES:
        raise I21ResidualOracleError("geometry-valid hypothesis bound failed")
    return tuple(hypotheses), tuple(rejections)


def _diagnostics(
    components: tuple[RigidComponent, ...],
    nontrivial_component_ids: frozenset[int],
    base_pairs: tuple[AffinityPair, ...],
    base_claims: tuple[RCCE4Claim, ...],
    base_diag: e22.CandidatePoolDiagnostics,
    spatial_pairs: tuple[SpatialPair, ...],
    spatial_same: int,
    spatial_cross: int,
    claims: tuple[RCCE4Claim, ...],
    relations: tuple[RelationCandidate, ...],
    hypotheses: tuple[PoseHypothesis, ...],
    rejections: tuple[GeometryRejection, ...],
) -> CandidatePoolDiagnostics:
    nontrivial_tiles = sum(
        component.size
        for component in components
        if component.component_id in nontrivial_component_ids
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
    base_same = base_diag.same_component_pairs
    base_cross = base_diag.cross_component_pairs
    diagnostics = CandidatePoolDiagnostics(
        component_count=len(components),
        nontrivial_components=len(nontrivial_component_ids),
        singleton_components=len(components) - len(nontrivial_component_ids),
        total_tiles=NUM_TILES,
        nontrivial_tiles=nontrivial_tiles,
        singleton_tiles=NUM_TILES - nontrivial_tiles,
        emitter_tiles=NUM_TILES,
        directed_valid_memberships=base_diag.directed_valid_memberships,
        input_logit_observations=base_diag.input_logit_observations,
        spatial_logit_values=SPATIAL_LOGIT_VALUES,
        spatial_selections=SPATIAL_SELECTIONS,
        spatial_pair_nominations=sum(pair.nomination_count for pair in spatial_pairs),
        base_affinity_pairs=len(base_pairs),
        spatial_pairs=len(spatial_pairs),
        unordered_affinity_pairs=len(base_pairs) + len(spatial_pairs),
        one_way_affinity_pairs=base_diag.one_way_affinity_pairs,
        reciprocal_affinity_pairs=base_diag.reciprocal_affinity_pairs,
        base_pre_component_filter_claims=4 * len(base_pairs),
        spatial_pre_component_filter_claims=4 * len(spatial_pairs),
        pre_component_filter_claims=4 * (len(base_pairs) + len(spatial_pairs)),
        base_same_component_pairs=base_same,
        spatial_same_component_pairs=spatial_same,
        same_component_pairs=base_same + spatial_same,
        same_component_claims_removed=4 * (base_same + spatial_same),
        base_cross_component_pairs=base_cross,
        spatial_cross_component_pairs=spatial_cross,
        cross_component_pairs=base_cross + spatial_cross,
        base_claims=len(base_claims),
        spatial_claims=len(claims) - len(base_claims),
        claims=len(claims),
        claim_logit_observations=base_diag.claim_logit_observations,
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
    if diagnostics.spatial_selections != SPATIAL_SELECTIONS:
        raise I21ResidualOracleError("spatial selection diagnostic drifted")
    if diagnostics.spatial_pair_nominations != SPATIAL_SELECTIONS:
        raise I21ResidualOracleError("spatial nomination diagnostic drifted")
    if diagnostics.unordered_affinity_pairs > MAX_ALL_UNORDERED_PAIRS:
        raise I21ResidualOracleError("combined affinity pair bound failed")
    if diagnostics.spatial_pre_component_filter_claims != 4 * diagnostics.spatial_pairs:
        raise I21ResidualOracleError("new literal claim accounting drifted")
    if diagnostics.pre_component_filter_claims != 4 * diagnostics.unordered_affinity_pairs:
        raise I21ResidualOracleError("combined literal claim accounting drifted")
    if diagnostics.pre_component_filter_claims > MAX_COMBINED_RCCE4_CLAIMS:
        raise I21ResidualOracleError("combined literal claim bound failed")
    if diagnostics.claims != 4 * diagnostics.cross_component_pairs:
        raise I21ResidualOracleError("cross-component claim accounting drifted")
    if diagnostics.claims > MAX_CROSS_COMPONENT_CLAIMS:
        raise I21ResidualOracleError("cross-component claim bound failed")
    if diagnostics.relation_candidates > MAX_RELATION_CANDIDATES:
        raise I21ResidualOracleError("relation candidate bound failed")
    if diagnostics.geometry_valid_hypotheses > MAX_GEOMETRY_HYPOTHESES:
        raise I21ResidualOracleError("geometry hypothesis bound failed")
    if (
        diagnostics.geometry_valid_hypotheses
        + diagnostics.geometry_rejected_relations
        != diagnostics.relation_candidates
    ):
        raise I21ResidualOracleError("geometry diagnostic accounting drifted")
    return diagnostics


def run_i21_residual_candidate_oracle(
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
) -> CandidatePoolResult:
    """Build the complete frozen E23 label-free residual-K64 candidate pool."""

    spatial = _validate_spatial_logits(spatial_logits)
    try:
        base = e22.run_rcce4_candidate_oracle(candidate_ids, raw_logits)
    except Exception as exc:
        raise I21ResidualOracleError(f"exact E22 base replay failed: {exc}") from exc
    if not isinstance(base, e22.CandidatePoolResult):
        raise I21ResidualOracleError("exact E22 base replay returned an invalid type")

    base_pairs = base.affinity_pairs
    base_claims = base.claims
    components = base.components
    owner = base.owner
    local_rows = base.local_rows
    local_cols = base.local_cols
    nontrivial_component_ids = base.nontrivial_component_ids
    base_diag = base.diagnostics
    base_memberships = sum(pair.membership_count for pair in base_pairs)
    if (
        not NUM_TILES <= base_memberships <= MAX_DIRECTED_MEMBERSHIPS
        or base_memberships != base_diag.directed_valid_memberships
    ):
        raise I21ResidualOracleError("exact E22 directed membership replay drifted")
    # The E22 relation/hypothesis inventory is not part of the E23 result.
    # Drop its owning result before allocating the much larger union objects.
    del base
    selected_ids, nomination_counts = _select_spatial_residuals(spatial, base_pairs)
    spatial_pairs = _build_spatial_pairs(nomination_counts, base_pairs)
    affinity_pairs: tuple[AffinityPair | SpatialPair, ...] = base_pairs + spatial_pairs
    if affinity_pairs[: len(base_pairs)] != base_pairs:
        raise I21ResidualOracleError("exact E22 affinity-pair prefix drifted")
    if any(pair.pair_id != index for index, pair in enumerate(affinity_pairs)):
        raise I21ResidualOracleError("combined affinity pair IDs are not contiguous")

    spatial_claims, spatial_same, spatial_cross = _build_spatial_claims(
        spatial_pairs, owner, first_claim_id=len(base_claims)
    )
    claims = base_claims + spatial_claims
    if claims[: len(base_claims)] != base_claims:
        raise I21ResidualOracleError("exact E22 retained-claim prefix drifted")
    if any(claim.claim_id != index for index, claim in enumerate(claims)):
        raise I21ResidualOracleError("combined claim IDs are not contiguous")
    if len(claims) > MAX_CROSS_COMPONENT_CLAIMS:
        raise I21ResidualOracleError("combined cross-component claim bound failed")
    seams: set[PhysicalSeam] = set()
    for claim in claims:
        seam = claim.physical_seam
        if seam in seams:
            raise I21ResidualOracleError(
                "combined RCCE-4 physical seams are not unique"
            )
        seams.add(seam)
    del seams

    try:
        relations = e22._build_relation_candidates(
            claims, local_rows, local_cols
        )
    except Exception as exc:
        raise I21ResidualOracleError(
            f"exact signed component-relation grouping failed: {exc}"
        ) from exc
    if len(relations) > MAX_RELATION_CANDIDATES:
        raise I21ResidualOracleError("relation candidate bound failed")
    hypotheses, rejections = filter_relation_geometry(
        relations,
        claims,
        components,
        owner,
        local_rows,
        local_cols,
    )
    diagnostics = _diagnostics(
        components,
        nontrivial_component_ids,
        base_pairs,
        base_claims,
        base_diag,
        spatial_pairs,
        spatial_same,
        spatial_cross,
        claims,
        relations,
        hypotheses,
        rejections,
    )
    return CandidatePoolResult(
        components=components,
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial_component_ids=nontrivial_component_ids,
        affinity_pairs=affinity_pairs,
        base_affinity_pairs=base_pairs,
        spatial_selected_ids=selected_ids,
        spatial_pairs=spatial_pairs,
        claims=claims,
        relation_candidates=relations,
        hypotheses=hypotheses,
        rejections=rejections,
        diagnostics=diagnostics,
    )


__all__ = [
    "AffinityPair",
    "CANDIDATE_WIDTH",
    "CandidatePoolDiagnostics",
    "CandidatePoolResult",
    "ComponentEntry",
    "DIRECTION_NAMES",
    "DOWN",
    "GeometryRejection",
    "GRID",
    "I21ResidualOracleError",
    "LEFT",
    "MAX_ALL_UNORDERED_PAIRS",
    "MAX_BASE_AFFINITY_PAIRS",
    "MAX_COMBINED_RCCE4_CLAIMS",
    "MAX_CROSS_COMPONENT_CLAIMS",
    "MAX_DIRECTED_MEMBERSHIPS",
    "MAX_GEOMETRY_HYPOTHESES",
    "MAX_NEW_RCCE4_CLAIMS",
    "MAX_RELATION_CANDIDATES",
    "MAX_SPATIAL_PAIRS_ABSOLUTE",
    "NUM_DIRECTIONS",
    "NUM_TILES",
    "PhysicalSeam",
    "PoseHypothesis",
    "PoseRelation",
    "RCCE4Claim",
    "RCCE4_CLAIM_ORDER",
    "RIGHT",
    "RelationCandidate",
    "RigidComponent",
    "SPATIAL_K",
    "SPATIAL_LOGIT_VALUES",
    "SPATIAL_SELECTIONS",
    "SpatialPair",
    "UP",
    "filter_relation_geometry",
    "run_i21_residual_candidate_oracle",
]
