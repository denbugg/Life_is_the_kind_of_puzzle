"""Frozen label-free E21 raw CC96-anchor pose-candidate inventory.

The core consumes only two raw dense directional score matrices.  It builds
the corrected deterministic CC96 partition, lets only nontrivial-component
tiles emit fixed positive top-eight directional claims, and groups every
exact signed component relation without collapsing alternative offsets.

No ground-truth data, image content, global layout, path closure, or search is
accepted by this module.  Oracle marking and exact-cluster evaluation belong
to the separate E21 evaluator after this function returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from solve_buddies import build_buddies_components


GRID = 24
NUM_TILES = GRID * GRID
COMPONENT_MAX_EDGES = 96
MIN_MARGIN = 0.0
CANDIDATE_TOP_K = 8
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # U, D, L, R

PhysicalSeam = tuple[int, int, int, int]
PoseRelation = tuple[int, int, int, int]  # u, v, dr, dc with u < v
ComponentEntry = tuple[int, int, int]  # tile, local row, local column


class CandidateOracleError(ValueError):
    """The frozen E21 input or candidate-inventory invariant failed closed."""


@dataclass(frozen=True)
class RigidComponent:
    component_id: int
    entries: tuple[ComponentEntry, ...]

    @property
    def tiles(self) -> tuple[int, ...]:
        return tuple(tile for tile, _row, _col in self.entries)

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def minimum_tile(self) -> int:
        return min(self.tiles)

    def positions(self) -> dict[int, tuple[int, int]]:
        return {tile: (row, col) for tile, row, col in self.entries}


@dataclass(frozen=True)
class CandidateClaim:
    claim_id: int
    score: float
    anchor: int
    target: int
    dy: int
    dx: int
    anchor_component: int
    target_component: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (self.anchor, self.target, self.dy, self.dx)


@dataclass(frozen=True)
class PoseHypothesis:
    hypothesis_id: int
    u: int
    v: int
    dr: int
    dc: int
    seam_scores: tuple[tuple[PhysicalSeam, float], ...]
    reciprocal_seams: tuple[PhysicalSeam, ...]

    @property
    def relation(self) -> PoseRelation:
        return (self.u, self.v, self.dr, self.dc)

    @property
    def physical_seams(self) -> tuple[PhysicalSeam, ...]:
        return tuple(seam for seam, _score in self.seam_scores)

    @property
    def unique_physical_seams(self) -> int:
        return len(self.seam_scores)

    @property
    def reciprocal_physical_seams(self) -> int:
        return len(self.reciprocal_seams)

    @property
    def direct_neural_sum(self) -> float:
        return float(sum(score for _seam, score in self.seam_scores))

    @property
    def direct_max_score(self) -> float:
        return float(max((score for _seam, score in self.seam_scores), default=0.0))


@dataclass(frozen=True)
class CandidatePoolDiagnostics:
    component_count: int
    nontrivial_components: int
    singleton_components: int
    total_tiles: int
    nontrivial_tiles: int
    singleton_tiles: int
    emitter_tiles: int
    directional_emitter_rows: int
    positive_top8_before_component_filter: int
    same_component_filtered: int
    claims: int
    nontrivial_target_claims: int
    singleton_target_claims: int
    hypotheses: int
    component_pairs: int
    component_pairs_with_alternative_offsets: int
    unique_physical_seams: int
    reciprocal_physical_seams: int


@dataclass(frozen=True)
class CandidatePoolResult:
    components: tuple[RigidComponent, ...]
    owner: np.ndarray
    local_rows: np.ndarray
    local_cols: np.ndarray
    nontrivial_component_ids: frozenset[int]
    claims: tuple[CandidateClaim, ...]
    hypotheses: tuple[PoseHypothesis, ...]
    diagnostics: CandidatePoolDiagnostics


@dataclass(frozen=True)
class _ClaimDiagnostics:
    emitter_tiles: int
    directional_emitter_rows: int
    positive_top8_before_component_filter: int
    same_component_filtered: int
    nontrivial_target_claims: int
    singleton_target_claims: int


def _dense(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (NUM_TILES, NUM_TILES):
        raise CandidateOracleError(f"{label} must be exactly 576x576")
    if array.dtype != np.float32:
        raise CandidateOracleError(f"{label} must have exact float32 dtype")
    # Never freeze the caller's buffer as a side effect.  A contiguous input
    # would otherwise be returned by ``ascontiguousarray`` itself and the
    # read-only flag below would leak outside this label-free core.
    matrix = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(matrix).all() or bool((matrix < 0.0).any()):
        raise CandidateOracleError(f"{label} must be finite and nonnegative")
    if bool((np.diag(matrix) != 0.0).any()):
        raise CandidateOracleError(f"{label} diagonal must be exactly zero")
    matrix.setflags(write=False)
    return matrix


def _normalise_component(
    positions: Mapping[int, tuple[int, int]],
) -> tuple[ComponentEntry, ...]:
    if not positions:
        raise CandidateOracleError("corrected CC96 builder returned an empty component")
    minimum_row = min(int(value[0]) for value in positions.values())
    minimum_col = min(int(value[1]) for value in positions.values())
    entries = tuple(
        sorted(
            (
                int(tile),
                int(value[0]) - minimum_row,
                int(value[1]) - minimum_col,
            )
            for tile, value in positions.items()
        )
    )
    if any(not 0 <= tile < NUM_TILES for tile, _row, _col in entries):
        raise CandidateOracleError("CC96 component contains an invalid tile ID")
    if len({tile for tile, _row, _col in entries}) != len(entries):
        raise CandidateOracleError("CC96 component repeats a tile")
    coordinates = {(row, col) for _tile, row, col in entries}
    if len(coordinates) != len(entries):
        raise CandidateOracleError("CC96 component contains a coordinate collision")
    if min(row for _tile, row, _col in entries) != 0 or min(
        col for _tile, _row, col in entries
    ) != 0:
        raise CandidateOracleError("CC96 component normalization failed")
    if max(row for _tile, row, _col in entries) >= GRID or max(
        col for _tile, _row, col in entries
    ) >= GRID:
        raise CandidateOracleError("CC96 component exceeds the 24x24 span")
    return entries


def build_components(
    right: np.ndarray, down: np.ndarray
) -> tuple[
    tuple[RigidComponent, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    frozenset[int],
]:
    """Build corrected CC96 geometry and the full deterministic partition."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    raw = build_buddies_components(
        r,
        d,
        max_edges=COMPONENT_MAX_EDGES,
        min_margin=MIN_MARGIN,
    )
    used: set[int] = set()
    entries: list[tuple[ComponentEntry, ...]] = []
    for raw_component in raw:
        normalized = _normalise_component(raw_component)
        tiles = {tile for tile, _row, _col in normalized}
        if used & tiles:
            raise CandidateOracleError("corrected CC96 components overlap in tile ID")
        used.update(tiles)
        entries.append(normalized)
    entries.extend(((tile, 0, 0),) for tile in range(NUM_TILES) if tile not in used)
    entries.sort(
        key=lambda value: (
            -len(value),
            min(tile for tile, _row, _col in value),
            value,
        )
    )
    components = tuple(
        RigidComponent(component_id=index, entries=value)
        for index, value in enumerate(entries)
    )
    if len(components) == 0:
        raise CandidateOracleError("CC96 full partition is empty")

    owner = np.full(NUM_TILES, -1, dtype=np.int64)
    local_rows = np.zeros(NUM_TILES, dtype=np.int64)
    local_cols = np.zeros(NUM_TILES, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            if owner[tile] >= 0:
                raise CandidateOracleError("CC96 full partition repeats a tile")
            owner[tile] = component.component_id
            local_rows[tile] = row
            local_cols[tile] = col
    expected_owner_ids = np.repeat(
        np.arange(len(components), dtype=np.int64),
        np.asarray([component.size for component in components], dtype=np.int64),
    )
    if not np.array_equal(np.sort(owner), expected_owner_ids):
        # The equality above simultaneously binds every component ID multiplicity.
        raise CandidateOracleError("CC96 owner multiplicities drifted")
    partition_tiles = np.concatenate(
        [np.asarray(component.tiles, dtype=np.int64) for component in components]
    )
    if not np.array_equal(np.sort(partition_tiles), np.arange(NUM_TILES)):
        raise CandidateOracleError("CC96 components do not partition all 576 tiles")
    nontrivial = frozenset(
        component.component_id for component in components if component.size >= 2
    )
    for value in (owner, local_rows, local_cols):
        value.setflags(write=False)
    return components, owner, local_rows, local_cols, nontrivial


def _direction_matrices(
    right: np.ndarray, down: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (down.T, down, right.T, right)


def _build_candidate_claims_with_diagnostics(
    right: np.ndarray,
    down: np.ndarray,
    components: tuple[RigidComponent, ...],
    owner: np.ndarray,
    nontrivial_component_ids: frozenset[int],
) -> tuple[tuple[CandidateClaim, ...], _ClaimDiagnostics]:
    r = _dense(right, label="right")
    d = _dense(down, label="down")
    owner_value = np.asarray(owner)
    if owner_value.shape != (NUM_TILES,) or owner_value.dtype != np.int64:
        raise CandidateOracleError("component owner must be int64 length 576")
    component_sizes = {component.component_id: component.size for component in components}
    if set(component_sizes) != set(range(len(components))):
        raise CandidateOracleError("component IDs are not contiguous and canonical")
    expected_owner = np.full(NUM_TILES, -1, dtype=np.int64)
    for component in components:
        if component.entries != tuple(sorted(component.entries)) or component.size < 1:
            raise CandidateOracleError("component entries are not canonical")
        for tile, _row, _col in component.entries:
            if not 0 <= int(tile) < NUM_TILES or expected_owner[int(tile)] >= 0:
                raise CandidateOracleError("components do not uniquely partition tile IDs")
            expected_owner[int(tile)] = component.component_id
    if not np.array_equal(expected_owner, owner_value):
        raise CandidateOracleError("component owner does not match component entries")
    if any(
        int(owner_value[tile]) not in component_sizes for tile in range(NUM_TILES)
    ):
        raise CandidateOracleError("component owner references an unknown component")
    expected_nontrivial = frozenset(
        component_id for component_id, size in component_sizes.items() if size >= 2
    )
    if nontrivial_component_ids != expected_nontrivial:
        raise CandidateOracleError("nontrivial component set drifted")

    tile_ids = np.arange(NUM_TILES, dtype=np.int64)
    claims: list[CandidateClaim] = []
    emitter_tiles = tuple(
        tile
        for tile in range(NUM_TILES)
        if int(owner_value[tile]) in nontrivial_component_ids
    )
    selected_before_filter = 0
    same_component_filtered = 0
    nontrivial_target_claims = 0
    singleton_target_claims = 0
    for direction, matrix in enumerate(_direction_matrices(r, d)):
        dy, dx = DELTAS[direction]
        for anchor in emitter_tiles:
            anchor_component = int(owner_value[anchor])
            order = np.lexsort((tile_ids, -matrix[anchor].astype(np.float64)))
            selected: list[tuple[int, float]] = []
            for target_value in order.tolist():
                target = int(target_value)
                score = float(matrix[anchor, target])
                if score <= 0.0:
                    break
                if target == anchor:
                    raise CandidateOracleError("positive dense diagonal escaped validation")
                selected.append((target, score))
                if len(selected) == CANDIDATE_TOP_K:
                    break
            selected_before_filter += len(selected)
            for target, score in selected:
                target_component = int(owner_value[target])
                if target_component == anchor_component:
                    same_component_filtered += 1
                    continue
                if target_component in nontrivial_component_ids:
                    nontrivial_target_claims += 1
                else:
                    if component_sizes[target_component] != 1:
                        raise CandidateOracleError("target component is neither rigid nor singleton")
                    singleton_target_claims += 1
                claims.append(
                    CandidateClaim(
                        claim_id=len(claims),
                        score=score,
                        anchor=anchor,
                        target=target,
                        dy=dy,
                        dx=dx,
                        anchor_component=anchor_component,
                        target_component=target_component,
                    )
                )
    diagnostics = _ClaimDiagnostics(
        emitter_tiles=len(emitter_tiles),
        directional_emitter_rows=len(emitter_tiles) * len(DELTAS),
        positive_top8_before_component_filter=selected_before_filter,
        same_component_filtered=same_component_filtered,
        nontrivial_target_claims=nontrivial_target_claims,
        singleton_target_claims=singleton_target_claims,
    )
    if selected_before_filter != same_component_filtered + len(claims):
        raise CandidateOracleError("top-eight prefilter accounting drifted")
    if len(claims) != nontrivial_target_claims + singleton_target_claims:
        raise CandidateOracleError("cross-component target accounting drifted")
    return tuple(claims), diagnostics


def build_candidate_claims(
    right: np.ndarray,
    down: np.ndarray,
    components: tuple[RigidComponent, ...],
    owner: np.ndarray,
    nontrivial_component_ids: frozenset[int],
) -> tuple[CandidateClaim, ...]:
    """Return cross-component claims after the literal prefilter top-eight."""

    claims, _diagnostics = _build_candidate_claims_with_diagnostics(
        right,
        down,
        components,
        owner,
        nontrivial_component_ids,
    )
    return claims


def _canonical_observation(
    claim: CandidateClaim,
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> tuple[PoseRelation, PhysicalSeam, int, float]:
    if not isfinite(float(claim.score)) or float(claim.score) <= 0.0:
        raise CandidateOracleError("candidate claim score must be finite and positive")
    if (int(claim.dy), int(claim.dx)) not in DELTAS:
        raise CandidateOracleError("candidate claim direction is not cardinal")
    anchor = int(claim.anchor)
    target = int(claim.target)
    if not (0 <= anchor < NUM_TILES and 0 <= target < NUM_TILES) or anchor == target:
        raise CandidateOracleError("candidate claim tile IDs are invalid")
    anchor_component = int(claim.anchor_component)
    target_component = int(claim.target_component)
    if (
        int(owner[anchor]) != anchor_component
        or int(owner[target]) != target_component
        or anchor_component == target_component
    ):
        raise CandidateOracleError("candidate claim component ownership drifted")

    raw_offset = (
        int(local_rows[anchor]) + int(claim.dy) - int(local_rows[target]),
        int(local_cols[anchor]) + int(claim.dx) - int(local_cols[target]),
    )
    if anchor_component < target_component:
        relation = (
            anchor_component,
            target_component,
            raw_offset[0],
            raw_offset[1],
        )
    else:
        relation = (
            target_component,
            anchor_component,
            -raw_offset[0],
            -raw_offset[1],
        )

    if (claim.dy, claim.dx) in ((0, 1), (1, 0)):
        seam = (anchor, target, int(claim.dy), int(claim.dx))
        orientation = 0
    else:
        seam = (target, anchor, -int(claim.dy), -int(claim.dx))
        orientation = 1
    return relation, seam, orientation, float(claim.score)


def build_pose_hypotheses(
    claims: tuple[CandidateClaim, ...],
    owner: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> tuple[PoseHypothesis, ...]:
    """Group all exact pair/offsets with canonical physical seam evidence."""

    for label, value in (
        ("owner", owner),
        ("local_rows", local_rows),
        ("local_cols", local_cols),
    ):
        array = np.asarray(value)
        if array.shape != (NUM_TILES,) or array.dtype != np.int64:
            raise CandidateOracleError(f"{label} must be int64 length 576")
    if tuple(claim.claim_id for claim in claims) != tuple(range(len(claims))):
        raise CandidateOracleError("candidate claim IDs are not contiguous and stable")

    grouped: dict[PoseRelation, dict[PhysicalSeam, dict[int, float]]] = {}
    seam_relation: dict[PhysicalSeam, PoseRelation] = {}
    for claim in claims:
        relation, seam, orientation, score = _canonical_observation(
            claim, owner, local_rows, local_cols
        )
        previous_relation = seam_relation.setdefault(seam, relation)
        if previous_relation != relation:
            raise CandidateOracleError("one physical seam implied multiple pose relations")
        observations = grouped.setdefault(relation, {}).setdefault(seam, {})
        observations[orientation] = max(observations.get(orientation, 0.0), score)

    output: list[PoseHypothesis] = []
    for hypothesis_id, relation in enumerate(sorted(grouped)):
        seams = grouped[relation]
        seam_scores = tuple(
            (seam, float(max(observations.values())))
            for seam, observations in sorted(seams.items())
        )
        reciprocal = tuple(
            seam
            for seam, observations in sorted(seams.items())
            if set(observations) == {0, 1}
        )
        hypothesis = PoseHypothesis(
            hypothesis_id=hypothesis_id,
            u=relation[0],
            v=relation[1],
            dr=relation[2],
            dc=relation[3],
            seam_scores=seam_scores,
            reciprocal_seams=reciprocal,
        )
        if hypothesis.u >= hypothesis.v or hypothesis.unique_physical_seams < 1:
            raise CandidateOracleError("pose hypothesis canonicalization failed")
        output.append(hypothesis)
    return tuple(output)


def run_posegraph_candidate_oracle(
    right: np.ndarray, down: np.ndarray
) -> CandidatePoolResult:
    """Build the complete frozen E21 label-free raw candidate pool."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    components, owner, local_rows, local_cols, nontrivial = build_components(r, d)
    claims, claim_diagnostics = _build_candidate_claims_with_diagnostics(
        r,
        d,
        components,
        owner,
        nontrivial,
    )
    hypotheses = build_pose_hypotheses(
        claims,
        owner,
        local_rows,
        local_cols,
    )
    pair_offsets: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for hypothesis in hypotheses:
        pair_offsets.setdefault((hypothesis.u, hypothesis.v), set()).add(
            (hypothesis.dr, hypothesis.dc)
        )
    nontrivial_tiles = sum(
        component.size
        for component in components
        if component.component_id in nontrivial
    )
    diagnostics = CandidatePoolDiagnostics(
        component_count=len(components),
        nontrivial_components=len(nontrivial),
        singleton_components=len(components) - len(nontrivial),
        total_tiles=NUM_TILES,
        nontrivial_tiles=nontrivial_tiles,
        singleton_tiles=NUM_TILES - nontrivial_tiles,
        emitter_tiles=claim_diagnostics.emitter_tiles,
        directional_emitter_rows=claim_diagnostics.directional_emitter_rows,
        positive_top8_before_component_filter=(
            claim_diagnostics.positive_top8_before_component_filter
        ),
        same_component_filtered=claim_diagnostics.same_component_filtered,
        claims=len(claims),
        nontrivial_target_claims=claim_diagnostics.nontrivial_target_claims,
        singleton_target_claims=claim_diagnostics.singleton_target_claims,
        hypotheses=len(hypotheses),
        component_pairs=len(pair_offsets),
        component_pairs_with_alternative_offsets=sum(
            len(offsets) > 1 for offsets in pair_offsets.values()
        ),
        unique_physical_seams=sum(
            hypothesis.unique_physical_seams for hypothesis in hypotheses
        ),
        reciprocal_physical_seams=sum(
            hypothesis.reciprocal_physical_seams for hypothesis in hypotheses
        ),
    )
    if diagnostics.emitter_tiles != diagnostics.nontrivial_tiles:
        raise CandidateOracleError("emitter tile count drifted from nontrivial coverage")
    if diagnostics.singleton_tiles != diagnostics.singleton_components:
        raise CandidateOracleError("singleton tile/component accounting drifted")
    if diagnostics.singleton_tiles + diagnostics.nontrivial_tiles != diagnostics.total_tiles:
        raise CandidateOracleError("CC96 partition diagnostics do not cover 576 tiles")
    return CandidatePoolResult(
        components=components,
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial_component_ids=nontrivial,
        claims=claims,
        hypotheses=hypotheses,
        diagnostics=diagnostics,
    )


__all__ = [
    "CANDIDATE_TOP_K",
    "COMPONENT_MAX_EDGES",
    "CandidateClaim",
    "CandidateOracleError",
    "CandidatePoolDiagnostics",
    "CandidatePoolResult",
    "ComponentEntry",
    "DELTAS",
    "GRID",
    "MIN_MARGIN",
    "NUM_TILES",
    "PhysicalSeam",
    "PoseHypothesis",
    "PoseRelation",
    "RigidComponent",
    "build_candidate_claims",
    "build_components",
    "build_pose_hypotheses",
    "run_posegraph_candidate_oracle",
]
