"""Fixed E20 triangle-supported signed-potential pose forest.

E20 is deliberately structure-only.  It consumes the frozen E18/E19 CC192
rigid islands and positive dense top-eight bridge claims, but it never builds
an absolute board, completes residual tiles, rotates a fragment, or consumes
ground-truth labels.  Exact two-leg closures rank a single irreversible
potential-DSU/Kruskal pass.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

import e18_absolute_frame_beam as e18


GRID = e18.GRID
NUM_TILES = e18.NUM_TILES
COMPONENT_MAX_EDGES = e18.COMPONENT_MAX_EDGES
MIN_MARGIN = e18.MIN_MARGIN
CANDIDATE_TOP_K = e18.CANDIDATE_TOP_K
INCIDENT_TOP_K = 8

PhysicalSeam = e18.PhysicalSeam
PoseRelation = tuple[int, int, int, int]  # u, v, dr, dc with u < v
RelativeEntry = tuple[int, int, int]  # tile, relative row, relative column
BBox = tuple[int, int, int, int]


class TrianglePoseError(ValueError):
    """An E20 input or signed-pose invariant failed closed."""


@dataclass(frozen=True)
class TriangleWitness:
    """One retained exact two-leg closure through a distinct intermediary."""

    intermediary: int
    first_hypothesis: int
    second_hypothesis: int
    bottleneck_sum: float
    bottleneck_max: float
    strong: bool


@dataclass(frozen=True)
class PoseHypothesis:
    """One canonical exact signed translation equation between two islands."""

    hypothesis_id: int
    u: int
    v: int
    dr: int
    dc: int
    seam_scores: tuple[tuple[PhysicalSeam, float], ...]
    reciprocal_seams: tuple[PhysicalSeam, ...]
    triangle_witnesses: tuple[TriangleWitness, ...] = ()

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

    @property
    def triangle_intermediates(self) -> tuple[int, ...]:
        return tuple(witness.intermediary for witness in self.triangle_witnesses)

    @property
    def strong_triangle_intermediates(self) -> tuple[int, ...]:
        return tuple(
            witness.intermediary
            for witness in self.triangle_witnesses
            if witness.strong
        )

    @property
    def triangle_bottleneck_sum(self) -> float:
        return float(sum(witness.bottleneck_sum for witness in self.triangle_witnesses))

    @property
    def independent_paths(self) -> int:
        # Reciprocity is deliberately absent: two observations of one physical
        # boundary are rank evidence, not two independent geometric paths.
        return self.unique_physical_seams + len(self.triangle_witnesses)

    @property
    def eligible(self) -> bool:
        return self.independent_paths >= 2


@dataclass(frozen=True)
class PoseCluster:
    """One normalized sparse rigid-island pose cluster; never a board."""

    component_ids: tuple[int, ...]
    translations: tuple[tuple[int, int, int], ...]
    relative_entries: tuple[RelativeEntry, ...]
    bbox: BBox
    bbox_height: int
    bbox_width: int
    legal_origin_bounds: BBox
    legal_origin_count: int
    tree_hypothesis_ids: tuple[int, ...]
    cycle_hypothesis_ids: tuple[int, ...]
    accepted_hypothesis_ids: tuple[int, ...]
    accepted_relations: tuple[PoseRelation, ...]
    component_contacts: tuple[tuple[int, int], ...]
    accepted_cross_seams: tuple[PhysicalSeam, ...]
    rigid_tiles: int
    rigid_coverage: float
    component_cycle_rank: int
    component_cycle_rank_ratio: float
    cross_neural_sum: float
    minimum_tile: int


@dataclass(frozen=True)
class TrianglePoseDiagnostics:
    cc192_component_count: int
    cc192_nontrivial_components: int
    cc192_nontrivial_tiles: int
    bridge_claims: int
    pose_hypotheses: int
    triangle_supported_hypotheses: int
    eligible_hypotheses: int
    weak_hypotheses: int
    eligible_processed: int
    tree_merges: int
    cycle_acceptances: int
    pose_conflicts: int
    contact_rejections: int
    collision_rejections: int
    span_rejections: int
    cluster_count: int
    selected_components: int
    selected_rigid_tiles: int


@dataclass(frozen=True)
class TrianglePoseResult:
    selected: PoseCluster
    clusters: tuple[PoseCluster, ...]
    hypotheses: tuple[PoseHypothesis, ...]
    diagnostics: TrianglePoseDiagnostics


def _add(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    return (first[0] + second[0], first[1] + second[1])


def _sub(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    return (first[0] - second[0], first[1] - second[1])


def _neg(value: tuple[int, int]) -> tuple[int, int]:
    return (-value[0], -value[1])


def _canonical_observation(
    claim: e18.BridgeClaim, graph: e18.GraphData
) -> tuple[PoseRelation, PhysicalSeam, int, float]:
    """Return relation, physical seam, observation orientation and score."""

    if not isfinite(float(claim.score)) or float(claim.score) <= 0.0:
        raise TrianglePoseError("bridge claims must have finite positive scores")
    if (int(claim.dy), int(claim.dx)) not in e18.DELTAS:
        raise TrianglePoseError("bridge claim direction is not cardinal")
    anchor = int(claim.anchor)
    target = int(claim.target)
    if not (0 <= anchor < NUM_TILES and 0 <= target < NUM_TILES):
        raise TrianglePoseError("bridge claim tile is outside 0..575")
    anchor_component = int(claim.anchor_component)
    target_component = int(claim.target_component)
    if (
        int(graph.owner[anchor]) != anchor_component
        or int(graph.owner[target]) != target_component
    ):
        raise TrianglePoseError("bridge claim component owner is inconsistent")
    if (
        anchor_component == target_component
        or anchor_component not in graph.nontrivial
        or target_component not in graph.nontrivial
    ):
        raise TrianglePoseError("E20 claims must cross two nontrivial components")

    raw_offset = (
        int(graph.local_rows[anchor])
        + int(claim.dy)
        - int(graph.local_rows[target]),
        int(graph.local_cols[anchor])
        + int(claim.dx)
        - int(graph.local_cols[target]),
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

    seam = e18.e15._physical_seam_identity(
        anchor, target, int(claim.dy), int(claim.dx)
    )
    first, second, seam_dy, seam_dx = seam
    if (
        anchor == first
        and target == second
        and (int(claim.dy), int(claim.dx)) == (seam_dy, seam_dx)
    ):
        orientation = 0
    elif (
        anchor == second
        and target == first
        and (int(claim.dy), int(claim.dx)) == (-seam_dy, -seam_dx)
    ):
        orientation = 1
    else:
        raise TrianglePoseError("physical seam canonicalization is inconsistent")
    return relation, seam, orientation, float(claim.score)


def build_pose_hypotheses(graph: e18.GraphData) -> tuple[PoseHypothesis, ...]:
    """Group every exact pair/offset while deduplicating physical boundaries."""

    by_relation: dict[
        PoseRelation, dict[PhysicalSeam, dict[int, float]]
    ] = {}
    seam_relation: dict[PhysicalSeam, PoseRelation] = {}
    for claim in graph.claims:
        relation, seam, orientation, score = _canonical_observation(claim, graph)
        previous_relation = seam_relation.setdefault(seam, relation)
        if previous_relation != relation:
            raise TrianglePoseError("one physical seam implied multiple pose equations")
        orientations = by_relation.setdefault(relation, {}).setdefault(seam, {})
        orientations[orientation] = max(orientations.get(orientation, 0.0), score)

    output: list[PoseHypothesis] = []
    for hypothesis_id, relation in enumerate(sorted(by_relation)):
        seams = by_relation[relation]
        seam_scores = tuple(
            (seam, float(max(orientations.values())))
            for seam, orientations in sorted(seams.items())
        )
        reciprocal = tuple(
            seam
            for seam, orientations in sorted(seams.items())
            if set(orientations) == {0, 1}
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
        if not hypothesis.physical_seams:
            raise TrianglePoseError("pose hypothesis has no physical seam")
        output.append(hypothesis)
    return tuple(output)


def _direct_incident_key(
    hypothesis: PoseHypothesis, component_id: int
) -> tuple[float | int, ...]:
    if component_id == hypothesis.u:
        other = hypothesis.v
    elif component_id == hypothesis.v:
        other = hypothesis.u
    else:
        raise TrianglePoseError("hypothesis is not incident to requested component")
    return (
        -hypothesis.unique_physical_seams,
        -hypothesis.reciprocal_physical_seams,
        -hypothesis.direct_neural_sum,
        -hypothesis.direct_max_score,
        other,
        hypothesis.u,
        hypothesis.v,
        hypothesis.dr,
        hypothesis.dc,
        hypothesis.hypothesis_id,
    )


def incident_top_hypotheses(
    hypotheses: Sequence[PoseHypothesis],
) -> dict[int, tuple[PoseHypothesis, ...]]:
    """Return the fixed top eight direct hypotheses at every component."""

    incident: dict[int, list[PoseHypothesis]] = {}
    seen_ids: set[int] = set()
    seen_relations: set[PoseRelation] = set()
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id in seen_ids:
            raise TrianglePoseError("hypothesis IDs are not unique")
        if hypothesis.relation in seen_relations:
            raise TrianglePoseError("pose relations are not unique")
        if hypothesis.u >= hypothesis.v:
            raise TrianglePoseError("pose hypothesis component order is not canonical")
        seen_ids.add(hypothesis.hypothesis_id)
        seen_relations.add(hypothesis.relation)
        incident.setdefault(hypothesis.u, []).append(hypothesis)
        incident.setdefault(hypothesis.v, []).append(hypothesis)
    return {
        component_id: tuple(
            sorted(
                values,
                key=lambda hypothesis: _direct_incident_key(
                    hypothesis, component_id
                ),
            )[:INCIDENT_TOP_K]
        )
        for component_id, values in incident.items()
    }


def _offset_from(
    hypothesis: PoseHypothesis, source: int, target: int
) -> tuple[int, int]:
    if source == hypothesis.u and target == hypothesis.v:
        return (hypothesis.dr, hypothesis.dc)
    if source == hypothesis.v and target == hypothesis.u:
        return (-hypothesis.dr, -hypothesis.dc)
    raise TrianglePoseError("triangle leg endpoints do not match its hypothesis")


def add_triangle_support(
    hypotheses: Sequence[PoseHypothesis],
) -> tuple[PoseHypothesis, ...]:
    """Add bounded exact two-leg witnesses without collapsing pair offsets."""

    values = tuple(hypotheses)
    by_relation = {hypothesis.relation: hypothesis for hypothesis in values}
    if len(by_relation) != len(values):
        raise TrianglePoseError("pose relations are not unique")
    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in values}
    if len(by_id) != len(values):
        raise TrianglePoseError("hypothesis IDs are not unique")
    top = incident_top_hypotheses(values)

    # target hypothesis -> intermediary -> best retained witness
    witnesses: dict[int, dict[int, TriangleWitness]] = {}
    for intermediary in sorted(top):
        legs = top[intermediary]
        if len(legs) > INCIDENT_TOP_K:
            raise TrianglePoseError("incident triangle leg bound was exceeded")
        for first_index in range(len(legs)):
            for second_index in range(first_index + 1, len(legs)):
                first = legs[first_index]
                second = legs[second_index]
                first_outer = first.v if first.u == intermediary else first.u
                second_outer = second.v if second.u == intermediary else second.u
                if first_outer == second_outer:
                    continue
                outer_u, outer_v = sorted((first_outer, second_outer))
                offset_to_u = _offset_from(
                    first if first_outer == outer_u else second,
                    intermediary,
                    outer_u,
                )
                offset_to_v = _offset_from(
                    first if first_outer == outer_v else second,
                    intermediary,
                    outer_v,
                )
                composed = _sub(offset_to_v, offset_to_u)
                target = by_relation.get(
                    (outer_u, outer_v, composed[0], composed[1])
                )
                if target is None:
                    continue
                leg_ids = tuple(sorted((first.hypothesis_id, second.hypothesis_id)))
                candidate = TriangleWitness(
                    intermediary=intermediary,
                    first_hypothesis=leg_ids[0],
                    second_hypothesis=leg_ids[1],
                    bottleneck_sum=float(
                        min(first.direct_neural_sum, second.direct_neural_sum)
                    ),
                    bottleneck_max=float(
                        min(first.direct_max_score, second.direct_max_score)
                    ),
                    strong=(
                        first.unique_physical_seams >= 2
                        and second.unique_physical_seams >= 2
                    ),
                )
                retained = witnesses.setdefault(target.hypothesis_id, {}).get(
                    intermediary
                )
                candidate_key = (
                    -candidate.bottleneck_sum,
                    -candidate.bottleneck_max,
                    candidate.first_hypothesis,
                    candidate.second_hypothesis,
                )
                if retained is None:
                    witnesses[target.hypothesis_id][intermediary] = candidate
                else:
                    retained_key = (
                        -retained.bottleneck_sum,
                        -retained.bottleneck_max,
                        retained.first_hypothesis,
                        retained.second_hypothesis,
                    )
                    if candidate_key < retained_key:
                        witnesses[target.hypothesis_id][intermediary] = candidate

    return tuple(
        replace(
            hypothesis,
            triangle_witnesses=tuple(
                witnesses.get(hypothesis.hypothesis_id, {}).get(intermediary)
                for intermediary in sorted(
                    witnesses.get(hypothesis.hypothesis_id, {})
                )
            ),
        )
        for hypothesis in values
    )


def _kruskal_key(hypothesis: PoseHypothesis) -> tuple[float | int, ...]:
    return (
        -hypothesis.independent_paths,
        -len(hypothesis.triangle_witnesses),
        -len(hypothesis.strong_triangle_intermediates),
        -hypothesis.unique_physical_seams,
        -hypothesis.reciprocal_physical_seams,
        -hypothesis.triangle_bottleneck_sum,
        -hypothesis.direct_neural_sum,
        -hypothesis.direct_max_score,
        hypothesis.u,
        hypothesis.v,
        hypothesis.dr,
        hypothesis.dc,
    )


class _PotentialDSU:
    """Signed Z^2 potential DSU plus sparse aggregate geometry."""

    def __init__(self, graph: e18.GraphData) -> None:
        self.graph = graph
        component_map = {component.component_id: component for component in graph.components}
        self.nodes = tuple(sorted(int(value) for value in graph.nontrivial))
        if not self.nodes:
            raise TrianglePoseError("CC192 has no nontrivial pose component")
        if any(node not in component_map for node in self.nodes):
            raise TrianglePoseError("nontrivial component lookup is incomplete")
        self.component_map = component_map
        self.parent = {node: node for node in self.nodes}
        self.potential = {node: (0, 0) for node in self.nodes}
        self.size = {node: 1 for node in self.nodes}
        self.entries: dict[int, dict[tuple[int, int], int]] = {}
        self.translations: dict[int, dict[int, tuple[int, int]]] = {}
        self.tree_ids: dict[int, set[int]] = {}
        self.cycle_ids: dict[int, set[int]] = {}
        for node in self.nodes:
            occupied: dict[tuple[int, int], int] = {}
            for tile, row, col in component_map[node].entries:
                coordinate = (int(row), int(col))
                if coordinate in occupied:
                    raise TrianglePoseError("CC192 component has a coordinate collision")
                occupied[coordinate] = int(tile)
            if len(occupied) < 2:
                raise TrianglePoseError("pose forest received a singleton component")
            rows = [row for row, _col in occupied]
            cols = [col for _row, col in occupied]
            if max(rows) - min(rows) >= GRID or max(cols) - min(cols) >= GRID:
                raise TrianglePoseError("CC192 component exceeds the 24x24 span")
            self.entries[node] = occupied
            self.translations[node] = {node: (0, 0)}
            self.tree_ids[node] = set()
            self.cycle_ids[node] = set()

    def find(self, node: int) -> tuple[int, tuple[int, int]]:
        if node not in self.parent:
            raise TrianglePoseError("pose hypothesis references a non-pose component")
        parent = self.parent[node]
        if parent == node:
            return node, (0, 0)
        root, parent_potential = self.find(parent)
        value = _add(self.potential[node], parent_potential)
        self.parent[node] = root
        self.potential[node] = value
        return root, value

    def _contacts_valid(
        self,
        hypothesis: PoseHypothesis,
        translations: Mapping[int, tuple[int, int]],
    ) -> bool:
        if not hypothesis.physical_seams:
            return False
        for first, second, dy, dx in hypothesis.physical_seams:
            if (dy, dx) not in ((0, 1), (1, 0)):
                return False
            first_component = int(self.graph.owner[first])
            second_component = int(self.graph.owner[second])
            if {first_component, second_component} != {hypothesis.u, hypothesis.v}:
                return False
            if first_component not in translations or second_component not in translations:
                return False
            first_coordinate = (
                translations[first_component][0] + int(self.graph.local_rows[first]),
                translations[first_component][1] + int(self.graph.local_cols[first]),
            )
            second_coordinate = (
                translations[second_component][0] + int(self.graph.local_rows[second]),
                translations[second_component][1] + int(self.graph.local_cols[second]),
            )
            if _sub(second_coordinate, first_coordinate) != (dy, dx):
                return False
        return True

    def accept(self, hypothesis: PoseHypothesis) -> str:
        """Process one eligible hypothesis and return one literal outcome."""

        if not hypothesis.eligible:
            raise TrianglePoseError("weak hypotheses are diagnostic only")
        root_u, delta_u = self.find(hypothesis.u)
        root_v, delta_v = self.find(hypothesis.v)
        equation = (hypothesis.dr, hypothesis.dc)
        if root_u == root_v:
            if _sub(delta_v, delta_u) != equation:
                return "conflict"
            root_translations = self.translations[root_u]
            if not self._contacts_valid(hypothesis, root_translations):
                return "contact"
            if hypothesis.hypothesis_id in self.tree_ids[root_u]:
                raise TrianglePoseError("tree hypothesis was processed twice")
            self.cycle_ids[root_u].add(hypothesis.hypothesis_id)
            return "cycle"

        # Express root_v in root_u's signed coordinate gauge.
        root_v_in_u = _sub(_add(equation, delta_u), delta_v)
        merged_translations = dict(self.translations[root_u])
        merged_translations.update(
            {
                component: _add(value, root_v_in_u)
                for component, value in self.translations[root_v].items()
            }
        )
        if not self._contacts_valid(hypothesis, merged_translations):
            return "contact"

        shifted_v = {
            _add(coordinate, root_v_in_u): tile
            for coordinate, tile in self.entries[root_v].items()
        }
        if set(self.entries[root_u]) & set(shifted_v):
            return "collision"
        merged_entries = dict(self.entries[root_u])
        merged_entries.update(shifted_v)
        rows = [row for row, _col in merged_entries]
        cols = [col for _row, col in merged_entries]
        if max(rows) - min(rows) >= GRID or max(cols) - min(cols) >= GRID:
            return "span"

        keep_u = (
            self.size[root_u] > self.size[root_v]
            or (self.size[root_u] == self.size[root_v] and root_u < root_v)
        )
        combined_tree = self.tree_ids[root_u] | self.tree_ids[root_v]
        combined_tree.add(hypothesis.hypothesis_id)
        combined_cycle = self.cycle_ids[root_u] | self.cycle_ids[root_v]
        if keep_u:
            self.parent[root_v] = root_u
            self.potential[root_v] = root_v_in_u
            self.size[root_u] += self.size[root_v]
            self.entries[root_u] = merged_entries
            self.translations[root_u] = merged_translations
            self.tree_ids[root_u] = combined_tree
            self.cycle_ids[root_u] = combined_cycle
            discarded = root_v
        else:
            root_u_in_v = _neg(root_v_in_u)
            shifted_u = {
                _add(coordinate, root_u_in_v): tile
                for coordinate, tile in self.entries[root_u].items()
            }
            retained_entries = dict(self.entries[root_v])
            retained_entries.update(shifted_u)
            retained_translations = dict(self.translations[root_v])
            retained_translations.update(
                {
                    component: _add(value, root_u_in_v)
                    for component, value in self.translations[root_u].items()
                }
            )
            self.parent[root_u] = root_v
            self.potential[root_u] = root_u_in_v
            self.size[root_v] += self.size[root_u]
            self.entries[root_v] = retained_entries
            self.translations[root_v] = retained_translations
            self.tree_ids[root_v] = combined_tree
            self.cycle_ids[root_v] = combined_cycle
            discarded = root_u
        del self.entries[discarded]
        del self.translations[discarded]
        del self.tree_ids[discarded]
        del self.cycle_ids[discarded]
        return "tree"


def _cluster(
    root: int,
    dsu: _PotentialDSU,
    hypotheses_by_id: Mapping[int, PoseHypothesis],
) -> PoseCluster:
    entries = dsu.entries[root]
    minimum_row = min(row for row, _col in entries)
    maximum_row = max(row for row, _col in entries)
    minimum_col = min(col for _row, col in entries)
    maximum_col = max(col for _row, col in entries)
    height = maximum_row - minimum_row + 1
    width = maximum_col - minimum_col + 1
    if height > GRID or width > GRID:
        raise TrianglePoseError("final cluster exceeds the legal relative span")
    normalized_entries = tuple(
        sorted(
            (tile, row - minimum_row, col - minimum_col)
            for (row, col), tile in entries.items()
        )
    )
    translations = tuple(
        sorted(
            (
                component,
                value[0] - minimum_row,
                value[1] - minimum_col,
            )
            for component, value in dsu.translations[root].items()
        )
    )
    component_ids = tuple(value[0] for value in translations)
    if component_ids != tuple(sorted(set(component_ids))):
        raise TrianglePoseError("final component translations are not canonical")
    if len({(row, col) for _tile, row, col in normalized_entries}) != len(
        normalized_entries
    ):
        raise TrianglePoseError("final normalized entries collide")

    tree_ids = tuple(sorted(dsu.tree_ids[root]))
    cycle_ids = tuple(sorted(dsu.cycle_ids[root]))
    if set(tree_ids) & set(cycle_ids):
        raise TrianglePoseError("tree and cycle hypothesis evidence overlaps")
    accepted_ids = tuple(sorted((*tree_ids, *cycle_ids)))
    accepted = [hypotheses_by_id[value] for value in accepted_ids]
    if any(not hypothesis.eligible for hypothesis in accepted):
        raise TrianglePoseError("weak hypothesis leaked into selected evidence")
    if len(tree_ids) != max(0, len(component_ids) - 1):
        raise TrianglePoseError("pose cluster tree edge count is inconsistent")

    seam_scores: dict[PhysicalSeam, float] = {}
    for hypothesis in accepted:
        for seam, score in hypothesis.seam_scores:
            seam_scores[seam] = max(seam_scores.get(seam, 0.0), float(score))
    seams = tuple(sorted(seam_scores))
    contacts = tuple(sorted({(hypothesis.u, hypothesis.v) for hypothesis in accepted}))
    cycle_rank = max(0, len(contacts) - len(component_ids) + 1)
    legal_bounds = (0, GRID - height, 0, GRID - width)
    legal_count = int((GRID - height + 1) * (GRID - width + 1))
    return PoseCluster(
        component_ids=component_ids,
        translations=translations,
        relative_entries=normalized_entries,
        bbox=(0, height - 1, 0, width - 1),
        bbox_height=height,
        bbox_width=width,
        legal_origin_bounds=legal_bounds,
        legal_origin_count=legal_count,
        tree_hypothesis_ids=tree_ids,
        cycle_hypothesis_ids=cycle_ids,
        accepted_hypothesis_ids=accepted_ids,
        accepted_relations=tuple(hypothesis.relation for hypothesis in accepted),
        component_contacts=contacts,
        accepted_cross_seams=seams,
        rigid_tiles=len(normalized_entries),
        rigid_coverage=float(len(normalized_entries) / NUM_TILES),
        component_cycle_rank=cycle_rank,
        component_cycle_rank_ratio=float(
            cycle_rank / max(1, len(component_ids) - 1)
        ),
        cross_neural_sum=float(sum(seam_scores.values())),
        minimum_tile=min(tile for tile, _row, _col in normalized_entries),
    )


def _cluster_selection_key(cluster: PoseCluster) -> tuple[object, ...]:
    return (
        -cluster.rigid_tiles,
        -cluster.component_cycle_rank,
        -len(cluster.accepted_cross_seams),
        -cluster.cross_neural_sum,
        cluster.minimum_tile,
        cluster.translations,
    )


def solve_pose_forest(graph: e18.GraphData) -> TrianglePoseResult:
    """Run the one fixed E20 supported-pose Kruskal pass."""

    direct = build_pose_hypotheses(graph)
    hypotheses = add_triangle_support(direct)
    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    eligible = sorted(
        (hypothesis for hypothesis in hypotheses if hypothesis.eligible),
        key=_kruskal_key,
    )
    dsu = _PotentialDSU(graph)
    outcomes = {
        "tree": 0,
        "cycle": 0,
        "conflict": 0,
        "contact": 0,
        "collision": 0,
        "span": 0,
    }
    for hypothesis in eligible:
        outcome = dsu.accept(hypothesis)
        if outcome not in outcomes:
            raise TrianglePoseError("unknown potential-DSU outcome")
        outcomes[outcome] += 1

    roots = tuple(sorted(node for node in dsu.nodes if dsu.find(node)[0] == node))
    clusters = tuple(
        sorted(
            (_cluster(root, dsu, by_id) for root in roots),
            key=_cluster_selection_key,
        )
    )
    if not clusters:
        raise TrianglePoseError("pose forest produced no sparse cluster")
    selected = clusters[0]
    diagnostics = TrianglePoseDiagnostics(
        cc192_component_count=len(graph.components),
        cc192_nontrivial_components=len(graph.nontrivial),
        cc192_nontrivial_tiles=sum(
            component.size
            for component in graph.components
            if component.component_id in graph.nontrivial
        ),
        bridge_claims=len(graph.claims),
        pose_hypotheses=len(hypotheses),
        triangle_supported_hypotheses=sum(
            bool(hypothesis.triangle_witnesses) for hypothesis in hypotheses
        ),
        eligible_hypotheses=len(eligible),
        weak_hypotheses=len(hypotheses) - len(eligible),
        eligible_processed=len(eligible),
        tree_merges=outcomes["tree"],
        cycle_acceptances=outcomes["cycle"],
        pose_conflicts=outcomes["conflict"],
        contact_rejections=outcomes["contact"],
        collision_rejections=outcomes["collision"],
        span_rejections=outcomes["span"],
        cluster_count=len(clusters),
        selected_components=len(selected.component_ids),
        selected_rigid_tiles=selected.rigid_tiles,
    )
    return TrianglePoseResult(
        selected=selected,
        clusters=clusters,
        hypotheses=hypotheses,
        diagnostics=diagnostics,
    )


def run_triangle_potential_dsu(
    right: np.ndarray, down: np.ndarray
) -> TrianglePoseResult:
    """Build exact E18 graph data and run the structure-only E20 solver."""

    graph = e18.build_graph_data(right, down)
    return solve_pose_forest(graph)


__all__ = [
    "BBox",
    "CANDIDATE_TOP_K",
    "COMPONENT_MAX_EDGES",
    "GRID",
    "INCIDENT_TOP_K",
    "MIN_MARGIN",
    "NUM_TILES",
    "PhysicalSeam",
    "PoseCluster",
    "PoseHypothesis",
    "PoseRelation",
    "TrianglePoseDiagnostics",
    "TrianglePoseError",
    "TrianglePoseResult",
    "TriangleWitness",
    "add_triangle_support",
    "build_pose_hypotheses",
    "incident_top_hypotheses",
    "run_triangle_potential_dsu",
    "solve_pose_forest",
]
