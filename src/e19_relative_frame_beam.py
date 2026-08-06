"""Fixed E19 global-translation quotient viability beam.

E19 changes one E18 search variable: globally shifted copies of the same
relative rigid-component layout are represented by one state.  The decoder
therefore works on a signed sparse coordinate plane with the largest CC192
component fixed at relative translation ``(0, 0)``.  It never constructs an
absolute 24x24 board, completes residual tiles, or consumes labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

import e18_absolute_frame_beam as e18


GRID = e18.GRID
NUM_TILES = e18.NUM_TILES
COMPONENT_MAX_EDGES = e18.COMPONENT_MAX_EDGES
MIN_MARGIN = e18.MIN_MARGIN
CANDIDATE_TOP_K = e18.CANDIDATE_TOP_K
BEAM_WIDTH = e18.BEAM_WIDTH
PROPOSALS_PER_STATE = e18.PROPOSALS_PER_STATE
MAX_ATTACHMENTS = e18.MAX_ATTACHMENTS
RELATIVE_LAYOUTS = e18.ABSOLUTE_LAYOUTS
EXPANSION_CAP = e18.EXPANSION_CAP
DELTAS = e18.DELTAS

BridgeClaim = e18.BridgeClaim
GraphData = e18.GraphData
PhysicalSeam = e18.PhysicalSeam
ComponentPair = e18.ComponentPair
RelativeEntry = tuple[int, int, int]  # tile, signed relative row, signed relative col
BBox = tuple[int, int, int, int]  # minimum row, maximum row, minimum col, maximum col
Proposal = tuple[int, int, int, tuple[int, ...], float, float]

build_graph_data = e18.build_graph_data


class RelativeFrameError(ValueError):
    """An E19 input or relative-search invariant violated the frozen contract."""


class RelativeFrameCapError(RelativeFrameError):
    """The cumulative distinct relative state/candidate cap was reached."""

    def __init__(self, proposal_evaluations: int, rounds: int) -> None:
        super().__init__(
            "relative beam reached the frozen cumulative proposal cap"
        )
        self.proposal_evaluations = int(proposal_evaluations)
        self.rounds = int(rounds)
        self.initial_states = 1
        self.cap_hit = True


def _bbox(entries: Sequence[RelativeEntry]) -> BBox:
    if not entries:
        raise RelativeFrameError("relative layout has no occupied tile")
    rows = [int(row) for _tile, row, _col in entries]
    cols = [int(col) for _tile, _row, col in entries]
    return (min(rows), max(rows), min(cols), max(cols))


def _bbox_shape(value: BBox) -> tuple[int, int]:
    return (value[1] - value[0] + 1, value[3] - value[2] + 1)


def _legal_origin_bounds(value: BBox) -> BBox:
    """Return inclusive legal absolute shifts without materialising an origin."""

    minimum_row, maximum_row, minimum_col, maximum_col = value
    return (
        -minimum_row,
        GRID - 1 - maximum_row,
        -minimum_col,
        GRID - 1 - maximum_col,
    )


def _legal_origin_count(value: BBox) -> int:
    row_low, row_high, col_low, col_high = _legal_origin_bounds(value)
    if row_low > row_high or col_low > col_high:
        return 0
    return int((row_high - row_low + 1) * (col_high - col_low + 1))


@dataclass(frozen=True)
class RelativeState:
    translations: tuple[tuple[int, int, int], ...]
    relative_entries: tuple[RelativeEntry, ...]
    bbox: BBox
    satisfied_bridge_claims: frozenset[int]
    component_contacts: frozenset[ComponentPair]
    cross_seams: frozenset[PhysicalSeam]
    cross_neural_sum: float
    cross_lab_sum: float
    rigid_tiles: int

    @property
    def component_cycle_rank(self) -> int:
        return max(0, len(self.component_contacts) - len(self.translations) + 1)

    @property
    def component_cycle_rank_ratio(self) -> float:
        return float(
            self.component_cycle_rank / max(1, len(self.translations) - 1)
        )


@dataclass(frozen=True)
class RelativeLayout:
    translations: tuple[tuple[int, int, int], ...]
    relative_entries: tuple[RelativeEntry, ...]
    satisfied_bridge_claims: tuple[int, ...]
    component_contacts: tuple[ComponentPair, ...]
    cross_seams: tuple[PhysicalSeam, ...]
    cross_neural_sum: float
    cross_lab_sum: float
    rigid_tiles: int
    rigid_coverage: float
    component_cycle_rank: int
    component_cycle_rank_ratio: float
    bbox: BBox
    bbox_height: int
    bbox_width: int
    legal_origin_bounds: BBox
    legal_origin_count: int


@dataclass(frozen=True)
class RelativeBeamDiagnostics:
    cc192_component_count: int
    cc192_nontrivial_components: int
    cc192_nontrivial_tiles: int
    root_component_id: int
    root_component_size: int
    initial_states: int
    bridge_claims: int
    rounds: int
    proposal_evaluations: int
    cap_hit: bool
    layouts_retained: int


@dataclass(frozen=True)
class RelativeBeamResult:
    layouts: tuple[RelativeLayout, ...]
    diagnostics: RelativeBeamDiagnostics


def _state_rank(
    state: RelativeState,
) -> tuple[int, int, int, int, int, float, float]:
    """Literal E18 path/cycle rank on translation-invariant evidence."""

    return (
        state.component_cycle_rank,
        len(state.satisfied_bridge_claims),
        state.rigid_tiles,
        len(state.component_contacts),
        len(state.cross_seams),
        float(state.cross_neural_sum),
        float(state.cross_lab_sum),
    )


def _occupied_map(state: RelativeState) -> dict[tuple[int, int], int]:
    occupied: dict[tuple[int, int], int] = {}
    for tile, row, col in state.relative_entries:
        coordinate = (int(row), int(col))
        if coordinate in occupied:
            raise RelativeFrameError("relative state repeats an occupied coordinate")
        occupied[coordinate] = int(tile)
    return occupied


def _translation_map(state: RelativeState) -> dict[int, tuple[int, int]]:
    return {component: (row, col) for component, row, col in state.translations}


def _validate_quotient_key(state: RelativeState) -> None:
    """Fail closed if a caller injects a globally shifted quotient state."""

    if not state.translations or state.translations[0] != (0, 0, 0):
        raise RelativeFrameError("relative quotient root must remain fixed at (0, 0)")
    component_ids = tuple(value[0] for value in state.translations)
    if component_ids != tuple(sorted(set(component_ids))):
        raise RelativeFrameError("relative translations must have unique sorted components")


def initial_relative_state(graph: GraphData) -> RelativeState:
    """Create the sole quotient root at relative translation ``(0, 0)``."""

    if not graph.components or graph.components[0].size < 2:
        raise RelativeFrameError("CC192 has no nontrivial root component")
    root = graph.components[0]
    entries = tuple(
        sorted((int(tile), int(row), int(col)) for tile, row, col in root.entries)
    )
    value = _bbox(entries)
    height, width = _bbox_shape(value)
    if height > GRID or width > GRID:
        raise RelativeFrameError("root component exceeds the relative 24x24 span")
    return RelativeState(
        translations=((root.component_id, 0, 0),),
        relative_entries=entries,
        bbox=value,
        satisfied_bridge_claims=frozenset(),
        component_contacts=frozenset(),
        cross_seams=frozenset(),
        cross_neural_sum=0.0,
        cross_lab_sum=0.0,
        rigid_tiles=root.size,
    )


def induced_relative_translations(
    state: RelativeState, graph: GraphData
) -> tuple[Proposal, ...]:
    """Return E18-ordered top-64 translations on the signed sparse plane."""

    _validate_quotient_key(state)
    placed = {component for component, _row, _col in state.translations}
    occupied = _occupied_map(state)
    grouped: dict[tuple[int, int, int], dict[int, float]] = {}
    for (row, col), anchor in sorted(occupied.items()):
        for direction, (dy, dx) in enumerate(DELTAS):
            frontier_row, frontier_col = row + dy, col + dx
            if (frontier_row, frontier_col) in occupied:
                continue
            for claim in graph.claims_by_frontier.get((anchor, direction), ()):
                component_id = claim.target_component
                if component_id in placed or component_id not in graph.nontrivial:
                    continue
                shift_row = frontier_row - int(graph.local_rows[claim.target])
                shift_col = frontier_col - int(graph.local_cols[claim.target])
                grouped.setdefault((component_id, shift_row, shift_col), {})[
                    claim.claim_id
                ] = claim.score
    ranked: list[Proposal] = [
        (
            component_id,
            shift_row,
            shift_col,
            tuple(sorted(scores)),
            float(sum(scores.values())),
            float(max(scores.values())),
        )
        for (component_id, shift_row, shift_col), scores in grouped.items()
    ]
    ranked.sort(
        key=lambda value: (
            -len(value[3]),
            -value[4],
            -value[5],
            value[0],
            value[1],
            value[2],
        )
    )
    return tuple(ranked[:PROPOSALS_PER_STATE])


def _satisfied_new_claims(
    component_id: int,
    translations: Mapping[int, tuple[int, int]],
    graph: GraphData,
) -> frozenset[int]:
    placed = set(translations)
    satisfied: set[int] = set()
    for claim in graph.claims_by_component.get(component_id, ()):
        if claim.anchor_component not in placed or claim.target_component not in placed:
            continue
        anchor_shift = translations[claim.anchor_component]
        target_shift = translations[claim.target_component]
        anchor_row = anchor_shift[0] + int(graph.local_rows[claim.anchor])
        anchor_col = anchor_shift[1] + int(graph.local_cols[claim.anchor])
        target_row = target_shift[0] + int(graph.local_rows[claim.target])
        target_col = target_shift[1] + int(graph.local_cols[claim.target])
        if (target_row - anchor_row, target_col - anchor_col) == (
            claim.dy,
            claim.dx,
        ):
            satisfied.add(claim.claim_id)
    return frozenset(satisfied)


def _seam_value(
    seam: PhysicalSeam, right: np.ndarray, down: np.ndarray
) -> float:
    first, second, dy, dx = seam
    return e18.e15._contact_value(first, second, dy, dx, right, down)


def place_relative_component(
    state: RelativeState,
    graph: GraphData,
    component_id: int,
    shift_row: int,
    shift_col: int,
    right: np.ndarray,
    down: np.ndarray,
    lab_right: np.ndarray,
    lab_down: np.ndarray,
) -> RelativeState | None:
    """Place one whole rigid island without any absolute-frame clipping."""

    _validate_quotient_key(state)
    placed_ids = {cid for cid, _row, _col in state.translations}
    if (
        component_id < 0
        or component_id >= len(graph.components)
        or component_id not in graph.nontrivial
        or component_id in placed_ids
    ):
        raise RelativeFrameError(
            "relative component ID is not an unplaced rigid island"
        )
    component = graph.components[component_id]
    occupied = _occupied_map(state)
    candidate: dict[tuple[int, int], int] = {}
    for tile, local_row, local_col in component.entries:
        coordinate = (local_row + shift_row, local_col + shift_col)
        if coordinate in occupied or coordinate in candidate:
            return None
        candidate[coordinate] = int(tile)

    candidate_entries = tuple(
        (tile, row, col) for (row, col), tile in candidate.items()
    )
    merged_bbox = _bbox((*state.relative_entries, *candidate_entries))
    height, width = _bbox_shape(merged_bbox)
    if height > GRID or width > GRID:
        return None

    new_seams: set[PhysicalSeam] = set()
    new_pairs: set[ComponentPair] = set()
    positive = 0
    for (row, col), tile in candidate.items():
        for dy, dx in DELTAS:
            neighbour = occupied.get((row + dy, col + dx))
            if neighbour is None:
                continue
            seam = e18.e15._physical_seam_identity(tile, neighbour, dy, dx)
            new_seams.add(seam)
            other_component = int(graph.owner[neighbour])
            pair = tuple(sorted((component_id, other_component)))
            if pair[0] == pair[1]:
                raise RelativeFrameError(
                    "relative cross contact remained inside one component"
                )
            new_pairs.add(pair)
            positive += int(_seam_value(seam, right, down) > 0.0)
    if positive == 0 or not new_seams:
        return None

    translations_map = _translation_map(state)
    translations_map[component_id] = (int(shift_row), int(shift_col))
    translations = tuple(
        sorted((cid, value[0], value[1]) for cid, value in translations_map.items())
    )
    if translations[0] != (0, 0, 0):
        raise RelativeFrameError("relative quotient root translation drifted")
    seams = state.cross_seams | frozenset(new_seams)
    neural_sum = float(
        sum(_seam_value(seam, right, down) for seam in sorted(seams))
    )
    lab_sum = float(
        sum(_seam_value(seam, lab_right, lab_down) for seam in sorted(seams))
    )
    if not isfinite(neural_sum) or not isfinite(lab_sum):
        raise RelativeFrameError("relative cross-seam evidence is non-finite")
    entries = tuple(sorted((*state.relative_entries, *candidate_entries)))
    return RelativeState(
        translations=translations,
        relative_entries=entries,
        bbox=merged_bbox,
        satisfied_bridge_claims=(
            state.satisfied_bridge_claims
            | _satisfied_new_claims(component_id, translations_map, graph)
        ),
        component_contacts=state.component_contacts | frozenset(new_pairs),
        cross_seams=seams,
        cross_neural_sum=neural_sum,
        cross_lab_sum=lab_sum,
        rigid_tiles=state.rigid_tiles + component.size,
    )


def _select_states(
    states: Sequence[RelativeState], count: int
) -> list[RelativeState]:
    """Dedupe exact quotient keys and return the frozen rank best-first."""

    unique: dict[tuple[tuple[int, int, int], ...], RelativeState] = {}
    for state in states:
        _validate_quotient_key(state)
        previous = unique.get(state.translations)
        if previous is None:
            unique[state.translations] = state
            continue
        if (
            _state_rank(previous) != _state_rank(state)
            or previous.relative_entries != state.relative_entries
            or previous.bbox != state.bbox
            or previous.satisfied_bridge_claims != state.satisfied_bridge_claims
            or previous.component_contacts != state.component_contacts
            or previous.cross_seams != state.cross_seams
        ):
            raise RelativeFrameError(
                "identical relative translations produced different evidence"
            )
    ranked = list(unique.values())
    ranked.sort(
        key=lambda state: (
            tuple(-float(value) for value in _state_rank(state)),
            state.translations,
        )
    )
    return ranked[:count]


def relative_path_beam(
    right: np.ndarray,
    down: np.ndarray,
    tiles: np.ndarray,
    graph: GraphData,
) -> tuple[tuple[RelativeState, ...], int, int, bool]:
    """Run the one-root signed relative beam and retain up to eight layouts."""

    r = e18._dense(right, label="right")
    d = e18._dense(down, label="down")
    tile_array = e18._tiles(tiles)
    lab_right, lab_down = e18.e15._lab_pair_matrices(tile_array)
    active = [initial_relative_state(graph)]
    deadends: list[RelativeState] = []
    evaluations = 0
    rounds = 0
    cap_hit = False
    evaluated_keys: set[
        tuple[tuple[tuple[int, int, int], ...], int, int, int]
    ] = set()

    for _round in range(MAX_ATTACHMENTS):
        pool: list[RelativeState] = []
        for state in active:
            children: list[RelativeState] = []
            for component_id, shift_row, shift_col, _claim_ids, _sum, _maximum in induced_relative_translations(
                state, graph
            ):
                evaluation_key = (
                    state.translations,
                    component_id,
                    shift_row,
                    shift_col,
                )
                if evaluation_key in evaluated_keys:
                    continue
                evaluated_keys.add(evaluation_key)
                evaluations += 1
                if evaluations >= EXPANSION_CAP:
                    cap_hit = True
                    break
                child = place_relative_component(
                    state,
                    graph,
                    component_id,
                    shift_row,
                    shift_col,
                    r,
                    d,
                    lab_right,
                    lab_down,
                )
                if child is not None:
                    children.append(child)
            if cap_hit:
                break
            if children:
                pool.extend(_select_states(children, PROPOSALS_PER_STATE))
            else:
                deadends.append(state)
        if cap_hit:
            break
        if not pool:
            active = []
            break
        active = _select_states(pool, BEAM_WIDTH)
        deadends = _select_states(deadends, BEAM_WIDTH)
        rounds += 1

    candidates = [*deadends, *active]
    if not candidates:
        raise RelativeFrameError("relative beam produced no partial layout")
    retained = tuple(_select_states(candidates, RELATIVE_LAYOUTS))
    if not retained:
        raise RelativeFrameError("relative beam retained no layout")
    return retained, evaluations, rounds, cap_hit


def _layout(state: RelativeState) -> RelativeLayout:
    _validate_quotient_key(state)
    height, width = _bbox_shape(state.bbox)
    legal_origin_bounds = _legal_origin_bounds(state.bbox)
    legal_origin_count = _legal_origin_count(state.bbox)
    return RelativeLayout(
        translations=state.translations,
        relative_entries=state.relative_entries,
        satisfied_bridge_claims=tuple(sorted(state.satisfied_bridge_claims)),
        component_contacts=tuple(sorted(state.component_contacts)),
        cross_seams=tuple(sorted(state.cross_seams)),
        cross_neural_sum=float(state.cross_neural_sum),
        cross_lab_sum=float(state.cross_lab_sum),
        rigid_tiles=state.rigid_tiles,
        rigid_coverage=float(state.rigid_tiles / NUM_TILES),
        component_cycle_rank=state.component_cycle_rank,
        component_cycle_rank_ratio=state.component_cycle_rank_ratio,
        bbox=state.bbox,
        bbox_height=height,
        bbox_width=width,
        legal_origin_bounds=legal_origin_bounds,
        legal_origin_count=legal_origin_count,
    )


def run_relative_frame(
    right: np.ndarray,
    down: np.ndarray,
    tiles: np.ndarray,
) -> RelativeBeamResult:
    """Run the frozen label-free E19 quotient and return structure only."""

    r = e18._dense(right, label="right")
    d = e18._dense(down, label="down")
    tile_array = e18._tiles(tiles)
    graph = build_graph_data(r, d)
    states, evaluations, rounds, cap_hit = relative_path_beam(
        r, d, tile_array, graph
    )
    if cap_hit:
        raise RelativeFrameCapError(evaluations, rounds)
    layouts = tuple(_layout(state) for state in states)
    if not layouts or any(layout.legal_origin_count < 1 for layout in layouts):
        raise RelativeFrameError("relative result has no legal analytic origin")
    return RelativeBeamResult(
        layouts=layouts,
        diagnostics=RelativeBeamDiagnostics(
            cc192_component_count=len(graph.components),
            cc192_nontrivial_components=len(graph.nontrivial),
            cc192_nontrivial_tiles=sum(
                component.size
                for component in graph.components
                if component.size >= 2
            ),
            root_component_id=0,
            root_component_size=graph.components[0].size,
            initial_states=1,
            bridge_claims=len(graph.claims),
            rounds=rounds,
            proposal_evaluations=evaluations,
            cap_hit=False,
            layouts_retained=len(layouts),
        ),
    )


__all__ = [
    "BEAM_WIDTH",
    "BBox",
    "BridgeClaim",
    "CANDIDATE_TOP_K",
    "COMPONENT_MAX_EDGES",
    "DELTAS",
    "EXPANSION_CAP",
    "GRID",
    "GraphData",
    "MAX_ATTACHMENTS",
    "MIN_MARGIN",
    "NUM_TILES",
    "PROPOSALS_PER_STATE",
    "RELATIVE_LAYOUTS",
    "RelativeBeamDiagnostics",
    "RelativeBeamResult",
    "RelativeFrameCapError",
    "RelativeFrameError",
    "RelativeLayout",
    "RelativeState",
    "induced_relative_translations",
    "initial_relative_state",
    "place_relative_component",
    "relative_path_beam",
    "run_relative_frame",
]
