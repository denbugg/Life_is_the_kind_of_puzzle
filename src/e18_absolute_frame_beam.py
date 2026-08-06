"""Fixed E18 CC192 absolute-frame sparse-path beam.

The decoder is deliberately label-free.  It receives frozen dense scores and
the original corrupted upright tiles, builds exact CC192 rigid islands, and
searches only integer translations in one hard 24x24 frame.  Target pixels and
permutations belong exclusively to the separate evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

import e15_frame_consensus as e15
from rank96_lab_selector import lab_depth1_board_score
from solve_buddies import build_buddies_components


GRID = 24
NUM_TILES = GRID * GRID
COMPONENT_MAX_EDGES = 192
MIN_MARGIN = 0.0
CANDIDATE_TOP_K = 8
BEAM_WIDTH = 256
PROPOSALS_PER_STATE = 64
MAX_ATTACHMENTS = 64
ABSOLUTE_LAYOUTS = 8
EXPANSION_CAP = 500_000
SCORE_FLOOR = 1.0e-8
HUNGARIAN_ROUNDS = 2
MIN_MULTI_CONTACTS = 2
IDENTITY_BONUS = 0.0
REPAIR_PASSES = 0

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # U, D, L, R
PhysicalSeam = tuple[int, int, int, int]
ComponentPair = tuple[int, int]


class AbsoluteFrameError(ValueError):
    """An E18 input or intermediate state violated the frozen contract."""


@dataclass(frozen=True)
class BridgeClaim:
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
class GraphData:
    components: tuple[e15.Component, ...]
    owner: np.ndarray
    local_rows: np.ndarray
    local_cols: np.ndarray
    nontrivial: frozenset[int]
    claims: tuple[BridgeClaim, ...]
    claims_by_frontier: Mapping[tuple[int, int], tuple[BridgeClaim, ...]]
    claims_by_component: Mapping[int, tuple[BridgeClaim, ...]]


@dataclass(frozen=True)
class PartialState:
    translations: tuple[tuple[int, int, int], ...]
    board: np.ndarray
    satisfied_bridge_claims: frozenset[int]
    component_contacts: frozenset[ComponentPair]
    cross_seams: frozenset[PhysicalSeam]
    cross_neural_sum: float
    cross_lab_sum: float
    rigid_tiles: int
    root_origin: tuple[int, int]

    @property
    def component_cycle_rank(self) -> int:
        return max(0, len(self.component_contacts) - len(self.translations) + 1)

    @property
    def component_cycle_rank_ratio(self) -> float:
        return float(
            self.component_cycle_rank / max(1, len(self.translations) - 1)
        )


@dataclass(frozen=True)
class ResidualCandidate:
    state: PartialState
    board: np.ndarray
    wave_commits: int
    wave_rounds: int
    hungarian_rounds: int
    terminal_neural_objective: float
    terminal_lab_tie_score: float


@dataclass(frozen=True)
class SolveDiagnostics:
    cc192_component_count: int
    cc192_nontrivial_components: int
    cc192_nontrivial_tiles: int
    root_component_id: int
    root_component_size: int
    root_origins_evaluated: int
    bridge_claims: int
    attachment_rounds: int
    proposal_evaluations: int
    expansion_cap_hit: bool
    absolute_layouts_retained: int
    rigid_components_placed: int
    rigid_tiles_placed: int
    rigid_coverage: float
    unplaced_nontrivial_components: int
    unplaced_nontrivial_tiles: int
    satisfied_bridge_claims: int
    unique_component_contacts: int
    unique_physical_cross_seams: int
    component_cycle_rank: int
    component_cycle_rank_ratio: float
    accepted_cross_seams: tuple[PhysicalSeam, ...]
    root_origin: tuple[int, int]
    translations: tuple[tuple[int, int, int], ...]
    wave_commits: int
    wave_rounds: int
    hungarian_rounds: int
    terminal_neural_objective: float
    terminal_lab_tie_score: float


@dataclass(frozen=True)
class SolveResult:
    board: np.ndarray
    diagnostics: SolveDiagnostics


def _dense(value: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.ascontiguousarray(value, dtype=np.float32)
    if matrix.shape != (NUM_TILES, NUM_TILES):
        raise AbsoluteFrameError(f"{label} must be 576x576")
    if not np.isfinite(matrix).all() or bool((matrix < 0.0).any()):
        raise AbsoluteFrameError(f"{label} must be finite and nonnegative")
    if bool((np.diag(matrix) != 0.0).any()):
        raise AbsoluteFrameError(f"{label} diagonal must be exactly zero")
    return matrix


def _tiles(value: np.ndarray) -> np.ndarray:
    tiles = np.asarray(value)
    if tiles.shape != (NUM_TILES, 20, 20, 3) or tiles.dtype != np.uint8:
        raise AbsoluteFrameError("tiles must be upright uint8 RGB 576x20x20x3")
    return np.ascontiguousarray(tiles)


def _strict_board(value: np.ndarray) -> np.ndarray:
    board = np.asarray(value)
    if board.shape != (NUM_TILES,) or board.dtype.kind not in "iu":
        raise AbsoluteFrameError("board must be an integer vector of length 576")
    board = board.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(board), np.arange(NUM_TILES, dtype=np.int64)):
        raise AbsoluteFrameError("board is not a strict tile permutation")
    return np.ascontiguousarray(board)


def _normalise_component(
    positions: Mapping[int, tuple[int, int]],
) -> tuple[tuple[int, int, int], ...]:
    try:
        return e15._normalise_positions(positions)
    except e15.FrameConsensusError as exc:
        raise AbsoluteFrameError(str(exc)) from exc


def build_components(
    right: np.ndarray, down: np.ndarray
) -> tuple[tuple[e15.Component, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Build exact CC192 geometry and stable component/tile lookup arrays."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    raw = build_buddies_components(
        r, d, max_edges=COMPONENT_MAX_EDGES, min_margin=MIN_MARGIN
    )
    used: set[int] = set()
    entries: list[tuple[tuple[int, int, int], ...]] = []
    for component in raw:
        normalised = _normalise_component(component)
        tiles = {tile for tile, _row, _col in normalised}
        if used & tiles:
            raise AbsoluteFrameError("CC192 components overlap in tile identity")
        used.update(tiles)
        entries.append(normalised)
    entries.extend(((tile, 0, 0),) for tile in range(NUM_TILES) if tile not in used)
    entries.sort(key=lambda value: (-len(value), min(item[0] for item in value), value))
    components = tuple(
        e15.Component(component_id=index, entries=value)
        for index, value in enumerate(entries)
    )
    owner = np.full(NUM_TILES, -1, dtype=np.int64)
    local_rows = np.zeros(NUM_TILES, dtype=np.int64)
    local_cols = np.zeros(NUM_TILES, dtype=np.int64)
    for component in components:
        for tile, row, col in component.entries:
            if owner[tile] >= 0:
                raise AbsoluteFrameError("CC192 component partition repeats a tile")
            owner[tile] = component.component_id
            local_rows[tile] = row
            local_cols[tile] = col
    if not np.array_equal(np.sort(np.concatenate([np.asarray(c.tiles) for c in components])), np.arange(NUM_TILES)):
        raise AbsoluteFrameError("CC192 components do not partition all tiles")
    if components[0].size < 2:
        raise AbsoluteFrameError("CC192 has no nontrivial root component")
    for value in (owner, local_rows, local_cols):
        value.setflags(write=False)
    return components, owner, local_rows, local_cols


def _direction_matrices(
    right: np.ndarray, down: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (down.T, down, right.T, right)


def build_bridge_claims(
    right: np.ndarray,
    down: np.ndarray,
    components: Sequence[e15.Component],
    owner: np.ndarray,
) -> tuple[
    tuple[BridgeClaim, ...],
    dict[tuple[int, int], tuple[BridgeClaim, ...]],
    dict[int, tuple[BridgeClaim, ...]],
]:
    """Freeze exact positive dense top-8 U/D/L/R cross-component claims."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    if np.asarray(owner).shape != (NUM_TILES,):
        raise AbsoluteFrameError("component owner must have length 576")
    nontrivial = {component.component_id for component in components if component.size >= 2}
    tile_ids = np.arange(NUM_TILES, dtype=np.int64)
    pending: list[tuple[float, int, int, int, int, int, int]] = []
    for direction, matrix in enumerate(_direction_matrices(r, d)):
        dy, dx = DELTAS[direction]
        for anchor in range(NUM_TILES):
            anchor_component = int(owner[anchor])
            if anchor_component not in nontrivial:
                continue
            order = np.lexsort((tile_ids, -matrix[anchor].astype(np.float64)))
            top_targets: list[tuple[int, float]] = []
            for target_value in order.tolist():
                target = int(target_value)
                score = float(matrix[anchor, target])
                if score <= 0.0:
                    break
                if target == anchor:
                    continue
                top_targets.append((target, score))
                if len(top_targets) == CANDIDATE_TOP_K:
                    break
            for target, score in top_targets:
                target_component = int(owner[target])
                if target_component == anchor_component or target_component not in nontrivial:
                    continue
                pending.append(
                    (
                        score,
                        anchor,
                        target,
                        dy,
                        dx,
                        anchor_component,
                        target_component,
                    )
                )
    claims = tuple(
        BridgeClaim(
            claim_id=index,
            score=value[0],
            anchor=value[1],
            target=value[2],
            dy=value[3],
            dx=value[4],
            anchor_component=value[5],
            target_component=value[6],
        )
        for index, value in enumerate(pending)
    )
    by_frontier_lists: dict[tuple[int, int], list[BridgeClaim]] = {}
    by_component_lists: dict[int, list[BridgeClaim]] = {
        component.component_id: [] for component in components
    }
    for claim in claims:
        direction = DELTAS.index((claim.dy, claim.dx))
        by_frontier_lists.setdefault((claim.anchor, direction), []).append(claim)
        by_component_lists[claim.anchor_component].append(claim)
        by_component_lists[claim.target_component].append(claim)
    by_frontier = {
        key: tuple(sorted(value, key=lambda claim: (-claim.score, claim.target, claim.claim_id)))
        for key, value in by_frontier_lists.items()
    }
    by_component = {
        key: tuple(sorted(value, key=lambda claim: claim.claim_id))
        for key, value in by_component_lists.items()
    }
    return claims, by_frontier, by_component


def build_graph_data(right: np.ndarray, down: np.ndarray) -> GraphData:
    components, owner, local_rows, local_cols = build_components(right, down)
    claims, by_frontier, by_component = build_bridge_claims(
        right, down, components, owner
    )
    return GraphData(
        components=components,
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial=frozenset(
            component.component_id for component in components if component.size >= 2
        ),
        claims=claims,
        claims_by_frontier=by_frontier,
        claims_by_component=by_component,
    )


def _component_shape(component: e15.Component) -> tuple[int, int]:
    return (
        max(row for _tile, row, _col in component.entries) + 1,
        max(col for _tile, _row, col in component.entries) + 1,
    )


def initial_absolute_states(graph: GraphData) -> tuple[PartialState, ...]:
    root = graph.components[0]
    height, width = _component_shape(root)
    states: list[PartialState] = []
    for shift_row in range(GRID - height + 1):
        for shift_col in range(GRID - width + 1):
            board = np.full((GRID, GRID), -1, dtype=np.int64)
            for tile, row, col in root.entries:
                board[row + shift_row, col + shift_col] = tile
            board.setflags(write=False)
            states.append(
                PartialState(
                    translations=((root.component_id, shift_row, shift_col),),
                    board=board,
                    satisfied_bridge_claims=frozenset(),
                    component_contacts=frozenset(),
                    cross_seams=frozenset(),
                    cross_neural_sum=0.0,
                    cross_lab_sum=0.0,
                    rigid_tiles=root.size,
                    root_origin=(shift_row, shift_col),
                )
            )
    if not states:
        raise AbsoluteFrameError("root component has no legal absolute origin")
    return tuple(states)


def _translation_map(state: PartialState) -> dict[int, tuple[int, int]]:
    return {component: (row, col) for component, row, col in state.translations}


def _state_rank(state: PartialState) -> tuple[int, int, int, int, int, float, float]:
    return (
        state.component_cycle_rank,
        len(state.satisfied_bridge_claims),
        state.rigid_tiles,
        len(state.component_contacts),
        len(state.cross_seams),
        float(state.cross_neural_sum),
        float(state.cross_lab_sum),
    )


def _metric_desc_key(values: Sequence[int | float]) -> tuple[float, ...]:
    return tuple(-float(value) for value in values)


def _diverse_tied_states(
    states: Sequence[PartialState], count: int
) -> list[PartialState]:
    """Select by exact rank, diversifying root origins only inside exact ties.

    The cached minimum distance is updated once per selected origin.  This is
    exactly equivalent to recomputing the distance to every prior selection,
    but avoids the former O(n*k^2) exact-tie path created by global shifts.
    """

    unique: dict[tuple[tuple[int, int, int], ...], PartialState] = {}
    for state in states:
        previous = unique.get(state.translations)
        if previous is None:
            unique[state.translations] = state
            continue
        if (
            _state_rank(previous) != _state_rank(state)
            or not np.array_equal(previous.board, state.board)
            or previous.satisfied_bridge_claims != state.satisfied_bridge_claims
            or previous.component_contacts != state.component_contacts
            or previous.cross_seams != state.cross_seams
        ):
            raise AbsoluteFrameError("identical translations produced different evidence")
    output: list[PartialState] = []
    groups: dict[tuple[int, int, int, int, int, float, float], list[PartialState]] = {}
    for state in unique.values():
        groups.setdefault(_state_rank(state), []).append(state)
    for rank in sorted(groups, reverse=True):
        tied = groups[rank]
        if not output and tied and len(output) < count:
            chosen_index = min(
                range(len(tied)),
                key=lambda index: (
                    (tied[index].root_origin[0] - (GRID - 1) / 2.0) ** 2
                    + (tied[index].root_origin[1] - (GRID - 1) / 2.0) ** 2,
                    tied[index].translations,
                ),
            )
            output.append(tied.pop(chosen_index))
        if not tied or len(output) >= count:
            if len(output) >= count:
                break
            continue
        minimum_distances = [
            min(
                (state.root_origin[0] - selected.root_origin[0]) ** 2
                + (state.root_origin[1] - selected.root_origin[1]) ** 2
                for selected in output
            )
            for state in tied
        ]
        while tied and len(output) < count:
            chosen_index = max(
                range(len(tied)),
                key=lambda index: (
                    minimum_distances[index],
                    tuple(
                        (-cid, -row, -col)
                        for cid, row, col in tied[index].translations
                    ),
                ),
            )
            chosen = tied.pop(chosen_index)
            minimum_distances.pop(chosen_index)
            output.append(chosen)
            for index, state in enumerate(tied):
                distance = (
                    (state.root_origin[0] - chosen.root_origin[0]) ** 2
                    + (state.root_origin[1] - chosen.root_origin[1]) ** 2
                )
                minimum_distances[index] = min(minimum_distances[index], distance)
        if len(output) >= count:
            break
    return output


def induced_translations(
    state: PartialState, graph: GraphData
) -> tuple[tuple[int, int, int, tuple[int, ...], float, float], ...]:
    """Return frozen top-64 distinct frontier-induced translations.

    The pre-geometry order is distinct supporting-claim count, claim-score
    sum, maximum claim score, then component/row/column, all descending for
    evidence and ascending for identifiers/coordinates.
    """

    placed = {component for component, _row, _col in state.translations}
    grouped: dict[tuple[int, int, int], dict[int, float]] = {}
    for row_value, col_value in np.argwhere(state.board >= 0):
        row, col = int(row_value), int(col_value)
        anchor = int(state.board[row, col])
        for direction, (dy, dx) in enumerate(DELTAS):
            frontier_row, frontier_col = row + dy, col + dx
            if not (0 <= frontier_row < GRID and 0 <= frontier_col < GRID):
                continue
            if state.board[frontier_row, frontier_col] >= 0:
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
    ranked = [
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


def _seam_value(
    seam: PhysicalSeam, right: np.ndarray, down: np.ndarray
) -> float:
    first, second, dy, dx = seam
    return e15._contact_value(first, second, dy, dx, right, down)


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


def place_induced_component(
    state: PartialState,
    graph: GraphData,
    component_id: int,
    shift_row: int,
    shift_col: int,
    right: np.ndarray,
    down: np.ndarray,
    lab_right: np.ndarray,
    lab_down: np.ndarray,
) -> PartialState | None:
    placed_ids = {cid for cid, _row, _col in state.translations}
    if (
        component_id < 0
        or component_id >= len(graph.components)
        or component_id not in graph.nontrivial
        or component_id in placed_ids
    ):
        raise AbsoluteFrameError("induced component ID is not an unplaced rigid island")
    component = graph.components[component_id]
    candidate: dict[tuple[int, int], int] = {}
    for tile, local_row, local_col in component.entries:
        row, col = local_row + shift_row, local_col + shift_col
        if not (0 <= row < GRID and 0 <= col < GRID):
            return None
        if state.board[row, col] >= 0 or (row, col) in candidate:
            return None
        candidate[(row, col)] = tile
    new_seams: set[PhysicalSeam] = set()
    new_pairs: set[ComponentPair] = set()
    positive = 0
    for (row, col), tile in candidate.items():
        for dy, dx in DELTAS:
            neighbour_row, neighbour_col = row + dy, col + dx
            if not (0 <= neighbour_row < GRID and 0 <= neighbour_col < GRID):
                continue
            neighbour = int(state.board[neighbour_row, neighbour_col])
            if neighbour < 0:
                continue
            seam = e15._physical_seam_identity(tile, neighbour, dy, dx)
            new_seams.add(seam)
            other_component = int(graph.owner[neighbour])
            pair = tuple(sorted((component_id, other_component)))
            if pair[0] == pair[1]:
                raise AbsoluteFrameError("cross contact remained inside one component")
            new_pairs.add(pair)
            positive += int(_seam_value(seam, right, down) > 0.0)
    if positive == 0 or not new_seams:
        return None
    translations_map = _translation_map(state)
    translations_map[component_id] = (shift_row, shift_col)
    translations = tuple(
        sorted((cid, value[0], value[1]) for cid, value in translations_map.items())
    )
    board = state.board.copy()
    for coordinate, tile in candidate.items():
        board[coordinate] = tile
    board.setflags(write=False)
    seams = state.cross_seams | frozenset(new_seams)
    neural_sum = float(
        sum(_seam_value(seam, right, down) for seam in sorted(seams))
    )
    lab_sum = float(
        sum(_seam_value(seam, lab_right, lab_down) for seam in sorted(seams))
    )
    if not isfinite(neural_sum) or not isfinite(lab_sum):
        raise AbsoluteFrameError("cross-seam evidence is non-finite")
    return PartialState(
        translations=translations,
        board=board,
        satisfied_bridge_claims=(
            state.satisfied_bridge_claims
            | _satisfied_new_claims(component_id, translations_map, graph)
        ),
        component_contacts=state.component_contacts | frozenset(new_pairs),
        cross_seams=seams,
        cross_neural_sum=neural_sum,
        cross_lab_sum=lab_sum,
        rigid_tiles=state.rigid_tiles + component.size,
        root_origin=state.root_origin,
    )


def absolute_path_beam(
    right: np.ndarray,
    down: np.ndarray,
    tiles: np.ndarray,
    graph: GraphData,
) -> tuple[tuple[PartialState, ...], int, int, bool, int]:
    """Run one cumulative absolute beam and retain eight partial layouts."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    tile_array = _tiles(tiles)
    lab_right, lab_down = e15._lab_pair_matrices(tile_array)
    active = list(initial_absolute_states(graph))
    root_origins = len(active)
    deadends: list[PartialState] = []
    evaluations = 0
    cap_hit = False
    rounds = 0
    for _round in range(MAX_ATTACHMENTS):
        pool: list[PartialState] = []
        for state in active:
            children: list[PartialState] = []
            for component_id, shift_row, shift_col, _claim_ids, _sum, _maximum in induced_translations(
                state, graph
            ):
                evaluations += 1
                if evaluations >= EXPANSION_CAP:
                    cap_hit = True
                    break
                child = place_induced_component(
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
                pool.extend(_diverse_tied_states(children, PROPOSALS_PER_STATE))
            else:
                deadends.append(state)
        if cap_hit:
            break
        if not pool:
            active = []
            break
        active = _diverse_tied_states(pool, BEAM_WIDTH)
        deadends = _diverse_tied_states(deadends, BEAM_WIDTH)
        rounds += 1
    candidates = [*deadends, *active]
    if not candidates:
        raise AbsoluteFrameError("absolute beam produced no partial layout")
    retained = tuple(_diverse_tied_states(candidates, ABSOLUTE_LAYOUTS))
    if not retained:
        raise AbsoluteFrameError("absolute beam retained no layout")
    return retained, evaluations, rounds, cap_hit, root_origins


def _final_rank(
    candidate: ResidualCandidate,
) -> tuple[int, int, int, int, int, float, float, float, float]:
    state = candidate.state
    return (
        state.component_cycle_rank,
        len(state.satisfied_bridge_claims),
        state.rigid_tiles,
        len(state.component_contacts),
        len(state.cross_seams),
        state.cross_neural_sum,
        state.cross_lab_sum,
        candidate.terminal_neural_objective,
        candidate.terminal_lab_tie_score,
    )


def solve_absolute_frame(
    right: np.ndarray,
    down: np.ndarray,
    tiles: np.ndarray,
) -> SolveResult:
    """Run the single frozen E18 decoder and return one upright strict board."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    tile_array = _tiles(tiles)
    graph = build_graph_data(r, d)
    partials, evaluations, rounds, cap_hit, root_origins = absolute_path_beam(
        r, d, tile_array, graph
    )
    if cap_hit:
        raise AbsoluteFrameError(
            "absolute beam reached the frozen cumulative proposal cap"
        )
    completed: list[ResidualCandidate] = []
    for state in partials:
        try:
            board, residual = e15.complete_residual(state.board, r, d)
            board = _strict_board(board)
            if residual.hungarian_rounds != HUNGARIAN_ROUNDS:
                raise AbsoluteFrameError(
                    "residual completion did not execute exactly two Hungarian rounds"
                )
            locked = state.board.reshape(-1) >= 0
            if not np.array_equal(
                board[locked], state.board.reshape(-1)[locked]
            ):
                raise AbsoluteFrameError(
                    "residual completion moved a tile in the locked rigid core"
                )
            terminal_neural = e15.terminal_neural_objective(board, r, d)
            terminal_lab = float(lab_depth1_board_score(tile_array, board))
        except e15.FrameConsensusError as exc:
            raise AbsoluteFrameError(str(exc)) from exc
        completed.append(
            ResidualCandidate(
                state=state,
                board=board,
                wave_commits=residual.wave_commits,
                wave_rounds=residual.wave_rounds,
                hungarian_rounds=residual.hungarian_rounds,
                terminal_neural_objective=terminal_neural,
                terminal_lab_tie_score=terminal_lab,
            )
        )
    completed.sort(
        key=lambda candidate: (
            _metric_desc_key(_final_rank(candidate)),
            candidate.state.translations,
        )
    )
    best = completed[0]
    state = best.state
    placed_ids = {component for component, _row, _col in state.translations}
    unplaced = [
        component
        for component in graph.components
        if component.size >= 2 and component.component_id not in placed_ids
    ]
    board = _strict_board(best.board).copy()
    board.setflags(write=False)
    return SolveResult(
        board=board,
        diagnostics=SolveDiagnostics(
            cc192_component_count=len(graph.components),
            cc192_nontrivial_components=len(graph.nontrivial),
            cc192_nontrivial_tiles=sum(
                component.size for component in graph.components if component.size >= 2
            ),
            root_component_id=0,
            root_component_size=graph.components[0].size,
            root_origins_evaluated=root_origins,
            bridge_claims=len(graph.claims),
            attachment_rounds=rounds,
            proposal_evaluations=evaluations,
            expansion_cap_hit=cap_hit,
            absolute_layouts_retained=len(partials),
            rigid_components_placed=len(state.translations),
            rigid_tiles_placed=state.rigid_tiles,
            rigid_coverage=float(state.rigid_tiles / NUM_TILES),
            unplaced_nontrivial_components=len(unplaced),
            unplaced_nontrivial_tiles=sum(component.size for component in unplaced),
            satisfied_bridge_claims=len(state.satisfied_bridge_claims),
            unique_component_contacts=len(state.component_contacts),
            unique_physical_cross_seams=len(state.cross_seams),
            component_cycle_rank=state.component_cycle_rank,
            component_cycle_rank_ratio=state.component_cycle_rank_ratio,
            accepted_cross_seams=tuple(sorted(state.cross_seams)),
            root_origin=state.root_origin,
            translations=state.translations,
            wave_commits=best.wave_commits,
            wave_rounds=best.wave_rounds,
            hungarian_rounds=best.hungarian_rounds,
            terminal_neural_objective=best.terminal_neural_objective,
            terminal_lab_tie_score=best.terminal_lab_tie_score,
        ),
    )


__all__ = [
    "ABSOLUTE_LAYOUTS",
    "AbsoluteFrameError",
    "BEAM_WIDTH",
    "BridgeClaim",
    "CANDIDATE_TOP_K",
    "COMPONENT_MAX_EDGES",
    "EXPANSION_CAP",
    "GraphData",
    "HUNGARIAN_ROUNDS",
    "IDENTITY_BONUS",
    "MAX_ATTACHMENTS",
    "MIN_MARGIN",
    "MIN_MULTI_CONTACTS",
    "PROPOSALS_PER_STATE",
    "PartialState",
    "REPAIR_PASSES",
    "SCORE_FLOOR",
    "SolveDiagnostics",
    "SolveResult",
    "absolute_path_beam",
    "build_bridge_claims",
    "build_components",
    "build_graph_data",
    "induced_translations",
    "initial_absolute_states",
    "place_induced_component",
    "solve_absolute_frame",
]
