"""Fixed E15 CC96-rigid / CC192 two-vote absolute-frame decoder.

The module is deliberately label-free.  Synthetic permutations are consumed
only by the separate evaluator after all components, hypotheses, and boards
have been selected.  Tiles remain upright; the only component degree of
freedom is an integer translation in a hard 24x24 frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from rank96_lab_selector import lab_depth1_board_score, scaled_lab_tiles
from solve_buddies import _candidate_edges, build_buddies_components


GRID = 24
NUM_TILES = GRID * GRID
SEED_MAX_EDGES = 96
VOTE_MAX_EDGES = 192
MIN_DISTINCT_SEAMS = 2
BEAM_WIDTH = 256
PROPOSALS_PER_STATE = 64
RELATIVE_LAYOUTS = 8
ABSOLUTE_LAYOUTS = 8
EXPANSION_CAP = 500_000
SCORE_FLOOR = 1.0e-8
HUNGARIAN_ROUNDS = 2
MIN_MULTI_CONTACTS = 2
NULL_WEIGHT = 0.0
REPAIR_PASSES = 0

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PhysicalSeam = tuple[int, int, int, int]


class FrameConsensusError(ValueError):
    """An E15 input or intermediate state violates the frozen contract."""


@dataclass(frozen=True)
class Component:
    component_id: int
    entries: tuple[tuple[int, int, int], ...]  # tile, local row, local col

    @property
    def tiles(self) -> tuple[int, ...]:
        return tuple(entry[0] for entry in self.entries)

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def minimum_tile(self) -> int:
        return min(self.tiles)

    def positions(self) -> dict[int, tuple[int, int]]:
        return {tile: (row, col) for tile, row, col in self.entries}


@dataclass(frozen=True)
class SeamClaim:
    score: float
    anchor: int
    target: int
    dy: int
    dx: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (self.anchor, self.target, self.dy, self.dx)


@dataclass(frozen=True)
class TranslationHypothesis:
    hypothesis_id: int
    left_component: int
    right_component: int
    offset_row: int  # translation(right) - translation(left)
    offset_col: int
    claims: tuple[SeamClaim, ...]

    @property
    def distinct_seams(self) -> int:
        return len(self.claims)

    @property
    def score_sum(self) -> float:
        return float(sum(claim.score for claim in self.claims))


@dataclass(frozen=True)
class RelativeState:
    translations: tuple[tuple[int, int, int], ...]  # component, row, col
    used_hypotheses: frozenset[int]
    seam_votes: int
    neural_score: float
    lab_tie_score: float


@dataclass(frozen=True)
class RigidGrowth:
    translations: tuple[tuple[int, int, int], ...]
    used_hypotheses: frozenset[int]
    hypothesis_seams: int
    hypothesis_score: float
    contact_seams: int
    contact_score: float
    lab_tie_score: float
    attachments: tuple[tuple[int, int], ...]  # component, supporting seams
    expansions: int
    cap_hit: bool


@dataclass(frozen=True)
class ResidualDiagnostics:
    wave_commits: int
    wave_rounds: int
    hungarian_rounds: int


@dataclass(frozen=True)
class SolveDiagnostics:
    seed_component_count: int
    seed_nontrivial_components: int
    seed_nontrivial_tiles: int
    eligible_hypotheses: int
    relative_layouts: int
    origins_evaluated: int
    origins_retained: int
    expansions: int
    expansion_cap_hit: bool
    rigid_components_placed: int
    rigid_tiles_placed: int
    rigid_coverage: float
    non_seed_attachment_supports: tuple[int, ...]
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
        raise FrameConsensusError(f"{label} must be 576x576")
    if not np.isfinite(matrix).all() or bool((matrix < 0.0).any()):
        raise FrameConsensusError(f"{label} must be finite and nonnegative")
    if bool((np.diag(matrix) != 0.0).any()):
        raise FrameConsensusError(f"{label} diagonal must be zero")
    return matrix


def _tiles(value: np.ndarray) -> np.ndarray:
    tiles = np.asarray(value)
    if tiles.shape != (NUM_TILES, 20, 20, 3) or tiles.dtype != np.uint8:
        raise FrameConsensusError("tiles must be upright uint8 RGB 576x20x20x3")
    return np.ascontiguousarray(tiles)


def _strict_board(value: np.ndarray) -> np.ndarray:
    board = np.asarray(value)
    if board.shape != (NUM_TILES,) or board.dtype.kind not in "iu":
        raise FrameConsensusError("board must be an integer vector of length 576")
    board = board.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(board), np.arange(NUM_TILES, dtype=np.int64)):
        raise FrameConsensusError("board is not a strict tile permutation")
    return np.ascontiguousarray(board)


def _normalise_positions(
    positions: Mapping[int, tuple[int, int]],
) -> tuple[tuple[int, int, int], ...]:
    if not positions:
        raise FrameConsensusError("empty component")
    minimum_row = min(row for row, _ in positions.values())
    minimum_col = min(col for _, col in positions.values())
    entries = tuple(
        sorted(
            (int(tile), int(row - minimum_row), int(col - minimum_col))
            for tile, (row, col) in positions.items()
        )
    )
    coordinates = [(row, col) for _tile, row, col in entries]
    if len(set(coordinates)) != len(coordinates):
        raise FrameConsensusError("component contains a coordinate collision")
    if max(row for row, _ in coordinates) >= GRID or max(col for _, col in coordinates) >= GRID:
        raise FrameConsensusError("component exceeds the 24x24 span")
    return entries


def build_seed_components(
    right: np.ndarray, down: np.ndarray
) -> tuple[tuple[Component, ...], np.ndarray]:
    """Build exact CC96 geometry and add every omitted tile as a singleton."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    raw = build_buddies_components(
        r, d, max_edges=SEED_MAX_EDGES, min_margin=0.0
    )
    used: set[int] = set()
    entries: list[tuple[tuple[int, int, int], ...]] = []
    for component in raw:
        normalised = _normalise_positions(component)
        component_tiles = {tile for tile, _row, _col in normalised}
        if used & component_tiles:
            raise FrameConsensusError("CC96 components overlap in tile identity")
        used.update(component_tiles)
        entries.append(normalised)
    entries.extend(((tile, 0, 0),) for tile in range(NUM_TILES) if tile not in used)
    entries.sort(key=lambda value: (min(item[0] for item in value), -len(value)))
    components = tuple(
        Component(component_id=index, entries=value)
        for index, value in enumerate(entries)
    )
    owner = np.empty(NUM_TILES, dtype=np.int64)
    for component in components:
        for tile in component.tiles:
            owner[tile] = component.component_id
    if not np.array_equal(np.sort(np.concatenate([np.asarray(c.tiles) for c in components])), np.arange(NUM_TILES)):
        raise FrameConsensusError("seed components do not partition all tiles")
    owner.setflags(write=False)
    return components, owner


def selected_claims(
    right: np.ndarray, down: np.ndarray, *, max_edges: int
) -> tuple[SeamClaim, ...]:
    if max_edges not in (SEED_MAX_EDGES, VOTE_MAX_EDGES):
        raise FrameConsensusError("E15 exposes only the fixed 96/192 prefixes")
    edges = _candidate_edges(
        _dense(right, label="right"),
        _dense(down, label="down"),
        max_edges=max_edges,
        min_margin=0.0,
    )
    return tuple(
        SeamClaim(float(score), int(a), int(b), int(dy), int(dx))
        for score, _margin, a, b, dy, dx in edges
    )


def _pair_geometry_legal(
    left: Component, right: Component, offset: tuple[int, int]
) -> bool:
    coordinates = {(row, col) for _tile, row, col in left.entries}
    moved = {
        (row + offset[0], col + offset[1])
        for _tile, row, col in right.entries
    }
    if coordinates & moved:
        return False
    union = coordinates | moved
    rows = [row for row, _ in union]
    cols = [col for _, col in union]
    return max(rows) - min(rows) < GRID and max(cols) - min(cols) < GRID


def build_translation_hypotheses(
    components: Sequence[Component],
    owner: np.ndarray,
    claims: Sequence[SeamClaim],
) -> tuple[TranslationHypothesis, ...]:
    """Group CC192 cross-component claims by one identical translation."""

    by_id = {component.component_id: component for component in components}
    local = {
        tile: (component.component_id, row, col)
        for component in components
        for tile, row, col in component.entries
    }
    grouped: dict[tuple[int, int, int, int], dict[tuple[int, int, int, int], SeamClaim]] = {}
    for claim in claims:
        left_id = int(owner[claim.anchor])
        right_id = int(owner[claim.target])
        if left_id == right_id:
            continue
        _ca, anchor_row, anchor_col = local[claim.anchor]
        _cb, target_row, target_col = local[claim.target]
        offset = (
            anchor_row + claim.dy - target_row,
            anchor_col + claim.dx - target_col,
        )
        if left_id > right_id:
            left_id, right_id = right_id, left_id
            offset = (-offset[0], -offset[1])
        key = (left_id, right_id, int(offset[0]), int(offset[1]))
        grouped.setdefault(key, {})[claim.identity] = claim

    pending: list[tuple[tuple[int, int, int, int], tuple[SeamClaim, ...]]] = []
    for key, unique in grouped.items():
        hypothesis_claims = tuple(
            sorted(unique.values(), key=lambda c: (-c.score, c.identity))
        )
        if len(hypothesis_claims) < MIN_DISTINCT_SEAMS:
            continue
        left_id, right_id, offset_row, offset_col = key
        if not _pair_geometry_legal(
            by_id[left_id], by_id[right_id], (offset_row, offset_col)
        ):
            continue
        pending.append((key, hypothesis_claims))
    pending.sort(
        key=lambda item: (
            -len(item[1]),
            -sum(claim.score for claim in item[1]),
            item[0],
        )
    )
    return tuple(
        TranslationHypothesis(
            hypothesis_id=index,
            left_component=key[0],
            right_component=key[1],
            offset_row=key[2],
            offset_col=key[3],
            claims=claims_value,
        )
        for index, (key, claims_value) in enumerate(pending)
    )


def _lab_pair_matrices(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = scaled_lab_tiles(_tiles(tiles))
    right = np.empty((NUM_TILES, NUM_TILES), dtype=np.float32)
    down = np.empty_like(right)
    for anchor in range(NUM_TILES):
        right_delta = lab[anchor, :, 18, :][None, :, :] - lab[:, :, 1, :]
        down_delta = lab[anchor, 18, :, :][None, :, :] - lab[:, 1, :, :]
        right[anchor] = -np.square(right_delta).mean(axis=(1, 2), dtype=np.float64)
        down[anchor] = -np.square(down_delta).mean(axis=(1, 2), dtype=np.float64)
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    return np.ascontiguousarray(right), np.ascontiguousarray(down)


def _contact_value(
    first: int,
    second: int,
    dy: int,
    dx: int,
    right: np.ndarray,
    down: np.ndarray,
) -> float:
    if (dy, dx) == (0, 1):
        return float(right[first, second])
    if (dy, dx) == (0, -1):
        return float(right[second, first])
    if (dy, dx) == (1, 0):
        return float(down[first, second])
    if (dy, dx) == (-1, 0):
        return float(down[second, first])
    raise FrameConsensusError("contact direction is not cardinal")


def _physical_seam_identity(
    first: int, second: int, dy: int, dx: int
) -> PhysicalSeam:
    """Canonicalise one physical contact to the right/down convention."""

    if (dy, dx) in ((0, 1), (1, 0)):
        return (int(first), int(second), int(dy), int(dx))
    if (dy, dx) == (0, -1):
        return (int(second), int(first), 0, 1)
    if (dy, dx) == (-1, 0):
        return (int(second), int(first), 1, 0)
    raise FrameConsensusError("physical seam direction is not cardinal")


def _hypothesis_evidence(
    hypothesis_ids: Iterable[int],
    hypotheses: Mapping[int, TranslationHypothesis],
) -> tuple[frozenset[PhysicalSeam], float]:
    """Return unique physical claims and their once-only frozen score sum."""

    scores: dict[PhysicalSeam, float] = {}
    for hypothesis_id in hypothesis_ids:
        hypothesis = hypotheses[int(hypothesis_id)]
        for claim in hypothesis.claims:
            identity = _physical_seam_identity(
                claim.anchor, claim.target, claim.dy, claim.dx
            )
            scores[identity] = max(scores.get(identity, -float("inf")), claim.score)
    return frozenset(scores), float(sum(scores.values()))


def _state_translations(state: RelativeState | RigidGrowth) -> dict[int, tuple[int, int]]:
    return {component: (row, col) for component, row, col in state.translations}


def _placed_coordinates(
    translations: Mapping[int, tuple[int, int]],
    components: Mapping[int, Component],
) -> dict[tuple[int, int], int]:
    board: dict[tuple[int, int], int] = {}
    for component_id, (shift_row, shift_col) in translations.items():
        for tile, row, col in components[component_id].entries:
            coordinate = (row + shift_row, col + shift_col)
            if coordinate in board:
                raise FrameConsensusError("relative state contains an overlap")
            board[coordinate] = tile
    return board


def _span_legal(coordinates: Iterable[tuple[int, int]]) -> bool:
    values = list(coordinates)
    if not values:
        return False
    rows = [row for row, _ in values]
    cols = [col for _, col in values]
    return max(rows) - min(rows) < GRID and max(cols) - min(cols) < GRID


def _hypothesis_consistent(
    hypothesis: TranslationHypothesis,
    translations: Mapping[int, tuple[int, int]],
) -> bool:
    if hypothesis.left_component not in translations or hypothesis.right_component not in translations:
        return False
    left = translations[hypothesis.left_component]
    right = translations[hypothesis.right_component]
    return (
        right[0] - left[0] == hypothesis.offset_row
        and right[1] - left[1] == hypothesis.offset_col
    )


def _relative_state_key(state: RelativeState) -> tuple[object, ...]:
    return (
        len(state.used_hypotheses),
        state.seam_votes,
        state.neural_score,
        state.lab_tie_score,
        len(state.translations),
        tuple((-component, -row, -col) for component, row, col in state.translations),
    )


def relative_translation_beam(
    components: Sequence[Component],
    hypotheses: Sequence[TranslationHypothesis],
    lab_right: np.ndarray,
    lab_down: np.ndarray,
) -> tuple[tuple[RelativeState, ...], int, bool]:
    """Place the hypothesis-connected core in a translation-invariant gauge."""

    component_map = {component.component_id: component for component in components}
    hypothesis_map = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    degree = {component.component_id: 0 for component in components}
    for hypothesis in hypotheses:
        degree[hypothesis.left_component] += 1
        degree[hypothesis.right_component] += 1
    seed = min(
        components,
        key=lambda component: (
            -degree[component.component_id],
            -component.size,
            component.minimum_tile,
        ),
    )
    beam = [
        RelativeState(
            translations=((seed.component_id, 0, 0),),
            used_hypotheses=frozenset(),
            seam_votes=0,
            neural_score=0.0,
            lab_tie_score=0.0,
        )
    ]
    expansions = 0
    cap_hit = False
    while beam:
        pool: list[RelativeState] = []
        for state in beam:
            translations = _state_translations(state)
            occupied = _placed_coordinates(translations, component_map)
            induced: dict[tuple[int, int, int], set[int]] = {}
            for hypothesis in hypotheses:
                left_placed = hypothesis.left_component in translations
                right_placed = hypothesis.right_component in translations
                if left_placed == right_placed:
                    continue
                if left_placed:
                    left_shift = translations[hypothesis.left_component]
                    candidate = (
                        hypothesis.right_component,
                        left_shift[0] + hypothesis.offset_row,
                        left_shift[1] + hypothesis.offset_col,
                    )
                else:
                    right_shift = translations[hypothesis.right_component]
                    candidate = (
                        hypothesis.left_component,
                        right_shift[0] - hypothesis.offset_row,
                        right_shift[1] - hypothesis.offset_col,
                    )
                induced.setdefault(candidate, set()).add(hypothesis.hypothesis_id)

            ranked: list[tuple[tuple[object, ...], RelativeState]] = []
            for (component_id, shift_row, shift_col), _inducers in induced.items():
                component = component_map[component_id]
                placed = {
                    (row + shift_row, col + shift_col): tile
                    for tile, row, col in component.entries
                }
                if set(placed) & set(occupied):
                    continue
                if not _span_legal([*occupied, *placed]):
                    continue
                proposal_translations = dict(translations)
                proposal_translations[component_id] = (shift_row, shift_col)
                newly_satisfied = {
                    hypothesis.hypothesis_id
                    for hypothesis in hypotheses
                    if hypothesis.hypothesis_id not in state.used_hypotheses
                    and _hypothesis_consistent(hypothesis, proposal_translations)
                }
                if not newly_satisfied:
                    continue
                expansions += 1
                if expansions >= EXPANSION_CAP:
                    cap_hit = True
                    break
                proposal_used = state.used_hypotheses | newly_satisfied
                proposal_seams, proposal_score = _hypothesis_evidence(
                    proposal_used, hypothesis_map
                )
                lab_score = 0.0
                for (row, col), tile in placed.items():
                    for dy, dx in DELTAS:
                        neighbour = occupied.get((row + dy, col + dx))
                        if neighbour is not None:
                            lab_score += _contact_value(
                                tile, neighbour, dy, dx, lab_right, lab_down
                            )
                updated = RelativeState(
                    translations=tuple(
                        sorted(
                            (cid, value[0], value[1])
                            for cid, value in proposal_translations.items()
                        )
                    ),
                    used_hypotheses=proposal_used,
                    seam_votes=len(proposal_seams),
                    neural_score=proposal_score,
                    lab_tie_score=state.lab_tie_score + lab_score,
                )
                incremental = (
                    len(newly_satisfied),
                    len(proposal_seams) - state.seam_votes,
                    proposal_score - state.neural_score,
                    lab_score,
                    component.size,
                    -component_id,
                    -shift_row,
                    -shift_col,
                )
                ranked.append((incremental, updated))
            if cap_hit:
                break
            ranked.sort(key=lambda item: item[0], reverse=True)
            pool.extend(item[1] for item in ranked[:PROPOSALS_PER_STATE])
        if cap_hit or not pool:
            break
        pool.sort(key=_relative_state_key, reverse=True)
        unique: list[RelativeState] = []
        seen: set[tuple[tuple[int, int, int], ...]] = set()
        for state in pool:
            if state.translations in seen:
                continue
            seen.add(state.translations)
            unique.append(state)
            if len(unique) >= BEAM_WIDTH:
                break
        beam = unique

    if not beam:
        raise FrameConsensusError("relative translation beam lost every state")
    beam.sort(key=_relative_state_key, reverse=True)
    return tuple(beam[:RELATIVE_LAYOUTS]), expansions, cap_hit


def _supported_contacts(
    placed: Mapping[tuple[int, int], int],
    candidate: Mapping[tuple[int, int], int],
    right: np.ndarray,
    down: np.ndarray,
    lab_right: np.ndarray,
    lab_down: np.ndarray,
) -> tuple[dict[PhysicalSeam, float], float]:
    neural: dict[PhysicalSeam, float] = {}
    lab = 0.0
    for (row, col), tile in candidate.items():
        for dy, dx in DELTAS:
            neighbour = placed.get((row + dy, col + dx))
            if neighbour is None:
                continue
            raw = _contact_value(tile, neighbour, dy, dx, right, down)
            if raw <= 0.0:
                continue
            identity = _physical_seam_identity(tile, neighbour, dy, dx)
            neural[identity] = max(neural.get(identity, -float("inf")), raw)
            lab += _contact_value(tile, neighbour, dy, dx, lab_right, lab_down)
    return neural, lab


def grow_rigid_multicontact(
    state: RelativeState,
    components: Sequence[Component],
    hypotheses: Sequence[TranslationHypothesis],
    right: np.ndarray,
    down: np.ndarray,
    lab_right: np.ndarray,
    lab_down: np.ndarray,
    *,
    prior_expansions: int,
) -> RigidGrowth:
    """Attach remaining non-singleton islands only through 2+ supported seams."""

    component_map = {component.component_id: component for component in components}
    hypothesis_map = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    translations = _state_translations(state)
    used_hypotheses = set(state.used_hypotheses)
    hypothesis_seam_ids, hypothesis_score = _hypothesis_evidence(
        used_hypotheses, hypothesis_map
    )
    contact_scores: dict[PhysicalSeam, float] = {}
    lab_score = state.lab_tie_score
    attachments: list[tuple[int, int]] = []
    expansions = prior_expansions
    cap_hit = expansions >= EXPANSION_CAP

    while not cap_hit:
        occupied = _placed_coordinates(translations, component_map)
        frontier: set[tuple[int, int]] = set()
        for row, col in occupied:
            for dy, dx in DELTAS:
                coordinate = (row + dy, col + dx)
                if coordinate not in occupied:
                    frontier.add(coordinate)
        best: tuple[
            tuple[object, ...],
            int,
            int,
            int,
            dict[PhysicalSeam, float],
            float,
            set[int],
            frozenset[PhysicalSeam],
            float,
        ] | None = None
        for component in components:
            if component.component_id in translations or component.size < 2:
                continue
            candidate_shifts = {
                (frontier_row - local_row, frontier_col - local_col)
                for frontier_row, frontier_col in frontier
                for _tile, local_row, local_col in component.entries
            }
            for shift_row, shift_col in sorted(candidate_shifts):
                candidate = {
                    (row + shift_row, col + shift_col): tile
                    for tile, row, col in component.entries
                }
                if set(candidate) & set(occupied):
                    continue
                if not _span_legal([*occupied, *candidate]):
                    continue
                contacts, lab = _supported_contacts(
                    occupied,
                    candidate,
                    right,
                    down,
                    lab_right,
                    lab_down,
                )
                support = len(contacts)
                if support < MIN_MULTI_CONTACTS:
                    continue
                expansions += 1
                if expansions >= EXPANSION_CAP:
                    cap_hit = True
                    break
                proposal_translations = dict(translations)
                proposal_translations[component.component_id] = (shift_row, shift_col)
                new_hypotheses = {
                    hypothesis.hypothesis_id
                    for hypothesis in hypotheses
                    if hypothesis.hypothesis_id not in used_hypotheses
                    and _hypothesis_consistent(hypothesis, proposal_translations)
                }
                proposal_used = used_hypotheses | new_hypotheses
                proposal_seams, proposal_hypothesis_score = _hypothesis_evidence(
                    proposal_used, hypothesis_map
                )
                proposal_contacts = dict(contact_scores)
                proposal_contacts.update(contacts)
                proposal_contact_score = sum(
                    score
                    for identity, score in proposal_contacts.items()
                    if identity not in proposal_seams
                )
                proposal_neural = proposal_hypothesis_score + proposal_contact_score
                key = (
                    len(proposal_used),
                    len(proposal_seams),
                    proposal_neural,
                    lab_score + lab,
                    component.size,
                    -component.component_id,
                    -shift_row,
                    -shift_col,
                )
                candidate_value = (
                    key,
                    component.component_id,
                    shift_row,
                    shift_col,
                    contacts,
                    lab,
                    new_hypotheses,
                    proposal_seams,
                    proposal_hypothesis_score,
                )
                if best is None or candidate_value[0] > best[0]:
                    best = candidate_value
            if cap_hit:
                break
        if cap_hit or best is None:
            break
        (
            _key,
            component_id,
            shift_row,
            shift_col,
            contacts,
            lab,
            new_hypotheses,
            hypothesis_seam_ids,
            hypothesis_score,
        ) = best
        translations[component_id] = (shift_row, shift_col)
        attachments.append((component_id, len(contacts)))
        contact_scores.update(contacts)
        lab_score += lab
        used_hypotheses.update(new_hypotheses)

    contact_scores = {
        identity: score
        for identity, score in contact_scores.items()
        if identity not in hypothesis_seam_ids
    }

    return RigidGrowth(
        translations=tuple(
            sorted((cid, value[0], value[1]) for cid, value in translations.items())
        ),
        used_hypotheses=frozenset(used_hypotheses),
        hypothesis_seams=len(hypothesis_seam_ids),
        hypothesis_score=hypothesis_score,
        contact_seams=len(contact_scores),
        contact_score=float(sum(contact_scores.values())),
        lab_tie_score=lab_score,
        attachments=tuple(attachments),
        expansions=expansions,
        cap_hit=cap_hit,
    )


def _grow_relative_layouts(
    relative: Sequence[RelativeState],
    components: Sequence[Component],
    hypotheses: Sequence[TranslationHypothesis],
    right: np.ndarray,
    down: np.ndarray,
    lab_right: np.ndarray,
    lab_down: np.ndarray,
    *,
    prior_expansions: int,
    prior_cap_hit: bool,
) -> tuple[tuple[RigidGrowth, ...], int, bool]:
    """Spend one cumulative expansion budget across every layout in a scene."""

    growths: list[RigidGrowth] = []
    expansions = int(prior_expansions)
    cap_hit = bool(prior_cap_hit or expansions >= EXPANSION_CAP)
    for state in relative:
        growth = grow_rigid_multicontact(
            state,
            components,
            hypotheses,
            right,
            down,
            lab_right,
            lab_down,
            prior_expansions=expansions,
        )
        growths.append(growth)
        expansions = growth.expansions
        cap_hit = bool(cap_hit or growth.cap_hit or expansions >= EXPANSION_CAP)
        if cap_hit:
            break
    if not growths:
        raise FrameConsensusError("E15 has no relative layout to grow")
    return tuple(growths), expansions, cap_hit


def _normalised_rigid_board(
    growth: RigidGrowth, components: Mapping[int, Component]
) -> tuple[dict[tuple[int, int], int], tuple[int, int]]:
    translations = _state_translations(growth)
    placed = _placed_coordinates(translations, components)
    minimum_row = min(row for row, _ in placed)
    minimum_col = min(col for _, col in placed)
    normalised = {
        (row - minimum_row, col - minimum_col): tile
        for (row, col), tile in placed.items()
    }
    height = max(row for row, _ in normalised) + 1
    width = max(col for _, col in normalised) + 1
    return normalised, (height, width)


def _cell_scores(
    board: np.ndarray,
    cells: Sequence[tuple[int, int]],
    tiles: Sequence[int],
    log_right: np.ndarray,
    log_down: np.ndarray,
    *,
    neighbour_mask: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.zeros((len(cells), len(tiles)), dtype=np.float64)
    tile_values = np.asarray(tiles, dtype=np.int64)
    for index, (row, col) in enumerate(cells):
        if col > 0 and board[row, col - 1] >= 0 and (
            neighbour_mask is None or neighbour_mask[row, col - 1]
        ):
            scores[index] += log_right[int(board[row, col - 1]), tile_values]
        if col + 1 < GRID and board[row, col + 1] >= 0 and (
            neighbour_mask is None or neighbour_mask[row, col + 1]
        ):
            scores[index] += log_right[tile_values, int(board[row, col + 1])]
        if row > 0 and board[row - 1, col] >= 0 and (
            neighbour_mask is None or neighbour_mask[row - 1, col]
        ):
            scores[index] += log_down[int(board[row - 1, col]), tile_values]
        if row + 1 < GRID and board[row + 1, col] >= 0 and (
            neighbour_mask is None or neighbour_mask[row + 1, col]
        ):
            scores[index] += log_down[tile_values, int(board[row + 1, col])]
    return scores


def _occupied_neighbour_count(board: np.ndarray, row: int, col: int) -> int:
    return sum(
        0 <= row + dy < GRID
        and 0 <= col + dx < GRID
        and board[row + dy, col + dx] >= 0
        for dy, dx in DELTAS
    )


def _mutual_best_pairs(scores: np.ndarray) -> list[tuple[int, int]]:
    if scores.size == 0:
        return []
    row_best = np.argmax(scores, axis=1)
    col_best = np.argmax(scores, axis=0)
    return [
        (row, int(column))
        for row, column in enumerate(row_best)
        if int(col_best[int(column)]) == row
    ]


def _origin_preview(
    board: np.ndarray,
    unused: Sequence[int],
    log_right: np.ndarray,
    log_down: np.ndarray,
) -> tuple[int, float]:
    cells = [
        (int(row), int(col))
        for row, col in np.argwhere(board < 0)
        if _occupied_neighbour_count(board, int(row), int(col)) >= MIN_MULTI_CONTACTS
    ]
    if not cells or not unused:
        return (0, 0.0)
    scores = _cell_scores(board, cells, unused, log_right, log_down)
    pairs = _mutual_best_pairs(scores)
    return (len(pairs), float(sum(scores[row, col] for row, col in pairs)))


def _diverse_top_origins(
    candidates: Sequence[tuple[tuple[object, ...], int, int, np.ndarray, object]],
    count: int,
) -> list[tuple[tuple[object, ...], int, int, np.ndarray, object]]:
    """Keep exact-score ties spatially diverse instead of pinning (0,0)."""

    remaining = list(candidates)
    output: list[tuple[tuple[object, ...], int, int, np.ndarray, object]] = []
    while remaining and len(output) < count:
        best_score = max(item[0] for item in remaining)
        tied = [item for item in remaining if item[0] == best_score]
        if not output:
            chosen = min(
                tied,
                key=lambda item: (
                    (item[1] - (GRID - 1) / 2.0) ** 2
                    + (item[2] - (GRID - 1) / 2.0) ** 2,
                    item[1],
                    item[2],
                ),
            )
        else:
            chosen = max(
                tied,
                key=lambda item: (
                    min(
                        (item[1] - selected[1]) ** 2
                        + (item[2] - selected[2]) ** 2
                        for selected in output
                    ),
                    -item[1],
                    -item[2],
                ),
            )
        output.append(chosen)
        remaining = [item for item in remaining if item is not chosen]
    return output


def _absolute_origin_candidates(
    growths: Sequence[RigidGrowth],
    components: Mapping[int, Component],
    log_right: np.ndarray,
    log_down: np.ndarray,
) -> tuple[
    tuple[tuple[RigidGrowth, tuple[int, float], int, int, np.ndarray], ...],
    int,
]:
    """Retain exactly one global set of at most eight absolute layouts."""

    origin_rows: list[
        tuple[tuple[object, ...], int, int, np.ndarray, object]
    ] = []
    evaluated = 0
    for growth in growths:
        rigid, (height, width) = _normalised_rigid_board(growth, components)
        used = set(rigid.values())
        unused = sorted(set(range(NUM_TILES)) - used)
        for shift_row in range(GRID - height + 1):
            for shift_col in range(GRID - width + 1):
                board = np.full((GRID, GRID), -1, dtype=np.int64)
                for (row, col), tile in rigid.items():
                    board[row + shift_row, col + shift_col] = tile
                preview = _origin_preview(board, unused, log_right, log_down)
                neural_evidence = growth.hypothesis_score + growth.contact_score
                preselection_key = (
                    len(growth.used_hypotheses),
                    growth.hypothesis_seams,
                    neural_evidence,
                    preview[0],
                    preview[1],
                    growth.lab_tie_score,
                )
                origin_rows.append(
                    (
                        preselection_key,
                        shift_row,
                        shift_col,
                        board,
                        (growth, preview),
                    )
                )
                evaluated += 1
    retained = _diverse_top_origins(origin_rows, ABSOLUTE_LAYOUTS)
    return (
        tuple(
            (payload[0], payload[1], shift_row, shift_col, board)
            for _score, shift_row, shift_col, board, payload in retained
        ),
        evaluated,
    )


def complete_residual(
    partial_board: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, ResidualDiagnostics]:
    """Run mutual-best 2+ neighbour waves, then exactly two Hungarian rounds."""

    board = np.asarray(partial_board, dtype=np.int64).copy()
    if board.shape != (GRID, GRID):
        raise FrameConsensusError("partial board must be 24x24")
    placed = board[board >= 0]
    if len(np.unique(placed)) != len(placed):
        raise FrameConsensusError("partial board repeats a tile")
    if bool(((placed < 0) | (placed >= NUM_TILES)).any()):
        raise FrameConsensusError("partial board contains an invalid tile ID")
    locked = board >= 0
    unused = sorted(set(range(NUM_TILES)) - set(map(int, placed.tolist())))
    log_right = np.log(np.maximum(_dense(right, label="right"), SCORE_FLOOR)).astype(np.float32)
    log_down = np.log(np.maximum(_dense(down, label="down"), SCORE_FLOOR)).astype(np.float32)
    wave_commits = 0
    wave_rounds = 0
    while unused:
        cells = [
            (int(row), int(col))
            for row, col in np.argwhere(board < 0)
            if _occupied_neighbour_count(board, int(row), int(col)) >= MIN_MULTI_CONTACTS
        ]
        if not cells:
            break
        scores = _cell_scores(board, cells, unused, log_right, log_down)
        pairs = _mutual_best_pairs(scores)
        if not pairs:
            break
        selected_tiles = {unused[column] for _row, column in pairs}
        for cell_index, tile_index in pairs:
            row, col = cells[cell_index]
            board[row, col] = unused[tile_index]
            locked[row, col] = True
        unused = [tile for tile in unused if tile not in selected_tiles]
        wave_commits += len(pairs)
        wave_rounds += 1

    residual_cells = [tuple(map(int, value)) for value in np.argwhere(board < 0)]
    residual_tiles = list(unused)
    completed_rounds = 0
    if residual_cells:
        if len(residual_cells) != len(residual_tiles):
            raise FrameConsensusError("residual cell/tile cardinality differs")
        draft = board.copy()
        for round_index in range(HUNGARIAN_ROUNDS):
            mask = locked if round_index == 0 else None
            scores = _cell_scores(
                draft,
                residual_cells,
                residual_tiles,
                log_right,
                log_down,
                neighbour_mask=mask,
            )
            row_indices, tile_indices = linear_sum_assignment(-scores)
            updated = draft.copy()
            for row_index, tile_index in zip(row_indices.tolist(), tile_indices.tolist()):
                row, col = residual_cells[row_index]
                updated[row, col] = residual_tiles[tile_index]
            draft = updated
            completed_rounds += 1
        board = draft
    flat = _strict_board(board.reshape(-1))
    return flat, ResidualDiagnostics(
        wave_commits=wave_commits,
        wave_rounds=wave_rounds,
        hungarian_rounds=completed_rounds,
    )


def terminal_neural_objective(
    board: np.ndarray, right: np.ndarray, down: np.ndarray
) -> float:
    value = _strict_board(board).reshape(GRID, GRID)
    log_right = np.log(np.maximum(_dense(right, label="right"), SCORE_FLOOR))
    log_down = np.log(np.maximum(_dense(down, label="down"), SCORE_FLOOR))
    horizontal = log_right[value[:, :-1], value[:, 1:]].sum(dtype=np.float64)
    vertical = log_down[value[:-1, :], value[1:, :]].sum(dtype=np.float64)
    result = float((horizontal + vertical) / (2 * GRID * (GRID - 1)))
    if not isfinite(result):
        raise FrameConsensusError("terminal neural objective is non-finite")
    return result


def solve_frame_consensus(
    right: np.ndarray,
    down: np.ndarray,
    tiles: np.ndarray,
) -> SolveResult:
    """Run the one fixed E15 decoder and return a strict upright tile board."""

    r = _dense(right, label="right")
    d = _dense(down, label="down")
    tile_array = _tiles(tiles)
    components, owner = build_seed_components(r, d)
    claims192 = selected_claims(r, d, max_edges=VOTE_MAX_EDGES)
    hypotheses = build_translation_hypotheses(components, owner, claims192)
    lab_right, lab_down = _lab_pair_matrices(tile_array)
    relative, expansions, cap_hit = relative_translation_beam(
        components, hypotheses, lab_right, lab_down
    )
    component_map = {component.component_id: component for component in components}
    growths, expansions, cap_hit = _grow_relative_layouts(
        relative,
        components,
        hypotheses,
        r,
        d,
        lab_right,
        lab_down,
        prior_expansions=expansions,
        prior_cap_hit=cap_hit,
    )

    log_right = np.log(np.maximum(r, SCORE_FLOOR)).astype(np.float32)
    log_down = np.log(np.maximum(d, SCORE_FLOOR)).astype(np.float32)
    absolute_candidates, origins_evaluated = _absolute_origin_candidates(
        growths, component_map, log_right, log_down
    )

    completed: list[
        tuple[tuple[object, ...], np.ndarray, RigidGrowth, ResidualDiagnostics, float, float]
    ] = []
    for growth, preview, shift_row, shift_col, partial in absolute_candidates:
        board, residual = complete_residual(partial, r, d)
        neural = terminal_neural_objective(board, r, d)
        lab = float(lab_depth1_board_score(tile_array, board))
        key = (
            len(growth.used_hypotheses),
            growth.hypothesis_seams,
            neural,
            lab,
            -shift_row,
            -shift_col,
        )
        completed.append((key, board, growth, residual, neural, lab))
    if not completed:
        raise FrameConsensusError("E15 produced no absolute board candidate")
    completed.sort(key=lambda item: item[0], reverse=True)
    _key, board, growth, residual, neural, lab = completed[0]
    placed_component_ids = {component for component, _row, _col in growth.translations}
    rigid_tiles = sum(component_map[index].size for index in placed_component_ids)
    supports = tuple(support for _component, support in growth.attachments)
    if any(support < MIN_MULTI_CONTACTS for support in supports):
        raise FrameConsensusError("a non-seed rigid attachment lacks two seams")
    result_board = _strict_board(board).copy()
    result_board.setflags(write=False)
    return SolveResult(
        board=result_board,
        diagnostics=SolveDiagnostics(
            seed_component_count=len(components),
            seed_nontrivial_components=sum(component.size >= 2 for component in components),
            seed_nontrivial_tiles=sum(component.size for component in components if component.size >= 2),
            eligible_hypotheses=len(hypotheses),
            relative_layouts=len(relative),
            origins_evaluated=origins_evaluated,
            origins_retained=len(absolute_candidates),
            expansions=expansions,
            expansion_cap_hit=cap_hit,
            rigid_components_placed=len(placed_component_ids),
            rigid_tiles_placed=rigid_tiles,
            rigid_coverage=float(rigid_tiles / NUM_TILES),
            non_seed_attachment_supports=supports,
            wave_commits=residual.wave_commits,
            wave_rounds=residual.wave_rounds,
            hungarian_rounds=residual.hungarian_rounds,
            terminal_neural_objective=neural,
            terminal_lab_tie_score=lab,
        ),
    )


__all__ = [
    "ABSOLUTE_LAYOUTS",
    "BEAM_WIDTH",
    "Component",
    "EXPANSION_CAP",
    "FrameConsensusError",
    "HUNGARIAN_ROUNDS",
    "MIN_DISTINCT_SEAMS",
    "NULL_WEIGHT",
    "PROPOSALS_PER_STATE",
    "RELATIVE_LAYOUTS",
    "REPAIR_PASSES",
    "SCORE_FLOOR",
    "SEED_MAX_EDGES",
    "SeamClaim",
    "SolveDiagnostics",
    "SolveResult",
    "TranslationHypothesis",
    "VOTE_MAX_EDGES",
    "build_seed_components",
    "build_translation_hypotheses",
    "complete_residual",
    "selected_claims",
    "solve_frame_consensus",
    "terminal_neural_objective",
]
