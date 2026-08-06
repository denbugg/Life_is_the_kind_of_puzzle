"""Translation-free beam assembly of reliable seed components.

Unlike the deterministic component packer, this solver does not anchor the
largest island to an arbitrary board corner.  It grows a connected layout in
relative coordinates, keeps several competing component translations, and
scores every contact made by a proposed component at once.

The implementation is intentionally a bounded gate.  It consumes the frozen
full-graph caches and reports exact synthetic metrics; it is not used on test
images unless it improves the held-out neighbour baseline.
"""
from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import GRID, NFRAG, WORK_ROOT
from eval_seeded_qap import dense_rd
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import build_buddies_components


DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class BeamState:
    board: dict[tuple[int, int], int]
    remaining: frozenset[int]
    score: float
    contacts: int


def complete_components(
    components: list[dict[int, tuple[int, int]]],
) -> list[dict[int, tuple[int, int]]]:
    """Add every tile omitted by the conservative edge graph as a singleton."""
    used = {tile for component in components for tile in component}
    result = [dict(component) for component in components]
    result.extend({tile: (0, 0)} for tile in range(NFRAG) if tile not in used)
    return result


def canonical_board(
    board: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    minimum_row = min(row for row, _ in board)
    minimum_col = min(col for _, col in board)
    if minimum_row == 0 and minimum_col == 0:
        return board
    return {
        (row - minimum_row, col - minimum_col): tile
        for (row, col), tile in board.items()
    }


def span_ok(coordinates: list[tuple[int, int]]) -> bool:
    rows = [value[0] for value in coordinates]
    cols = [value[1] for value in coordinates]
    return max(rows) - min(rows) < GRID and max(cols) - min(cols) < GRID


def edge_value(
    first: int,
    second: int,
    direction: int,
    right: np.ndarray,
    down: np.ndarray,
) -> float:
    if direction == 0:
        return float(down[second, first])
    if direction == 1:
        return float(down[first, second])
    if direction == 2:
        return float(right[second, first])
    return float(right[first, second])


def top_targets(
    right: np.ndarray,
    down: np.ndarray,
    count: int,
) -> np.ndarray:
    """Top candidate tile IDs for U/D/L/R from every anchor tile."""
    matrices = (down.T, down, right.T, right)
    output = np.empty((4, NFRAG, count), dtype=np.int64)
    for direction, matrix in enumerate(matrices):
        work = matrix.copy()
        np.fill_diagonal(work, -np.inf)
        slots = np.argpartition(-work, count - 1, axis=1)[:, :count]
        values = np.take_along_axis(work, slots, axis=1)
        order = np.argsort(-values, axis=1)
        output[direction] = np.take_along_axis(slots, order, axis=1)
    return output


def component_index(
    components: list[dict[int, tuple[int, int]]],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    owner = np.empty(NFRAG, dtype=np.int64)
    local: dict[int, tuple[int, int]] = {}
    for index, component in enumerate(components):
        for tile, coordinate in component.items():
            owner[tile] = index
            local[tile] = coordinate
    return owner, local


def proposal_delta(
    board: dict[tuple[int, int], int],
    component: dict[int, tuple[int, int]],
    shift: tuple[int, int],
    right: np.ndarray,
    down: np.ndarray,
    contact_bonus: float,
) -> tuple[float, int, list[tuple[int, int]]] | None:
    coordinates = [
        (row + shift[0], col + shift[1]) for row, col in component.values()
    ]
    if any(coordinate in board for coordinate in coordinates):
        return None
    if not span_ok([*board, *coordinates]):
        return None
    placed = {
        (row + shift[0], col + shift[1]): tile
        for tile, (row, col) in component.items()
    }
    score = 0.0
    contacts = 0
    for (row, col), tile in placed.items():
        for direction, (dr, dc) in enumerate(DELTAS):
            neighbour = board.get((row + dr, col + dc))
            if neighbour is None:
                continue
            score += edge_value(neighbour, tile, direction ^ 1, right, down)
            contacts += 1
    if not contacts:
        return None
    score += contact_bonus * max(0, contacts - 1)
    return score, contacts, coordinates


def state_proposals(
    state: BeamState,
    components: list[dict[int, tuple[int, int]]],
    owner: np.ndarray,
    local: dict[int, tuple[int, int]],
    top: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    *,
    per_state: int,
    contact_bonus: float,
    allowed_component: int | None = None,
) -> list[BeamState]:
    candidates: set[tuple[int, int, int]] = set()
    frontier_cells: set[tuple[int, int]] = set()
    for (row, col), anchor in state.board.items():
        for direction, (dr, dc) in enumerate(DELTAS):
            frontier = (row + dr, col + dc)
            if frontier in state.board:
                continue
            frontier_cells.add(frontier)
            for target in top[direction, anchor]:
                component_id = int(owner[target])
                if component_id not in state.remaining:
                    continue
                if (
                    allowed_component is not None
                    and component_id != allowed_component
                ):
                    continue
                target_local = local[int(target)]
                candidates.add(
                    (
                        component_id,
                        frontier[0] - target_local[0],
                        frontier[1] - target_local[1],
                    )
                )
    # Near the end, a remaining tile may no longer occur in any truncated
    # frontier shortlist.  Enumerating the small residual set prevents an
    # artificial top-k dead end.  It is cheap only in this late regime.
    residual_components = (
        (allowed_component,)
        if allowed_component is not None
        else state.remaining
    )
    if (
        allowed_component is not None
        or len(state.remaining) <= max(64, 2 * top.shape[-1])
    ):
        for component_id in residual_components:
            if component_id not in state.remaining:
                continue
            component = components[component_id]
            for target, target_local in component.items():
                for frontier in frontier_cells:
                    candidates.add(
                        (
                            component_id,
                            frontier[0] - target_local[0],
                            frontier[1] - target_local[1],
                        )
                    )

    ranked: list[tuple[float, int, int, int, int, list[tuple[int, int]]]] = []
    for component_id, shift_row, shift_col in candidates:
        result = proposal_delta(
            state.board,
            components[component_id],
            (shift_row, shift_col),
            right,
            down,
            contact_bonus,
        )
        if result is None:
            continue
        delta, contacts, coordinates = result
        # A deterministic tie-breaker is required because component IDs are
        # otherwise sensitive to set iteration order.
        heapq.heappush(
            ranked,
            (
                -delta,
                -contacts,
                component_id,
                shift_row,
                shift_col,
                coordinates,
            ),
        )

    output: list[BeamState] = []
    for _ in range(min(per_state, len(ranked))):
        neg_delta, neg_contacts, component_id, shift_row, shift_col, _ = heapq.heappop(
            ranked
        )
        board = dict(state.board)
        for tile, (row, col) in components[component_id].items():
            board[(row + shift_row, col + shift_col)] = tile
        board = canonical_board(board)
        output.append(
            BeamState(
                board=board,
                remaining=state.remaining - {component_id},
                score=state.score - neg_delta,
                contacts=state.contacts - neg_contacts,
            )
        )
    return output


def signature(state: BeamState) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        sorted((row, col, tile) for (row, col), tile in state.board.items())
    )


def finish_singletons(
    state: BeamState,
    components: list[dict[int, tuple[int, int]]],
    right: np.ndarray,
    down: np.ndarray,
    contact_bonus: float,
) -> BeamState | None:
    """Explode only a dead-end residual component set and fill its tiles.

    This is an exact-cover escape hatch, not the main growth rule.  All
    components already placed by the beam remain rigid.
    """
    board = dict(state.board)
    unused = {
        tile
        for component_id in state.remaining
        for tile in components[component_id]
    }
    score, contacts = state.score, state.contacts
    while unused:
        frontier = set()
        for row, col in board:
            for dr, dc in DELTAS:
                coordinate = (row + dr, col + dc)
                if coordinate not in board:
                    frontier.add(coordinate)
        best = None
        for coordinate in frontier:
            if not span_ok([*board, coordinate]):
                continue
            row, col = coordinate
            for tile in unused:
                delta = 0.0
                count = 0
                for direction, (dr, dc) in enumerate(DELTAS):
                    neighbour = board.get((row + dr, col + dc))
                    if neighbour is None:
                        continue
                    delta += edge_value(
                        neighbour, tile, direction ^ 1, right, down
                    )
                    count += 1
                if not count:
                    continue
                delta += contact_bonus * max(0, count - 1)
                candidate = (delta, count, -row, -col, -tile)
                if best is None or candidate > best[0]:
                    best = (candidate, coordinate, tile)
        if best is None:
            return None
        (delta, count, *_), coordinate, tile = best
        board[coordinate] = tile
        unused.remove(tile)
        score += delta
        contacts += count
        board = canonical_board(board)
    return BeamState(board, frozenset(), score, contacts)


def solve_component_beam(
    right: np.ndarray,
    down: np.ndarray,
    components: list[dict[int, tuple[int, int]]],
    *,
    beam_width: int,
    candidate_topk: int,
    per_state: int,
    seed_components: int,
    contact_bonus: float,
    singleton_growth: bool = False,
    fixed_order: bool = False,
) -> tuple[np.ndarray | None, dict[str, float]]:
    components = complete_components(components)
    if singleton_growth:
        seed_id = min(
            range(len(components)), key=lambda index: (-len(components[index]), index)
        )
        seed = components[seed_id]
        seed_tiles = set(seed)
        components = [seed] + [
            {tile: (0, 0)} for tile in range(NFRAG) if tile not in seed_tiles
        ]
        seed_components = 1
    owner, local = component_index(components)
    top = top_targets(right, down, candidate_topk)
    seed_ids = sorted(
        range(len(components)), key=lambda index: (-len(components[index]), index)
    )[:seed_components]
    all_components = frozenset(range(len(components)))
    component_order = sorted(
        range(len(components)), key=lambda index: (-len(components[index]), index)
    )
    beam = []
    for component_id in seed_ids:
        board = canonical_board(
            {coordinate: tile for tile, coordinate in components[component_id].items()}
        )
        beam.append(
            BeamState(board, all_components - {component_id}, 0.0, 0)
        )

    expanded = 0
    while beam and beam[0].remaining:
        pool: list[BeamState] = []
        for state in beam:
            allowed_component = None
            if fixed_order:
                allowed_component = next(
                    component_id
                    for component_id in component_order
                    if component_id in state.remaining
                )
            proposals = state_proposals(
                state,
                components,
                owner,
                local,
                top,
                right,
                down,
                per_state=per_state,
                contact_bonus=contact_bonus,
                allowed_component=allowed_component,
            )
            expanded += len(proposals)
            pool.extend(proposals)
        if not pool:
            completed = [
                result
                for state in beam
                if (
                    result := finish_singletons(
                        state, components, right, down, contact_bonus
                    )
                )
                is not None
            ]
            if completed:
                beam = completed
            break
        pool.sort(
            key=lambda state: (
                state.score,
                state.contacts,
                len(state.board),
            ),
            reverse=True,
        )
        unique: list[BeamState] = []
        seen = set()
        for state in pool:
            key = signature(state)
            if key in seen:
                continue
            seen.add(key)
            unique.append(state)
            if len(unique) >= beam_width:
                break
        beam = unique

    complete = [state for state in beam if not state.remaining]
    if not complete:
        return None, {
            "complete": 0.0,
            "placed_tiles": float(max((len(state.board) for state in beam), default=0)),
            "expanded": float(expanded),
        }
    best = max(complete, key=lambda state: (state.score, state.contacts))
    board = canonical_board(best.board)
    if len(board) != NFRAG:
        raise AssertionError("complete component state did not place every tile")
    if max(row for row, _ in board) >= GRID or max(col for _, col in board) >= GRID:
        raise AssertionError("complete component state exceeds the 24x24 frame")
    placement = np.empty(NFRAG, dtype=np.int64)
    for (row, col), tile in board.items():
        placement[row * GRID + col] = tile
    return placement, {
        "complete": 1.0,
        "placed_tiles": float(NFRAG),
        "expanded": float(expanded),
        "score": float(best.score),
        "contacts": float(best.contacts),
        "components": float(len(components)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--images", default="50")
    parser.add_argument("--component-edges", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--candidate-topk", type=int, default=8)
    parser.add_argument("--per-state", type=int, default=24)
    parser.add_argument("--seed-components", type=int, default=8)
    parser.add_argument("--contact-bonus", type=float, default=0.05)
    parser.add_argument(
        "--singleton-growth",
        action="store_true",
        help="keep only the largest seed island rigid; grow all other tiles individually",
    )
    parser.add_argument(
        "--fixed-order",
        action="store_true",
        help="place components in one shared size-descending order across the beam",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "component_beam_gate.json",
    )
    args = parser.parse_args()
    rows = []
    for image in (int(value) for value in args.images.split(",")):
        path = args.cache_dir / f"image_{image:04d}_k64.npz"
        stored = np.load(path)
        candidates = torch.from_numpy(stored["candidate_ids"]).long()
        scores = (
            torch.from_numpy(stored["candidate_scores"])
            .float()
            .reshape(NFRAG, 4, -1)
            .permute(1, 0, 2)
        )
        right_t, down_t = dense_rd(candidates, scores)
        right, down = right_t.numpy(), down_t.numpy()
        components = build_buddies_components(
            right, down, max_edges=args.component_edges, min_margin=0.0
        )
        placement, diagnostics = solve_component_beam(
            right,
            down,
            components,
            beam_width=args.beam_width,
            candidate_topk=args.candidate_topk,
            per_state=args.per_state,
            seed_components=args.seed_components,
            contact_bonus=args.contact_bonus,
            singleton_growth=args.singleton_growth,
            fixed_order=args.fixed_order,
        )
        if placement is None:
            metrics = {"placement": 0.0, "neighbour": 0.0, **diagnostics}
        else:
            truth = np.argsort(stored["permutation"])
            exact, _ = placement_accuracy(placement, truth)
            neighbour, horizontal, vertical = neighbour_accuracy(placement, truth)
            metrics = {
                "placement": exact,
                "neighbour": neighbour,
                "right": horizontal,
                "down": vertical,
                **diagnostics,
            }
        rows.append(metrics)
        print(json.dumps({"image": image, **metrics}), flush=True)
    summary = {
        key: float(np.mean([row.get(key, 0.0) for row in rows]))
        for key in rows[0]
    }
    report = {
        "experiment": "translation_free_component_beam",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": summary,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
