"""Complete 24x24 puzzle decoder for dense right/down compatibility matrices.

The decoder deliberately separates the learned pair score from the discrete
layout problem.  A coordinate-consistent component supplies several anchors;
each anchor is completed greedily, synchronously refined with exact Hungarian
assignments, and polished with pair swaps.  The best complete permutation is
returned.  Ground truth is never used by the solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class SolveResult:
    board: np.ndarray  # board[cell] -> tile
    objective: float
    anchor_size: int
    candidates: int


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Make logits from different directions comparable without flattening ranks."""
    x = np.asarray(matrix, dtype=np.float64).copy()
    finite = np.isfinite(x)
    replacement = float(np.min(x[finite])) - 20.0 if np.any(finite) else -20.0
    x[~finite] = replacement
    median = np.median(x, axis=1, keepdims=True)
    centered = x - median
    # One global scale preserves the model's confidence margins.  Row-wise MAD
    # made an ambiguous tile as influential as a distinctive one.
    scale = 1.4826 * np.median(np.abs(centered)) + 1e-4
    x = np.clip(centered / scale, -12.0, 12.0)
    np.fill_diagonal(x, -30.0)
    return x.astype(np.float32)


def board_objective(board: np.ndarray, right: np.ndarray, down: np.ndarray, side: int) -> float:
    grid = np.asarray(board).reshape(side, side)
    return float(
        right[grid[:, :-1], grid[:, 1:]].sum(dtype=np.float64)
        + down[grid[:-1], grid[1:]].sum(dtype=np.float64)
    )


def _anchor_shifts(coords: dict[int, tuple[int, int]], side: int) -> list[tuple[int, int]]:
    if not coords:
        return [(0, 0)]
    rows = np.asarray([p[0] for p in coords.values()])
    cols = np.asarray([p[1] for p in coords.values()])
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    row_room, col_room = side - height, side - width
    row_choices = sorted(set((0, row_room // 2, row_room)))
    col_choices = sorted(set((0, col_room // 2, col_room)))
    return [(r - int(rows.min()), c - int(cols.min()))
            for r in row_choices for c in col_choices]


def _place_anchor(coords: dict[int, tuple[int, int]], shift: tuple[int, int], side: int) -> np.ndarray:
    board = np.full(side * side, -1, np.int32)
    dr, dc = shift
    for tile, (row, col) in coords.items():
        row, col = row + dr, col + dc
        if 0 <= row < side and 0 <= col < side:
            cell = row * side + col
            if board[cell] < 0:
                board[cell] = int(tile)
    return board


def _extract_components(right: np.ndarray, down: np.ndarray, tiles: np.ndarray,
                        side: int, topk: int = 5) -> list[dict[int, tuple[int, int]]]:
    """Coordinate-consistent Kruskal forest for tiles outside the main anchor."""
    ids = np.asarray(tiles, dtype=np.int32)
    if not len(ids):
        return []
    components = {int(tile): {int(tile): (0, 0)} for tile in ids}
    owner = {int(tile): int(tile) for tile in ids}
    candidates = []
    for direction, full in enumerate((right, down)):
        matrix = full[np.ix_(ids, ids)]
        k = min(topk, max(1, len(ids) - 1))
        order = np.argsort(-matrix, axis=1)[:, :k]
        reverse_order = np.argsort(-matrix, axis=0)
        reverse_rank = np.empty_like(reverse_order)
        reverse_rank[reverse_order, np.arange(len(ids))[None, :]] = np.arange(len(ids))[:, None]
        for local_i, local_js in enumerate(order):
            for local_j in local_js:
                i, j = int(ids[local_i]), int(ids[local_j])
                weight = float(matrix[local_i, local_j]) + 1.0 / (1.0 + reverse_rank[local_i, local_j])
                candidates.append((weight, i, j, direction))
    candidates.sort(reverse=True)
    for _weight, i, j, direction in candidates:
        ci, cj = owner[i], owner[j]
        if ci == cj:
            continue
        dr, dc = ((0, 1), (1, 0))[direction]
        left, other = components[ci], components[cj]
        ri, co_i = left[i]
        rj, co_j = other[j]
        shift = (ri + dr - rj, co_i + dc - co_j)
        shifted = {tile: (r + shift[0], c + shift[1]) for tile, (r, c) in other.items()}
        if set(left.values()) & set(shifted.values()):
            continue
        merged = {**left, **shifted}
        rows, cols = zip(*merged.values())
        if max(rows) - min(rows) + 1 > side or max(cols) - min(cols) + 1 > side:
            continue
        components[ci] = merged
        del components[cj]
        for tile in shifted:
            owner[tile] = ci
    return sorted(components.values(), key=len, reverse=True)


def _placement_gain(board: np.ndarray, cells_to_tiles: dict[int, int],
                    right: np.ndarray, down: np.ndarray, side: int) -> tuple[float, int]:
    gain, contacts = 0.0, 0
    for cell, tile in cells_to_tiles.items():
        row, col = divmod(cell, side)
        for neighbour, direction, forward in (
            (cell - 1, 0, False), (cell + 1, 0, True),
            (cell - side, 1, False), (cell + side, 1, True),
        ):
            if neighbour < 0 or neighbour >= len(board):
                continue
            nr, nc = divmod(neighbour, side)
            if abs(nr - row) + abs(nc - col) != 1 or board[neighbour] < 0:
                continue
            other = int(board[neighbour])
            matrix = right if direction == 0 else down
            gain += float(matrix[tile, other] if forward else matrix[other, tile])
            contacts += 1
    return gain, contacts


def _pack_components(anchor_coords: dict[int, tuple[int, int]], components: list[dict[int, tuple[int, int]]],
                     right: np.ndarray, down: np.ndarray, side: int,
                     beam_width: int) -> list[np.ndarray]:
    """Small beam over rigid component translations; unmatched pieces remain free."""
    states: list[tuple[float, np.ndarray]] = []
    for shift in _anchor_shifts(anchor_coords, side):
        states.append((0.0, _place_anchor(anchor_coords, shift, side)))
    states = states[:max(1, beam_width * 2)]
    for component in components:
        if len(component) <= 1:
            continue
        rows = np.asarray([coord[0] for coord in component.values()])
        cols = np.asarray([coord[1] for coord in component.values()])
        min_r, min_c = int(rows.min()), int(cols.min())
        height, width = int(rows.max() - min_r + 1), int(cols.max() - min_c + 1)
        proposals: list[tuple[float, np.ndarray]] = []
        for state_score, board in states:
            local = []
            for base_r in range(side - height + 1):
                for base_c in range(side - width + 1):
                    mapping = {
                        (base_r + r - min_r) * side + (base_c + c - min_c): int(tile)
                        for tile, (r, c) in component.items()
                    }
                    cells = np.fromiter(mapping.keys(), dtype=np.int32)
                    if np.any(board[cells] >= 0):
                        continue
                    gain, contacts = _placement_gain(board, mapping, right, down, side)
                    if contacts:
                        local.append((gain + 0.20 * contacts, mapping))
            # Keep several alternatives per state so an early translation can be revisited.
            local.sort(key=lambda item: item[0], reverse=True)
            for gain, mapping in local[:max(2, beam_width)]:
                candidate = board.copy()
                for cell, tile in mapping.items():
                    candidate[cell] = tile
                proposals.append((state_score + gain, candidate))
            if not local:
                proposals.append((state_score, board))
        proposals.sort(key=lambda item: (item[0], np.count_nonzero(item[1] >= 0)), reverse=True)
        states = proposals[:max(1, beam_width)]
    return [board for _, board in states]


def _cell_scores(board: np.ndarray, tiles: np.ndarray, cells: np.ndarray,
                 right: np.ndarray, down: np.ndarray, side: int) -> np.ndarray:
    """Compatibility of every candidate tile with currently occupied neighbours."""
    score = np.zeros((len(tiles), len(cells)), np.float32)
    for column, cell in enumerate(cells):
        row, col = divmod(int(cell), side)
        if col and board[cell - 1] >= 0:
            score[:, column] += right[int(board[cell - 1]), tiles]
        if col + 1 < side and board[cell + 1] >= 0:
            score[:, column] += right[tiles, int(board[cell + 1])]
        if row and board[cell - side] >= 0:
            score[:, column] += down[int(board[cell - side]), tiles]
        if row + 1 < side and board[cell + side] >= 0:
            score[:, column] += down[tiles, int(board[cell + side])]
    return score


def _frontier_complete(board: np.ndarray, right: np.ndarray, down: np.ndarray,
                       side: int, rng: np.random.Generator) -> np.ndarray:
    result = board.copy()
    n = side * side
    used = np.zeros(n, bool)
    used[result[result >= 0]] = True
    remaining = np.flatnonzero(~used).astype(np.int32)
    while len(remaining):
        free = np.flatnonzero(result < 0).astype(np.int32)
        contact = np.zeros(len(free), np.int8)
        for k, cell in enumerate(free):
            row, col = divmod(int(cell), side)
            contact[k] = sum((
                col > 0 and result[cell - 1] >= 0,
                col + 1 < side and result[cell + 1] >= 0,
                row > 0 and result[cell - side] >= 0,
                row + 1 < side and result[cell + side] >= 0,
            ))
        best_contact = int(contact.max(initial=0))
        frontier = free[contact == best_contact]
        if best_contact == 0:
            cell = int(frontier[rng.integers(len(frontier))])
            tile_index = int(rng.integers(len(remaining)))
        else:
            values = _cell_scores(result, remaining, frontier, right, down, side)
            tile_index, cell_index = np.unravel_index(int(np.argmax(values)), values.shape)
            cell = int(frontier[cell_index])
        result[cell] = int(remaining[tile_index])
        remaining = np.delete(remaining, tile_index)
    return result


def _hungarian_refine(board: np.ndarray, right: np.ndarray, down: np.ndarray,
                      side: int, rounds: int,
                      movable_cells: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    current = board.copy()
    best = current.copy()
    best_score = board_objective(best, right, down, side)
    cells = (np.arange(side * side, dtype=np.int32) if movable_cells is None
             else np.asarray(movable_cells, dtype=np.int32))
    seen = {current.tobytes()}
    for _ in range(rounds):
        tiles = current[cells].copy()
        unary = _cell_scores(current, tiles, cells, right, down, side)
        # A small inertia term prevents unstable two-cycles while still allowing
        # low-confidence pieces to move under multi-neighbour evidence.
        lookup = {int(tile): row for row, tile in enumerate(tiles)}
        unary[[lookup[int(current[cell])] for cell in cells], np.arange(len(cells))] += 0.15
        tile_rows, assigned_cells = linear_sum_assignment(-unary.astype(np.float64))
        candidate = current.copy()
        candidate[cells[assigned_cells]] = tiles[tile_rows]
        key = candidate.tobytes()
        score = board_objective(candidate, right, down, side)
        if score > best_score:
            best, best_score = candidate.copy(), score
        if key in seen:
            break
        seen.add(key)
        current = candidate
    return best, best_score


def _incident_edges(cell: int, side: int) -> set[tuple[int, int, int]]:
    row, col = divmod(cell, side)
    edges: set[tuple[int, int, int]] = set()
    if col:
        edges.add((cell - 1, cell, 0))
    if col + 1 < side:
        edges.add((cell, cell + 1, 0))
    if row:
        edges.add((cell - side, cell, 1))
    if row + 1 < side:
        edges.add((cell, cell + side, 1))
    return edges


def _edge_value(board: np.ndarray, edge: tuple[int, int, int],
                right: np.ndarray, down: np.ndarray) -> float:
    a, b, direction = edge
    return float((right if direction == 0 else down)[board[a], board[b]])


def _swap_polish(board: np.ndarray, right: np.ndarray, down: np.ndarray,
                 side: int, proposals: int, rng: np.random.Generator,
                 movable_cells: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    current = board.copy()
    score = board_objective(current, right, down, side)
    choices = (np.arange(len(current), dtype=np.int32) if movable_cells is None
               else np.asarray(movable_cells, dtype=np.int32))
    if len(choices) < 2:
        return current, score
    for _ in range(proposals):
        a, b = rng.choice(choices, 2, replace=False)
        affected = _incident_edges(int(a), side) | _incident_edges(int(b), side)
        before = sum(_edge_value(current, edge, right, down) for edge in affected)
        current[a], current[b] = current[b], current[a]
        after = sum(_edge_value(current, edge, right, down) for edge in affected)
        delta = after - before
        if delta > 1e-7:
            score += delta
        else:
            current[a], current[b] = current[b], current[a]
    return current, float(score)


def solve_complete(right: np.ndarray, down: np.ndarray, side: int,
                   anchor_coords: dict[int, tuple[int, int]], *, seed: int = 1337,
                   beam_width: int = 4, hungarian_rounds: int = 5,
                   swap_proposals: int = 12000) -> SolveResult:
    """Return a complete, duplicate-free board using no target information."""
    n = side * side
    if right.shape != (n, n) or down.shape != (n, n):
        raise ValueError(f"expected two {(n, n)} matrices")
    rscore, dscore = _normalise(right), _normalise(down)
    rng = np.random.default_rng(seed)
    # Validation showed that greedily packing secondary components slightly
    # reduced true adjacency (4.13% vs 4.15%).  Keep only the strongest anchor
    # rigid and let global assignment handle every remaining tile.
    packed = [_place_anchor(anchor_coords, shift, side)
              for shift in _anchor_shifts(anchor_coords, side)]
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for initial in packed:
        complete = _frontier_complete(initial, rscore, dscore, side, rng)
        fixed = np.flatnonzero(initial >= 0).astype(np.int32)
        movable = np.flatnonzero(initial < 0).astype(np.int32)
        candidates.append((board_objective(complete, rscore, dscore, side), complete, fixed, movable))
    candidates.sort(key=lambda item: item[0], reverse=True)
    beam = candidates[:max(1, beam_width)]
    best_board, best_score = beam[0][1], beam[0][0]
    for _, candidate, _fixed, movable in beam:
        refined, refined_score = _hungarian_refine(
            candidate, rscore, dscore, side, hungarian_rounds, movable)
        polished, polished_score = _swap_polish(
            refined, rscore, dscore, side, swap_proposals, rng, movable)
        if polished_score > best_score:
            best_board, best_score = polished, polished_score
    if len(best_board) != n or not np.array_equal(np.sort(best_board), np.arange(n)):
        raise RuntimeError("global solver violated the full-permutation contract")
    return SolveResult(best_board, best_score, len(anchor_coords), len(candidates))


def placement_metrics(board: np.ndarray, true_positions: np.ndarray, side: int) -> dict[str, float]:
    """Direct and translation-aligned diagnostics plus true adjacency accuracy."""
    predicted = np.empty_like(true_positions)
    predicted[board] = np.arange(len(board))
    direct = float(np.mean(predicted == true_positions))
    shifts: dict[tuple[int, int], int] = {}
    for tile, cell in enumerate(predicted):
        pr, pc = divmod(int(cell), side)
        tr, tc = divmod(int(true_positions[tile]), side)
        shifts[(tr - pr, tc - pc)] = shifts.get((tr - pr, tc - pc), 0) + 1
    aligned = max(shifts.values()) / len(board)
    grid = board.reshape(side, side)
    left_pos = true_positions[grid[:, :-1]]
    right_pos = true_positions[grid[:, 1:]]
    top_pos = true_positions[grid[:-1, :]]
    bottom_pos = true_positions[grid[1:, :]]
    right_acc = np.mean((right_pos - left_pos == 1) & (right_pos // side == left_pos // side))
    down_acc = np.mean(bottom_pos - top_pos == side)
    return {
        "coverage": 1.0,
        "direct_placement": direct,
        "translation_aligned_placement": float(aligned),
        "adjacency": float(0.5 * (right_acc + down_acc)),
    }
