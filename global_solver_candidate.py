"""Optimized global jigsaw solver for 24x24 grids."""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

GRID = 24
N = GRID * GRID
POSITION_WEIGHT = 0.11
STEPS = 400000
BEST_BUDDY_MARGIN = 0.5


def _hungarian_layout(pos):
    """Return the baseline position-only initializer."""
    tile_indices, position_indices = linear_sum_assignment(-pos)
    layout = np.empty(N, dtype=np.int32)
    layout[position_indices] = tile_indices
    return layout


def _second_best_margin(matrix, axis):
    """Top-1 minus top-2 margin along rows (1) or columns (0)."""
    top_two = np.partition(matrix, -2, axis=axis)
    if axis == 1:
        return top_two[:, -1] - top_two[:, -2]
    return top_two[-1, :] - top_two[-2, :]


def _best_buddy_component_layout(right, down, pos):
    """Place coordinate-consistent reciprocal high-margin components.

    Edges are admitted strongest-first.  A merge is rejected if its relative
    coordinates conflict, overlap another member, or cannot fit on the board.
    Components are then anchored by their summed position logits; remaining
    tiles retain the baseline Hungarian position assignment on free cells.
    """
    best_right = np.argmax(right, axis=1)
    best_left = np.argmax(right, axis=0)
    best_down = np.argmax(down, axis=1)
    best_up = np.argmax(down, axis=0)
    right_row_margin = _second_best_margin(right, 1)
    right_col_margin = _second_best_margin(right, 0)
    down_row_margin = _second_best_margin(down, 1)
    down_col_margin = _second_best_margin(down, 0)

    edges = []
    for tile in range(N):
        neighbour = int(best_right[tile])
        confidence = float(min(right_row_margin[tile], right_col_margin[neighbour]))
        if best_left[neighbour] == tile and confidence >= BEST_BUDDY_MARGIN:
            edges.append((confidence, tile, neighbour, 0, 1))
        neighbour = int(best_down[tile])
        confidence = float(min(down_row_margin[tile], down_col_margin[neighbour]))
        if best_up[neighbour] == tile and confidence >= BEST_BUDDY_MARGIN:
            edges.append((confidence, tile, neighbour, 1, 0))
    edges.sort(reverse=True)

    roots = np.arange(N, dtype=np.int32)
    components = {tile: {tile: (0, 0)} for tile in range(N)}
    component_confidence = {tile: 0.0 for tile in range(N)}
    for confidence, source, target, dr, dc in edges:
        source_root, target_root = int(roots[source]), int(roots[target])
        source_coord = components[source_root][source]
        target_coord = components[target_root][target]
        required = (source_coord[0] + dr, source_coord[1] + dc)
        if source_root == target_root:
            continue
        shift = (required[0] - target_coord[0], required[1] - target_coord[1])
        shifted = {
            tile: (coord[0] + shift[0], coord[1] + shift[1])
            for tile, coord in components[target_root].items()
        }
        merged_coords = list(components[source_root].values()) + list(shifted.values())
        rows = [coord[0] for coord in merged_coords]
        cols = [coord[1] for coord in merged_coords]
        if (len(set(merged_coords)) != len(merged_coords)
                or max(rows) - min(rows) >= GRID
                or max(cols) - min(cols) >= GRID):
            continue
        components[source_root].update(shifted)
        for tile in shifted:
            roots[tile] = source_root
        del components[target_root]
        component_confidence[source_root] += component_confidence.pop(target_root) + confidence

    # Strong/larger components claim positions first.  Singletons and any
    # component that cannot be placed are completed by Hungarian matching.
    ordered = sorted(
        (root for root, members in components.items() if len(members) >= 2),
        key=lambda root: (len(components[root]), component_confidence[root]),
        reverse=True,
    )
    layout = np.full(N, -1, dtype=np.int32)
    occupied = np.zeros(N, dtype=bool)
    placed_tiles = np.zeros(N, dtype=bool)
    for root in ordered:
        members = components[root]
        min_row = min(coord[0] for coord in members.values())
        min_col = min(coord[1] for coord in members.values())
        normalized = {
            tile: (coord[0] - min_row, coord[1] - min_col)
            for tile, coord in members.items()
        }
        height = max(coord[0] for coord in normalized.values()) + 1
        width = max(coord[1] for coord in normalized.values()) + 1
        best_score, best_positions = -np.inf, None
        for row in range(GRID - height + 1):
            for col in range(GRID - width + 1):
                positions = [
                    (row + coord[0]) * GRID + col + coord[1]
                    for coord in normalized.values()
                ]
                if occupied[positions].any():
                    continue
                score = sum(
                    float(pos[tile, position])
                    for tile, position in zip(normalized, positions)
                )
                if score > best_score:
                    best_score, best_positions = score, positions
        if best_positions is None:
            continue
        for tile, position in zip(normalized, best_positions):
            layout[position] = tile
            occupied[position] = True
            placed_tiles[tile] = True

    remaining_tiles = np.flatnonzero(~placed_tiles)
    remaining_positions = np.flatnonzero(~occupied)
    tile_indices, position_indices = linear_sum_assignment(
        -pos[np.ix_(remaining_tiles, remaining_positions)]
    )
    layout[remaining_positions[position_indices]] = remaining_tiles[tile_indices]
    return layout


def objective(layout, right, down, weighted_pos):
    """Calculate the global objective score for a layout."""
    score = 0.0
    for position in range(N):
        tile = layout[position]
        score += weighted_pos[tile, position]
        if position % GRID < GRID - 1:
            score += right[tile, layout[position + 1]]
        if position < N - GRID:
            score += down[tile, layout[position + GRID]]
    return float(score)


def solve_layout(right, down, pos, seed):
    """Return a permutation: tile index at every row-major board position."""
    rng = np.random.default_rng(seed)

    # E4 changes only the initializer; the SA below is intentionally unchanged.
    initializer = os.getenv("INITIALIZER_MODE", "hungarian")
    if initializer == "hungarian":
        layout = _hungarian_layout(pos)
    elif initializer == "best_buddy":
        layout = _best_buddy_component_layout(right, down, pos)
    else:
        raise ValueError(f"unknown INITIALIZER_MODE={initializer}")
    weighted_pos = POSITION_WEIGHT * pos

    # Precompute best neighbors for heuristic moves
    best_right = np.argmax(right, axis=1)
    best_down = np.argmax(down, axis=1)
    best_left = np.argmax(right, axis=0)
    best_up = np.argmax(down, axis=0)

    inverse_layout = np.empty(N, dtype=np.int32)
    inverse_layout[layout] = np.arange(N)
    
    current_score = objective(layout, right, down, weighted_pos)
    best_layout = layout.copy()
    best_score = current_score
    
    # SA parameters
    start_temp, end_temp = 1.0, 0.0001
    log_temp_ratio = np.log(end_temp / start_temp)

    # Constants
    GRID_VAL = GRID
    N_VAL = N
    RIGHT_LIMIT = GRID - 1
    DOWN_LIMIT = N - GRID

    for step in range(STEPS):
        # Move selection
        if rng.random() < 0.1:
            a, b = rng.integers(N_VAL), rng.integers(N_VAL)
        else:
            # Heuristic: try to fix a neighbor of a random anchor
            a = b = -1
            for _ in range(8):
                anchor = int(rng.integers(N_VAL))
                anchor_tile = layout[anchor]
                direction = int(rng.integers(4))
                
                if direction == 0: # Right
                    if anchor % GRID_VAL != RIGHT_LIMIT:
                        a, b = anchor + 1, inverse_layout[best_right[anchor_tile]]
                elif direction == 1: # Down
                    if anchor < DOWN_LIMIT:
                        a, b = anchor + GRID_VAL, inverse_layout[best_down[anchor_tile]]
                elif direction == 2: # Left
                    if anchor % GRID_VAL != 0:
                        a, b = anchor - 1, inverse_layout[best_left[anchor_tile]]
                else: # Up
                    if anchor >= GRID_VAL:
                        a, b = anchor - GRID_VAL, inverse_layout[best_up[anchor_tile]]
                
                if a != -1 and a != b:
                    break
            else:
                a, b = rng.integers(N_VAL), rng.integers(N_VAL)

        if a == b:
            continue

        # Determine affected positions for delta calculation
        affected = {a, b}
        if a % GRID_VAL > 0: affected.add(a - 1)
        if a >= GRID_VAL: affected.add(a - GRID_VAL)
        if b % GRID_VAL > 0: affected.add(b - 1)
        if b >= GRID_VAL: affected.add(b - GRID_VAL)

        # Delta calculation
        before = 0.0
        for p in affected:
            t = layout[p]
            before += weighted_pos[t, p]
            if p % GRID_VAL < RIGHT_LIMIT:
                before += right[t, layout[p + 1]]
            if p < DOWN_LIMIT:
                before += down[t, layout[p + GRID_VAL]]

        tile_a, tile_b = layout[a], layout[b]
        layout[a], layout[b] = tile_b, tile_a

        after = 0.0
        for p in affected:
            t = layout[p]
            after += weighted_pos[t, p]
            if p % GRID_VAL < RIGHT_LIMIT:
                after += right[t, layout[p + 1]]
            if p < DOWN_LIMIT:
                after += down[t, layout[p + GRID_VAL]]

        delta = after - before
        
        # SA Acceptance
        temperature = start_temp * np.exp(log_temp_ratio * (step / STEPS))
        
        if delta >= 0 or (temperature > 1e-7 and rng.random() < np.exp(delta / temperature)):
            current_score += delta
            inverse_layout[tile_a], inverse_layout[tile_b] = b, a
            if current_score > best_score:
                best_score = current_score
                best_layout = layout.copy()
        else:
            # Reject: swap back
            layout[a], layout[b] = tile_a, tile_b

    return best_layout
