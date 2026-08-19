"""Optimized global jigsaw solver for 24x24 grids."""
import numpy as np
from scipy.optimize import linear_sum_assignment

GRID = 24
N = GRID * GRID
POSITION_WEIGHT = 0.11
STEPS = 400000


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

    # Initial layout using global position scores
    tile_indices, position_indices = linear_sum_assignment(-pos)
    layout = np.empty(N, dtype=np.int32)
    layout[position_indices] = tile_indices
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
