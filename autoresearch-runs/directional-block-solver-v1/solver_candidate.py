"""Experimental block-preserving global jigsaw solver for 24x24 grids."""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

GRID = 24
N = GRID * GRID
POSITION_WEIGHT = 0.11
STEPS = 400000
BLOCK_STEPS = 60000


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

    mode = os.getenv("BLOCK_MODE", "baseline")
    if mode == "baseline":
        return best_layout

    # Start the large-neighborhood phase from the best single-tile state.
    layout = best_layout.copy()
    current_score = best_score

    def affected_origins(positions):
        affected = set()
        for p in positions:
            affected.add(p)
            if p % GRID_VAL > 0:
                affected.add(p - 1)
            if p >= GRID_VAL:
                affected.add(p - GRID_VAL)
        return affected

    def partial_score(origins):
        value = 0.0
        for p in origins:
            t = layout[p]
            value += weighted_pos[t, p]
            if p % GRID_VAL < RIGHT_LIMIT:
                value += right[t, layout[p + 1]]
            if p < DOWN_LIMIT:
                value += down[t, layout[p + GRID_VAL]]
        return value

    if mode in ("two_side", "two_side_block2"):
        inverse_layout = np.empty(N_VAL, dtype=np.int32)
        inverse_layout[layout] = np.arange(N_VAL)

        def incident_support(p):
            tile = layout[p]
            value = weighted_pos[tile, p]
            if p % GRID_VAL > 0:
                value += right[layout[p - 1], tile]
            if p % GRID_VAL < RIGHT_LIMIT:
                value += right[tile, layout[p + 1]]
            if p >= GRID_VAL:
                value += down[layout[p - GRID_VAL], tile]
            if p < DOWN_LIMIT:
                value += down[tile, layout[p + GRID_VAL]]
            return value

        def exact_swap_delta(a, b):
            origins = affected_origins([a, b])
            before = partial_score(origins)
            tile_a, tile_b = layout[a], layout[b]
            layout[a], layout[b] = tile_b, tile_a
            delta = partial_score(origins) - before
            layout[a], layout[b] = tile_a, tile_b
            return delta

        for _ in range(12000):
            sampled = rng.integers(N_VAL, size=24)
            position = min((int(p) for p in sampled), key=incident_support)
            fit = weighted_pos[:, position].copy()
            if position % GRID_VAL > 0:
                fit += right[layout[position - 1], :]
            if position % GRID_VAL < RIGHT_LIMIT:
                fit += right[:, layout[position + 1]]
            if position >= GRID_VAL:
                fit += down[layout[position - GRID_VAL], :]
            if position < DOWN_LIMIT:
                fit += down[:, layout[position + GRID_VAL]]
            top_tiles = np.argpartition(fit, -16)[-16:]
            best_delta = 0.0
            best_other = -1
            for tile in top_tiles:
                other = int(inverse_layout[int(tile)])
                if other == position:
                    continue
                delta = exact_swap_delta(position, other)
                if delta > best_delta:
                    best_delta = delta
                    best_other = other
            if best_other >= 0:
                tile_a, tile_b = layout[position], layout[best_other]
                layout[position], layout[best_other] = tile_b, tile_a
                inverse_layout[tile_a], inverse_layout[tile_b] = best_other, position
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_layout = layout.copy()

        if mode == "two_side":
            return best_layout
        # Hybrid: polish the two-side result with the existing conservative 2x2 phase.
        layout = best_layout.copy()
        current_score = best_score
        mode = "block2"

    if mode == "guided_block2":
        block_steps = 3500
        candidates_per_step = 24

        def block_at(row, col):
            return [row * GRID_VAL + col, row * GRID_VAL + col + 1,
                    (row + 1) * GRID_VAL + col, (row + 1) * GRID_VAL + col + 1]

        def try_swap(group_a, group_b):
            origins = affected_origins(group_a + group_b)
            before = partial_score(origins)
            old_a = layout[group_a].copy()
            old_b = layout[group_b].copy()
            layout[group_a] = old_b
            layout[group_b] = old_a
            delta = partial_score(origins) - before
            layout[group_a] = old_a
            layout[group_b] = old_b
            return delta

        for _ in range(block_steps):
            # Pick the weakest of several sampled blocks by its local objective.
            weak_group = None
            weak_value = np.inf
            for _ in range(16):
                row = int(rng.integers(GRID_VAL - 1))
                col = int(rng.integers(GRID_VAL - 1))
                group = block_at(row, col)
                value = partial_score(affected_origins(group))
                if value < weak_value:
                    weak_value = value
                    weak_group = group

            best_delta = 0.0
            best_group = None
            weak_set = set(weak_group)
            for _ in range(candidates_per_step):
                row = int(rng.integers(GRID_VAL - 1))
                col = int(rng.integers(GRID_VAL - 1))
                candidate = block_at(row, col)
                if weak_set & set(candidate):
                    continue
                delta = try_swap(weak_group, candidate)
                if delta > best_delta:
                    best_delta = delta
                    best_group = candidate

            if best_group is not None:
                old_a = layout[weak_group].copy()
                old_b = layout[best_group].copy()
                layout[weak_group] = old_b
                layout[best_group] = old_a
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_layout = layout.copy()
        return best_layout

    for _ in range(BLOCK_STEPS):
        proposal = mode
        if mode == "mixed":
            proposal = "block2" if rng.random() < 0.55 else "segment4"

        if proposal == "block2":
            ar, ac = int(rng.integers(GRID_VAL - 1)), int(rng.integers(GRID_VAL - 1))
            br, bc = int(rng.integers(GRID_VAL - 1)), int(rng.integers(GRID_VAL - 1))
            if abs(ar - br) < 2 and abs(ac - bc) < 2:
                continue
            group_a = [ar * GRID_VAL + ac, ar * GRID_VAL + ac + 1,
                       (ar + 1) * GRID_VAL + ac, (ar + 1) * GRID_VAL + ac + 1]
            group_b = [br * GRID_VAL + bc, br * GRID_VAL + bc + 1,
                       (br + 1) * GRID_VAL + bc, (br + 1) * GRID_VAL + bc + 1]
        elif proposal == "segment4":
            horizontal = rng.random() < 0.5
            if horizontal:
                ar, br = int(rng.integers(GRID_VAL)), int(rng.integers(GRID_VAL))
                ac, bc = int(rng.integers(GRID_VAL - 3)), int(rng.integers(GRID_VAL - 3))
                group_a = [ar * GRID_VAL + ac + k for k in range(4)]
                group_b = [br * GRID_VAL + bc + k for k in range(4)]
            else:
                ar, br = int(rng.integers(GRID_VAL - 3)), int(rng.integers(GRID_VAL - 3))
                ac, bc = int(rng.integers(GRID_VAL)), int(rng.integers(GRID_VAL))
                group_a = [(ar + k) * GRID_VAL + ac for k in range(4)]
                group_b = [(br + k) * GRID_VAL + bc for k in range(4)]
            if set(group_a) & set(group_b):
                continue
        else:
            raise ValueError(f"unknown BLOCK_MODE={mode}")

        changed = group_a + group_b
        origins = affected_origins(changed)
        before = partial_score(origins)
        old_a = layout[group_a].copy()
        old_b = layout[group_b].copy()
        layout[group_a] = old_b
        layout[group_b] = old_a
        delta = partial_score(origins) - before
        if delta >= 0:
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_layout = layout.copy()
        else:
            layout[group_a] = old_a
            layout[group_b] = old_b

    return best_layout
