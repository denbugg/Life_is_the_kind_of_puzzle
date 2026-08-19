# cython: boundscheck=False, wraparound=False, initializedcheck=False
"""Compiled form of the unchanged simulated-annealing hot loop."""
import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t float32_t
ctypedef cnp.int32_t int32_t
ctypedef cnp.int64_t int64_t


cpdef tuple run_sa(
    cnp.ndarray[float32_t, ndim=2] right,
    cnp.ndarray[float32_t, ndim=2] down,
    cnp.ndarray[float32_t, ndim=2] weighted_pos,
    cnp.ndarray[int32_t, ndim=1] layout,
    cnp.ndarray[int32_t, ndim=1] inverse_layout,
    cnp.ndarray[int64_t, ndim=1] best_right,
    cnp.ndarray[int64_t, ndim=1] best_down,
    cnp.ndarray[int64_t, ndim=1] best_left,
    cnp.ndarray[int64_t, ndim=1] best_up,
    object rng,
    double current_score,
    double log_temp_ratio,
    int steps,
):
    """Run the baseline loop with typed indexing and the original RNG object."""
    cdef:
        int step, a, b, anchor, anchor_tile, direction, p, tile_a, tile_b, t
        int grid = 24
        int n = 576
        int right_limit = 23
        int down_limit = 552
        int attempt
        double before, after, delta, temperature
        double start_temp = 1.0
        double best_score = current_score
        object affected, p_object
        cnp.ndarray[int32_t, ndim=1] best_layout = layout.copy()

    for step in range(steps):
        if rng.random() < 0.1:
            a = rng.integers(n)
            b = rng.integers(n)
        else:
            a = -1
            b = -1
            for attempt in range(8):
                anchor = rng.integers(n)
                anchor_tile = layout[anchor]
                direction = rng.integers(4)
                if direction == 0:
                    if anchor % grid != right_limit:
                        a = anchor + 1
                        b = inverse_layout[best_right[anchor_tile]]
                elif direction == 1:
                    if anchor < down_limit:
                        a = anchor + grid
                        b = inverse_layout[best_down[anchor_tile]]
                elif direction == 2:
                    if anchor % grid != 0:
                        a = anchor - 1
                        b = inverse_layout[best_left[anchor_tile]]
                else:
                    if anchor >= grid:
                        a = anchor - grid
                        b = inverse_layout[best_up[anchor_tile]]
                if a != -1 and a != b:
                    break
            else:
                a = rng.integers(n)
                b = rng.integers(n)

        if a == b:
            continue

        # Keep a Python set so affected-origin iteration order matches baseline.
        affected = {a, b}
        if a % grid > 0:
            affected.add(a - 1)
        if a >= grid:
            affected.add(a - grid)
        if b % grid > 0:
            affected.add(b - 1)
        if b >= grid:
            affected.add(b - grid)

        before = 0.0
        for p_object in affected:
            p = p_object
            t = layout[p]
            before += weighted_pos[t, p]
            if p % grid < right_limit:
                before += right[t, layout[p + 1]]
            if p < down_limit:
                before += down[t, layout[p + grid]]

        tile_a = layout[a]
        tile_b = layout[b]
        layout[a] = tile_b
        layout[b] = tile_a

        after = 0.0
        for p_object in affected:
            p = p_object
            t = layout[p]
            after += weighted_pos[t, p]
            if p % grid < right_limit:
                after += right[t, layout[p + 1]]
            if p < down_limit:
                after += down[t, layout[p + grid]]

        delta = after - before
        # NumPy exp is intentional: preserve the baseline acceptance math.
        temperature = start_temp * np.exp(log_temp_ratio * (step / <double>steps))
        if delta >= 0 or (temperature > 1e-7 and rng.random() < np.exp(delta / temperature)):
            current_score += delta
            inverse_layout[tile_a] = b
            inverse_layout[tile_b] = a
            if current_score > best_score:
                best_score = current_score
                best_layout = layout.copy()
        else:
            layout[a] = tile_a
            layout[b] = tile_b

    return best_layout, best_score
