"""Direct minimisation of the global seam objective over permutations.

The solvers tried before this one -- LP synchronisation, loop-verified growth,
belief propagation -- all work by *selecting* edges, so each edge they accept
has to be right on its own.  On real boards mutual top-1 edges are 21% correct,
which is nowhere near enough, and that is why every one of them collapsed to
chance.

Minimising the summed cost over all 1104 grid adjacencies asks nothing of any
single edge.  A layout is scored by 1104 noisy terms at once, so per-edge errors
average out instead of accumulating.  The landscape supports it: the true layout
costs 3.3x less than a random one, one swap away is already 1.4% worse, and only
4.3% of single swaps improve on the truth -- those being interchangeable flat
tiles, which cost almost nothing in SSIM anyway.

Swap deltas touch at most seven edges, so a move is O(1) and tens of millions of
moves are affordable.

Use it as a REFINER, not a constructor.  From a random start it cannot solve
even a clean board -- 13x the true cost after 16M moves with candidate-biased
proposals, where greedy construction reaches place_acc 0.9965 -- because a
576-element permutation has barriers no pairwise move can cross.  Pass a
constructed layout as `init` and it polishes; without one it wanders.

Its diagnostic use stands on its own: pointed at real boards from random it
reaches 0.858x the TRUE cost, so a layout cheaper than the truth exists and is
easy to find.  That is a property of the cost matrix, not of the search, and it
is why every edge-selection solver in this repo collapsed.
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _touch(lay, p, skip, CH, CV, G, N):
    """Cost of edges incident to position p, ignoring the one leading to `skip`."""
    s = 0.0
    if p % G != G - 1 and p + 1 != skip:
        s += CH[lay[p], lay[p + 1]]
    if p % G != 0 and p - 1 != skip:
        s += CH[lay[p - 1], lay[p]]
    if p < N - G and p + G != skip:
        s += CV[lay[p], lay[p + G]]
    if p >= G and p - G != skip:
        s += CV[lay[p - G], lay[p]]
    return s


@njit(cache=True, fastmath=True)
def total_cost(lay, CH, CV, G, N):
    s = 0.0
    for p in range(N):
        if p % G != G - 1:
            s += CH[lay[p], lay[p + 1]]
        if p < N - G:
            s += CV[lay[p], lay[p + G]]
    return s


@njit(cache=True, fastmath=True)
def _anneal(lay, CH, CV, G, N, iters, t0, t1, seed, cand, pcand):
    """cand[t, k] = the k-th cheapest right/below partner of tile t.

    Uniformly random swaps are nearly all absurd at 576 tiles -- the pair has no
    reason to belong anywhere near each other -- so the chain spends its budget
    rejecting.  Half the proposals are instead drawn from the cost matrix's own
    shortlist: pick a position, take a tile the scores say could follow its left
    neighbour, and swap that tile in.  These are the moves that can actually
    close a seam, and it is the difference between never solving a clean board
    and solving it outright.
    """
    np.random.seed(seed)
    ratio = (t1 / t0) ** (1.0 / iters)
    T = t0
    cur = total_cost(lay, CH, CV, G, N)
    best = cur
    best_lay = lay.copy()
    K = cand.shape[1]
    for _ in range(iters):
        p = np.random.randint(0, N)
        if np.random.random() < pcand and p % G != 0:
            # a tile the scores like as a follower of whatever sits to the left
            want = cand[lay[p - 1], np.random.randint(0, K)]
            q = -1
            for i in range(N):
                if lay[i] == want:
                    q = i
                    break
            if q < 0:
                q = np.random.randint(0, N)
        else:
            q = np.random.randint(0, N)
        if p != q:
            # `skip=p` on the second call so an edge between p and q is not double counted
            before = _touch(lay, p, -1, CH, CV, G, N) + _touch(lay, q, p, CH, CV, G, N)
            t = lay[p]; lay[p] = lay[q]; lay[q] = t
            after = _touch(lay, p, -1, CH, CV, G, N) + _touch(lay, q, p, CH, CV, G, N)
            d = after - before
            if d <= 0.0 or np.random.random() < np.exp(-d / T):
                cur += d
                if cur < best:
                    best = cur
                    for i in range(N):
                        best_lay[i] = lay[i]
            else:
                t = lay[p]; lay[p] = lay[q]; lay[q] = t
        T *= ratio
    for i in range(N):
        lay[i] = best_lay[i]
    return best


@njit(cache=True, fastmath=True)
def _polish(lay, CH, CV, G, N, sweeps):
    """Exhaustive 2-opt: keep swapping any improving pair until none is left."""
    cur = total_cost(lay, CH, CV, G, N)
    for _ in range(sweeps):
        moved = 0
        for p in range(N):
            for q in range(p + 1, N):
                before = _touch(lay, p, -1, CH, CV, G, N) + _touch(lay, q, p, CH, CV, G, N)
                t = lay[p]; lay[p] = lay[q]; lay[q] = t
                after = _touch(lay, p, -1, CH, CV, G, N) + _touch(lay, q, p, CH, CV, G, N)
                if after < before - 1e-9:
                    cur += after - before
                    moved += 1
                else:
                    t = lay[p]; lay[p] = lay[q]; lay[q] = t
        if moved == 0:
            break
    return cur


def solve_anneal(cost_h, cost_v, grid=24, iters=8_000_000, restarts=4,
                 t_hi=0.35, t_lo=0.002, sweeps=8, seed=0, k_cand=8, p_cand=0.5,
                 init=None):
    """Return lay with lay[position] = tile index.

    Temperatures are given as fractions of the typical swap penalty, measured
    from the matrices themselves, so the schedule transfers across boards and
    across cost scales (MGC and ridge differ by orders of magnitude).
    """
    N = grid * grid
    CH = np.ascontiguousarray(cost_h, dtype=np.float64)
    CV = np.ascontiguousarray(cost_v, dtype=np.float64)
    scale = float(np.median(CH) + np.median(CV))
    # shortlist of plausible right-hand partners, used to propose sane swaps
    cand = np.argsort(CH, axis=1)[:, :k_cand].astype(np.int64).copy()
    best_lay, best_cost = None, np.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        if init is None:
            lay = rng.permutation(N).astype(np.int64)
        else:
            # refining: start cold, or the schedule undoes the construction
            lay = np.ascontiguousarray(init, dtype=np.int64).copy()
        hi = (t_hi if init is None else t_hi * 0.15) * scale
        _anneal(lay, CH, CV, grid, N, int(iters), hi, t_lo * scale,
                seed + r, cand, p_cand)
        c = _polish(lay, CH, CV, grid, N, sweeps)
        if c < best_cost:
            best_cost, best_lay = c, lay.copy()
    return best_lay, best_cost
