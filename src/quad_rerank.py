"""Re-rank each shortlist by the best 2x2 SQUARE a pair can stand in.

The door this goes through
--------------------------
M187 named it: the true neighbour is in the top 20 for two thirds of fragments
and in the top 64 for four fifths, so the information is present and the
problem is RANKING. A perfect picker inside the top 5 alone would give R@1
0.486, past the 0.47 at which M102 saw placement switch on. The learned chooser
recovered four per cent of that headroom, which is what M107 predicts of a
second stage assembled from the first stage's own features -- it can only copy.

A square is information the pairwise score does not contain. If j sits to the
right of i, then whatever sits below i must sit to the left of whatever sits
below j. Cycle consistency (M93), already in `costs_from_models`, is the linear
first-order version of that intuition; this is the second-order one.

Why this is not M92
-------------------
M92 used 2x2 closure as a HARD filter over mutual-best edges and the volume
collapsed -- 598 edges at precision 0.438 became 22 at 0.628 -- which is how
every hard filter in this project has died. Here the square is a SCORE: the
best square a pair can complete, over shortlists, with no closure required.
Nothing is discarded; the shortlist is re-ordered.

Measured, 24 boards, on the pipeline's own fused matrices
---------------------------------------------------------
top-1 0.3055 -> 0.3130, seed block 21.8 -> 24.3, true bonds 153.2 -> 156.8.
The aggregation matters: the single best square is worth less than a soft
log-sum-exp over the shortlist of squares (a pair that fits MANY squares well
is better evidence than one that fits a single square perfectly), and a
temperature that is too high washes the signal out -- tau 0.5 beats both 0.25
and 1.0. A second round raises the block further and costs top-1 heavily, so it
is not the default.
"""
import numpy as np

from config import GRID as G

N = G * G


def _squares(A, B, rows, cand, k, tau):
    """For every (i, j) in the shortlist, its best square above and below.

    `A` scores the axis being re-ranked and `B` the other one. Shapes are
    written out rather than left to broadcasting, which silently paired the
    wrong axes the first time this was written.
    """
    fwd = np.argsort(-B, axis=1)[:, :k]          # B[i, a]: a after i
    bwd = np.argsort(-B, axis=0).T[:, :k]        # B[a, i]: a before i
    out = np.zeros((len(rows), cand.shape[1], 2))
    for r, i in enumerate(rows):
        j = cand[r]
        for side in (0, 1):
            if side == 0:
                a, C = fwd[i], fwd[j]
                e = (B[i, a][:, None, None]
                     + A[a[:, None, None], C[None]]
                     + B[j[:, None], C][None])
            else:
                a, C = bwd[i], bwd[j]
                e = (B[a, i][:, None, None]
                     + A[a[:, None, None], C[None]]
                     + B[C, j[:, None]][None])
            f = e.transpose(1, 0, 2).reshape(len(j), -1) / 3.0
            m = f.max(1, keepdims=True)
            out[r, :, side] = (f.max(1) if tau <= 0 else
                               m[:, 0] + tau * np.log(
                                   np.exp((f - m) / tau).sum(1)))
    return out


def _bonus(A, B, is_last, k, tau, weight, short):
    """A with the square bonus ADDED on its shortlist, the rest untouched.

    Adding rather than replacing is what makes more than one round meaningful:
    an earlier version overwrote everything outside the shortlist with -1e9, so
    a second round read its squares off a crippled matrix and lost 0.05 of
    top-1 for it.
    """
    rows = np.array([i for i in range(N) if not is_last(i)])
    cand = np.argsort(-A[rows], axis=1)[:, :short]
    q = _squares(A, B, rows, cand, k, tau)
    t = q[..., 0] + q[..., 1]
    # CENTRE the term per row. These are log-assignments, so the square score is
    # NEGATIVE, and an uncentred "bonus" pushes shortlist members below the
    # candidates outside the shortlist, which get no term at all. Measured, that
    # cost recall at every depth past the first -- depth 5 fell 0.4829 to 0.4611
    # on 22 boards of 24, and depth 20 fell 0.0594 on all 24 -- while top-1
    # rose. Centred, the term re-orders WITHIN the shortlist and cannot evict
    # anything from it.
    t = t - t.mean(1, keepdims=True)
    add = np.zeros((len(rows), N))
    np.put_along_axis(add, cand, weight * t, axis=1)
    out = A.copy()
    out[rows] += add
    return out


def rerank_scores(H, V, k=16, tau=0.5, weight=0.75, rounds=1, short=20):
    """Both SCORE matrices, re-ranked by square evidence. Higher is better.

    This is the form the vote harvest needs: `harvest_votes._calibrated`
    returns log-assignments, one pair per scorer, and each scorer's mutual-best
    set is taken from them. Re-ranking there improves the opinion each scorer
    casts, which is upstream of everything the vote decides.
    """
    H = np.array(H, np.float64, copy=True)
    V = np.array(V, np.float64, copy=True)
    dh, dv = np.diag(H).copy(), np.diag(V).copy()
    np.fill_diagonal(H, -1e9)
    np.fill_diagonal(V, -1e9)
    for _ in range(max(rounds, 0)):
        Hn = _bonus(H, V, lambda i: i % G == G - 1, k, tau, weight, short)
        Vn = _bonus(V, H, lambda i: i // G == G - 1, k, tau, weight, short)
        H, V = Hn, Vn
    np.fill_diagonal(H, dh)
    np.fill_diagonal(V, dv)
    return H, V


def quad_rerank(CH, CV, k=16, tau=0.5, weight=0.75, rounds=1, short=20):
    """Both COST matrices, re-ranked by square evidence. Lower is better."""
    H, V = rerank_scores(-np.asarray(CH, np.float64),
                         -np.asarray(CV, np.float64),
                         k, tau, weight, rounds, short)
    out = []
    for L in (H, V):
        C = -L
        C -= C.min()
        np.fill_diagonal(C, 0.0)
        out.append(np.ascontiguousarray(C))
    return out
