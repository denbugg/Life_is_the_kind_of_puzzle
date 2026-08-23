"""Turn matcher descriptors into the cost matrices solvers actually want.

Three stages, each measured:

1. Log-probability at the model's own learned temperature, not 1 - cosine.
   Unit-norm dot products compress into a narrow band -- median relative margin
   0.0300 against MGC's 0.1299 -- and build_matches weights edges by the SQUARE
   of that margin, so the weighting degenerates to uniform, which M45 showed is
   exactly where the LP breaks (5% outlier tolerance instead of 15%).

2. Sinkhorn, which imposes the constraint the answer satisfies anyway: every
   tile has one right-hand neighbour and is one tile's right-hand neighbour.
   Raised R@1 0.261 -> 0.287, mutual edges 459 -> 590, margin 0.0300 -> 0.1449.

3. Cycle consistency between the two axes.  For the true layout the
   right-neighbour matrix H and the below-neighbour matrix V commute: going
   down from i, right, then up must land where going right from i lands, so
   V H V^T is evidence about H and H V H^T is evidence about V.  Loop
   verification (M11, M92) is the hard version of the same test and fails here
   because it DISCARDS -- on real boards it cut 598 edges at precision 0.438
   down to 22 at 0.628, and connectivity is what assembly runs out of (M87).
   Multiplying beliefs instead reinforces a weak-but-consistent edge, and gains
   on both counts at once: real boards 0.270 -> 0.312 R@1, 601 -> 696 edges,
   precision 0.436 -> 0.458; at severity 0.4, 0.433 -> 0.516, 659 -> 776,
   0.626 -> 0.679.  No training involved.

4. Acyclicity.  H decomposes into 24 chains of 24 tiles, so it has no cycles;
   evidence against edge (i, j) is the belief in the return path.  Another +0.02
   edge precision, saturating near weight 4.

The consistency weight was re-tuned against the current matcher and moved from
0.50 to 0.35: precision 0.4471 -> 0.4616 on six held-out boards, for slightly
fewer edges (675 against 701).  Worth redoing whenever the matcher changes --
these constants were fitted to an earlier one and had drifted.
"""
from __future__ import annotations

import numpy as np
import torch

from seam_embed import board_logits


def _sink(L, iters=20, slack=0):
    """Sinkhorn in log space, optionally with a slack row and column.

    Plain double stochasticity asserts that EVERY tile has a right-hand
    neighbour and is some tile's right-hand neighbour.  That is false for a
    24x24 board: the 24 tiles of the rightmost column have no right neighbour
    and the 24 of the leftmost are nobody's.  Forcing them to claim one invents
    24 false edges per axis, and cycle consistency then propagates the lie.

    With slack, one extra column absorbs "no right neighbour" and one extra row
    absorbs "no left neighbour", each carrying `slack` units of mass, so the
    true structure -- a partial matching of 552, not a permutation of 576 -- is
    what gets normalised.
    """
    if slack <= 0:
        for _ in range(iters):
            L = L - torch.logsumexp(L, 1, keepdim=True)
            L = L - torch.logsumexp(L, 0, keepdim=True)
        return L

    n = L.shape[0]
    dev, dt = L.device, L.dtype
    # the slack entries start neutral; their mass is set by the target marginals
    A = torch.zeros(n + 1, n + 1, device=dev, dtype=dt)
    A[:n, :n] = L
    A[n, n] = -1e4                          # slack-to-slack is meaningless
    r = torch.ones(n + 1, device=dev, dtype=dt)
    c = torch.ones(n + 1, device=dev, dtype=dt)
    r[n] = c[n] = float(slack)
    lr, lc = r.log(), c.log()
    for _ in range(iters):
        A = A - torch.logsumexp(A, 1, keepdim=True) + lr[:, None]
        A = A - torch.logsumexp(A, 0, keepdim=True) + lc[None, :]
    return A[:n, :n]


def _acyclic(M):
    """Log-probability that the return path does not exist, for 2- and 3-cycles.

    The true right-neighbour matrix decomposes into 24 chains of 24 tiles and so
    has no cycles at all, yet nothing else in this pipeline forbids believing
    both that B is right of A and that A is right of B: Sinkhorn only balances
    marginals and cycle consistency only compares the two axes to each other.
    Worth about +0.02 edge precision, saturating around weight 4.
    """
    return (torch.log(torch.clamp(1.0 - M.t(), min=1e-6))
            + torch.log(torch.clamp(1.0 - (M @ M).t(), min=1e-6)))


def cycle_consistency(logit_h, logit_v, rounds=3, weight=0.35, iters=20, slack=0,
                      acyclic=3.0):
    """Refine two log-assignment matrices against each other. Returns log H, V."""
    H, V = _sink(logit_h, iters, slack), _sink(logit_v, iters, slack)
    for _ in range(rounds):
        Hm, Vm = H.exp(), V.exp()
        ev_h = torch.log(torch.clamp(Vm @ Hm @ Vm.t(), min=1e-12))
        ev_v = torch.log(torch.clamp(Hm @ Vm @ Hm.t(), min=1e-12))
        if acyclic > 0:
            ev_h = ev_h + acyclic * _acyclic(Hm)
            ev_v = ev_v + acyclic * _acyclic(Vm)
        H = _sink(H + weight * ev_h, iters, slack)
        V = _sink(V + weight * ev_v, iters, slack)
    return H, V


def costs_from_model(model, tiles, rounds=3, weight=0.35, device="cuda", slack=0):
    """(n,20,20,3) float tiles -> (cost_h, cost_v) numpy, lower is better.

    Costs are shifted to be non-negative because build_matches forms ratios of
    them; the diagonal is zeroed so callers can mask it themselves.
    """
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in model(x)[:4]]
    scale = model.logit_scale.exp().detach()
    lg = []
    for ax in ("h", "v"):
        # float32 throughout: logits span roughly +-30 after the temperature, so
        # logsumexp is comfortable, and fp64 matmuls run at 1/64 rate on this
        # card -- 915 ms per call against 79 ms, which dominated the re-ranker's
        # training step
        A = board_logits(desc, ax, getattr(model, "modes", 1)).float() * scale
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1], rounds, weight, slack=slack)
    out = []
    for L in (H, V):
        C = (-L).cpu().numpy()
        C -= C.min()
        np.fill_diagonal(C, 0.0)
        out.append(np.ascontiguousarray(C))
    return out


def costs_from_models(models, tiles, rounds=3, weight=0.35, device="cuda", slack=0):
    """Cost matrices from one matcher or several, combined pessimistically.

    M201 measured how two matchers of comparable strength should be combined,
    and it is not by addition.  Adding two calibrated log-assignments
    manufactures mutual-best pairs of mediocre quality -- 150 extra edges at
    precision 0.404 where either matcher alone reads about 0.47.  The
    elementwise MINIMUM leaves the edge count where it was and raises precision
    to 0.496: an edge survives only if BOTH matchers believe it, which is the
    conservative reading of two comparable posteriors.

    Each matrix is Sinkhorned on its own first -- that is what makes the two
    comparable -- and cycle consistency runs once on the result, exactly as it
    runs on a single view.

    Combine only matchers of similar strength.  Adding weaker checkpoints
    degrades the result monotonically: min of 2 gives 0.496, of 3 0.492, of 4
    0.482, of 6 0.479, against 0.485 for the best one alone.
    """
    if not isinstance(models, (list, tuple)):
        return costs_from_model(models, tiles, rounds, weight, device, slack)
    if len(models) == 1:
        return costs_from_model(models[0], tiles, rounds, weight, device, slack)

    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).to(device)
    per = []
    for model in models:
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            desc = [t.float() for t in model(x)[:4]]
        scale = model.logit_scale.exp().detach()
        lg = []
        for ax in ("h", "v"):
            A = board_logits(desc, ax, getattr(model, "modes", 1)).float() * scale
            A.fill_diagonal_(-1e4)
            lg.append(_sink(A, slack=slack))
        per.append(lg)
    lh = torch.stack([p[0] for p in per]).amin(0)
    lv = torch.stack([p[1] for p in per]).amin(0)
    H, V = cycle_consistency(lh, lv, rounds, weight, slack=slack)
    out = []
    for L in (H, V):
        C = (-L).cpu().numpy()
        C -= C.min()
        np.fill_diagonal(C, 0.0)
        out.append(np.ascontiguousarray(C))
    return out
