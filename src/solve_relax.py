"""Relaxation labelling over tile-to-position beliefs (after Vardi et al., 2023).

How this differs from everything already here
---------------------------------------------
Our calibration works on PAIR matrices: H says which tile follows which, V which
sits below which, and cycle consistency makes the two agree (M93).  That
produces better edges, and edges then have to be handed to a constructor.
Relaxation labelling keeps a belief per tile over the 576 POSITIONS instead, so
the global arrangement is the variable being optimised and no separate
construction step is needed.

It is the family belief propagation belongs to, and BP failed here for a
diagnosed reason: it enforced no exclusivity during message passing, so many
tiles claimed the same slot and a final Hungarian could not repair diverged
beliefs (M64).  Projecting onto the doubly stochastic set after every update
fixes exactly that -- it is the constraint the answer satisfies by definition.

The update.  Support for "tile i sits at position p" is the belief that some
tile j, compatible with i across direction d, sits at the position p is offset
from:

    q_i(p) = sum_d  sum_j  A_d[i, j] * P[j, p + d]

with A_d the calibrated affinity across d.  Beliefs are then multiplied by their
support and renormalised, which is the classic relaxation step, and the multi-
phase part is the Sinkhorn projection that keeps the result a feasible
assignment rather than letting mass pile onto popular slots.

Honest scope: the measured knee for every solver tried here sits at edge
precision about 0.72 and real boards supply 0.50 (M102), so this is not expected
to assemble them.  It is worth having because it optimises the quantity the
others only approach indirectly.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import GRID as G, NFRAG as N


def _affinities(cost_h, cost_v, tau, device):
    """Row-normalised affinity per direction: right, left, down, up."""
    out = []
    for C in (cost_h, cost_v):
        A = torch.as_tensor(np.ascontiguousarray(C), dtype=torch.float32, device=device)
        A = A - A.min()
        A = torch.exp(-A / max(tau, 1e-6))
        A.fill_diagonal_(0.0)
        out.append(A / A.sum(1, keepdim=True).clamp_min(1e-12))
        out.append((A / A.sum(0, keepdim=True).clamp_min(1e-12)).t().contiguous())
    # right, left, down, up
    return [out[0], out[1], out[2], out[3]]


def _shift(P, dr, dc, grid=G):
    """Move each position belief to the slot offset by (dr, dc); off-board is zero."""
    n = P.shape[0]
    Q = P.reshape(n, grid, grid)
    out = torch.zeros_like(Q)
    r0, r1 = max(0, -dr), grid - max(0, dr)
    c0, c1 = max(0, -dc), grid - max(0, dc)
    out[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc] = Q[:, r0:r1, c0:c1]
    return out.reshape(n, grid * grid)


def _sinkhorn(P, iters=10):
    for _ in range(iters):
        P = P / P.sum(1, keepdim=True).clamp_min(1e-12)
        P = P / P.sum(0, keepdim=True).clamp_min(1e-12)
    return P


def solve_relax(cost_h, cost_v, grid=G, rounds=60, tau=None, sink_iters=10,
                damping=0.7, device="cuda", init=None):
    """Return lay with lay[position] = tile index."""
    n = grid * grid
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if tau is None:
        tau = float(np.median(np.abs(cost_h - cost_h.min())) + 1e-6)
    A = _affinities(cost_h, cost_v, tau, dev)
    # A_d[i, j] means j sits at pos(i) + off_d, so the support for i at p needs
    # P[j, p + off_d].  _shift(P, dr, dc) MOVES belief to (r+dr, c+dc), i.e. it
    # reads P[j, p - (dr, dc)] -- so the shifts are the negated offsets.  Getting
    # this backwards is silent: it still converges, to nonsense.  With oracle
    # affinities the wrong sign scored place_acc 0.0000.
    offs = ((0, -1), (0, 1), (-1, 0), (1, 0))

    if init is None:
        P = torch.full((n, n), 1.0 / n, device=dev)
    else:
        P = torch.full((n, n), 0.2 / n, device=dev)
        lay = np.asarray(init)
        P[torch.as_tensor(lay, device=dev), torch.arange(n, device=dev)] += 0.8
    P = _sinkhorn(P, sink_iters)

    for _ in range(rounds):
        q = torch.zeros_like(P)
        for Ad, (dr, dc) in zip(A, offs):
            q = q + Ad @ _shift(P, dr, dc, grid)
        q = q / q.sum(1, keepdim=True).clamp_min(1e-12)
        P = P * (q ** damping)
        P = _sinkhorn(P, sink_iters)

    tile, pos = linear_sum_assignment(-P.double().cpu().numpy())
    lay = np.empty(n, np.int64)
    lay[pos] = tile
    return lay
