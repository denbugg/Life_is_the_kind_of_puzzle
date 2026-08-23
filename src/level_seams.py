"""Undo the per-fragment brightness the generator handed out, from the seams.

`distort.py` gives every fragment its own contrast around its own grey mean and
its own additive offset, both drawn independently, so the assembled board wears
a 24x24 grid of brightness steps.  That grid is exactly what the organisers ask
to have removed -- "level the brightness, remove the artefacts" -- and it is
also the largest single source of low-frequency structure that does not belong
to the photograph.

The estimate needs no knowledge of which fragment is which.  Wherever two
fragments touch, the true image was continuous across that seam, so the step we
observe there is the difference of their two offsets.  With 1104 seams on the
board and 576 unknowns per channel that is an over-determined linear system,
and the same construction panorama stitchers use for gain compensation:

    minimise  sum over seams (o_a - o_b + (v_a - v_b))^2  +  lam * sum o_i^2

where v_a, v_b are the mean colours of the touching strips.  The regulariser is
what keeps the solution from drifting as a whole -- the seam terms only fix the
offsets up to a constant, and that constant is the board's mean, which we want
left alone.

This is photometric only: no fragment moves, none is resized or warped, and the
correction is a single number per fragment per channel.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

from config import GRID as G

N = G * G


def _strip_means(tiles, strip):
    """Mean colour of each fragment's four border strips: (N, 4, 3).

    Order is left, right, top, bottom.
    """
    t = tiles.astype(np.float32)
    return np.stack([t[:, :, :strip].mean((1, 2)), t[:, :, -strip:].mean((1, 2)),
                     t[:, :strip].mean((1, 2)), t[:, -strip:].mean((1, 2))], 1)


def seam_offsets(tiles, lay, lam=0.02, strip=2):
    """Additive per-fragment, per-channel offset: (N, 3), indexed by fragment.

    `lay[cell] = fragment`, cells row-major on the GxG board.
    """
    v = _strip_means(tiles, strip)
    lay = np.asarray(lay, np.int64)
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for cell in range(N):
        y, x = divmod(cell, G)
        a = int(lay[cell])
        if x + 1 < G:                       # a's right meets b's left
            b = int(lay[cell + 1])
            rows += [r, r]; cols += [a, b]; vals += [1.0, -1.0]
            rhs.append(v[b, 0] - v[a, 1])
            r += 1
        if y + 1 < G:                       # a's bottom meets b's top
            b = int(lay[cell + G])
            rows += [r, r]; cols += [a, b]; vals += [1.0, -1.0]
            rhs.append(v[b, 2] - v[a, 3])
            r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, N)).tocsr()
    B = np.asarray(rhs, np.float64)                       # (r, 3)
    M = (A.T @ A + lam * eye(N, format="csr")).tocsc()
    o = np.stack([spsolve(M, A.T @ B[:, c]) for c in range(3)], 1)
    return (o - o.mean(0, keepdims=True)).astype(np.float32)


def _seam_pairs(lay):
    """(a, b, side_a, side_b) for every seam; sides index `_strip_means`."""
    lay = np.asarray(lay, np.int64)
    out = []
    for cell in range(N):
        y, x = divmod(cell, G)
        if x + 1 < G:
            out.append((int(lay[cell]), int(lay[cell + 1]), 1, 0))
        if y + 1 < G:
            out.append((int(lay[cell]), int(lay[cell + G]), 3, 2))
    return out


def seam_offsets_and_gains(tiles, lay, lam_o=0.02, lam_g=2.0, rounds=6, strip=2):
    """Per-fragment gain and offset for `x -> g * (x - mu) + mu + o`.

    The generator draws contrast from 0.70 to 1.30 around each fragment's own
    mean, so an offset alone cannot undo it.  The seam evidence for a gain is
    weaker than for an offset -- two fragments can agree on their border means
    and still differ in contrast -- so the gain carries a much heavier prior
    toward 1, and the two are solved by alternation rather than jointly.

    DO NOT SHIP THIS ON A LAYOUT THAT IS MOSTLY WRONG (M185).  On the true
    layout it is worth +0.0083 over offsets alone.  On a wrong one the solved
    gains have mean 0.745 and 85% of them fall outside [0.6, 1.6] -- far outside
    anything the generator produces -- because the only way to make a seam that
    was never continuous look continuous is to crush the contrast on both sides.
    The score rises while the detail falls from 31.3 to 21.1: it is smoothing
    wearing a restoration's clothes.
    """
    v = _strip_means(tiles, strip)
    pivot = tiles.mean((1, 2))
    pairs = _seam_pairs(lay)
    r = len(pairs)
    ra = np.array([p[0] for p in pairs])
    rb = np.array([p[1] for p in pairs])
    va = np.stack([v[p[0], p[2]] for p in pairs])
    vb = np.stack([v[p[1], p[3]] for p in pairs])
    rows = np.repeat(np.arange(r), 2)
    cols = np.stack([ra, rb], 1).ravel()
    A = coo_matrix((np.tile([1.0, -1.0], r), (rows, cols)), shape=(r, N)).tocsr()
    Mo = (A.T @ A + lam_o * eye(N, format="csr")).tocsc()
    g = np.ones((N, 3), np.float64)
    o = np.zeros((N, 3), np.float64)
    for _ in range(rounds):
        rhs = ((g[rb] * (vb - pivot[rb]) + pivot[rb])
               - (g[ra] * (va - pivot[ra]) + pivot[ra]))
        o = np.stack([spsolve(Mo, A.T @ rhs[:, c]) for c in range(3)], 1)
        o -= o.mean(0, keepdims=True)
        da, db = va - pivot[ra], vb - pivot[rb]
        rhs2 = (o[rb] - o[ra]) + (pivot[rb] - pivot[ra])
        for c in range(3):
            vals = np.stack([da[:, c], -db[:, c]], 1).ravel()
            B = coo_matrix((vals, (rows, cols)), shape=(r, N)).tocsr()
            Mg = (B.T @ B + lam_g * eye(N, format="csr")).tocsc()
            g[:, c] = spsolve(Mg, B.T @ rhs2[:, c] + lam_g * np.ones(N))
        g = np.clip(g, 0.5, 2.0)
    return o.astype(np.float32), g.astype(np.float32)


def level(tiles, lay, lam=0.02, strip=2, gains=False):
    """Fragments with their seam-estimated photometry removed."""
    x = tiles.astype(np.float32)
    if not gains:
        o = seam_offsets(tiles, lay, lam, strip)
        return np.clip(x + o[:, None, None, :], 0, 255)
    o, g = seam_offsets_and_gains(tiles, lay, lam, strip=strip)
    mu = x.mean((1, 2))[:, None, None, :]
    return np.clip(g[:, None, None, :] * (x - mu) + mu + o[:, None, None, :],
                   0, 255)
