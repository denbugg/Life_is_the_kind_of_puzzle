"""The edge harvest: agreement across architectures AND inputs at once.

Every selection scheme this project tried before diversified along one axis and
they all landed on the same ceiling -- loop closure 0.154 of the board inside
internally-correct components, two-model agreement 0.158, a seven-model vote
0.159, an unfrozen trunk 0.158 -- which M163 recorded as an information limit.

It was not one.  Architecture and input produce different mistakes, and
requiring agreement across BOTH at once reaches 0.259 (M212, M214), two thirds
past that ceiling.  Nine scorers: three matchers, each run on the raw tiles and
on the output of each per-tile restorer, every one through the full calibrated
path, and an edge kept when enough of them independently call it mutually best.

Six votes of nine is the operating point, and it is not the cleanest one (M215).
Clean coverage keeps rising to nine votes while the packed layout's adjacency
peaks at six and then falls, because a stricter filter leaves the packer fewer
edges and it fills the rest with the seam score M160 measured at precision
0.349.  The best harvest is the cleanest one that still constrains the whole
board: adjacency 0.234 against the previous 0.192, at equal placement.
"""
from __future__ import annotations

import numpy as np
import torch

from config import GRID as G
from seam_cost import _sink, cycle_consistency
from seam_embed import board_logits

N = G * G


@torch.no_grad()
def _calibrated(model, tiles, device, orient=(0, 0, 0)):
    """One of the eight symmetries of the board (M236).

    The matcher's four heads do not share weights, so showing it the board
    transposed or flipped makes a different pair of them judge the same seam.
    A left-right flip additionally reverses the horizontal relation, so the
    result is transposed back as a matrix. Eight readings per model per input,
    from weights that are already trained.
    """
    tr, lr, ud = orient
    if lr:
        tiles = tiles[:, :, ::-1]
    if ud:
        tiles = tiles[:, ::-1]
    if tr:
        tiles = tiles.transpose(0, 2, 1, 3)
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).to(device)
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in model(x)[:4]]
    scale = model.logit_scale.exp().detach()
    lg = []
    for ax in ("h", "v"):
        # a multi-mode checkpoint must be SCORED with its modes or the
        # extra sub-vectors are silently read as one long descriptor
        A = board_logits(desc, ax, getattr(model, "modes", 1)).float() * scale
        A.fill_diagonal_(-1e4)
        lg.append(_sink(A))
    H, V = lg
    if tr:
        H, V = V, H
    if lr:
        H = H.t().contiguous()
    if ud:
        V = V.t().contiguous()
    H, V = cycle_consistency(H, V, 3, 0.35)
    return (H.cpu().numpy().astype(np.float64),
            V.cpu().numpy().astype(np.float64))


def _mutual(M, offset):
    """Mutual-best edges of one log-assignment matrix, keyed (i, j, offset)."""
    D = M.copy()
    np.fill_diagonal(D, -np.inf)
    forward, back = D.argmax(1), D.argmax(0)
    part = np.partition(D, -2, axis=1)
    return {(i, int(forward[i]), offset): float(part[i, -1] - part[i, -2])
            for i in range(N) if int(back[int(forward[i])]) == i}


ORIENTATIONS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def votes_for_target(sets, target):
    """The highest vote bar whose harvest still reaches `target` edges.

    M344 measured what decides a board's outcome: the NUMBER of voted edges
    predicts the assembly's adjacency at 0.955, better than the harvest's true
    edge precision at 0.652, so volume decides and purity follows.  The bar that
    sets that volume was global -- thirteen of eighteen on every board alike --
    while the count clearing it varies board to board by a factor of several.
    M288 and M289 swept the bar, but swept it globally.

    Choosing the bar per board to a fixed volume beats the fixed bar AT MATCHED
    VOLUME (M346): against a fixed eight yielding 431 edges, a per-board target
    of 400 yields 441 and lifts SSIM from 0.0983 to 0.1050 and fragment
    placement from 0.0021 to 0.0145.  So the gain is the per-board choice and
    not the lower bar.
    """
    counts = {}
    for s_ in sets:
        for e in s_:
            counts[e] = counts.get(e, 0) + 1
    for v in range(len(sets), 0, -1):
        if sum(1 for c in counts.values() if c >= v) >= target:
            return v
    return 1


def voted_edges(models, inputs, device="cuda", votes=8, orientations=2,
                margin=0.0, target=0):
    """Edges enough of the scorers agree on, strongly enough.

    With `target`, the vote bar is chosen PER BOARD so the harvest reaches that
    many edges rather than clearing a fixed bar -- see `votes_for_target`.

    The scorers are every matcher, on every input, in `orientations` of the
    board's eight symmetries.  `inputs` lists the tile arrays -- the raw tiles
    first, then one per restorer.  The returned weight is the MINIMUM margin any
    scorer gave the edge, the pessimistic reading M201 found to be the right way
    to combine comparable opinions.

    Two knobs, not one (M224): votes count how many scorers picked the edge and
    margin measures how far ahead they picked it, and they reach different
    places -- votes alone top out at precision 0.942 while eight votes at margin
    2.0 reach 0.997.

    Twenty-four scorers match or beat seventy-two, and which twenty-four depends
    on the goal (M237).  Three architectures on the raw input alone, all eight
    orientations, maximise ADJACENCY at 0.271.  One architecture on three inputs,
    all eight orientations, maximise PLACEMENT at 0.0132, which is the quantity
    the metric pays for.  Two architectures, three inputs, four orientations
    maximise CLEAN COVERAGE at 0.288.  All three beat the eighteen-scorer
    arrangement they replaced.
    """
    sets = _scorer_sets(models, inputs, device, orientations)
    if target:
        votes = votes_for_target(sets, target)
    seen = set()
    for s in sets:
        seen |= set(s)
    out = {}
    for e in seen:
        hit = [s[e] for s in sets if e in s]
        if len(hit) >= votes and min(hit) >= margin:
            out[e] = min(hit)
    return out


def _scorer_sets(models, inputs, device, orientations):
    sets = []
    for model in models:
        for tiles in inputs:
            for orient in ORIENTATIONS[:orientations]:
                H, V = _calibrated(model, tiles, device, orient)
                sets.append({**_mutual(H, (0, 1)), **_mutual(V, (1, 0))})
    return sets


def voted_pool(models, inputs, device="cuda", orientations=2):
    """Every pair ANY scorer called mutually best, with how many did.

    The threshold throws this away, and the edges below it are what a
    corroboration test needs: an offset between two components that TWO
    independent tile pairs agree on is right far more often than one a single
    pair proposes -- 0.938 against 0.111 in M248 -- and most of those second
    pairs sit under the vote threshold.
    """
    pool = {}
    for s in _scorer_sets(models, inputs, device, orientations):
        for e in s:
            pool[e] = pool.get(e, 0) + 1
    return pool
