"""Harvest edges by CHOOSING inside the shortlist instead of taking the top.

Where this sits
---------------
M187 states the door and M409 sizes it: the true neighbour is in the top five
for about 48% of fragments, which holds 543 correct bonds a board once the
square re-ranking is centred (M441), against the 450 to 500 M407 puts the
percolation knee at -- and the top-1 harvest delivers about 348. So the
shortlist is past the knee and the whole question is which of the five is taken.

M412 built the model that answers it and rejected it for a reason it stated
itself: 136 boards is not enough for a transformer over five candidates, and
"the experiment has not yet been RUN at the scale it needs". M438 runs it on
2901.

This is the inference side: the trained chooser picks one of five or abstains,
and the picks become the harvest. The features must match training exactly --
the raw score over ten, its lead over the shortlist's best, the rank, and
whether it IS the best -- because the model was fitted on those and nothing
else.
"""
import numpy as np
import torch

from choose5 import K, Choose5, seam_patch
from config import GRID as G

N = G * G


def load_chooser(path, device="cuda"):
    c = torch.load(path, map_location=device, weights_only=False)
    a = c.get("args", {})
    m = Choose5(a.get("ch", 64), a.get("dim", 192), a.get("strip", 4),
                a.get("layers", 3), encoder=a.get("encoder", "cnn")).to(device)
    m.load_state_dict(c["model"])
    m.strip = a.get("strip", 4)
    m.eval()
    return m


def _shortlist(M, k=K):
    D = np.array(M, np.float64)
    np.fill_diagonal(D, -1e9)
    idx = np.argpartition(-D, k, axis=1)[:, :k]
    val = np.take_along_axis(D, idx, axis=1)
    o = np.argsort(-val, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(val, o, 1)


def select_confident(edges, keep=0, floor=None):
    """Apply a board-adaptive confidence floor and an optional safety cap."""
    ranked = sorted(edges.items(), key=lambda kv: -kv[1])
    if floor is not None:
        ranked = [item for item in ranked if item[1] >= floor]
    if keep:
        ranked = ranked[:keep]
    return dict(ranked)


def chooser_edges(model, tiles, CH, CV, device="cuda", keep=0, floor=None):
    """{(i, j, offset): weight} -- one edge per fragment the chooser commits to.

    An abstention emits nothing, which is the point: NONE is the right answer
    for about half the fragments, and M412 measured that a model trained
    without discounting it collapses to abstaining everywhere.
    """
    strip = getattr(model, "strip", 4)
    x = torch.from_numpy(np.ascontiguousarray(tiles)).float().to(device)
    out = {}
    for M, axis, off, last in ((-np.asarray(CH, np.float64), "h", (0, 1),
                                lambda i: i % G == G - 1),
                               (-np.asarray(CV, np.float64), "v", (1, 0),
                                lambda i: i // G == G - 1)):
        idx, val = _shortlist(M)
        rows = np.array([i for i in range(N) if not last(i)])
        if not len(rows):
            continue
        ii = torch.from_numpy(idx[rows].astype(np.int64)).to(device)
        vv = torch.from_numpy(val[rows]).float().to(device)
        src = torch.from_numpy(rows).to(device).repeat_interleave(K)
        dst = ii.reshape(-1)
        patch = seam_patch(x, src, dst, axis, strip).reshape(
            len(rows), K, 3, 20, 2 * strip)
        rank = torch.arange(K, device=device, dtype=torch.float32)
        z = vv - vv[:, :1]
        sc = torch.stack([vv / 10.0, z, rank.expand(len(rows), K),
                          (z == 0).float()], -1)
        with torch.no_grad():
            logits = model(patch, sc)
        pick = logits.argmax(1)
        top2 = logits.topk(2, dim=1).values
        conf = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        pick = pick.cpu().numpy()
        for r, i in enumerate(rows):
            if pick[r] >= K:                      # abstained
                continue
            out[(int(i), int(idx[i, pick[r]]), off)] = float(conf[r])
    return select_confident(out, keep, floor)
