"""Which fragments sit on the edge of the photograph, and where that puts them.

A cardboard jigsaw gives the frame away: the border pieces have a straight side.
Ours do not.  Here "I am on the right border" can only mean "no fragment
continues me to the right", which is a statement about the matcher, and two
experiments read it off the matcher's SCORES and found almost nothing -- M228
and M229 measured AUC 0.567 to 0.638 and closed the thread on the grounds that
the left and right sides, the only purely structural ones, sat at 0.567.

They were reading the wrong number.  Sinkhorn normalises the score matrix to
double stochasticity, which asserts that every fragment HAS a right-hand
neighbour; for a 24x24 board that is false for 24 of them.  M97 added a slack
row and column to absorb exactly that, measured no change in edge precision and
shelved it -- but the slack column IS the border detector, and `_sink` discards
it on the way out.  Read instead of discarded it gives AUC 0.702 top, 0.679
bottom, 0.701 left, 0.705 right, and the four sides come out EQUAL, which is
what tells us this is structural rather than M229's content prior in disguise.

The eight orientations do not help here (0.687 against 0.696) and neither does
the amount of slack, anywhere from 1 to 24: unlike an edge decision, the slack
is already a marginal over the whole matrix, so the symmetries are reading a
statistic that has averaged their differences away.
"""
from __future__ import annotations

import numpy as np
import torch

from config import GRID as G
from seam_embed import board_logits

N = G * G
SIDES = ("top", "bottom", "left", "right")
BORDER_NET = "border_net_v2.pt"


def _sink_slack(L, slack, iters=20):
    """Sinkhorn with a slack row and column, returning what they absorb.

    Two vectors: how much mass a tile sends to "nothing continues me", and how
    much it receives from "nothing leads into me".
    """
    n = L.shape[0]
    A = torch.zeros(n + 1, n + 1, device=L.device, dtype=L.dtype)
    A[:n, :n] = L
    A[n, n] = -1e4
    r = torch.ones(n + 1, device=L.device, dtype=L.dtype)
    c = torch.ones(n + 1, device=L.device, dtype=L.dtype)
    r[n] = c[n] = float(slack)
    lr, lc = r.log(), c.log()
    for _ in range(iters):
        A = A - torch.logsumexp(A, 1, keepdim=True) + lr[:, None]
        A = A - torch.logsumexp(A, 0, keepdim=True) + lc[None, :]
    return A[:n, n].exp(), A[n, :n].exp()


@torch.no_grad()
def border_scores(models, tiles, device="cuda", slack=6):
    """Per-tile evidence for each of the four borders, averaged over models."""
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).to(device)
    acc = {s: np.zeros(N) for s in SIDES}
    for model in models:
        with torch.autocast("cuda", torch.float16):
            desc = [t.float() for t in model(x)[:4]]
        scale = model.logit_scale.exp().detach()
        for ax, (src, dst) in (("h", ("right", "left")), ("v", ("bottom", "top"))):
            A = board_logits(desc, ax).float() * scale
            A.fill_diagonal_(-1e4)
            s_out, s_in = _sink_slack(A, slack)
            acc[src] += s_out.cpu().numpy().astype(np.float64)
            acc[dst] += s_in.cpu().numpy().astype(np.float64)
    return {s: v / max(len(models), 1) for s, v in acc.items()}


@torch.no_grad()
def content_scores(net, tiles, device="cuda"):
    """The other border detector: what the fragment DEPICTS, not what follows it.

    Trained directly on "is this fragment on that edge of the photograph"
    (src/train_border.py), it reads 0.714 on the top and 0.550 on the right --
    M67's vertical composition, and retraining on 2.5 times the data moved the
    mean only from 0.609 to 0.615, so the lopsidedness is its ceiling rather
    than undertraining.  It fails where the structural detector does not, which
    is the whole reason to add it.
    """
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2)
    x = x.float().to(device) / 255.0
    with torch.autocast(device, torch.float16):
        v = net(x).float().cpu().numpy().astype(np.float64)
    return dict(zip(SIDES, v.T))


def border_prior(scores, content=None, content_weight=0.5):
    """(N, G, G) bonus for putting a tile in a cell, from the border evidence.

    Standardised per side, because the four are not on a common scale and a
    weight swept against the seam objective has to mean the same thing on each.
    Only the four edges of the board carry a term: the detector answers "is this
    tile on that border", and it has nothing to say about a cell in the middle.

    At HALF weight the content detector lifts every side of the structural one
    -- 0.735 / 0.720 / 0.718 / 0.682 against 0.702 / 0.679 / 0.701 / 0.705 --
    and at full weight it drags the right side down to 0.644, which is the side
    it is worst at.
    """
    p = np.zeros((N, G, G))
    z = {}
    for s in SIDES:
        v = np.asarray(scores[s], np.float64)
        z[s] = (v - v.mean()) / (v.std() + 1e-9)
        if content is not None:
            c = np.asarray(content[s], np.float64)
            z[s] = z[s] + content_weight * (c - c.mean()) / (c.std() + 1e-9)
    p[:, 0, :] += z["top"][:, None]
    p[:, G - 1, :] += z["bottom"][:, None]
    p[:, :, 0] += z["left"][:, None]
    p[:, :, G - 1] += z["right"][:, None]
    return p
