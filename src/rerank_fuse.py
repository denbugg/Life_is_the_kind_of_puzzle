"""Splice re-ranker scores into the retriever's full score matrix.

The re-ranker only ever sees a shortlist -- scoring all 576x576 pairs jointly is
what made the earlier pair CNN untrainable (M20).  But every solver here needs a
full matrix, and the calibration that makes those solvers work (Sinkhorn, cycle
consistency, acyclicity) is defined on one.

So the shortlist entries are overwritten with re-ranker scores and everything
else keeps the retriever's opinion.  The two live on different scales, so the
re-ranker's row is affinely mapped onto the retriever's statistics over the same
candidates before substitution; that keeps the row comparable to the untouched
tail and to every other row.

Calibration runs AFTER the splice, not before, because cycle consistency has to
see the improved numbers -- it was worth R@1 0.270 -> 0.312 on retriever scores
alone (M93) and its input is now better.
"""
from __future__ import annotations

import numpy as np
import torch

from seam_cost import cycle_consistency
from seam_embed import board_logits
from seam_rerank import build_patches

G = 24
AXES = (("h", 1, lambda p, n: p % G != G - 1), ("v", G, lambda p, n: p < n - G))


@torch.no_grad()
def fused_logits(retr, rerank, tiles, k=20, width=20, chunk=16, blend=0.7):
    """Return {axis: (n, n) log-score tensor} with shortlist entries re-scored."""
    n = tiles.shape[0]
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in retr(tiles)[:4]]
    scale = retr.logit_scale.exp().detach()
    out = {}
    for axis, _step, _ok in AXES:
        S = board_logits(desc, axis).float() * scale
        S.fill_diagonal_(-1e4)
        rows = torch.arange(n, device=tiles.device)
        cand = S.topk(k, dim=1).indices

        sc = []
        li = rows.repeat_interleave(k)
        ri = cand.reshape(-1)
        rsc = S.gather(1, cand).reshape(-1)
        for i in range(0, li.numel(), chunk * k):
            with torch.autocast("cuda", torch.float16):
                sc.append(rerank(build_patches(tiles, li[i:i + chunk * k],
                                               ri[i:i + chunk * k], axis, width,
                                               rsc[i:i + chunk * k])))
        sc = torch.cat(sc).float().reshape(n, k)

        # map the re-ranker's row onto the retriever's scale over the same
        # candidates, so untouched entries stay comparable
        old = S.gather(1, cand)
        sc = ((sc - sc.mean(1, keepdim=True)) / (sc.std(1, keepdim=True) + 1e-6)
              * old.std(1, keepdim=True) + old.mean(1, keepdim=True))
        # Blend rather than replace.  The re-ranker is trained only on rows whose
        # true neighbour made the shortlist, so on the ~28% of rows where it did
        # not, the second stage is confidently choosing among wrong answers and
        # its certainty is unearned.  Keeping some of the retriever's opinion
        # caps the damage on exactly those rows.
        S = S.scatter(1, cand, blend * sc + (1.0 - blend) * old)
        S.fill_diagonal_(-1e4)
        out[axis] = S
    return out


@torch.no_grad()
def fused_costs(retr, rerank, tiles_np, k=20, width=20, rounds=3, weight=0.5,
                device="cuda", blend=0.7):
    """(n,20,20,3) float tiles -> (cost_h, cost_v) numpy, lower is better."""
    tiles = torch.from_numpy(np.ascontiguousarray(tiles_np)).permute(0, 3, 1, 2).to(device)
    lg = fused_logits(retr, rerank, tiles, k, width, blend=blend)
    H, V = cycle_consistency(lg["h"], lg["v"], rounds, weight)
    out = []
    for L in (H, V):
        C = (-L).cpu().numpy()
        C -= C.min()
        np.fill_diagonal(C, 0.0)
        out.append(np.ascontiguousarray(C))
    return out
