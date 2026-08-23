"""Holistic full-pair compatibility scorer, trained in the deployment domain.

Background
----------
R8 (autoresearch-runs/pazzle-fixed-orientation-20260813) scored a concatenated
tile PAIR jointly instead of factorising compatibility into independent tile
embeddings, and beat every previous scorer: CAL Recall@20 58.80% versus 47.83%
for the frozen directional Siamese.  It was then rejected at G2 because on the
real rank96 graph its coverage collapsed to 22.51%.

That collapse is explained: R8 trained on CanvasDataset(real_prob=0.0), and the
synthetic forward model understates the photometric spread.  Measured on real
(dirty, clean) pairs, contrast a_std is 0.271; distort.py with
CONTRAST=(0.70,1.30) produces 0.161.  Everything else matches closely (residual
noise 13.1 vs 13.3, autocorrelation 0.732 vs 0.735, JPEG blockiness 1.49 vs
1.50, b_std 21 vs 23).

So the architecture was never the problem, the training domain was.  This
module keeps the joint-pair idea and trains on REAL fragment pairs whose
adjacency comes from build_restore_labels.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FS, GRID as G, NFRAG as N


class PairCompat(nn.Module):
    """Scores a joined tile pair.  Input (B,3,20,40): left half is the anchor,
    right half the candidate placed to its right.  Vertical seams are fed by
    transposing the two tiles before joining, so one head serves both axes."""

    def __init__(self, ch: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1, padding_mode="reflect"), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect"), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.GroupNorm(8, ch * 2), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 4), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(ch * 4, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """pair: (B,3,20,40) in [0,255] -> (B,) logit."""
        m = pair.mean(dim=(1, 2, 3), keepdim=True)
        s = pair.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        x = (pair - m) / s
        # 4th plane marks which half a column belongs to, so the network can
        # locate the seam without having to infer it from content.
        side = torch.zeros_like(x[:, :1])
        side[:, :, :, FS:] = 1.0
        h = self.net(torch.cat([x, side], dim=1))
        return self.head(h.mean(dim=(2, 3))).squeeze(1)


def join_h(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a,b: (B,3,20,20) -> (B,3,20,40), b placed to the right of a."""
    return torch.cat([a, b], dim=3)


def join_v(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """b placed below a.  Transposed so the vertical seam becomes vertical in
    the joined image; tiles themselves are never rotated in the output."""
    return torch.cat([a.transpose(2, 3), b.transpose(2, 3)], dim=3)


@torch.no_grad()
def dense_scores(model: PairCompat, tiles: torch.Tensor, axis: str,
                 batch: int = 2048, shortlist: np.ndarray | None = None) -> np.ndarray:
    """Full 576x576 logit matrix (higher = more compatible).

    `shortlist` (N,K) restricts scoring to candidate columns; entries outside it
    get -inf.  Scoring all 331k pairs per axis is affordable but the shortlist
    keeps end-to-end inference on 700 boards inside a sane budget.

    Keep `batch` under ~4000 pairs.  Measured on the 8 GB card, a forward+
    backward costs 0.234 s at 2112 pairs and 17.4 s at 6272: past that point the
    activations no longer fit and silently spill into WDDM shared memory.
    """
    n = len(tiles)
    join = join_h if axis == "h" else join_v
    out = np.full((n, n), -np.inf, np.float32)
    if shortlist is None:
        rows = np.repeat(np.arange(n), n)
        cols = np.tile(np.arange(n), n)
    else:
        rows = np.repeat(np.arange(n), shortlist.shape[1])
        cols = shortlist.reshape(-1)
    for s in range(0, len(rows), batch):
        r = torch.from_numpy(rows[s:s + batch]).to(tiles.device)
        c = torch.from_numpy(cols[s:s + batch]).to(tiles.device)
        with torch.autocast("cuda", torch.float16, enabled=tiles.is_cuda):
            v = model(join(tiles[r], tiles[c]))
        out[rows[s:s + batch], cols[s:s + batch]] = v.float().cpu().numpy()
    np.fill_diagonal(out, -np.inf)
    return out


def pair_metrics(score: np.ndarray, axis: str) -> dict[str, float]:
    """score in TRUE grid order, higher = better.  Mirrors restore_tile.seam_metrics."""
    step = 1 if axis == "h" else G
    edge = (lambda p: (p % G) != G - 1) if axis == "h" else (lambda p: p < N - G)
    rows = np.array([p for p in range(N) if edge(p)])
    order = np.argsort(-score[rows], axis=1)
    rank = np.array([np.where(order[k] == rows[k] + step)[0][0] for k in range(len(rows))])
    best_f, best_b = np.argmax(score, 1), np.argmax(score, 0)
    bb = [(i, best_f[i]) for i in range(N) if best_b[best_f[i]] == i]
    ok = sum(1 for i, j in bb if edge(i) and j == i + step)
    return {"R1": float((rank == 0).mean()), "R20": float((rank < 20).mean()),
            "bb_prec": float(ok / max(1, len(bb)))}
