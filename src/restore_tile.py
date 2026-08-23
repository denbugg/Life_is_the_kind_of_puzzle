"""Pre-assembly tile restoration: dirty 20x20 fragment -> clean 20x20 fragment.

Why this exists
---------------
Measured seam-matching budget (best-buddy precision, what a solver needs):

    clean tiles                        0.946
    clean + intra-tile 3x3 blur        0.863
    dirty + ORACLE photometry          0.295
    dirty as-is                        0.113

So the ceiling is not the matcher, it is the input.  Every scorer in this repo
was fitted on the 0.113 row.  This module attacks the row itself.

Design
------
The generator applies, per fragment, a SCALAR affine  x -> a*(x-pivot)+pivot+b
before noise/blur/JPEG.  Normalising a tile by its own mean/std therefore
removes a and b exactly.  The trunk runs in that normalised space and only has
to recover STRUCTURE; a separate global head re-predicts the absolute mean/std.
Mixing the two in one conv stack makes the network fight itself, because b is
not recoverable from tile content while structure is.

The restoration target is the ORIGINAL tile, not a denoised one: undoing the
3x3 blur is worth 0.863 -> 0.946 of solver headroom.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GRID as G, FS, NFRAG as N


# --------------------------------------------------------------------------- #
# tiles
# --------------------------------------------------------------------------- #
def to_frags(img: np.ndarray) -> np.ndarray:
    """(480,480,3) -> (576,20,20,3), row-major grid order."""
    return img.reshape(G, FS, G, FS, 3).transpose(0, 2, 1, 3, 4).reshape(N, FS, FS, 3)


def from_frags(frags: np.ndarray) -> np.ndarray:
    """(576,20,20,3) -> (480,480,3)."""
    return frags.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def blur3_np(x: np.ndarray) -> np.ndarray:
    """Separable 3x3 Gaussian with reflect padding, matching the generator."""
    xp = np.pad(x, ((0, 0), (1, 1), (0, 0), (0, 0)), "reflect")
    x = .25 * xp[:, :-2] + .5 * xp[:, 1:-1] + .25 * xp[:, 2:]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), "reflect")
    return .25 * xp[:, :, :-2] + .5 * xp[:, :, 1:-1] + .25 * xp[:, :, 2:]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, padding_mode="reflect")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, padding_mode="reflect")
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return x + self.n2(self.c2(h))


class TileRestorer(nn.Module):
    """dirty tile -> clean tile.  Full resolution throughout (20x20 is too small
    to pool), receptive field widened with dilation instead of downsampling."""

    def __init__(self, ch: int = 96, blocks: int = 6, residual: bool = False,
                 checkpoint: bool = False, ycc: bool = False):
        super().__init__()
        # Predict a CORRECTION to the normalised input rather than the structure
        # from scratch, so the network starts from identity.  Adds no weights,
        # hence checkpoints stay loadable; older ones were trained without it and
        # must keep residual=False.
        self.residual = residual
        # Recompute block activations in the backward pass.  A 576-tile board
        # at ch=128/blocks=8 otherwise exceeds the 8 GB card and silently spills
        # into WDDM shared memory, where training emits no step at all.
        self.checkpoint = checkpoint
        # Extra input planes carrying the YCrCb view.  JPEG quantises chroma
        # separately and subsamples it 4:2:0, so the colour channels survive the
        # corruption far better: measured residual sigma is 11.68 on Y against
        # 5.52 and 4.85 on Cr and Cb.  In RGB that advantage is smeared across
        # all three planes, so the split is handed to the network explicitly.
        self.ycc = ycc
        # 4 input planes: normalised RGB + a constant plane carrying tile std
        self.stem = nn.Conv2d(7 if ycc else 4, ch, 3, padding=1, padding_mode="reflect")
        dil = [1, 2, 3, 2, 1, 1][:blocks] + [1] * max(0, blocks - 6)
        self.body = nn.Sequential(*[ResBlock(ch, d) for d in dil])
        self.head = nn.Conv2d(ch, 3, 3, padding=1, padding_mode="reflect")
        # global head: predicts the CLEAN tile's mean/std from pooled features
        # plus the observed statistics (b is only partly recoverable, so this
        # head learns the Bayes shrinkage rather than an exact inverse).
        self.glob = nn.Sequential(
            nn.Linear(ch + 6, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,20,20) float in [0,255].  Returns (B,3,20,20) in [0,255]."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        xn = (x - mean) / std
        std_plane = (std / 64.0).expand(-1, -1, FS, FS)[:, :1]
        planes = [xn, std_plane]
        if self.ycc:
            r, g, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            y = 0.299 * r + 0.587 * g + 0.114 * bl
            cr = (r - y) * 0.713
            cb = (bl - y) * 0.564
            yc = torch.cat([y, cr, cb], dim=1)
            yc = (yc - yc.mean(dim=(2, 3), keepdim=True)) / yc.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
            planes.append(yc)
        h = self.stem(torch.cat(planes, dim=1))
        if self.checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint_sequential
            h = checkpoint_sequential(self.body, len(self.body), h, use_reentrant=False)
        else:
            h = self.body(h)
        struct = self.head(h)                                    # normalised structure
        if self.residual:
            struct = struct + xn

        pooled = h.mean(dim=(2, 3))
        stats = torch.cat([mean.flatten(1) / 128.0, std.flatten(1) / 64.0], dim=1)
        g = self.glob(torch.cat([pooled, stats], dim=1))
        # predict clean stats as a correction of the observed ones
        out_mean = mean.flatten(1) + 30.0 * torch.tanh(g[:, :3])
        out_std = std.flatten(1) * torch.exp(g[:, 3:].clamp(-1.5, 1.5))

        struct = struct - struct.mean(dim=(2, 3), keepdim=True)
        struct = struct / struct.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        return struct * out_std[:, :, None, None] + out_mean[:, :, None, None]


# --------------------------------------------------------------------------- #
# seam scoring on restored tiles (the gate metric)
# --------------------------------------------------------------------------- #
def ridge_cost(tiles: np.ndarray, w: float = 0.03, cols: int = 3, axis: str = "h") -> np.ndarray:
    """cost[i,j] = var(d) + w*mean(d)^2 over the facing border strips.

    The per-tile brightness b is a nuisance parameter: fitting it freely (w=0)
    throws away real DC signal, ignoring it (w=1) eats the +-30 jitter, so the
    ridge optimum sits in between.

    Defaults are the held-out optimum ON RESTORED tiles (bb_prec 0.4117).  Raw
    tiles peak elsewhere (w=0.12, cols=2, bb_prec 0.3438): restoration already
    removes part of the brightness error, which both devalues the DC term and
    widens the usable border strip.
    """
    n_tiles = len(tiles)
    if axis == "h":
        A = tiles[:, :, -cols:, :].reshape(n_tiles, -1, 3)
        B = tiles[:, :, :cols, :].reshape(n_tiles, -1, 3)
    else:
        A = tiles[:, -cols:, :, :].reshape(n_tiles, -1, 3)
        B = tiles[:, :cols, :, :].reshape(n_tiles, -1, 3)
    n = A.shape[1]
    m2 = ((A ** 2).sum(1)[:, None, :] + (B ** 2).sum(1)[None, :, :]
          - 2 * np.einsum("ikc,jkc->ijc", A, B)) / n
    mu = A.mean(1)[:, None, :] - B.mean(1)[None, :, :]
    return (m2 - mu ** 2 + w * mu ** 2).sum(-1)


def ridge_cost_torch(tiles: torch.Tensor, w: float = 0.03, cols: int = 3,
                     axis: str = "h") -> torch.Tensor:
    """Differentiable twin of ridge_cost.  tiles: (N,3,20,20) -> (N,N) cost.

    Being differentiable is the point: it lets the restorer be trained directly
    on "make true neighbours findable" instead of on pixel L1, which converges
    to the conditional mean and smooths away the border microstructure that
    carries the entire adjacency signal.
    """
    if axis == "h":
        a, b = tiles[:, :, :, -cols:], tiles[:, :, :, :cols]
    else:
        a, b = tiles[:, :, -cols:, :], tiles[:, :, :cols, :]
    n_tiles = tiles.shape[0]
    a = a.reshape(n_tiles, 3, -1)
    b = b.reshape(n_tiles, 3, -1)
    n = a.shape[2]
    m2 = ((a ** 2).sum(2)[:, None, :] + (b ** 2).sum(2)[None, :, :]
          - 2 * torch.einsum("icn,jcn->ijc", a, b)) / n
    mu = a.mean(2)[:, None, :] - b.mean(2)[None, :, :]
    return (m2 - mu ** 2 + w * mu ** 2).sum(-1)


def seam_infonce(tiles: torch.Tensor, inv_temp: torch.Tensor, good: torch.Tensor,
                 w: float = 0.03, cols: int = 3, metric: str = "ridge",
                 hard_k: int = 0) -> torch.Tensor:
    """Contrastive seam loss over a whole board given in TRUE grid order.

    `good` is the per-position label-confidence mask.  A row counts only when
    BOTH its anchor and its true neighbour are confidently placed: the Hungarian
    matcher is 0.825 accurate overall, so unmasked rows would teach the model
    adjacencies that do not exist.
    """
    total = tiles.new_zeros(())
    n_used = 0
    metrics = ("ridge", "mgc") if metric == "both" else (metric,)
    for axis, step in (("h", 1), ("v", G)):
      for met in metrics:
          if met == "mgc":
              # Train on the measure we will actually score with.  MGC reads
              # gradients across the seam and reaches bb_prec 0.994 on clean tiles
              # versus 0.796 for the ridge cost, but it is destroyed by residual
              # noise, so the restorer must be optimised for it directly.
              # Both are kept because they dominate in different noise regimes:
              # on real restored tiles ridge scores 0.396 and MGC 0.113, while on
              # clean tiles MGC wins 0.994 to 0.796.
              from mgc import mgc_cost_torch
              cost = mgc_cost_torch(tiles, axis)
          else:
              cost = ridge_cost_torch(tiles, w, cols, axis)
          # Row-standardise before the temperature: raw ridge costs run in the
          # thousands, so an unnormalised scale drives softmax to one-hot and the
          # loss sits far above ln(576) no matter what the model does.
          cost = (cost - cost.mean(1, keepdim=True)) / cost.std(1, keepdim=True).clamp_min(1e-6)
          logits = -cost * inv_temp.exp()
          logits = logits - torch.eye(len(tiles), device=tiles.device) * 1e4   # mask self
          on_grid = torch.tensor(
              [((p % G) != G - 1 if axis == "h" else p < N - G) for p in range(N)],
              device=tiles.device)
          idx = torch.nonzero(on_grid & (good > 0) & torch.roll(good > 0, -step), as_tuple=True)[0]
          if len(idx) == 0:
              continue
          if hard_k and hard_k < len(tiles) - 1:
              # Focus on the confusable tail.  Averaged over all 575 negatives
              # the loss is dominated by tiles that are trivially far away and
              # contribute almost no gradient; restricting it to the hardest few
              # spends every step on the competitors that actually cost us R@1.
              rows = logits[idx]
              keep = rows.topk(hard_k, dim=1).indices
              keep = torch.cat([(idx + step).unsqueeze(1), keep], dim=1)
              sub = torch.gather(rows, 1, keep)
              total = total + F.cross_entropy(
                  sub, torch.zeros(len(idx), dtype=torch.long, device=tiles.device))
          else:
              total = total + F.cross_entropy(logits[idx], idx + step)
          n_used += 1

    return total / max(1, n_used)


def seam_metrics(tiles: np.ndarray, w: float = 0.03, cols: int = 3,
                 metric: str = "ridge") -> dict[str, float]:
    """tiles must be in TRUE grid order.  Reports R@1/R@20/best-buddy precision.

    With metric="both" the two measures are reported side by side and the
    headline figure is the better of them, since the solver is free to use
    whichever wins: ridge dominates on noisy tiles (0.396 vs 0.113) and MGC on
    clean ones (0.994 vs 0.796).
    """
    if metric == "both":
        a = seam_metrics(tiles, w, cols, "ridge")
        b = seam_metrics(tiles, w, cols, "mgc")
        out = {f"ridge_{k}": v for k, v in a.items()}
        out.update({f"mgc_{k}": v for k, v in b.items()})
        for k in ("R1", "R20", "bb_prec"):
            out[k] = max(a[k], b[k])
        return out
    out = {}
    for axis, step, edge in (("h", 1, lambda p: (p % G) != G - 1),
                             ("v", G, lambda p: p < N - G)):
        if metric == "mgc":
            from mgc import mgc_cost
            C = mgc_cost(tiles, axis)
        else:
            C = ridge_cost(tiles, w, cols, axis)
        np.fill_diagonal(C, np.inf)
        rows = np.array([p for p in range(N) if edge(p)])
        order = np.argsort(C[rows], axis=1)
        rank = np.array([np.where(order[k] == rows[k] + step)[0][0] for k in range(len(rows))])
        best_f, best_b = np.argmin(C, 1), np.argmin(C, 0)
        bb = [(i, best_f[i]) for i in range(N) if best_b[best_f[i]] == i]
        ok = sum(1 for i, j in bb if edge(i) and j == i + step)
        out[f"R1_{axis}"] = float((rank == 0).mean())
        out[f"R20_{axis}"] = float((rank < 20).mean())
        out[f"bb_{axis}"] = float(ok / max(1, len(bb)))
    out["R1"] = 0.5 * (out["R1_h"] + out["R1_v"])
    out["R20"] = 0.5 * (out["R20_h"] + out["R20_v"])
    out["bb_prec"] = 0.5 * (out["bb_h"] + out["bb_v"])
    return out
