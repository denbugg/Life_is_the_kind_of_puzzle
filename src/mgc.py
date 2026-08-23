"""Mahalanobis Gradient Compatibility (Gallagher, CVPR 2012).

The repo dismissed classical gradient compatibility as "bb ~= 0.05", but that
was measured on the fully corrupted tiles, where nothing survives.  The open
question is different: our plain ridge/L2 seam cost tops out at R@1 = 0.788 on
PERFECTLY CLEAN tiles, and every solver built on it stalls, whereas published
jigsaw solvers assemble reliably from R@1 ~0.9-0.95.  So before blaming the
solver we should check whether the compatibility measure itself is the limit.

MGC predicts the colour gradient across the seam from the gradient INSIDE each
tile, and penalises deviation under the Mahalanobis metric of that tile's own
gradient distribution.  This normalises by local texture, which plain L2 does
not: a busy tile and a flat tile are otherwise incomparable.
"""
from __future__ import annotations

import numpy as np


def _dir_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """cost[i,j] for tile j placed to the RIGHT of tile i.

    left/right are the facing 2-column strips: left[i] = last two columns of i
    (as [..., -2], [..., -1]), right[j] = first two columns of j.
    """
    n = len(left)
    # expected gradient just past tile i's right edge, and its covariance
    gi = left[:, :, 1, :] - left[:, :, 0, :]                    # (n, rows, 3)
    mu_i = gi.mean(1)                                           # (n, 3)
    cov_i = np.einsum("nrc,nrd->ncd", gi - mu_i[:, None], gi - mu_i[:, None]) / gi.shape[1]
    cov_i += np.eye(3) * 1.0                                    # ridge, keeps it invertible
    inv_i = np.linalg.inv(cov_i)                                # (n, 3, 3)
    # same, looking leftwards out of tile j
    gj = right[:, :, 0, :] - right[:, :, 1, :]
    mu_j = gj.mean(1)
    cov_j = np.einsum("nrc,nrd->ncd", gj - mu_j[:, None], gj - mu_j[:, None]) / gj.shape[1]
    cov_j += np.eye(3) * 1.0
    inv_j = np.linalg.inv(cov_j)

    # actual gradient across the seam, for every ordered pair
    a_edge = left[:, :, 1, :]                                   # (n, rows, 3) last col of i
    b_edge = right[:, :, 0, :]                                  # (n, rows, 3) first col of j
    d_ij = b_edge[None, :, :, :] - a_edge[:, None, :, :]         # (i, j, rows, 3)
    r_ij = d_ij - mu_i[:, None, None, :]
    cost_ij = np.einsum("ijrc,icd,ijrd->ij", r_ij, inv_i, r_ij)
    r_ji = -d_ij - mu_j[None, :, None, :]
    cost_ji = np.einsum("ijrc,jcd,ijrd->ij", r_ji, inv_j, r_ji)
    return (cost_ij + cost_ji) / d_ij.shape[2]


def mgc_cost(tiles: np.ndarray, axis: str = "h") -> np.ndarray:
    """tiles (N,20,20,3) -> cost[i,j], lower = better. axis 'h': j right of i."""
    t = tiles.astype(np.float64)
    if axis == "v":
        t = t.transpose(0, 2, 1, 3)                             # vertical seam -> horizontal
    left = np.stack([t[:, :, -2, :], t[:, :, -1, :]], axis=2)   # (N, rows, 2, 3)
    right = np.stack([t[:, :, 0, :], t[:, :, 1, :]], axis=2)
    return _dir_cost(left, right)


def mgc_cost_torch(tiles: "torch.Tensor", axis: str = "h", ridge: float = 1.0) -> "torch.Tensor":
    """Differentiable MGC. tiles: (N,3,20,20) -> (N,N) cost, lower = better.

    Needed because the restorer must be trained on the measure it will actually
    be scored with.  Trained against the plain ridge seam cost it optimises
    pixel VALUES, while MGC reads pixel GRADIENTS -- which is why a restorer
    good for ridge (bb 0.396) is useless for MGC (bb 0.113), even though MGC
    reaches 0.994 on clean tiles versus ridge's 0.796.
    """
    import torch

    t = tiles.permute(0, 2, 3, 1)                               # (N,20,20,3)
    if axis == "v":
        t = t.permute(0, 2, 1, 3)
    n, rows = t.shape[0], t.shape[1]
    eye = torch.eye(3, device=t.device, dtype=t.dtype) * ridge

    def stats(g):
        mu = g.mean(1)
        d = g - mu[:, None]
        cov = torch.einsum("nrc,nrd->ncd", d, d) / g.shape[1] + eye
        return mu, torch.linalg.inv(cov)

    mu_i, inv_i = stats(t[:, :, -1, :] - t[:, :, -2, :])
    mu_j, inv_j = stats(t[:, :, 0, :] - t[:, :, 1, :])
    d_ij = t[None, :, :, 0, :] - t[:, None, :, -1, :]           # (i,j,rows,3)
    r_ij = d_ij - mu_i[:, None, None, :]
    c_ij = torch.einsum("ijrc,icd,ijrd->ij", r_ij, inv_i, r_ij)
    r_ji = -d_ij - mu_j[None, :, None, :]
    c_ji = torch.einsum("ijrc,jcd,ijrd->ij", r_ji, inv_j, r_ji)
    return (c_ij + c_ji) / rows
