"""Small, label-aware metrics for the Canvas-first prototype."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import GRID, NFRAG
from placement_metrics import neighbour_accuracy


def canvas_patches(canvas: torch.Tensor, patch: int) -> torch.Tensor:
    """``(B,3,grid*p,grid*p)`` canvas -> ``(B,576,3*p*p)`` descriptors."""
    b, c, h, w = canvas.shape
    if c != 3 or h != GRID * patch or w != GRID * patch:
        raise ValueError(f"expected (B,3,{GRID * patch},{GRID * patch}), got {tuple(canvas.shape)}")
    return (canvas.reshape(b, c, GRID, patch, GRID, patch)
                  .permute(0, 2, 4, 1, 3, 5)
                  .reshape(b, NFRAG, c * patch * patch))


def symmetric_assignment_ce(logits: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Row and column CE for logits indexed as ``tile input -> clean slot``."""
    b, n, m = logits.shape
    if n != NFRAG or m != NFRAG:
        raise ValueError(f"expected 576x576 logits, got {tuple(logits.shape)}")
    row = torch.nn.functional.cross_entropy(logits.reshape(b * n, m), perm.reshape(b * n))
    inv = torch.argsort(perm, dim=1)
    col = torch.nn.functional.cross_entropy(logits.transpose(1, 2).reshape(b * n, m), inv.reshape(b * n))
    return 0.5 * (row + col)


def hard_assignment(logits: torch.Tensor) -> np.ndarray:
    """Hungarian decode of one tile-to-slot score matrix -> ``place[slot]=tile``."""
    x = logits.detach().float().cpu().numpy()
    if x.shape != (NFRAG, NFRAG):
        raise ValueError(f"expected (576,576), got {x.shape}")
    rows, cols = linear_sum_assignment(-x)
    place = np.empty(NFRAG, dtype=np.int64)
    place[cols] = rows
    return place


def rank_summary(logits: torch.Tensor, perm: torch.Tensor) -> dict[str, float]:
    """R@1/R@5/R@20 and true-slot median rank for a batch."""
    order = logits.argsort(dim=-1, descending=True)
    target = perm[..., None]
    ranks = (order == target).nonzero(as_tuple=False)[:, -1].float() + 1
    return {
        "r1": float((ranks <= 1).float().mean()),
        "r5": float((ranks <= 5).float().mean()),
        "r20": float((ranks <= 20).float().mean()),
        "median_rank": float(ranks.median()),
    }


def decoded_geometry(logits: torch.Tensor, perm: torch.Tensor) -> dict[str, float]:
    """Exact synthetic placement and neighbour scores after Hungarian decoding."""
    places, place_acc, neigh = [], [], []
    for score, p in zip(logits, perm):
        place = hard_assignment(score)
        inv = np.argsort(p.detach().cpu().numpy().astype(np.int64))
        places.append(place)
        place_acc.append(float(np.mean(place == inv)))
        neigh.append(neighbour_accuracy(place, inv)[0])
    return {
        "place_acc": float(np.mean(place_acc)),
        "neighbour_acc": float(np.mean(neigh)),
        "places": places,
    }
