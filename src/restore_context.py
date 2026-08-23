"""Context-aware tile restoration: denoise a tile using its placed neighbours.

Why
---
A lone 20x20 fragment is a hard denoising target: at residual sigma ~13 there is
simply not enough surrounding signal, and non-local methods have nothing to
average over.  Measured consequence: the context-free restorer plateaus around
ridge bb_prec 0.40, which maps to roughly 0.30 SSIM -- above the current
submission but short of the 0.64 placement (0.41 SSIM) the solver reaches on
clean-blur quality tiles.

But placement and restoration feed each other.  Once a partial layout exists,
a tile with two or three correct neighbours can be denoised over a 60x60
window instead of 20x20.  That is a different problem, and a much easier one.

Loop:
    context-free restore -> solve (~0.3-0.4) -> context restore -> re-solve

The circle closes because step 1 only has to reach ~0.3, not ~0.9, and every
iteration hands the next one cleaner input.

The network takes a 3x3 tile neighbourhood with a validity mask, so it degrades
gracefully: missing neighbours are zeroed and flagged, and the model still works
when the centre tile stands alone (the mask is all-zero except the centre).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FS
from restore_tile import ResBlock


class ContextRestorer(nn.Module):
    """(3x3 tile block + neighbour mask) -> clean centre tile.

    Input  (B, 4, 60, 60): RGB of the 3x3 block plus one plane marking which of
                           the nine slots actually carry a placed tile.
    Output (B, 3, 20, 20): the restored centre tile.
    """

    def __init__(self, ch: int = 64, blocks: int = 6):
        super().__init__()
        self.stem = nn.Conv2d(4, ch, 3, padding=1, padding_mode="reflect")
        dil = [1, 2, 3, 2, 1, 1][:blocks] + [1] * max(0, blocks - 6)
        self.body = nn.Sequential(*[ResBlock(ch, d) for d in dil])
        self.head = nn.Conv2d(ch, 3, 3, padding=1, padding_mode="reflect")
        self.glob = nn.Sequential(
            nn.Linear(ch + 6, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 6),
        )

    def forward(self, block: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """block: (B,3,60,60) in [0,255]; mask: (B,1,60,60) in {0,1}."""
        centre = block[:, :, FS:2 * FS, FS:2 * FS]
        mean = centre.mean(dim=(2, 3), keepdim=True)
        std = centre.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        # Normalise the whole block by the CENTRE statistics: neighbours carry
        # their own independent a,b, and the mask lets the net discount them.
        xn = (block - mean) / std
        h = self.body(self.stem(torch.cat([xn * mask, mask], dim=1)))
        h = h[:, :, FS:2 * FS, FS:2 * FS]
        struct = self.head(h) + xn[:, :, FS:2 * FS, FS:2 * FS]

        pooled = h.mean(dim=(2, 3))
        stats = torch.cat([mean.flatten(1) / 128.0, std.flatten(1) / 64.0], dim=1)
        g = self.glob(torch.cat([pooled, stats], dim=1))
        out_mean = mean.flatten(1) + 30.0 * torch.tanh(g[:, :3])
        out_std = std.flatten(1) * torch.exp(g[:, 3:].clamp(-1.5, 1.5))

        struct = struct - struct.mean(dim=(2, 3), keepdim=True)
        struct = struct / struct.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
        return struct * out_std[:, :, None, None] + out_mean[:, :, None, None]


def build_blocks(tiles: np.ndarray, board: np.ndarray, grid: int,
                 keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Assemble per-position 3x3 neighbourhoods from a (partial) layout.

    tiles: (N,20,20,3);  board[p] = tile index at position p, -1 if empty.
    keep:  optional bool per position, marks neighbours trusted enough to use.
    Returns block (N,60,60,3) and mask (N,60,60,1).
    """
    n = grid * grid
    block = np.zeros((n, 3 * FS, 3 * FS, 3), np.float32)
    mask = np.zeros((n, 3 * FS, 3 * FS, 1), np.float32)
    for p in range(n):
        r0, c0 = divmod(p, grid)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < grid and 0 <= c < grid):
                    continue
                q = r * grid + c
                t = board[q]
                if t < 0:
                    continue
                if keep is not None and (dr or dc) and not keep[q]:
                    continue
                ys, xs = (dr + 1) * FS, (dc + 1) * FS
                block[p, ys:ys + FS, xs:xs + FS] = tiles[t]
                mask[p, ys:ys + FS, xs:xs + FS] = 1.0
    return block, mask


# Memory note: the 3x3 block input is 60x60, nine times the pixels of a bare
# tile, so activations scale accordingly.  On the 8 GB card ch=96 with
# tile_batch=192 does not fit and silently spills into WDDM shared memory --
# training then emits no step at all.  Keep ch*tile_batch modest (ch=64 with
# tile_batch=64 is comfortable) and watch that nvidia-smi stays well under
# 7 GB.
