"""Restoration anchored on the predicted colour field, not on the assembly.

Why the anchor matters more than the architecture
-------------------------------------------------
`RestoreNet` adds its correction to its own input, and when that input is our
assembled board the model starts at -0.25 over the flat fill (M145) and spends
its whole budget climbing back out.  Six thousand steps got it to -0.030
(M147), still below a constant.

The coarse field starts at +0.0159 (M144 CAL).  Anchoring the residual there
instead means step 0 already sits at the best arm measured so far, and training
can only add.  The assembled board is still supplied -- as extra input channels
-- because it is the only place the real image structure lives; the model just
is not forced to treat it as the answer.

    output = field + head(unet([board, field]))

with the head zero-initialised, so an untrained model reproduces the field
exactly and any gain is unambiguous.  Same discipline as M142's blind twin and
M144's zero-residual rendering contract.

The honest-submission test travels with it: swap the input for another board's
and the output has to change.  A fill fails that by construction (M146).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models import ConvBlock


class FieldRestore(nn.Module):
    """(board, field) -> restored image, as a residual above the field."""

    def __init__(self, base=48, depth=4, in_ch=6):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth - 1):
            self.enc.append(ConvBlock(chs[i], chs[i]))
            self.down.append(nn.Conv2d(chs[i], chs[i + 1], 2, stride=2))
        self.mid = ConvBlock(chs[-1], chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(base, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, board, field, clamp=True):
        h = self.stem(torch.cat([board, field], 1))
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, s], 1))
        out = field + self.head(h)
        return torch.clamp(out, 0, 1) if clamp else out
