"""A DRUNet-style denoiser of the 96x96 PICTURE, for use as a plug-and-play prior.

What it is for, and what it is NOT for
--------------------------------------
Not restoration. That axis is closed by two bounds rather than by a plateau:
M301 leaves denoising 0.31 dB of headroom and M302 shows the per-fragment affine
is predictable only to about a tenth of its error from a fragment alone. Nothing
here is meant to touch a submitted pixel, and SSIM is not measured.

It exists because of one measurement. `prior_projection.py` alternated an image
prior with the Hungarian projection onto the bag and a random arrangement stayed
EXACTLY at chance for fifteen rounds under every classical prior -- blur at three
widths, total variation at two weights -- while the same priors started from the
truth kept 0.84 of it. The controls were clean and the reading was a design
requirement: a REGULARISING prior only destroys structure, so the arrangement
has to come from a prior that CREATES it. A denoiser run at a large noise level
is exactly that; it is the object diffusion samplers are built from.

Trained on OUR corruption and OUR pictures on purpose. M471 measured that a
generic photograph prior is worth nothing for placement -- a stranger's picture
places 0.0030 against a chance of 0.0017, and the mean of forty places 0.0007 --
so a downloaded general-purpose denoiser would be the thing already measured
dead. The 7000 boards give exactly the matched pairs this needs.

Design follows Zhang et al.'s plug-and-play denoiser prior: the noise level is
handed to the network as an extra input plane, so one model serves the whole
annealing schedule instead of one model per level.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x), inplace=True))


class MapDenoiser(nn.Module):
    """Four channels in -- the picture and its noise level -- three out.

    Bias-free convolutions throughout. A bias-free network is exactly
    scale-equivariant in the input, which is what lets one set of weights hold
    across an annealing schedule that spans two orders of magnitude of noise
    instead of drifting at the ends.
    """

    def __init__(self, base=48, blocks=2):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.head = nn.Conv2d(4, c1, 3, padding=1, bias=False)
        self.e1 = nn.Sequential(*[ResBlock(c1) for _ in range(blocks)])
        self.d1 = nn.Conv2d(c1, c2, 2, stride=2, bias=False)
        self.e2 = nn.Sequential(*[ResBlock(c2) for _ in range(blocks)])
        self.d2 = nn.Conv2d(c2, c3, 2, stride=2, bias=False)
        self.mid = nn.Sequential(*[ResBlock(c3) for _ in range(blocks + 1)])
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2, bias=False)
        self.f2 = nn.Sequential(*[ResBlock(c2) for _ in range(blocks)])
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2, bias=False)
        self.f1 = nn.Sequential(*[ResBlock(c1) for _ in range(blocks)])
        self.tail = nn.Conv2d(c1, 3, 3, padding=1, bias=False)

    def forward(self, x, sigma):
        """x is (B, 3, H, W) in [-1, 1]; sigma is (B,) on the same scale."""
        s = sigma.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        h1 = self.e1(self.head(torch.cat([x, s], 1)))
        h2 = self.e2(self.d1(h1))
        m = self.mid(self.d2(h2))
        u = self.f2(self.u2(m) + h2)
        u = self.f1(self.u1(u) + h1)
        return x - self.tail(u)          # residual: the network predicts noise
