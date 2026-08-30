"""A conditional diffusion model of the PICTURE, conditioned on the bag.

Why this exists
---------------
M471 named the target of this family in one number. Cells described at 4x4 per
fragment and fragments assigned by Hungarian: the true map places 260 fragments
and scores SSIM 0.4292, the same map carrying 32 RMSE of noise still places 210
and scores 0.3902 -- above the 0.38 a competitor reached, with no seam evidence
anywhere. The tolerance is enormous against a signal whose own spread is 50.7.

The same experiment killed every shortcut to it. A stranger's photograph places
0.0030 where chance is 0.0017, and the mean of forty photographs places 0.0007,
so a generic prior carries NOTHING and the map must be predicted from THIS bag.

M387 measured our existing field at spread 2.05 against the true cells' 57.37 --
a nearly constant map, which places at 0.0027 despite a better RMSE than noise
that places at 0.1008. That collapse is not a capability limit, it is what the
LOSS demands: a regression trained with MSE converges to the conditional mean of
the picture given the bag, and the mean of many photographs is flat. A sampler
does not average its posterior, so it is sharp by construction, and sharpness is
what assignment needs.

`prior_projection.py` then measured the classical form of the same idea and it
went nowhere: alternating a blur or a total-variation prior with the Hungarian
projection leaves a random arrangement exactly at chance for fifteen rounds,
while a mild prior started from the truth keeps 0.84 of it. The controls are
clean, and the reading is a design requirement rather than a refutation -- a
REGULARISING prior only destroys structure, so the arrangement must come from a
prior that CREATES it. That is what a generative model is for.

What it is
----------
    bag       576 corrupted fragments, unordered, through a permutation
              invariant transformer -- no positional encoding anywhere, so the
              conditioning cannot leak the answer
    picture   the clean image at 96x96, which is 4x4 per cell, denoised by a
              small U-Net that cross-attends to the bag at its coarse levels

Trained as an ordinary epsilon-prediction DDPM. At sampling time the bag is a
hard constraint the picture must satisfy exactly, and `project` enforces it by
Hungarian assignment, which is the discrete analogue of the data-consistency
step in a diffusion solver for inverse problems.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

GRID = 24            # cells a side
SUB = 4              # pixels a cell in the map -- M428's resolution
RES = GRID * SUB     # 96
N = GRID * GRID


# ---------------------------------------------------------------- schedule
def cosine_betas(T, s=0.008):
    """Nichol and Dhariwal's cosine schedule; flat noise near the ends."""
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    a = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = a / a[0]
    return torch.clip(1 - a[1:] / a[:-1], 0, 0.999).float()


class Schedule:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.betas = cosine_betas(T).to(device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.sqrt_abar = self.abar.sqrt()
        self.sqrt_one_minus = (1 - self.abar).sqrt()

    def add_noise(self, x0, t, eps):
        return (self.sqrt_abar[t][:, None, None, None] * x0
                + self.sqrt_one_minus[t][:, None, None, None] * eps)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device)
                      / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


# ------------------------------------------------------------ bag encoder
class BagEncoder(nn.Module):
    """The 576 fragments as an unordered set.

    There is no positional encoding and no ordering anywhere in this module, so
    self-attention over the tokens is exactly permutation invariant and the
    fragments' storage order -- which in the caches happens to be the true cell
    order -- cannot reach the model.
    """

    def __init__(self, d=192, layers=4, heads=6, view=8):
        super().__init__()
        self.view = view
        self.inp = nn.Linear(view * view * 3 + 6, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, batch_first=True,
            norm_first=True, dropout=0.0, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.LayerNorm(d)

    def features(self, tiles):
        """(B, N, 20, 20, 3) in [0,255] to a per-fragment description."""
        b, n = tiles.shape[:2]
        x = tiles.permute(0, 1, 4, 2, 3).reshape(b * n, 3, 20, 20) / 127.5 - 1.0
        v = F.adaptive_avg_pool2d(x, self.view).reshape(b, n, -1)
        m = x.mean((2, 3)).reshape(b, n, 3)
        s = x.std((2, 3)).reshape(b, n, 3)
        return torch.cat([v, m, s], -1)

    def forward(self, tiles):
        return self.out(self.enc(self.inp(self.features(tiles))))


# ------------------------------------------------------------------ U-Net
class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(t)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CrossAttn(nn.Module):
    """Pixels attend to the bag. This is the only path the content takes in."""

    def __init__(self, ch, d, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.kv = nn.Linear(d, 2 * ch)
        self.o = nn.Conv2d(ch, ch, 1)
        self.heads = heads
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, ctx):
        b, c, h, w = x.shape
        q = self.q(self.norm(x)).reshape(b, self.heads, c // self.heads, h * w)
        k, v = self.kv(ctx).chunk(2, -1)
        k = k.reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(b, -1, self.heads, c // self.heads).permute(0, 2, 1, 3)
        a = torch.softmax(q.transpose(2, 3) @ k / math.sqrt(c // self.heads), -1)
        y = (a @ v).transpose(2, 3).reshape(b, c, h, w)
        return x + self.o(y)


class FieldUNet(nn.Module):
    def __init__(self, d=192, base=64, tdim=256):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(),
                                  nn.Linear(tdim, tdim))
        c1, c2, c3 = base, base * 2, base * 3
        self.inp = nn.Conv2d(3, c1, 3, padding=1)
        self.d1 = ResBlock(c1, c1, tdim)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)   # 96 -> 48
        self.d2 = ResBlock(c1, c2, tdim)
        self.x2 = CrossAttn(c2, d)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)   # 48 -> 24
        self.d3 = ResBlock(c2, c3, tdim)
        self.x3 = CrossAttn(c3, d)
        self.mid1 = ResBlock(c3, c3, tdim)
        self.xm = CrossAttn(c3, d)
        self.mid2 = ResBlock(c3, c3, tdim)
        self.u3 = ResBlock(c3 + c3, c2, tdim)
        self.x4 = CrossAttn(c2, d)
        self.u2 = ResBlock(c2 + c2, c1, tdim)
        self.u1 = ResBlock(c1 + c1, c1, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(),
                                 nn.Conv2d(c1, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, ctx):
        e = self.temb(timestep_embedding(t, self.tdim))
        h1 = self.d1(self.inp(x), e)
        h2 = self.x2(self.d2(self.down1(h1), e), ctx)
        h3 = self.x3(self.d3(self.down2(h2), e), ctx)
        m = self.mid2(self.xm(self.mid1(h3, e), ctx), e)
        u = self.u3(torch.cat([m, h3], 1), e)
        u = self.x4(u, ctx)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u2(torch.cat([u, h2], 1), e)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u1(torch.cat([u, h1], 1), e)
        return self.out(u)


class FieldDiffusion(nn.Module):
    def __init__(self, d=192, layers=4, heads=6, base=64, view=8):
        super().__init__()
        self.bag = BagEncoder(d, layers, heads, view)
        self.net = FieldUNet(d, base)

    def forward(self, x, t, tiles=None, ctx=None):
        if ctx is None:
            ctx = self.bag(tiles)
        return self.net(x, t, ctx)


# ------------------------------------------------------------- projection
def cell_desc(img):
    """(RES, RES, 3) picture to (N, SUB*SUB*3) cell descriptors."""
    a = img.reshape(GRID, SUB, GRID, SUB, 3).transpose(0, 2, 1, 3, 4)
    return a.reshape(N, -1)


def frag_desc(frags):
    """(N, 20, 20, 3) fragments to the same description by block averages."""
    s = frags.shape[1] // SUB
    a = frags.reshape(len(frags), SUB, s, SUB, s, 3).mean((2, 4))
    return a.reshape(len(frags), -1)


def project(img, frags):
    """The assignment of fragments to cells that best explains a picture.

    This is the data-consistency step: whatever the prior imagines, the answer
    must be a permutation of the bag we were actually given.
    """
    A, B = cell_desc(img), frag_desc(frags)
    C = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T)
    r, c = linear_sum_assignment(C)
    order = np.empty(N, np.int64)
    order[r] = c
    return order


def render(frags, order):
    """The 96x96 picture implied by an assignment, for the next prior step."""
    img = np.zeros((RES, RES, 3), np.float32)
    blocks = frag_desc(frags)[order].reshape(N, SUB, SUB, 3)
    for cell in range(N):
        y, x = divmod(cell, GRID)
        img[y * SUB:(y + 1) * SUB, x * SUB:(x + 1) * SUB] = blocks[cell]
    return img


@torch.no_grad()
def sample(model, tiles, sched, steps=100, frags=None, snap_from=0.0,
           device="cuda", generator=None):
    """DDIM, optionally snapping onto the bag once the picture has form.

    `snap_from` is the fraction of the schedule after which each step's clean
    estimate is replaced by the picture its own best assignment would draw. Early
    steps are left free: the constraint is exact but uninformative while the
    estimate is still noise, and imposing it then only locks in an arbitrary
    permutation, which is what `prior_projection.py` measured.
    """
    b = tiles.shape[0]
    ctx = model.bag(tiles)
    x = torch.randn(b, 3, RES, RES, device=device, generator=generator)
    ts = torch.linspace(sched.T - 1, 0, steps).long().to(device)
    for k, t in enumerate(ts):
        tt = t.expand(b)
        eps = model(x, tt, ctx=ctx)
        ab = sched.abar[t]
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x0 = x0.clamp(-1, 1)
        if frags is not None and k / max(len(ts) - 1, 1) >= snap_from:
            for i in range(b):
                img = (x0[i].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                o = project(img, frags[i])
                x0[i] = torch.from_numpy(
                    render(frags[i], o) / 127.5 - 1.0).permute(2, 0, 1).to(
                        device, x0.dtype)
        if k + 1 < len(ts):
            ab_next = sched.abar[ts[k + 1]]
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        else:
            x = x0
    return x
