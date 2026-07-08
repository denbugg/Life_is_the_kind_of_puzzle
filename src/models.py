"""Models: RestoreNet (U-Net) + CompatNet (siamese edge-embeddings) + SSIM losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# SSIM / MS-SSIM losses (differentiable). Metric is uniform win=7; MS-SSIM
# (Gaussian) is a strong training proxy that also lifts the single-scale score.
# ----------------------------------------------------------------------------
def _gauss(win, sigma):
    c = torch.arange(win).float() - (win - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    return (g / g.sum())


def _ssim_map(x, y, win, sigma, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ch = x.shape[1]
    k = _gauss(win, sigma).to(x.device, x.dtype)
    k2 = (k[:, None] * k[None, :]).expand(ch, 1, win, win)
    pad = win // 2
    def filt(z):
        return F.conv2d(F.pad(z, [pad] * 4, mode="reflect"), k2, groups=ch)
    mux, muy = filt(x), filt(y)
    mx2, my2, mxy = mux * mux, muy * muy, mux * muy
    sx = filt(x * x) - mx2
    sy = filt(y * y) - my2
    sxy = filt(x * y) - mxy
    cs = (2 * sxy + C2) / (sx + sy + C2)
    ss = ((2 * mxy + C1) / (mx2 + my2 + C1)) * cs
    return ss.mean([1, 2, 3]), cs.mean([1, 2, 3])


def ssim_loss(x, y, win=11, sigma=1.5):
    ss, _ = _ssim_map(x, y, win, sigma)
    return 1 - ss.mean()


def msssim_loss(x, y, win=11, sigma=1.5,
                weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    w = torch.tensor(weights, device=x.device, dtype=x.dtype)
    levels = len(weights)
    mcs = []
    for i in range(levels):
        ss, cs = _ssim_map(x, y, win, sigma)
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    ssim_last = torch.relu(ss)
    mcs = torch.stack(mcs, 0)              # (levels-1, B)
    pw = w[:-1].unsqueeze(1)
    out = torch.prod(mcs ** pw, 0) * (ssim_last ** w[-1])
    return 1 - out.mean()


def restore_loss(pred, target, alpha=0.84):
    """MS-SSIM + L1 (Zhao et al.). pred/target in [0,1]."""
    return alpha * msssim_loss(pred, target) + (1 - alpha) * F.l1_loss(pred, target)


# ----------------------------------------------------------------------------
# RestoreNet: U-Net with global residual (predicts correction added to input).
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return self.act(h + self.skip(x))


class RestoreNet(nn.Module):
    def __init__(self, base=48, depth=4):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        chs = [base * (2 ** i) for i in range(depth)]           # 48,96,192,384
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

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, s], 1))
        return torch.clamp(x + self.head(h), 0, 1)


# ----------------------------------------------------------------------------
# CompatNet: encode each 20x20 fragment into 4 edge embeddings (R,L,T,B).
# Right-neighbor compat(i,j) = <eR[i], eL[j]>; below compat(i,j) = <eB[i], eT[j]>.
# All-pairs at test = two matrix products -> very fast.
# ----------------------------------------------------------------------------
class CompatNet(nn.Module):
    def __init__(self, dim=160, strip=4, C=96):
        super().__init__()
        self.strip = strip
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.GELU(),
            ConvBlock(48, 64),
            ConvBlock(64, C),
            ConvBlock(C, C),
        )
        flat = 20 * strip * C
        def head():
            return nn.Sequential(nn.Flatten(), nn.Linear(flat, 320), nn.GELU(),
                                 nn.Linear(320, dim))
        self.hR, self.hL, self.hT, self.hB = head(), head(), head(), head()
        self.logit_scale = nn.Parameter(torch.tensor(2.3))   # ~1/temp, learned

    def embed(self, frag):
        """frag: (B,3,20,20) in [0,1]. Returns normalized eR,eL,eT,eB (B,dim)."""
        f = self.backbone(frag)
        s = self.strip
        eR = self.hR(f[:, :, :, -s:])
        eL = self.hL(f[:, :, :, :s])
        eB = self.hB(f[:, :, -s:, :])
        eT = self.hT(f[:, :, :s, :])
        n = lambda e: F.normalize(e, dim=-1)
        return n(eR), n(eL), n(eT), n(eB)


class GNBlock(nn.Module):
    """Residual conv block with GroupNorm (stabilizes the deeper pairwise net and
    helps it stay invariant to the per-fragment brightness/contrast shifts)."""
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(h + self.skip(x))


class PairwiseNet(nn.Module):
    """Sees both fragments straddling the seam (3,20,40) -> compatibility logit.
    v2: SEAM-AWARE (keeps spatial layout instead of global-avg-pooling it away, so
    the model can localize continuity at the seam), GroupNorm, wider (C=96).
    autocast lives INSIDE forward so fp16 survives nn.DataParallel replica threads."""
    def __init__(self, C=96):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1), nn.GroupNorm(8, C), nn.GELU(),
            GNBlock(C, C),
            nn.Conv2d(C, 2 * C, 3, stride=2, padding=1), nn.GroupNorm(8, 2 * C), nn.GELU(),  # 10x20
            GNBlock(2 * C, 2 * C),
            nn.Conv2d(2 * C, 4 * C, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * C), nn.GELU(),  # 5x10
            GNBlock(4 * C, 4 * C),
            nn.Conv2d(4 * C, C, 1), nn.GELU(),      # neck: cheap channel squeeze, keep 5x10 layout
            nn.Flatten(),                            # 5*10*C -> seam position preserved
        )
        self.head = nn.Sequential(nn.Linear(5 * 10 * C, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, pair):        # pair: (B,3,20,40)
        with torch.autocast("cuda", dtype=torch.float16, enabled=pair.is_cuda):
            return self.head(self.body(pair)).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
