"""Preprocessing used only for matching/scoring, not for final restoration."""
import os
import numpy as np
import torch
import torch.nn as nn
from config import CKPT_DIR


class _Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(4, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(4, c),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MatchDenoiser(nn.Module):
    """Tiny residual CNN for denoise-for-matching on 20x20 fragments."""
    def __init__(self, base=32, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.body = nn.Sequential(*[_Block(base) for _ in range(blocks)])
        self.head = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        h = self.body(self.stem(x))
        return torch.clamp(x + self.head(h), 0, 1)


def photometric_normalize_np(frags):
    """Per-tile affine normalization to the image-level RGB mean/std."""
    x = frags.astype(np.float32)
    tile_mean = x.mean(axis=(1, 2), keepdims=True)
    tile_std = x.std(axis=(1, 2), keepdims=True)
    glob_mean = x.mean(axis=(0, 1, 2), keepdims=True)
    glob_std = x.std(axis=(0, 1, 2), keepdims=True)
    y = (x - tile_mean) / (tile_std + 1e-6) * (glob_std + 1e-6) + glob_mean
    return np.clip(y, 0, 255).round().astype(np.uint8)


def photometric_normalize_tensor(frags):
    """Tensor version for (B,N,3,20,20) or (N,3,20,20), values in [0,1]."""
    squeeze = frags.dim() == 4
    if squeeze:
        frags = frags.unsqueeze(0)
    mean = frags.mean(dim=(-1, -2), keepdim=True)
    std = frags.std(dim=(-1, -2), keepdim=True)
    gmean = frags.mean(dim=(1, 3, 4), keepdim=True)
    gstd = frags.std(dim=(1, 3, 4), keepdim=True)
    out = (frags - mean) / (std + 1e-6) * (gstd + 1e-6) + gmean
    out = out.clamp(0, 1)
    return out.squeeze(0) if squeeze else out


def load_match_denoiser(tag="matchden", which="best", device="cuda"):
    for name in (f"{tag}_{which}.pt", f"{tag}_last.pt", f"{tag}_best.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=device)
            m = MatchDenoiser(base=ck.get("base", 32), blocks=ck.get("blocks", 4)).to(device)
            m.load_state_dict(ck["model"])
            m.eval()
            print(f"loaded {name} step={ck.get('step')} val={ck.get('val')}", flush=True)
            return m, ck
    return None, None


@torch.no_grad()
def apply_match_denoiser_np(frags, model, device="cuda", bs=512):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(frags)).permute(0, 3, 1, 2).float().div_(255)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            y = model(xb).float()
        out.append(y.cpu())
    y = torch.cat(out).permute(0, 2, 3, 1).clamp(0, 1).mul(255).round().numpy()
    return y.astype(np.uint8)


def preprocess_frags_np(frags, mode="raw", denoiser=None, device="cuda"):
    if mode == "raw":
        return frags
    if mode == "norm":
        return photometric_normalize_np(frags)
    if mode == "denoise":
        if denoiser is None:
            raise ValueError("mode='denoise' requires a loaded MatchDenoiser")
        return apply_match_denoiser_np(frags, denoiser, device=device)
    if mode == "denoise_norm":
        if denoiser is None:
            raise ValueError("mode='denoise_norm' requires a loaded MatchDenoiser")
        return photometric_normalize_np(apply_match_denoiser_np(frags, denoiser, device=device))
    raise ValueError(f"unknown preprocess mode: {mode}")

