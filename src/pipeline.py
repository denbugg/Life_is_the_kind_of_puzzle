"""Shared inference pipeline: load models, solve + restore a single image."""
import os
import numpy as np
import torch
from config import CKPT_DIR, IMG
from imgio import to_frags, assemble
from models import CompatNet, RestoreNet, PairwiseNet
from solve import solve_image

DEV = "cuda"


def load_compat(tag="compat", which="best"):
    for name in (f"{tag}_{which}.pt", f"{tag}_last.pt", f"{tag}_best.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEV)
            m = CompatNet().to(DEV); m.load_state_dict(ck["model"]); m.eval()
            return m, ck
    raise FileNotFoundError(f"no compat checkpoint for {tag}")


def load_restore(tag="restore", which="best"):
    for name in (f"{tag}_{which}.pt", f"{tag}_last.pt", f"{tag}_best.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEV)
            m = RestoreNet(base=ck.get("base", 48)).to(DEV); m.load_state_dict(ck["model"]); m.eval()
            return m, ck
    return None, None


def _load_pair_ckpt(name):
    p = os.path.join(CKPT_DIR, name)
    if os.path.exists(p):
        ck = torch.load(p, map_location=DEV)
        m = PairwiseNet().to(DEV); m.load_state_dict(ck["model"]); m.eval()
        return m, ck
    return None, None


def load_pair(tag="pair", which="best"):
    """Returns (models, ck): a LIST of PairwiseNets to ensemble. Prefers the
    two-GPU ensemble members pair0/pair1; falls back to a single `pair` model."""
    models, ck0 = [], None
    for t in ("pair0", "pair1"):
        for name in (f"{t}_{which}.pt", f"{t}_last.pt", f"{t}_best.pt"):
            m, ck = _load_pair_ckpt(name)
            if m is not None:
                models.append(m); ck0 = ck0 or ck; break
    if not models:
        for name in (f"{tag}_{which}.pt", f"{tag}_last.pt", f"{tag}_best.pt"):
            m, ck = _load_pair_ckpt(name)
            if m is not None:
                models.append(m); ck0 = ck; break
    if not models:
        return None, None
    print(f"pair ensemble: {len(models)} model(s)", flush=True)
    return models, ck0


@torch.no_grad()
def restore_full(model, img_np):
    """Run restoration on a full 480x480 uint8 image."""
    if model is None:
        return img_np
    x = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255
    with torch.autocast("cuda", dtype=torch.float16):
        y = model(x)
    return (y[0].permute(1, 2, 0).float().clamp(0, 1) * 255).round().cpu().numpy().astype(np.uint8)


def nlm_restore(img_np, h=10):
    """Classical Non-Local-Means denoise. Measured to lift GT-assembled SSIM
    0.447 -> ~0.57 (h~10-12), far above the current undertrained RestoreNet."""
    import cv2
    return cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(img_np), None, h, h, 7, 21)


def restore_apply(restore, img_np, nlm=False, h=10):
    return nlm_restore(img_np, h) if nlm else restore_full(restore, img_np)


def process(frags_np, compat, restore, solve_kw=None, pair=None, rescore_kw=None,
            nlm=False, h=10):
    solve_kw = solve_kw or {}
    place, R, D, v = solve_image(frags_np, compat, DEV, pair_model=pair,
                                 rescore_kw=rescore_kw, **solve_kw)
    assembled = assemble(frags_np, place)
    out = restore_apply(restore, assembled, nlm=nlm, h=h)
    return out, place, assembled
