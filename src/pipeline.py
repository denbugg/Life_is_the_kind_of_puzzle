"""Shared inference pipeline: load models, solve + restore a single image."""
import os
import numpy as np
import torch
from config import CKPT_DIR, IMG
from imgio import to_frags, assemble
from models import CompatNet, RestoreNet
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


@torch.no_grad()
def restore_full(model, img_np):
    """Run restoration on a full 480x480 uint8 image."""
    if model is None:
        return img_np
    x = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255
    with torch.autocast("cuda", dtype=torch.float16):
        y = model(x)
    return (y[0].permute(1, 2, 0).float().clamp(0, 1) * 255).round().cpu().numpy().astype(np.uint8)


def process(frags_np, compat, restore, solve_kw=None):
    solve_kw = solve_kw or {}
    place, R, D, v = solve_image(frags_np, compat, DEV, **solve_kw)
    assembled = assemble(frags_np, place)
    out = restore_full(restore, assembled)
    return out, place, assembled
