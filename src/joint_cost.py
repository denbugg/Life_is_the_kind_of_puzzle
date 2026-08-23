"""Cost matrices from the joint seam scorer, for the island harvest.

The harvest is the binding constraint on the island route.  Loop closure turns
tile-level edges into 2x2 blocks and the yield depends steeply on the edges it
is fed, because four independent judgements have to coincide: a matcher that is
14% better per edge is far more than 14% better per quad.

The retriever alone supplies mutual-edge precision 0.438.  The joint scorer over
its own trunk reaches 0.494 (seam_joint_v1, 5000 steps) and has never been
pointed at the harvest -- it was built to be read as a number, not to produce
costs.  This produces them, following the same shape as `rerank_fuse`: the
retriever's calibrated scores everywhere, the joint scorer's opinion spliced
into the shortlist it was trained on, then cycle consistency over the result.

The blend is deliberate.  The joint scorer only ever saw rows whose true
neighbour made the shortlist, so on rows where it did not, its confidence is
unearned; keeping part of the retriever's opinion caps the damage there.
"""
from __future__ import annotations

import numpy as np
import torch

from seam_cost import cycle_consistency
from seam_embed import board_logits


@torch.no_grad()
def calibrated(frozen, tiles, rounds=3, weight=0.35):
    """The retriever's scores AFTER Sinkhorn and cycle consistency.

    This is not a detail.  The joint head takes the retriever's score as an
    input channel, and it was trained on the CALIBRATED score; feeding it the
    raw logits instead is out of distribution and destroys it -- measured, not
    guessed: edge precision came out at 0.318 against the retriever's own 0.449,
    and at blend 1.0 it collapsed to 0.053.
    """
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in frozen(tiles)[:4]]
    scale = frozen.logit_scale.exp().detach()
    lg = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax).float() * scale
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1], rounds, weight)
    H, V = H.clone(), V.clone()
    H.fill_diagonal_(-1e4)
    V.fill_diagonal_(-1e4)
    return {"h": H, "v": V}


@torch.no_grad()
def joint_logits(model, frozen, tiles, k=20, chunk=24, blend=0.7):
    """{axis: (n,n) log-score} with shortlist entries re-scored by the joint head."""
    n = tiles.shape[0]
    cal = calibrated(frozen, tiles)
    feats = model.features(tiles)
    out = {}
    for axis in ("h", "v"):
        S = cal[axis].clone()
        rows = torch.arange(n, device=tiles.device)
        cand = S.topk(k, dim=1).indices
        li = rows.repeat_interleave(k)
        ri = cand.reshape(-1)
        rsc = S.gather(1, cand).reshape(-1)

        sc = []
        for i in range(0, li.numel(), chunk * k):
            sl = slice(i, i + chunk * k)
            with torch.autocast("cuda", torch.float16):
                sc.append(model(feats, li[sl], ri[sl], axis, rsc[sl]))
        sc = torch.cat(sc).float().reshape(n, k)

        # put the joint head's row on the retriever's scale so untouched
        # entries stay comparable with the ones we replaced
        old = S.gather(1, cand)
        sc = ((sc - sc.mean(1, keepdim=True)) / (sc.std(1, keepdim=True) + 1e-6)
              * old.std(1, keepdim=True) + old.mean(1, keepdim=True))
        S = S.scatter(1, cand, blend * sc + (1.0 - blend) * old)
        S.fill_diagonal_(-1e4)
        out[axis] = S
    return out


@torch.no_grad()
def costs_from_joint(model, frozen, tiles_np, k=20, rounds=0, weight=0.35,
                     blend=0.7, device="cuda"):
    """(n,20,20,3) float tiles -> (cost_h, cost_v) numpy, lower is better."""
    tiles = torch.from_numpy(np.ascontiguousarray(tiles_np)).permute(0, 3, 1, 2).to(device)
    lg = joint_logits(model, frozen, tiles, k=k, blend=blend)
    # `joint_logits` splices into ALREADY calibrated scores, so consistency has
    # been applied once; `rounds > 0` re-applies it on top of the joint head's
    # opinion, which is a different thing and is left off by default
    if rounds:
        H, V = cycle_consistency(lg["h"], lg["v"], rounds, weight)
    else:
        H, V = lg["h"], lg["v"]
    out = []
    for L in (H, V):
        C = (-L).cpu().numpy()
        C -= C.min()
        np.fill_diagonal(C, 0.0)
        out.append(np.ascontiguousarray(C))
    return out


def load_joint(path, device="cuda"):
    """Return (joint model, frozen retriever) ready for `costs_from_joint`."""
    from pathlib import Path

    from config import CKPT_DIR
    from seam_embed import SeamEmbed
    from seam_joint import SeamJoint

    ck = torch.load(Path(CKPT_DIR) / path, map_location=device, weights_only=False)
    a = ck["args"]
    rk = torch.load(Path(CKPT_DIR) / a["retriever"], map_location=device,
                    weights_only=False)
    ta = rk["args"]

    def build():
        m = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                      ta.get("head", "global")).to(device)
        m.load_state_dict(rk["model"])
        m.eval()
        return m

    trunk, frozen = build(), build()
    model = SeamJoint(trunk, a["strip"], a["ch"], a["blocks"],
                      freeze_trunk=True).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    return model, frozen, ck
