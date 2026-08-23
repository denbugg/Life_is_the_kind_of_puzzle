"""Train the 2x2 verifier on the question it is actually asked, as a RESIDUAL.

Two design faults sank the binary version, and the first eval measured both.

Wrong objective.  Binary training asks "is this quad real", one at a time; the
harvest asks "which of these fifty candidates is real, if any".  A separation
model says a scorer with AUC around 0.8 picks the true quad out of fifty about
17% of the time, and the trained net scored 0.177 of covered anchors -- the
prediction was exact, so the ceiling was the objective, not the architecture.
The loss here is a softmax over the candidate set the harvest actually builds,
the same objective that made the retriever work (M79).

Too little information.  M107's rule is that a second stage must never see less
than the stage it corrects, and the plain verifier sees pixels where the
pairwise sum sees the matcher's learned scores -- it picked 0.177 against the
sum's 0.560.  The fix is not to feed it the sum as an input, which M107 also
measured and which teaches copying, but to ADD it: the final score is

    pairwise_sum (standardised per anchor)  +  w * plaquette(quad)

What it has to learn is what four independent seam scores cannot express: the
junction where four corners meet.

The correction is NOT gated by a learned scalar.  That was the first attempt and
it walked into this repo's oldest trap -- a gate initialised at zero on a signal
the model can ignore stays at zero, because the network's gradient is scaled by
the gate and the gate's gradient needs the network to already be useful.  After
1000 steps the gate read -0.025, the loss had not moved and the pick rate sat
below the baseline it started from.  The correction is added at unit weight
instead, standardised per anchor so the two terms are commensurable: the model
starts BELOW the pairwise sum, which is the honest starting point, and has to
earn its way past it with a full gradient from the first step.

Abstention is a threshold, not a class.  Only a quarter of anchors have their
true quad in the candidate set at all, and a NONE class over that distribution
is a trap: the first attempt reached "pick 0.7639" at cover 0.2361, which is
exactly 1 - 0.2361, because always answering NONE is the local optimum and the
metric was counting it as a win.  Training runs on covered anchors only, and the
harvest decides what to accept from the margin -- the same way the pairwise sum
is used, which keeps the comparison honest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CACHE_DIR, CKPT_DIR, GRID as G, TRAIN_INP
from plaquette import PlaquetteNet, count_params
from restore_tile import to_frags
from seam_cost import cycle_consistency
from seam_embed import SeamEmbed, board_logits

N = G * G
ANCHORS = np.array([i for i in range(N) if i % G != G - 1 and i < N - G])


def load_rgb(path):
    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


def load_matcher(name, dev):
    ck = torch.load(Path(CKPT_DIR) / name, map_location=dev, weights_only=False)
    a = ck["args"]
    m = SeamEmbed(a["ch"], a["blocks"], a["dim"], a["strip"],
                  a.get("head", "global")).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def calibrated(model, tiles, dev):
    x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).to(dev)
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in model(x)[:4]]
    scale = model.logit_scale.exp().detach()
    lg = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax).float() * scale
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1], 3, 0.35)
    H, V = H.clone(), V.clone()
    H.fill_diagonal_(-1e4)
    V.fill_diagonal_(-1e4)
    return H, V


def candidates(H, V, k, br_top):
    """(A, k*k*br_top, 4) quads and their pairwise sums, as the harvest builds them."""
    a = torch.as_tensor(ANCHORS, device=H.device)
    R = torch.topk(H, k, dim=1).indices[a]                   # (A, k)
    D = torch.topk(V, k, dim=1).indices[a]
    S = V[R].unsqueeze(1) + H[D].unsqueeze(2)                # (A, k_d, k_r, N)
    S = S.reshape(a.numel(), k * k, N)
    br = torch.topk(S, br_top, dim=2).indices                # (A, k*k, br_top)
    rr = R[:, None, :].expand(-1, k, k).reshape(a.numel(), k * k)
    dd = D[:, :, None].expand(-1, k, k).reshape(a.numel(), k * k)
    quad = torch.stack([a[:, None, None].expand(-1, k * k, br_top),
                        rr[:, :, None].expand(-1, -1, br_top),
                        dd[:, :, None].expand(-1, -1, br_top),
                        br], -1).reshape(a.numel(), k * k * br_top, 4)
    pair = (H[quad[..., 0], quad[..., 1]] + V[quad[..., 0], quad[..., 2]]
            + H[quad[..., 2], quad[..., 3]] + V[quad[..., 1], quad[..., 3]])
    return quad, pair


def build_cache(matcher, names, inv, dev, out, k, br_top):
    per = k * k * br_top
    tiles = np.empty((len(names), N, 20, 20, 3), np.uint8)
    cand = np.empty((len(names), ANCHORS.size, per, 4), np.int16)
    pair = np.empty((len(names), ANCHORS.size, per), np.float16)
    t0 = time.time()
    for b, nm in enumerate(names):
        t = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv[b].astype(np.int64)]
        tiles[b] = t.astype(np.uint8)
        q, p = candidates(*calibrated(matcher, t, dev), k, br_top)
        cand[b] = q.cpu().numpy().astype(np.int16)
        pair[b] = p.cpu().numpy().astype(np.float16)
        if (b + 1) % 50 == 0:
            print(f"  cached {b+1}/{len(names)}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out, tiles=tiles, cand=cand, pair=pair)
    return {"tiles": tiles, "cand": cand, "pair": pair}


class Residual(nn.Module):
    """The pairwise sum, plus a learned correction from the assembled patch."""

    def __init__(self, width):
        super().__init__()
        self.net = PlaquetteNet(width)

    def forward(self, quads, pair_z, per):
        with torch.autocast("cuda", torch.float16):
            r = self.net(quads).float().reshape(-1, per)
        r = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
        return pair_z + r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matcher", default="seam_embed_v1.pt")
    ap.add_argument("--boards", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--br-top", type=int, default=2)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--cache", default="plaquette_listwise_cache.npz")
    ap.add_argument("--out", default="plaquette_listwise.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(x) for x in blob["names"][: a.boards]]
    inv = blob["inv"][: a.boards]
    cpath = Path(CACHE_DIR) / a.cache
    if cpath.exists():
        z = np.load(cpath)
        cache = {k: z[k] for k in ("tiles", "cand", "pair")}
        print(f"cache loaded: {cache['tiles'].shape[0]} boards", flush=True)
    else:
        print(f"building cache for {len(names)} boards", flush=True)
        cache = build_cache(load_matcher(a.matcher, dev), names, inv, dev, cpath,
                            a.k, a.br_top)

    truth = np.stack([ANCHORS, ANCHORS + 1, ANCHORS + G, ANCHORS + G + 1], 1)
    n_boards = cache["tiles"].shape[0]
    hold = max(1, n_boards // 10)
    train_b = n_boards - hold
    per = a.k * a.k * a.br_top

    model = Residual(a.width).to(dev)
    print(f"width {a.width}, {count_params(model):,} parameters, {per} candidates "
          f"per anchor, {train_b} train boards", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler("cuda")

    # which (board, anchor) pairs have their true quad in the candidate set, and
    # where it sits.  Computed once: the rejection sampling this replaced cost
    # 2.6 s per step, all of it on the CPU.
    _hit = (cache["cand"].astype(np.int64) == truth[None, :, None, :]).all(3)
    _cov = np.argwhere(_hit.any(2))                       # (M, 2) board, anchor
    _tgt = _hit.argmax(2)
    _train = _cov[_cov[:, 0] < train_b]
    _val = _cov[_cov[:, 0] >= train_b]
    print(f"covered anchors: {len(_cov)} of {_hit.shape[0] * _hit.shape[1]} "
          f"({len(_cov) / _hit.size * per:.3f})", flush=True)

    def batch(rng, pool):
        pick = pool[rng.integers(0, len(pool), a.anchors)]
        b, s = pick[:, 0], pick[:, 1]
        q = cache["cand"][b, s].astype(np.int64)
        p = cache["pair"][b, s].astype(np.float32)
        p = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-6)
        x = cache["tiles"][b[:, None, None], q]
        x = torch.from_numpy(np.ascontiguousarray(x)).to(dev, torch.float32)
        x = x.permute(0, 1, 2, 5, 3, 4).reshape(-1, 4, 3, 20, 20) / 255.0
        return (x, torch.from_numpy(p).to(dev),
                torch.from_numpy(_tgt[b, s]).to(dev))

    def evaluate():
        vrng = np.random.default_rng(99)
        model.eval()
        hit = base = tot = 0
        with torch.no_grad():
            for _ in range(24):
                x, pz, tgt = batch(vrng, _val)
                s = model(x, pz, per)
                hit += int((s.argmax(1) == tgt).sum())
                base += int((pz.argmax(1) == tgt).sum())
                tot += tgt.numel()
        model.train()
        return hit / tot, base / tot

    rng = np.random.default_rng(0)
    run_loss, best, t0 = [], 0.0, time.time()
    for step in range(1, a.steps + 1):
        x, pz, tgt = batch(rng, _train)
        loss = F.cross_entropy(model(x, pz, per), tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss.append(float(loss.detach()))
        if step % a.eval_every == 0 or step == a.steps:
            acc, base = evaluate()
            print(f"step {step:6d}  loss {np.mean(run_loss[-200:]):.4f}  "
                  f"pick {acc:.4f}  pairwise {base:.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
            if acc >= best:
                best = acc
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "step": step,
                            "eval": {"pick": acc, "pairwise": base}},
                           Path(CKPT_DIR) / a.out)
    print(json.dumps({"best_pick": best, "out": a.out}), flush=True)


if __name__ == "__main__":
    main()
