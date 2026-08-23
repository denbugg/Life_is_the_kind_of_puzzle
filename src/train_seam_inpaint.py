"""Train the seam inpainter on true neighbour pairs.

The network sees a joined pair with the strip at the join removed and predicts
the CLEAN content of that strip.  Only true neighbours are used for training:
compatibility at inference is the disagreement between the prediction and what
was actually observed there, which is small for a genuine neighbour (both are
the same clean transition plus noise) and large for a stranger, whose context
constrains the strip towards something else entirely.

Motivation: every measure tried so far compares the two facing border strips
directly and saturates near R@1 0.17, because those strips carry 26% more error
than the tile interior.  Here the prediction is formed from the cleaner
interior of both pieces, so the comparison is not noise against noise.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import CACHE_DIR, CKPT_DIR, GRID as G, NFRAG as N, TRAIN_INP, TRAIN_TGT, VAL_COUNT
from distort import distort_frags
from restore_tile import blur3_np, ridge_cost, to_frags
from seam_inpaint import SeamInpainter, join_pair, observed_strip

H_EDGE = lambda p: p % G != G - 1
V_EDGE = lambda p: p < N - G
AXES = (("h", 1, H_EDGE), ("v", G, V_EDGE))


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


class Pairs(Dataset):
    def __init__(self, names, inv, margin, thr, synth_prob):
        self.names, self.inv, self.margin = names, inv, margin
        self.thr, self.synth = thr, synth_prob

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = self.names[k]
        clean = to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.float32)
        if self.synth and np.random.rand() < self.synth:
            tiles = distort_frags(clean.astype(np.uint8), np.random.default_rng()).astype(np.float32)
            good = np.ones(N, bool)
        else:
            dirty = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)
            tiles = dirty[self.inv[k].astype(np.int64)]
            good = self.margin[k] >= self.thr
        target = blur3_np(clean)                      # clean transition, blur retained
        return (torch.from_numpy(tiles).permute(0, 3, 1, 2),
                torch.from_numpy(target).permute(0, 3, 1, 2),
                torch.from_numpy(good.astype(np.uint8)))


@torch.no_grad()
def evaluate(model, names, inv, device, n_boards, hole, shortlist=64):
    """R@1 / R@20 using inpainting disagreement as the compatibility score."""
    model.eval()
    rows = []
    for k in range(n_boards):
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(np.float32)
        tiles = tiles[inv[k].astype(np.int64)]
        t = torch.from_numpy(tiles).permute(0, 3, 1, 2).to(device)
        got = []
        for axis, step, edge in AXES:
            sl = np.argsort(ridge_cost(tiles, axis=axis), axis=1)[:, :shortlist]
            score = np.full((N, N), np.inf, np.float32)
            rr = np.repeat(np.arange(N), shortlist)
            cc = sl.reshape(-1)
            for s in range(0, len(rr), 2048):
                ri = torch.from_numpy(rr[s:s + 2048]).to(device)
                ci = torch.from_numpy(cc[s:s + 2048]).to(device)
                with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                    pair = join_pair(t[ri], t[ci], axis)
                    pred = model(pair)
                    err = (pred.float() - observed_strip(pair, hole).float()).abs().mean(dim=(1, 2, 3))
                score[rr[s:s + 2048], cc[s:s + 2048]] = err.float().cpu().numpy()
            np.fill_diagonal(score, np.inf)
            idx = np.array([p for p in range(N) if edge(p)])
            order = np.argsort(score[idx], axis=1)
            rk = np.array([np.where(order[i] == idx[i] + step)[0][0] for i in range(len(idx))])
            got.append([(rk == 0).mean(), (rk < 20).mean()])
        rows.append(np.mean(got, axis=0))
    model.train()
    return np.mean(rows, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT_DIR) / "seam_inpaint.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--pairs-per-board", type=int, default=192)
    ap.add_argument("--hole", type=int, default=4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--synth-prob", type=float, default=0.5)
    ap.add_argument("--margin-quantile", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    blob = np.load(args.labels, allow_pickle=True)
    names, inv, margin = blob["names"], blob["inv"], blob["margin"]
    thr = float(np.quantile(margin, args.margin_quantile))
    n_val = min(VAL_COUNT, len(names) // 4)
    tr = slice(0, len(names) - n_val)
    va = slice(len(names) - n_val, len(names))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeamInpainter(args.ch, args.blocks, args.hole).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}", flush=True)

    dl = DataLoader(Pairs(names[tr], inv[tr], margin[tr], thr, args.synth_prob), batch_size=1,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(0)

    best, step, started = -1.0, 0, time.perf_counter()
    while step < args.steps:
        for tiles, target, good in dl:
            if step >= args.steps:
                break
            tiles = tiles[0].to(device)
            target = target[0].to(device)
            good = good[0].numpy()
            opt.zero_grad(set_to_none=True)
            losses = []
            for axis, stp, edge in AXES:
                cand = [p for p in range(N) if edge(p) and good[p] and good[p + stp]]
                if not cand:
                    continue
                sel = rng.choice(np.array(cand), size=min(args.pairs_per_board, len(cand)), replace=False)
                a = torch.from_numpy(sel).to(device)
                b = torch.from_numpy(sel + stp).to(device)
                with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                    pred = model(join_pair(tiles[a], tiles[b], axis))
                    tgt = observed_strip(join_pair(target[a], target[b], axis), args.hole)
                    loss = (pred.float() - tgt.float()).abs().mean() / 2
                scaler.scale(loss).backward()
                losses.append(loss.detach())
            if not losses:
                continue
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                r1, r20 = evaluate(model, names[va], inv[va], device, args.eval_boards, args.hole)
                flag = ""
                if r1 > best:
                    best = r1
                    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step, "R1": best}, args.ckpt)
                    flag = "  *saved"
                print(f"step {step:5d}  l1 {torch.stack(losses).mean().item():6.2f}  "
                      f"R@1 {r1:.3f}  R@20 {r20:.3f}  "
                      f"{(time.perf_counter()-started)/60:.1f}min{flag}", flush=True)

    print(f"best R@1 = {best:.4f}   ckpt={args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
