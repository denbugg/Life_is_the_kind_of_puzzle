"""Train the per-tile position prior that breaks the toroidal ambiguity.

The metric that matters is not per-tile accuracy -- a single tile predicts its
own row band at 0.21 against chance 0.167 and that will not improve much (M67).
It is whether 576 such weak votes, summed, pick the right cyclic shift out of
576.  So evaluation rolls a KNOWN-correct layout by a random shift and asks
whether the prior rolls it back.

Trained on synthetic boards, where the true row and column of every tile are
exact.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CACHE_DIR, CKPT_DIR, NFRAG as N, TRAIN_INP, TRAIN_TGT
from distort import distort_frags, distort_frags_scaled
from restore_tile import to_frags
from row_prior import RowPrior, best_shift

G = 24


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    def __init__(self, names, mix=0.0):
        self.names, self.mix = names, mix

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        clean = to_frags(load_rgb(Path(TRAIN_TGT) / str(self.names[k]))).astype(np.uint8)
        rng = np.random.default_rng()
        d = (distort_frags_scaled(clean, rng, float(np.random.rand()))
             if self.mix and np.random.rand() < self.mix
             else distort_frags(clean, rng)).astype(np.float32)
        return torch.from_numpy(d).permute(0, 3, 1, 2)


@torch.no_grad()
def evaluate(model, names, inv_all, n_boards, dev):
    """Per-tile accuracy, and the number that matters: shift recovery."""
    model.eval()
    rows = []
    rng = np.random.default_rng(0)
    for k in range(n_boards):
        nm = str(names[k])
        d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv_all[k].astype(np.int64)]
        x = torch.from_numpy(d).permute(0, 3, 1, 2).to(dev)
        with torch.autocast("cuda", torch.float16):
            rl, cl = model(x)
        pos = torch.arange(N, device=dev)
        r_acc = (rl.float().argmax(1) == pos // G).float().mean().item()
        c_acc = (cl.float().argmax(1) == pos % G).float().mean().item()

        # a correct layout, rolled away and asked to come back
        truth = np.arange(N)
        dr, dc = int(rng.integers(G)), int(rng.integers(G))
        rolled = np.roll(truth.reshape(G, G), (dr, dc), axis=(0, 1)).ravel()
        back = best_shift(rolled, d, model, device=dev)
        rows.append([r_acc, c_acc, float(np.mean(back == truth))])
    model.train()
    v = np.mean(rows, axis=0)
    return {"row_acc": v[0], "col_acc": v[1], "shift_recovered": v[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=5)
    ap.add_argument("--out", default="row_prior.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = len(names) - 300
    dl = DataLoader(Boards(names[:cut], a.mix), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)

    model = RowPrior(a.ch, a.blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    tgt_r = (torch.arange(N, device=dev) // G)
    tgt_c = (torch.arange(N, device=dev) % G)

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x in dl:
            if step >= a.steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            with torch.autocast("cuda", torch.float16):
                for b in range(x.shape[0]):
                    rl, cl = model(x[b].to(dev, non_blocking=True))
                    loss = loss + 0.5 * (F.cross_entropy(rl, tgt_r)
                                         + F.cross_entropy(cl, tgt_c))
                loss = loss / x.shape[0]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(float(loss.detach()))
            step += 1
            if step % 200 == 0:
                print(f"step {step:5d}  loss {np.mean(run[-200:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f} s/step  "
                      f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                e = evaluate(model, names[cut:], inv[cut:], a.eval_boards, dev)
                print(f"  [eval @ {step}] row {e['row_acc']:.3f} col {e['col_acc']:.3f} "
                      f"(chance 0.042); SHIFT RECOVERED {e['shift_recovered']:.3f}",
                      flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
