"""Train the iterative discrete assembler.

Masking schedule matters.  Revealing a uniformly random fraction each step means
the model rarely sees the regime it starts inference in -- everything hidden --
and rarely the one it ends in.  Sampling the fraction from a distribution that
covers both ends, and including the all-hidden case outright, keeps every round
of the decode in distribution.

Reports place_acc, the only number that pays: through the submission's own
post-processing SSIM is 0.192 at chance placement, 0.228 at 0.10 and 0.267 at
0.20 (M103).  Chance is 0.0017.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from assemble_net import cost_planes
from config import CACHE_DIR, CKPT_DIR, NFRAG as N, TRAIN_INP, TRAIN_TGT
from distort import distort_frags, distort_frags_scaled
from iter_assemble import IterAssemble, decode
from restore_tile import to_frags
from seam_cost import cycle_consistency
from seam_embed import SeamEmbed, board_logits

G = 24


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    def __init__(self, names, inv, real_prob=0.0, mix=0.0, severity=-1.0):
        self.names, self.inv, self.real_prob = names, inv, real_prob
        self.mix, self.severity = mix, severity

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = str(self.names[k])
        if np.random.rand() < self.real_prob:
            d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
                self.inv[k].astype(np.int64)]
        else:
            clean = to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.uint8)
            rng = np.random.default_rng()
            if self.severity >= 0.0:
                d = distort_frags_scaled(clean, rng, self.severity).astype(np.float32)
            elif self.mix and np.random.rand() < self.mix:
                d = distort_frags_scaled(clean, rng,
                                         float(np.random.rand())).astype(np.float32)
            else:
                d = distort_frags(clean, rng).astype(np.float32)
        perm = np.random.permutation(N).astype(np.int64)
        return torch.from_numpy(d[perm]).permute(0, 3, 1, 2), torch.from_numpy(perm)


@torch.no_grad()
def encode(retr, tiles):
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in retr(tiles)[:4]]
    s = tiles.flatten(2)
    tok = torch.cat([torch.cat(desc, 1), s.mean(-1) / 255.0, s.std(-1) / 255.0], 1)
    lg = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax).float() * retr.logit_scale.exp().detach()
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1])
    return tok, cost_planes(-H, -V)


@torch.no_grad()
def evaluate(model, retr, names, inv_all, a, dev):
    model.eval()
    out = []
    for k in range(a.eval_boards):
        nm = str(names[k])
        if a.eval_severity >= 0.0:
            d = distort_frags_scaled(
                to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.uint8),
                np.random.default_rng(k), a.eval_severity).astype(np.float32)
        else:
            d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
                inv_all[k].astype(np.int64)]
        perm = np.random.permutation(N).astype(np.int64)
        tiles = torch.from_numpy(d[perm]).permute(0, 3, 1, 2).to(dev)
        tok, planes = encode(retr, tiles)
        rl, cl = decode(model, tok, planes, a.rounds, device=dev)
        cost = -model.slot_logits(rl, cl).double().cpu().numpy()
        _, slot = linear_sum_assignment(cost)
        out.append(float(np.mean(slot == perm)))
    model.train()
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="retr_frozen.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--mix-init", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--severity", type=float, default=-1.0)
    ap.add_argument("--eval-severity", type=float, default=-1.0)
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-boards", type=int, default=4)
    ap.add_argument("--out", default="iter_asm_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.retriever, map_location=dev, weights_only=False)
    ta = ck["args"]
    retr = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                     ta.get("head", "global")).to(dev)
    retr.load_state_dict(ck["model"])
    retr.eval()
    for p in retr.parameters():
        p.requires_grad_(False)
    print(f"retriever {a.retriever} step {ck.get('step')}: {ck.get('eval')}", flush=True)

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = len(names) - 300
    dl = DataLoader(Boards(names[:cut], inv[:cut], a.real_prob, a.mix, a.severity),
                    batch_size=1, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=a.workers > 0)

    model = IterAssemble(4 * ta["dim"] + 6, a.d, a.heads, a.layers, a.ff,
                         mix_init=a.mix_init).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    step, t0, run, acc_run = 0, time.time(), [], []
    while step < a.steps:
        for x, perm in dl:
            if step >= a.steps:
                break
            tiles = x[0].to(dev, non_blocking=True)
            slot = perm[0].to(dev).long()
            tr, tc = slot // G, slot % G
            tok, planes = encode(retr, tiles)

            # Weight the reveals HIGH.  With nothing revealed the task is not
            # merely hard, it is ill-posed -- absolute position is defined only
            # by the global arrangement -- so the only learnable answer is the
            # content prior, and training a quarter of the steps that way taught
            # exactly that: revealing 98% of the board still gave row accuracy
            # 0.104 where the cost graph supports about 0.50.  Teach the
            # learnable skill first and keep a little of the hard end.
            u = float(np.random.rand())
            # capped below 1: a step with nothing hidden gives cross-entropy over
            # an empty tensor, which is nan and poisons every later step
            frac = 0.0 if u < 0.05 else min(0.95, max(0.15, u ** 0.5))
            keep = torch.rand(N, device=dev) < frac
            if bool(keep.all()):
                keep[torch.randint(N, (8,), device=dev)] = False
            rows = torch.where(keep, tr, torch.full_like(tr, G))
            cols = torch.where(keep, tc, torch.full_like(tc, G))

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16):
                rl, cl = model(tok, rows, cols, planes)
                hid = ~keep
                loss = 0.5 * (F.cross_entropy(rl[hid], tr[hid])
                              + F.cross_entropy(cl[hid], tc[hid]))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(float(loss.detach()))
            with torch.no_grad():
                acc_run.append(float((rl[hid].argmax(1) == tr[hid]).float().mean()))
            step += 1
            if step % 200 == 0:
                print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  "
                      f"row_acc {np.mean(acc_run[-200:]):.4f} (chance 0.042)  "
                      f"{(time.time()-t0)/step:.2f} s/step  "
                      f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                acc = evaluate(model, retr, names[cut:], inv[cut:], a, dev)
                print(f"  [eval @ {step}] place_acc {acc:.4f}  (chance 0.0017, best "
                      f"solver here 0.008; SSIM 0.192 at chance, 0.228 at 0.10)",
                      flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": {"place_acc": acc}, "step": step},
                           Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
