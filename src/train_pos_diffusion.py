"""Train positional diffusion on top of the frozen matcher.

Reports place_acc, which is the only number that pays: SSIM through the
submission's own post-processing is 0.192 at chance placement, 0.228 at 0.10,
0.267 at 0.20 and 0.631 at a perfect layout (M103).  Chance is 0.0017 and the
best solver in this repo reaches 0.008 on real boards.

Tiles are shuffled before encoding.  Attention is permutation-equivariant so the
true order cannot leak, but shuffling costs nothing and matches inference.
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
from pos_diffusion import PosDiffusion, cosine_alphas, grid_targets, sample
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
    """Tokens from the matcher's descriptors, and calibrated planes for attention."""
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


def place_acc(pred, true_slot, grid=G):
    """Hungarian from continuous coordinates onto the lattice, then accuracy."""
    cells = grid_targets(N, grid, pred.device)
    cost = torch.cdist(pred, cells).double().cpu().numpy()
    _, slot = linear_sum_assignment(cost)
    return float(np.mean(slot == true_slot.cpu().numpy()))


@torch.no_grad()
def evaluate(model, retr, names, inv_all, a, dev, alphas):
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
        pred = sample(model, tok, planes, alphas, a.sample_steps, seed=k, device=dev)
        tgt = torch.from_numpy(perm).to(dev)
        out.append(place_acc(pred, tgt))
    model.train()
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="retr_frozen.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--mix-init", type=float, default=1.0,
                    help="initial weight on the seam-cost attention bias")
    ap.add_argument("--timesteps", type=int, default=100)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--severity", type=float, default=-1.0)
    ap.add_argument("--eval-severity", type=float, default=-1.0)
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-boards", type=int, default=4)
    ap.add_argument("--out", default="pos_diff_v1.pt")
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

    in_dim = 4 * ta["dim"] + 6
    model = PosDiffusion(in_dim, a.d, a.heads, a.layers, a.ff,
                         mix_init=a.mix_init).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    alphas = cosine_alphas(a.timesteps, device=dev)
    cells = grid_targets(N, G, dev)

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x, perm in dl:
            if step >= a.steps:
                break
            tiles = x[0].to(dev, non_blocking=True)
            slot = perm[0].to(dev).long()
            x0 = cells[slot]                      # token t belongs at slot perm[t]
            tok, planes = encode(retr, tiles)

            t = torch.randint(1, a.timesteps + 1, (1,), device=dev)
            a_t = alphas[t]
            noise = torch.randn_like(x0)
            xt = a_t.sqrt() * x0 + (1 - a_t).sqrt() * noise

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16):
                pred = model(tok, xt, t, planes)
                loss = F.smooth_l1_loss(pred.float(), x0, beta=0.05)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(float(loss.detach()))
            step += 1
            if step % 200 == 0:
                print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f} s/step  "
                      f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                acc = evaluate(model, retr, names[cut:], inv[cut:], a, dev, alphas)
                print(f"  [eval @ {step}] place_acc {acc:.4f}  (chance 0.0017, best "
                      f"solver here 0.008; SSIM 0.192 at chance, 0.228 at 0.10)",
                      flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": {"place_acc": acc}, "step": step},
                           Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
