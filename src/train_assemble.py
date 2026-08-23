"""Train the whole-board assembler on top of a frozen seam matcher.

Reports the metric that actually matters -- place_acc, the fraction of tiles put
in their true slot -- rather than a proxy.  Chance is 0.0017, every solver in
this repo sits there on real boards, and SSIM is roughly 0.083 + 0.39 * place_acc
off raw dirty pixels, so beating the 0.2375 submission honestly needs about 0.40.

Tiles are shuffled before being fed in.  Attention is permutation-equivariant so
the true order could not leak anyway, but shuffling costs nothing and matches
inference exactly.
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

from assemble_net import AssembleNet, cost_planes, sinkhorn_log
from config import CACHE_DIR, CKPT_DIR, NFRAG as N, TRAIN_INP, TRAIN_TGT
from distort import distort_frags, distort_frags_scaled
from restore_tile import to_frags
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
                d = distort_frags_scaled(
                    clean, rng, float(np.random.rand())).astype(np.float32)
            else:
                d = distort_frags(clean, rng).astype(np.float32)
        # d is indexed by TRUE position, so token t holds the tile from
        # position perm[t] -- that is where it belongs
        perm = np.random.permutation(N).astype(np.int64)
        return torch.from_numpy(d[perm]).permute(0, 3, 1, 2), torch.from_numpy(perm)


@torch.no_grad()
def encode(retr, tiles, scale):
    """Tokens from the matcher's descriptors, plus Sinkhorn-calibrated planes."""
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in retr(tiles)[:4]]
    tok = torch.cat(desc, 1)
    s = tiles.flatten(2)
    tok = torch.cat([tok, s.mean(-1) / 255.0, s.std(-1) / 255.0], 1)
    planes = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax) * scale
        A.fill_diagonal_(-1e4)
        planes.append(sinkhorn_log(A, 1.0, 20))
    return tok, cost_planes(-planes[0], -planes[1])


def place_acc(logits, true_slot):
    """Hungarian assignment, then the fraction of tiles landing in their slot.

    true_slot[t] is where token t belongs; linear_sum_assignment returns tokens
    in order, so its column indices are the slots it chose for each.
    """
    _, slot = linear_sum_assignment(-logits.double().cpu().numpy())
    return float(np.mean(slot == true_slot.cpu().numpy()))


@torch.no_grad()
def evaluate(model, retr, names, inv_all, n_boards, dev, scale, severity=-1.0):
    model.eval()
    out = []
    for k in range(n_boards):
        nm = str(names[k])
        if severity >= 0.0:
            d = distort_frags_scaled(
                to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.uint8),
                np.random.default_rng(k), severity).astype(np.float32)
        else:
            d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
                inv_all[k].astype(np.int64)]
        perm = np.random.permutation(N).astype(np.int64)
        tiles = torch.from_numpy(d[perm]).permute(0, 3, 1, 2).to(dev)
        tok, planes = encode(retr, tiles, scale)
        row_lg, col_lg = model(tok, planes)
        logits = model.slot_logits(row_lg, col_lg)
        tgt = torch.from_numpy(perm).to(dev).long()   # token t belongs at perm[t]
        out.append([place_acc(logits, tgt),
                    float((row_lg.argmax(1) == tgt // G).float().mean()),
                    float((col_lg.argmax(1) == tgt % G).float().mean())])
    model.train()
    v = np.mean(out, axis=0)
    return {"place_acc": v[0], "row_acc": v[1], "col_acc": v[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="seam_embed_v1.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--severity", type=float, default=-1.0,
                    help="fix corruption strength; -1 uses the production chain")
    ap.add_argument("--eval-severity", type=float, default=-1.0,
                    help="evaluate on synthetic boards at this strength "
                         "instead of real ones")
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=5)
    ap.add_argument("--out", default="assemble_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.retriever, map_location=dev, weights_only=False)
    ta = ck["args"]
    retr = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                     ta.get("head", "global"),
                     predict=ta.get("predict_weight", 0) > 0).to(dev)
    retr.load_state_dict(ck["model"])
    retr.eval()
    for p in retr.parameters():
        p.requires_grad_(False)
    scale = float(retr.logit_scale.exp().detach())
    print(f"retriever step {ck.get('step')}: {ck.get('eval')}", flush=True)

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = len(names) - 300
    dl = DataLoader(Boards(names[:cut], inv[:cut], a.real_prob, a.mix, a.severity),
                    batch_size=1,
                    shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)

    in_dim = 4 * ta["dim"] * (20 if ta.get("head") == "local" else 1) + 6
    model = AssembleNet(in_dim, a.d, a.heads, a.layers, a.ff).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x, perm in dl:
            if step >= a.steps:
                break
            tiles = x[0].to(dev, non_blocking=True)
            tgt = perm[0].to(dev).long()         # token t belongs at slot tgt[t]
            tok, planes = encode(retr, tiles, scale)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16):
                row_lg, col_lg = model(tok, planes)
                loss = 0.5 * (F.cross_entropy(row_lg, tgt // G)
                              + F.cross_entropy(col_lg, tgt % G))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(float(loss.detach()))
            step += 1
            if step % 100 == 0:
                print(f"step {step:5d}  loss {np.mean(run[-100:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f} s/step  "
                      f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                e = evaluate(model, retr, names[cut:], inv[cut:], a.eval_boards,
                             dev, scale, a.eval_severity)
                print(f"  [eval @ {step}] place_acc {e['place_acc']:.4f}  "
                      f"row {e['row_acc']:.3f} col {e['col_acc']:.3f}  "
                      f"(chance 0.0017 / 0.042; need ~0.40)", flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
