"""Train the board-context refinement on top of a frozen matcher.

Gated on mutual-edge precision, which is what assembly runs on: the frozen
retriever supplies 0.44 calibrated and the knee is 0.72 (M102).  The refinement
is initialised to zero, so step 0 reproduces the retriever exactly and any
movement is attributable.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from assemble_net import cost_planes
from config import CACHE_DIR, CKPT_DIR, NFRAG as N, TRAIN_INP, TRAIN_TGT
from distort import distort_frags, distort_frags_scaled
from restore_tile import to_frags
from seam_context import SeamContext
from seam_cost import cycle_consistency
from seam_embed import SeamEmbed, board_logits, infonce

G = 24


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    def __init__(self, names, inv, real_prob=0.0, mix=0.0):
        self.names, self.inv, self.real_prob, self.mix = names, inv, real_prob, mix

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
            d = (distort_frags_scaled(clean, rng, float(np.random.rand()))
                 if self.mix and np.random.rand() < self.mix
                 else distort_frags(clean, rng)).astype(np.float32)
        return torch.from_numpy(d).permute(0, 3, 1, 2)


@torch.no_grad()
def board_planes(embed, tiles):
    """Calibrated seam scores, used only as an attention bias."""
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in embed(tiles)[:4]]
    lg = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax).float() * embed.logit_scale.exp().detach()
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1])
    return cost_planes(-H, -V)


def edge_stats(desc, scale):
    """R@1 and mutual-edge precision on the CALIBRATED scores, as solvers see them."""
    lg = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax).float() * scale
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1])
    r1, n, ok = [], 0, 0
    for L, step, valid in ((H, 1, lambda p: p % G != G - 1),
                           (V, G, lambda p: p < N - G)):
        C = (-L).cpu().numpy()
        np.fill_diagonal(C, np.inf)
        idx = np.array([p for p in range(N) if valid(p)])
        r1.append((C[idx].argmin(1) == idx + step).mean())
        fwd, back = C.argmin(1), C.argmin(0)
        for i in range(N):
            if back[fwd[i]] == i:
                n += 1
                ok += int(valid(i) and fwd[i] == i + step)
    return float(np.mean(r1)), n, ok / max(1, n)


@torch.no_grad()
def evaluate(model, names, inv_all, n_boards, dev):
    model.eval()
    rows = []
    for k in range(n_boards):
        nm = str(names[k])
        d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv_all[k].astype(np.int64)]
        tiles = torch.from_numpy(d).permute(0, 3, 1, 2).to(dev)
        planes = board_planes(model.embed, tiles)
        base = model.base(tiles)
        with torch.autocast("cuda", torch.float16):
            ref = [t.float() for t in model(tiles, planes)]
        b = edge_stats(base, model.embed.logit_scale.exp())
        r = edge_stats(ref, model.logit_scale.exp())
        rows.append([b[0], b[2], r[0], r[2]])
    model.train()
    v = np.mean(rows, axis=0)
    return {"base_R@1": v[0], "base_prec": v[1], "R@1": v[2], "prec": v[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="retr_frozen.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--mix-init", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-boards", type=int, default=5)
    ap.add_argument("--out", default="seam_ctx_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.retriever, map_location=dev, weights_only=False)
    ta = ck["args"]
    embed = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                      ta.get("head", "global")).to(dev)
    embed.load_state_dict(ck["model"])
    embed.eval()
    model = SeamContext(embed, a.d, a.heads, a.layers, a.ff,
                        mix_init=a.mix_init).to(dev)
    print(f"retriever {a.retriever} step {ck.get('step')}: {ck.get('eval')}", flush=True)

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = len(names) - 300
    dl = DataLoader(Boards(names[:cut], inv[:cut], a.real_prob, a.mix), batch_size=1,
                    shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x in dl:
            if step >= a.steps:
                break
            tiles = x[0].to(dev, non_blocking=True)
            planes = board_planes(model.embed, tiles)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16):
                desc = model(tiles, planes)
                loss, _ = infonce(desc, model.logit_scale.exp())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
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
                e = evaluate(model, names[cut:], inv[cut:], a.eval_boards, dev)
                print(f"  [eval @ {step}] precision {e['prec']:.4f} vs frozen "
                      f"{e['base_prec']:.4f}; R@1 {e['R@1']:.4f} vs {e['base_R@1']:.4f}"
                      f"   (knee at 0.72)", flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
