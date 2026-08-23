"""Train the joint seam head that sits on the retriever's own trunk.

The number to watch is mutual-edge precision, not R@1: assembly switches on near
0.72 and the retriever alone supplies 0.44 (M102).  A second stage that merely
matches the retriever -- which both earlier designs did, one by copying its
score outright -- is worth nothing, so the log prints the retriever's figure on
the same boards beside it every time.
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
from seam_cost import cycle_consistency
from seam_embed import SeamEmbed, board_logits
from seam_joint import SeamJoint

G = 24
AXES = (("h", 1, lambda p: p % G != G - 1), ("v", G, lambda p: p < N - G))


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
def calibrated(embed, tiles):
    """The retriever's own calibrated scores, which also form the shortlist."""
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in embed(tiles)[:4]]
    lg = []
    for ax in ("h", "v"):
        A = board_logits(desc, ax).float() * embed.logit_scale.exp().detach()
        A.fill_diagonal_(-1e4)
        lg.append(A)
    H, V = cycle_consistency(lg[0], lg[1])
    H = H.clone(); V = V.clone()
    H.fill_diagonal_(-1e4); V.fill_diagonal_(-1e4)
    return {"h": H, "v": V}


def score_pairs(model, feats, li, ri, axis, rsc, chunk, k, drop=0.0):
    out = []
    for i in range(0, li.numel(), chunk * k):
        sl = slice(i, i + chunk * k)
        with torch.autocast("cuda", torch.float16):
            out.append(model(feats, li[sl], ri[sl], axis, rsc[sl], drop))
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, frozen, names, inv_all, a, dev):
    model.eval()
    rows = []
    for kk in range(a.eval_boards):
        nm = str(names[kk])
        d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv_all[kk].astype(np.int64)]
        tiles = torch.from_numpy(d).permute(0, 3, 1, 2).to(dev)
        cal = calibrated(frozen, tiles)
        feats = model.features(tiles)
        per = []
        for axis, step, ok in AXES:
            S = cal[axis]
            idx = torch.tensor([p for p in range(N) if ok(p)], device=dev)
            cand = S[idx].topk(a.k, 1).indices
            tgt = idx + step
            cover = (cand == tgt[:, None]).any(1)
            base = (S[idx].argmax(1) == tgt).float().mean().item()
            base_pick = (((S[idx].argmax(1) == tgt) & cover).float().sum()
                         / cover.float().sum()).item()
            li = idx.repeat_interleave(a.k)
            ri = cand.reshape(-1)
            rsc = S[idx].gather(1, cand).reshape(-1)
            sc = score_pairs(model, feats, li, ri, axis, rsc, a.chunk, a.k).reshape(-1, a.k)
            chosen = cand[torch.arange(len(idx), device=dev), sc.float().argmax(1)]
            pick = (((chosen == tgt) & cover).float().sum() / cover.float().sum()).item()

            M = torch.full((N, N), -1e4, device=dev)
            M[li, ri] = sc.reshape(-1).float()
            mutual = M.argmax(0)[chosen] == idx
            prec = (((chosen == tgt) & mutual).sum() / max(1, int(mutual.sum()))).item()
            per.append([base, cover.float().mean().item(), base_pick, pick,
                        (chosen == tgt).float().mean().item(), prec])
        rows.append(np.mean(per, axis=0))
    model.train()
    v = np.mean(rows, axis=0)
    return {"retr_R@1": v[0], "cover": v[1], "retr_pick": v[2], "pick": v[3],
            "R@1": v[4], "mut_prec": v[5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="retr_frozen.pt")
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--strip", type=int, default=4)
    ap.add_argument("--unfreeze", action="store_true",
                    help="train the trunk too, not just the joint head")
    ap.add_argument("--trunk-lr", type=float, default=5e-5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--row-frac", type=float, default=0.5)
    ap.add_argument("--score-drop", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=5)
    ap.add_argument("--out", default="seam_joint_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    ck = torch.load(Path(CKPT_DIR) / a.retriever, map_location=dev, weights_only=False)
    ta = ck["args"]
    embed = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                      ta.get("head", "global")).to(dev)
    embed.load_state_dict(ck["model"])
    embed.eval()
    # the shortlist must not move while the trunk is being trained, so a second
    # frozen copy supplies it -- otherwise the candidate set drifts under the
    # model that is learning to rank it
    frozen = SeamEmbed(ta["ch"], ta["blocks"], ta["dim"], ta["strip"],
                       ta.get("head", "global")).to(dev)
    frozen.load_state_dict(ck["model"])
    frozen.eval()
    for p_ in frozen.parameters():
        p_.requires_grad_(False)
    model = SeamJoint(embed, a.strip, a.ch, a.blocks,
                      freeze_trunk=not a.unfreeze).to(dev)
    print(f"retriever {a.retriever} step {ck.get('step')}: {ck.get('eval')}", flush=True)

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = len(names) - 300
    dl = DataLoader(Boards(names[:cut], inv[:cut], a.real_prob, a.mix), batch_size=1,
                    shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)

    head = [p for n_, p in model.named_parameters()
            if p.requires_grad and not n_.startswith("embed.")]
    trunk = [p for n_, p in model.named_parameters()
             if p.requires_grad and n_.startswith("embed.")]
    params = head + trunk
    groups = [{"params": head, "lr": a.lr}]
    if trunk:
        # the trunk carries 17000 steps of contrastive training; a head-sized
        # learning rate would wash it out before the head knows what to ask for
        groups.append({"params": trunk, "lr": a.trunk_lr})
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, [g["lr"] for g in groups], total_steps=a.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x in dl:
            if step >= a.steps:
                break
            tiles = x[0].to(dev, non_blocking=True)
            cal = calibrated(frozen, tiles)
            feats = model.features(tiles)
            # Two-phase backward.  The trunk is shared by every chunk of the
            # step, and calling backward once per chunk with retain_graph makes
            # each of the ~140 chunks re-traverse it AND keeps the whole graph
            # alive: measured, the run never finished 100 steps in 35 minutes
            # and sat at 7.7 GiB of 8, i.e. spilling into WDDM shared memory.
            # Instead the chunks run against a detached leaf, each freeing its
            # own small graph, their gradients accumulate in `leaf.grad`, and
            # the trunk is traversed exactly once at the end.
            leaf = feats.detach().requires_grad_(True) if a.unfreeze else feats
            opt.zero_grad(set_to_none=True)
            tot, nb = 0.0, 0
            for axis, step_off, ok in AXES:
                S = cal[axis]
                rows = torch.tensor([p for p in range(N) if ok(p)], device=dev)
                cand = S[rows].topk(a.k, 1).indices
                tgt = rows + step_off
                hit = cand == tgt[:, None]
                keep = hit.any(1)
                rows, cand = rows[keep], cand[keep]
                pos = hit.float().argmax(1)[keep]
                if rows.numel() == 0:
                    continue
                if 0.0 < a.row_frac < 1.0:
                    sel = torch.randperm(rows.numel(), device=dev)[
                        :max(1, int(rows.numel() * a.row_frac))]
                    rows, cand, pos = rows[sel], cand[sel], pos[sel]
                rsc = S[rows].gather(1, cand)
                nch = max(1, (rows.numel() + a.chunk - 1) // a.chunk)
                for c0 in range(0, rows.numel(), a.chunk):
                    rw, cd = rows[c0:c0 + a.chunk], cand[c0:c0 + a.chunk]
                    li = rw.repeat_interleave(a.k)
                    ri = cd.reshape(-1)
                    rs = rsc[c0:c0 + a.chunk].reshape(-1)
                    with torch.autocast("cuda", torch.float16):
                        s = model(leaf, li, ri, axis, rs, a.score_drop)
                        loss = F.cross_entropy(s.reshape(-1, a.k),
                                               pos[c0:c0 + a.chunk]) / nch
                    scaler.scale(loss).backward()
                    tot += float(loss.detach())
                nb += 1
            if a.unfreeze and leaf.grad is not None:
                feats.backward(leaf.grad)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(tot / max(1, nb))
            step += 1
            if step % 100 == 0:
                print(f"step {step:5d}  loss {np.mean(run[-100:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f} s/step  "
                      f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                e = evaluate(model, frozen, names[cut:], inv[cut:], a, dev)
                print(f"  [eval @ {step}] pick {e['pick']:.4f} vs retriever "
                      f"{e['retr_pick']:.4f}; R@1 {e['R@1']:.4f} vs {e['retr_R@1']:.4f}; "
                      f"MUTUAL PRECISION {e['mut_prec']:.4f} vs 0.44 "
                      f"(cover {e['cover']:.3f}, need 0.72)", flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
