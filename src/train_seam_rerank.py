"""Train the shortlist re-ranker behind a frozen descriptor retriever.

Rows are kept only when the true neighbour is actually inside the shortlist:
that is precisely the population the re-ranker will face, since a row whose true
neighbour was never retrieved is lost whatever the second stage does.  The
ceiling is therefore the retriever's R@K, and the number to watch is the product
-- retriever R@K times re-ranker accuracy -- against the retriever's own R@1.
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
from seam_rerank import SeamRerank, build_patches

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
def shortlist(retr, tiles, k, row_frac=1.0):
    """Per axis: kept rows, their top-k candidates, and the truth's slot."""
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in retr(tiles)[:4]]
    # Shortlists are drawn from the calibrated scores, not raw cosines: cycle
    # consistency alone lifts R@1 0.270 -> 0.312 on real boards, so it widens
    # the ceiling the re-ranker is working under (its cover is R@K).
    lg = []
    for axis in ("h", "v"):
        A = (board_logits(desc, axis).float() * retr.logit_scale.exp().detach())
        A.fill_diagonal_(-1e4)
        lg.append(A)
    HH, VV = cycle_consistency(lg[0], lg[1])
    cal = {"h": HH.float(), "v": VV.float()}
    out = []
    for axis, step, ok in AXES:
        S = cal[axis].clone()
        S.fill_diagonal_(-1e4)
        rows = torch.tensor([p for p in range(N) if ok(p)], device=tiles.device)
        cand = S[rows].topk(k, dim=1).indices
        tgt = rows + step
        hit = cand == tgt[:, None]
        keep = hit.any(1)
        rows, cand, pos = rows[keep], cand[keep], hit.float().argmax(1)[keep]
        sc = S[rows].gather(1, cand)
        if 0.0 < row_frac < 1.0 and rows.numel() > 0:
            # A step costs one patch per (row, candidate) pair, 23k of them for a
            # whole board.  Rows are independent softmaxes, so taking a random
            # subset each step buys proportionally more optimiser steps for the
            # same compute, which is the better trade at this batch size.
            sel = torch.randperm(rows.numel(), device=rows.device)[
                :max(1, int(rows.numel() * row_frac))]
            rows, cand, pos, sc = rows[sel], cand[sel], pos[sel], sc[sel]
        out.append((axis, rows, cand, pos, sc))
    return out


@torch.no_grad()
def evaluate(model, retr, names, inv_all, a, dev):
    model.eval()
    rows = []
    for kk in range(a.eval_boards):
        nm = str(names[kk])
        d = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv_all[kk].astype(np.int64)]
        tiles = torch.from_numpy(d).permute(0, 3, 1, 2).to(dev)
        with torch.autocast("cuda", torch.float16):
            desc = [t.float() for t in retr(tiles)[:4]]
        lg = []
        for axis in ("h", "v"):
            A = (board_logits(desc, axis).float() * retr.logit_scale.exp().detach())
            A.fill_diagonal_(-1e4)
            lg.append(A)
        HH, VV = cycle_consistency(lg[0], lg[1])
        cal = {"h": HH.float(), "v": VV.float()}
        per = []
        for axis, step, ok in AXES:
            S = cal[axis].clone()
            S.fill_diagonal_(-1e4)
            idx = torch.tensor([p for p in range(N) if ok(p)], device=dev)
            cand = S[idx].topk(a.k, dim=1).indices
            tgt = idx + step
            base = (S[idx].argmax(1) == tgt).float().mean().item()
            cover = (cand == tgt[:, None]).any(1).float().mean().item()
            li = idx.repeat_interleave(a.k)
            ri = cand.reshape(-1)
            rsc = S[idx].gather(1, cand).reshape(-1)
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                sc = torch.cat([
                    model(build_patches(tiles, li[i:i + a.chunk * a.k],
                                        ri[i:i + a.chunk * a.k], axis, a.width,
                                        rsc[i:i + a.chunk * a.k]))
                    for i in range(0, li.numel(), a.chunk * a.k)]).reshape(-1, a.k)
            chosen = cand[torch.arange(len(idx), device=dev), sc.argmax(1)]

            # Mutual-edge precision, which is what actually gates assembly:
            # M88 measured 0.669 precision giving 16% relative assembly while
            # MGC's 0.917 gives 60%, so R@1 alone says little.  Scores outside
            # each row's shortlist are unknown, so the matrix is sparse and a
            # tile's best left-hand partner is taken over that support only.
            S = torch.full((N, N), -1e4, device=dev)
            S[li, ri] = sc.reshape(-1).float()
            back = S.argmax(0)
            fwd = chosen
            mutual = back[fwd] == idx
            prec = ((fwd == tgt) & mutual).sum().item() / max(1, mutual.sum().item())
            per.append([base, cover, (chosen == tgt).float().mean().item(),
                        mutual.sum().item(), prec])
        rows.append(np.mean(per, axis=0))
    model.train()
    v = np.mean(rows, axis=0)
    return {"base": v[0], "cover": v[1], "R@1": v[2],
            "pick": v[2] / max(v[1], 1e-9), "mutual": v[3], "mut_prec": v[4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="seam_embed_v1.pt")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--width", type=int, default=20,
                    help="columns kept from each tile; 20 is the whole tile")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--score-drop", type=float, default=0.5,
                    help="chance of hiding the retriever score in training")
    ap.add_argument("--chunk", type=int, default=16,
                    help="rows per backward pass; caps activation memory")
    ap.add_argument("--row-frac", type=float, default=1.0,
                    help="fraction of rows scored per step")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix", type=float, default=0.0)
    ap.add_argument("--real-prob", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=5)
    ap.add_argument("--out", default="seam_rerank_v1.pt")
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
    print(f"retriever from step {ck.get('step')}: {ck.get('eval')}", flush=True)

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = len(names) - 300
    dl = DataLoader(Boards(names[:cut], inv[:cut], a.real_prob, a.mix), batch_size=1,
                    shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)

    model = SeamRerank(a.ch, a.blocks, a.width, use_score=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for x in dl:
            if step >= a.steps:
                break
            tiles = x[0].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            tot, nb = 0.0, 0
            for axis, rows, cand, pos, rsc in shortlist(retr, tiles, a.k, a.row_frac):
                if rows.numel() == 0:
                    continue
                # Rows are chunked and backwarded separately.  A whole axis is
                # 11520 patches of 6x20x8, whose activations spill past the 8 GB
                # card into WDDM shared memory and turn a 0.2 s step into 13 s.
                # Each row is its own softmax, so chunking changes nothing but
                # the peak.
                nch = max(1, (rows.numel() + a.chunk - 1) // a.chunk)
                for c0 in range(0, rows.numel(), a.chunk):
                    rw = rows[c0:c0 + a.chunk]
                    cd = cand[c0:c0 + a.chunk]
                    li = rw.repeat_interleave(a.k)
                    ri = cd.reshape(-1)
                    rs = rsc[c0:c0 + a.chunk].reshape(-1)
                    with torch.autocast("cuda", torch.float16):
                        s = model(build_patches(tiles, li, ri, axis, a.width, rs,
                                                a.score_drop))
                        loss = F.cross_entropy(s.reshape(-1, a.k),
                                               pos[c0:c0 + a.chunk]) / nch
                    scaler.scale(loss).backward()
                    # each chunk already carries 1/nch of the axis loss, so
                    # summing them reconstructs the mean, not nch times it
                    tot += float(loss.detach())
                nb += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                e = evaluate(model, retr, names[cut:], inv[cut:], a, dev)
                print(f"  [eval @ {step}] R@1 {e['R@1']:.4f} = cover "
                      f"{e['cover']:.4f} x pick {e['pick']:.4f}; mutual "
                      f"{e['mutual']:.0f} at precision {e['mut_prec']:.3f}"
                      f"   (retriever alone R@1 {e['base']:.4f}, its mutual "
                      f"precision 0.44)", flush=True)
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
