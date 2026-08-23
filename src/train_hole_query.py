"""A JOINT query over a cell's neighbourhood, scored against the whole board.

Every scorer in this repository is a function of TWO fragments.  Even the
cross-encoder of M267 reads a pair.  M204 measured what a cell's neighbourhood
is worth by SUMMING calibrated seam log-probabilities over the known
neighbours -- one neighbour 0.324, two 0.511, three 0.608, four 0.669 -- and
M273 re-derived the same curve.  A sum cannot see that the continuations must
agree with each other at the CORNERS where three fragments meet, and cannot
weigh a strong side against a washed-out one.  Nothing here has ever asked the
neighbourhood as one question.

`train_hole_filler.py` was written for that and never run, as a cross-encoder:
one forward per (neighbourhood, candidate) pair.  That makes a 576-way
objective unaffordable, which is why it defaulted to seven negatives -- and
M107 is explicit that this is the failure mode rather than the fix:

    the failure was budget, not architecture.  The retriever spent 17000 steps
    on a 576-way contrastive objective and a fresh re-ranker is asked to
    rediscover that from a 20-way signal.  Inheriting the trunk removes the
    rediscovery.

So the shape here is two towers.  The candidate side IS the retriever, frozen:
a fragment's key is its four seam descriptors concatenated, which already cost
17000 steps of 576-way contrastive training.  The query side is new and is
where the joint reasoning lives -- one convolutional pass over the whole 3x3
neighbourhood with the centre blank, so the corners, the mask and the relative
strength of the four sides are all visible at once.  Scoring is then a dot
product against all 576 fragments of the board, so the objective is 576-way at
the price of one forward per hole.

M105's other lesson is respected by construction: the query sees strictly more
than the retriever does, a 60x60 neighbourhood against a 20x20 tile.  And there
is no score plane to copy, so M107's copying shortcut cannot arise.

The number to beat is M204's, measured the same way: top-1 among 576 with k
true neighbours revealed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, TRAIN_TGT
from distort import distort_frags
from restore_tile import to_frags
from seam_embed import SeamEmbed

N = G * G
DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
SLOTS = (1, 3, 5, 7)


def rgb(path):
    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


class HoleQuery(nn.Module):
    """A 3x3 neighbourhood with an empty centre -> one query vector.

    Five input planes: three of colour, one marking which cells are present and
    one marking the centre.  The centre is left BLANK, so the network describes
    the hole rather than judging a candidate already in it, which is what makes
    a single forward pass serve all 576 candidates.
    """

    def __init__(self, ch=64, blocks=4, dim=768):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(5, ch, 3, padding=1),
                                  nn.GroupNorm(8, ch), nn.GELU())
        body, c = [], ch
        for _ in range(blocks):
            body += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                     nn.GroupNorm(8, c * 2), nn.GELU(),
                     nn.Conv2d(c * 2, c * 2, 3, padding=1),
                     nn.GroupNorm(8, c * 2), nn.GELU()]
            c *= 2
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(c, dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.5))

    def forward(self, x):
        v = self.head(self.body(self.stem(x)).mean((2, 3)))
        return F.normalize(v, dim=-1)


def load_retriever(name, dev):
    ck = torch.load(Path(CKPT_DIR) / name, map_location=dev, weights_only=False)
    a = ck["args"]
    m = SeamEmbed(a["ch"], a["blocks"], a["dim"], a["strip"], a.get("head", "global"),
                  predict=any(k.startswith("pred.") for k in ck["model"])).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def board_keys(retriever, frags, dev):
    """Each fragment's four seam descriptors, concatenated and re-normalised."""
    x = torch.from_numpy(np.ascontiguousarray(frags)).permute(0, 3, 1, 2).to(dev)
    with torch.autocast("cuda", torch.float16):
        desc = [t.float() for t in retriever(x)[:4]]
    return F.normalize(torch.cat(desc, dim=1), dim=-1)


def patches(frags, cells, rng, keep_range=(1, 5)):
    """One 5-plane neighbourhood per cell, with a random subset of neighbours."""
    out = np.zeros((len(cells), 3 * FS, 3 * FS, 5), np.float32)
    for n, c in enumerate(cells):
        r, q = divmod(int(c), G)
        present = []
        for k, (dr, dq) in enumerate(DIRS):
            rr, qq = r + dr, q + dq
            if 0 <= rr < G and 0 <= qq < G:
                present.append((SLOTS[k], rr * G + qq))
        rng.shuffle(present)
        keep = int(rng.integers(*keep_range))
        for slot, cell in present[:keep]:
            sr, sc = divmod(slot, 3)
            sl = (slice(sr * FS, (sr + 1) * FS), slice(sc * FS, (sc + 1) * FS))
            out[n, sl[0], sl[1], :3] = frags[cell]
            out[n, sl[0], sl[1], 3] = 255.0
        out[n, FS:2 * FS, FS:2 * FS, 4] = 255.0
    return out


def to_tensor(p, dev):
    return torch.from_numpy(p).permute(0, 3, 1, 2).to(dev) / 255.0 - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="seam_embed_v3.pt")
    ap.add_argument("--boards", type=int, default=1200)
    ap.add_argument("--holes", type=int, default=24, help="holes per step")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-boards", type=int, default=12)
    ap.add_argument("--out", default="hole_query.pt")
    a = ap.parse_args()
    dev = "cuda"

    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(x) for x in blob["names"]]
    print(f"loading {a.boards} train and {a.eval_boards} val boards", flush=True)
    train = [to_frags(rgb(Path(TRAIN_TGT) / n)) for n in names[:a.boards]]
    val = [to_frags(rgb(Path(TRAIN_TGT) / n)) for n in names[-a.eval_boards:]]

    retriever = load_retriever(a.retriever, dev)
    model = HoleQuery(a.ch, a.blocks).to(dev)
    dim = board_keys(retriever, train[0].astype(np.float32), dev).shape[1]
    if dim != model.head.out_features:
        model.head = nn.Linear(model.head.in_features, dim).to(dev)
        print(f"query dimension set to the retriever's {dim}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def evaluate():
        """M204's question exactly: top-1 among 576 with k true neighbours."""
        model.eval()
        vr = np.random.default_rng(7)
        hit = {k: [0, 0] for k in (1, 2, 3, 4)}
        for b in val:
            frags = distort_frags(b.astype(np.uint8), vr).astype(np.float32)
            keys = board_keys(retriever, frags, dev)
            cells = vr.permutation(N)[:64]
            for k in (1, 2, 3, 4):
                p = patches(frags, cells, vr, (k, k + 1))
                with torch.autocast("cuda", torch.float16):
                    q = model(to_tensor(p, dev)).float()
                s = q @ keys.t()
                # the revealed neighbours are on the board and cannot be the answer
                for n, c in enumerate(cells):
                    r, qq = divmod(int(c), G)
                    for dr, dq in DIRS:
                        rr, cc = r + dr, qq + dq
                        if 0 <= rr < G and 0 <= cc < G:
                            s[n, rr * G + cc] = -1e4
                hit[k][0] += int((s.argmax(1).cpu().numpy() == cells).sum())
                hit[k][1] += len(cells)
        model.train()
        return {k: v[0] / max(v[1], 1) for k, v in hit.items()}

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M trainable, "
          f"keys frozen from {a.retriever}", flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = train[int(rng.integers(len(train)))]
        frags = distort_frags(b.astype(np.uint8), rng).astype(np.float32)
        keys = board_keys(retriever, frags, dev)
        cells = rng.permutation(N)[:a.holes]
        p = patches(frags, cells, rng)
        with torch.autocast("cuda", torch.float16):
            q = model(to_tensor(p, dev))
        logits = model.logit_scale.exp() * (q.float() @ keys.t())
        for n, c in enumerate(cells):
            r, qq = divmod(int(c), G)
            for dr, dq in DIRS:
                rr, cc = r + dr, qq + dq
                if 0 <= rr < G and 0 <= cc < G:
                    logits[n, rr * G + cc] = -1e4
        loss = F.cross_entropy(
            logits, torch.from_numpy(cells.astype(np.int64)).to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"{(time.time() - t0) / step:.2f} s/step  "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB",
                  flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            e = evaluate()
            print(f"  [eval @ {step}] top-1 among 576 by neighbours known: "
                  + "  ".join(f"{k} -> {v:.4f}" for k, v in e.items())
                  + "   (M204 summed: 0.324 0.511 0.608 0.669)", flush=True)
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "eval": e, "step": step}, Path(CKPT_DIR) / a.out)


if __name__ == "__main__":
    main()
