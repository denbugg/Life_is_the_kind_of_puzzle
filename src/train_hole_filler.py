"""What fits this HOLE? A scorer over a cell's whole neighbourhood at once.

Every scorer in this repo answers about one seam.  M273 measured the question a
person actually asks -- which fragment belongs in a cell whose neighbours are
known -- and found it far easier: summing the existing seam costs over four
known neighbours gives top-1 0.629 against 0.316 for a single seam, with the
true fragment at mean rank 20 of 576.

But that sum is a weak way to ask it.  Four independent seam costs added
together cannot see that the four continuations must agree with EACH OTHER
around the corners, and cannot weigh a strong side against a washed-out one.
The hole is one object and deserves one model.

The lessons of M105 and M107 are built in.  The second stage must see MORE than
the stage it corrects, so it gets the full 3x3 neighbourhood rather than strips
-- a narrow band scored 0.188 where the retriever scored 0.484, because the
retriever reads whole tiles and uses the interior to clean the ring.  And it
must not be handed the first stage's answer unconditionally, or it learns to
copy: the seam-cost feature is dropped half the time in training.

Missing neighbours are normal -- a cell on the frontier has one or two -- so the
mask is part of the input and the training distribution covers every count from
one to four.
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

N = G * G


def load_rgb(path):
    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


class HoleFiller(nn.Module):
    """A 3x3 neighbourhood with a candidate in the middle -> one score.

    Six input planes: three of colour, one marking which of the nine cells are
    actually present, one marking the centre, and one carrying the summed seam
    cost of this candidate against the known neighbours -- dropped at random in
    training so the pixels have to carry the decision.
    """

    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(6, ch, 3, padding=1),
                                  nn.GroupNorm(8, ch), nn.GELU())
        body, c = [], ch
        for _ in range(blocks):
            body += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                     nn.GroupNorm(8, c * 2), nn.GELU(),
                     nn.Conv2d(c * 2, c * 2, 3, padding=1),
                     nn.GroupNorm(8, c * 2), nn.GELU()]
            c *= 2
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(c, 1)

    def forward(self, x):
        return self.head(self.body(self.stem(x)).mean((2, 3))).squeeze(-1)


def _patch(frags, cells, present, centre_frag, cost_plane):
    """Assemble the 6-plane 3x3 patch for one (neighbourhood, candidate) pair."""
    img = np.zeros((3 * FS, 3 * FS, 6), np.float32)
    for k in range(9):
        r, c = divmod(k, 3)
        sl = (slice(r * FS, (r + 1) * FS), slice(c * FS, (c + 1) * FS))
        if k == 4:
            img[sl[0], sl[1], :3] = centre_frag
            img[sl[0], sl[1], 3] = 255.0
            img[sl[0], sl[1], 4] = 255.0
        elif present[k]:
            img[sl[0], sl[1], :3] = frags[cells[k]]
            img[sl[0], sl[1], 3] = 255.0
    img[:, :, 5] = cost_plane
    return img


def sample_batch(boards, batch, rng, negatives=7, drop=0.5):
    """One true fragment and `negatives` wrong ones for each sampled hole."""
    xs, ys = [], []
    for _ in range(batch):
        b = boards[rng.integers(len(boards))]
        frags = distort_frags(b.astype(np.uint8), rng).astype(np.float32)
        c = int(rng.integers(N))
        r, q = divmod(c, G)
        cells, present = [0] * 9, [False] * 9
        # only the four edge-adjacent neighbours are evidence; the diagonals are
        # left empty because nothing in the pipeline ever asserts a diagonal
        for k, (dr, dq) in enumerate(((-1, 0), (0, -1), (0, 1), (1, 0))):
            rr, qq = r + dr, q + dq
            slot = (1, 3, 5, 7)[k]
            if 0 <= rr < G and 0 <= qq < G:
                cells[slot] = rr * G + qq
                present[slot] = True
        keep = rng.integers(1, 5)                 # one to four neighbours known
        idx = [s for s in (1, 3, 5, 7) if present[s]]
        rng.shuffle(idx)
        for s in idx[keep:]:
            present[s] = False
        if not any(present):
            present[idx[0]] = True

        cands = [c]
        while len(cands) < 1 + negatives:
            t = int(rng.integers(N))
            if t != c:
                cands.append(t)
        for t in cands:
            plane = 0.0 if rng.random() < drop else 255.0
            xs.append(_patch(frags, cells, present, frags[t], plane))
            ys.append(1.0 if t == c else 0.0)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    return x, torch.tensor(ys), negatives + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boards", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=24, help="holes per step")
    ap.add_argument("--negatives", type=int, default=7)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default="hole_filler.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(n) for n in blob["names"]]
    print(f"loading {a.boards} train and 24 val boards", flush=True)
    train = [to_frags(load_rgb(Path(TRAIN_TGT) / n)) for n in names[: a.boards]]
    val = [to_frags(load_rgb(Path(TRAIN_TGT) / n)) for n in names[-24:]]

    model = HoleFiller(a.ch, a.blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)

    def evaluate():
        """Pick rate within the group, which is what growth actually needs."""
        model.eval()
        vr = np.random.default_rng(7)
        hit = tot = 0
        with torch.no_grad():
            for _ in range(24):
                x, y, gsz = sample_batch(val, 16, vr, a.negatives, drop=1.0)
                with torch.autocast("cuda", torch.float16):
                    s = model(x.to(dev)).float().cpu().reshape(-1, gsz)
                hit += int((s.argmax(1) == 0).sum())
                tot += s.shape[0]
        model.train()
        return hit / tot

    run, t0, best = [], time.time(), 0.0
    for step in range(1, a.steps + 1):
        x, y, gsz = sample_batch(train, a.batch, rng, a.negatives)
        with torch.autocast("cuda", torch.float16):
            s = model(x.to(dev)).float().reshape(-1, gsz)
            loss = F.cross_entropy(s, torch.zeros(s.shape[0], dtype=torch.long,
                                                  device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(float(loss.detach()))
        if step % a.eval_every == 0 or step == a.steps:
            acc = evaluate()
            print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  "
                  f"pick {acc:.4f}  (chance {1 / (a.negatives + 1):.4f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "pick": acc, "step": step},
                           Path(CKPT_DIR) / a.out)
    print(f"best pick {best:.4f} -> {Path(CKPT_DIR) / a.out}")
    print("summing seam costs over four known neighbours picks the true "
          "fragment out of ALL 576 at 0.629 (M273)")


if __name__ == "__main__":
    main()
