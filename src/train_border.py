"""Is this fragment on the edge of the photograph? Asked of its CONTENT.

M246 built a border detector out of the matcher: Sinkhorn's slack column says
"nothing continues me", which reaches AUC 0.70 on all four sides.  That detector
is structural -- it never looks at what the fragment depicts.

Content is a separate question with a separate answer.  Photographs are framed:
sky and ceiling at the top, ground and floor at the bottom, and the outermost
ring of a frame is where a composition puts its margins.  M67 measured a coarse
row band from one CLEAN fragment at 0.337 against a chance of 0.250 and read it
as thin, but M234 then showed M67's conclusion was about the object it measured
rather than about photographs, and nobody has ever trained the border question
directly.  This trains it.

The generator is the reason to expect little and to check anyway: it rescales
contrast around each fragment's OWN mean and adds a per-fragment brightness
offset, so vignetting -- the one cue that would make this trivial -- is erased
by construction.  What survives is hue, texture and structure.
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

SIDES = ("top", "bottom", "left", "right")


def load_rgb(path):
    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


class BorderNet(nn.Module):
    """One corrupted fragment -> four independent border logits."""

    def __init__(self, width=48, blocks=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1),
                                  nn.GroupNorm(8, width), nn.GELU())
        body, ch = [], width
        for _ in range(blocks):
            body += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                     nn.GroupNorm(8, ch * 2), nn.GELU(),
                     nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
                     nn.GroupNorm(8, ch * 2), nn.GELU()]
            ch *= 2
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(ch, 4)

    def forward(self, x):
        return self.head(self.body(self.stem(x)).mean((2, 3)))


def sample_batch(boards, batch, rng, border_frac=0.5):
    """Fragments with their four border labels.

    Only 92 cells of 576 are on a border and only 24 on any one side, so a
    natural sample is 84% negatives on every head at once.  Half the batch is
    drawn from border cells to keep the gradient alive; the loss carries no
    further weighting, and evaluation uses the natural distribution.
    """
    r, c = np.arange(G) // 1, np.arange(G)
    border_cells = np.array([i for i in range(G * G)
                             if i // G in (0, G - 1) or i % G in (0, G - 1)])
    inner_cells = np.setdiff1d(np.arange(G * G), border_cells)
    xs, ys = [], []
    n_b = int(batch * border_frac)
    for cells, n in ((border_cells, n_b), (inner_cells, batch - n_b)):
        if n == 0:
            continue
        b = boards[rng.integers(len(boards))]
        idx = cells[rng.integers(len(cells), size=n)]
        xs.append(distort_frags(b[idx].astype(np.uint8), rng).astype(np.float32))
        rr, cc = idx // G, idx % G
        ys.append(np.stack([rr == 0, rr == G - 1, cc == 0, cc == G - 1], 1))
    x = torch.from_numpy(np.concatenate(xs)).permute(0, 3, 1, 2).float() / 255.0
    return x, torch.from_numpy(np.concatenate(ys).astype(np.float32))


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s, r = score[order], np.empty(len(score))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    p = int(label.sum())
    return float((r[label].sum() - p * (p - 1) / 2) / (p * (len(score) - p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boards", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="border_net.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(n) for n in blob["names"]]
    print(f"loading {a.boards} train and 40 val boards", flush=True)
    train = [to_frags(load_rgb(Path(TRAIN_TGT) / n)) for n in names[: a.boards]]
    val = [to_frags(load_rgb(Path(TRAIN_TGT) / n)) for n in names[-40:]]

    model = BorderNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)

    def evaluate():
        """Whole boards at the natural 92-of-576 rate, which is what we deploy on."""
        model.eval()
        vr = np.random.default_rng(99)
        S, Y = [], []
        with torch.no_grad():
            for b in val[:12]:
                f = distort_frags(b.astype(np.uint8), vr).astype(np.float32)
                x = torch.from_numpy(f).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.autocast("cuda", torch.float16):
                    S.append(model(x).float().cpu().numpy())
                rr, cc = np.arange(G * G) // G, np.arange(G * G) % G
                Y.append(np.stack([rr == 0, rr == G - 1, cc == 0, cc == G - 1], 1))
        model.train()
        S, Y = np.concatenate(S), np.concatenate(Y)
        return [auc(S[:, k], Y[:, k]) for k in range(4)]

    run, t0, best = [], time.time(), 0.0
    for step in range(1, a.steps + 1):
        x, y = sample_batch(train, a.batch, rng)
        with torch.autocast("cuda", torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(x.to(dev)).float(),
                                                      y.to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(float(loss.detach()))
        if step % a.eval_every == 0 or step == a.steps:
            au = evaluate()
            m = float(np.mean(au))
            print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  AUC "
                  + " ".join(f"{s} {v:.3f}" for s, v in zip(SIDES, au))
                  + f"  mean {m:.3f}  {time.time() - t0:.0f}s", flush=True)
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "auc": au}, Path(CKPT_DIR) / a.out)
    print(f"best mean AUC {best:.3f} -> {Path(CKPT_DIR) / a.out}")
    print("the structural detector (M246) reaches 0.702 / 0.679 / 0.701 / 0.705")


if __name__ == "__main__":
    main()
