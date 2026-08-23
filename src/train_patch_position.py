"""Where in the photograph does a MULTI-TILE patch belong?

Seven attempts at absolute placement have failed and every one asked the
question of a single 20x20 fragment or of a colour field.  M67 is the closest:
predicting a coarse row band from one CLEAN tile reaches 0.337 against a chance
of 0.250, and the column carries nothing at all, which was read as "photographs
have vertical composition and no horizontal systematics".

That reading may be right about tiles and wrong about patches.  A 6x6 patch is
120 pixels of real photograph -- a face, a horizon, a floor, a stretch of
signage -- and "where in the frame does this come from" is a question a
convolutional network can answer for a patch where it cannot for a fragment.
The harvest now produces components of up to 34 tiles, so if the signal grows
with size there is something to place them with.

The patch is corrupted the way the real data is: every one of its fragments gets
its own contrast, brightness, noise and JPEG quality, independently.  The task
is a coarse band, not an exact cell -- M70 measured that coarse position pays
nothing directly, but a band would narrow the search a component must be placed
into, which is what M217 said the problem needs.
"""
from __future__ import annotations

import argparse
import json
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


def load_rgb(path):
    return np.ascontiguousarray(cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, ::-1])


class PatchPosition(nn.Module):
    """A k*FS square of corrupted photograph -> two coarse band distributions.

    With `masked`, a fourth channel marks which fragments are actually present.
    Real components are not squares: they have holes and ragged edges, and a
    model trained on full patches would be out of distribution on them, which
    M199 measured as the single most expensive mistake available here.
    """

    def __init__(self, bands=6, width=48, blocks=4, masked=False):
        super().__init__()
        self.masked = masked
        self.stem = nn.Sequential(nn.Conv2d(4 if masked else 3, width, 3, padding=1),
                                  nn.GroupNorm(8, width), nn.GELU())
        body = []
        ch = width
        for i in range(blocks):
            body += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                     nn.GroupNorm(8, ch * 2), nn.GELU(),
                     nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
                     nn.GroupNorm(8, ch * 2), nn.GELU()]
            ch *= 2
        self.body = nn.Sequential(*body)
        self.row = nn.Linear(ch, bands)
        self.col = nn.Linear(ch, bands)

    def forward(self, x):
        h = self.body(self.stem(x)).mean((2, 3))
        return self.row(h), self.col(h)


def sample_batch(clean_boards, k, bands, batch, rng, mask_frac=0.0):
    """Random k x k tile patches, each fragment corrupted on its own.

    `mask_frac` drops that share of the fragments and appends a presence
    channel, so the patch looks like a real component: ragged and holed.
    """
    xs, rs, cs = [], [], []
    for _ in range(batch):
        b = clean_boards[rng.integers(len(clean_boards))]
        r0 = int(rng.integers(G - k + 1))
        c0 = int(rng.integers(G - k + 1))
        idx = np.array([(r0 + dy) * G + (c0 + dx)
                        for dy in range(k) for dx in range(k)])
        frags = distort_frags(b[idx].astype(np.uint8), rng).astype(np.float32)
        if mask_frac > 0:
            keep = rng.random(k * k) >= mask_frac * rng.random()
            if keep.sum() < 2:
                keep[:2] = True
            frags = frags * keep[:, None, None, None]
            m = np.repeat(np.repeat(keep.reshape(k, k).astype(np.float32),
                                    FS, 0), FS, 1)
        patch = frags.reshape(k, k, FS, FS, 3).transpose(0, 2, 1, 3, 4)
        img = patch.reshape(k * FS, k * FS, 3)
        if mask_frac > 0:
            img = np.concatenate([img, m[:, :, None] * 255.0], 2)
        xs.append(img)
        # the band of the patch CENTRE, so a patch that spans two bands is not
        # forced to choose the one its corner happens to fall in
        rs.append(min(int((r0 + k / 2) * bands / G), bands - 1))
        cs.append(min(int((c0 + k / 2) * bands / G), bands - 1))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    return x, torch.tensor(rs), torch.tensor(cs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6, help="patch side, in tiles")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--boards", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--mask-frac", type=float, default=0.0,
                    help="drop up to this share of the fragments and add a "
                         "presence channel, so the patch resembles a component")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names = [str(n) for n in blob["names"]]
    train_names = names[: a.boards]
    val_names = names[-100:]
    print(f"loading {len(train_names)} train and {len(val_names)} val boards",
          flush=True)
    train = [to_frags(load_rgb(Path(TRAIN_TGT) / n)) for n in train_names]
    val = [to_frags(load_rgb(Path(TRAIN_TGT) / n)) for n in val_names]

    model = PatchPosition(a.bands, masked=a.mask_frac > 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)

    def evaluate():
        model.eval()
        vr = np.random.default_rng(99)
        hits_r = hits_c = tot = 0
        with torch.no_grad():
            for _ in range(16):
                x, r, c = sample_batch(val, a.k, a.bands, 64, vr,
                                       a.mask_frac)
                with torch.autocast("cuda", torch.float16):
                    pr, pc = model(x.to(dev))
                hits_r += int((pr.argmax(1).cpu() == r).sum())
                hits_c += int((pc.argmax(1).cpu() == c).sum())
                tot += r.numel()
        model.train()
        return hits_r / tot, hits_c / tot

    run, t0, best = [], time.time(), 0.0
    for step in range(1, a.steps + 1):
        x, r, c = sample_batch(train, a.k, a.bands, a.batch, rng,
                               a.mask_frac)
        with torch.autocast("cuda", torch.float16):
            pr, pc = model(x.to(dev))
            loss = F.cross_entropy(pr.float(), r.to(dev)) + \
                F.cross_entropy(pc.float(), c.to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(float(loss.detach()))
        if step % a.eval_every == 0 or step == a.steps:
            ar, ac = evaluate()
            print(f"step {step:6d}  loss {np.mean(run[-200:]):.4f}  "
                  f"row {ar:.4f}  col {ac:.4f}  (chance {1 / a.bands:.4f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
            if ar + ac > best and a.out:
                best = ar + ac
                torch.save({"model": model.state_dict(), "args": vars(a),
                            "step": step, "eval": {"row": ar, "col": ac}},
                           Path(CKPT_DIR) / a.out)
    print(json.dumps({"k": a.k, "bands": a.bands, "best_sum": best}), flush=True)


if __name__ == "__main__":
    main()
