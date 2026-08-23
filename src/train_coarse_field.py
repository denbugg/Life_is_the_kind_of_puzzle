"""Train the coarse colour-field predictor, and prove it uses the image.

The scoreboard is the one M137 established: SSIM minus the SSIM of a flat fill
at our own tiles' mean colour.  On that scale the deployed submission is -0.141,
our best layout -0.002, the true layout +0.131, and the leader roughly +0.02.
M138 priced this model's target: a correct 4x4 version of the image is +0.046
and an 8x8 is +0.069.

The control that matters
------------------------
A model like this can score above zero in two quite different ways.  It can read
THIS image's palette and texture -- which is information, and is the point -- or
it can emit the average photograph, a fixed bright-above-dark-below template
that fits most pictures a little.  The second is a prior, not a recovery, and
would be worth reporting separately rather than banking.

So every run has a --blind twin: the same model with the tile features zeroed,
leaving only the flat colour and a free learned template.  Whatever the blind
arm scores is the generic prior; the difference is what the bag of tiles is
actually worth.  Report both, always.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader, Dataset

from coarse_field import CoarseField, render
from config import CACHE_DIR, CKPT_DIR, TRAIN_INP, TRAIN_TGT
from models import ssim_loss
from restore_tile import to_frags


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    """Bag of dirty tiles -> the clean full-resolution target."""

    def __init__(self, names, inv, augment=True):
        self.names, self.inv, self.augment = names, inv, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = str(self.names[k])
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            self.inv[k].astype(np.int64)]
        tgt = load_rgb(Path(TRAIN_TGT) / nm)
        if self.augment:
            # rotating the whole picture rotates every tile and permutes the
            # bag; the bag is unordered, so only the rotation survives
            r = np.random.randint(4)
            if r:
                tiles = np.rot90(tiles, r, axes=(1, 2))
                tgt = np.rot90(tgt, r)
            if np.random.rand() < 0.5:
                tiles, tgt = tiles[:, :, ::-1], tgt[:, ::-1]
        tiles = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2)
        tgt = torch.from_numpy(np.ascontiguousarray(tgt)).permute(2, 0, 1).float() / 255.0
        return tiles.float(), tgt


@torch.no_grad()
def evaluate(model, names, inv, dev, limit, blind):
    model.eval()
    gains, flats = [], []
    for k in range(limit):
        nm = str(names[k])
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv[k].astype(np.int64)]
        tgt = load_rgb(Path(TRAIN_TGT) / nm)
        x = torch.from_numpy(tiles).permute(0, 3, 1, 2)[None].to(dev)
        if blind:
            x = x * 0.0 + x.mean((1, 3, 4), keepdim=True)
        img = render(model(x))[0].permute(1, 2, 0).cpu().numpy()
        img = np.rint(img * 255.0).clip(0, 255).astype(np.uint8)
        flat = np.zeros_like(tgt)
        flat[:] = np.rint(tiles.reshape(-1, 3).mean(0)).clip(0, 255).astype(np.uint8)
        fs = float(ssim_fn(flat, tgt, channel_axis=2, data_range=255))
        gains.append(float(ssim_fn(img, tgt, channel_axis=2, data_range=255)) - fs)
        flats.append(fs)
    model.train()
    return float(np.mean(gains)), float(np.mean(flats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="field resolution")
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--blind", action="store_true",
                    help="hide the tiles; measures the generic-photograph prior")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--train-boards", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=60)
    ap.add_argument("--out", default="coarse_field_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    names, inv = blob["names"], blob["inv"]
    cut = min(a.train_boards, len(names) - 300)
    model = CoarseField(a.n, a.ch, a.dim, a.hidden).to(dev)
    print(f"CoarseField n={a.n} blind={a.blind}: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    dl = DataLoader(Boards(names[:cut], inv[:cut]), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best, step, t0, run = -9.0, 0, time.time(), []
    while step < a.steps:
        for x, y in dl:
            if step >= a.steps:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if a.blind:
                # keep the flat colour, destroy everything else about the tiles
                x = x * 0.0 + x.mean((1, 3, 4), keepdim=True)
            with torch.autocast("cuda", torch.float16):
                field = model(x)
            # the loss stays in fp32: SSIM divides by local variances, which
            # underflow in half precision
            loss = ssim_loss(render(field.float()), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"step {step:6d}  loss {np.mean(run[-100:]):.4f}  "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                g, fs = evaluate(model, names[-300:], inv[-300:], dev,
                                 a.eval_boards, a.blind)
                print(f"  eval step {step}: gain over flat {g:+.4f} "
                      f"(flat itself {fs:.4f})", flush=True)
                if g > best:
                    best = g
                    torch.save({"model": model.state_dict(), "args": vars(a),
                                "step": step, "eval": {"gain": g}},
                               Path(CKPT_DIR) / a.out)
    print(f"\nbest gain over flat: {best:+.4f}", flush=True)
    print("oracle ceilings (M138): 3x3 +0.032, 4x4 +0.046, 8x8 +0.069, "
          "24x24 +0.127; leader about +0.02; current submission -0.141")


if __name__ == "__main__":
    main()
