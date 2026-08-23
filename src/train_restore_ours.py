"""Train the restoration net on the boards our own solver produces.

Why this is not the R5 net again
--------------------------------
R5 was trained on a CORRECTLY ordered corrupted board against the clean image --
a pure denoiser -- and the submission applies it to a scrambled one.  It has
never seen its own input distribution, and it is still worth 0.11 SSIM against
assembly's 0.024 (M133).  It is the one component of the chain that was
inherited rather than built, and the only one nobody has ever fitted to the data
it actually receives.

What the target is
------------------
The clean image, at full 480x480, under MS-SSIM + L1.  Not a denoising proxy:
the loss is the metric.  The input carries very little high-frequency
information about the target -- our layout places 0.6% of tiles correctly -- so
a well-fitted model will smooth heavily, and that is the data speaking rather
than a metric trick.  What is NOT allowed is discarding the input: the log
therefore prints the output's spatial standard deviation beside the score, and a
model whose output variance collapses towards a flat fill is a failure however
well it scores.

Baselines this has to beat, measured on 8 held-out boards (M134):
  our chain today, R5 + NLM       0.2145
  plain Gaussian blur sigma 4     0.2268
  R5 then Gaussian blur sigma 4   0.2388
  plain Gaussian blur sigma 6     0.2513
  R5 then Gaussian blur sigma 6   0.2591
  perfect restoration, our layout 0.1395   (honest per-tile restoration of a
                                            scrambled board scores LOWER than
                                            smoothing it -- the layout is what
                                            is missing, not tile quality)
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

import infer_rank96 as rank96
from config import CACHE_DIR, CKPT_DIR, FS, GRID as G, TRAIN_INP, TRAIN_TGT
import torch.nn.functional as F

from models import RestoreNet, restore_loss
from restore_tile import to_frags


def detail_std(x, factor=16):
    """Spread of what survives a low-pass -- the texture a viewer reads.

    M145 measured the exchange rate between this and SSIM, and M174 showed the
    whole submission decision lives on that curve: the layout is worth 0.0005 at
    full restoration while the restoration strength is worth 0.13.  Choosing the
    operating point by BLENDING two networks is leaving score on the table; this
    makes the floor a constraint the network is trained under.

    A box low-pass at 1/16 resolution stands in for the 12 px Gaussian the
    report uses -- close enough to constrain, and far cheaper to differentiate.
    """
    low = F.interpolate(F.avg_pool2d(x, factor), size=x.shape[-2:],
                        mode="bilinear", align_corners=False)
    return (x - low).std()


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


def assemble(tiles, lay):
    x = np.clip(tiles[np.asarray(lay)], 0, 255).astype(np.uint8)
    return x.reshape(G, G, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(G * FS, G * FS, 3)


def load_layouts(paths):
    names, lays = [], []
    for p in paths:
        b = np.load(p, allow_pickle=True)
        names += [str(x) for x in b["names"]]
        lays.append(b["lay"])
    return names, np.concatenate(lays)


class Boards(Dataset):
    """Assembled dirty board -> clean target, with D4 augmentation."""

    def __init__(self, names, lays, inv_by_name, augment=True):
        self.names, self.lays = names, lays
        self.inv, self.augment = inv_by_name, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = self.names[k]
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            self.inv[nm].astype(np.int64)]
        x = assemble(tiles, self.lays[k].astype(np.int64))
        y = load_rgb(Path(TRAIN_TGT) / nm)
        if self.augment:
            r = np.random.randint(4)
            if r:
                x, y = np.rot90(x, r), np.rot90(y, r)
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1).float() / 255.0
        return x, y


@torch.no_grad()
def evaluate(model, names, lays, inv_by_name, dev, limit, nlm=False, swap=False):
    """Gain over the flat fill, plus the two numbers that catch a dead run.

    Absolute SSIM is not reported: M137 established that on this task it mostly
    measures proximity to a constant, so the flat fill at our own tiles' mean
    colour is the zero of the scale.

    `swap` feeds board k the tiles of board k+1 while scoring against board k's
    target.  That is the acceptance test for the whole submission: a fill's
    output does not change when the input does, a real restorer's does.
    """
    model.eval()
    gains, sd, sat, ss_nlm = [], [], [], []
    n = min(limit, len(names))
    for k in range(n):
        nm = names[k]
        tiles = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)[
            inv_by_name[nm].astype(np.int64)]
        src = k if not swap else (k + 1) % n
        snm = names[src]
        stiles = tiles if src == k else to_frags(
            load_rgb(Path(TRAIN_INP) / snm)).astype(np.float32)[
                inv_by_name[snm].astype(np.int64)]
        img = assemble(stiles, lays[src].astype(np.int64))
        tgt = load_rgb(Path(TRAIN_TGT) / nm)
        t = torch.from_numpy(img).to(dev, torch.float32).permute(2, 0, 1)[None] / 255.0
        raw = model(t, clamp=False)
        sat.append(float(((raw <= 0.0) | (raw >= 1.0)).float().mean()))
        o = raw.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.rint(o * 255.0).clip(0, 255).astype(np.uint8)
        flat = np.zeros_like(tgt)
        flat[:] = np.rint(tiles.reshape(-1, 3).mean(0)).clip(0, 255).astype(np.uint8)
        base = float(ssim_fn(flat, tgt, channel_axis=2, data_range=255))
        gains.append(float(ssim_fn(out, tgt, channel_axis=2, data_range=255)) - base)
        f = out.astype(np.float32)
        sd.append(float((f - cv2.GaussianBlur(f, (0, 0), 12.0)).std()))
        if nlm:
            ss_nlm.append(float(ssim_fn(rank96.fixed_nlm(out), tgt,
                                        channel_axis=2, data_range=255)) - base)
    model.train()
    return (float(np.mean(gains)), float(np.mean(sd)), float(np.mean(sat)),
            float(np.mean(ss_nlm)) if ss_nlm else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-layouts", nargs="+",
                    default=[str(Path(CACHE_DIR) / f"layouts_tr_{i}.npz")
                             for i in range(3)])
    ap.add_argument("--val-layouts", default=str(Path(CACHE_DIR) / "layouts_val.npz"))
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--init", default="", help="warm start from this checkpoint path")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.84,
                    help="MS-SSIM weight in the loss; the rest is L1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=24)
    ap.add_argument("--detail-floor", type=float, default=0.0,
                    help="keep the output's texture at or above this level; "
                         "0 disables the constraint")
    ap.add_argument("--detail-weight", type=float, default=0.02)
    ap.add_argument("--out", default="restore_ours_v1.pt")
    a = ap.parse_args()

    dev = "cuda"
    blob = np.load(Path(CACHE_DIR) / "restore_labels.npz", allow_pickle=True)
    inv_by_name = {str(n): v for n, v in zip(blob["names"], blob["inv"])}

    tr_names, tr_lays = load_layouts(a.train_layouts)
    va_names, va_lays = load_layouts([a.val_layouts])
    print(f"train {len(tr_names)} boards, val {len(va_names)}", flush=True)

    model = RestoreNet(base=a.base, depth=a.depth).to(dev)
    # Start as the identity, so step 0 emits the assembled board unchanged and
    # every later number is an improvement on a known baseline.  It also keeps
    # the run alive: a randomly initialised head is already comparable in
    # magnitude to the image itself, which drove M147 straight into the dead
    # clamp before it had taken 200 steps.
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    if a.init:
        pay = torch.load(a.init, map_location=dev, weights_only=False)
        st = (pay.get("model") or pay.get("model_state_dict")
              or pay.get("state_dict") or pay)
        model.load_state_dict(st, strict=True)
        print(f"warm start from {a.init}", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"RestoreNet base {a.base} depth {a.depth}: {n_par/1e6:.2f}M", flush=True)

    dl = DataLoader(Boards(tr_names, tr_lays, inv_by_name), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best, step, t0 = -1.0, 0, time.time()
    run = []
    while step < a.steps:
        for x, y in dl:
            if step >= a.steps:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", torch.float16):
                pred = model(x, clamp=False)
            # the loss stays in fp32: MS-SSIM divides by local variances, which
            # underflow in half precision and take the whole loss to nan
            loss = restore_loss(pred.float(), y, a.alpha)
            if a.detail_floor > 0:
                # one-sided: nothing is asked of the model until its output is
                # smoother than the floor, so the term costs nothing above it
                d = detail_std(pred.float().clamp(0, 1)) * 255.0
                loss = loss + a.detail_weight * torch.relu(a.detail_floor - d) ** 2
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
                s, sd, sat, _ = evaluate(model, va_names, va_lays, inv_by_name,
                                         dev, a.eval_boards)
                flags = ""
                if sat > 0.5:
                    flags += "  <-- SATURATED, gradient is dead"
                if sd < 1.0:
                    flags += "  <-- collapsing towards a flat fill"
                print(f"  eval step {step}: gain over flat {s:+.4f}  "
                      f"detail {sd:.1f}  clipped {sat:.3f}{flags}", flush=True)
                if s > best:
                    best = s
                    torch.save({"model": model.state_dict(), "args": vars(a),
                                "step": step,
                                "eval": {"gain": s, "std": sd, "clipped": sat}},
                               Path(CKPT_DIR) / a.out)
    s, sd, sat, s_nlm = evaluate(model, va_names, va_lays, inv_by_name, dev,
                                 a.eval_boards, nlm=True)
    sw, sw_sd, _, _ = evaluate(model, va_names, va_lays, inv_by_name, dev,
                               a.eval_boards, swap=True)
    print(f"\nfinal: gain over flat {s:+.4f}  (+NLM {s_nlm:+.4f})  "
          f"out_std {sd:.1f}  clipped {sat:.3f}  best {best:+.4f}", flush=True)
    print(f"swapped input: gain {sw:+.4f}, out_std {sw_sd:.1f} -- the honesty "
          f"test, this must be clearly WORSE than {s:+.4f}", flush=True)


if __name__ == "__main__":
    main()
