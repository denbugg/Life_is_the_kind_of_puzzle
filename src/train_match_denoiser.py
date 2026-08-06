"""Train a tiny denoiser used only to improve matching, not final image quality."""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_TGT, CKPT_DIR, SEED
from imgio import load, to_frags, train_val_split
from distort import distort_frags
from match_preprocess import MatchDenoiser

DEV = "cuda"


class FragmentDenoiseDataset(Dataset):
    """Loads one clean image and samples several fragments from it.

    Returning fragment packs avoids decoding the same PNG once per 20x20 tile.
    The training loop flattens (image_batch, frags_per_image) into a fragment batch.
    """
    def __init__(self, names, frags_per_image=32):
        self.names = names
        self.fpi = frags_per_image

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        nm = self.names[i % len(self.names)]
        clean_frags = to_frags(load(os.path.join(TRAIN_TGT, nm)))
        idx = rng.integers(0, len(clean_frags), size=self.fpi)
        clean = clean_frags[idx]
        dist = distort_frags(clean, rng)
        x = torch.from_numpy(np.ascontiguousarray(dist)).permute(0, 3, 1, 2).float() / 255
        y = torch.from_numpy(np.ascontiguousarray(clean)).permute(0, 3, 1, 2).float() / 255
        return x, y


def flatten_pack(x):
    if x.dim() == 5:
        b, k = x.shape[:2]
        return x.reshape(b * k, *x.shape[2:])
    return x


def grad_loss(pred, target):
    px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


@torch.no_grad()
def evaluate(model, dl, n=20):
    model.eval()
    vals = []
    for k, (x, y) in enumerate(dl):
        if k >= n:
            break
        x = flatten_pack(x).to(DEV, non_blocking=True)
        y = flatten_pack(y).to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x).float()
        vals.append(F.l1_loss(p, y).item())
    model.train()
    return float(np.mean(vals)) if vals else 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256, help="effective fragment batch size")
    ap.add_argument("--frags_per_image", type=int, default=32)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="matchden")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    trn, val = train_val_split()
    image_bs = max(1, args.bs // max(1, args.frags_per_image))
    dl = DataLoader(FragmentDenoiseDataset(trn, args.frags_per_image), batch_size=image_bs,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0, pin_memory=True)
    vdl = DataLoader(FragmentDenoiseDataset(val, min(8, args.frags_per_image)), batch_size=image_bs,
                     shuffle=False, num_workers=max(1, min(3, args.workers)), pin_memory=True)
    print(f"effective fragment bs={image_bs * args.frags_per_image} "
          f"({image_bs} images x {args.frags_per_image} frags)", flush=True)

    model = MatchDenoiser(base=args.base, blocks=args.blocks).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best = 1e9
    step = 0
    t0 = time.time()
    while step < args.steps:
        for x, y in dl:
            x = flatten_pack(x).to(DEV, non_blocking=True)
            y = flatten_pack(y).to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(x)
                loss = F.l1_loss(p.float(), y) + 0.25 * grad_loss(p.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.1e} {(time.time()-t0)/max(1,step):.3f}s/it",
                      flush=True)
            if step % 500 == 0 and step > 0:
                va = evaluate(model, vdl)
                print(f"  [VAL] frag_l1={va:.5f}", flush=True)
                torch.save({"model": model.state_dict(), "step": step, "val": va,
                            "base": args.base, "blocks": args.blocks},
                           os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
                if va < best:
                    best = va
                    torch.save({"model": model.state_dict(), "step": step, "val": va,
                                "base": args.base, "blocks": args.blocks},
                               os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            step += 1
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "step": step, "val": best,
                "base": args.base, "blocks": args.blocks},
               os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
    print(f"done. best val frag_l1={best:.5f}", flush=True)


if __name__ == "__main__":
    main()
