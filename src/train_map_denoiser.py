"""Train the 96x96 picture denoiser that the plug-and-play assembler needs.

Nothing here is scored in SSIM and nothing here touches a submitted pixel: the
denoiser is a PRIOR, and the only thing that matters about it is whether the
assembly loop built on it moves placement.

Trained across the whole noise range the annealing schedule will visit, with the
level handed in as an input plane so one set of weights covers all of it. The
held-out read is the denoising error at a few fixed levels, purely as a sanity
check that it learned the picture statistics of THIS corpus.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CACHE_DIR, CKPT_DIR
from map_denoiser import MapDenoiser

HELD = 300
LEVELS = (0.1, 0.25, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="field_cache.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--sigma-max", type=float, default=1.2,
                    help="on the [-1,1] picture scale; 1.0 is about 127 grey "
                         "levels, well past where structure has to be invented")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="map_denoiser.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    z = np.load(Path(CACHE_DIR) / a.cache)
    pic = z["pic"]
    tr = np.arange(len(pic) - HELD)
    ev = np.arange(len(pic) - HELD, len(pic))[:64]
    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if a.device == "auto" else a.device)
    model = MapDenoiser(a.base, a.blocks).to(dev)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters "
          f"on {dev}; {len(tr)} train pictures", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=max(a.epochs * (len(tr) // a.batch), 1),
        pct_start=0.05)
    scaler = torch.amp.GradScaler(dev)

    def to_t(idx):
        x = torch.from_numpy(pic[idx].astype(np.float32)).to(dev)
        return x.permute(0, 3, 1, 2) / 127.5 - 1.0

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = []
        for s in LEVELS:
            errs = []
            for k in range(0, len(ev), 32):
                x = to_t(ev[k:k + 32])
                g = torch.Generator(device=dev).manual_seed(7)
                n = torch.randn(x.shape, device=dev, generator=g) * s
                y = model(x + n, torch.full((len(x),), s, device=dev))
                errs.append(float(((y - x) ** 2).mean().sqrt()) * 127.5)
            out.append(np.mean(errs))
        model.train()
        return out

    print("[init] held-out RMSE in grey levels at sigma "
          + " ".join(f"{s:.2f}" for s in LEVELS) + ": "
          + " ".join(f"{v:.1f}" for v in evaluate()), flush=True)
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        run, t0 = [], time.time()
        for k in range(0, len(perm) - a.batch + 1, a.batch):
            x = to_t(np.sort(perm[k:k + a.batch]))
            if rng.random() < 0.5:
                x = torch.flip(x, [3])
            s = torch.rand(len(x), device=dev) * a.sigma_max
            xn = x + torch.randn_like(x) * s.reshape(-1, 1, 1, 1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16):
                loss = F.mse_loss(model(xn, s), x)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(float(loss.detach()))
        msg = f"[epoch {ep}] loss {np.mean(run):.5f} {(time.time()-t0)/60:.1f} min"
        if ep % 3 == 0 or ep == a.epochs - 1:
            msg += "  RMSE " + " ".join(f"{v:.1f}" for v in evaluate())
            torch.save({"model": model.state_dict(),
                        "args": {"base": a.base, "blocks": a.blocks}},
                       Path(CKPT_DIR) / a.out)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
