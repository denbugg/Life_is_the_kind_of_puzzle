"""Train the field diffusion, and read it as PLACEMENT rather than as loss.

The number that decides this is not the denoising loss. M471 measured the whole
curve that matters: the true 96x96 picture places 0.4520 of the fragments and
scores SSIM 0.4292; the same picture carrying 32 RMSE of noise places 0.3637 and
scores 0.3902, above the competitor; at 64 RMSE it places 0.2261 and scores
0.3181, which is BELOW the flat fill of 0.3514 and therefore worthless. So the
model has one target -- get the picture within about 32 grey levels a sub-cell --
and every evaluation here reports RMSE against the true map, the placement the
map supports, and nothing else.

The control is the point of the exercise. `--mode regress` trains the identical
network with an MSE loss to predict the picture in one shot, which is what
`coarse_field` did and what M387 found collapsed to a near-constant map: spread
2.05 against the true cells' 57.37, placing 0.0027. If the regression collapses
here too and the sampler does not, the diagnosis in M471 is confirmed on a model
of ordinary size, and the difference is the loss and not the capacity.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import CACHE_DIR, CKPT_DIR
from field_diffusion import (GRID, N, RES, FieldDiffusion, Schedule, cell_desc,
                             sample)

HELD = 300


def placement(pred, bag4):
    """Hungarian on the 4x4 description, exactly as M428 and M471 score it."""
    A = cell_desc(pred)
    B = bag4.reshape(N, -1).astype(np.float64)
    C = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T)
    r, c = linear_sum_assignment(C)
    o = np.empty(N, np.int64)
    o[r] = c
    return float((o == np.arange(N)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="field_cache.npz")
    ap.add_argument("--mode", choices=("diffusion", "regress"),
                    default="diffusion")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--eval-boards", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--train-boards", type=int, default=0,
                    help="limit the FIT prefix; 0 uses every non-held board")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="cap batches per epoch for a reproducible smoke run")
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, cuda, or an explicit torch device")
    ap.add_argument("--snap-from", type=float, default=1.01,
                    help="fraction of the schedule after which each step is "
                         "projected onto the bag; above 1 disables it, which is "
                         "the honest reading of the prior on its own")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="field_diff.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    z = np.load(Path(CACHE_DIR) / a.cache)
    bag8, bag4, stats, pic = z["bag8"], z["bag4"], z["stats"], z["pic"]
    n = len(pic)
    n_train = n - HELD
    if a.train_boards:
        n_train = min(n_train, a.train_boards)
    tr = np.arange(0, n_train)
    ev = np.arange(n - HELD, n)[:a.eval_boards]
    print(f"{len(tr)} train boards, {len(ev)} evaluated, picture at "
          f"{RES}x{RES}", flush=True)

    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if a.device == "auto" else a.device)
    model = FieldDiffusion(a.d, a.layers, a.heads, a.base).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(Path(CKPT_DIR) / a.resume,
                                         map_location=dev)["model"])
        print(f"resumed from {a.resume}", flush=True)
    npar = sum(p.numel() for p in model.parameters())
    print(f"{npar/1e6:.1f}M parameters on {dev}", flush=True)
    sched = Schedule(a.steps, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    batches_per_epoch = len(tr) // a.batch
    if a.max_batches:
        batches_per_epoch = min(batches_per_epoch, a.max_batches)
    if batches_per_epoch < 1:
        raise ValueError("training split must contain at least one full batch")
    total = a.epochs * batches_per_epoch
    lrs = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(
        total, 1), pct_start=0.05)
    scaler = torch.amp.GradScaler(dev)

    def bag_of(idx):
        """The encoder consumes 20x20 fragments; the cache holds the 8x8 view,
        so it is expanded back and the encoder's own pooling is a no-op."""
        b8 = torch.from_numpy(bag8[idx].astype(np.float32)).to(dev)
        st = torch.from_numpy(stats[idx].astype(np.float32)).to(dev)
        return b8, st

    def encode(idx):
        b8, st = bag_of(idx)
        b, nn_ = b8.shape[:2]
        # channel-major, matching BagEncoder.features, which pools a
        # (3, 20, 20) tensor -- the cache stores the view spatially and
        # feeding it in that order would silently scramble the input
        v = b8.permute(0, 1, 4, 2, 3).reshape(b, nn_, -1) / 127.5 - 1.0
        m = st[..., :3] / 127.5 - 1.0
        s = st[..., 3:] / 127.5
        return model.bag.out(model.bag.enc(model.bag.inp(
            torch.cat([v, m, s], -1))))

    # the oracle anchor: what the TRUE picture places, on these very boards
    anchor = float(np.mean([placement(pic[i].astype(np.float64), bag4[i])
                            for i in ev]))
    print(f"anchor: the true picture places {anchor:.4f} on these boards "
          f"(M471 reads 0.4520)", flush=True)

    def evaluate():
        model.eval()
        rm, pl, condition_delta, wrong_pl = [], [], [], []
        with torch.no_grad():
            for ev_no, i in enumerate(ev):
                ctx = encode(np.array([i]))
                wrong_ctx = encode(np.array([ev[(ev_no + 1) % len(ev)]]))
                gen = torch.Generator(device=dev).manual_seed(a.seed + int(i))
                start = torch.randn(1, 3, RES, RES, device=dev,
                                    generator=gen)

                def predict(given_ctx):
                    if a.mode == "regress":
                        return model.net(
                            torch.zeros(1, 3, RES, RES, device=dev),
                            torch.zeros(1, dtype=torch.long, device=dev),
                            given_ctx)
                    x = start.clone()
                    ts = torch.linspace(sched.T - 1, 0,
                                        a.sample_steps).long().to(dev)
                    for k, t in enumerate(ts):
                        eps = model.net(x, t.expand(1), given_ctx)
                        ab = sched.abar[t]
                        x0 = ((x - (1 - ab).sqrt() * eps)
                              / ab.sqrt()).clamp(-1, 1)
                        if k + 1 < len(ts):
                            an = sched.abar[ts[k + 1]]
                            e2 = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
                            x = an.sqrt() * x0 + (1 - an).sqrt() * e2
                    return x0

                x0 = predict(ctx)
                wrong = predict(wrong_ctx)
                img = (x0[0].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5
                wrong_img = ((wrong[0].permute(1, 2, 0).float().cpu().numpy()
                              + 1.0) * 127.5)
                rm.append(float(np.sqrt(((img - pic[i]) ** 2).mean())))
                pl.append(placement(img.astype(np.float64), bag4[i]))
                condition_delta.append(float(np.sqrt(
                    ((img - wrong_img) ** 2).mean())))
                wrong_pl.append(placement(wrong_img.astype(np.float64), bag4[i]))
        model.train()
        return (float(np.mean(rm)), float(np.mean(pl)), float(img.std()),
                float(np.mean(condition_delta)), float(np.mean(wrong_pl)))

    r, p, sd, cd, wp = evaluate()
    print(f"[init] RMSE {r:.2f}  placed {p:.4f}  map spread {sd:.2f}  "
          f"bag_delta {cd:.2f}  wrong_bag {wp:.4f}", flush=True)
    best = p
    for ep in range(a.epochs):
        perm = np.random.permutation(tr)
        run, t0 = [], time.time()
        for batch_no, k in enumerate(range(0, len(perm) - a.batch + 1,
                                            a.batch)):
            if a.max_batches and batch_no >= a.max_batches:
                break
            idx = np.sort(perm[k:k + a.batch])
            x0 = torch.from_numpy(pic[idx].astype(np.float32)).to(dev)
            x0 = (x0.permute(0, 3, 1, 2) / 127.5 - 1.0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16):
                ctx = encode(idx)
                if a.mode == "regress":
                    out = model.net(torch.zeros_like(x0),
                                    torch.zeros(len(idx), dtype=torch.long,
                                                device=dev), ctx)
                    loss = torch.nn.functional.mse_loss(out, x0)
                else:
                    t = torch.randint(0, sched.T, (len(idx),), device=dev)
                    eps = torch.randn_like(x0)
                    xt = sched.add_noise(x0, t, eps)
                    loss = torch.nn.functional.mse_loss(
                        model.net(xt, t, ctx), eps)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            lrs.step()
            run.append(float(loss.detach()))
        msg = (f"[epoch {ep}] loss {np.mean(run):.4f} "
               f"{(time.time()-t0)/60:.1f} min")
        if ep % a.eval_every == 0 or ep == a.epochs - 1:
            r, p, sd, cd, wp = evaluate()
            msg += (f"  RMSE {r:.2f}  placed {p:.4f}  spread {sd:.2f} "
                    f"bag_delta {cd:.2f} wrong_bag {wp:.4f}")
            torch.save({"model": model.state_dict(),
                        "args": {k: getattr(a, k) for k in
                                 ("d", "layers", "heads", "base", "steps",
                                  "mode")},
                        "placed": p, "rmse": r},
                       Path(CKPT_DIR) / a.out)
            if p > best:
                best = p
                torch.save({"model": model.state_dict(),
                            "args": {k: getattr(a, k) for k in
                                     ("d", "layers", "heads", "base", "steps",
                                      "mode")},
                            "placed": p, "rmse": r},
                           Path(CKPT_DIR) / (a.out[:-3] + "_best.pt"))
                msg += "  *"
        print(msg, flush=True)
    print(f"best placement {best:.4f}; the true picture reaches {anchor:.4f} "
          f"and 32 RMSE of noise still reaches 0.3637")


if __name__ == "__main__":
    main()
