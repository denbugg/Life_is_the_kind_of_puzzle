"""Train RestoreNet on (distorted-in-correct-order, clean) pairs.
Loss = MS-SSIM + L1. Validation reports the true skimage SSIM metric, both for
the distorted input (baseline) and the restored output (lift)."""
import os, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim
from config import CKPT_DIR, TRAIN_TGT, SEED
from imgio import load, to_frags, from_frags, train_val_split
from datasets import RestoreDataset, real_recon
from distort import distort_frags
from models import RestoreNet, restore_loss, count_params

DEV = "cuda"


@torch.no_grad()
def evaluate(model, val_names, n=48, use_real=True):
    model.eval()
    rng = np.random.default_rng(0)
    base, rest = [], []
    for nm in val_names[:n]:
        clean = load(os.path.join(TRAIN_TGT, nm))
        rr = real_recon(nm) if use_real else None
        if rr is not None:
            dist = from_frags(rr[0])
        else:
            dist = from_frags(distort_frags(to_frags(clean), rng))
        x = torch.from_numpy(dist).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255
        with torch.autocast("cuda", dtype=torch.float16):
            y = model(x)
        out = (y[0].permute(1, 2, 0).float().clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        base.append(sk_ssim(clean, dist.astype(np.uint8), channel_axis=2, data_range=255))
        rest.append(sk_ssim(clean, out, channel_axis=2, data_range=255))
    model.train()
    return float(np.mean(base)), float(np.mean(rest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--real_prob", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--tag", default="restore")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    trn, val = train_val_split()
    dl = DataLoader(RestoreDataset(trn, args.crop, args.real_prob), batch_size=args.bs,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=True, prefetch_factor=4, pin_memory=True)
    model = RestoreNet(base=args.base).to(DEV)
    print(f"RestoreNet params: {count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    step = 0; t0 = time.time(); best = 0.0; ema = 0.0
    while step < args.steps:
        for di, cl in dl:
            di, cl = di.to(DEV, non_blocking=True), cl.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(di)
                loss = restore_loss(pred.float(), cl.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            ema = 0.98 * ema + 0.02 * loss.item() if step else loss.item()
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {ema:.4f} lr {sched.get_last_lr()[0]:.1e} "
                      f"{(time.time()-t0)/max(1,step):.2f}s/it", flush=True)
            if step % 1000 == 0 and step > 0:
                b, r = evaluate(model, val)
                print(f"  [VAL] SSIM base(distorted) {b:.4f} -> restored {r:.4f}  (+{r-b:.4f})", flush=True)
                torch.save({"model": model.state_dict(), "step": step, "val": r, "base": args.base},
                           os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
                if r > best:
                    best = r
                    torch.save({"model": model.state_dict(), "step": step, "val": r, "base": args.base},
                               os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
                    print(f"  saved best restored SSIM={best:.4f}", flush=True)
            step += 1
            if step >= args.steps: break
    torch.save({"model": model.state_dict(), "step": step, "base": args.base},
               os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
    print(f"done. best restored SSIM={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
