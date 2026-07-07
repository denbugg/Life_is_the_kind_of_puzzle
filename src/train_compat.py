"""Train CompatNet with symmetric InfoNCE over in-image candidates.
Key logged metric: neighbor top-1 accuracy (does the true right/below neighbor
rank #1 among all 576 fragments?) -> directly predicts puzzle solvability.
"""
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import GRID, NFRAG, FS, CKPT_DIR, SEED
from imgio import train_val_split
from datasets import CompatDataset
from models import CompatNet, count_params

DEV = "cuda"


def grid_targets():
    p = np.arange(NFRAG)
    h = p[p % GRID != GRID - 1]
    v = p[p // GRID != GRID - 1]
    return (torch.tensor(h, device=DEV), torch.tensor(h + 1, device=DEV),
            torch.tensor(v, device=DEV), torch.tensor(v + GRID, device=DEV))


def losses_and_acc(model, frags, ha, ht, va, vt):
    B, N = frags.shape[:2]
    eR, eL, eT, eB = model.embed(frags.view(B * N, 3, FS, FS).to(DEV, non_blocking=True))
    d = eR.shape[-1]
    eR, eL, eT, eB = [e.view(B, N, d) for e in (eR, eL, eT, eB)]
    scale = model.logit_scale.exp().clamp(max=100)
    neg_inf = torch.finfo(eR.dtype).min
    loss = 0.0; hacc = 0.0; vacc = 0.0
    for b in range(B):
        lh = scale * (eR[b] @ eL[b].t())
        lv = scale * (eB[b] @ eT[b].t())
        idx = torch.arange(N, device=DEV)
        lh[idx, idx] = neg_inf; lv[idx, idx] = neg_inf
        loss = loss + F.cross_entropy(lh[ha], ht) + F.cross_entropy(lh.t()[ht], ha)
        loss = loss + F.cross_entropy(lv[va], vt) + F.cross_entropy(lv.t()[vt], va)
        hacc += (lh[ha].argmax(1) == ht).float().mean().item()
        vacc += (lv[va].argmax(1) == vt).float().mean().item()
    return loss / B, hacc / B, vacc / B


@torch.no_grad()
def evaluate(model, loader, ha, ht, va, vt, n=40):
    model.eval(); H = []; V = []
    for k, frags in enumerate(loader):
        if k >= n: break
        with torch.autocast("cuda", dtype=torch.float16):
            _, h, v = losses_and_acc(model, frags, ha, ht, va, vt)
        H.append(h); V.append(v)
    model.train()
    return float(np.mean(H)), float(np.mean(V))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--real_prob", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="compat")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    trn, val = train_val_split()
    dl = DataLoader(CompatDataset(trn, args.real_prob), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=True,
                    prefetch_factor=2, pin_memory=True)
    vdl = DataLoader(CompatDataset(val, real_prob=1.0), batch_size=args.bs, shuffle=False,
                     num_workers=4, persistent_workers=True)
    model = CompatNet().to(DEV)
    print(f"CompatNet params: {count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    ha, ht, va, vt = grid_targets()

    step = 0; t0 = time.time(); best = 0.0
    ema_h = ema_v = 0.0
    while step < args.steps:
        for frags in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                loss, h, v = losses_and_acc(model, frags, ha, ht, va, vt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            ema_h = 0.98 * ema_h + 0.02 * h if step else h
            ema_v = 0.98 * ema_v + 0.02 * v if step else v
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.3f} "
                      f"H@1 {ema_h:.3f} V@1 {ema_v:.3f} lr {sched.get_last_lr()[0]:.1e} "
                      f"{(time.time()-t0)/max(1,step):.2f}s/it", flush=True)
            if step % 500 == 0 and step > 0:
                vh, vv = evaluate(model, vdl, ha, ht, va, vt)
                print(f"  [VAL real] H@1 {vh:.3f} V@1 {vv:.3f}", flush=True)
                score = 0.5 * (vh + vv)
                torch.save({"model": model.state_dict(), "step": step, "val": score},
                           os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
                if score > best:
                    best = score
                    torch.save({"model": model.state_dict(), "step": step, "val": score},
                               os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
                    print(f"  saved best val={best:.3f}", flush=True)
            step += 1
            if step >= args.steps: break
    torch.save({"model": model.state_dict(), "step": step},
               os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
    print(f"done. best val H/V@1 = {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
