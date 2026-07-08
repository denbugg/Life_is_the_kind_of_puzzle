"""Train PairwiseNet with InfoNCE over sampled candidates.
Fragments arrive in correct order (index == grid position): right neighbor of
position a is a+1, down neighbor is a+GRID. For each anchor we score the true
neighbor against M-1 random candidates and apply softmax cross-entropy."""
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import GRID, NFRAG, FS, CKPT_DIR, SEED
from imgio import train_val_split
from datasets import CompatDataset
from models import PairwiseNet, count_params

DEV = "cuda"


def make_pairs(frags, anchors, cand, transpose):
    """frags:(N,3,20,20) anchors:(A,) cand:(A,M) -> pairs (A*M,3,20,40)."""
    f = frags.transpose(-1, -2) if transpose else frags
    A, M = cand.shape
    left = f[anchors][:, None].expand(A, M, 3, FS, FS)
    right = f[cand]
    pair = torch.cat([left, right], dim=-1)         # (A,M,3,20,40)
    return pair.reshape(A * M, 3, FS, 2 * FS)


def build(frags, has_right, offset, nA, M, gen):
    N = frags.shape[0]
    idx = has_right[torch.randint(len(has_right), (nA,), generator=gen, device=DEV)]
    true = idx + offset
    cand = torch.randint(N, (nA, M), generator=gen, device=DEV)
    cand[:, 0] = true
    return idx, cand


def step_loss(model, frags_b, ph, pv, nA, M, gen):
    """Build ALL pairs for the step and score them in ONE model() call. This is
    critical for DataParallel: it replicates the model once and splits the big
    pair batch across GPUs, instead of paying replication overhead per sub-call."""
    chunks = []
    for frags in frags_b:
        for has, off, tr in ((ph, 1, False), (pv, GRID, True)):
            a, cand = build(frags, has, off, nA, M, gen)
            chunks.append(make_pairs(frags, a, cand, tr))     # (nA*M, 3, 20, 40)
    G = len(chunks)
    logits = model(torch.cat(chunks, 0)).view(G, nA, M)       # one call; order preserved
    tgt = torch.zeros(G * nA, dtype=torch.long, device=DEV)
    loss = F.cross_entropy(logits.reshape(G * nA, M), tgt)
    acc = (logits.argmax(-1) == 0).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(model, vdl, ph, pv, gen, M=32, nA=48, n=16):
    """Small, memory-safe val (same pair count scale as a train step)."""
    model.eval(); A = []
    for k, fb in enumerate(vdl):
        if k >= n: break
        _, a = step_loss(model, fb.to(DEV), ph, pv, nA, M, gen)
        A.append(a)
    model.train()
    return float(np.mean(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--nA", type=int, default=48)
    ap.add_argument("--M", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--real_prob", type=float, default=0.6)   # train on REAL degradation, not just synthetic
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)   # differ across ensemble members
    ap.add_argument("--tag", default="pair")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=DEV); gen.manual_seed(args.seed)

    trn, val = train_val_split()
    dl = DataLoader(CompatDataset(trn, args.real_prob), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=True,
                    prefetch_factor=3, pin_memory=True)
    vdl = DataLoader(CompatDataset(val, real_prob=1.0), batch_size=args.bs, shuffle=False,
                     num_workers=3, persistent_workers=True)
    p = np.arange(NFRAG)
    ph = torch.tensor(p[p % GRID != GRID - 1], device=DEV)
    pv = torch.tensor(p[p // GRID != GRID - 1], device=DEV)

    model = PairwiseNet().to(DEV)
    print(f"PairwiseNet params: {count_params(model):,}", flush=True)
    ngpu = torch.cuda.device_count()
    print(f"CUDA devices: {ngpu} -> {[torch.cuda.get_device_name(i) for i in range(ngpu)]}", flush=True)
    if ngpu > 1:                                    # use BOTH T4s: DataParallel splits the pair batch
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {ngpu} GPUs enabled", flush=True)
    def state_dict():
        return (model.module if hasattr(model, "module") else model).state_dict()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    step = 0; t0 = time.time(); best = 0.0; ema = 0.0
    while step < args.steps:
        for fb in dl:
            fb = fb.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, acc = step_loss(model, fb, ph, pv, args.nA, args.M, gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ema = 0.98 * ema + 0.02 * acc if step else acc
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.3f} acc@{args.M} {ema:.3f} "
                      f"lr {sched.get_last_lr()[0]:.1e} {(time.time()-t0)/max(1,step):.2f}s/it", flush=True)
            if step % 500 == 0 and step > 0:
                va = evaluate(model, vdl, ph, pv, gen)
                print(f"  [VAL real] acc@48 {va:.3f}", flush=True)
                torch.save({"model": state_dict(), "step": step, "val": va},
                           os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
                if va > best:
                    best = va
                    torch.save({"model": state_dict(), "step": step, "val": va},
                               os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            step += 1
            if step >= args.steps: break
    torch.save({"model": state_dict(), "step": step}, os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
    print(f"done. best val acc@48={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
