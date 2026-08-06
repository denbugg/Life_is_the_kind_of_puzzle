"""Train the DINOv2 + Sinkhorn AssemblerNet. DINOv2 features are PRECOMPUTED into an
in-memory cache (frozen encoder never in the loop); the assembler trains on cached
features with a random shuffle giving perfect permutation labels. See NEW_CONCEPT.md.

  # overfit sanity (THE gate): can it learn to assemble 8 images at all?
  python train_assembler.py --overfit 8 --steps 1500
  # small generalisation run:
  python train_assembler.py --train_n 800 --steps 4000 --evaln 12
"""
import os, time, argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as sk_ssim
from config import GRID, NFRAG, TRAIN_INP, TRAIN_TGT, CACHE_DIR, CKPT_DIR, SEED
from imgio import load, to_frags, assemble, train_val_split
from distort import distort_frags
from dino import DinoEncoder
from assembler import AssemblerNet, assemble_loss, count_params

DEV = "cuda"


def build_cache(dino, names, kind):
    """kind='synth': distort clean in GRID order (perfect labels).
       kind='real' : to_frags(input) in INPUT order (for eval vs recover GT)."""
    cache = {}
    for k, nm in enumerate(names):
        if kind == "synth":
            frags = distort_frags(to_frags(load(os.path.join(TRAIN_TGT, nm))), np.random.default_rng(k + 1))
        else:
            frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        cache[nm] = dino.encode(frags)                       # (576, feat) cpu, order per `kind`
        if (k + 1) % 200 == 0:
            print(f"  cached {k+1}/{len(names)} ({kind})", flush=True)
    return cache


@torch.no_grad()
def eval_real(model, val_names, real_cache, gt_inv):
    model.eval()
    accs, ssims = [], []
    for nm in val_names:
        feats = real_cache[nm][None].to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            logP, _ = model(feats)
        P = logP[0].float().cpu().numpy()                    # (frag, cell)
        r, c = linear_sum_assignment(-P)
        place = np.empty(NFRAG, np.int64); place[c] = r       # cell -> fragment
        inv = gt_inv[nm].astype(np.int64)
        accs.append(float(np.mean(place == inv)))
        frags = to_frags(load(os.path.join(TRAIN_INP, nm)))
        tgt = load(os.path.join(TRAIN_TGT, nm))
        ssims.append(sk_ssim(tgt, assemble(frags, place), channel_axis=2, data_range=255))
    model.train()
    return float(np.mean(accs)), float(np.mean(ssims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dsize", type=int, default=98)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=800)
    ap.add_argument("--evaln", type=int, default=8)
    ap.add_argument("--tag", default="asm")
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    trn, val = train_val_split()
    z = np.load(os.path.join(CACHE_DIR, "perms.npz"), allow_pickle=True)
    gt_inv = {n: z["inv"][i] for i, n in enumerate(z["names"])}

    dino = DinoEncoder(size=args.dsize, device=DEV)
    print(f"DINOv2 feat_dim={dino.feat_dim}", flush=True)
    names = trn[:args.overfit] if args.overfit else trn[:args.train_n]
    val_names = (names if args.overfit else val)[:args.evaln]
    print(f"precomputing features: {len(names)} train (synth) + {len(val_names)} eval...", flush=True)
    synth = build_cache(dino, names, "synth")
    real = build_cache(dino, val_names, "real")
    cache_t = torch.stack([synth[nm] for nm in names])       # (K, 576, feat) grid order

    model = AssemblerNet(dino.feat_dim, d=args.d, layers=args.layers).to(DEV)
    print(f"AssemblerNet params: {count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    K, fd = cache_t.shape[0], dino.feat_dim
    g = torch.Generator().manual_seed(SEED)

    t0 = time.time(); ema = 0.0
    for step in range(args.steps):
        idx = torch.randint(K, (args.bs,), generator=g)
        batch = cache_t[idx]                                  # (bs,576,feat) grid order
        sig = torch.stack([torch.randperm(NFRAG, generator=g) for _ in range(args.bs)])  # (bs,576)
        fed = torch.gather(batch, 1, sig[..., None].expand(-1, -1, fd)).to(DEV)
        tgt = sig.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            logP, _ = model(fed)
            loss = assemble_loss(logP.float(), tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tr = (logP.argmax(2) == tgt).float().mean().item()
        ema = 0.98 * ema + 0.02 * tr if step else tr
        if step % 50 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.3f} train_acc {ema:.3f} "
                  f"lr {sched.get_last_lr()[0]:.1e} {(time.time()-t0)/max(1,step):.2f}s/it", flush=True)
        if step % 500 == 0 and step > 0:
            a, s = eval_real(model, val_names, real, gt_inv)
            tagv = "OVERFIT(same imgs, real degr.)" if args.overfit else "VAL"
            print(f"  [{tagv}] placement_acc {a:.3f}  solve_SSIM {s:.3f}", flush=True)
    a, s = eval_real(model, val_names, real, gt_inv)
    print(f"FINAL placement_acc {a:.3f}  solve_SSIM {s:.3f}", flush=True)
    torch.save({"model": model.state_dict(), "step": args.steps, "args": vars(args)},
               os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
