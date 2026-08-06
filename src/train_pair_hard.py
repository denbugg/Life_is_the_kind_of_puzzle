"""Fine-tune PairwiseNet on mined hard negatives."""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import GRID, NFRAG, FS, TRAIN_TGT, CKPT_DIR, CACHE_DIR, SEED
from imgio import load, to_frags, train_val_split
from distort import distort_frags
from datasets import real_recon, CompatDataset
from models import PairwiseNet, count_params
from match_preprocess import photometric_normalize_tensor, load_match_denoiser

DEV = "cuda"


class HardPairDataset(Dataset):
    def __init__(self, hard_path, real_prob=0.6):
        z = np.load(hard_path, allow_pickle=True)
        self.names = [str(x) for x in z["names"]]
        self.right = z["right"].astype(np.int64)
        self.down = z["down"].astype(np.int64)
        self.real_prob = real_prob

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        nm = self.names[i]
        use_real = self.real_prob > 0 and rng.random() < self.real_prob
        if use_real:
            rr = real_recon(nm)
            frags = rr[0] if rr is not None else None
        else:
            frags = None
        if frags is None:
            frags = distort_frags(to_frags(load(os.path.join(TRAIN_TGT, nm))), rng)
        t = torch.from_numpy(np.ascontiguousarray(frags)).permute(0, 3, 1, 2).float() / 255
        return t, torch.from_numpy(self.right[i]), torch.from_numpy(self.down[i])


def make_pairs(frags, anchors, cand, transpose):
    f = frags.transpose(-1, -2) if transpose else frags
    A, M = cand.shape
    left = f[anchors][:, None].expand(A, M, 3, FS, FS)
    right = f[cand]
    pair = torch.cat([left, right], dim=-1)
    return pair.reshape(A * M, 3, FS, 2 * FS)


def choose_candidates(pool, anchors, offset, M, gen):
    A, K = anchors.shape[0], pool.shape[1]
    cand = torch.empty((A, M), dtype=torch.long, device=anchors.device)
    true = anchors + offset
    cand[:, 0] = true
    for i in range(A):
        vals = pool[anchors[i]].to(anchors.device).long()
        vals = vals[(vals >= 0) & (vals != anchors[i]) & (vals != true[i])]
        if len(vals) >= M - 1:
            p = torch.randperm(len(vals), generator=gen, device=anchors.device)[:M - 1]
            cand[i, 1:] = vals[p]
        else:
            if len(vals):
                cand[i, 1:1 + len(vals)] = vals
            need = M - 1 - len(vals)
            fill = torch.randint(NFRAG, (need,), generator=gen, device=anchors.device)
            fill = torch.where((fill == anchors[i]) | (fill == true[i]), (fill + 7) % NFRAG, fill)
            cand[i, 1 + len(vals):] = fill
    return cand


def apply_preprocess(frags, mode, denoiser):
    if mode == "raw":
        return frags
    if mode == "norm":
        return photometric_normalize_tensor(frags)
    if mode in ("denoise", "denoise_norm"):
        if denoiser is None:
            raise ValueError("denoise preprocess requires checkpoint")
        B, N = frags.shape[:2]
        flat = frags.reshape(B * N, 3, FS, FS)
        with torch.no_grad():
            out = []
            for i in range(0, len(flat), 1024):
                out.append(denoiser(flat[i:i + 1024]).float())
            frags = torch.cat(out, 0).reshape(B, N, 3, FS, FS)
        return photometric_normalize_tensor(frags) if mode == "denoise_norm" else frags
    raise ValueError(f"unknown preprocess mode: {mode}")


def step_loss(model, frags_b, rneg_b, dneg_b, ph, pv, nA, M, gen):
    chunks = []
    for bi, frags in enumerate(frags_b):
        for has, off, hard, tr in ((ph, 1, rneg_b[bi], False), (pv, GRID, dneg_b[bi], True)):
            anchors = has[torch.randint(len(has), (nA,), generator=gen, device=DEV)]
            cand = choose_candidates(hard, anchors, off, M, gen)
            chunks.append(make_pairs(frags, anchors, cand, tr))
    G = len(chunks)
    logits = model(torch.cat(chunks, 0)).view(G, nA, M)
    tgt = torch.zeros(G * nA, dtype=torch.long, device=DEV)
    loss = F.cross_entropy(logits.reshape(G * nA, M), tgt)
    acc = (logits.argmax(-1) == 0).float().mean().item()
    return loss, acc


def random_val_loss(model, vdl, ph, pv, gen, nA=48, M=48, n=8):
    model.eval()
    vals = []
    from train_pair import step_loss as random_step_loss
    with torch.no_grad():
        for k, fb in enumerate(vdl):
            if k >= n:
                break
            _, acc = random_step_loss(model, fb.to(DEV), ph, pv, nA, M, gen)
            vals.append(acc)
    model.train()
    return float(np.mean(vals)) if vals else 0.0


def load_init(tag="pair", which="best"):
    for name in (f"{tag}_{which}.pt", f"{tag}_last.pt", f"{tag}_best.pt"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEV)
            m = PairwiseNet().to(DEV)
            m.load_state_dict(ck["model"])
            print(f"loaded init {name} step={ck.get('step')} val={ck.get('val')}", flush=True)
            return m
    return PairwiseNet().to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default="")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--nA", type=int, default=48)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real_prob", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--preprocess", choices=("raw", "norm", "denoise", "denoise_norm"), default="raw")
    ap.add_argument("--denoise_tag", default="matchden")
    ap.add_argument("--init_tag", default="pair")
    ap.add_argument("--tag", default="pair_hard")
    args = ap.parse_args()

    hard_path = args.hard or os.path.join(CACHE_DIR, f"hardneg_train_{args.preprocess}_K48.npz")
    if not os.path.exists(hard_path):
        raise FileNotFoundError(hard_path)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED)

    dl = DataLoader(HardPairDataset(hard_path, args.real_prob), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0,
                    prefetch_factor=3 if args.workers > 0 else None, pin_memory=True)
    _, val = train_val_split()
    vdl = DataLoader(CompatDataset(val, real_prob=1.0), batch_size=args.bs, shuffle=False,
                     num_workers=max(1, min(3, args.workers)), persistent_workers=True)

    p = np.arange(NFRAG)
    ph = torch.tensor(p[p % GRID != GRID - 1], device=DEV)
    pv = torch.tensor(p[p // GRID != GRID - 1], device=DEV)
    model = load_init(args.init_tag).to(DEV)
    print(f"PairwiseNet params: {count_params(model):,}", flush=True)
    denoiser = None
    if args.preprocess in ("denoise", "denoise_norm"):
        denoiser, _ = load_match_denoiser(args.denoise_tag, device=DEV)
        if denoiser is None:
            raise FileNotFoundError("no matching denoiser checkpoint found")
        denoiser.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    step = 0
    t0 = time.time()
    while step < args.steps:
        for fb, rn, dn in dl:
            fb = fb.to(DEV, non_blocking=True)
            rn = rn.to(DEV, non_blocking=True)
            dn = dn.to(DEV, non_blocking=True)
            fb = apply_preprocess(fb, args.preprocess, denoiser)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, acc = step_loss(model, fb, rn, dn, ph, pv, args.nA, args.M, gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.3f} hard_acc@{args.M} {acc:.3f} "
                      f"lr {sched.get_last_lr()[0]:.1e} {(time.time()-t0)/max(1,step):.2f}s/it",
                      flush=True)
            if step % 500 == 0 and step > 0:
                va = random_val_loss(model, vdl, ph, pv, gen, M=max(args.M, 48))
                print(f"  [VAL random] acc@{max(args.M,48)} {va:.3f}", flush=True)
                torch.save({"model": model.state_dict(), "step": step, "val": va},
                           os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
                if va > best:
                    best = va
                    torch.save({"model": model.state_dict(), "step": step, "val": va},
                               os.path.join(CKPT_DIR, f"{args.tag}_best.pt"))
            step += 1
            if step >= args.steps:
                break
    torch.save({"model": model.state_dict(), "step": step, "val": best},
               os.path.join(CKPT_DIR, f"{args.tag}_last.pt"))
    print(f"done. best random-val acc={best:.3f}", flush=True)


if __name__ == "__main__":
    main()

