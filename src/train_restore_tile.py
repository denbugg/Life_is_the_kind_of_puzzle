"""Train the pre-assembly tile restorer on REAL (dirty, clean) fragment pairs.

Training data comes from build_restore_labels.py, which recovers each train
board's permutation by Hungarian matching of photometrically-normalised tiles.
Raw match accuracy is 0.825, but the assignment margin is well calibrated, so
only positions above a margin quantile are used (top 50% -> 0.996 accuracy).

Synthetic boards can be mixed in via --synth-prob.  distort.py was verified
against real pairs and matches them closely: residual noise 13.1 vs 13.3,
spatial autocorrelation 0.732 vs 0.735, JPEG blockiness 1.49 vs 1.50, and
contrast percentiles agreeing within 0.02 (robust spread 0.193 vs 0.194).
Beware the naive check: plain std reads 0.401 vs 0.223, which looks like a 40%
domain gap but is entirely outliers from flat tiles, where fitting a slope
divides by a near-zero variance.

Synthetic boards are worth mixing in because they corrupt the same clean tile
differently every epoch and their labels are exact, whereas recovered
permutations are 0.996 accurate on the retained half and 0.825 overall.

Objective
---------
Pure pixel L1 plateaus at held-out bb_prec 0.225 (vs 0.189 for the raw ridge
scorer): L1 converges to the conditional mean, which smooths away exactly the
border microstructure adjacency depends on.  The same failure sank the earlier
MatchDenoiser (D1), which improved L1 while leaving ranking untouched.

So the dominant term is a CONTRASTIVE seam loss: the ridge seam cost is
differentiable, so the restorer can be trained directly on "make the true
neighbour rank first" over all 576 tiles.  L1 is kept at low weight purely as
an anchor that stops the output drifting off the natural image manifold.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import CACHE_DIR, CKPT_DIR, FS, NFRAG as N, TRAIN_INP, TRAIN_TGT, VAL_COUNT
from distort import distort_frags
from mgc import mgc_cost
from restore_tile import TileRestorer, blur3_np, seam_infonce, seam_metrics, to_frags


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable: {path}")
    return np.ascontiguousarray(img[:, :, ::-1])


class Boards(Dataset):
    """One item = a whole board in TRUE grid order plus its label-confidence mask.

    The seam loss needs every tile present (all 575 non-self tiles are its
    negatives), so unlike a pure-L1 dataset this cannot drop low-margin
    positions; they are masked in the loss instead.
    """

    def __init__(self, names, inv, margin, thr, synth_prob=0.0, target_mode='clean'):
        self.curriculum = 1.0
        self.names, self.inv, self.margin, self.thr = names, inv, margin, thr
        self.synth_prob, self.target_mode = synth_prob, target_mode

    def __len__(self):
        return len(self.names)

    def __getitem__(self, k):
        nm = self.names[k]
        clean = to_frags(load_rgb(Path(TRAIN_TGT) / nm)).astype(np.float32)
        if self.target_mode == "noisefree":
            # Aim at the blurred clean tile rather than the original.  The
            # degradation decomposition shows blur+JPEG alone still score R@1
            # 0.521, above the 0.47 assembly threshold, while noise alone drops
            # it to 0.100.  Deconvolving the blur is therefore unnecessary and
            # merely spends capacity on an ill-posed inverse; the tractable
            # target is "same tile, minus the noise".
            clean = blur3_np(clean)
        if self.synth_prob and np.random.rand() < self.synth_prob:
            # Synthetic board: re-corrupt the clean tiles ourselves.  Verified
            # against real pairs -- residual noise 13.1 vs 13.3, autocorrelation
            # 0.732 vs 0.735, JPEG blockiness 1.49 vs 1.50, and contrast
            # percentiles matching within 0.02 (robust spread 0.193 vs 0.194).
            # Unlike real pairs this gives unlimited augmentation (a tile can be
            # corrupted differently every epoch) and EXACT labels, where recovered
            # permutations are only 0.996 accurate on the retained half.
            # distort_frags needs a Generator: its default np.random path calls
            # .integers(), which the legacy RandomState does not provide.
            x = distort_frags(clean.astype(np.uint8),
                              np.random.default_rng()).astype(np.float32)
            if self.curriculum < 1.0:
                # Sample the corruption STRENGTH per board rather than ramping it
                # over training: a DataLoader with workers copies the dataset, so
                # a schedule mutated in the main process never reaches them.
                # Mixing easy and full corruption also keeps a usable gradient --
                # at full strength the contrastive signal is nearly flat, and the
                # model has little to latch onto.
                lo = self.curriculum
                strength = lo + (1.0 - lo) * np.random.rand()
                x = clean + (x - clean) * strength
            good = np.ones(N, np.float32)
        else:
            dirty = to_frags(load_rgb(Path(TRAIN_INP) / nm)).astype(np.float32)
            x = dirty[self.inv[k].astype(np.int64)]            # dirty tile at grid pos p
            good = (self.margin[k] >= self.thr).astype(np.float32)
        return (torch.from_numpy(x).permute(0, 3, 1, 2),
                torch.from_numpy(clean).permute(0, 3, 1, 2),
                torch.from_numpy(good))


@torch.no_grad()
def evaluate(model, names, inv, device, n_boards, w, cols, metric='ridge'):
    """Restore each held-out board's tiles IN TRUE ORDER and score the seams."""
    model.eval()
    got, base = [], []
    for k in range(n_boards):
        dirty = to_frags(load_rgb(Path(TRAIN_INP) / names[k])).astype(np.float32)
        ordered = dirty[inv[k].astype(np.int64)]
        x = torch.from_numpy(ordered).permute(0, 3, 1, 2).to(device)
        with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            out = torch.cat([model(x[i:i + 288]) for i in range(0, len(x), 288)])
        rec = out.float().permute(0, 2, 3, 1).clamp(0, 255).cpu().numpy()
        got.append(seam_metrics(rec, w, cols, metric))
        base.append(seam_metrics(ordered, w, cols, metric))
    model.train()
    agg = lambda rows, key: float(np.mean([r[key] for r in rows]))
    return ({k: agg(got, k) for k in got[0]}, {k: agg(base, k) for k in base[0]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path(CACHE_DIR) / "restore_labels.npz")
    ap.add_argument("--ckpt", type=Path, default=Path(CKPT_DIR) / "tile_restorer.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--boards-per-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--synth-prob", type=float, default=0.0,
                    help="fraction of boards re-corrupted synthetically (exact labels, unlimited augmentation)")
    ap.add_argument("--curriculum-start", type=float, default=1.0,
                    help="lowest synthetic corruption strength; 1.0 = always full corruption")
    ap.add_argument("--hard-k", type=int, default=0,
                    help="restrict the contrastive loss to the K hardest negatives (0 = all 575)")
    ap.add_argument("--ycc", action="store_true",
                    help="feed a YCrCb view too; chroma survives JPEG far better than luma")
    ap.add_argument("--checkpoint", action="store_true",
                    help="gradient checkpointing; needed above ch=96/blocks=6 on 8 GB")
    ap.add_argument("--residual", action="store_true",
                    help="predict a correction to the input instead of structure from scratch")
    ap.add_argument("--margin-quantile", type=float, default=0.5,
                    help="keep positions above this margin quantile (0.5 -> 0.996 label acc)")
    ap.add_argument("--seam-weight", type=float, default=1.0)
    ap.add_argument("--l1-weight", type=float, default=0.02,
                    help="manifold anchor only; pure L1 plateaus at bb_prec 0.225")
    ap.add_argument("--edge-weight", type=float, default=1.0,
                    help="L1 weight on the border ring the matcher actually reads")
    ap.add_argument("--interior-weight", type=float, default=0.0,
                    help="L1 weight inside the ring; oracle tests show the interior is irrelevant")
    ap.add_argument("--target-mode", choices=("clean", "noisefree"), default="clean",
                    help="noisefree targets blur3(clean): denoising only, ceiling R@1 0.521")
    ap.add_argument("--grad-weight", type=float, default=1.0,
                    help="L1 on the residual gradient; decorrelates the error MGC reads")
    ap.add_argument("--ring-width", type=int, default=2,
                    help="ring thickness; 2px recovers the full ceiling, 1px reaches R@1 0.549")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-boards", type=int, default=8)
    ap.add_argument("--ridge-w", type=float, default=0.03)
    ap.add_argument("--ridge-cols", type=int, default=3)
    ap.add_argument("--seam-metric", choices=("ridge", "mgc", "both"), default="ridge",
                    help="metric the contrastive seam loss optimises")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    blob = np.load(args.labels, allow_pickle=True)
    names, inv, margin = blob["names"], blob["inv"], blob["margin"]
    thr = float(np.quantile(margin, args.margin_quantile))
    n_val = min(VAL_COUNT, len(names) // 4)
    tr_slice, va_slice = slice(0, len(names) - n_val), slice(len(names) - n_val, len(names))
    print(f"images: train={len(names)-n_val} val={n_val}  margin_thr={thr:.4f}  "
          f"kept={(margin[tr_slice] >= thr).mean()*100:.1f}% of positions", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TileRestorer(args.ch, args.blocks, args.residual, args.checkpoint, args.ycc).to(device)
    inv_temp = nn.Parameter(torch.tensor(0.7, device=device))
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={device}", flush=True)

    ds = Boards(names[tr_slice], inv[tr_slice], margin[tr_slice], thr, args.synth_prob,
                args.target_mode)
    if args.curriculum_start < 1.0:
        ds.curriculum = args.curriculum_start
    dl = DataLoader(ds, batch_size=args.boards_per_batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(list(model.parameters()) + [inv_temp], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    # Only the outer ring matters.  Oracle experiment: grafting the true 2px
    # border onto a restored tile lifts R@1 from 0.088 to 0.774 -- the exact
    # clean_blur ceiling -- while grafting the true 16x16 INTERIOR changes
    # nothing at all (0.088).  A 1px ring already reaches 0.549.  So spending
    # capacity on the interior is wasted; the loss weights the ring and can
    # zero the interior entirely.
    wmap = torch.full((1, 1, FS, FS), args.interior_weight, device=device)
    r = args.ring_width
    wmap[:, :, :r, :] = args.edge_weight; wmap[:, :, -r:, :] = args.edge_weight
    wmap[:, :, :, :r] = args.edge_weight; wmap[:, :, :, -r:] = args.edge_weight

    best, step, started = -1.0, 0, time.perf_counter()
    while step < args.steps:
        for xb, yb, gb in dl:
            if step >= args.steps:
                break
            seam_acc, l1_acc = [], []
            for x, y, g in zip(xb, yb, gb):
                x, y, g = x.to(device), y.to(device), g.to(device)
                with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                    out = model(x)
                out = out.float()
                seam_acc.append(seam_infonce(out, inv_temp, g, args.ridge_w, args.ridge_cols,
                                             args.seam_metric, args.hard_k))
                keep = g > 0
                err = (out[keep] - y[keep])
                loss_px = (err.abs() * wmap).mean()
                # Gradient term: plain L1 lets the network under-restore high
                # frequencies, which makes the residual CORRELATED (measured
                # autocorrelation 0.72 versus 0.00 for white noise).  That costs
                # 2.5x: at ring sigma 14 a white residual scores R@1 0.495 and a
                # correlated one 0.194.  MGC reads gradients, so match gradients.
                gx = (err[:, :, :, 1:] - err[:, :, :, :-1]).abs()
                gy = (err[:, :, 1:, :] - err[:, :, :-1, :]).abs()
                loss_gr = (gx * wmap[:, :, :, 1:]).mean() + (gy * wmap[:, :, 1:, :]).mean()
                l1_acc.append(loss_px + args.grad_weight * loss_gr)
            seam = torch.stack(seam_acc).mean()
            l1 = torch.stack(l1_acc).mean()
            loss = args.seam_weight * seam + args.l1_weight * l1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(model.parameters()) + [inv_temp], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                got, base = evaluate(model, names[va_slice], inv[va_slice], device,
                                     args.eval_boards, args.ridge_w, args.ridge_cols,
                                     args.seam_metric)
                flag = ""
                if got["bb_prec"] > best:
                    best = got["bb_prec"]
                    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "args": vars(args),
                                "step": step, "bb_prec": best}, args.ckpt)
                    flag = "  *saved"
                print(f"step {step:5d}  seam {seam.item():6.3f}  l1 {l1.item():6.2f}  "
                      f"bb_prec {got['bb_prec']:.3f} (raw {base['bb_prec']:.3f})  "
                      f"R@1 {got['R1']:.3f} (raw {base['R1']:.3f})  "
                      f"R@20 {got['R20']:.3f} (raw {base['R20']:.3f})  "
                      f"{(time.perf_counter()-started)/60:.1f}min{flag}", flush=True)

    print(f"best held-out bb_prec = {best:.4f}   ckpt={args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
