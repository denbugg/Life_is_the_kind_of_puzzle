"""Train the clean-image VAE prior and optional unordered-bag latent initialiser.

Usage:

    # First establish that a global latent can reconstruct held-out 96px canvases.
    python src/train_latent_prior.py --stage prior --steps 6000 --bs 24

    # Then teach a set encoder to initialise that latent from shuffled tiles.
    python src/train_latent_prior.py --stage bag --vae_ckpt artifacts/latent/prior_best.pt

The per-puzzle OT optimisation is intentionally kept in ``eval_latent_ot.py``;
this script only learns the image manifold and a cheap initial point on it.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import SEED
from imgio import train_val_split
from latent_canvas_model import BagLatentEncoder, CanvasVAE, count_params
from latent_data import CleanCanvasDataset


def _loader(dataset: Any, bs: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    kw: dict[str, Any] = dict(batch_size=bs, shuffle=shuffle, num_workers=workers,
                              pin_memory=device.type == "cuda", drop_last=shuffle)
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kw)


def _vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, dict[str, float]]:
    rec = F.l1_loss(recon, target)
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    loss = rec + beta * kl
    return loss, {"rec": float(rec.detach()), "kl": float(kl.detach()), "total": float(loss.detach())}


def _save(path: str, model: torch.nn.Module, step: int, args: argparse.Namespace, metrics: dict[str, float], kind: str) -> None:
    if kind == "prior":
        kwargs = {"image_size": args.image_size, "zdim": args.zdim, "base": args.base}
    else:
        kwargs = {"zdim": args.zdim, "d": args.bag_d}
    torch.save({"model": model.state_dict(), "model_kwargs": kwargs, "step": step,
                "args": vars(args), "metrics": metrics}, path)


def _load_vae(path: str, args: argparse.Namespace, device: torch.device) -> CanvasVAE:
    payload = torch.load(path, map_location=device)
    kw = payload.get("model_kwargs", {}) if isinstance(payload, dict) else {}
    vae = CanvasVAE(image_size=kw.get("image_size", args.image_size), zdim=kw.get("zdim", args.zdim),
                    base=kw.get("base", args.base)).to(device)
    vae.load_state_dict(payload["model"] if isinstance(payload, dict) else payload)
    return vae


@torch.no_grad()
def _eval_prior(vae: CanvasVAE, loader: DataLoader, device: torch.device, beta: float, n: int) -> dict[str, float]:
    vae.eval(); rows: list[dict[str, float]] = []
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    seen = 0
    for target in loader:
        target = target.to(device, non_blocking=True)
        with amp:
            mu, logvar = vae.encode(target)
            recon = vae.decode(mu)
            _, row = _vae_loss(recon.float(), target.float(), mu.float(), logvar.float(), beta)
        rows.append(row); seen += target.shape[0]
        if seen >= n:
            break
    vae.train()
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


@torch.no_grad()
def _eval_bag(bag: BagLatentEncoder, vae: CanvasVAE, loader: DataLoader, device: torch.device, n: int) -> dict[str, float]:
    bag.eval(); vae.eval(); rows: list[dict[str, float]] = []; seen = 0
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    for batch in loader:
        tiles = batch["tiles"].to(device, non_blocking=True)
        canvas = batch["canvas"].to(device, non_blocking=True)
        with amp:
            target_z, _ = vae.encode(canvas)
            z = bag(tiles)
            recon = vae.decode(z)
        rows.append({"z_mse": float(F.mse_loss(z.float(), target_z.float())),
                     "canvas_l1": float(F.l1_loss(recon.float(), canvas.float()))})
        seen += tiles.shape[0]
        if seen >= n:
            break
    bag.train(); vae.eval()
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("prior", "bag"), required=True)
    ap.add_argument("--steps", type=int, default=6_000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1e-4, help="VAE KL coefficient")
    ap.add_argument("--deterministic", action="store_true", help="train decoder from mu rather than sampled z (AE gate)")
    ap.add_argument("--bag_recon_weight", type=float, default=1.0)
    ap.add_argument("--real_prob", type=float, default=0.5, help="real bag fraction for --stage bag")
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=96)
    ap.add_argument("--zdim", type=int, default=256)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--bag_d", type=int, default=128)
    ap.add_argument("--vae_ckpt", help="frozen VAE checkpoint required for --stage bag")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "latent"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.steps <= 0 or args.bs <= 0 or args.workers < 0:
        ap.error("--steps/--bs must be positive and --workers non-negative")
    if args.image_size != 24 * args.patch:
        ap.error("--image_size must equal 24 * --patch")
    if args.stage == "bag" and not args.vae_ckpt:
        ap.error("--vae_ckpt is required for --stage bag")
    os.makedirs(args.out_dir, exist_ok=True)
    args.tag = args.tag or args.stage

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_names, val_names = train_val_split()

    if args.stage == "prior":
        model: torch.nn.Module = CanvasVAE(args.image_size, args.zdim, args.base).to(device)
        train_dl = _loader(CleanCanvasDataset(train_names, args.patch), args.bs, args.workers, True, device)
        val_dl = _loader(CleanCanvasDataset(val_names, args.patch), args.bs, min(args.workers, 2), False, device)
        print(f"prior params={count_params(model):,} device={device}", flush=True)
    else:
        vae = _load_vae(args.vae_ckpt, args, device).eval()
        for p in vae.parameters(): p.requires_grad_(False)
        model = BagLatentEncoder(args.zdim, args.bag_d).to(device)
        train_dl = _loader(CanvasDataset(train_names, patch=args.patch, real_prob=args.real_prob, seed=args.seed), args.bs, args.workers, True, device)
        val_dl = _loader(CanvasDataset(val_names, patch=args.patch, real_prob=1.0, seed=args.seed + 10_000), args.bs, min(args.workers, 2), False, device)
        print(f"bag params={count_params(model):,}; frozen VAE={count_params(vae):,}; device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.08)
    best = float("inf"); started = time.time(); it = iter(train_dl)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl); batch = next(it)
        with amp:
            if args.stage == "prior":
                target = batch.to(device, non_blocking=True)
                mu, logvar = model.encode(target)  # type: ignore[union-attr]
                z = mu if args.deterministic else model.reparameterize(mu, logvar)  # type: ignore[union-attr]
                recon = model.decode(z)  # type: ignore[union-attr]
                loss, terms = _vae_loss(recon.float(), target.float(), mu.float(), logvar.float(), args.beta)
            else:
                tiles = batch["tiles"].to(device, non_blocking=True)
                canvas = batch["canvas"].to(device, non_blocking=True)
                with torch.no_grad():
                    target_z, _ = vae.encode(canvas)
                z = model(tiles)  # type: ignore[operator]
                recon = vae.decode(z)
                z_mse = F.mse_loss(z.float(), target_z.float())
                rec = F.l1_loss(recon.float(), canvas.float())
                loss = z_mse + args.bag_recon_weight * rec
                terms = {"z_mse": float(z_mse.detach()), "canvas_l1": float(rec.detach()), "total": float(loss.detach())}
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if step % 50 == 0:
            print(f"step {step}/{args.steps} " + " ".join(f"{k}={v:.4f}" for k, v in terms.items()) +
                  f" lr={sched.get_last_lr()[0]:.2e} {(time.time() - started)/max(1,step):.2f}s/it", flush=True)
        if step > 0 and step % args.eval_every == 0:
            if args.stage == "prior":
                metrics = _eval_prior(model, val_dl, device, args.beta, args.eval_n)  # type: ignore[arg-type]
                key = metrics["rec"]
            else:
                metrics = _eval_bag(model, vae, val_dl, device, args.eval_n)  # type: ignore[arg-type]
                key = metrics["canvas_l1"]
            print("[VAL] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            _save(os.path.join(args.out_dir, f"{args.tag}_last.pt"), model, step, args, metrics, args.stage)
            if key < best:
                best = key
                _save(os.path.join(args.out_dir, f"{args.tag}_best.pt"), model, step, args, metrics, args.stage)
                print(f"saved best {key:.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
