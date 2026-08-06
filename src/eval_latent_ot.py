"""Gate a latent-canvas VAE with per-image differentiable OT fitting.

For each held-out clean target this script makes a fresh synthetic puzzle
(``target -> distort every tile -> shuffle``).  It then asks whether the VAE
manifold can be fitted directly from the unordered bag: decode a latent canvas,
match its 24x24 low-pass cells to input low-pass tiles with Sinkhorn OT, and
differentiate that objective with respect to the latent.  No recovered real
permutations are used here.
"""
from __future__ import annotations

import argparse
import inspect
import os
import random
from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim

from canvas_data import CanvasDataset
from canvas_metrics import canvas_patches, decoded_geometry, hard_assignment, rank_summary
from config import FS, GRID, NFRAG, SEED
from imgio import from_frags, train_val_split
from latent_canvas_model import BagLatentEncoder, CanvasVAE


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _checkpoint_kwargs(cls: type[nn.Module], checkpoint: Any) -> dict[str, Any]:
    """Use constructor settings saved by the trainer, while ignoring train-only args."""
    cfg: dict[str, Any] = {}
    ck = _mapping(checkpoint)
    for key in ("model_kwargs", "kwargs", "config", "args"):
        value = _mapping(ck.get(key))
        cfg.update(value)
        # A compact checkpoint may nest the relevant kwargs under the model name.
        for nested in ("vae_kwargs", "bag_kwargs", "model_kwargs"):
            cfg.update(_mapping(value.get(nested)))
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in cfg.items() if k in params and k != "self"}


def _state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    ck = _mapping(checkpoint)
    for key in ("model", "state_dict", "vae", "bag", "encoder"):
        value = ck.get(key)
        if isinstance(value, Mapping) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    raise KeyError("checkpoint needs a model/state_dict/vae/bag tensor state dictionary")


def _without_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = dict(state)
    for prefix in ("module.", "model.", "vae.", "bag."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def _load_model(path: str, cls: type[nn.Module], device: torch.device) -> nn.Module:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, cls):
        model = checkpoint
    else:
        model = cls(**_checkpoint_kwargs(cls, checkpoint))
        model.load_state_dict(_without_prefix(_state_dict(checkpoint)), strict=True)
    return model.to(device).eval()


def _first_tensor(value: Any, keys: tuple[str, ...]) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for key in keys:
            if torch.is_tensor(value.get(key)):
                return value[key]
        for item in value.values():
            if torch.is_tensor(item):
                return item
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"cannot find tensor in {type(value).__name__}")


def encode_mean(vae: CanvasVAE, canvas: torch.Tensor) -> torch.Tensor:
    """The first encode output is conventionally the deterministic VAE mean."""
    return _first_tensor(vae.encode(canvas), ("mu", "z", "latent", "mean"))


def decode_canvas(vae: CanvasVAE, z: torch.Tensor, patch: int) -> torch.Tensor:
    canvas = _first_tensor(vae.decode(z), ("canvas", "recon", "decoded", "image"))
    if canvas.ndim == 3:
        canvas = canvas.unsqueeze(0)
    if canvas.ndim != 4:
        raise ValueError(f"VAE decoder must return BCHW canvas, got {tuple(canvas.shape)}")
    if canvas.shape[1] != 3 and canvas.shape[-1] == 3:
        canvas = canvas.permute(0, 3, 1, 2).contiguous()
    expected = GRID * patch
    if tuple(canvas.shape[1:]) != (3, expected, expected):
        raise ValueError(
            f"decoder canvas must be (B,3,{expected},{expected}) for --patch={patch}, got {tuple(canvas.shape)}"
        )
    return canvas


def normalized_tile_descriptors(tiles: torch.Tensor, patch: int) -> torch.Tensor:
    """Exact per-tile pooling; it never blends information across puzzle seams."""
    b, n, c, h, w = tiles.shape
    if (n, c, h, w) != (NFRAG, 3, FS, FS) or FS % patch:
        raise ValueError(f"expected tiles (B,{NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}")
    x = F.avg_pool2d(tiles.reshape(b * n, c, h, w), FS // patch).reshape(b, n, -1)
    return F.normalize(x - x.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6)


def normalized_canvas_descriptors(canvas: torch.Tensor, patch: int) -> torch.Tensor:
    x = canvas_patches(canvas, patch)
    return F.normalize(x - x.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6)


def scores_for_latent(vae: CanvasVAE, z: torch.Tensor, tile_desc: torch.Tensor, patch: int) -> torch.Tensor:
    canvas_desc = normalized_canvas_descriptors(decode_canvas(vae, z, patch), patch)
    return tile_desc @ canvas_desc.transpose(1, 2)


def sinkhorn(scores: torch.Tensor, tau: float, iters: int) -> torch.Tensor:
    """Differentiable uniform OT plan, with every row and column summing to one."""
    log_p = scores / tau
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


def fit_latent(
    vae: CanvasVAE,
    z_init: torch.Tensor,
    tile_desc: torch.Tensor,
    *,
    patch: int,
    iters: int,
    lr: float,
    tau: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, float, float]:
    """Fit just z; all VAE parameters remain frozen while decode(z) stays differentiable."""
    z = nn.Parameter(z_init.detach().clone())
    optimizer = torch.optim.Adam([z], lr=lr)
    first_loss = float("nan")
    last_loss = float("nan")
    for step in range(iters):
        scores = scores_for_latent(vae, z, tile_desc, patch)
        plan = sinkhorn(scores, tau, sinkhorn_iters)
        # Maximizing cosine affinity under a soft bijection is the entropic OT
        # objective up to its entropy term (the plan itself is differentiable).
        loss = -(plan * scores).sum(dim=(-1, -2)).mean() / NFRAG
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite latent OT loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 5.0)
        optimizer.step()
        if step == 0:
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
    return z.detach(), first_loss, last_loss


def _solve_ssim(tiles: torch.Tensor, clean: torch.Tensor, scores: torch.Tensor) -> float:
    place = hard_assignment(scores[0])
    frags = tiles[0].detach().float().cpu().permute(0, 2, 3, 1).numpy()
    target = clean[0].detach().float().cpu().permute(1, 2, 0).numpy()
    assembled = from_frags(frags[place])
    return float(sk_ssim(target, assembled, channel_axis=2, data_range=1.0))


@torch.no_grad()
def measure(
    vae: CanvasVAE,
    z: torch.Tensor,
    tile_desc: torch.Tensor,
    tiles: torch.Tensor,
    clean: torch.Tensor,
    perm: torch.Tensor,
    patch: int,
) -> dict[str, float]:
    scores = scores_for_latent(vae, z, tile_desc, patch)
    rank = rank_summary(scores, perm)
    geometry = decoded_geometry(scores, perm)
    return {
        "r1": rank["r1"],
        "r5": rank["r5"],
        "placement": geometry["place_acc"],
        "neighbour": geometry["neighbour_acc"],
        "solve_ssim": _solve_ssim(tiles, clean, scores),
    }


@torch.no_grad()
def bag_latent(bag: BagLatentEncoder, tiles: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    z = _first_tensor(bag(tiles), ("z", "latent", "mu", "mean"))
    if z.shape != reference.shape:
        if z.numel() == reference.numel():
            z = z.reshape_as(reference)
        else:
            raise ValueError(f"bag encoder produced {tuple(z.shape)}, expected latent {tuple(reference.shape)}")
    return z


def _print_row(label: str, values: Mapping[str, float]) -> None:
    print(
        f"{label:9s} R@1={values['r1']:.4f} R@5={values['r5']:.4f} "
        f"placement={values['placement']:.4f} neighbour={values['neighbour']:.4f} "
        f"solveSSIM={values['solve_ssim']:.4f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="number of synthetic held-out images")
    ap.add_argument("--iters", type=int, default=250, help="Adam steps for each image latent")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.08, help="Sinkhorn temperature")
    ap.add_argument("--ckpt", required=True, help="CanvasVAE checkpoint")
    ap.add_argument("--bag_ckpt", default=None, help="optional BagLatentEncoder checkpoint")
    ap.add_argument("--patch", type=int, default=4, help="low-pass descriptor side per 20px tile")
    ap.add_argument("--sinkhorn_iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None, help="default: cuda when available, otherwise cpu")
    args = ap.parse_args()
    if args.n < 1 or args.iters < 1 or args.sinkhorn_iters < 1:
        ap.error("--n, --iters and --sinkhorn_iters must be positive")
    if args.lr <= 0 or args.tau <= 0:
        ap.error("--lr and --tau must be positive")
    if FS % args.patch:
        ap.error("--patch must divide the fragment size")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = _load_model(args.ckpt, CanvasVAE, device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    _, val_names = train_val_split()
    names = val_names[:args.n]
    if not names:
        raise RuntimeError("held-out split is empty")
    dataset = CanvasDataset(names, patch=args.patch, real_prob=0.0, seed=args.seed)

    # Infer the expected bag latent shape from a clean VAE encoding before
    # instantiating the optional set encoder.
    probe = dataset[0]["canvas"].unsqueeze(0).to(device)
    with torch.no_grad():
        latent_probe = encode_mean(vae, probe)
    bag = None
    init_name = "prior"
    if args.bag_ckpt:
        bag = _load_model(args.bag_ckpt, BagLatentEncoder, device)
        for parameter in bag.parameters():
            parameter.requires_grad_(False)
        init_name = "bag"

    print(
        f"device={device} n={len(names)} patch={args.patch} iters={args.iters} "
        f"tau={args.tau:g} init={init_name}",
        flush=True,
    )
    totals: dict[str, defaultdict[str, list[float]]] = {
        key: defaultdict(list) for key in ("oracle", "init", "optimized")
    }
    for index, name in enumerate(names):
        item = dataset[index]
        tiles = item["tiles"].unsqueeze(0).to(device)
        canvas = item["canvas"].unsqueeze(0).to(device)
        clean = item["clean"].unsqueeze(0).to(device)
        perm = item["perm"].unsqueeze(0).to(device).long()
        tile_desc = normalized_tile_descriptors(tiles, args.patch)
        with torch.no_grad():
            z_oracle = encode_mean(vae, canvas)
            z_init = bag_latent(bag, tiles, z_oracle) if bag is not None else torch.zeros_like(z_oracle)
            rows = {
                "oracle": measure(vae, z_oracle, tile_desc, tiles, clean, perm, args.patch),
                "init": measure(vae, z_init, tile_desc, tiles, clean, perm, args.patch),
            }
        z_opt, first_loss, last_loss = fit_latent(
            vae, z_init, tile_desc, patch=args.patch, iters=args.iters, lr=args.lr,
            tau=args.tau, sinkhorn_iters=args.sinkhorn_iters,
        )
        rows["optimized"] = measure(vae, z_opt, tile_desc, tiles, clean, perm, args.patch)
        print(f"[{index + 1}/{len(names)}] {name} OT={first_loss:.4f}->{last_loss:.4f}", flush=True)
        for label, values in rows.items():
            _print_row(label, values)
            for key, value in values.items():
                totals[label][key].append(value)

    print("\n== mean synthetic held-out metrics ==", flush=True)
    for label in ("oracle", "init", "optimized"):
        _print_row(label, {key: float(np.mean(values)) for key, values in totals[label].items()})


if __name__ == "__main__":
    main()
