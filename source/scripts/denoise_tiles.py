#!/usr/bin/env python3
"""Tilewise denoising pipeline for the VSOS puzzle task.

The script keeps the shuffled tile layout fixed. It learns to map each corrupted
20x20 input tile to a cleaner version of the same content, using pseudo-labels
created by matching train input tiles to clean target tiles from the same image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

GRID = 24
TILE = 20
SIZE = GRID * TILE


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(path)


def split_tiles(img: np.ndarray) -> np.ndarray:
    if img.shape != (SIZE, SIZE, 3):
        raise ValueError(f"expected {(SIZE, SIZE, 3)}, got {img.shape}")
    return (
        img.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * GRID, TILE, TILE, 3)
    )


def merge_tiles(tiles: np.ndarray) -> np.ndarray:
    if tiles.shape != (GRID * GRID, TILE, TILE, 3):
        raise ValueError(f"expected {(GRID * GRID, TILE, TILE, 3)}, got {tiles.shape}")
    return tiles.reshape(GRID, GRID, TILE, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(SIZE, SIZE, 3)


def tile_features(tiles: np.ndarray, bins: int = 5) -> np.ndarray:
    """Robust-ish descriptors for matching noisy/contrast-shifted 20px tiles."""
    t = tiles.astype(np.float32)
    pooled = t.reshape(-1, bins, TILE // bins, bins, TILE // bins, 3).mean(axis=(2, 4))
    flat = pooled.reshape(len(tiles), -1)
    normalized = flat - flat.mean(axis=1, keepdims=True)
    normalized = normalized / (normalized.std(axis=1, keepdims=True) + 1e-6)

    # Keep a weak absolute-color cue. It helps on repeated textures but is not
    # allowed to dominate brightness/contrast-corrupted descriptors.
    raw = (flat / 255.0) * 0.35
    means = (t.mean(axis=(1, 2)) / 255.0) * 0.35
    stds = (t.std(axis=(1, 2)) / 255.0) * 0.35
    return np.concatenate([normalized, raw, means, stds], axis=1).astype(np.float32)


def match_tiles(input_img: np.ndarray, target_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_tiles = split_tiles(input_img)
    y_tiles = split_tiles(target_img)
    x_feat = tile_features(x_tiles)
    y_feat = tile_features(y_tiles)
    diff = x_feat[:, None, :] - y_feat[None, :, :]
    cost = np.mean(diff * diff, axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = np.empty(GRID * GRID, dtype=np.int16)
    assigned_cost = np.empty(GRID * GRID, dtype=np.float32)
    mapping[row_ind] = col_ind.astype(np.int16)
    assigned_cost[row_ind] = cost[row_ind, col_ind].astype(np.float32)

    nearest_two = np.partition(cost, kth=1, axis=1)[:, :2]
    nearest_two.sort(axis=1)
    margin = (nearest_two[:, 1] - nearest_two[:, 0]).astype(np.float32)
    return mapping, assigned_cost, margin


def load_maps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=False)
    names = data["names"].astype(str)
    maps = data["maps"].astype(np.int16)
    costs = data["costs"].astype(np.float32) if "costs" in data else None
    margins = data["margins"].astype(np.float32) if "margins" in data else None
    return names, maps, costs, margins


def cmd_build_maps(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    input_dir = data_root / "train" / "inputs"
    target_dir = data_root / "train" / "targets"
    names = sorted(p.name for p in input_dir.glob("*.png") if (target_dir / p.name).exists())
    if args.offset:
        names = names[args.offset :]
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit("no matched train input/target PNG names found")

    maps = np.empty((len(names), GRID * GRID), dtype=np.int16)
    costs = np.empty((len(names), GRID * GRID), dtype=np.float32)
    margins = np.empty((len(names), GRID * GRID), dtype=np.float32)
    started = time.time()

    for i, name in enumerate(tqdm(names, desc="matching images")):
        mapping, assigned_cost, margin = match_tiles(read_rgb(input_dir / name), read_rgb(target_dir / name))
        maps[i] = mapping
        costs[i] = assigned_cost
        margins[i] = margin

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        names=np.asarray(names),
        maps=maps,
        costs=costs,
        margins=margins,
        meta=json.dumps(
            {
                "data_root": str(data_root),
                "count": len(names),
                "grid": GRID,
                "tile": TILE,
                "mean_cost": float(costs.mean()),
                "median_cost": float(np.median(costs)),
                "mean_margin": float(margins.mean()),
                "seconds": time.time() - started,
            },
            sort_keys=True,
        ),
    )
    print(f"saved {out}")
    print(f"images={len(names)} mean_cost={costs.mean():.4f} median_cost={np.median(costs):.4f} mean_margin={margins.mean():.4f}")


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    return torch, nn, F, DataLoader, Dataset


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class TileCacheDatasetBase:
    pass


def make_dataset_class(Dataset):
    class TileCacheDataset(Dataset):
        def __init__(
            self,
            data_root: Path,
            names: np.ndarray,
            maps: np.ndarray,
            image_indices: np.ndarray,
            augment: bool,
            max_pairs: int | None = None,
            max_cost: float | None = None,
            costs: np.ndarray | None = None,
            seed: int = 42,
        ) -> None:
            xs = []
            ys = []
            input_dir = data_root / "train" / "inputs"
            target_dir = data_root / "train" / "targets"
            for map_row, idx in enumerate(tqdm(image_indices, desc="caching tile pairs")):
                name = str(names[idx])
                x_tiles = split_tiles(read_rgb(input_dir / name))
                y_tiles = split_tiles(read_rgb(target_dir / name))
                clean_shuffled = y_tiles[maps[idx]]
                if max_cost is not None and costs is not None:
                    keep = costs[idx] <= max_cost
                    x_tiles = x_tiles[keep]
                    clean_shuffled = clean_shuffled[keep]
                xs.append(x_tiles)
                ys.append(clean_shuffled)
            self.x = np.concatenate(xs, axis=0)
            self.y = np.concatenate(ys, axis=0)
            if max_pairs and max_pairs < len(self.x):
                rng = np.random.default_rng(seed)
                keep = rng.choice(len(self.x), size=max_pairs, replace=False)
                self.x = self.x[keep]
                self.y = self.y[keep]
            self.augment = augment

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx: int):
            x = self.x[idx]
            y = self.y[idx]
            if self.augment:
                if random.random() < 0.5:
                    x = x[:, ::-1].copy()
                    y = y[:, ::-1].copy()
                if random.random() < 0.5:
                    x = x[::-1].copy()
                    y = y[::-1].copy()
                if random.random() < 0.5:
                    x = x.transpose(1, 0, 2).copy()
                    y = y.transpose(1, 0, 2).copy()
            x = np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32) / 255.0
            y = np.ascontiguousarray(y.transpose(2, 0, 1)).astype(np.float32) / 255.0
            return x, y

    return TileCacheDataset


def make_model_class(nn):
    class ResBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
            )

        def forward(self, x):
            return x + self.net(x) * 0.2

    class TileRestorer(nn.Module):
        def __init__(self, width: int = 64, depth: int = 8) -> None:
            super().__init__()
            self.head = nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect"),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
            self.tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")

        def forward(self, x):
            residual = self.tail(self.body(self.head(x)))
            return (x + residual).clamp(0.0, 1.0)

    return TileRestorer


def gradient_loss(F, pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, targ_dx) + F.l1_loss(pred_dy, targ_dy)


def border_l1(F, pred, target, band: int = 3):
    mask = pred.new_zeros((1, 1, TILE, TILE))
    mask[:, :, :band, :] = 1
    mask[:, :, -band:, :] = 1
    mask[:, :, :, :band] = 1
    mask[:, :, :, -band:] = 1
    return F.l1_loss(pred * mask, target * mask)


def tensor_to_image(tensor, np_module=np) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    arr = np_module.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr.transpose(0, 2, 3, 1)


def restore_image_model(model, torch, img: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    tiles = split_tiles(img)
    outs = []
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch = tiles[start : start + batch_size].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            pred = model(x)
            outs.append(tensor_to_image(pred))
    return merge_tiles(np.concatenate(outs, axis=0))


def restore_image_classical(img: np.ndarray, h: float = 7.0, h_color: float = 7.0) -> np.ndarray:
    try:
        import cv2

        out_tiles = []
        for tile in split_tiles(img):
            bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h_color, 5, 11)
            out_tiles.append(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
        return merge_tiles(np.asarray(out_tiles, dtype=np.uint8))
    except Exception:
        from scipy.ndimage import median_filter

        return median_filter(img, size=(1, 1, 1)).astype(np.uint8)


def pseudo_clean_shuffled(data_root: Path, name: str, mapping: np.ndarray) -> np.ndarray:
    target_tiles = split_tiles(read_rgb(data_root / "train" / "targets" / name))
    return merge_tiles(target_tiles[mapping])


def image_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    return {
        "ssim": float(structural_similarity(target, pred, channel_axis=2, data_range=255)),
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=255)),
        "mae": float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32)))),
    }


def evaluate_restorer(
    data_root: Path,
    names: np.ndarray,
    maps: np.ndarray,
    indices: np.ndarray,
    model,
    torch,
    device: str,
    batch_size: int,
    method: str,
    blend_raw: float = 0.0,
) -> dict[str, float]:
    raw_scores = []
    pred_scores = []
    for idx in tqdm(indices, desc="evaluating"):
        name = str(names[idx])
        img = read_rgb(data_root / "train" / "inputs" / name)
        target = pseudo_clean_shuffled(data_root, name, maps[idx])
        if method == "copy":
            pred = img
        elif method == "classical":
            pred = restore_image_classical(img)
        else:
            pred = restore_image_model(model, torch, img, device, batch_size)
            if blend_raw:
                pred = np.clip((1.0 - blend_raw) * pred.astype(np.float32) + blend_raw * img.astype(np.float32), 0, 255).astype(np.uint8)
        raw_scores.append(image_metrics(img, target))
        pred_scores.append(image_metrics(pred, target))

    result = {}
    for key in raw_scores[0]:
        result[f"raw_{key}"] = float(np.mean([s[key] for s in raw_scores]))
        result[f"pred_{key}"] = float(np.mean([s[key] for s in pred_scores]))
        result[f"delta_{key}"] = result[f"pred_{key}"] - result[f"raw_{key}"]
    return result


def cmd_train(args: argparse.Namespace) -> None:
    torch, nn, F, DataLoader, Dataset = import_torch()
    TileCacheDataset = make_dataset_class(Dataset)
    TileRestorer = make_model_class(nn)
    seed_all(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    names, maps, costs, _margins = load_maps(Path(args.maps))
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(names))
    if args.shuffle_images:
        rng.shuffle(order)
    val_images = min(args.val_images, len(order))
    train_pool = order[:-val_images] if val_images else order
    val_idx = order[-val_images:] if val_images else order[: min(8, len(order))]
    if args.train_images:
        train_pool = train_pool[: args.train_images]
    if len(train_pool) == 0:
        raise SystemExit("no training images selected")

    max_cost = None
    if args.cost_quantile and costs is not None:
        max_cost = float(np.quantile(costs[train_pool].reshape(-1), args.cost_quantile))
        print(f"cost filter q={args.cost_quantile} max_cost={max_cost:.4f}")

    train_ds = TileCacheDataset(
        data_root,
        names,
        maps,
        train_pool,
        augment=True,
        max_pairs=args.max_pairs,
        max_cost=max_cost,
        costs=costs,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)

    device = choose_device(torch, args.device)
    model = TileRestorer(width=args.width, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    print(f"device={device} train_pairs={len(train_ds)} val_images={len(val_idx)}")

    best_ssim = -math.inf
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    saved_args = {k: v for k, v in vars(args).items() if k != "func"}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            if args.grad_weight:
                loss = loss + args.grad_weight * gradient_loss(F, pred, yb)
            if args.border_weight:
                loss = loss + args.border_weight * border_l1(F, pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
            pbar.set_postfix(loss=total / max(count, 1))
        scheduler.step()

        metrics = evaluate_restorer(data_root, names, maps, val_idx, model, torch, device, args.eval_batch_size, "model")
        metrics["epoch"] = epoch
        metrics["train_loss"] = total / max(count, 1)
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))

        if metrics["pred_ssim"] > best_ssim:
            best_ssim = metrics["pred_ssim"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "width": args.width,
                    "depth": args.depth,
                    "grid": GRID,
                    "tile": TILE,
                    "args": saved_args,
                    "history": history,
                },
                out,
            )
            print(f"saved best checkpoint {out} pred_ssim={best_ssim:.6f}")


def load_model(checkpoint: Path, device: str):
    torch, nn, _F, _DataLoader, _Dataset = import_torch()
    TileRestorer = make_model_class(nn)
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        # Compatibility for checkpoints produced before metadata was restricted
        # to plain data. Use this only for checkpoints created in this workspace.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TileRestorer(width=int(ckpt.get("width", 64)), depth=int(ckpt.get("depth", 8)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return torch, model


def cmd_eval(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    names, maps, _costs, _margins = load_maps(Path(args.maps))
    indices = np.arange(len(names))
    if args.val_images:
        indices = indices[-args.val_images :]
    if args.method == "model":
        torch, _nn, _F, _DataLoader, _Dataset = import_torch()
        device = choose_device(torch, args.device)
        torch, model = load_model(Path(args.checkpoint), device)
    else:
        torch = None
        model = None
        device = "cpu"
    metrics = evaluate_restorer(data_root, names, maps, indices, model, torch, device, args.batch_size, args.method, args.blend_raw)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_apply(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    names = sorted(p.name for p in input_dir.glob("*.png"))
    if args.limit:
        names = names[: args.limit]
    if args.method == "model":
        torch, _nn, _F, _DataLoader, _Dataset = import_torch()
        device = choose_device(torch, args.device)
        torch, model = load_model(Path(args.checkpoint), device)
    else:
        torch = None
        model = None
        device = "cpu"
    print(f"applying method={args.method} device={device} files={len(names)}")
    for name in tqdm(names, desc="restoring"):
        img = read_rgb(input_dir / name)
        if args.method == "copy":
            pred = img
        elif args.method == "classical":
            pred = restore_image_classical(img)
        else:
            pred = restore_image_model(model, torch, img, device, args.batch_size)
            if args.blend_raw:
                pred = np.clip((1.0 - args.blend_raw) * pred.astype(np.float32) + args.blend_raw * img.astype(np.float32), 0, 255).astype(np.uint8)
        write_rgb(out_dir / name, pred)


def cmd_zip_dir(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.png"))
    if len(files) == 0:
        raise SystemExit(f"no PNG files in {input_dir}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in tqdm(files, desc="zipping"):
            zf.write(path, arcname=path.name)
    print(f"wrote {out} files={len(files)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-maps", help="match train input tiles to clean target tiles")
    p.add_argument("--data-root", default="puzzle")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(func=cmd_build_maps)

    p = sub.add_parser("train", help="train residual tile restorer")
    p.add_argument("--data-root", default="puzzle")
    p.add_argument("--maps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-images", type=int, default=0)
    p.add_argument("--val-images", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-weight", type=float, default=0.15)
    p.add_argument("--border-weight", type=float, default=0.20)
    p.add_argument("--cost-quantile", type=float, default=0.0)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--shuffle-images", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="evaluate copy/classical/model against pseudo-clean shuffled targets")
    p.add_argument("--data-root", default="puzzle")
    p.add_argument("--maps", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--method", choices=["copy", "classical", "model"], default="model")
    p.add_argument("--val-images", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--blend-raw", type=float, default=0.0, help="for model eval, blend this fraction of raw input back into predictions")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("apply", help="restore all PNG files in an input directory")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--method", choices=["copy", "classical", "model"], default="model")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--blend-raw", type=float, default=0.0, help="for model apply, blend this fraction of raw input back into predictions")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("zip-dir", help="zip restored PNG files at archive root")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_zip_dir)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
