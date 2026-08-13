"""PGA1: compact hierarchical set-to-slot Sinkhorn evidence gate for ORBIT-24.

The input order is intentionally absent from every feature.  Each example is an
unordered tile set; fixed learned grid-slot queries produce full tile->slot
logits.  Sinkhorn is used only while learning, then Hungarian produces an exact
bijection.  Clean targets are read only to create synthetic labels and post-hoc
held-out SSIM metrics; no test images are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim

from config import FS, GRID, NFRAG, TRAIN_INP, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags


DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot")
DEFAULT_SPLIT = DEFAULT_WORK / "source_disjoint_split_v1.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(seed: int, name: str, salt: str) -> int:
    raw = f"{seed}\0{name}\0{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32)


def log_sinkhorn(logits: torch.Tensor, temperature: float, rounds: int) -> torch.Tensor:
    """Log-space tile->slot doubly-stochastic relaxation."""
    z = logits / temperature
    for _ in range(rounds):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return z


def inverse_permutation(tile_to_slot: torch.Tensor) -> torch.Tensor:
    bsz, count = tile_to_slot.shape
    out = torch.empty_like(tile_to_slot)
    rows = torch.arange(count, device=tile_to_slot.device).expand(bsz, -1)
    out.scatter_(1, tile_to_slot, rows)
    return out


def tiles_to_image(tiles: np.ndarray) -> np.ndarray:
    if tiles.shape != (NFRAG, FS, FS, 3):
        raise ValueError(f"expected ({NFRAG},{FS},{FS},3), got {tiles.shape}")
    return tiles.reshape(GRID, GRID, FS, FS, 3).transpose(0, 2, 1, 3, 4).reshape(GRID * FS, GRID * FS, 3)


def tile_low_frequency(tiles: torch.Tensor) -> torch.Tensor:
    # (B,N,3,20,20) -> (B,N,3*4*4), retaining only an auxiliary global layout signal.
    bsz, count, channels, height, width = tiles.shape
    x = F.avg_pool2d(tiles.reshape(bsz * count, channels, height, width), kernel_size=5, stride=5)
    return x.reshape(bsz, count, -1)


def macro_index(slot: np.ndarray) -> np.ndarray:
    row, col = np.divmod(slot, GRID)
    return (row // 4) * 6 + (col // 4)


class TileStem(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        mid = max(24, d // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(3, mid, 3, padding=1), nn.GroupNorm(4, mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1), nn.GroupNorm(4, mid), nn.GELU(),
            nn.Conv2d(mid, d, 3, stride=2, padding=1), nn.GroupNorm(8, d), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.stats = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        bsz, count, channels, height, width = tiles.shape
        flat = tiles.reshape(bsz * count, channels, height, width)
        visual = self.conv(flat).flatten(1).reshape(bsz, count, -1)
        stats = torch.cat((tiles.mean(dim=(-1, -2)), tiles.std(dim=(-1, -2), correction=0)), dim=-1)
        return self.norm(visual + self.stats(stats))


class SetBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 3), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 3, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class PGASlotTransformer(nn.Module):
    def __init__(self, d: int = 96, heads: int = 4, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("d must divide heads")
        self.d = d
        self.stem = TileStem(d)
        self.set_blocks = nn.ModuleList([SetBlock(d, heads, dropout) for _ in range(layers)])
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing="ij"), dim=-1).reshape(NFRAG, 2)
        freqs = torch.tensor([1.0, 2.0, 4.0, 8.0])
        fixed = torch.cat([torch.sin(math.pi * coords[:, :1] * freqs), torch.cos(math.pi * coords[:, :1] * freqs), torch.sin(math.pi * coords[:, 1:] * freqs), torch.cos(math.pi * coords[:, 1:] * freqs)], dim=1)
        self.register_buffer("fixed_slot_features", fixed, persistent=False)
        self.slot_base = nn.Parameter(torch.randn(NFRAG, d) * 0.02)
        self.slot_project = nn.Sequential(nn.Linear(fixed.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.cross_norm = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.tile_proj = nn.Linear(d, d, bias=False)
        self.slot_proj = nn.Linear(d, d, bias=False)
        self.tile_bias = nn.Linear(d, 1, bias=False)
        self.slot_bias = nn.Linear(d, 1, bias=False)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        token = self.stem(tiles)
        for block in self.set_blocks:
            token = block(token)
        bsz = token.shape[0]
        slots = self.slot_base + self.slot_project(self.fixed_slot_features)
        slots = slots.unsqueeze(0).expand(bsz, -1, -1)
        query = self.cross_norm(slots)
        slots = slots + self.cross(query, token, token, need_weights=False)[0]
        scale = self.d ** -0.5
        scores = torch.einsum("bnd,bmd->bnm", self.tile_proj(token), self.slot_proj(slots)) * scale
        return scores + self.tile_bias(token) + self.slot_bias(slots).transpose(1, 2)


def loss_components(logits: torch.Tensor, labels: torch.Tensor, tiles: torch.Tensor, clean_tiles: torch.Tensor, temperature: float, rounds: int, aux_weight: float) -> dict[str, torch.Tensor]:
    logp = log_sinkhorn(logits, temperature, rounds)
    row_nll = -logp.gather(2, labels.unsqueeze(-1)).mean()
    inv = inverse_permutation(labels)
    col_nll = -logp.transpose(1, 2).gather(2, inv.unsqueeze(-1)).mean()
    soft = logp.exp()
    pred_low = torch.bmm(soft.transpose(1, 2), tile_low_frequency(tiles))
    clean_low = tile_low_frequency(clean_tiles)
    low = F.smooth_l1_loss(pred_low, clean_low)
    total = 0.5 * (row_nll + col_nll) + aux_weight * low
    return {"total": total, "row": row_nll, "col": col_nll, "low": low}


class TileBank:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.clean: list[np.ndarray] = []
        for ordinal, name in enumerate(names, 1):
            image = load(str(Path(TRAIN_TGT) / name))
            fragments = np.ascontiguousarray(to_frags(image))
            if fragments.shape != (NFRAG, FS, FS, 3):
                raise RuntimeError(f"unexpected fragments {fragments.shape} for {name}")
            self.clean.append(fragments)
            if ordinal % 64 == 0:
                print(f"preloaded {ordinal}/{len(names)} clean boards", flush=True)

    def synthetic_batch(self, indices: np.ndarray, step: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dirty_rows: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        clean_rows: list[np.ndarray] = []
        for local, index in enumerate(indices.tolist()):
            clean = self.clean[index]
            rng = np.random.default_rng((seed + 1_000_003 * step + 7_919 * index + local) % (2**32))
            dirty = np.ascontiguousarray(distort_frags(clean, rng))
            permutation = rng.permutation(NFRAG).astype(np.int64)
            dirty_rows.append(dirty[permutation])
            labels.append(permutation)
            clean_rows.append(clean)
        dirty_t = torch.from_numpy(np.stack(dirty_rows)).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device, non_blocking=True)
        clean_t = torch.from_numpy(np.stack(clean_rows)).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device, non_blocking=True)
        label_t = torch.from_numpy(np.stack(labels)).long().to(device, non_blocking=True)
        return dirty_t, label_t, clean_t


def decode_hungarian(logits: torch.Tensor) -> np.ndarray:
    scores = logits.detach().float().cpu().numpy()
    output = np.empty((scores.shape[0], NFRAG), dtype=np.int64)
    for batch_index, matrix in enumerate(scores):
        rows, cols = linear_sum_assignment(-matrix)
        output[batch_index, rows] = cols
    return output


def model_ssim_for_assignment(dirty_tiles: np.ndarray, tile_to_slot: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    assembled = np.empty_like(dirty_tiles)
    assembled[tile_to_slot] = dirty_tiles
    raw = tiles_to_image(dirty_tiles)
    recovered = tiles_to_image(assembled)
    return (
        float(ssim(raw, target, channel_axis=2, data_range=255)),
        float(ssim(recovered, target, channel_axis=2, data_range=255)),
    )


def run_eval(model: PGASlotTransformer, bank: TileBank, names: list[str], seed: int, device: torch.device, temperature: float, rounds: int, real_eval: bool) -> dict[str, float]:
    model.eval()
    synth_top1: list[float] = []
    synth_macro: list[float] = []
    synth_raw_ssim: list[float] = []
    synth_ssim: list[float] = []
    real_raw_ssim: list[float] = []
    real_ssim: list[float] = []
    with torch.no_grad():
        for index, name in enumerate(names):
            dirty_t, label_t, _ = bank.synthetic_batch(np.array([index]), step=0, seed=stable_seed(seed, name, "eval"), device=device)
            logits = model(dirty_t)
            pred = decode_hungarian(logits)[0]
            true = label_t.cpu().numpy()[0]
            synth_top1.append(float((pred == true).mean()))
            synth_macro.append(float((macro_index(pred) == macro_index(true)).mean()))
            clean_target = tiles_to_image(bank.clean[index])
            dirty_uint8 = np.clip(np.rint(dirty_t[0].permute(0, 2, 3, 1).cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            raw_score, recovered_score = model_ssim_for_assignment(dirty_uint8, pred, clean_target)
            synth_raw_ssim.append(raw_score)
            synth_ssim.append(recovered_score)
            if real_eval:
                real_dirty = np.ascontiguousarray(to_frags(load(str(Path(TRAIN_INP) / name))))
                real_t = torch.from_numpy(real_dirty[None]).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device)
                real_pred = decode_hungarian(model(real_t))[0]
                raw_score, recovered_score = model_ssim_for_assignment(real_dirty, real_pred, clean_target)
                real_raw_ssim.append(raw_score)
                real_ssim.append(recovered_score)
    result = {
        "synthetic_tile_top1": float(np.mean(synth_top1)),
        "synthetic_macro_membership": float(np.mean(synth_macro)),
        "synthetic_raw_ssim": float(np.mean(synth_raw_ssim)),
        "synthetic_ssim": float(np.mean(synth_ssim)),
        "synthetic_ssim_delta": float(np.mean(synth_ssim) - np.mean(synth_raw_ssim)),
    }
    if real_eval:
        result.update({
            "real_raw_ssim": float(np.mean(real_raw_ssim)),
            "real_ssim": float(np.mean(real_ssim)),
            "real_ssim_delta": float(np.mean(real_ssim) - np.mean(real_raw_ssim)),
        })
    return result


def autocast_for(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def read_split(path: Path, partition: str, limit: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload["splits"][partition]
    if limit > 0:
        names = names[:limit]
    if not names:
        raise ValueError("empty selected split")
    return names


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "overfit", "gate"), default="smoke")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--seed", type=int, default=2413)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--train_n", type=int, default=8)
    parser.add_argument("--eval_n", type=int, default=4)
    parser.add_argument("--d", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--sinkhorn_rounds", type=int, default=8)
    parser.add_argument("--aux_weight", type=float, default=0.03)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.batch < 1 or args.steps < 1 or args.eval_n < 1:
        parser.error("steps, batch and eval_n must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PGA1 gate is local-GPU-only by task requirement")
    torch.backends.cudnn.benchmark = True
    train_limit = 2 if args.mode == "overfit" else args.train_n
    train_names = read_split(args.split, "fit", train_limit)
    dev_names = read_split(args.split, "dev", args.eval_n)
    all_names = train_names + [name for name in dev_names if name not in set(train_names)]
    bank_all = TileBank(all_names)
    train_bank = TileBank.__new__(TileBank)
    train_bank.names, train_bank.clean = train_names, bank_all.clean[:len(train_names)]
    dev_bank = TileBank.__new__(TileBank)
    dev_bank.names, dev_bank.clean = dev_names, bank_all.clean[len(train_names):]
    model = PGASlotTransformer(args.d, args.heads, args.layers, args.dropout).to(device)
    params = sum(param.numel() for param in model.parameters())
    print(json.dumps({"mode": args.mode, "device": str(device), "params": params, "train_n": len(train_names), "dev_n": len(dev_names)}), flush=True)
    if args.mode == "smoke":
        tiles, labels, clean = train_bank.synthetic_batch(np.array([0]), step=1, seed=args.seed, device=device)
        with autocast_for(device):
            logits = model(tiles)
            losses = loss_components(logits, labels, tiles, clean, args.temperature, args.sinkhorn_rounds, args.aux_weight)
        summary = {key: float(value.detach().float()) for key, value in losses.items()}
        summary.update({"shape": list(logits.shape), "params": params, "cuda_max_allocated": int(torch.cuda.max_memory_allocated())})
        print(json.dumps(summary, indent=2), flush=True)
        if args.report:
            save_json(args.report, {"experiment": "PGA1_smoke", "args": vars(args), "summary": summary})
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(args.seed + 17)
    started = time.time()
    for step in range(1, args.steps + 1):
        chosen = generator.integers(0, len(train_names), size=args.batch, endpoint=False)
        tiles, labels, clean = train_bank.synthetic_batch(chosen, step=step, seed=args.seed, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device):
            logits = model(tiles)
            losses = loss_components(logits, labels, tiles, clean, args.temperature, args.sinkhorn_rounds, args.aux_weight)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            pred = decode_hungarian(logits)[0]
            truth = labels[0].detach().cpu().numpy()
            print(json.dumps({"step": step, "loss": float(losses['total'].detach().float()), "row": float(losses['row'].detach().float()), "low": float(losses['low'].detach().float()), "batch_top1": float((pred == truth).mean()), "seconds": round(time.time() - started, 2)}), flush=True)

    eval_names = train_names if args.mode == "overfit" else dev_names
    eval_bank = train_bank if args.mode == "overfit" else dev_bank
    metrics = run_eval(model, eval_bank, eval_names, args.seed, device, args.temperature, args.sinkhorn_rounds, real_eval=(args.mode == "gate"))
    gate = {
        "overfit_min_tile_top1": 0.95,
        "macro_min": 0.0656,
        "synthetic_ssim_delta_min": 0.0,
        "real_ssim_delta_min": 0.0,
    }
    decision = {
        "overfit_pass": metrics["synthetic_tile_top1"] >= gate["overfit_min_tile_top1"] if args.mode == "overfit" else None,
        "macro_pass": metrics["synthetic_macro_membership"] >= gate["macro_min"] if args.mode == "gate" else None,
        "synthetic_ssim_pass": metrics["synthetic_ssim_delta"] > gate["synthetic_ssim_delta_min"] if args.mode == "gate" else None,
        "real_ssim_pass": metrics.get("real_ssim_delta", -float("inf")) > gate["real_ssim_delta_min"] if args.mode == "gate" else None,
    }
    decision["pass"] = bool(decision["overfit_pass"]) if args.mode == "overfit" else bool(decision["macro_pass"] and decision["synthetic_ssim_pass"] and decision["real_ssim_pass"])
    report = {"experiment": "PGA1_set_slot_sinkhorn", "mode": args.mode, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "params": params, "metrics": metrics, "gate": gate, "decision": decision, "elapsed_seconds": time.time() - started}
    destination = args.report or (args.work / f"{args.mode}_report.json")
    save_json(destination, report)
    checkpoint = args.checkpoint or (args.work / f"{args.mode}_best.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, checkpoint)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
