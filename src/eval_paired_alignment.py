"""Stage 1 oracle gate for the factorized 2D-hole scorer (branch E).

Every prior geometry branch matched a dirty tile against *other* tiles: seam
compatibility (destroyed by independent per-tile noise), absolute grid slot
(no transferable signal), or a soft candidate graph (recall ~0.67 but no
decoder converts it into a placement).  This gate asks a question none of
them tested directly: after one independent application of the challenge
degradation (contrast/brightness affine, sigma 40-55 noise, 3x3 blur, JPEG
35-50), does a tile's content still identify *which specific clean patch it
came from*, among hundreds of visually similar candidates from the same
photograph?

This is deliberately not a seam or position experiment.  Two small encoders
(one for dirty tiles, one for clean tiles) are trained with a symmetric
InfoNCE / CLIP-style contrastive loss on exact synthetic (dirty_i, clean_i)
pairs -- no shuffle, no permutation cache, since a per-tile CNN with no
positional input cannot see grid order anyway.  The held-out gate reports
retrieval of the true counterpart among the *same photograph's* 576 tiles
(the practically relevant pool for the later halo-conditioned scorer) and,
diagnostically, among a much larger cross-image pool.

Only if this gate passes does branch E proceed to stage 2 (a clean masked
halo model) and stage 3 (combining both to rank real dirty candidates from
predicted-clean context).  Nothing here touches assembly or test images.

Examples
--------

    python src/eval_paired_alignment.py --smoke
    python src/eval_paired_alignment.py --steps 1500 --bs 4 --tiles-per-image 192 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import CKPT_DIR, FS, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags, train_val_split


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TileEncoder(nn.Module):
    """Small per-tile CNN with no positional input; raw + exposure-normalized view."""

    def __init__(self, embed_dim: int = 128, width: int = 0) -> None:
        super().__init__()
        if embed_dim < 8:
            raise ValueError("embed_dim must be at least 8")
        base = width or max(24, embed_dim // 5)
        middle = max(32, embed_dim // 2)
        self.embed_dim = int(embed_dim)
        self.features = nn.Sequential(
            nn.Conv2d(6, base, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base), base),
            nn.GELU(),
            nn.Conv2d(base, middle, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(middle), middle),
            nn.GELU(),
            nn.Conv2d(middle, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(embed_dim), embed_dim),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 4 or tuple(tiles.shape[1:]) != (3, FS, FS):
            raise ValueError(f"tiles must have shape (N,3,{FS},{FS}), got {tuple(tiles.shape)}")
        mean = tiles.mean(dim=(-3, -2, -1), keepdim=True)
        rms = (tiles - mean).square().mean(dim=(-3, -2, -1), keepdim=True)
        normalized = ((tiles - mean) / rms.add(1.0e-5).sqrt()).clamp(-5.0, 5.0)
        x = self.features(torch.cat((tiles, normalized), dim=1))
        flat = x.flatten(start_dim=2)
        stats = torch.cat(
            (flat.mean(dim=-1), flat.var(dim=-1, unbiased=False).add(1.0e-6).sqrt()), dim=-1
        )
        return F.normalize(self.project(stats), dim=-1)


class PairedAlignment(nn.Module):
    """Two independent tile encoders plus a learnable CLIP-style temperature."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.dirty_encoder = TileEncoder(embed_dim)
        self.clean_encoder = TileEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def scale(self) -> Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(self, dirty: Tensor, clean: Tensor) -> tuple[Tensor, Tensor]:
        return self.dirty_encoder(dirty), self.clean_encoder(clean)


def symmetric_info_nce(dirty_embed: Tensor, clean_embed: Tensor, scale: Tensor) -> Tensor:
    """CLIP-style symmetric contrastive loss; the diagonal is the true pairing."""
    if dirty_embed.shape != clean_embed.shape or dirty_embed.ndim != 2:
        raise ValueError("dirty_embed and clean_embed must share shape (N,D)")
    logits = (dirty_embed @ clean_embed.t()) * scale
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def rank_of_diagonal(similarity: Tensor) -> Tensor:
    """1-based rank of the true (diagonal) match within each row."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square (N,N) matrix")
    true_score = similarity.diagonal().unsqueeze(1)
    return similarity.gt(true_score).sum(dim=1) + 1


def retrieval_summary(rank: Tensor, prefix: str) -> dict[str, float]:
    rank = rank.float()
    return {
        f"{prefix}_r1": float(rank.le(1).float().mean()),
        f"{prefix}_r5": float(rank.le(5).float().mean()),
        f"{prefix}_r10": float(rank.le(10).float().mean()),
        f"{prefix}_median_rank": float(rank.median()),
        f"{prefix}_mrr": float(rank.reciprocal().mean()),
        f"{prefix}_n": int(rank.numel()),
    }


def _to_tensor(tiles_uint8: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles_uint8))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .to(device)
    )


def _synthetic_pair(name: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    clean = to_frags(load(os.path.join(TRAIN_TGT, name)))
    dirty = distort_frags(clean, rng)
    return dirty, clean


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(
    model: PairedAlignment, names: list[str], *, device: torch.device, seed: int
) -> dict[str, float]:
    """Held-out retrieval: same-image pool (practical) and pooled cross-image (diagnostic)."""
    was_training = model.training
    model.eval()
    per_image_c2d: list[Tensor] = []
    per_image_d2c: list[Tensor] = []
    all_dirty: list[Tensor] = []
    all_clean: list[Tensor] = []
    for index, name in enumerate(names):
        rng = np.random.default_rng(seed + index * 7919)
        dirty, clean = _synthetic_pair(name, rng)
        dirty_t = _to_tensor(dirty, device)
        clean_t = _to_tensor(clean, device)
        with _autocast(device):
            dirty_embed, clean_embed = model(dirty_t, clean_t)
        dirty_embed, clean_embed = dirty_embed.float(), clean_embed.float()
        similarity_c2d = clean_embed @ dirty_embed.t()
        similarity_d2c = dirty_embed @ clean_embed.t()
        per_image_c2d.append(rank_of_diagonal(similarity_c2d))
        per_image_d2c.append(rank_of_diagonal(similarity_d2c))
        all_dirty.append(dirty_embed)
        all_clean.append(clean_embed)
    if was_training:
        model.train()

    same_image = {
        **retrieval_summary(torch.cat(per_image_c2d), "same_image_clean_to_dirty"),
        **retrieval_summary(torch.cat(per_image_d2c), "same_image_dirty_to_clean"),
    }
    pooled_dirty = torch.cat(all_dirty)
    pooled_clean = torch.cat(all_clean)
    pooled_similarity = pooled_clean @ pooled_dirty.t()
    pooled = retrieval_summary(rank_of_diagonal(pooled_similarity), "pooled_clean_to_dirty")
    pooled["pooled_images"] = len(names)
    pooled["pooled_candidates"] = int(pooled_dirty.shape[0])
    return {**same_image, **pooled}


def _checkpoint(path: str, model: PairedAlignment, *, step: int, metrics: dict[str, Any]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "embed_dim": model.dirty_encoder.embed_dim,
            "step": int(step),
            "metrics": metrics,
        },
        path,
    )


def smoke(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    """Data-free contract test: shapes, gradients, and a perfect-similarity rank check."""
    torch.manual_seed(4242)
    model = PairedAlignment(embed_dim=16).to(device)
    dirty = torch.rand(12, 3, FS, FS, device=device, requires_grad=True)
    clean = torch.rand(12, 3, FS, FS, device=device, requires_grad=True)
    dirty_embed, clean_embed = model(dirty, clean)
    if dirty_embed.shape != (12, 16) or clean_embed.shape != (12, 16):
        raise AssertionError(f"unexpected embedding shapes {dirty_embed.shape} {clean_embed.shape}")
    if not torch.allclose(dirty_embed.norm(dim=-1), torch.ones(12, device=device), atol=1.0e-5):
        raise AssertionError("dirty embeddings are not L2-normalized")
    loss = symmetric_info_nce(dirty_embed, clean_embed, model.scale())
    loss.backward()
    if dirty.grad is None or not torch.isfinite(dirty.grad).all():
        raise AssertionError("paired alignment loss lost a finite input gradient")
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
        raise AssertionError("model parameters received no gradient")

    perfect = torch.eye(8, device=device) * 10.0 - 5.0
    rank = rank_of_diagonal(perfect)
    metrics = retrieval_summary(rank, "toy")
    if metrics["toy_r1"] < 0.999 or metrics["toy_mrr"] < 0.999:
        raise AssertionError(f"perfect-similarity smoke rank check failed: {metrics}")

    # Permuting both axes by the same order keeps the diagonal the true match
    # (both embedding sets are re-indexed together, as a shuffled bag would be).
    permutation = torch.randperm(8)
    shuffled = perfect[permutation][:, permutation]
    shuffled_rank = rank_of_diagonal(shuffled)
    if retrieval_summary(shuffled_rank, "shuf")["shuf_r1"] < 0.999:
        raise AssertionError("rank_of_diagonal is not invariant to joint row/column reindexing")

    return {
        "loss": float(loss.detach()),
        "parameters": sum(p.numel() for p in model.parameters()),
        **metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--bs", type=int, default=4, help="images per optimizer step")
    parser.add_argument("--tiles-per-image", "--tiles_per_image", dest="tiles_per_image", type=int, default=192)
    parser.add_argument("--embed-dim", "--embed_dim", dest="embed_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=300)
    parser.add_argument("--eval-images", "--eval_images", dest="eval_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="paired_alignment")
    parser.add_argument("--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", default=CKPT_DIR)
    parser.add_argument(
        "--report", type=Path, default=Path("E:/pazzle_work/gates/paired_alignment_gate.json")
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return args
    if args.steps < 1 or args.bs < 1 or args.tiles_per_image < 1 or args.eval_images < 1:
        parser.error("--steps, --bs, --tiles-per-image, and --eval-images must be positive")
    if args.tiles_per_image > NFRAG:
        parser.error(f"--tiles-per-image must not exceed {NFRAG}")
    if args.embed_dim < 8 or args.lr <= 0.0 or args.eval_every < 1:
        parser.error("invalid --embed-dim/--lr/--eval-every")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        print(f"[paired-alignment smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    train_names, val_names = train_val_split()
    if len(val_names) < args.eval_images:
        raise ValueError(f"--eval-images exceeds the held-out pool ({len(val_names)})")

    model = PairedAlignment(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} PairedAlignment params={sum(p.numel() for p in model.parameters()):,} "
        f"embed_dim={args.embed_dim} bs={args.bs} tiles/image={args.tiles_per_image} "
        "objective=symmetric-InfoNCE(dirty_i, clean_i) on exact synthetic pairs",
        flush=True,
    )
    print(
        "question: does one independent application of the challenge degradation still let a "
        "tile's own clean counterpart be found among ~576 same-photo candidates?",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 1)
    best_gate_metric = -float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        names = [train_names[int(i)] for i in rng.integers(0, len(train_names), size=args.bs)]
        dirty_batch: list[np.ndarray] = []
        clean_batch: list[np.ndarray] = []
        for name in names:
            image_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            dirty, clean = _synthetic_pair(name, image_rng)
            picked = image_rng.choice(NFRAG, size=args.tiles_per_image, replace=False)
            dirty_batch.append(dirty[picked])
            clean_batch.append(clean[picked])
        dirty_tensor = _to_tensor(np.concatenate(dirty_batch), device)
        clean_tensor = _to_tensor(np.concatenate(clean_batch), device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            dirty_embed, clean_embed = model(dirty_tensor, clean_tensor)
            loss = symmetric_info_nce(dirty_embed.float(), clean_embed.float(), model.scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                train_rank = rank_of_diagonal((dirty_embed.float() @ clean_embed.float().t()))
                train_metrics = retrieval_summary(train_rank, "train_in_batch")
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"in_batch_r1={train_metrics['train_in_batch_r1']:.4f} "
                f"in_batch_r5={train_metrics['train_in_batch_r5']:.4f} "
                f"candidates={dirty_embed.shape[0]} lr={scheduler.get_last_lr()[0]:.3e} "
                f"{elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val_names[: args.eval_images], device=device, seed=args.seed + 9973)
            print(f"[SYN paired-alignment held-out] step={step} {metrics}", flush=True)
            last_path = os.path.join(args.ckpt_dir, f"{args.tag}_last.pt")
            _checkpoint(last_path, model, step=step, metrics=metrics)
            gate_metric = metrics["same_image_clean_to_dirty_r1"]
            if gate_metric > best_gate_metric:
                best_gate_metric = gate_metric
                best_path = os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")
                _checkpoint(best_path, model, step=step, metrics=metrics)
                print(f"saved best same_image_clean_to_dirty_r1={best_gate_metric:.4f}", flush=True)

    final_metrics = evaluate(model, val_names[: args.eval_images], device=device, seed=args.seed + 9973)
    passed = (
        final_metrics["same_image_clean_to_dirty_r1"] >= 0.05
        and final_metrics["same_image_clean_to_dirty_r5"] >= 0.15
    )
    report = {
        "experiment": "stage1_paired_alignment_dirty_to_clean_tile_identity",
        "question": (
            "does one independent challenge-degradation instance still let a tile's own clean "
            "counterpart be identified among ~576 same-photo candidates?"
        ),
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "device": str(device)},
        "final_metrics": final_metrics,
        "gate": {
            "rule": (
                "same_image_clean_to_dirty_r1 >= 0.05 (~29x chance of 1/576) "
                "AND same_image_clean_to_dirty_r5 >= 0.15"
            ),
            "chance_r1": 1.0 / NFRAG,
            "chance_r5": 5.0 / NFRAG,
            "pass": bool(passed),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> proceed to stage 2 (clean halo model)" if passed else "FAILED -> close branch E stage 1"
    print(f"\n=== stage 1 gate {verdict} ===", flush=True)
    print(json.dumps(report["gate"], indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
