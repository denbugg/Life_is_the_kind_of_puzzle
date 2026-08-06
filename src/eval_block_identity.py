"""Branch G stage 1: does one dirty tile identify its containing 4x4 macro-block?

This project's own earliest diagnostic (macro_oracle) established the real
bottleneck precisely: given the TRUE 16-tile group for a 4x4 block, the
existing scorer already solves that block with placement~=0.68,
neighbour~=0.72. Every later branch (affinity graphs, candidate-matched
listwise ranking, Frontier Pointer, halo context) tried to recover exact
tile-to-tile ADJACENCY and capped at recall/precision far below what
assembly needs, because a single 20x20 tile's *local* neighbourhood does not
determine its identity (branch E, this session).

This asks a different, coarser question with a fundamentally different
positive signal-to-noise ratio: not "which exact tile is my neighbour" but
"which of 36 same-image 4x4 (80x80) macro-blocks do I belong to". An 80x80
clean block carries ~16x the content of one 20x20 tile, and stage-1's own
result (paired_alignment: R@1=76% for tile self-identity among 576) proves a
correctly-posed corruption-invariant contrastive objective recovers identity
signal that similarity-based affinity graphs miss entirely. If a single
dirty tile can point to its containing block well above chance, downstream
group assignment does not need every tile correct -- 16 independent noisy
votes per block concentrate quickly (see eval_block_group_assignment.py),
directly re-enabling the ALREADY VALIDATED per-16-group local solver instead
of building yet another new assembler.

Examples
--------

    python src/eval_block_identity.py --smoke
    python src/eval_block_identity.py --steps 1500 --bs 4 --tiles-per-image 192 --device cuda
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

from config import CKPT_DIR, FS, GRID, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags, train_val_split


MACRO = 4  # tiles per block edge (4x4 = 16 tiles per block, matching the macro_oracle finding)
BLOCKS_PER_SIDE = GRID // MACRO
NUM_BLOCKS = BLOCKS_PER_SIDE * BLOCKS_PER_SIDE
BLOCK_PX = MACRO * FS
if GRID % MACRO:
    raise RuntimeError("GRID must be divisible by MACRO")

_ROWS, _COLS = np.divmod(np.arange(NFRAG), GRID)
TILE_BLOCK_ID = ((_ROWS // MACRO) * BLOCKS_PER_SIDE + (_COLS // MACRO)).astype(np.int64)


def to_macro_blocks(image: np.ndarray, block_px: int = BLOCK_PX) -> np.ndarray:
    """(g*block_px, g*block_px, 3) -> (g*g, block_px, block_px, 3), row-major."""
    height, width = image.shape[:2]
    block_h, block_w = height // block_px, width // block_px
    reshaped = image.reshape(block_h, block_px, block_w, block_px, 3).transpose(0, 2, 1, 3, 4)
    return reshaped.reshape(block_h * block_w, block_px, block_px, 3)


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvStack(nn.Module):
    """Strided conv stem shared by both encoders; depth adapts to input size."""

    def __init__(self, in_size: int, embed_dim: int) -> None:
        super().__init__()
        base = max(24, embed_dim // 5)
        middle = max(32, embed_dim // 2)
        downsamples = max(1, round(math.log2(in_size / 3)))
        channels = [3] + [base] * max(0, downsamples - 2) + [middle, embed_dim]
        channels = channels[: downsamples + 1]
        if len(channels) < 2:
            channels = [3, embed_dim]
        layers: list[nn.Module] = []
        for index in range(len(channels) - 1):
            c_in, c_out = channels[index], channels[index + 1]
            layers += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_groups(c_out), c_out),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TileToBlockEncoder(nn.Module):
    """Maps one 20x20 DIRTY tile into the shared block-identity space."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.stem = _ConvStack(FS, embed_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(2 * self.stem.out_channels),
            nn.Linear(2 * self.stem.out_channels, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 4 or tuple(tiles.shape[1:]) != (3, FS, FS):
            raise ValueError(f"tiles must have shape (N,3,{FS},{FS}), got {tuple(tiles.shape)}")
        mean = tiles.mean(dim=(-3, -2, -1), keepdim=True)
        rms = (tiles - mean).square().mean(dim=(-3, -2, -1), keepdim=True)
        normalized = ((tiles - mean) / rms.add(1.0e-5).sqrt()).clamp(-5.0, 5.0)
        # Raw and exposure-normalized views go through the same shared stem as
        # two separate forward passes (the stem takes 3 input channels), then
        # their pooled stats are concatenated for the projection head.
        x = self.stem(tiles)
        x_norm = self.stem(normalized)
        flat = x.flatten(start_dim=2)
        flat_norm = x_norm.flatten(start_dim=2)
        stats = torch.cat((flat.mean(dim=-1), flat_norm.mean(dim=-1)), dim=-1)
        return F.normalize(self.project(stats), dim=-1)


class BlockEncoder(nn.Module):
    """Maps one 80x80 CLEAN macro-block into the shared identity space."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.stem = _ConvStack(BLOCK_PX, embed_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(self.stem.out_channels),
            nn.Linear(self.stem.out_channels, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, blocks: Tensor) -> Tensor:
        if blocks.ndim != 4 or tuple(blocks.shape[1:]) != (3, BLOCK_PX, BLOCK_PX):
            raise ValueError(f"blocks must have shape (N,3,{BLOCK_PX},{BLOCK_PX})")
        x = self.stem(blocks)
        flat = x.flatten(start_dim=2)
        stats = flat.mean(dim=-1)
        return F.normalize(self.project(stats), dim=-1)


class BlockIdentity(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.tile_encoder = TileToBlockEncoder(embed_dim)
        self.block_encoder = BlockEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def scale(self) -> Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def _to_tensor(images_uint8: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(images_uint8))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .to(device)
    )


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def rank_metrics(logits: Tensor, target: Tensor, prefix: str) -> dict[str, float]:
    true_score = logits.gather(1, target[:, None]).squeeze(1)
    rank = logits.gt(true_score[:, None]).sum(dim=1) + 1
    rank = rank.float()
    return {
        f"{prefix}_r1": float(rank.le(1).float().mean()),
        f"{prefix}_r5": float(rank.le(5).float().mean()),
        f"{prefix}_median_rank": float(rank.median()),
        f"{prefix}_mrr": float(rank.reciprocal().mean()),
        f"{prefix}_n": int(rank.numel()),
    }


def _synthetic_sample(name: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    clean = load(os.path.join(TRAIN_TGT, name))
    clean_tiles = to_frags(clean)
    dirty_tiles = distort_frags(clean_tiles, rng)
    clean_blocks = to_macro_blocks(clean)
    return dirty_tiles, clean_blocks


@torch.inference_mode()
def evaluate(
    model: BlockIdentity, names: list[str], *, device: torch.device, seed: int, tiles_per_image: int = NFRAG
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    same_image_ranks: list[Tensor] = []
    all_tile_embed: list[Tensor] = []
    all_block_embed: list[Tensor] = []
    all_targets: list[Tensor] = []
    offset = 0
    for index, name in enumerate(names):
        rng = np.random.default_rng(seed + index * 7919)
        dirty_tiles, clean_blocks = _synthetic_sample(name, rng)
        if tiles_per_image < NFRAG:
            picked = rng.choice(NFRAG, size=tiles_per_image, replace=False)
        else:
            picked = np.arange(NFRAG)
        dirty_t = _to_tensor(dirty_tiles[picked], device)
        block_t = _to_tensor(clean_blocks, device)
        target = torch.from_numpy(TILE_BLOCK_ID[picked]).to(device)
        with _autocast(device):
            tile_embed = model.tile_encoder(dirty_t).float()
            block_embed = model.block_encoder(block_t).float()
        logits = tile_embed @ block_embed.t() * model.scale()
        rank = logits.gt(logits.gather(1, target[:, None])).sum(dim=1) + 1
        same_image_ranks.append(rank.float())
        all_tile_embed.append(tile_embed)
        all_block_embed.append(block_embed)
        all_targets.append(target + offset)
        offset += NUM_BLOCKS
    if was_training:
        model.train()

    same_rank = torch.cat(same_image_ranks)
    result = {
        "same_image_r1": float(same_rank.le(1).float().mean()),
        "same_image_r5": float(same_rank.le(5).float().mean()),
        "same_image_median_rank": float(same_rank.median()),
        "same_image_mrr": float(same_rank.reciprocal().mean()),
        "same_image_n": int(same_rank.numel()),
    }
    pooled_tile = torch.cat(all_tile_embed)
    pooled_block = torch.cat(all_block_embed)
    pooled_target = torch.cat(all_targets)
    pooled_logits = pooled_tile @ pooled_block.t() * model.scale()
    pooled_rank = pooled_logits.gt(pooled_logits.gather(1, pooled_target[:, None])).sum(dim=1) + 1
    result.update(
        {
            "pooled_r1": float(pooled_rank.le(1).float().mean()),
            "pooled_r5": float(pooled_rank.le(5).float().mean()),
            "pooled_images": len(names),
            "pooled_candidates": int(pooled_block.shape[0]),
        }
    )
    return result


def _checkpoint(path: str, model: BlockIdentity, *, step: int, metrics: dict[str, Any]) -> None:
    torch.save(
        {"model": model.state_dict(), "embed_dim": model.tile_encoder.embed_dim, "step": int(step), "metrics": metrics},
        path,
    )


def smoke(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    """Data-free contract test: shapes, block-geometry consistency, gradients."""
    if NUM_BLOCKS != 36 or BLOCK_PX != 80:
        raise AssertionError(f"unexpected macro geometry: {NUM_BLOCKS} blocks of {BLOCK_PX}px")
    if TILE_BLOCK_ID.shape != (NFRAG,) or TILE_BLOCK_ID.min() != 0 or TILE_BLOCK_ID.max() != NUM_BLOCKS - 1:
        raise AssertionError("TILE_BLOCK_ID lookup is malformed")
    counts = np.bincount(TILE_BLOCK_ID, minlength=NUM_BLOCKS)
    if not np.all(counts == MACRO * MACRO):
        raise AssertionError("every macro-block must contain exactly MACRO*MACRO tiles")

    # to_macro_blocks must be the exact block-level analogue of imgio.to_frags:
    # tiling the full image directly should equal tiling each block's own tiles.
    rng = np.random.default_rng(11)
    fake_image = rng.integers(0, 256, size=(GRID * FS, GRID * FS, 3), dtype=np.uint8)
    blocks = to_macro_blocks(fake_image)
    tiles = to_frags(fake_image)
    for block_id in (0, 5, 35):
        member_tiles = np.flatnonzero(TILE_BLOCK_ID == block_id)
        block_row, block_col = divmod(block_id, BLOCKS_PER_SIDE)
        rebuilt = np.zeros((BLOCK_PX, BLOCK_PX, 3), dtype=np.uint8)
        for tile_id in member_tiles:
            local_row = _ROWS[tile_id] - block_row * MACRO
            local_col = _COLS[tile_id] - block_col * MACRO
            rebuilt[local_row * FS : (local_row + 1) * FS, local_col * FS : (local_col + 1) * FS] = tiles[tile_id]
        if not np.array_equal(rebuilt, blocks[block_id]):
            raise AssertionError(f"to_macro_blocks disagrees with to_frags tiling for block {block_id}")

    torch.manual_seed(4321)
    model = BlockIdentity(embed_dim=16)
    tile_input = torch.rand(20, 3, FS, FS, requires_grad=True)
    block_input = torch.rand(NUM_BLOCKS, 3, BLOCK_PX, BLOCK_PX, requires_grad=True)
    tile_embed = model.tile_encoder(tile_input)
    block_embed = model.block_encoder(block_input)
    if tile_embed.shape != (20, 16) or block_embed.shape != (NUM_BLOCKS, 16):
        raise AssertionError(f"unexpected embedding shapes {tile_embed.shape} {block_embed.shape}")
    if not torch.allclose(tile_embed.norm(dim=-1), torch.ones(20), atol=1.0e-5):
        raise AssertionError("tile embeddings are not L2-normalized")
    target = torch.from_numpy(TILE_BLOCK_ID[:20]).long()
    logits = tile_embed @ block_embed.t() * model.scale()
    loss = F.cross_entropy(logits, target)
    loss.backward()
    if tile_input.grad is None or not torch.isfinite(tile_input.grad).all():
        raise AssertionError("loss lost a finite gradient into tile_input")
    if block_input.grad is None or not torch.isfinite(block_input.grad).all():
        raise AssertionError("loss lost a finite gradient into block_input")
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
        raise AssertionError("model parameters received no gradient")

    perfect_logits = F.one_hot(target, NUM_BLOCKS).float() * 10.0 - 5.0
    metrics = rank_metrics(perfect_logits, target, "toy")
    if metrics["toy_r1"] < 0.999:
        raise AssertionError(f"perfect-similarity smoke rank check failed: {metrics}")
    return {"loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters()), **metrics}


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
    parser.add_argument("--tag", default="block_identity")
    parser.add_argument("--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", default=CKPT_DIR)
    parser.add_argument("--report", type=Path, default=Path("E:/pazzle_work/gates/block_identity_gate.json"))
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
        print(f"[block-identity smoke] device={device} {smoke(device)}", flush=True)
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

    model = BlockIdentity(embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} BlockIdentity params={sum(p.numel() for p in model.parameters()):,} "
        f"embed_dim={args.embed_dim} bs={args.bs} tiles/image={args.tiles_per_image} "
        f"macro={MACRO}x{MACRO} blocks={NUM_BLOCKS} block_px={BLOCK_PX} "
        "objective=CE(dirty tile embedding vs its true containing block among 36 same-image blocks)",
        flush=True,
    )
    print(
        "question: does one independent challenge-degradation instance still point a single tile "
        "to its containing 4x4 macro-block, among 36 same-photo candidates?",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 5)
    best_gate_metric = -float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        names = [train_names[int(i)] for i in rng.integers(0, len(train_names), size=args.bs)]
        dirty_batch: list[np.ndarray] = []
        block_batch: list[np.ndarray] = []
        target_batch: list[np.ndarray] = []
        for image_index, name in enumerate(names):
            image_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            dirty_tiles, clean_blocks = _synthetic_sample(name, image_rng)
            picked = image_rng.choice(NFRAG, size=args.tiles_per_image, replace=False)
            dirty_batch.append(dirty_tiles[picked])
            block_batch.append(clean_blocks)
            target_batch.append(TILE_BLOCK_ID[picked] + image_index * NUM_BLOCKS)
        dirty_tensor = _to_tensor(np.concatenate(dirty_batch), device)
        block_tensor = _to_tensor(np.concatenate(block_batch), device)
        target_tensor = torch.from_numpy(np.concatenate(target_batch)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            tile_embed = model.tile_encoder(dirty_tensor).float()
            block_embed = model.block_encoder(block_tensor).float()
            logits = tile_embed @ block_embed.t() * model.scale()
            loss = F.cross_entropy(logits, target_tensor)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                train_metrics = rank_metrics(logits, target_tensor, "train")
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"train_r1={train_metrics['train_r1']:.4f} train_r5={train_metrics['train_r5']:.4f} "
                f"tiles={dirty_tensor.shape[0]} blocks={block_tensor.shape[0]} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val_names[: args.eval_images], device=device, seed=args.seed + 9973)
            print(f"[SYN block-identity held-out] step={step} {metrics}", flush=True)
            last_path = os.path.join(args.ckpt_dir, f"{args.tag}_last.pt")
            _checkpoint(last_path, model, step=step, metrics=metrics)
            gate_metric = metrics["same_image_r1"]
            if gate_metric > best_gate_metric:
                best_gate_metric = gate_metric
                best_path = os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")
                _checkpoint(best_path, model, step=step, metrics=metrics)
                print(f"saved best same_image_r1={best_gate_metric:.4f}", flush=True)

    final_metrics = evaluate(model, val_names[: args.eval_images], device=device, seed=args.seed + 9973)
    passed = final_metrics["same_image_r1"] >= 0.15
    report = {
        "experiment": "stage_g1_tile_to_block_identity",
        "question": (
            "does one independent challenge-degradation instance still let a tile identify its "
            "containing 4x4 (16-tile) macro-block among 36 same-photo candidates?"
        ),
        "macro_geometry": {"macro": MACRO, "blocks_per_side": BLOCKS_PER_SIDE, "num_blocks": NUM_BLOCKS, "block_px": BLOCK_PX},
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "device": str(device)},
        "final_metrics": final_metrics,
        "gate": {
            "rule": "same_image_r1 >= 0.15 (chance 1/36=2.78%, ~5.4x)",
            "chance_r1": 1.0 / NUM_BLOCKS,
            "chance_r5": 5.0 / NUM_BLOCKS,
            "pass": bool(passed),
            "note": (
                "this per-tile gate is intentionally lenient: the decisive test is whether "
                "capacitated 16-per-block joint assignment over all 576 tiles recovers clean "
                "groups (see eval_block_group_assignment.py), since per-tile votes can be noisy "
                "and still yield accurate groups after aggregation."
            ),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> test capacitated group assignment" if passed else "FAILED -> close branch G"
    print(f"\n=== stage G1 gate {verdict} ===", flush=True)
    print(json.dumps(report["gate"], indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
