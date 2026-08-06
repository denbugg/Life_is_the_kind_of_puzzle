"""Train/evaluate a direct Siamese same-4x4-block embedding.

This is branch G4: it fixes the train/inference mismatch of the successful G1
information gate.  G1 pulled dirty tiles toward unavailable clean 80x80 block
prototypes; G4 directly pulls *different dirty sibling tiles* together and
pushes dirty tiles from other blocks of the same image apart.

Examples
--------

    python src/train_block_siamese.py --smoke
    python src/train_block_siamese.py --steps 1500 --device cuda
    python src/train_block_siamese.py --eval-only --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from block_siamese import (
    BlockSiamese,
    balanced_spherical_kmeans,
    clustering_metrics,
    same_block_retrieval_metrics,
    sibling_supervised_contrastive_loss,
    smoke,
)
from config import CKPT_DIR, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from eval_block_identity import (
    BLOCKS_PER_SIDE,
    MACRO,
    NUM_BLOCKS,
    TILE_BLOCK_ID,
    BlockIdentity,
)
from imgio import load, to_frags, train_val_split


def _autocast(device: torch.device, enabled: bool = False):
    return (
        torch.autocast("cuda", dtype=torch.float16)
        if enabled and device.type == "cuda"
        else nullcontext()
    )


def _tiles_tensor(tiles: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .to(device)
    )


def _balanced_tile_sample(rng: np.random.Generator, tiles_per_block: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    for block in range(NUM_BLOCKS):
        members = np.flatnonzero(TILE_BLOCK_ID == block)
        selected.append(rng.choice(members, size=tiles_per_block, replace=False))
    picked = np.concatenate(selected)
    return picked[rng.permutation(len(picked))]


def _two_dirty_views(
    name: str,
    picked: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = to_frags(load(os.path.join(TRAIN_TGT, name)))[picked]
    view_a = distort_frags(clean, np.random.default_rng(rng.integers(0, 2**31 - 1)))
    view_b = distort_frags(clean, np.random.default_rng(rng.integers(0, 2**31 - 1)))
    return np.stack((view_a, view_b), axis=0)


def _load_g1_initialization(model: BlockSiamese, path: str) -> dict[str, Any] | None:
    if not path or path.lower() == "none":
        return None
    if not os.path.isfile(path):
        print(f"initialization checkpoint not found, training from scratch: {path}", flush=True)
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embed_dim = int(payload.get("embed_dim", model.embed_dim))
    if embed_dim != model.embed_dim:
        raise ValueError(
            f"G1 checkpoint embed_dim={embed_dim} does not match requested embed_dim={model.embed_dim}"
        )
    g1 = BlockIdentity(embed_dim=embed_dim)
    g1.load_state_dict(payload["model"], strict=True)
    model.encoder.load_state_dict(g1.tile_encoder.state_dict(), strict=True)
    return {
        "path": path,
        "step": payload.get("step"),
        "metrics": payload.get("metrics"),
    }


def _save_checkpoint(
    path: str,
    model: BlockSiamese,
    *,
    step: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "embed_dim": model.embed_dim,
            "step": int(step),
            "metrics": metrics,
            "config": config,
            "objective": (
                "direct dirty-dirty sibling supervised contrastive; "
                "same-tile cross-view pairs excluded"
            ),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[BlockSiamese, dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BlockSiamese(embed_dim=int(payload["embed_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def _embed_dirty_image(
    model: BlockSiamese,
    name: str,
    *,
    device: torch.device,
    seed: int,
    batch_size: int = NFRAG,
    amp: bool = False,
) -> np.ndarray:
    clean_tiles = to_frags(load(os.path.join(TRAIN_TGT, name)))
    dirty = distort_frags(clean_tiles, np.random.default_rng(seed))
    embeddings: list[Tensor] = []
    for start in range(0, NFRAG, batch_size):
        tiles = _tiles_tensor(dirty[start : start + batch_size], device)
        with _autocast(device, amp):
            embeddings.append(model(tiles).float().cpu())
    return torch.cat(embeddings).numpy()


@torch.inference_mode()
def evaluate(
    model: BlockSiamese,
    names: list[str],
    *,
    device: torch.device,
    seed: int,
    cluster_iterations: int,
    cluster_restarts: int,
    amp: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    per_image: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        image_seed = seed + index * 7919
        embeddings = _embed_dirty_image(
            model, name, device=device, seed=image_seed, amp=amp
        )
        retrieval = same_block_retrieval_metrics(embeddings).as_dict()
        assignment, objective = balanced_spherical_kmeans(
            embeddings,
            iterations=cluster_iterations,
            restarts=cluster_restarts,
            seed=image_seed + 17,
        )
        cluster = clustering_metrics(assignment)
        row = {
            "image": name,
            **retrieval,
            **cluster,
            "cluster_objective": objective,
        }
        per_image.append(row)
        print(
            f"  {name}: top1={row['top1_same_block']:.3f} "
            f"recip={row['reciprocal_precision']:.3f}/{row['reciprocal_edges']} "
            f"purity={row['purity']:.3f} perfect={row['perfect_blocks']}/{NUM_BLOCKS} "
            f"near={row['near_perfect_blocks']}/{NUM_BLOCKS}",
            flush=True,
        )
    if was_training:
        model.train()

    scalar_keys = [
        "top1_same_block",
        "precision_at_5",
        "recall_at_5",
        "reciprocal_precision",
        "reciprocal_edges",
        "purity",
        "perfect_blocks",
        "near_perfect_blocks",
        "mean_best_overlap",
        "min_best_overlap",
    ]
    mean = {
        f"mean_{key}": float(np.mean([float(row[key]) for row in per_image]))
        for key in scalar_keys
    }
    return {"per_image": per_image, **mean, "images": len(per_image)}


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-images", type=int, default=2)
    parser.add_argument("--tiles-per-block", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-images", type=int, default=8)
    parser.add_argument("--cluster-iterations", type=int, default=20)
    parser.add_argument("--cluster-restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="block_siamese")
    parser.add_argument("--ckpt-dir", default=CKPT_DIR)
    parser.add_argument(
        "--init",
        default=os.path.join(CKPT_DIR, "block_identity_best.pt"),
        help="G1 block-identity checkpoint, or 'none'",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint for --eval-only (default: <ckpt-dir>/<tag>_best.pt)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("E:/pazzle_work/gates/block_siamese_gate.json"),
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=None,
        help="optional JSON output for --eval-only",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "use fp16 autocast; off by default because this contrastive objective "
            "can overflow fp16 gradients on the local RTX 2070"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return args
    if args.steps < 1 and not args.eval_only:
        parser.error("--steps must be positive")
    if args.batch_images < 1 or args.eval_images < 1:
        parser.error("--batch-images and --eval-images must be positive")
    if not 2 <= args.tiles_per_block <= MACRO * MACRO:
        parser.error(f"--tiles-per-block must be in [2,{MACRO * MACRO}]")
    if args.eval_every < 1 or args.cluster_iterations < 1 or args.cluster_restarts < 1:
        parser.error("evaluation and clustering counts must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(f"[block-siamese smoke] {smoke(args.seed)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_names, val_names = train_val_split()
    if len(val_names) < args.eval_images:
        raise ValueError(f"--eval-images exceeds held-out pool ({len(val_names)})")
    eval_names = val_names[: args.eval_images]
    checkpoint = args.checkpoint or os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")

    if args.eval_only:
        model, payload = load_checkpoint(checkpoint, device)
        print(
            f"eval-only checkpoint={checkpoint} step={payload.get('step')} device={device}",
            flush=True,
        )
        metrics = evaluate(
            model,
            eval_names,
            device=device,
            seed=args.seed + 9973,
            cluster_iterations=args.cluster_iterations,
            cluster_restarts=args.cluster_restarts,
            amp=args.amp,
        )
        if args.eval_report is not None:
            args.eval_report.parent.mkdir(parents=True, exist_ok=True)
            args.eval_report.write_text(
                json.dumps(
                    {
                        "experiment": "stage_g4_extended_eval",
                        "checkpoint": checkpoint,
                        "checkpoint_step": payload.get("step"),
                        "config": _jsonable_config(args),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"eval report saved to {args.eval_report}", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return

    model = BlockSiamese(embed_dim=args.embed_dim).to(device)
    initialization = _load_g1_initialization(model, args.init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )
    config = _jsonable_config(args)
    print(
        f"device={device} params={sum(parameter.numel() for parameter in model.parameters()):,} "
        f"embed_dim={args.embed_dim} images/step={args.batch_images} "
        f"tiles/block={args.tiles_per_block} views=2 "
        f"amp={args.amp} "
        f"init={initialization or 'scratch'}",
        flush=True,
    )
    print(
        "objective=direct dirty-dirty sibling SupCon within each image; "
        "same-source-tile cross-view pairs are excluded",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 41)
    best_purity = -np.inf
    best_metrics: dict[str, Any] | None = None
    started = time.time()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image_losses: list[Tensor] = []
        for _ in range(args.batch_images):
            name = train_names[int(rng.integers(0, len(train_names)))]
            image_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            picked = _balanced_tile_sample(image_rng, args.tiles_per_block)
            views = _two_dirty_views(name, picked, image_rng)
            tensor = _tiles_tensor(views.reshape(-1, *views.shape[2:]), device)
            with _autocast(device, args.amp):
                embeddings = model(tensor).reshape(2, len(picked), -1)
                image_loss = sibling_supervised_contrastive_loss(
                    embeddings,
                    torch.from_numpy(TILE_BLOCK_ID[picked]).to(device),
                    torch.from_numpy(picked.astype(np.int64)).to(device),
                    scale=model.scale(),
                )
            image_losses.append(image_loss)
        loss = torch.stack(image_losses).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_applied = scaler.get_scale() >= old_scale
        if optimizer_step_applied:
            scheduler.step()

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"scale={float(model.scale().detach()):.2f} grad={float(gradient_norm):.3f} "
                f"update={int(optimizer_step_applied)} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            print(f"[held-out block-siamese] step={step}", flush=True)
            metrics = evaluate(
                model,
                eval_names,
                device=device,
                seed=args.seed + 9973,
                cluster_iterations=args.cluster_iterations,
                cluster_restarts=args.cluster_restarts,
                amp=args.amp,
            )
            print(
                f"[mean] top1={metrics['mean_top1_same_block']:.4f} "
                f"recip={metrics['mean_reciprocal_precision']:.4f} "
                f"purity={metrics['mean_purity']:.4f} "
                f"perfect={metrics['mean_perfect_blocks']:.2f} "
                f"near={metrics['mean_near_perfect_blocks']:.2f}",
                flush=True,
            )
            last_path = os.path.join(args.ckpt_dir, f"{args.tag}_last.pt")
            _save_checkpoint(last_path, model, step=step, metrics=metrics, config=config)
            if metrics["mean_purity"] > best_purity:
                best_purity = metrics["mean_purity"]
                best_metrics = metrics
                _save_checkpoint(checkpoint, model, step=step, metrics=metrics, config=config)
                print(f"saved best mean_purity={best_purity:.4f} -> {checkpoint}", flush=True)

    if best_metrics is None:
        raise RuntimeError("training finished without an evaluation")
    passed = (
        best_metrics["mean_top1_same_block"] >= 0.40
        and best_metrics["mean_purity"] >= 0.35
        and best_metrics["mean_perfect_blocks"] >= 1.0
    )
    report = {
        "experiment": "stage_g4_direct_dirty_dirty_block_siamese",
        "question": (
            "does directly optimizing dirty sibling similarity recover clean-reference-free "
            "balanced 4x4 source groups strongly enough for the validated local solver?"
        ),
        "initialization": initialization,
        "config": config,
        "best_metrics": best_metrics,
        "baselines": {
            "g2_proxy_objective_mean_purity": 0.2456597222222222,
            "g3_proxy_objective_top1_same_block": 0.222,
            "random_embedding_matched_purity": 0.1388888888888889,
        },
        "gate": {
            "rule": (
                "mean top1 same-block >= 0.40 AND mean purity >= 0.35 "
                "AND at least 1 perfect 16-tile block per image on average"
            ),
            "pass": bool(passed),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> integrate recovered groups with macro local solver" if passed else (
        "FAILED -> direct same-block Siamese improved signal but does not yet justify end-to-end rollout"
    )
    print(f"\n=== stage G4 gate {verdict} ===", flush=True)
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
