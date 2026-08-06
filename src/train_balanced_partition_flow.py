"""Train and gate discrete refinement of balanced 36x16 macro partitions."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from balanced_partition_flow import (
    BalancedPartitionRefiner,
    capacity_preserving_corruption,
    iterative_refine,
    refinement_loss,
    smoke_test,
)
from block_siamese import (
    BlockSiamese,
    balanced_spherical_kmeans,
    clustering_metrics,
)
from config import NFRAG, SEED, TRAIN_TGT, WORK_ROOT
from distort import distort_frags
from eval_block_identity import NUM_BLOCKS, TILE_BLOCK_ID
from imgio import load, to_frags, train_val_split
from train_block_siamese import load_checkpoint


class PuzzlePool:
    def __init__(self, names: list[str], *, max_images: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        if max_images and len(names) > max_images:
            chosen = rng.choice(len(names), size=max_images, replace=False)
            names = [names[int(index)] for index in chosen]
        self.names = names
        self.clean = np.stack(
            [to_frags(load(os.path.join(TRAIN_TGT, name))) for name in names]
        )
        self.rng = rng

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dirty_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        for _ in range(batch_size):
            image = self.clean[int(self.rng.integers(len(self.clean)))]
            dirty = distort_frags(image, self.rng)
            permutation = self.rng.permutation(NFRAG)
            dirty_rows.append(dirty[permutation])
            label_rows.append(TILE_BLOCK_ID[permutation])
        dirty_np = np.stack(dirty_rows)
        tiles = (
            torch.from_numpy(np.ascontiguousarray(dirty_np))
            .permute(0, 1, 4, 2, 3)
            .float()
            .div_(255.0)
            .to(device)
        )
        labels = torch.from_numpy(np.stack(label_rows)).long().to(device)
        return tiles, labels


def randomize_group_ids(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    output = torch.empty_like(labels)
    for image in range(labels.shape[0]):
        permutation = torch.randperm(
            NUM_BLOCKS, device=labels.device, generator=generator
        )
        output[image] = permutation[labels[image]]
    return output


@torch.no_grad()
def encode(encoder: BlockSiamese, tiles: torch.Tensor) -> torch.Tensor:
    batch, count = tiles.shape[:2]
    return encoder(tiles.flatten(0, 1)).float().reshape(batch, count, -1)


def cluster_metrics_for_labels(
    assignment: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    rows = [
        clustering_metrics(
            assignment[index].detach().cpu().numpy(),
            labels[index].detach().cpu().numpy(),
        )
        for index in range(assignment.shape[0])
    ]
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


@torch.inference_mode()
def evaluate(
    model: BalancedPartitionRefiner,
    encoder: BlockSiamese,
    sampler: PuzzlePool,
    *,
    images: int,
    device: torch.device,
    cluster_iterations: int,
    cluster_restarts: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    tiles, labels = sampler.sample(images, device)
    embeddings = encode(encoder, tiles)
    initial_rows: list[np.ndarray] = []
    for image in range(images):
        assignment, _ = balanced_spherical_kmeans(
            embeddings[image].cpu().numpy(),
            iterations=cluster_iterations,
            restarts=cluster_restarts,
            seed=seed + image * 7919,
        )
        initial_rows.append(assignment)
    initial = torch.from_numpy(np.stack(initial_rows)).long().to(device)
    stages = iterative_refine(model, embeddings, initial)
    initial_metrics = cluster_metrics_for_labels(initial, labels)
    stage_metrics = [
        cluster_metrics_for_labels(stage, labels) for stage in stages
    ]
    return {
        "initial": initial_metrics,
        "stages": stage_metrics,
        "final": stage_metrics[-1],
    }


def gate_report(
    result: dict[str, Any],
    *,
    checkpoint: Path,
    train_steps: int,
    heldout_images: int,
) -> dict[str, Any]:
    final = result["final"]
    initial = result["initial"]
    thresholds = {
        "purity": 0.35,
        "purity_improvement": 0.05,
        "perfect_blocks": 1.0,
        "near_perfect_blocks": 0.5,
    }
    observed = {
        "purity": final["purity"],
        "purity_improvement": final["purity"] - initial["purity"],
        "perfect_blocks": final["perfect_blocks"],
        "near_perfect_blocks": final["near_perfect_blocks"],
    }
    checks = {
        key: observed[key] >= threshold for key, threshold in thresholds.items()
    }
    return {
        "experiment": "balanced_partition_discrete_flow",
        "status": "pass" if all(checks.values()) else "fail",
        "best": result,
        "observed": observed,
        "thresholds": thresholds,
        "checks": checks,
        "checkpoint": str(checkpoint),
        "train_steps": train_steps,
        "heldout_images": heldout_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-images", type=int, default=96)
    parser.add_argument("--val-images", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--cluster-iterations", type=int, default=12)
    parser.add_argument("--cluster-restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--encoder",
        default=str(Path(WORK_ROOT) / "ckpt" / "block_siamese_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(WORK_ROOT) / "balanced_partition_flow" / "best.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "balanced_partition_flow_gate.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.smoke:
        print(json.dumps(smoke_test(device), indent=2))
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_names, val_names = train_val_split()
    train_pool = PuzzlePool(
        train_names, max_images=args.cache_images, seed=args.seed
    )
    val_pool = PuzzlePool(
        val_names[: args.val_images],
        max_images=args.val_images,
        seed=args.seed + 40_000,
    )
    encoder, encoder_payload = load_checkpoint(args.encoder, device)
    encoder.requires_grad_(False).eval()
    model = BalancedPartitionRefiner(
        embed_dim=encoder.embed_dim,
        d_model=args.d_model,
        groups=NUM_BLOCKS,
        capacity=NFRAG // NUM_BLOCKS,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    if args.eval_only:
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        result = evaluate(
            model,
            encoder,
            val_pool,
            images=args.val_images,
            device=device,
            cluster_iterations=args.cluster_iterations,
            cluster_restarts=args.cluster_restarts,
            seed=args.seed + 80_000,
        )
        report = gate_report(
            result,
            checkpoint=args.output,
            train_steps=int(payload.get("step", 0)),
            heldout_images=args.val_images,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_purity = -1.0
    best_result: dict[str, Any] = {}
    rolling: dict[str, float] = {}
    started = time.time()
    for step in range(1, args.steps + 1):
        tiles, true_groups = train_pool.sample(args.batch_size, device)
        with torch.no_grad():
            embeddings = encode(encoder, tiles)
        target = randomize_group_ids(true_groups, generator)
        requested_noise = torch.empty(
            args.batch_size, device=device
        ).uniform_(0.45, 0.85, generator=generator)
        noisy, actual_noise = capacity_preserving_corruption(
            target, requested_noise, generator=generator
        )
        logits = model(embeddings, noisy, actual_noise)
        loss, parts = refinement_loss(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        denoise_accuracy = float(logits.argmax(dim=-1).eq(target).float().mean())
        for key, value in {
            **parts,
            "input_accuracy": float(noisy.eq(target).float().mean()),
            "denoise_accuracy": denoise_accuracy,
        }.items():
            rolling[key] = rolling.get(key, 0.0) + value
        if step % args.log_every == 0:
            row = {key: value / args.log_every for key, value in rolling.items()}
            row.update(
                {
                    "step": step,
                    "grad_norm": float(grad_norm),
                    "elapsed": time.time() - started,
                }
            )
            print(json.dumps(row), flush=True)
            rolling.clear()
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(
                model,
                encoder,
                val_pool,
                images=args.val_images,
                device=device,
                cluster_iterations=args.cluster_iterations,
                cluster_restarts=args.cluster_restarts,
                seed=args.seed + 80_000,
            )
            print(json.dumps({"step": step, "validation": result}), flush=True)
            if result["final"]["purity"] > best_purity:
                best_purity = result["final"]["purity"]
                best_result = result
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_kwargs": model.model_kwargs,
                        "step": step,
                        "metrics": result,
                        "encoder_step": encoder_payload.get("step"),
                    },
                    args.output,
                )
    report = gate_report(
        best_result,
        checkpoint=args.output,
        train_steps=args.steps,
        heldout_images=args.val_images,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report, "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
