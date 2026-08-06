"""Train and gate the 4x4 relative-coordinate flow puzzle solver.

Examples:
    python src/train_relative_flow.py --smoke
    python src/train_relative_flow.py --steps 1600 --batch-size 16 --device cuda
"""
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
from torch import Tensor

from config import CKPT_DIR, FS, GRID, SEED, TRAIN_TGT, WORK_ROOT
from distort import distort_frags
from imgio import load, to_frags, train_val_split
from relative_flow import (
    RelativeCoordinateFlow,
    arrangement_metrics,
    flow_matching_loss,
    grid_coordinates,
    hungarian_slots,
    integrate_flow,
    load_paired_dirty_encoder,
    smoke_test,
)


def _tiles_tensor(tiles: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 1, 4, 2, 3)
        .float()
        .div_(255.0)
        .to(device)
    )


class BlockSampler:
    """In-memory clean images with fresh independent tile degradation per sample."""

    def __init__(self, names: list[str], *, side: int, max_images: int, seed: int) -> None:
        if not names:
            raise ValueError("empty image list")
        self.side = int(side)
        self.rng = np.random.default_rng(seed)
        if max_images > 0 and len(names) > max_images:
            indices = self.rng.choice(len(names), size=max_images, replace=False)
            names = [names[int(index)] for index in indices]
        self.names = list(names)
        self.clean = np.stack(
            [to_frags(load(os.path.join(TRAIN_TGT, name))).reshape(GRID, GRID, FS, FS, 3) for name in names]
        )

    def sample_numpy(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = self.side * self.side
        blocks = np.empty((batch_size, count, FS, FS, 3), dtype=np.uint8)
        target_slots = np.empty((batch_size, count), dtype=np.int64)
        for batch_index in range(batch_size):
            image_index = int(self.rng.integers(len(self.clean)))
            row = int(self.rng.integers(GRID - self.side + 1))
            col = int(self.rng.integers(GRID - self.side + 1))
            clean_block = self.clean[image_index, row : row + self.side, col : col + self.side]
            blocks[batch_index] = clean_block.reshape(count, FS, FS, 3)

        dirty = distort_frags(blocks.reshape(-1, FS, FS, 3), self.rng).reshape(blocks.shape)
        shuffled = np.empty_like(dirty)
        for batch_index in range(batch_size):
            permutation = self.rng.permutation(count)
            shuffled[batch_index] = dirty[batch_index, permutation]
            target_slots[batch_index] = permutation
        coordinates = (
            grid_coordinates(self.side).numpy()[target_slots]
        )
        return shuffled, coordinates.astype(np.float32), target_slots

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        tiles, coordinates, slots = self.sample_numpy(batch_size)
        return (
            _tiles_tensor(tiles, device),
            torch.from_numpy(coordinates).to(device),
            torch.from_numpy(slots).to(device),
        )


def make_validation_suite(
    names: list[str],
    *,
    side: int,
    images: int,
    blocks_per_image: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen held-out blocks: new images, degradation, crops, and shuffles."""
    sampler = BlockSampler(
        names[:images], side=side, max_images=images, seed=seed
    )
    return sampler.sample_numpy(len(sampler.names) * blocks_per_image)


@torch.inference_mode()
def evaluate(
    model: RelativeCoordinateFlow,
    suite: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    device: torch.device,
    integration_steps: int,
    seed: int,
    batch_size: int = 32,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    tiles_np, target_coordinates_np, true_slots_np = suite
    predicted_parts: list[Tensor] = []
    coordinate_sse = 0.0
    coordinate_n = 0
    for start in range(0, len(tiles_np), batch_size):
        stop = min(start + batch_size, len(tiles_np))
        tiles = _tiles_tensor(tiles_np[start:stop], device)
        predicted_coordinates = integrate_flow(
            model, tiles, steps=integration_steps, seed=seed + start
        )
        target_coordinates = torch.from_numpy(target_coordinates_np[start:stop]).to(device)
        coordinate_sse += float((predicted_coordinates - target_coordinates).square().sum())
        coordinate_n += predicted_coordinates.numel()
        predicted_parts.append(hungarian_slots(predicted_coordinates, model.side).cpu())
    predicted = torch.cat(predicted_parts)
    truth = torch.from_numpy(true_slots_np)
    metrics = arrangement_metrics(predicted, truth, model.side)
    metrics["coordinate_rmse"] = float((coordinate_sse / coordinate_n) ** 0.5)
    metrics["puzzles"] = int(len(tiles_np))
    if was_training:
        model.train()
    return metrics


def _save_checkpoint(
    path: Path,
    model: RelativeCoordinateFlow,
    *,
    step: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "step": int(step),
            "metrics": metrics,
            "model_args": {
                "side": args.side,
                "d_model": args.d_model,
                "layers": args.layers,
                "heads": args.heads,
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_names, val_names = train_val_split()
    sampler = BlockSampler(
        train_names,
        side=args.side,
        max_images=args.cache_images,
        seed=args.seed,
    )
    validation = make_validation_suite(
        val_names,
        side=args.side,
        images=args.val_images,
        blocks_per_image=args.val_blocks_per_image,
        seed=args.seed + 100_003,
    )
    model = RelativeCoordinateFlow(
        side=args.side,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    initialization: dict[str, Any] | None = None
    if args.init_encoder and Path(args.init_encoder).exists():
        initialization = load_paired_dirty_encoder(model, args.init_encoder)
        print(json.dumps({"encoder_initialization": initialization}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=args.lr * 0.1
    )
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "relative_flow_4x4_best.pt"
    best_placement = -1.0
    best_metrics: dict[str, float] = {}
    start_time = time.time()

    baseline = evaluate(
        model,
        validation,
        device=device,
        integration_steps=args.integration_steps,
        seed=args.seed + 700_001,
    )
    print(json.dumps({"step": 0, "validation": baseline}), flush=True)
    rolling: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        model.train()
        tiles, coordinates, _ = sampler.sample(args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = flow_matching_loss(
            model,
            tiles,
            coordinates,
            pair_weight=args.pair_weight,
            grid_weight=args.grid_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {parts}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        for key, value in parts.items():
            rolling[key] = rolling.get(key, 0.0) + value

        if step % args.log_every == 0:
            log = {key: value / args.log_every for key, value in rolling.items()}
            log.update(
                {
                    "step": step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "elapsed_sec": round(time.time() - start_time, 1),
                }
            )
            print(json.dumps(log), flush=True)
            rolling.clear()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                validation,
                device=device,
                integration_steps=args.integration_steps,
                seed=args.seed + 700_001,
            )
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            if metrics["placement_accuracy"] > best_placement:
                best_placement = metrics["placement_accuracy"]
                best_metrics = metrics
                _save_checkpoint(best_path, model, step=step, metrics=metrics, args=args)

    gate = {
        "experiment": "relative_coordinate_flow_4x4",
        "status": (
            "pass"
            if best_metrics.get("placement_accuracy", 0.0) >= args.placement_gate
            and best_metrics.get("neighbor_accuracy", 0.0) >= args.neighbor_gate
            else "fail"
        ),
        "thresholds": {
            "placement_accuracy": args.placement_gate,
            "neighbor_accuracy": args.neighbor_gate,
        },
        "baseline": baseline,
        "best": best_metrics,
        "checkpoint": str(best_path),
        "initialization": initialization,
        "train_steps": args.steps,
        "train_cached_images": len(sampler.names),
        "heldout_images": min(args.val_images, len(val_names)),
    }
    gate_dir = Path(WORK_ROOT) / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    gate_path = gate_dir / "relative_flow_4x4_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "gate_path": str(gate_path)}), flush=True)
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--grid-weight", type=float, default=0.25)
    parser.add_argument("--integration-steps", type=int, default=20)
    parser.add_argument("--cache-images", type=int, default=128)
    parser.add_argument("--val-images", type=int, default=16)
    parser.add_argument("--val-blocks-per-image", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--placement-gate", type=float, default=0.85)
    parser.add_argument("--neighbor-gate", type=float, default=0.90)
    parser.add_argument(
        "--init-encoder",
        default=str(Path(CKPT_DIR) / "paired_alignment_best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(WORK_ROOT) / "relative_flow"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.smoke:
        print(json.dumps(smoke_test(torch.device(parsed.device)), indent=2))
    else:
        train(parsed)
