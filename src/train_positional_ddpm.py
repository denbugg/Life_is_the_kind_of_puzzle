"""Train and gate the full 24x24 fixed-orientation Positional DDPM solver."""
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
from torch import Tensor

from config import CKPT_DIR, FS, GRID, SEED, TRAIN_TGT, WORK_ROOT
from distort import distort_frags
from imgio import load, to_frags, train_val_split
from positional_ddpm import (
    PositionalDDPM,
    arrangement_metrics,
    ddim_sample,
    diffusion_loss,
    grid_coordinates,
    hungarian_slots,
    load_paired_dirty_encoder,
    smoke_test,
)
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores


def tiles_tensor(tiles: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device)


class FullBoardSampler:
    def __init__(
        self,
        names: list[str],
        *,
        max_images: int,
        seed: int,
        precompute: bool = False,
        distortion_variants: int = 1,
    ) -> None:
        rng = np.random.default_rng(seed)
        if max_images > 0 and len(names) > max_images:
            chosen = rng.choice(len(names), size=max_images, replace=False)
            names = [names[int(index)] for index in chosen]
        clean = np.stack([to_frags(load(os.path.join(TRAIN_TGT, name))) for name in names])
        self.rng = rng
        self.precomputed: np.ndarray | None = None
        if precompute:
            variants: list[np.ndarray] = []
            for _ in range(max(1, int(distortion_variants))):
                variants.append(np.stack([distort_frags(board, self.rng) for board in clean]))
            self.precomputed = np.concatenate(variants, axis=0)
            self.clean = None
        else:
            self.clean = clean

    def sample_numpy(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        dirty = np.empty((batch_size, GRID * GRID, FS, FS, 3), dtype=np.uint8)
        permutations = np.empty((batch_size, GRID * GRID), dtype=np.int64)
        for batch_index in range(batch_size):
            if self.precomputed is not None:
                degraded = self.precomputed[int(self.rng.integers(len(self.precomputed)))]
            else:
                assert self.clean is not None
                clean = self.clean[int(self.rng.integers(len(self.clean)))]
                degraded = distort_frags(clean, self.rng)
            permutation = self.rng.permutation(GRID * GRID)
            dirty[batch_index] = degraded[permutation]
            permutations[batch_index] = permutation
        return dirty, permutations

    def sample(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        tiles, slots = self.sample_numpy(batch_size)
        true_slots = torch.from_numpy(slots).to(device)
        coordinates = grid_coordinates(GRID, device=device)[true_slots]
        return tiles_tensor(tiles, device), coordinates, true_slots


@torch.inference_mode()
def evaluate(
    model: PositionalDDPM,
    suite: tuple[np.ndarray, np.ndarray],
    *,
    device: torch.device,
    sample_steps: int,
    seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    tiles_np, true_slots_np = suite
    predictions: list[Tensor] = []
    edge_r1_values: list[float] = []
    edge_solver_placement: list[float] = []
    edge_solver_neighbor: list[float] = []
    coordinate_sse = 0.0
    coordinate_n = 0
    for index in range(len(tiles_np)):
        tiles = tiles_tensor(tiles_np[index:index + 1], device)
        predicted_coordinates = ddim_sample(model, tiles, sample_steps=sample_steps, seed=seed + index)
        true_slots = torch.from_numpy(true_slots_np[index:index + 1]).to(device)
        target = grid_coordinates(GRID, device=device)[true_slots]
        tile_features = model.encode_tiles(tiles)
        edge_scores = model.directional_edge_scores(tile_features)[0]
        inverse = torch.empty_like(true_slots)
        inverse.scatter_(1, true_slots, torch.arange(GRID * GRID, device=device)[None])
        row = torch.div(true_slots, GRID, rounding_mode="floor")
        col = true_slots.remainder(GRID)
        valid_by_direction = (row.gt(0), row.lt(GRID - 1), col.gt(0), col.lt(GRID - 1))
        deltas = (-GRID, GRID, -1, 1)
        correct = 0
        total = 0
        for direction, (valid, delta) in enumerate(zip(valid_by_direction, deltas)):
            target_tile = inverse.gather(1, (true_slots + delta).clamp(0, GRID * GRID - 1))
            correct += int(edge_scores[direction][valid[0]].argmax(-1).eq(target_tile[0][valid[0]]).sum())
            total += int(valid.sum())
        edge_r1_values.append(correct / total)
        probability = edge_scores.float().softmax(dim=-1).cpu().numpy()
        right = 0.5 * (probability[3] + probability[2].T)
        down = 0.5 * (probability[1] + probability[0].T)
        np.fill_diagonal(right, 0.0)
        np.fill_diagonal(down, 0.0)
        edge_placement, _ = solve_buddies_from_scores(right, down, max_edges=384, repair_passes=0)
        truth_placement = np.argsort(true_slots_np[index])
        edge_solver_placement.append(placement_accuracy(edge_placement, truth_placement)[0])
        edge_solver_neighbor.append(neighbour_accuracy(edge_placement, truth_placement)[0])
        coordinate_sse += float((predicted_coordinates - target).square().sum())
        coordinate_n += predicted_coordinates.numel()
        predictions.append(hungarian_slots(predicted_coordinates, GRID).cpu())
    predicted = torch.cat(predictions)
    truth = torch.from_numpy(true_slots_np)
    metrics = arrangement_metrics(predicted, truth, GRID)
    metrics["coordinate_rmse"] = float((coordinate_sse / coordinate_n) ** 0.5)
    metrics["puzzles"] = int(len(tiles_np))
    metrics["edge_r1"] = float(np.mean(edge_r1_values))
    metrics["edge_solver_placement"] = float(np.mean(edge_solver_placement))
    metrics["edge_solver_neighbor"] = float(np.mean(edge_solver_neighbor))
    if was_training:
        model.train()
    return metrics


def save_checkpoint(path: Path, model: PositionalDDPM, optimizer: torch.optim.Optimizer, step: int, metrics: dict[str, Any], args: argparse.Namespace) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "model_args": {
                "side": GRID,
                "tile_dim": 128,
                "d_model": args.d_model,
                "layers": args.layers,
                "heads": args.heads,
                "diffusion_steps": args.diffusion_steps,
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
    train_sampler = FullBoardSampler(
        train_names,
        max_images=args.cache_images,
        seed=args.seed,
        precompute=not args.overfit,
        distortion_variants=args.distortion_variants,
    )
    validation_sampler = FullBoardSampler(
        val_names[:args.val_images],
        max_images=args.val_images,
        seed=args.seed + 100_003,
        precompute=True,
    )
    validation = validation_sampler.sample_numpy(args.val_images)
    fixed_training = train_sampler.sample_numpy(1) if args.overfit else None

    model = PositionalDDPM(
        side=GRID,
        tile_dim=128,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    initialization = None
    if args.init_encoder and Path(args.init_encoder).exists():
        initialization = load_paired_dirty_encoder(model, args.init_encoder)
        print(json.dumps({"encoder_initialization": initialization}), flush=True)
    if args.freeze_encoder:
        for parameter in model.tile_backbone.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "overfit" if args.overfit else "train"
    best_path = output_dir / f"positional_ddpm_{tag}_best.pt"
    latest_path = output_dir / f"positional_ddpm_{tag}_latest.pt"
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    rolling: dict[str, float] = {}
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        if fixed_training is None:
            tiles, coordinates, _ = train_sampler.sample(args.batch_size, device)
        else:
            fixed_tiles, fixed_slots = fixed_training
            # Repeat the same board so each optimizer step observes several
            # independent timesteps/noise draws.  With batch=1, a single rare
            # low-noise timestep caused violent overfit regressions.
            tiles = tiles_tensor(fixed_tiles, device).expand(args.batch_size, -1, -1, -1, -1)
            true_slots = torch.from_numpy(fixed_slots).to(device).expand(args.batch_size, -1)
            coordinates = grid_coordinates(GRID, device=device)[true_slots]
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with autocast:
            loss, parts = diffusion_loss(
                model,
                tiles,
                coordinates,
                grid_weight=args.grid_weight,
                edge_weight=args.edge_weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {parts}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        for key, value in parts.items():
            rolling[key] = rolling.get(key, 0.0) + value

        if step % args.log_every == 0:
            log = {key: value / args.log_every for key, value in rolling.items()}
            log.update({"step": step, "grad_norm": float(grad_norm), "elapsed_sec": round(time.time() - start_time, 1)})
            if device.type == "cuda":
                log["max_cuda_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            print(json.dumps(log), flush=True)
            rolling.clear()

        if step % args.eval_every == 0 or step == args.steps:
            suite = fixed_training if args.overfit else validation
            metrics = evaluate(
                model,
                suite,
                device=device,
                sample_steps=args.sample_steps,
                seed=args.seed + 700_001,
            )
            print(json.dumps({"step": step, "evaluation": metrics, "overfit": args.overfit}), flush=True)
            save_checkpoint(latest_path, model, optimizer, step, metrics, args)
            selection_score = (
                metrics.get("edge_solver_neighbor", metrics["neighbor_accuracy"])
                if args.edge_weight > 0.0
                else metrics["neighbor_accuracy"]
            )
            if selection_score > best_score:
                best_score = selection_score
                best_metrics = metrics
                save_checkpoint(best_path, model, optimizer, step, metrics, args)

    report = {
        "experiment": "full_board_positional_ddpm_fixed_orientation",
        "overfit": args.overfit,
        "best_checkpoint": str(best_path),
        "best_metrics": best_metrics,
        "initialization": initialization,
        "args": vars(args),
        "elapsed_sec": round(time.time() - start_time, 1),
    }
    report_path = output_dir / f"positional_ddpm_{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--diffusion-steps", type=int, default=300)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grid-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cache-images", type=int, default=192)
    parser.add_argument("--distortion-variants", type=int, default=1)
    parser.add_argument("--val-images", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=str(Path(WORK_ROOT) / "positional_ddpm"))
    parser.add_argument("--init-encoder", default=str(Path(CKPT_DIR) / "paired_alignment_best.pt"))
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke_test(torch.device(args.device))), flush=True)
        return
    train(args)


if __name__ == "__main__":
    main()
