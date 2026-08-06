"""Train stochastic boundary hypotheses around the frozen MatchDenoiser."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from config import FS, SEED, TRAIN_TGT, WORK_ROOT
from distort import distort_frags
from imgio import load, to_frags, train_val_split
from match_preprocess import load_match_denoiser
from posterior_edge import PosteriorEdgeRestorer, best_of_k_edge_loss, boundary_pixels, smoke_test


class TilePool:
    def __init__(self, names: list[str], *, images: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        if images < len(names):
            indices = rng.choice(len(names), size=images, replace=False)
            names = [names[int(index)] for index in indices]
        self.clean = np.concatenate(
            [to_frags(load(os.path.join(TRAIN_TGT, name))) for name in names],
            axis=0,
        )
        self.rng = rng

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.rng.integers(0, len(self.clean), size=batch_size)
        clean = self.clean[indices]
        dirty = distort_frags(clean, self.rng)
        def tensor(value: np.ndarray) -> torch.Tensor:
            return (
                torch.from_numpy(np.ascontiguousarray(value))
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
                .to(device)
            )
        return tensor(dirty), tensor(clean)


@torch.inference_mode()
def evaluate(
    model: PosteriorEdgeRestorer,
    denoiser: torch.nn.Module,
    sampler: TilePool,
    *,
    batches: int,
    batch_size: int,
    hypotheses: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    sums = {
        "deterministic_edge_l1": 0.0,
        "sample0_edge_l1": 0.0,
        "oracle_edge_l1": 0.0,
        "edge_diversity": 0.0,
    }
    for _ in range(batches):
        dirty, clean = sampler.sample(batch_size, device)
        mean = denoiser(dirty).float()
        samples = model.sample(dirty, mean, hypotheses=hypotheses, generator=generator)
        target = boundary_pixels(clean).unsqueeze(0)
        sample_error = (boundary_pixels(samples) - target).abs().mean(dim=(-1, -2))
        sums["deterministic_edge_l1"] += float(
            (boundary_pixels(mean) - target[0]).abs().mean()
        )
        sums["sample0_edge_l1"] += float(sample_error[0].mean())
        sums["oracle_edge_l1"] += float(sample_error.min(dim=0).values.mean())
        edge = boundary_pixels(samples)
        pair = (edge[:, None] - edge[None, :]).abs().mean(dim=(-1, -2))
        mask = ~torch.eye(hypotheses, dtype=torch.bool, device=device)
        sums["edge_diversity"] += float(pair.masked_select(mask[:, :, None]).mean())
    return {key: value / batches for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--cache-images", type=int, default=96)
    parser.add_argument("--val-images", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(WORK_ROOT) / "posterior_edge" / "posterior_edge_best.pt",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.smoke:
        print(json.dumps(smoke_test(device), indent=2))
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_names, val_names = train_val_split()
    train_pool = TilePool(train_names, images=args.cache_images, seed=args.seed)
    val_pool = TilePool(val_names, images=args.val_images, seed=args.seed + 50_000)
    denoiser, denoiser_payload = load_match_denoiser("matchden", device=str(device))
    if denoiser is None:
        raise FileNotFoundError("matchden checkpoint is required")
    denoiser.requires_grad_(False).eval()
    model = PosteriorEdgeRestorer(
        width=args.width,
        blocks=args.blocks,
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )
    best_oracle = float("inf")
    best_metrics: dict[str, float] = {}
    rolling: dict[str, float] = {}
    start = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        model.train()
        dirty, clean = train_pool.sample(args.batch_size, device)
        with torch.no_grad():
            mean = denoiser(dirty).float()
        samples = model.sample(dirty, mean, hypotheses=args.hypotheses)
        loss, parts = best_of_k_edge_loss(samples, clean, mean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        for key, value in parts.items():
            rolling[key] = rolling.get(key, 0.0) + value
        if step % args.log_every == 0:
            row = {key: value / args.log_every for key, value in rolling.items()}
            row.update({"step": step, "grad_norm": float(grad_norm), "elapsed": time.time() - start})
            print(json.dumps(row), flush=True)
            rolling.clear()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                denoiser,
                val_pool,
                batches=4,
                batch_size=64,
                hypotheses=args.hypotheses,
                device=device,
                seed=args.seed + 70_000,
            )
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            if metrics["oracle_edge_l1"] < best_oracle:
                best_oracle = metrics["oracle_edge_l1"]
                best_metrics = metrics
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_kwargs": model.model_kwargs,
                        "step": step,
                        "metrics": metrics,
                        "denoiser_step": denoiser_payload.get("step"),
                    },
                    args.output,
                )
    print(json.dumps({"best": best_metrics, "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
