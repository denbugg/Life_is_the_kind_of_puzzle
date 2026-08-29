"""Continue V10 on larger boards with explicit hardest-negative ranking loss."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

for module_file in Path("/kaggle/input").glob("**/train_puzzle_transformer_v10.py"):
    sys.path.insert(0, str(module_file.parent))
    break
import train_puzzle_transformer_v10 as v10


OUT = Path("/kaggle/working/puzzle_hard_v18")
SEED = 20260901


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1200
    side: int = 16
    grad_accum: int = 2
    warmup_steps: int = 80
    lr: float = 5.0e-5
    min_lr: float = 4.0e-6
    weight_decay: float = 0.04
    hard_weight: float = 0.15
    hard_margin: float = 0.5
    real_probability: float = 0.8
    log_every: int = 10
    validate_every: int = 400
    validation_boards: int = 4
    holdout_boards: int = 8


def log(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def find_v10() -> Path:
    for path in sorted(Path("/kaggle/input").glob("**/best.pt")):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("model_config", {}).get("depth") == 14 and "model" in state:
                return path
        except Exception:
            pass
    raise FileNotFoundError("V10 best.pt not mounted")


def hard_negative_loss(logits: torch.Tensor, positions: torch.Tensor,
                       reliable: torch.Tensor, side: int, direction: int,
                       margin: float) -> torch.Tensor:
    target, valid = v10.targets_for_direction(positions, reliable, side, direction)
    masked = logits.masked_fill(~reliable[:, None, :], -1e4)
    bi, source = valid.nonzero(as_tuple=True)
    destination = target[bi, source]
    positive = masked[bi, source, destination]
    candidates = masked[bi, source].clone()
    candidates[torch.arange(len(source), device=logits.device), destination] = -1e4
    hardest = candidates.max(dim=1).values
    return F.relu(margin + hardest - positive).mean()


def combined_loss(right: torch.Tensor, down: torch.Tensor, positions: torch.Tensor,
                  reliable: torch.Tensor, side: int, config: TrainConfig):
    listwise, details = v10.dense_listwise_loss(right, down, positions, reliable, side)
    hard_right = hard_negative_loss(
        right, positions, reliable, side, 0, config.hard_margin)
    hard_down = hard_negative_loss(
        down, positions, reliable, side, 1, config.hard_margin)
    hard = 0.5 * (hard_right + hard_down)
    return listwise + config.hard_weight * hard, {
        **details, "listwise_loss": float(listwise.detach()),
        "hard_loss": float(hard.detach())}


def lr_at(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return config.lr * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * progress))


def save(path: Path, model, optimizer, step: int, config: TrainConfig,
         best: float, source: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model.config), "train_config": asdict(config),
                "best_validation_score": best, "source_checkpoint": str(source)}, temporary)
    temporary.replace(path)


def main() -> None:
    seed_all(SEED); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    source = find_v10()
    state = torch.load(source, map_location="cpu", weights_only=True)
    model = v10.DensePuzzleTransformer(
        v10.ModelConfig(**state["model_config"]), use_checkpoint=True)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device)
    # Preserve learned low-level denoising and most global reasoning. Fine-tune
    # the last four transformer blocks plus edge projections/fusion parameters.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.layers[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for module in (model.output_norm, model.side_norms, model.projections):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.context_gate.requires_grad_(True)
    model.baseline_gate.requires_grad_(True)
    model.residual_gate.requires_grad_(True)
    model.logit_scales.requires_grad_(True)
    config = TrainConfig()
    _root, inputs, targets = v10.find_dataset()
    alignment = v10.build_alignment_cache(inputs, targets, OUT / "alignment_unused.npz")
    train_sampler = v10.SceneSampler(inputs, targets, alignment, 0, 6700, SEED + 1)
    validation_sampler = v10.SceneSampler(inputs, targets, alignment, 6700, 6800, SEED + 2)
    holdout_sampler = v10.SceneSampler(inputs, targets, alignment, 6800, 7000, SEED + 3)
    winner = v10.load_winner(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, growth_interval=2000)
    generator = torch.Generator(device=device).manual_seed(SEED + 99)
    optimizer.zero_grad(set_to_none=True)
    best, accumulated, started = -1.0, 0.0, time.perf_counter()
    log(event="start", device=torch.cuda.get_device_name(), source=str(source),
        model_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters), config=asdict(config))
    for step in range(1, config.steps + 1):
        model.train()
        x, positions, reliable, is_real, scene = train_sampler.sample(
            config.side, real_probability=config.real_probability)
        x = x.to(device); positions = positions.to(device); reliable = reliable.to(device)
        if not is_real:
            x = v10.augment_tiles(x, generator, strong=True)
        baseline = v10.winner_scores(x, winner)
        with torch.autocast("cuda", dtype=torch.float16):
            right, down = model(x, baseline)
            loss, details = combined_loss(right, down, positions, reliable, config.side, config)
        scaler.scale(loss / config.grad_accum).backward()
        accumulated += float(loss.detach())
        if step % config.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            lr = lr_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm, lr = 0.0, optimizer.param_groups[0]["lr"]
        if step == 1 or step % config.log_every == 0:
            log(event="train", step=step, scene=scene, real=is_real,
                reliable_fraction=float(reliable.float().mean()),
                loss=accumulated / (1 if step == 1 else config.log_every),
                lr=lr, grad_norm=grad_norm, gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                seconds=time.perf_counter() - started, **details)
            accumulated = 0.0
        if step % config.validate_every == 0:
            result = v10.evaluate(model, validation_sampler,
                                  list(range(6736, 6736 + config.validation_boards)),
                                  device, synthetic=False, winner=winner)
            log(event="validation", step=step, **result)
            if result["solver_score"] > best:
                best = result["solver_score"]
                save(OUT / "best.pt", model, optimizer, step, config, best, source)
            save(OUT / "latest.pt", model, optimizer, step, config, best, source)
    holdout = v10.evaluate(model, holdout_sampler,
                           list(range(6929, 6929 + config.holdout_boards)),
                           device, synthetic=False, winner=winner)
    report = {"schema": "puzzle-hard-finetune-v18", "seed": SEED,
              "source": str(source), "config": asdict(config),
              "best_validation_score": best, "holdout_6929_6936": holdout,
              "seconds": time.perf_counter() - started}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save(OUT / "final.pt", model, optimizer, config.steps, config, best, source)
    log(event="complete", report=report)


if __name__ == "__main__":
    main()
