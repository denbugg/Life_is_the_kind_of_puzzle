"""Training utilities for exact synthetic tile restoration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from .degradation import SyntheticTileDegrader
from .losses import LossWeights, RestorationLoss
from .metrics import ordered_image_ssim, tile_metrics
from .model import FullResolutionTileNAF, TileNAFNet, model_parameter_count
from .tiles import split_tiles_numpy


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    manifest: str
    output: str
    model: str = "tile-naf"
    train_images: int = 256
    val_images: int = 16
    val_tiles_per_image: int = 576
    steps: int = 1000
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    seed: int = 20260710
    device: str = "auto"
    log_interval: int = 25
    eval_interval: int = 250
    ssim_start_fraction: float = 0.75
    variant_weights: tuple[float, float, float] = (1.0, 0.0, 0.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"train", "val", "audit"}
    if set(payload.get("splits", {})) != required:
        raise ValueError(f"manifest must contain exactly {sorted(required)}")
    overlap = set(payload["excluded_test_overlap"])
    if any(overlap & set(payload["splits"][split]) for split in required):
        raise ValueError("test-overlap source leaked into a denoising split")
    return payload


class CleanTileStore:
    """Preloaded uint8 target tiles for fast random exact-pair sampling."""

    def __init__(self, target_dir: Path, names: list[str]) -> None:
        self.names = list(names)
        self.tiles = np.empty((len(names), 576, 20, 20, 3), dtype=np.uint8)
        started = time.perf_counter()
        for index, name in enumerate(names):
            image = np.asarray(Image.open(target_dir / name).convert("RGB"), dtype=np.uint8)
            self.tiles[index] = split_tiles_numpy(image)
        print(
            json.dumps(
                {
                    "event": "tile_store_loaded",
                    "images": len(names),
                    "tiles": len(names) * 576,
                    "gib": self.tiles.nbytes / (1024**3),
                    "seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        image_indices = rng.integers(0, len(self.names), size=batch_size)
        tile_indices = rng.integers(0, 576, size=batch_size)
        batch = self.tiles[image_indices, tile_indices]
        return torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).float().div_(255.0)


def random_dihedral(tiles: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    transforms = rng.integers(0, 8, size=len(tiles))
    output = torch.empty_like(tiles)
    for transform in range(8):
        indices_np = np.flatnonzero(transforms == transform)
        if len(indices_np) == 0:
            continue
        indices = torch.as_tensor(indices_np, device=tiles.device)
        selected = tiles[indices]
        selected = torch.rot90(selected, transform % 4, dims=(-2, -1))
        if transform >= 4:
            selected = torch.flip(selected, dims=(-1,))
        output[indices] = selected
    return output


def build_model(name: str) -> nn.Module:
    if name == "tile-naf":
        return TileNAFNet()
    if name == "full-naf":
        return FullResolutionTileNAF()
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    model_parameters = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(model_parameters[name], alpha=1.0 - decay)
    model_buffers = dict(model.named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(model_buffers[name])


def make_fixed_validation(
    target_dir: Path,
    names: list[str],
    tiles_per_image: int,
    seed: int,
    degrader: SyntheticTileDegrader,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    clean_parts = []
    source_names = []
    for name in names:
        image = np.asarray(Image.open(target_dir / name).convert("RGB"), dtype=np.uint8)
        tiles = split_tiles_numpy(image)
        if tiles_per_image < 576:
            indices = np.sort(rng.choice(576, size=tiles_per_image, replace=False))
            tiles = tiles[indices]
        clean_parts.append(tiles)
        source_names.extend([name] * len(tiles))
    clean = np.concatenate(clean_parts)

    torch_generator = torch.Generator().manual_seed(seed)
    corrupt_parts = []
    for start in range(0, len(clean), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(clean[start : start + batch_size].transpose(0, 3, 1, 2)))
        batch = batch.float().div_(255.0)
        corrupted, _ = degrader(batch, generator=torch_generator)
        corrupt_parts.append(
            np.clip(np.rint(corrupted.numpy().transpose(0, 2, 3, 1) * 255.0), 0, 255).astype(np.uint8)
        )
    return np.concatenate(corrupt_parts), clean, source_names


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    corrupt: np.ndarray,
    clean: np.ndarray,
    device: torch.device,
    batch_size: int,
    complete_images: bool = False,
) -> dict[str, float]:
    model.eval()
    predictions = []
    for start in range(0, len(clean), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(corrupt[start : start + batch_size].transpose(0, 3, 1, 2))
        ).float().div_(255.0).to(device)
        restored = model(batch)
        predictions.append(
            np.clip(np.rint(restored.cpu().numpy().transpose(0, 2, 3, 1) * 255.0), 0, 255).astype(np.uint8)
        )
    predictions_array = np.concatenate(predictions)
    metrics = tile_metrics(predictions_array, clean)
    if complete_images:
        metrics["ordered_image_ssim"] = ordered_image_ssim(predictions_array, clean)
    return metrics


def train(config: TrainConfig) -> dict:
    seed_everything(config.seed)
    root = Path(config.data_root)
    manifest_path = Path(config.manifest)
    manifest = load_manifest(manifest_path)
    target_dir = root / "train" / "targets"
    train_names = manifest["splits"]["train"][: config.train_images or None]
    val_names = manifest["splits"]["val"][: config.val_images or None]
    if not train_names or not val_names:
        raise ValueError("empty train or validation selection")

    device = choose_device(config.device)
    rng = np.random.default_rng(config.seed)
    store = CleanTileStore(target_dir, train_names)
    degrader = SyntheticTileDegrader(variant_weights=config.variant_weights)
    validation_corrupt, validation_clean, _validation_sources = make_fixed_validation(
        target_dir,
        val_names,
        config.val_tiles_per_image,
        config.seed + 1,
        degrader,
        config.eval_batch_size,
    )
    raw_metrics = tile_metrics(validation_corrupt, validation_clean)

    model = build_model(config.model).to(device)
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.steps, 1),
        eta_min=config.learning_rate * 0.05,
    )
    warmup_loss = RestorationLoss(LossWeights(ssim=0.0))
    final_loss = RestorationLoss(LossWeights(ssim=0.10))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    print(
        json.dumps(
            {
                "event": "train_start",
                "config": asdict(config),
                "device": str(device),
                "parameters": model_parameter_count(model),
                "manifest_sha256": manifest_sha256,
                "raw_validation": raw_metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.perf_counter()
    running = {}
    best_ssim = -math.inf
    history = []
    ssim_start = int(config.steps * config.ssim_start_fraction)

    for step in range(1, config.steps + 1):
        clean = store.sample(config.batch_size, rng).to(device)
        clean = random_dihedral(clean, rng)
        with torch.no_grad():
            corrupt, parameters = degrader(clean)
        criterion = final_loss if step > ssim_start else warmup_loss

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction, parameter_prediction = model(corrupt, return_aux=True)
            loss, components = criterion(
                prediction,
                clean,
                parameter_prediction,
                parameters.normalized(),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # Bias-correct the early EMA so short smoke runs do not mostly measure
        # the identity initialization. The decay reaches the configured value
        # naturally during long training.
        ema_decay = min(config.ema_decay, (1.0 + step) / (10.0 + step))
        update_ema(ema, model, ema_decay)

        for key, value in components.items():
            running[key] = running.get(key, 0.0) + float(value.cpu())

        if step % config.log_interval == 0 or step == 1:
            divisor = config.log_interval if step % config.log_interval == 0 else 1
            payload = {
                "event": "train_step",
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
                **{key: value / divisor for key, value in running.items()},
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            running = {}

        should_evaluate = step % config.eval_interval == 0 or step == config.steps
        if should_evaluate:
            metrics = evaluate_model(
                ema,
                validation_corrupt,
                validation_clean,
                device,
                config.eval_batch_size,
                complete_images=config.val_tiles_per_image == 576,
            )
            record = {"step": step, **metrics}
            history.append(record)
            print(json.dumps({"event": "validation", **record}, sort_keys=True), flush=True)
            if metrics["tile_ssim"] > best_ssim:
                best_ssim = metrics["tile_ssim"]
                checkpoint = {
                    "schema_version": 1,
                    "model_name": config.model,
                    "model_state": model.state_dict(),
                    "ema_state": ema.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(config),
                    "manifest_sha256": manifest_sha256,
                    "raw_validation": raw_metrics,
                    "history": history,
                    "best_validation": record,
                }
                torch.save(checkpoint, output)
                print(json.dumps({"event": "checkpoint_saved", "path": str(output), **record}), flush=True)

    return {
        "output": str(output),
        "raw_validation": raw_metrics,
        "best_ssim": best_ssim,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
