"""True contiguous 5x5 block supervision for the isolated TileNAF restorer.

The deployed model still receives one 20x20 tile at a time.  During fine-tuning,
however, batches are sampled as contiguous 5x5 regions from a clean source image,
corrupted independently per tile, and reassembled for losses that cross the real
tile boundaries.  This targets border fidelity without assuming that puzzle order
is known at inference time.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Mapping

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from .degradation import SyntheticTileDegrader
from .losses import (
    LossWeights,
    RestorationLoss,
    charbonnier,
    gradient_charbonnier,
    skimage_like_ssim,
)
from .metrics import tile_metrics
from .model import TileNAFNet, model_parameter_count
from .tiles import merge_tiles_numpy
from .training import (
    FixedValidationPlan,
    atomic_torch_save,
    choose_device,
    evaluate_model,
    load_manifest,
    make_fixed_validation_plan,
    render_fixed_validation,
    resolved_device_fingerprint,
    runtime_versions,
    seed_everything,
    update_ema,
)


TILE = 20
GRID = 24
BLOCK_TILES = 5
BLOCK_PIXELS = BLOCK_TILES * TILE
TILES_PER_BLOCK = BLOCK_TILES * BLOCK_TILES
EXPECTED_INIT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name_hash(names: list[str] | tuple[str, ...]) -> str:
    payload = json.dumps(
        sorted(str(name) for name in names),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_fingerprint() -> str:
    package = Path(__file__).resolve().parent
    paths = (
        "__init__.py",
        "block5x5.py",
        "degradation.py",
        "losses.py",
        "metrics.py",
        "model.py",
        "tiles.py",
        "training.py",
    )
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode("utf-8"))
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def _split_patch(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch)
    if patch.shape != (BLOCK_PIXELS, BLOCK_PIXELS, 3):
        raise ValueError(f"expected {BLOCK_PIXELS}x{BLOCK_PIXELS} RGB patch, got {patch.shape}")
    return (
        patch.reshape(BLOCK_TILES, TILE, BLOCK_TILES, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(TILES_PER_BLOCK, TILE, TILE, 3)
    )


def assemble_blocks(tiles: torch.Tensor, block_tiles: int = BLOCK_TILES) -> torch.Tensor:
    """Reassemble ``B x K^2 x C x H x W`` tiles into ``B x C x KH x KW``."""
    if tiles.ndim != 6:
        raise ValueError(f"expected BxKxKxCxHxW tiles, got {tuple(tiles.shape)}")
    batch, rows, columns, channels, height, width = tiles.shape
    if rows != block_tiles or columns != block_tiles or (height, width) != (TILE, TILE):
        raise ValueError("unexpected block geometry")
    return (
        tiles.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .reshape(batch, channels, rows * height, columns * width)
    )


class CleanBlockStore:
    """Preload whole clean source images and sample aligned contiguous 5x5 blocks."""

    def __init__(self, target_dir: Path, names: list[str]) -> None:
        if not names:
            raise ValueError("block store requires at least one source")
        self.names = list(names)
        self.images = np.empty((len(names), GRID * TILE, GRID * TILE, 3), dtype=np.uint8)
        digest = hashlib.sha256()
        started = time.perf_counter()
        for index, name in enumerate(names):
            path = target_dir / name
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            if image.shape != (GRID * TILE, GRID * TILE, 3):
                raise ValueError(f"unexpected target shape for {name}: {image.shape}")
            self.images[index] = image
            digest.update(name.encode("utf-8"))
            digest.update(image.tobytes())
        self.sha256 = digest.hexdigest()
        print(
            json.dumps(
                {
                    "event": "block_store_loaded",
                    "images": len(names),
                    "gib": self.images.nbytes / (1024**3),
                    "sha256": self.sha256,
                    "seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output = np.empty(
            (batch_size, TILES_PER_BLOCK, TILE, TILE, 3), dtype=np.uint8
        )
        image_indices = rng.integers(0, len(self.names), size=batch_size)
        top_rows = rng.integers(0, GRID - BLOCK_TILES + 1, size=batch_size)
        left_columns = rng.integers(0, GRID - BLOCK_TILES + 1, size=batch_size)
        transforms = rng.integers(0, 8, size=batch_size)
        for index, (image_index, row, column, transform) in enumerate(
            zip(image_indices, top_rows, left_columns, transforms, strict=True)
        ):
            y0, x0 = int(row) * TILE, int(column) * TILE
            patch = self.images[
                int(image_index),
                y0 : y0 + BLOCK_PIXELS,
                x0 : x0 + BLOCK_PIXELS,
            ]
            patch = np.rot90(patch, int(transform) % 4, axes=(0, 1))
            if int(transform) >= 4:
                patch = np.flip(patch, axis=1)
            output[index] = _split_patch(np.ascontiguousarray(patch))
        return (
            torch.from_numpy(np.ascontiguousarray(output.transpose(0, 1, 4, 2, 3)))
            .float()
            .div_(255.0)
        )


def seam_gradient_charbonnier(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match cross-tile finite differences at every internal 5x5 seam."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must be matching BCHW blocks")
    if tuple(prediction.shape[-2:]) != (BLOCK_PIXELS, BLOCK_PIXELS):
        raise ValueError("seam loss requires a 100x100 block")
    values: list[torch.Tensor] = []
    for boundary in range(TILE, BLOCK_PIXELS, TILE):
        pred_dx = prediction[:, :, :, boundary] - prediction[:, :, :, boundary - 1]
        true_dx = target[:, :, :, boundary] - target[:, :, :, boundary - 1]
        pred_dy = prediction[:, :, boundary, :] - prediction[:, :, boundary - 1, :]
        true_dy = target[:, :, boundary, :] - target[:, :, boundary - 1, :]
        values.append(charbonnier(pred_dx - true_dx).mean())
        values.append(charbonnier(pred_dy - true_dy).mean())
    return torch.stack(values).mean()


def neighbouring_tile_mean_loss(prediction_tiles: torch.Tensor, target_tiles: torch.Tensor) -> torch.Tensor:
    """Preserve low-frequency colour changes between adjacent true tiles."""
    if prediction_tiles.shape != target_tiles.shape or prediction_tiles.ndim != 6:
        raise ValueError("expected matching Bx5x5xCx20x20 tensors")
    pred_mean = prediction_tiles.mean(dim=(-2, -1))
    true_mean = target_tiles.mean(dim=(-2, -1))
    pred_horizontal = pred_mean[:, :, 1:] - pred_mean[:, :, :-1]
    true_horizontal = true_mean[:, :, 1:] - true_mean[:, :, :-1]
    pred_vertical = pred_mean[:, 1:] - pred_mean[:, :-1]
    true_vertical = true_mean[:, 1:] - true_mean[:, :-1]
    return 0.5 * (
        F.smooth_l1_loss(pred_horizontal, true_horizontal)
        + F.smooth_l1_loss(pred_vertical, true_vertical)
    )


@dataclass(frozen=True)
class BlockLossWeights:
    tile_ssim: float = 0.05
    tile_gradient: float = 0.03
    tile_boundary_extra: float = 0.75
    block_ssim: float = 0.08
    block_gradient: float = 0.04
    seam_gradient: float = 0.12
    neighbour_mean: float = 0.03


class Block5x5Loss(nn.Module):
    def __init__(self, weights: BlockLossWeights) -> None:
        super().__init__()
        self.weights = weights
        self.tile_loss = RestorationLoss(
            LossWeights(
                ssim=weights.tile_ssim,
                gradient=weights.tile_gradient,
                boundary_extra=weights.tile_boundary_extra,
            )
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        parameter_prediction: torch.Tensor,
        parameter_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.ndim != 6:
            raise ValueError("expected matching Bx5x5xCx20x20 tensors")
        batch = prediction.shape[0]
        flat_prediction = prediction.reshape(batch * TILES_PER_BLOCK, 3, TILE, TILE)
        flat_target = target.reshape(batch * TILES_PER_BLOCK, 3, TILE, TILE)
        tile_total, tile_components = self.tile_loss(
            flat_prediction,
            flat_target,
            parameter_prediction,
            parameter_target,
        )
        prediction_block = assemble_blocks(prediction)
        target_block = assemble_blocks(target)
        block_ssim = skimage_like_ssim(prediction_block, target_block)
        block_gradient = gradient_charbonnier(prediction_block, target_block)
        seam_gradient = seam_gradient_charbonnier(prediction_block, target_block)
        neighbour_mean = neighbouring_tile_mean_loss(prediction, target)
        total = (
            tile_total
            + self.weights.block_ssim * (1.0 - block_ssim)
            + self.weights.block_gradient * block_gradient
            + self.weights.seam_gradient * seam_gradient
            + self.weights.neighbour_mean * neighbour_mean
        )
        components = {
            **{f"tile_{key}": value for key, value in tile_components.items()},
            "block_ssim": block_ssim.detach(),
            "block_gradient": block_gradient.detach(),
            "seam_gradient": seam_gradient.detach(),
            "neighbour_mean": neighbour_mean.detach(),
            "total": total.detach(),
        }
        return total, components


@dataclass(frozen=True)
class Block5x5TrainConfig:
    data_root: str
    manifest: str
    protocol: str
    init_checkpoint: str
    output: str
    variant: str
    train_images: int = 2048
    steps: int = 6000
    block_batch_size: int = 8
    eval_batch_size: int = 512
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    eval_interval: int = 1000
    log_interval: int = 50
    seed: int = 2026071205
    device: str = "auto"
    tile_ssim: float = 0.05
    tile_gradient: float = 0.03
    tile_boundary_extra: float = 0.75
    block_ssim: float = 0.08
    block_gradient: float = 0.04
    seam_gradient: float = 0.12
    neighbour_mean: float = 0.03


def _validate_train_config(config: Block5x5TrainConfig) -> None:
    if config.variant not in {"moderate", "strong"}:
        raise ValueError("variant must be moderate or strong")
    for name in (
        "train_images",
        "steps",
        "block_batch_size",
        "eval_batch_size",
        "eval_interval",
        "log_interval",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.train_images > 4900:
        raise ValueError("train_images cannot exceed the frozen train split")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("invalid optimizer configuration")
    if not 0.0 <= config.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    weights = BlockLossWeights(
        config.tile_ssim,
        config.tile_gradient,
        config.tile_boundary_extra,
        config.block_ssim,
        config.block_gradient,
        config.seam_gradient,
        config.neighbour_mean,
    )
    if any(value < 0 or not math.isfinite(value) for value in asdict(weights).values()):
        raise ValueError("loss weights must be finite and non-negative")


def load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != "denoise_block5x5_protocol":
        raise ValueError("unsupported block5x5 protocol")
    required = {"development", "frozen_gate"}
    if set(payload.get("source_partitions", {})) != required:
        raise ValueError("protocol source partitions must be development and frozen_gate")
    for partition in required:
        record = payload["source_partitions"][partition]
        names = record.get("names")
        if not isinstance(names, list) or len(names) != record.get("count"):
            raise ValueError(f"invalid {partition} source list")
        if canonical_name_hash(names) != record.get("names_sha256"):
            raise ValueError(f"{partition} source hash mismatch")
    development = set(payload["source_partitions"]["development"]["names"])
    gate = set(payload["source_partitions"]["frozen_gate"]["names"])
    if development & gate:
        raise ValueError("development and frozen gate overlap")
    return payload


def _official_image_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(target, prediction, channel_axis=2, data_range=255))


def _source_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    metrics = tile_metrics(prediction, target)
    metrics["ordered_image_ssim"] = _official_image_ssim(
        merge_tiles_numpy(prediction), merge_tiles_numpy(target)
    )
    return metrics


def evaluate_validation_by_source(
    model: nn.Module,
    corrupt: np.ndarray,
    clean: np.ndarray,
    source_names: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if len(corrupt) != len(clean) or len(clean) != len(source_names) * 576:
        raise ValueError("validation arrays do not contain complete source images")
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(corrupt), batch_size):
            batch = (
                torch.from_numpy(
                    np.ascontiguousarray(corrupt[start : start + batch_size].transpose(0, 3, 1, 2))
                )
                .float()
                .div_(255.0)
                .to(device)
            )
            restored = model(batch)
            predictions.append(
                restored.detach()
                .cpu()
                .mul(255.0)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(0, 2, 3, 1)
                .numpy()
            )
    prediction = np.concatenate(predictions)
    per_source: list[dict[str, object]] = []
    for index, name in enumerate(source_names):
        start = index * 576
        metrics = _source_metrics(prediction[start : start + 576], clean[start : start + 576])
        per_source.append({"source": name, **metrics})
    macro = {
        key: float(np.mean([float(record[key]) for record in per_source]))
        for key in per_source[0]
        if key != "source"
    }
    return macro, per_source


def _balanced_metrics(panels: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(values) for values in panels.values()))
    return {
        key: float(np.mean([float(values[key]) for values in panels.values()]))
        for key in sorted(keys)
    }


def _load_initial_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    if sha256_file(checkpoint_path) != EXPECTED_INIT_SHA256:
        raise ValueError("initial checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "tile-naf":
        raise ValueError("initial checkpoint is not TileNAF")
    state = checkpoint.get("ema_state")
    if not isinstance(state, dict):
        raise ValueError("initial checkpoint has no EMA state")
    model = TileNAFNet()
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, checkpoint


def train_block5x5(config: Block5x5TrainConfig) -> dict:
    _validate_train_config(config)
    seed_everything(config.seed)
    protocol_path = Path(config.protocol)
    protocol = load_protocol(protocol_path)
    manifest_path = Path(config.manifest)
    manifest = load_manifest(manifest_path)
    expected_manifest = protocol["inputs"]["manifest_sha256"]
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("manifest hash does not match protocol")
    init_path = Path(config.init_checkpoint)
    if sha256_file(init_path) != protocol["inputs"]["init_checkpoint_sha256"]:
        raise ValueError("initial checkpoint hash does not match protocol")

    development_names = list(protocol["source_partitions"]["development"]["names"])
    validation_set = set(manifest["splits"]["val"])
    train_set = set(manifest["splits"]["train"])
    if not set(development_names) <= validation_set or set(development_names) & train_set:
        raise ValueError("development names are not a whole-source held-out validation subset")
    gate_names = set(protocol["source_partitions"]["frozen_gate"]["names"])
    if set(development_names) & gate_names:
        raise ValueError("development sources overlap frozen gate")

    device = choose_device(config.device)
    rng = np.random.default_rng(config.seed)
    target_dir = Path(config.data_root) / "train" / "targets"
    train_names = list(manifest["splits"]["train"][: config.train_images])
    store = CleanBlockStore(target_dir, train_names)
    model, initial_checkpoint = _load_initial_model(init_path, device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    degrader = SyntheticTileDegrader()
    plan: FixedValidationPlan = make_fixed_validation_plan(
        target_dir,
        development_names,
        576,
        int(protocol["panel_seeds"]["development"]),
        degrader,
    )
    primary_corrupt = render_fixed_validation(
        plan, degrader, config.eval_batch_size, codec="kornia"
    )
    independent_corrupt = render_fixed_validation(
        plan, degrader, min(config.eval_batch_size, 512), codec="pillow"
    )

    baseline_panels: dict[str, dict[str, float]] = {}
    baseline_per_source: dict[str, list[dict[str, object]]] = {}
    for panel_name, corrupt in (
        ("primary_kornia", primary_corrupt),
        ("independent_libjpeg", independent_corrupt),
    ):
        macro, per_source = evaluate_validation_by_source(
            model,
            corrupt,
            plan.clean,
            development_names,
            device,
            config.eval_batch_size,
        )
        baseline_panels[panel_name] = macro
        baseline_per_source[panel_name] = per_source

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.learning_rate * 0.05,
    )
    criterion = Block5x5Loss(
        BlockLossWeights(
            tile_ssim=config.tile_ssim,
            tile_gradient=config.tile_gradient,
            tile_boundary_extra=config.tile_boundary_extra,
            block_ssim=config.block_ssim,
            block_gradient=config.block_gradient,
            seam_gradient=config.seam_gradient,
            neighbour_mean=config.neighbour_mean,
        )
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed + 1000)

    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest = output.with_name(f"{output.stem}_latest{output.suffix}")
    history: list[dict[str, object]] = []
    best_key = (-math.inf, -math.inf)
    best_step: int | None = None
    running: dict[str, float] = {}
    started = time.perf_counter()
    versions = runtime_versions()
    source_sha256 = code_fingerprint()
    metadata = {
        "schema_version": 1,
        "kind": "tile_naf_block5x5_finetune",
        "model_name": "tile-naf",
        "variant": config.variant,
        "config": asdict(config),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "init_checkpoint_sha256": sha256_file(init_path),
        "training_data_sha256": store.sha256,
        "training_source_names": train_names,
        "training_source_names_sha256": canonical_name_hash(train_names),
        "development_source_names": development_names,
        "development_source_names_sha256": canonical_name_hash(development_names),
        "frozen_gate_source_names_sha256": protocol["source_partitions"]["frozen_gate"][
            "names_sha256"
        ],
        "source_code_sha256": source_sha256,
        "runtime_versions": versions,
        "resolved_device_fingerprint": resolved_device_fingerprint(device),
        "parameter_count": model_parameter_count(model),
        "baseline_development": {
            "panels": baseline_panels,
            "balanced": _balanced_metrics(baseline_panels),
            "per_source": baseline_per_source,
        },
        "python": platform.python_version(),
    }
    print(json.dumps({"event": "block5x5_train_start", **metadata}, sort_keys=True), flush=True)

    for step in range(1, config.steps + 1):
        clean = store.sample(config.block_batch_size, rng).to(device)
        flat_clean = clean.reshape(config.block_batch_size * TILES_PER_BLOCK, 3, TILE, TILE)
        with torch.no_grad():
            parameters = degrader.sample_parameters(len(flat_clean), device, torch_generator)
            noise = torch.randn(
                flat_clean.shape,
                device=device,
                dtype=flat_clean.dtype,
                generator=torch_generator,
            )
            corrupt, parameters = degrader(flat_clean, params=parameters, noise=noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction, parameter_prediction = model(corrupt, return_aux=True)
            prediction_blocks = prediction.reshape(
                config.block_batch_size, BLOCK_TILES, BLOCK_TILES, 3, TILE, TILE
            )
            clean_blocks = clean.reshape(
                config.block_batch_size, BLOCK_TILES, BLOCK_TILES, 3, TILE, TILE
            )
            loss, components = criterion(
                prediction_blocks,
                clean_blocks,
                parameter_prediction,
                parameters.normalized(),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema_decay = min(config.ema_decay, (1.0 + step) / (10.0 + step))
        update_ema(ema, model, ema_decay)

        for key, value in components.items():
            running[key] = running.get(key, 0.0) + float(value.cpu())
        if step == 1 or step % config.log_interval == 0:
            divisor = 1 if step == 1 else config.log_interval
            print(
                json.dumps(
                    {
                        "event": "block5x5_train_step",
                        "variant": config.variant,
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.perf_counter() - started,
                        **{key: value / divisor for key, value in running.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            running = {}

        if step % config.eval_interval and step != config.steps:
            continue
        candidate_panels: dict[str, dict[str, float]] = {}
        candidate_per_source: dict[str, list[dict[str, object]]] = {}
        for panel_name, corrupt in (
            ("primary_kornia", primary_corrupt),
            ("independent_libjpeg", independent_corrupt),
        ):
            macro, per_source = evaluate_validation_by_source(
                ema,
                corrupt,
                plan.clean,
                development_names,
                device,
                config.eval_batch_size,
            )
            candidate_panels[panel_name] = macro
            candidate_per_source[panel_name] = per_source
        candidate_balanced = _balanced_metrics(candidate_panels)
        baseline_balanced = _balanced_metrics(baseline_panels)
        deltas = {
            key: candidate_balanced[key] - baseline_balanced[key]
            for key in candidate_balanced
            if key in baseline_balanced
        }
        safety = {
            "tile_ssim_noninferior_0_0005": deltas["tile_ssim"] >= -0.0005,
            "boundary_mae_growth_at_most_0_2pct": candidate_balanced["boundary_mae"]
            <= baseline_balanced["boundary_mae"] * 1.002,
            "gradient_mae_growth_at_most_0_2pct": candidate_balanced["gradient_mae"]
            <= baseline_balanced["gradient_mae"] * 1.002,
            "both_panel_ordered_ssim_positive": all(
                candidate_panels[name]["ordered_image_ssim"]
                > baseline_panels[name]["ordered_image_ssim"]
                for name in candidate_panels
            ),
        }
        record: dict[str, object] = {
            "step": step,
            "panels": candidate_panels,
            "balanced": candidate_balanced,
            "deltas_vs_initial": deltas,
            "safety": safety,
            "eligible_for_dev_selection": all(safety.values()),
            "per_source": candidate_per_source,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "event": "block5x5_validation",
                    "variant": config.variant,
                    **{key: value for key, value in record.items() if key != "per_source"},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        key = (
            float(deltas["ordered_image_ssim"]) if all(safety.values()) else -math.inf,
            float(deltas["tile_ssim"]),
        )
        improved = key > best_key
        if improved:
            best_key = key
            best_step = step
        checkpoint = {
            **metadata,
            "step": step,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "latest_development": record,
            "best_step": best_step,
            "best_key": list(best_key),
        }
        atomic_torch_save(checkpoint, latest)
        if improved:
            checkpoint["best_development"] = record
            atomic_torch_save(checkpoint, output)
            print(
                json.dumps(
                    {
                        "event": "block5x5_checkpoint_saved",
                        "variant": config.variant,
                        "path": str(output),
                        "step": step,
                        "best_key": list(best_key),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return {
        "variant": config.variant,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "best_step": best_step,
        "best_key": list(best_key),
        "history_count": len(history),
        "seconds": time.perf_counter() - started,
    }


__all__ = [
    "BLOCK_PIXELS",
    "BLOCK_TILES",
    "Block5x5Loss",
    "Block5x5TrainConfig",
    "BlockLossWeights",
    "CleanBlockStore",
    "assemble_blocks",
    "canonical_name_hash",
    "code_fingerprint",
    "evaluate_validation_by_source",
    "load_protocol",
    "neighbouring_tile_mean_loss",
    "seam_gradient_charbonnier",
    "sha256_file",
    "train_block5x5",
]
