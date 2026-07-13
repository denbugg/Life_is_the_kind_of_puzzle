"""Training utilities for exact synthetic tile restoration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, features as pillow_features
import torch
from torch import nn

from .degradation import DegradationParameters, SyntheticTileDegrader, pillow_libjpeg_degrade
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
    libjpeg_val_images: int = 4
    resume: str | None = None
    init_weights: str | None = None
    loss_ssim: float = 0.10
    loss_gradient: float = 0.05
    loss_boundary_extra: float = 0.50


# These fields change the optimization trajectory.  Resume must therefore use
# the exact values stored in the checkpoint; silently changing any of them
# makes the loaded optimizer/scheduler/RNG state internally inconsistent.
_RESUME_CRITICAL_FIELDS = (
    "model",
    "train_images",
    "steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "ema_decay",
    "seed",
    "ssim_start_fraction",
    "variant_weights",
    "val_images",
    "val_tiles_per_image",
    "eval_batch_size",
    "eval_interval",
    "libjpeg_val_images",
    "output",
    "init_weights",
    "loss_ssim",
    "loss_gradient",
    "loss_boundary_extra",
)


def validate_resume_compatibility(
    checkpoint: dict,
    config: TrainConfig,
    manifest_sha256: str,
    training_data_sha256: str | None = None,
    validation_data_sha256: str | None = None,
    source_code_sha256: str | None = None,
) -> None:
    """Reject resume configurations that cannot preserve the saved trajectory."""
    if checkpoint.get("schema_version") != 2:
        raise ValueError("resume requires a schema_version=2 checkpoint")
    if checkpoint.get("model_name") != config.model:
        raise ValueError("resume checkpoint model does not match requested model")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("resume checkpoint manifest hash mismatch")
    if training_data_sha256 is not None and checkpoint.get("training_data_sha256") != training_data_sha256:
        raise ValueError("resume checkpoint training-data fingerprint mismatch")
    if validation_data_sha256 is not None and checkpoint.get("validation_data_sha256") != validation_data_sha256:
        raise ValueError("resume checkpoint validation-data fingerprint mismatch")
    if source_code_sha256 is not None and checkpoint.get("source_code_sha256") != source_code_sha256:
        raise ValueError("resume checkpoint source-code fingerprint mismatch")

    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("resume checkpoint is missing its training config")
    mismatches = []
    current = asdict(config)
    for field in _RESUME_CRITICAL_FIELDS:
        saved = saved_config.get(field)
        requested = current[field]
        if field == "variant_weights":
            saved = tuple(saved) if saved is not None else saved
            requested = tuple(requested)
        if saved != requested:
            mismatches.append(f"{field}: saved={saved!r}, requested={requested!r}")
    if mismatches:
        raise ValueError("resume training configuration mismatch: " + "; ".join(mismatches))

    step = checkpoint.get("step")
    if not isinstance(step, int) or not 0 <= step <= config.steps:
        raise ValueError(f"resume checkpoint has invalid step {step!r} for {config.steps} total steps")
    required_state = (
        "model_state",
        "ema_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
    )
    missing = [key for key in required_state if key not in checkpoint]
    if missing:
        raise ValueError(f"resume checkpoint is missing state: {', '.join(missing)}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_train_config(config: TrainConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.train_images < 0 or config.val_images < 0:
        raise ValueError("train_images and val_images must be non-negative")
    if config.batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.log_interval <= 0 or config.eval_interval <= 0:
        raise ValueError("log/eval intervals must be positive")
    if not 1 <= config.val_tiles_per_image <= 576:
        raise ValueError("val_tiles_per_image must be in [1, 576]")
    if config.libjpeg_val_images < 0:
        raise ValueError("libjpeg_val_images must be non-negative")
    if not 0.0 <= config.ssim_start_fraction <= 1.0:
        raise ValueError("ssim_start_fraction must be in [0, 1]")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0.0 <= config.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if min(config.loss_ssim, config.loss_gradient, config.loss_boundary_extra) < 0:
        raise ValueError("loss weights must be non-negative")
    if config.resume and config.init_weights:
        raise ValueError("init_weights and resume are mutually exclusive")
    SyntheticTileDegrader(variant_weights=config.variant_weights)


def runtime_versions() -> dict[str, str]:
    import cv2
    import PIL
    import kornia
    import scipy
    import skimage

    try:
        libjpeg_turbo = str(pillow_features.check_feature("libjpeg_turbo"))
        libjpeg_turbo_version = str(pillow_features.version_feature("libjpeg_turbo"))
    except (ValueError, AttributeError):
        libjpeg_turbo = "unknown"
        libjpeg_turbo_version = "unknown"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "jpeg_codec": str(pillow_features.version_codec("jpg")),
        "libjpeg_turbo": libjpeg_turbo,
        "libjpeg_turbo_version": libjpeg_turbo_version,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "kornia": kornia.__version__,
    }


def source_code_fingerprint() -> str:
    """Hash every source file that can change the synthetic training trajectory."""
    package_dir = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "degradation.py",
        "losses.py",
        "metrics.py",
        "model.py",
        "tiles.py",
        "training.py",
    )
    digest = hashlib.sha256()
    for name in names:
        path = package_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolved_device_fingerprint(device: torch.device) -> dict:
    result = {"type": device.type, "machine": platform.machine()}
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "index": int(index),
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(properties.total_memory),
            }
        )
    elif device.type == "mps":
        result["mac_ver"] = platform.mac_ver()[0]
    else:
        result["processor"] = platform.processor()
    return result


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rng_state(rng: np.random.Generator, device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict, rng: np.random.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if device.type == "mps" and "torch_mps" in state and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["torch_mps"])


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
        digest = hashlib.sha256()
        started = time.perf_counter()
        for index, name in enumerate(names):
            image = np.asarray(Image.open(target_dir / name).convert("RGB"), dtype=np.uint8)
            self.tiles[index] = split_tiles_numpy(image)
            digest.update(name.encode("utf-8"))
            digest.update(image.tobytes())
        self.sha256 = digest.hexdigest()
        print(
            json.dumps(
                {
                    "event": "tile_store_loaded",
                    "images": len(names),
                    "tiles": len(names) * 576,
                    "gib": self.tiles.nbytes / (1024**3),
                    "sha256": self.sha256,
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


@dataclass(frozen=True)
class FixedValidationPlan:
    clean: np.ndarray
    parameters: DegradationParameters
    noise: torch.Tensor
    source_names: list[str]


def make_fixed_validation_plan(
    target_dir: Path,
    names: list[str],
    tiles_per_image: int,
    seed: int,
    degrader: SyntheticTileDegrader,
) -> FixedValidationPlan:
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

    parameter_generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator().manual_seed(seed + 1)
    parameters = degrader.sample_parameters(len(clean), torch.device("cpu"), parameter_generator)
    noise = torch.randn((len(clean), 3, 20, 20), generator=noise_generator)
    return FixedValidationPlan(
        clean=clean,
        parameters=parameters,
        noise=noise,
        source_names=source_names,
    )


@torch.no_grad()
def render_fixed_validation(
    plan: FixedValidationPlan,
    degrader: SyntheticTileDegrader,
    batch_size: int = 512,
    codec: str = "kornia",
) -> np.ndarray:
    if codec not in {"kornia", "pillow"}:
        raise ValueError(f"unsupported validation codec {codec}")
    corrupt_parts = []
    for start in range(0, len(plan.clean), batch_size):
        stop = min(start + batch_size, len(plan.clean))
        batch = torch.from_numpy(
            np.ascontiguousarray(plan.clean[start:stop].transpose(0, 3, 1, 2))
        )
        batch = batch.float().div_(255.0)
        noise = plan.noise[start:stop]
        chunk_parameters = plan.parameters.index(slice(start, stop))
        if codec == "kornia":
            corrupted, _ = degrader(batch, params=chunk_parameters, noise=noise)
            corrupt_parts.append(
                np.clip(np.rint(corrupted.numpy().transpose(0, 2, 3, 1) * 255.0), 0, 255).astype(np.uint8)
            )
        else:
            corrupt_parts.append(
                pillow_libjpeg_degrade(
                    plan.clean[start:stop],
                    chunk_parameters,
                    noise.numpy().transpose(0, 2, 3, 1),
                )
            )
    return np.concatenate(corrupt_parts)


def make_fixed_validation(
    target_dir: Path,
    names: list[str],
    tiles_per_image: int,
    seed: int,
    degrader: SyntheticTileDegrader,
    batch_size: int = 512,
    codec: str = "kornia",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    plan = make_fixed_validation_plan(target_dir, names, tiles_per_image, seed, degrader)
    corrupt = render_fixed_validation(plan, degrader, batch_size, codec)
    return corrupt, plan.clean, plan.source_names


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
    validate_train_config(config)
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
    validation_plan = make_fixed_validation_plan(
        target_dir,
        val_names,
        config.val_tiles_per_image,
        config.seed + 1,
        degrader,
    )
    validation_corrupt = render_fixed_validation(
        validation_plan, degrader, config.eval_batch_size, codec="kornia"
    )
    validation_clean = validation_plan.clean
    validation_digest = hashlib.sha256()
    for name in val_names:
        validation_digest.update(name.encode("utf-8"))
    validation_digest.update(validation_clean.tobytes())
    validation_data_sha256 = validation_digest.hexdigest()
    raw_metrics = tile_metrics(validation_corrupt, validation_clean)
    libjpeg_names = val_names[: min(config.libjpeg_val_images, len(val_names))]
    paired_kornia_corrupt = libjpeg_corrupt = libjpeg_clean = None
    raw_paired_kornia_metrics = None
    raw_libjpeg_metrics = None
    if libjpeg_names:
        codec_plan = make_fixed_validation_plan(
            target_dir,
            libjpeg_names,
            config.val_tiles_per_image,
            config.seed + 2,
            degrader,
        )
        paired_kornia_corrupt = render_fixed_validation(
            codec_plan,
            degrader,
            min(config.eval_batch_size, 512),
            codec="kornia",
        )
        libjpeg_corrupt = render_fixed_validation(
            codec_plan,
            degrader,
            min(config.eval_batch_size, 512),
            codec="pillow",
        )
        libjpeg_clean = codec_plan.clean
        raw_paired_kornia_metrics = tile_metrics(paired_kornia_corrupt, libjpeg_clean)
        raw_libjpeg_metrics = tile_metrics(libjpeg_corrupt, libjpeg_clean)

    model = build_model(config.model).to(device)
    ema = copy.deepcopy(model).to(device).eval()
    init_weights_sha256 = None
    if config.init_weights:
        if config.resume:
            raise ValueError("init_weights and resume are mutually exclusive")
        init_path = Path(config.init_weights)
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        if initial.get("model_name") != config.model:
            raise ValueError("init checkpoint model does not match requested model")
        initial_state = initial.get("ema_state", initial.get("model_state"))
        if not isinstance(initial_state, dict):
            raise ValueError("init checkpoint has no model or EMA state")
        model.load_state_dict(initial_state, strict=True)
        ema.load_state_dict(initial_state, strict=True)
        init_weights_sha256 = hashlib.sha256(init_path.read_bytes()).hexdigest()
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
    warmup_loss = RestorationLoss(
        LossWeights(
            ssim=0.0,
            gradient=config.loss_gradient,
            boundary_extra=config.loss_boundary_extra,
        )
    )
    final_loss = RestorationLoss(
        LossWeights(
            ssim=config.loss_ssim,
            gradient=config.loss_gradient,
            boundary_extra=config.loss_boundary_extra,
        )
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_output = output.with_name(output.stem + "_latest" + output.suffix)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    versions = runtime_versions()
    source_sha256 = source_code_fingerprint()
    device_fingerprint = resolved_device_fingerprint(device)

    start_step = 0
    best_ssim = -math.inf
    history = []
    if config.resume:
        resume_path = Path(config.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_compatibility(
            checkpoint,
            config,
            manifest_sha256,
            store.sha256,
            validation_data_sha256,
            source_sha256,
        )
        if checkpoint.get("resolved_device_fingerprint") != device_fingerprint:
            raise ValueError(
                "resume checkpoint device mismatch: "
                f"saved={checkpoint.get('resolved_device_fingerprint')!r}, "
                f"requested={device_fingerprint!r}"
            )
        saved_versions = checkpoint.get("runtime_versions", {})
        version_drift = {
            key: {"saved": saved_versions.get(key), "current": value}
            for key, value in versions.items()
            if saved_versions.get(key) != value
        }
        if version_drift:
            raise ValueError(f"resume runtime version mismatch: {version_drift}")
        model.load_state_dict(checkpoint["model_state"])
        ema.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_step = int(checkpoint["step"])
        best_ssim = float(checkpoint.get("best_ssim", -math.inf))
        history = list(checkpoint.get("history", []))
        restore_rng_state(checkpoint["rng_state"], rng, device)
        print(
            json.dumps(
                {"event": "resumed", "path": str(resume_path), "step": start_step, "best_ssim": best_ssim},
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "train_start",
                "config": asdict(config),
                "device": str(device),
                "parameters": model_parameter_count(model),
                "manifest_sha256": manifest_sha256,
                "training_data_sha256": store.sha256,
                "validation_data_sha256": validation_data_sha256,
                "source_code_sha256": source_sha256,
                "resolved_device_fingerprint": device_fingerprint,
                "runtime_versions": versions,
                "raw_validation": raw_metrics,
                "raw_paired_kornia_validation": raw_paired_kornia_metrics,
                "raw_libjpeg_validation": raw_libjpeg_metrics,
                "init_weights_sha256": init_weights_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.perf_counter()
    running = {}
    ssim_start = int(config.steps * config.ssim_start_fraction)

    for step in range(start_step + 1, config.steps + 1):
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
            if libjpeg_corrupt is not None and libjpeg_clean is not None:
                paired_kornia_metrics = evaluate_model(
                    ema,
                    paired_kornia_corrupt,
                    libjpeg_clean,
                    device,
                    min(config.eval_batch_size, 256),
                    complete_images=config.val_tiles_per_image == 576,
                )
                libjpeg_metrics = evaluate_model(
                    ema,
                    libjpeg_corrupt,
                    libjpeg_clean,
                    device,
                    min(config.eval_batch_size, 256),
                    complete_images=config.val_tiles_per_image == 576,
                )
                metrics.update(
                    {f"paired_kornia_{key}": value for key, value in paired_kornia_metrics.items()}
                )
                metrics.update({f"libjpeg_{key}": value for key, value in libjpeg_metrics.items()})
            record = {"step": step, **metrics}
            history.append(record)
            print(json.dumps({"event": "validation", **record}, sort_keys=True), flush=True)
            improved = metrics["tile_ssim"] > best_ssim
            if improved:
                best_ssim = metrics["tile_ssim"]
            checkpoint = {
                "schema_version": 2,
                "model_name": config.model,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng_state": capture_rng_state(rng, device),
                "step": step,
                "best_ssim": best_ssim,
                "config": asdict(config),
                "resolved_device": str(device),
                "resolved_device_fingerprint": device_fingerprint,
                "runtime_versions": versions,
                "manifest_sha256": manifest_sha256,
                "training_data_sha256": store.sha256,
                "validation_data_sha256": validation_data_sha256,
                "source_code_sha256": source_sha256,
                "raw_validation": raw_metrics,
                "raw_paired_kornia_validation": raw_paired_kornia_metrics,
                "raw_libjpeg_validation": raw_libjpeg_metrics,
                "init_weights_sha256": init_weights_sha256,
                "history": history,
                "latest_validation": record,
            }
            atomic_torch_save(checkpoint, latest_output)
            print(json.dumps({"event": "latest_checkpoint_saved", "path": str(latest_output), "step": step}), flush=True)
            if improved:
                checkpoint["best_validation"] = record
                atomic_torch_save(checkpoint, output)
                print(json.dumps({"event": "checkpoint_saved", "path": str(output), **record}), flush=True)

    return {
        "output": str(output),
        "raw_validation": raw_metrics,
        "best_ssim": best_ssim,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
