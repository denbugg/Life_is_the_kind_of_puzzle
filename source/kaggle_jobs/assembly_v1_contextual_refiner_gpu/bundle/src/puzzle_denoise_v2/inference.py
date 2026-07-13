"""Order-preserving inference utilities for shuffled 24x24 tile images."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .training import build_model, choose_device
from .tiles import merge_tiles_numpy, split_tiles_numpy


_PROVENANCE_FIELDS = (
    "schema_version",
    "kind",
    "manifest_sha256",
    "source_code_sha256",
    "fine_tune_code_sha256",
    "init_checkpoint_sha256",
    "legacy_checkpoint_sha256",
    "validation_quarantine_sha256",
    "maps_1024_sha256",
    "train_pairs_sha256",
    "val_pairs_sha256",
    "training_data_sha256",
    "validation_data_sha256",
    "training_pixels_sha256",
    "validation_pixels_sha256",
    "best_ssim",
    "best_real_ssim",
    "promotion_status",
    "safe_for_inference",
    "runtime_versions",
    "resolved_device_fingerprint",
    "source_split",
    "best_validation",
    "gate_validation",
)


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(item) for key, item in value.items())
    return False


def _fine_tune_promotion_issues(checkpoint: dict) -> list[str]:
    kind = checkpoint.get("kind")
    if kind == "conservative_real_pair_fine_tune_rollback":
        issues = []
        if checkpoint.get("rolled_back") is not True:
            issues.append("rollback checkpoint is not explicitly marked rolled_back=true")
        if checkpoint.get("promotion_status") != "rollback_safe":
            issues.append(
                "rollback checkpoint promotion_status is not 'rollback_safe'"
            )
        if checkpoint.get("safe_for_inference") is not True:
            issues.append("rollback checkpoint is not explicitly safe_for_inference=true")
        if checkpoint.get("step") != 0:
            issues.append(f"rollback checkpoint step is {checkpoint.get('step')!r}, not 0")
        return issues
    if kind != "conservative_real_pair_fine_tune":
        return []

    issues = []
    if checkpoint.get("rolled_back") is True:
        issues.append("promoted fine-tune checkpoint is marked rolled_back=true")
    step = checkpoint.get("step")
    best_step = checkpoint.get("best_step")
    valid_steps = (
        isinstance(step, int)
        and not isinstance(step, bool)
        and isinstance(best_step, int)
        and not isinstance(best_step, bool)
    )
    if not valid_steps or step != best_step:
        issues.append(f"fine-tune step {step!r} does not match best_step {best_step!r}")
    best_validation = checkpoint.get("best_validation")
    if not isinstance(best_validation, dict):
        issues.append("fine-tune checkpoint has no best_validation record")
    elif valid_steps and best_validation.get("step") != best_step:
        issues.append(
            "fine-tune best_validation step "
            f"{best_validation.get('step')!r} does not match best_step {best_step!r}"
        )
    gate_validation = checkpoint.get("gate_validation")
    if not isinstance(gate_validation, dict):
        issues.append("fine-tune checkpoint has no frozen gate_validation record")
    else:
        if gate_validation.get("panel") != "frozen_gate":
            issues.append(
                "fine-tune gate_validation panel is "
                f"{gate_validation.get('panel')!r}, not 'frozen_gate'"
            )
        if valid_steps and gate_validation.get("selected_step") != best_step:
            issues.append(
                "fine-tune gate_validation selected_step "
                f"{gate_validation.get('selected_step')!r} does not match best_step {best_step!r}"
            )
        assessment = gate_validation.get("assessment")
        if not isinstance(assessment, dict) or assessment.get("eligible") is not True:
            issues.append("fine-tune frozen gate assessment is not explicitly eligible=true")
    if checkpoint.get("promotion_status") != "promoted":
        issues.append(
            f"fine-tune promotion_status is {checkpoint.get('promotion_status')!r}, not 'promoted'"
        )
    if checkpoint.get("safe_for_inference") is not True:
        issues.append("fine-tune checkpoint is not explicitly marked safe_for_inference=true")
    return issues


def load_restorer(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    state: str = "ema",
    allow_unpromoted: bool = False,
) -> tuple[torch.nn.Module, torch.device, dict]:
    if state not in {"ema", "model"}:
        raise ValueError("state must be 'ema' or 'model'")
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint_resolved = checkpoint_path.resolve(strict=True)
    checkpoint_is_latest = checkpoint_resolved.stem.endswith("_latest")
    if checkpoint_is_latest and not allow_unpromoted:
        raise ValueError(
            "refusing *_latest.pt checkpoint by default; "
            "pass allow_unpromoted=True only for expert debugging"
        )
    checkpoint = torch.load(checkpoint_resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    model_name = checkpoint.get("model_name")
    if model_name not in {"tile-naf", "full-naf"}:
        raise ValueError(f"unsupported checkpoint model {model_name!r}")
    state_key = "ema_state" if state == "ema" else "model_state"
    if state_key not in checkpoint:
        raise ValueError(f"checkpoint does not contain {state_key}")
    promotion_issues = []
    if checkpoint_is_latest:
        promotion_issues.append("checkpoint filename ends with *_latest.pt")
    promotion_issues.extend(_fine_tune_promotion_issues(checkpoint))
    if promotion_issues and not allow_unpromoted:
        raise ValueError(
            "refusing unpromoted fine-tune checkpoint: "
            + "; ".join(promotion_issues)
            + "; pass allow_unpromoted=True only for expert debugging"
        )

    resolved_device = choose_device(device)
    model = build_model(model_name)
    model.load_state_dict(checkpoint[state_key])
    model.to(resolved_device).eval()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_resolved": str(checkpoint_resolved),
        "checkpoint_sha256": hashlib.sha256(checkpoint_resolved.read_bytes()).hexdigest(),
        "checkpoint_is_latest": checkpoint_is_latest,
        "allow_unpromoted": allow_unpromoted,
        "promotion_issues": promotion_issues,
        "model_name": model_name,
        "state": state,
        "device": str(resolved_device),
        "step": checkpoint.get("step"),
        "best_step": checkpoint.get("best_step"),
        "rolled_back": bool(checkpoint.get("rolled_back", False)),
    }
    for key in _PROVENANCE_FIELDS:
        if key in checkpoint and _json_safe(checkpoint[key]):
            metadata[key] = checkpoint[key]
    return model, resolved_device, metadata


def _validate_uint8_tiles(tiles: np.ndarray) -> np.ndarray:
    tiles = np.asarray(tiles)
    if tiles.ndim != 4 or tiles.shape[1:] != (20, 20, 3):
        raise ValueError(f"expected Nx20x20x3 tiles, got {tiles.shape}")
    if tiles.dtype != np.uint8:
        raise TypeError(f"expected uint8 tiles, got {tiles.dtype}")
    if len(tiles) == 0:
        raise ValueError("tile array must not be empty")
    return tiles


@torch.no_grad()
def restore_tiles_uint8(
    model: torch.nn.Module,
    tiles: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    tiles = _validate_uint8_tiles(tiles)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    restored_parts = []
    model.eval()
    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(
            np.ascontiguousarray(tiles[start : start + batch_size].transpose(0, 3, 1, 2))
        ).float().div_(255.0).to(device)
        restored = model(batch)
        restored_parts.append(
            restored.detach()
            .float()
            .cpu()
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .numpy()
        )
    return np.concatenate(restored_parts)


def restore_shuffled_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Restore every tile independently and preserve its exact input slot."""
    image = np.asarray(image)
    if image.shape != (480, 480, 3) or image.dtype != np.uint8:
        raise ValueError("image must be an RGB uint8 array with shape 480x480x3")
    input_tiles = split_tiles_numpy(image)
    output_tiles = restore_tiles_uint8(model, input_tiles, device, batch_size)
    return merge_tiles_numpy(output_tiles)


def restore_png(
    model: torch.nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    batch_size: int = 512,
    *,
    overwrite: bool = False,
) -> None:
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    input_resolved = input_path.resolve(strict=True)
    output_resolved = output_path.resolve(strict=False)
    if input_resolved == output_resolved:
        raise ValueError("input and output PNG paths must be different")
    if output_path.exists() and input_path.samefile(output_path):
        raise ValueError("input and output PNG paths refer to the same file")
    if output_path.is_symlink():
        raise FileExistsError(f"refusing to write through output symlink: {output_path}")
    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(f"output PNG path is a directory: {output_path}")
        if not overwrite:
            raise FileExistsError(f"output PNG already exists; pass overwrite=True: {output_path}")
    with Image.open(input_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    restored = restore_shuffled_image(model, image, device, batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise FileExistsError(f"refusing to write through output symlink: {output_path}")
    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(f"output PNG path became a directory: {output_path}")
        if not overwrite:
            raise FileExistsError(f"output PNG appeared during inference: {output_path}")
    Image.fromarray(restored, mode="RGB").save(output_path, format="PNG")
