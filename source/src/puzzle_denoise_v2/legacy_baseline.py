"""Strict, isolated loader for the pre-v2 ``TileRestorer`` baseline.

The legacy checkpoint predates the versioned v2 checkpoint format.  This
module intentionally keeps its architecture and loading contract separate
from the current restorers so that a baseline evaluation cannot silently load
the wrong model or reinterpret a partially compatible state dictionary.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn


LEGACY_TILE_RESTORER_1024_Q90_FILENAME = "tile_restorer_1024_q90.pt"
LEGACY_TILE_RESTORER_1024_Q90_SHA256 = (
    "d1df5a4e4852c821d79f72063866cf1fe09fb1beff913a4fb1034466d6ead96e"
)
LEGACY_WIDTH = 64
LEGACY_DEPTH = 8
LEGACY_GRID = 24
LEGACY_TILE = 20

_CHECKPOINT_KEYS = frozenset(
    {
        "model_state",
        "width",
        "depth",
        "grid",
        "tile",
        "args",
        "history",
    }
)


class ResBlock(nn.Module):
    """Exact residual block used by ``scripts/denoise_tiles.py``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.2


class TileRestorer(nn.Module):
    """Exact legacy residual tile restorer, kept distinct from v2 models."""

    def __init__(self, width: int = LEGACY_WIDTH, depth: int = LEGACY_DEPTH) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        self.head = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.tail(self.body(self.head(x)))
        return (x + residual).clamp(0.0, 1.0)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without reading the file at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"checkpoint path is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_device(device: str | torch.device) -> torch.device:
    """Resolve an implicit CUDA device to the concrete current index."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA prediction device requested, but CUDA is unavailable")
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("expected_sha256 must be a string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return normalized


def _strict_int(checkpoint: Mapping[str, object], key: str, expected: int) -> None:
    value = checkpoint[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"legacy checkpoint {key!r} must be an integer")
    if value != expected:
        raise ValueError(
            f"legacy checkpoint {key!r} mismatch: expected {expected}, got {value}"
        )


def _validate_state_schema(
    state: object,
    expected_state: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError("legacy checkpoint model_state must be a mapping")
    actual_keys = list(state.keys())
    expected_keys = list(expected_state.keys())
    if actual_keys != expected_keys:
        missing = [key for key in expected_keys if key not in state]
        unexpected = [key for key in actual_keys if key not in expected_state]
        raise ValueError(
            "legacy checkpoint state schema mismatch: "
            f"missing={missing}, unexpected={unexpected}, order_matches=False"
        )
    for key, expected_tensor in expected_state.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"legacy checkpoint state {key!r} is not a tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"legacy checkpoint state {key!r} must use strided layout")
        if tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"legacy checkpoint state {key!r} dtype mismatch: "
                f"expected {expected_tensor.dtype}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"legacy checkpoint state {key!r} shape mismatch: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"legacy checkpoint state {key!r} contains non-finite values")
    return state


def load_legacy_tile_restorer(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = LEGACY_TILE_RESTORER_1024_Q90_SHA256,
    expected_width: int = LEGACY_WIDTH,
    expected_depth: int = LEGACY_DEPTH,
    device: str | torch.device = "cpu",
) -> tuple[TileRestorer, torch.device, dict[str, object]]:
    """Load the known legacy model only after provenance and schema checks.

    SHA-256 is checked before deserialization.  ``weights_only=True`` is
    deliberate: this loader never falls back to unrestricted pickle loading.
    The exact top-level keys, geometry metadata, state key order, tensor shapes,
    dtypes, layouts, and finiteness are all required to match.
    """
    expected_sha256 = _validate_sha256(expected_sha256)
    if isinstance(expected_width, bool) or not isinstance(expected_width, int) or expected_width <= 0:
        raise ValueError("expected_width must be a positive integer")
    if isinstance(expected_depth, bool) or not isinstance(expected_depth, int) or expected_depth <= 0:
        raise ValueError("expected_depth must be a positive integer")

    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"checkpoint path is not a regular file: {resolved}")
    payload = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError(
            "legacy checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    # Deserialize the exact bytes that were hashed, avoiding a path-level race
    # between provenance verification and loading.
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict:
        raise ValueError("legacy checkpoint must contain a plain dictionary")
    actual_keys = frozenset(checkpoint)
    if actual_keys != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - actual_keys)
        unexpected = sorted(actual_keys - _CHECKPOINT_KEYS)
        raise ValueError(
            f"legacy checkpoint keys mismatch: missing={missing}, unexpected={unexpected}"
        )

    _strict_int(checkpoint, "width", expected_width)
    _strict_int(checkpoint, "depth", expected_depth)
    _strict_int(checkpoint, "grid", LEGACY_GRID)
    _strict_int(checkpoint, "tile", LEGACY_TILE)
    if type(checkpoint["args"]) is not dict:
        raise ValueError("legacy checkpoint args must be a plain dictionary")
    if type(checkpoint["history"]) is not list:
        raise ValueError("legacy checkpoint history must be a plain list")

    model = TileRestorer(width=expected_width, depth=expected_depth)
    state = _validate_state_schema(checkpoint["model_state"], model.state_dict())
    model.load_state_dict(state, strict=True)
    resolved_device = _canonical_device(device)
    model.to(resolved_device).eval()

    metadata: dict[str, object] = {
        "kind": "legacy_tile_restorer_1024_q90",
        "checkpoint_resolved": str(resolved),
        "checkpoint_sha256": actual_sha256,
        "width": expected_width,
        "depth": expected_depth,
        "grid": LEGACY_GRID,
        "tile": LEGACY_TILE,
        "state_entries": len(state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(resolved_device),
    }
    return model, resolved_device, metadata


def predict_legacy_tiles_uint8(
    model: nn.Module,
    tiles: np.ndarray,
    device: str | torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Restore ``N x 20 x 20 x 3`` tiles with legacy uint8 conversion.

    The final ``+0.5`` followed by an integer cast intentionally reproduces
    ``tensor_to_image`` from ``scripts/denoise_tiles.py`` instead of using
    framework-dependent half-to-even rounding.
    """
    tiles = np.asarray(tiles)
    if tiles.ndim != 4 or tiles.shape[1:] != (LEGACY_TILE, LEGACY_TILE, 3):
        raise ValueError(f"expected Nx20x20x3 tiles, got {tiles.shape}")
    if tiles.dtype != np.uint8:
        raise TypeError(f"expected uint8 tiles, got {tiles.dtype}")
    if len(tiles) == 0:
        raise ValueError("tile array must not be empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    resolved_device = _canonical_device(device)
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.device != resolved_device:
        raise ValueError(
            f"model is on {parameter.device}, but prediction device is {resolved_device}"
        )

    model.eval()
    restored_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            batch = (
                np.ascontiguousarray(
                    tiles[start : start + batch_size].transpose(0, 3, 1, 2)
                ).astype(np.float32)
                / 255.0
            )
            prediction = model(torch.from_numpy(batch).to(resolved_device))
            if not isinstance(prediction, torch.Tensor) or prediction.shape != torch.Size(batch.shape):
                actual_shape = getattr(prediction, "shape", None)
                raise ValueError(
                    "legacy model must return a tensor matching its input shape; "
                    f"got {actual_shape}"
                )
            array = prediction.detach().cpu().numpy()
            array = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
            restored_parts.append(array.transpose(0, 2, 3, 1))
    return np.concatenate(restored_parts, axis=0)


__all__ = [
    "LEGACY_DEPTH",
    "LEGACY_GRID",
    "LEGACY_TILE",
    "LEGACY_TILE_RESTORER_1024_Q90_FILENAME",
    "LEGACY_TILE_RESTORER_1024_Q90_SHA256",
    "LEGACY_WIDTH",
    "ResBlock",
    "TileRestorer",
    "load_legacy_tile_restorer",
    "predict_legacy_tiles_uint8",
    "sha256_file",
]
