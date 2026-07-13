"""Small boundary-continuation network for 20x20 puzzle tiles.

``ContinuationNet0`` consumes a canonicalized raw/restored tile pair encoded
as six channels.  It keeps the native tile resolution throughout the trunk,
predicts the first four RGB columns beyond the right edge, and reconstructs
the complete clean query tile as an auxiliary output.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


SCHEMA_VERSION = 1
CHECKPOINT_KIND = "puzzle_assembly_continuation_net0_checkpoint"
MODEL_NAME = "continuation_net0"
TILE_SIZE = 20
INPUT_CHANNELS = 6
WIDTH = 48
GROUPS = 8
CONTINUATION_WIDTH = 4
COLLAPSE_WIDTH = 8
HORIZONTAL_DILATIONS = (1, 1, 2, 2, 4, 1)


class _HorizontalResidualBlock(nn.Module):
    """Full-resolution residual block with horizontal-only dilation."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        padding = (1, int(dilation))
        dilation_2d = (1, int(dilation))
        self.first = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=padding,
            dilation=dilation_2d,
            bias=False,
        )
        self.first_norm = nn.GroupNorm(GROUPS, width)
        self.second = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=padding,
            dilation=dilation_2d,
            bias=False,
        )
        self.second_norm = nn.GroupNorm(GROUPS, width)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.activation(self.first_norm(self.first(values)))
        values = self.second_norm(self.second(values))
        return self.activation(residual + values)


class ContinuationNet0(nn.Module):
    """Predict a clean four-column continuation and reconstruct the query.

    The caller is responsible for rotating/flipping a directional query into
    the canonical form in which the requested neighbour lies to the right.
    Inputs must be finite floating-point tensors with shape ``(N, 6, 20, 20)``.
    Both returned images are sigmoid probabilities in ``[0, 1]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, WIDTH, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, WIDTH),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                _HorizontalResidualBlock(WIDTH, dilation)
                for dilation in HORIZONTAL_DILATIONS
            ]
        )
        # Applied to the rightmost eight feature columns.  The 1x8 kernel is a
        # learned collapse rather than a fixed average or a pooling operation.
        self.continuation_collapse = nn.Sequential(
            nn.Conv2d(
                WIDTH,
                WIDTH,
                kernel_size=(1, COLLAPSE_WIDTH),
                bias=False,
            ),
            nn.GroupNorm(GROUPS, WIDTH),
            nn.SiLU(),
        )
        self.continuation_projection = nn.Conv2d(
            WIDTH, 3 * CONTINUATION_WIDTH, kernel_size=1
        )
        self.reconstruction_projection = nn.Conv2d(WIDTH, 3, kernel_size=1)

    @staticmethod
    def config() -> dict[str, Any]:
        """Return the exact serializable architecture contract."""

        return {
            "model_name": MODEL_NAME,
            "tile_size": TILE_SIZE,
            "input_channels": INPUT_CHANNELS,
            "width": WIDTH,
            "group_norm_groups": GROUPS,
            "horizontal_dilations": list(HORIZONTAL_DILATIONS),
            "collapse_width": COLLAPSE_WIDTH,
            "continuation_width": CONTINUATION_WIDTH,
            "residual_blocks": len(HORIZONTAL_DILATIONS),
            "pooling": False,
        }

    @staticmethod
    def _validate_input(values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        expected = (INPUT_CHANNELS, TILE_SIZE, TILE_SIZE)
        if values.ndim != 4 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"values must have shape (N,{expected[0]},{expected[1]},{expected[2]})")
        if values.shape[0] < 1:
            raise ValueError("values batch must be non-empty")
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_input(values)
        features = self.blocks(self.input(values))

        right_band = features[..., -COLLAPSE_WIDTH:]
        collapsed = self.continuation_collapse(right_band)
        continuation_logits = self.continuation_projection(collapsed)
        batch = int(values.shape[0])
        continuation = continuation_logits.reshape(
            batch, 3, CONTINUATION_WIDTH, TILE_SIZE
        ).permute(0, 1, 3, 2)
        reconstruction = self.reconstruction_projection(features)
        return {
            "continuation": continuation.sigmoid(),
            "reconstruction": reconstruction.sigmoid(),
        }


def _validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "model_config",
        "model_state",
        "metadata",
        "optimizer_state",
        "training_state",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("ContinuationNet0 checkpoint schema mismatch")
    if payload["schema_version"] != SCHEMA_VERSION or payload["kind"] != CHECKPOINT_KIND:
        raise RuntimeError("ContinuationNet0 checkpoint identity mismatch")
    if payload["model_config"] != ContinuationNet0.config():
        raise RuntimeError("ContinuationNet0 checkpoint config mismatch")
    if not isinstance(payload["model_state"], Mapping):
        raise RuntimeError("ContinuationNet0 checkpoint model_state must be a mapping")
    if not isinstance(payload["metadata"], dict):
        raise RuntimeError("ContinuationNet0 checkpoint metadata must be a dict")
    if payload["metadata"].get("safe_for_submission") is not False:
        raise RuntimeError("ContinuationNet0 checkpoint is missing fail-closed metadata")
    for field in ("optimizer_state", "training_state"):
        value = payload[field]
        if value is not None and not isinstance(value, dict):
            raise RuntimeError(f"ContinuationNet0 checkpoint {field} must be dict or None")
    return payload


def save_continuation_net0_checkpoint(
    path: str | Path,
    model: ContinuationNet0,
    *,
    metadata: Mapping[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save a strictly versioned ContinuationNet0 checkpoint."""

    if not isinstance(model, ContinuationNet0):
        raise TypeError("model must be ContinuationNet0")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    metadata_value = dict(metadata or {})
    if metadata_value.get("safe_for_submission") not in (None, False):
        raise ValueError("safe_for_submission cannot be true")
    metadata_value["safe_for_submission"] = False
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "model_config": model.config(),
        "model_state": model.state_dict(),
        "metadata": metadata_value,
        "optimizer_state": optimizer_state,
        "training_state": training_state,
    }
    _validate_checkpoint_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_continuation_net0_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate the exact checkpoint envelope."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return _validate_checkpoint_payload(payload)


def load_continuation_net0_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ContinuationNet0, dict[str, Any]]:
    """Restore a ContinuationNet0 with strict state-dict validation."""

    payload = load_continuation_net0_checkpoint_payload(
        path, map_location=map_location
    )
    model = ContinuationNet0()
    model.load_state_dict(payload["model_state"], strict=True)
    return model, dict(payload["metadata"])

