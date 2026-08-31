"""Target-free structural border prior from the historical TASKA matcher.

The prior asks which tiles have no plausible predecessor or successor along
each axis.  It does so with the slack row and column of the matcher's
doubly-stochastic projection.  Only the current dirty tile bag and frozen
matcher logits enter this module; it has no API for targets, filenames,
absolute tile ids, or a content-border model.

``right_down_logits`` is intentionally the only matcher interface required
here.  The TASKA matcher owns descriptor extraction and learned logit scaling,
so this module does not duplicate or reinterpret its trunk.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch

SIDES = ("top", "bottom", "left", "right")
_TILE_SIZE = 20
_MASKED_SELF_LOGIT = -1.0e4


class TaskaMatcherLogitProvider(Protocol):
    """Minimal public interface needed from a frozen TASKA seam matcher.

    The returned tensors are scaled, high-is-good raw logits, before diagonal
    masking or Sinkhorn normalisation.  Their tile axes must follow the input
    tensor's tile axis exactly.
    """

    def right_down_logits(
        self,
        tiles_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _validated_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be CPU, CUDA, or MPS")
    return device


def _tile_tensor(
    value: np.ndarray | torch.Tensor,
    *,
    grid: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    count = grid * grid
    if isinstance(value, torch.Tensor):
        tiles = value.detach()
        if tiles.shape != (count, _TILE_SIZE, _TILE_SIZE, 3):
            raise ValueError(
                "tiles must have shape "
                f"{(count, _TILE_SIZE, _TILE_SIZE, 3)}, got {tuple(tiles.shape)}"
            )
        if tiles.dtype == torch.bool or not (
            tiles.dtype.is_floating_point
            or tiles.dtype
            in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        ):
            raise TypeError("tiles must have a real numeric dtype")
        tensor = tiles.to(device=device, dtype=torch.float32)
    else:
        tiles_np = np.asarray(value)
        if tiles_np.shape != (count, _TILE_SIZE, _TILE_SIZE, 3):
            raise ValueError(
                "tiles must have shape "
                f"{(count, _TILE_SIZE, _TILE_SIZE, 3)}, got {tiles_np.shape}"
            )
        if tiles_np.dtype.kind not in "iuf":
            raise TypeError("tiles must have a real numeric dtype")
        tensor = torch.from_numpy(np.ascontiguousarray(tiles_np)).to(
            device=device,
            dtype=torch.float32,
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("tiles must contain only finite values")
    return tensor.permute(0, 3, 1, 2).contiguous()


def _validated_logits(
    value: torch.Tensor,
    *,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {tuple(value.shape)}")
    logits = value.detach().to(device=device, dtype=torch.float32).clone()
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    logits.fill_diagonal_(_MASKED_SELF_LOGIT)
    return logits


def _slack_sinkhorn_border_mass(
    logits: torch.Tensor,
    *,
    slack: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return outgoing and incoming mass absorbed by the slack states."""

    if not np.isfinite(slack) or slack <= 0:
        raise ValueError("slack must be finite and positive")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("sinkhorn_iterations must be a positive integer")
    count = logits.shape[0]
    augmented = torch.zeros(
        (count + 1, count + 1),
        device=logits.device,
        dtype=logits.dtype,
    )
    augmented[:count, :count] = logits
    augmented[count, count] = _MASKED_SELF_LOGIT
    row_mass = torch.ones(count + 1, device=logits.device, dtype=logits.dtype)
    column_mass = torch.ones(count + 1, device=logits.device, dtype=logits.dtype)
    row_mass[count] = float(slack)
    column_mass[count] = float(slack)
    log_row_mass = row_mass.log()
    log_column_mass = column_mass.log()
    for _ in range(iterations):
        augmented = (
            augmented
            - torch.logsumexp(augmented, dim=1, keepdim=True)
            + log_row_mass[:, None]
        )
        augmented = (
            augmented
            - torch.logsumexp(augmented, dim=0, keepdim=True)
            + log_column_mass[None, :]
        )
    outgoing = augmented[:count, count].exp()
    incoming = augmented[count, :count].exp()
    return outgoing, incoming


def structural_border_scores(
    models: Sequence[TaskaMatcherLogitProvider],
    tiles: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    grid: int = 24,
    slack: float = 6.0,
    sinkhorn_iterations: int = 20,
) -> dict[str, np.ndarray]:
    """Infer per-tile evidence for the four board sides.

    This exactly follows the structural half of historical ``border_prior.py``:
    mask self-links, run slack Sinkhorn independently on raw right/down logits,
    map outgoing/incoming mass to the appropriate sides, and average each side
    across matchers.  The frozen v3 and local matchers are the intended pair.
    """

    providers = tuple(models)
    if not providers:
        raise ValueError("models must contain at least one TASKA matcher")
    if any(not callable(getattr(model, "right_down_logits", None)) for model in providers):
        raise TypeError("every model must implement right_down_logits(tiles_tensor)")
    resolved_device = _validated_device(device)
    tensor = _tile_tensor(tiles, grid=grid, device=resolved_device)
    count = grid * grid
    # Historical code accumulated the float32 matcher outputs in NumPy float64.
    # Keeping these four small vectors on CPU also avoids unsupported float64
    # tensors on MPS without changing the matcher or Sinkhorn device.
    accumulators = {side: np.zeros(count, dtype=np.float64) for side in SIDES}
    with torch.inference_mode():
        for index, model in enumerate(providers):
            pair = model.right_down_logits(tensor)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    f"models[{index}].right_down_logits must return a two-tensor tuple"
                )
            right = _validated_logits(
                pair[0],
                count=count,
                device=resolved_device,
                name=f"models[{index}] right logits",
            )
            down = _validated_logits(
                pair[1],
                count=count,
                device=resolved_device,
                name=f"models[{index}] down logits",
            )
            right_out, right_in = _slack_sinkhorn_border_mass(
                right,
                slack=slack,
                iterations=sinkhorn_iterations,
            )
            down_out, down_in = _slack_sinkhorn_border_mass(
                down,
                slack=slack,
                iterations=sinkhorn_iterations,
            )
            accumulators["right"] += right_out.cpu().numpy().astype(np.float64)
            accumulators["left"] += right_in.cpu().numpy().astype(np.float64)
            accumulators["bottom"] += down_out.cpu().numpy().astype(np.float64)
            accumulators["top"] += down_in.cpu().numpy().astype(np.float64)

    divisor = float(len(providers))
    result = {
        side: np.ascontiguousarray(accumulators[side] / divisor, dtype=np.float64)
        for side in SIDES
    }
    if any(values.shape != (count,) or not np.isfinite(values).all() for values in result.values()):
        raise RuntimeError("structural border inference produced invalid side scores")
    return result


def structural_border_unary(
    models: Sequence[TaskaMatcherLogitProvider],
    tiles: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    grid: int = 24,
    slack: float = 6.0,
    sinkhorn_iterations: int = 20,
) -> np.ndarray:
    """Return the historical border-only ``(tile, row, column)`` bonus."""

    scores = structural_border_scores(
        models,
        tiles,
        device=device,
        grid=grid,
        slack=slack,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    count = grid * grid
    standardised: dict[str, np.ndarray] = {}
    for side in SIDES:
        values = scores[side]
        standardised[side] = (values - values.mean()) / (values.std() + 1.0e-9)

    unary = np.zeros((count, grid, grid), dtype=np.float64)
    unary[:, 0, :] += standardised["top"][:, None]
    unary[:, grid - 1, :] += standardised["bottom"][:, None]
    unary[:, :, 0] += standardised["left"][:, None]
    unary[:, :, grid - 1] += standardised["right"][:, None]
    if not np.isfinite(unary).all():
        raise RuntimeError("structural border unary contains non-finite values")
    return np.ascontiguousarray(unary)


__all__ = [
    "SIDES",
    "TaskaMatcherLogitProvider",
    "structural_border_scores",
    "structural_border_unary",
]
