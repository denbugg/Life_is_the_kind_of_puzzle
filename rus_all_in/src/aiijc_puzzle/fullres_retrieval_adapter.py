"""No-downsample tile adapter trained for frozen Socket neighbour retrieval.

The adapter is deliberately matcher-only.  It maps each upright dirty 20x20
tile to a bounded auxiliary view while every future layout/output must still
use the original raw tile.  Unlike the earlier boundary-pixel denoiser, the
primary objective back-propagates exact-neighbour ranking through a frozen d64
SocketMatcher.  A weak boundary reconstruction term prevents unconstrained
score-space adversarial residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    boundary_denoising_loss,
    model_config_dict,
)
from aiijc_puzzle.socket_matcher import SocketMatcher, socket_matching_loss


@dataclass(frozen=True)
class RetrievalAdapterConfig:
    """Frozen architecture and loss contract for the bounded pilot."""

    width: int = 32
    blocks: int = 8
    residual_limit: float = 32.0 / 255.0
    border_width: int = 6
    boundary_guard_weight: float = 0.25
    socket_border_weight: float = 0.10
    socket_raw_rank_weight: float = 0.25

    def validate(self) -> None:
        FullResolutionDenoiserConfig(
            width=self.width,
            blocks=self.blocks,
            residual_limit=self.residual_limit,
        ).validate()
        if not 1 <= self.border_width <= 10:
            raise ValueError("border_width must be in [1, 10]")
        for name, value in (
            ("boundary_guard_weight", self.boundary_guard_weight),
            ("socket_border_weight", self.socket_border_weight),
            ("socket_raw_rank_weight", self.socket_raw_rank_weight),
        ):
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


class FullResolutionRetrievalAdapter(FullResolutionBoundaryDenoiser):
    """Zero-initialised stride-one NAF adapter with a retrieval contract."""

    def __init__(self, config: RetrievalAdapterConfig | None = None) -> None:
        if config is None:
            config = RetrievalAdapterConfig()
        config.validate()
        self.retrieval_config = config
        super().__init__(
            FullResolutionDenoiserConfig(
                width=config.width,
                blocks=config.blocks,
                residual_limit=config.residual_limit,
            )
        )


@dataclass(frozen=True)
class RetrievalAdapterLoss:
    """Differentiable total and detached diagnostics for one exact board."""

    total: torch.Tensor
    socket: torch.Tensor
    boundary: torch.Tensor
    socket_diagnostics: dict[str, float]
    boundary_diagnostics: dict[str, float]


def retrieval_adapter_loss(
    adapter: FullResolutionRetrievalAdapter,
    frozen_socket: SocketMatcher,
    dirty_tiles: torch.Tensor,
    clean_tiles: torch.Tensor,
    tile_at_position: torch.Tensor,
    *,
    grid: int = 24,
) -> RetrievalAdapterLoss:
    """Train one adapted board with exact bidirectional Socket supervision."""

    if not isinstance(adapter, FullResolutionRetrievalAdapter):
        raise TypeError("adapter must be a FullResolutionRetrievalAdapter")
    if not isinstance(frozen_socket, SocketMatcher):
        raise TypeError("frozen_socket must be a SocketMatcher")
    count = grid * grid
    expected = (count, 3, 20, 20)
    if dirty_tiles.shape != expected or clean_tiles.shape != expected:
        raise ValueError(f"dirty_tiles and clean_tiles must have shape {expected}")
    if tile_at_position.shape != (1, count):
        raise ValueError(f"tile_at_position must have shape {(1, count)}")
    if any(parameter.requires_grad for parameter in frozen_socket.parameters()):
        raise ValueError("SocketMatcher parameters must be frozen")
    if frozen_socket.training:
        raise ValueError("SocketMatcher must stay in eval mode")

    adapted = adapter(dirty_tiles)
    socket_output = frozen_socket(adapted.unsqueeze(0), grid=grid)
    socket_loss, socket_diagnostics = socket_matching_loss(
        socket_output,
        tile_at_position,
        grid=grid,
        border_weight=adapter.retrieval_config.socket_border_weight,
        raw_rank_weight=adapter.retrieval_config.socket_raw_rank_weight,
    )
    boundary_loss, boundary_terms = boundary_denoising_loss(
        adapted,
        clean_tiles,
        dirty_tiles,
        border_width=adapter.retrieval_config.border_width,
    )
    total = (
        socket_loss
        + adapter.retrieval_config.boundary_guard_weight * boundary_loss
    )
    return RetrievalAdapterLoss(
        total=total,
        socket=socket_loss,
        boundary=boundary_loss,
        socket_diagnostics=socket_diagnostics,
        boundary_diagnostics={
            name: float(value.detach()) for name, value in boundary_terms.items()
        },
    )


def retrieval_adapter_contract(
    adapter: FullResolutionRetrievalAdapter,
) -> dict[str, Any]:
    """Return the portable matcher-only architecture/loss contract."""

    config = adapter.retrieval_config
    return {
        "architecture": "fullres-20x20-naf-frozen-socket-retrieval-adapter-v1",
        "model_config": model_config_dict(adapter),
        "retrieval_config": {
            "border_width": config.border_width,
            "boundary_guard_weight": config.boundary_guard_weight,
            "socket_border_weight": config.socket_border_weight,
            "socket_raw_rank_weight": config.socket_raw_rank_weight,
        },
        "spatial_downsampling": False,
        "feature_resolution_every_block": [20, 20],
        "socket_parameters_frozen": True,
        "socket_mode": "eval",
        "raw_parallel_control_required": True,
        "matcher_view_only": True,
        "restored_pixels_are_output_material": False,
    }


__all__ = [
    "FullResolutionRetrievalAdapter",
    "RetrievalAdapterConfig",
    "RetrievalAdapterLoss",
    "retrieval_adapter_contract",
    "retrieval_adapter_loss",
]
