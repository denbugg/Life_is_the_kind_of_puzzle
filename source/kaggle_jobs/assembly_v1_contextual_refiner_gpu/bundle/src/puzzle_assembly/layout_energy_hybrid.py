"""Bounded raw-only salvage utilities for a frozen layout-energy critic.

This module intentionally does not train, denoise, assemble a first pass, or
open targets.  It only combines a frozen critic heatmap with exact local seam
deltas around an already frozen HBT/QAP layout.  Target scoring belongs to the
separate evaluation script so predictions can be serialized first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .layout_energy_transformer import LayoutEnergyConfig, LayoutEnergyTransformer


@dataclass(frozen=True)
class DirectionalSeam:
    right: np.ndarray
    down: np.ndarray

    def __post_init__(self) -> None:
        right = np.asarray(self.right, dtype=np.float32)
        down = np.asarray(self.down, dtype=np.float32)
        if right.ndim != 2 or right.shape[0] != right.shape[1]:
            raise ValueError("right seam matrix must be square")
        if down.shape != right.shape:
            raise ValueError("right/down seam matrices must have identical shapes")
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "down", down)


@dataclass(frozen=True, order=True)
class SwapProposal:
    delta: float
    first: int
    second: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "first": int(self.first),
            "second": int(self.second),
            "seam_delta": float(self.delta),
        }


@dataclass(frozen=True)
class CriticScoreBatch:
    energies: np.ndarray
    error_probabilities: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_layout(values: np.ndarray | Sequence[int], *, count: int) -> np.ndarray:
    layout = np.asarray(values)
    if layout.shape != (count,):
        raise ValueError(f"layout must have shape {(count,)}, got {layout.shape}")
    if not np.issubdtype(layout.dtype, np.integer):
        raise TypeError("layout must be integral")
    layout = layout.astype(np.int32, copy=False)
    if not np.array_equal(np.sort(layout), np.arange(count, dtype=np.int32)):
        raise ValueError("layout must be a permutation")
    return layout


def _pairwise_l1(left: np.ndarray, right: np.ndarray, *, chunk_size: int) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32).reshape(len(left), -1)
    right = np.asarray(right, dtype=np.float32).reshape(len(right), -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError("seam feature dimensions differ")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = np.empty((len(left), len(right)), dtype=np.float32)
    for start in range(0, len(left), chunk_size):
        block = left[start : start + chunk_size]
        output[start : start + len(block)] = np.mean(
            np.abs(block[:, None] - right[None]), axis=2, dtype=np.float32
        )
    return output


def raw_border_l1_seam(
    raw_tiles: np.ndarray,
    *,
    strip: int = 2,
    chunk_size: int = 64,
) -> DirectionalSeam:
    """Build the same physical raw RGB border-L1 family used by the pilot."""

    tiles = np.asarray(raw_tiles)
    if tiles.ndim != 4 or tiles.shape[-1] != 3 or tiles.shape[1] != tiles.shape[2]:
        raise ValueError("raw_tiles must have shape NxTxTx3")
    if tiles.dtype != np.uint8:
        raise TypeError("raw_tiles must be uint8")
    if strip <= 0 or strip > tiles.shape[1]:
        raise ValueError("strip must fit within a tile")
    values = tiles.astype(np.float32) / 255.0
    right = _pairwise_l1(
        values[:, :, -strip:, :], values[:, :, :strip, :], chunk_size=chunk_size
    )
    down = _pairwise_l1(
        values[:, -strip:, :, :], values[:, :strip, :, :], chunk_size=chunk_size
    )
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return DirectionalSeam(right=right, down=down)


def _grid_size(count: int) -> int:
    grid = int(round(math.sqrt(count)))
    if grid * grid != count:
        raise ValueError("layout tile count must be a square")
    return grid


def seam_objective(layout: np.ndarray, seam: DirectionalSeam) -> float:
    order = validate_layout(layout, count=len(seam.right))
    grid_size = _grid_size(len(order))
    grid = order.reshape(grid_size, grid_size)
    right = seam.right[grid[:, :-1], grid[:, 1:]]
    down = seam.down[grid[:-1, :], grid[1:, :]]
    values = np.concatenate([right.ravel(), down.ravel()])
    if not bool(np.isfinite(values).all()):
        raise ValueError("layout seam objective contains non-finite values")
    return float(values.sum(dtype=np.float64))


def local_seam_costs(layout: np.ndarray, seam: DirectionalSeam) -> np.ndarray:
    order = validate_layout(layout, count=len(seam.right))
    grid_size = _grid_size(len(order))
    grid = order.reshape(grid_size, grid_size)
    costs = np.zeros(len(order), dtype=np.float64)
    for row in range(grid_size):
        for column in range(grid_size - 1):
            first = row * grid_size + column
            second = first + 1
            value = float(seam.right[grid[row, column], grid[row, column + 1]])
            costs[first] += value
            costs[second] += value
    for row in range(grid_size - 1):
        for column in range(grid_size):
            first = row * grid_size + column
            second = first + grid_size
            value = float(seam.down[grid[row, column], grid[row + 1, column]])
            costs[first] += value
            costs[second] += value
    if not bool(np.isfinite(costs).all()):
        raise ValueError("local seam costs contain non-finite values")
    return costs


def _incident_edges(position: int, grid_size: int) -> set[tuple[str, int, int]]:
    row, column = divmod(int(position), grid_size)
    edges: set[tuple[str, int, int]] = set()
    if column > 0:
        edges.add(("right", position - 1, position))
    if column + 1 < grid_size:
        edges.add(("right", position, position + 1))
    if row > 0:
        edges.add(("down", position - grid_size, position))
    if row + 1 < grid_size:
        edges.add(("down", position, position + grid_size))
    return edges


def _swap_seam_delta_validated(
    order: np.ndarray,
    seam: DirectionalSeam,
    first: int,
    second: int,
    *,
    grid_size: int,
) -> float:
    if not 0 <= first < len(order) or not 0 <= second < len(order):
        raise IndexError("swap positions out of range")
    if first == second:
        return 0.0
    edges = _incident_edges(first, grid_size) | _incident_edges(second, grid_size)

    def after_tile(position: int) -> int:
        if position == first:
            return int(order[second])
        if position == second:
            return int(order[first])
        return int(order[position])

    old = 0.0
    new = 0.0
    for direction, left, right in edges:
        matrix = seam.right if direction == "right" else seam.down
        old += float(matrix[int(order[left]), int(order[right])])
        new += float(matrix[after_tile(left), after_tile(right)])
    return float(new - old)


def swap_seam_delta(
    layout: np.ndarray,
    seam: DirectionalSeam,
    first: int,
    second: int,
) -> float:
    order = validate_layout(layout, count=len(seam.right))
    return _swap_seam_delta_validated(
        order,
        seam,
        first,
        second,
        grid_size=_grid_size(len(order)),
    )


def top_delta_swaps(
    layout: np.ndarray,
    seam: DirectionalSeam,
    suspect_positions: Iterable[int],
    *,
    budget: int,
) -> tuple[SwapProposal, ...]:
    """Return exactly ``budget`` deterministic best seam swaps touching suspects."""

    order = validate_layout(layout, count=len(seam.right))
    if budget <= 0:
        raise ValueError("budget must be positive")
    suspects = sorted({int(value) for value in suspect_positions})
    if not suspects or suspects[0] < 0 or suspects[-1] >= len(order):
        raise ValueError("suspect_positions must be non-empty and in range")
    pairs: set[tuple[int, int]] = set()
    for first in suspects:
        for second in range(len(order)):
            if first == second:
                continue
            pairs.add((min(first, second), max(first, second)))
    if len(pairs) < budget:
        raise ValueError(f"candidate budget {budget} exceeds {len(pairs)} unique swaps")
    grid_size = _grid_size(len(order))
    proposals = [
        SwapProposal(
            delta=_swap_seam_delta_validated(
                order,
                seam,
                first,
                second,
                grid_size=grid_size,
            ),
            first=first,
            second=second,
        )
        for first, second in pairs
    ]
    proposals.sort(key=lambda item: (item.delta, item.first, item.second))
    return tuple(proposals[:budget])


def apply_swap(layout: np.ndarray, proposal: SwapProposal) -> np.ndarray:
    order = validate_layout(layout, count=len(layout)).copy()
    order[proposal.first], order[proposal.second] = (
        order[proposal.second],
        order[proposal.first],
    )
    return order


def seam_select_or_noop(
    layout: np.ndarray,
    proposals: Sequence[SwapProposal],
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, SwapProposal | None]:
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if not proposals:
        raise ValueError("proposals must be non-empty")
    best = min(proposals, key=lambda item: (item.delta, item.first, item.second))
    if best.delta >= -tolerance:
        return np.asarray(layout, dtype=np.int32).copy(), None
    return apply_swap(np.asarray(layout), best), best


def load_failed_frozen_critic(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device | str,
) -> tuple[LayoutEnergyTransformer, dict[str, Any]]:
    """Load only the exact failed v1 critic; never reinterpret it as promoted."""

    path = Path(checkpoint_path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("kind") != "raw_layout_energy_transformer_checkpoint":
        raise ValueError("wrong checkpoint kind")
    if payload.get("safe_for_submission") is not False:
        raise ValueError("critic checkpoint must remain explicitly unsafe")
    if payload.get("development_gate_passed") is not False:
        raise ValueError("this diagnostic accepts only the failed frozen pilot")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint model_config is missing")
    config = LayoutEnergyConfig(**model_config)
    config.validate()
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model_state is missing")
    model = LayoutEnergyTransformer(config)
    model.load_state_dict(model_state, strict=True)
    model.to(torch.device(device)).eval()
    return model, payload


class FrozenCriticScorer:
    """Cache one raw tile encoding and score many position-to-slot layouts."""

    def __init__(
        self,
        model: LayoutEnergyTransformer,
        raw_tiles: np.ndarray,
        *,
        device: torch.device | str,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        tiles = np.asarray(raw_tiles)
        config = model.config
        expected = (config.tile_count, config.tile_size, config.tile_size, 3)
        if tiles.shape != expected or tiles.dtype != np.uint8:
            raise ValueError(f"raw_tiles must be uint8 {expected}")
        tensor = torch.from_numpy(
            np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
        ).float().div_(255.0).unsqueeze(0).to(self.device)
        use_amp = autocast_dtype is not None and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype or torch.float16,
            enabled=use_amp,
        ):
            self.encoded = model.encode_tiles(tensor)[0]

    def score(
        self,
        layouts: np.ndarray | Sequence[np.ndarray],
        *,
        batch_size: int,
    ) -> CriticScoreBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        rows = np.asarray(layouts)
        if rows.ndim == 1:
            rows = rows[None]
        rows = np.stack(
            [validate_layout(row, count=self.model.config.tile_count) for row in rows]
        )
        energies: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        use_amp = self.autocast_dtype is not None and self.device.type == "cuda"
        for start in range(0, len(rows), batch_size):
            block = rows[start : start + batch_size]
            indices = torch.from_numpy(block).to(self.device, dtype=torch.long)
            ordered = self.encoded.unsqueeze(0).expand(len(block), -1, -1).gather(
                1,
                indices.unsqueeze(2).expand(
                    -1, -1, self.model.config.d_model
                ),
            )
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype or torch.float16,
                enabled=use_amp,
            ):
                output = self.model.score_encoded_tiles(ordered)
            energies.append(output.energy.float().cpu().numpy())
            probabilities.append(
                output.local_error_logits.float().sigmoid().cpu().numpy()
            )
        return CriticScoreBatch(
            energies=np.concatenate(energies),
            error_probabilities=np.concatenate(probabilities),
        )


def critic_rerank_or_noop(
    layout: np.ndarray,
    proposals: Sequence[SwapProposal],
    scorer: FrozenCriticScorer,
    *,
    rerank_budget: int,
    batch_size: int,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, SwapProposal | None, np.ndarray]:
    """Rerank the best seam proposals by frozen energy, retaining no-op."""

    if rerank_budget <= 0 or rerank_budget > len(proposals):
        raise ValueError("rerank_budget must fit within proposals")
    shortlist = sorted(
        proposals, key=lambda item: (item.delta, item.first, item.second)
    )[:rerank_budget]
    candidates = [np.asarray(layout, dtype=np.int32)] + [
        apply_swap(np.asarray(layout), proposal) for proposal in shortlist
    ]
    scores = scorer.score(candidates, batch_size=batch_size)
    best_index = int(np.argmin(scores.energies))
    if best_index == 0 or scores.energies[best_index] >= scores.energies[0] - tolerance:
        return candidates[0].copy(), None, scores.energies
    return candidates[best_index], shortlist[best_index - 1], scores.energies


__all__ = [
    "CriticScoreBatch",
    "DirectionalSeam",
    "FrozenCriticScorer",
    "SwapProposal",
    "apply_swap",
    "critic_rerank_or_noop",
    "load_failed_frozen_critic",
    "local_seam_costs",
    "raw_border_l1_seam",
    "seam_objective",
    "seam_select_or_noop",
    "sha256_array",
    "sha256_file",
    "swap_seam_delta",
    "top_delta_swaps",
    "validate_layout",
]
