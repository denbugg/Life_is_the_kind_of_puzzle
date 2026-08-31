"""Bounded Pasha883 reranking of SocketMatcher's top candidate edges.

This module is deliberately narrower than the matched full-score diagnostic:
Socket partial OT supplies at most 32 candidates per outgoing socket and the
archived Pasha883 CNN evaluates only those ordered pairs.  A fixed equal-weight
average of within-candidate row-rank percentiles becomes an optional component
edge priority for the unchanged Socket decoder.  The original partial-OT
assignment, dustbin mass, hard projection, border unary and QAP objective are
not replaced.

No target, reference layout, filename, source identifier or absolute position
is accepted by the API.  The dense priority matrices returned for decoder
compatibility contain only bounded sparse evidence: every unscored pair and
every self-pair receives the same finite masked value.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    SocketDecodeResult,
    decode_socket_assignments,
)

DEFAULT_TOP_K = 32
MAX_TOP_K = 32
MASKED_PRIORITY = -1.0


@dataclass(frozen=True)
class PashaSocketTopKRerank:
    """Sparse Pasha evidence and finite decoder priority matrices."""

    right_candidates: np.ndarray
    down_candidates: np.ndarray
    right_pasha_logits: np.ndarray
    down_pasha_logits: np.ndarray
    right_priority: np.ndarray
    down_priority: np.ndarray
    top_k: int
    pair_evaluations: int
    pasha_seconds: float

    def component_edge_priority(self) -> dict[str, np.ndarray]:
        """Return the mapping accepted by ``decode_socket_assignments``."""

        return {"right": self.right_priority, "down": self.down_priority}

    def report(self) -> dict[str, Any]:
        """Return compact, JSON-compatible bounded-compute diagnostics."""

        count = int(self.right_candidates.shape[0])
        full_pair_evaluations = 2 * count * count
        return {
            "method": "pasha883-on-socket-partial-ot-topk-rank50-v1",
            "tile_count": count,
            "top_k": self.top_k,
            "pair_evaluations": self.pair_evaluations,
            "full_directional_pair_evaluations": full_pair_evaluations,
            "pair_evaluation_reduction_factor": (
                full_pair_evaluations / self.pair_evaluations
            ),
            "pasha_seconds": self.pasha_seconds,
            "fusion": "0.5*Socket within-top-k row rank + 0.5*Pasha within-top-k row rank",
            "self_masked": True,
            "unscored_pairs_masked": True,
        }


@dataclass(frozen=True)
class PashaSocketTopKDecode:
    """Unchanged Socket decode plus the bounded priority evidence it consumed."""

    decoder: SocketDecodeResult
    rerank: PashaSocketTopKRerank
    decoder_seconds: float

    def report(self, *, include_layout: bool = False) -> dict[str, Any]:
        return {
            "rerank": self.rerank.report(),
            "decoder_seconds": self.decoder_seconds,
            "decoder": self.decoder.report(include_layout=include_layout),
        }


def _validate_top_k(top_k: int, *, count: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    maximum = min(MAX_TOP_K, count - 1)
    if not 1 <= top_k <= maximum:
        raise ValueError(f"top_k must be in [1, {maximum}]")
    return top_k


def _score_matrix(value: Any, *, name: str, count: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if np.isnan(matrix).any() or np.isposinf(matrix).any():
        raise ValueError(f"{name} contains NaN or positive infinity")
    off_diagonal = ~np.eye(count, dtype=bool)
    if not np.isfinite(matrix[off_diagonal]).all():
        raise ValueError(f"{name} contains a non-finite non-self score")
    return np.ascontiguousarray(matrix)


def select_socket_topk_candidates(
    scores: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> np.ndarray:
    """Select stable high-score candidates per row while excluding self."""

    raw = scores
    if hasattr(raw, "shape") and len(raw.shape) == 2:
        count = int(raw.shape[0])
    else:
        raw = np.asarray(raw)
        if raw.ndim != 2:
            raise ValueError("scores must be a square matrix")
        count = int(raw.shape[0])
    if count < 2:
        raise ValueError("at least two tiles are required")
    matrix = _score_matrix(raw, name="scores", count=count)
    top_k = _validate_top_k(top_k, count=count)
    masked = matrix.copy()
    masked[np.arange(count), np.arange(count)] = -np.inf
    # Full stable sorting is negligible next to CNN inference and makes the
    # candidate contract deterministic, including the rare score-tie case.
    candidates = np.argsort(-masked, axis=1, kind="stable")[:, :top_k]
    if np.any(candidates == np.arange(count)[:, None]):
        raise RuntimeError("self-pair escaped the top-k mask")
    return np.ascontiguousarray(candidates, dtype=np.int32)


def _validate_candidates(
    candidates: Any,
    *,
    count: int,
    top_k: int,
    name: str,
) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.int64)
    if value.shape != (count, top_k):
        raise ValueError(f"{name} must have shape {(count, top_k)}, got {value.shape}")
    if np.any((value < 0) | (value >= count)):
        raise ValueError(f"{name} contains an out-of-range tile")
    if np.any(value == np.arange(count)[:, None]):
        raise ValueError(f"{name} contains a self-pair")
    if np.any(np.sort(value, axis=1)[:, 1:] == np.sort(value, axis=1)[:, :-1]):
        raise ValueError(f"{name} contains duplicate candidates")
    return np.ascontiguousarray(value, dtype=np.int32)


def _row_rank_percentiles(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Stable descending ordinal percentiles with candidate-id tie breaking."""

    rows, width = values.shape
    output = np.empty((rows, width), dtype=np.float32)
    percentiles = np.linspace(1.0, 0.0, width, dtype=np.float32)
    for row in range(rows):
        order = np.lexsort((candidates[row], -values[row]))
        output[row, order] = percentiles
    return output


def fuse_socket_pasha_topk_rank_percentiles(
    socket_scores: Any,
    candidates: Any,
    pasha_logits: Any,
    *,
    masked_priority: float = MASKED_PRIORITY,
) -> np.ndarray:
    """Create a finite sparse 50/50 rank priority matrix for one axis.

    Both models are ranked only within Socket's selected candidate set.  No
    unscored Pasha value is imputed: non-candidates and self stay masked below
    the admitted ``[0, 1]`` priority range.
    """

    raw_candidates = np.asarray(candidates)
    if raw_candidates.ndim != 2:
        raise ValueError("candidates must be a two-dimensional matrix")
    count, top_k = raw_candidates.shape
    _validate_top_k(int(top_k), count=count)
    selected = _validate_candidates(
        raw_candidates,
        count=count,
        top_k=top_k,
        name="candidates",
    )
    socket = _score_matrix(socket_scores, name="socket_scores", count=count)
    pasha = np.asarray(pasha_logits, dtype=np.float32)
    if pasha.shape != (count, top_k) or not np.isfinite(pasha).all():
        raise ValueError(f"pasha_logits must be finite with shape {(count, top_k)}")
    if not np.isfinite(masked_priority) or not masked_priority < 0:
        raise ValueError("masked_priority must be finite and below zero")

    rows = np.arange(count)[:, None]
    socket_selected = socket[rows, selected]
    socket_rank = _row_rank_percentiles(socket_selected, selected)
    pasha_rank = _row_rank_percentiles(pasha, selected)
    priority = np.full((count, count), masked_priority, dtype=np.float32)
    priority[rows, selected] = 0.5 * socket_rank + 0.5 * pasha_rank
    priority[np.arange(count), np.arange(count)] = masked_priority
    return np.ascontiguousarray(priority)


def _validate_tiles(tiles: Any) -> np.ndarray:
    value = np.asarray(tiles)
    if value.ndim != 4 or value.shape[1:] != (20, 20, 3) or value.dtype != np.uint8:
        raise ValueError(
            "tiles must be uint8 with shape (tile_count, 20, 20, 3), "
            f"got {value.dtype} {value.shape}"
        )
    if len(value) < 2:
        raise ValueError("at least two tiles are required")
    return np.ascontiguousarray(value)


@torch.inference_mode()
def _score_selected_pairs(
    model: nn.Module,
    features: torch.Tensor,
    candidates: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    count, top_k = candidates.shape
    total = count * top_k
    sources = torch.arange(count, device=features.device).repeat_interleave(top_k)
    targets = torch.from_numpy(candidates.reshape(-1).astype(np.int64)).to(features.device)
    output = np.empty(total, dtype=np.float32)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        pair = torch.cat(
            (features[sources[start:stop]], features[targets[start:stop]]),
            dim=-1,
        )
        logits = model(pair)
        if logits.shape != (stop - start,):
            raise ValueError(
                "Pasha model must return one logit per ordered pair, "
                f"got {tuple(logits.shape)}"
            )
        value = logits.float().cpu().numpy()
        if not np.isfinite(value).all():
            raise ValueError("Pasha model returned a non-finite logit")
        output[start:stop] = value
    return output.reshape(count, top_k)


@torch.inference_mode()
def rerank_socket_topk_with_pasha(
    model: nn.Module,
    tiles: Any,
    socket_right_scores: Any,
    socket_down_scores: Any,
    *,
    device: torch.device,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 2048,
) -> PashaSocketTopKRerank:
    """Evaluate Pasha on exactly ``2 * tile_count * top_k`` candidate pairs."""

    source = _validate_tiles(tiles)
    count = len(source)
    top_k = _validate_top_k(top_k, count=count)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if model.training:
        raise ValueError("Pasha model must be in eval mode")
    right = _score_matrix(socket_right_scores, name="socket_right_scores", count=count)
    down = _score_matrix(socket_down_scores, name="socket_down_scores", count=count)
    right_candidates = select_socket_topk_candidates(right, top_k=top_k)
    down_candidates = select_socket_topk_candidates(down, top_k=top_k)
    tensor = (
        torch.from_numpy(source)
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )

    started = perf_counter()
    right_logits = _score_selected_pairs(
        model,
        tensor,
        right_candidates,
        batch_size=batch_size,
    )
    # Exact historical vertical contract: transpose each square tile first,
    # then concatenate the ordered pair into a 20x40 crop.
    down_logits = _score_selected_pairs(
        model,
        tensor.transpose(-1, -2),
        down_candidates,
        batch_size=batch_size,
    )
    elapsed = perf_counter() - started
    right_priority = fuse_socket_pasha_topk_rank_percentiles(
        right,
        right_candidates,
        right_logits,
    )
    down_priority = fuse_socket_pasha_topk_rank_percentiles(
        down,
        down_candidates,
        down_logits,
    )
    return PashaSocketTopKRerank(
        right_candidates=right_candidates,
        down_candidates=down_candidates,
        right_pasha_logits=np.ascontiguousarray(right_logits),
        down_pasha_logits=np.ascontiguousarray(down_logits),
        right_priority=right_priority,
        down_priority=down_priority,
        top_k=top_k,
        pair_evaluations=2 * count * top_k,
        pasha_seconds=elapsed,
    )


def _assignment_real_block(value: Any, *, count: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    matrix = np.asarray(value)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.shape != (count + 1, count + 1):
        raise ValueError(
            f"{name} must have shape {(count + 1, count + 1)}, got {matrix.shape}"
        )
    return _score_matrix(matrix[:count, :count], name=f"{name} real block", count=count)


def decode_socket_with_pasha_topk_priority(
    model: nn.Module,
    tiles: Any,
    right_log_assignment: Any,
    down_log_assignment: Any,
    *,
    device: torch.device,
    grid: int = 24,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 2048,
    config: SocketDecoderConfig | None = None,
) -> PashaSocketTopKDecode:
    """Rerank bounded candidates and reprioritise only decoder components."""

    source = _validate_tiles(tiles)
    count = len(source)
    if grid * grid != count:
        raise ValueError(f"tile count {count} is not grid**2 for grid={grid}")
    right_real = _assignment_real_block(
        right_log_assignment,
        count=count,
        name="right_log_assignment",
    )
    down_real = _assignment_real_block(
        down_log_assignment,
        count=count,
        name="down_log_assignment",
    )
    rerank = rerank_socket_topk_with_pasha(
        model,
        source,
        right_real,
        down_real,
        device=device,
        top_k=top_k,
        batch_size=batch_size,
    )
    started = perf_counter()
    decoded = decode_socket_assignments(
        right_log_assignment,
        down_log_assignment,
        grid=grid,
        config=config,
        component_edge_priority=rerank.component_edge_priority(),
    )
    return PashaSocketTopKDecode(
        decoder=decoded,
        rerank=rerank,
        decoder_seconds=perf_counter() - started,
    )


__all__ = [
    "DEFAULT_TOP_K",
    "MASKED_PRIORITY",
    "MAX_TOP_K",
    "PashaSocketTopKDecode",
    "PashaSocketTopKRerank",
    "decode_socket_with_pasha_topk_priority",
    "fuse_socket_pasha_topk_rank_percentiles",
    "rerank_socket_topk_with_pasha",
    "select_socket_topk_candidates",
]
