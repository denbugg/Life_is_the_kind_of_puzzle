"""Vectorized relation-local verification for raw, adapter and DINO emitters.

The verifier never predicts an absolute position and never changes pixels.  It
reranks one target-blind union of directional neighbour candidates using an
ordered raw seam and ordered frozen DINO boundary tokens.  Scalar emitter
scores/ranks are auxiliary evidence rather than the complete representation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

EMITTERS = ("raw_d64_ot", "adapter_step1600", "dinov2_boundary")
SIDE_NAMES = ("right", "left", "bottom", "top")
TOP_K = 32
DINO_PROJECTION_DIM = 16
DINO_PROJECTION_SEED = 20260912
AUXILIARY_DIM = 19


@dataclass(frozen=True)
class CandidatePool:
    """One padded, target-blind three-emitter candidate union."""

    candidates: np.ndarray
    valid: np.ndarray
    auxiliary: np.ndarray
    raw_baseline: np.ndarray
    emitter_topk: np.ndarray
    identity_digest: str


def fixed_dino_projection(
    input_dim: int,
    *,
    output_dim: int = DINO_PROJECTION_DIM,
    seed: int = DINO_PROJECTION_SEED,
) -> np.ndarray:
    """Return a deterministic orthonormal projection for frozen DINO tokens."""

    if isinstance(input_dim, bool) or input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if isinstance(output_dim, bool) or not 1 <= output_dim <= input_dim:
        raise ValueError("output_dim must be in [1, input_dim]")
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((input_dim, output_dim))
    projection, _ = np.linalg.qr(matrix, mode="reduced")
    result = np.ascontiguousarray(projection, dtype=np.float32)
    result.setflags(write=False)
    return result


def compress_dino_boundary_tokens(
    patch_tokens: Any,
    projection: Any,
    *,
    band_width: int = 2,
) -> np.ndarray:
    """Keep ordered opposing DINO boundary bands after a fixed projection."""

    values = np.asarray(patch_tokens, dtype=np.float32)
    project = np.asarray(projection, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:3] != (7, 7):
        raise ValueError("patch_tokens must have shape N x 7 x 7 x D")
    if project.ndim != 2 or project.shape[0] != values.shape[-1]:
        raise ValueError("projection does not match the DINO token dimension")
    if isinstance(band_width, bool) or not 1 <= band_width <= 7:
        raise ValueError("band_width must be in [1, 7]")
    if not np.isfinite(values).all() or not np.isfinite(project).all():
        raise ValueError("DINO tokens and projection must be finite")

    right = values[:, :, -band_width:, :][:, :, ::-1, :]
    left = values[:, :, :band_width, :]
    bottom = values[:, -band_width:, :, :][:, ::-1, :, :]
    top = values[:, :band_width, :, :]
    sides = (right, left, bottom, top)
    compressed = []
    for side in sides:
        sequence = side.reshape(len(values), -1, values.shape[-1]) @ project
        norm = np.linalg.norm(sequence, axis=-1, keepdims=True)
        compressed.append(sequence / np.maximum(norm, 1e-6))
    result = np.ascontiguousarray(np.stack(compressed), dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("compressed DINO boundary tokens are non-finite")
    return result


def ordered_raw_side_sequences(tiles: Any) -> np.ndarray:
    """Return RGB boundary values and signed inward gradients for four sides."""

    value = np.asarray(tiles)
    if value.ndim != 4 or value.shape[1:] != (20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("tiles must be uint8 N x 20 x 20 x 3")
    scaled = value.astype(np.float32) / 127.5 - 1.0

    def join(boundary: np.ndarray, interior: np.ndarray) -> np.ndarray:
        gradient = 0.5 * (boundary - interior)
        return np.concatenate((boundary, gradient), axis=-1)

    sides = (
        join(scaled[:, :, -1, :], scaled[:, :, -2, :]),
        join(scaled[:, :, 0, :], scaled[:, :, 1, :]),
        join(scaled[:, -1, :, :], scaled[:, -2, :, :]),
        join(scaled[:, 0, :, :], scaled[:, 1, :, :]),
    )
    return np.ascontiguousarray(np.stack(sides), dtype=np.float32)


def _validated_scores(scores: Mapping[str, tuple[Any, Any]]) -> dict[str, np.ndarray]:
    if tuple(scores) != EMITTERS:
        raise ValueError(f"scores must have emitter order {EMITTERS}")
    result: dict[str, np.ndarray] = {}
    count = -1
    for emitter, axes in scores.items():
        if not isinstance(axes, tuple) or len(axes) != 2:
            raise ValueError(f"{emitter} must provide right/down score matrices")
        stacked = np.stack(
            [np.asarray(axis, dtype=np.float32) for axis in axes], axis=0
        )
        if stacked.ndim != 3 or stacked.shape[1] != stacked.shape[2]:
            raise ValueError(f"{emitter} scores must be two aligned square matrices")
        if count < 0:
            count = stacked.shape[1]
        if stacked.shape != (2, count, count) or not np.isfinite(stacked).all():
            raise ValueError("all emitter scores must be aligned and finite")
        result[emitter] = np.ascontiguousarray(stacked)
    return result


def _stable_orders(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(values)
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    outgoing = np.argsort(-work, axis=1, kind="stable")[:, :k]
    incoming = np.argsort(-work, axis=0, kind="stable")[:k]
    outgoing_rank = np.full((count, count), k, dtype=np.int16)
    incoming_rank = np.full((count, count), k, dtype=np.int16)
    rows = np.arange(count)[:, None]
    outgoing_rank[rows, outgoing] = np.arange(k, dtype=np.int16)[None]
    columns = np.arange(count)[None]
    incoming_rank[incoming, columns] = np.arange(k, dtype=np.int16)[:, None]
    return outgoing_rank, incoming_rank


def _score_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(values)
    mask = ~np.eye(count, dtype=bool)
    row = np.where(mask, values, 0.0)
    row_mean = row.sum(axis=1) / (count - 1)
    row_std = np.sqrt(
        (np.where(mask, (values - row_mean[:, None]) ** 2, 0.0).sum(axis=1))
        / (count - 1)
        + 1e-6
    )
    column_mean = row.sum(axis=0) / (count - 1)
    column_std = np.sqrt(
        (
            np.where(mask, (values - column_mean[None]) ** 2, 0.0).sum(axis=0)
            / (count - 1)
        )
        + 1e-6
    )
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    row_top = work.max(axis=1)
    return (
        np.stack((row_mean, row_std, row_top), axis=1),
        np.stack((column_mean, column_std), axis=1),
        work,
    )


def candidate_pool_digest(
    candidates: np.ndarray,
    valid: np.ndarray,
    emitter_topk: np.ndarray,
) -> str:
    """Hash only identities/membership, never learned scores or labels."""

    digest = hashlib.sha256()
    for value in (candidates, valid, emitter_topk):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_candidate_pool(
    scores: Mapping[str, tuple[Any, Any]],
    *,
    top_k: int = TOP_K,
) -> CandidatePool:
    """Build one stable top-k union and dirty-visible scalar auxiliaries."""

    matrices = _validated_scores(scores)
    count = next(iter(matrices.values())).shape[1]
    if isinstance(top_k, bool) or not 1 <= top_k < count:
        raise ValueError("top_k must be in [1, tile_count)")
    max_candidates = len(EMITTERS) * top_k
    candidates = np.full((2, count, max_candidates), -1, dtype=np.int32)
    valid = np.zeros_like(candidates, dtype=bool)
    auxiliary = np.zeros((2, count, max_candidates, AUXILIARY_DIM), dtype=np.float32)
    raw_baseline = np.full((2, count, max_candidates), -1e4, dtype=np.float32)
    emitter_topk = np.empty((len(EMITTERS), 2, count, top_k), dtype=np.int32)

    for axis in range(2):
        orders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        statistics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for emitter_index, emitter in enumerate(EMITTERS):
            matrix = matrices[emitter][axis]
            outgoing_rank, incoming_rank = _stable_orders(matrix, top_k)
            orders[emitter] = (outgoing_rank, incoming_rank)
            statistics[emitter] = _score_statistics(matrix)
            emitter_topk[emitter_index, axis] = np.argsort(
                -statistics[emitter][2], axis=1, kind="stable"
            )[:, :top_k]

        for anchor in range(count):
            union: list[int] = []
            seen: set[int] = set()
            for emitter_index in range(len(EMITTERS)):
                for target in emitter_topk[emitter_index, axis, anchor]:
                    item = int(target)
                    if item not in seen:
                        seen.add(item)
                        union.append(item)
            length = len(union)
            candidates[axis, anchor, :length] = union
            valid[axis, anchor, :length] = True
            for slot, target in enumerate(union):
                columns: list[float] = []
                support = 0
                for emitter in EMITTERS:
                    matrix = matrices[emitter][axis]
                    outgoing_rank, incoming_rank = orders[emitter]
                    row_stats, column_stats, _ = statistics[emitter]
                    member = outgoing_rank[anchor, target] < top_k
                    support += int(member)
                    row_std = float(row_stats[anchor, 1])
                    column_std = float(column_stats[target, 1])
                    score = float(matrix[anchor, target])
                    columns.extend(
                        (
                            float(member),
                            float(outgoing_rank[anchor, target]) / top_k,
                            float(incoming_rank[anchor, target]) / top_k,
                            float(np.clip((score - row_stats[anchor, 0]) / row_std, -8, 8)),
                            float(
                                np.clip(
                                    (score - column_stats[target, 0]) / column_std,
                                    -8,
                                    8,
                                )
                            ),
                            float(
                                np.clip(
                                    (score - row_stats[anchor, 2]) / row_std,
                                    -8,
                                    0,
                                )
                            ),
                        )
                    )
                columns.append(support / len(EMITTERS))
                auxiliary[axis, anchor, slot] = columns
                raw_baseline[axis, anchor, slot] = columns[3]

    # The raw candidates are a hard subset of every row's union.  This is a
    # roster invariant, not a learned gate.
    for axis in range(2):
        for anchor in range(count):
            union = set(candidates[axis, anchor, valid[axis, anchor]].tolist())
            raw = set(emitter_topk[0, axis, anchor].tolist())
            if not raw.issubset(union):
                raise RuntimeError("raw top-k candidates were not preserved")
    digest = candidate_pool_digest(candidates, valid, emitter_topk)
    return CandidatePool(
        candidates=np.ascontiguousarray(candidates),
        valid=np.ascontiguousarray(valid),
        auxiliary=np.ascontiguousarray(auxiliary),
        raw_baseline=np.ascontiguousarray(raw_baseline),
        emitter_topk=np.ascontiguousarray(emitter_topk),
        identity_digest=digest,
    )


class _SequenceEncoder(nn.Module):
    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        groups = math.gcd(width, 4)
        self.network = nn.Sequential(
            nn.Conv1d(channels, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded = self.network(value)
        return torch.cat((encoded.mean(2), encoded.amax(2)), dim=1)


class TriEmitterEdgeVerifier(nn.Module):
    """One vectorized listwise residual scorer over the fixed candidate union."""

    def __init__(
        self,
        *,
        dino_dim: int = DINO_PROJECTION_DIM,
        auxiliary_dim: int = AUXILIARY_DIM,
        width: int = 32,
        hidden: int = 96,
    ) -> None:
        super().__init__()
        if dino_dim <= 0 or auxiliary_dim <= 0 or width <= 0 or hidden <= 0:
            raise ValueError("model dimensions must be positive")
        self.dino_dim = dino_dim
        self.auxiliary_dim = auxiliary_dim
        self.width = width
        self.hidden = hidden
        self.raw_seam = _SequenceEncoder(4 * 6, width)
        self.dino_boundary = _SequenceEncoder(4 * dino_dim, width)
        self.direction = nn.Embedding(2, 8)
        self.auxiliary = nn.Sequential(
            nn.LayerNorm(auxiliary_dim),
            nn.Linear(auxiliary_dim, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(5 * width + 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        raw_sides: torch.Tensor,
        dino_sides: torch.Tensor,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        directions: torch.Tensor,
        auxiliary: torch.Tensor,
        raw_baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score padded query rows without any per-edge Python model call."""

        if candidates.ndim != 2 or valid.shape != candidates.shape:
            raise ValueError("candidates/valid must be aligned B x K tensors")
        batch, candidate_count = candidates.shape
        if anchors.shape != (batch,) or directions.shape != (batch,):
            raise ValueError("anchors/directions must have shape B")
        if auxiliary.shape != (batch, candidate_count, self.auxiliary_dim):
            raise ValueError("auxiliary tensor shape changed")
        if raw_baseline.shape != candidates.shape:
            raise ValueError("raw_baseline shape changed")
        safe_candidates = candidates.clamp_min(0)
        flat_target = safe_candidates.reshape(-1)
        flat_anchor = anchors[:, None].expand(-1, candidate_count).reshape(-1)
        flat_direction = directions[:, None].expand(-1, candidate_count).reshape(-1)
        source_side = 2 * flat_direction
        target_side = source_side + 1

        raw_source = raw_sides[source_side, flat_anchor]
        raw_target = raw_sides[target_side, flat_target]
        raw_pair = torch.cat(
            (raw_source, raw_target, raw_source - raw_target, raw_source * raw_target),
            dim=2,
        ).transpose(1, 2)
        dino_source = dino_sides[source_side, flat_anchor]
        dino_target = dino_sides[target_side, flat_target]
        dino_pair = torch.cat(
            (
                dino_source,
                dino_target,
                dino_source - dino_target,
                dino_source * dino_target,
            ),
            dim=2,
        ).transpose(1, 2)
        relation = torch.cat(
            (
                self.raw_seam(raw_pair),
                self.dino_boundary(dino_pair),
                self.auxiliary(auxiliary.reshape(-1, self.auxiliary_dim)),
                self.direction(flat_direction),
            ),
            dim=1,
        )
        delta = self.head(relation).reshape(batch, candidate_count)
        logits = raw_baseline + delta
        logits = logits.masked_fill(~valid, -1e4)
        return logits, delta.masked_fill(~valid, 0.0)


def sparse_reciprocal_evidence(
    candidates: Any,
    valid: Any,
    logits: Any,
) -> dict[str, np.ndarray]:
    """Compute reciprocal top-one evidence on the scored sparse union."""

    ids = np.asarray(candidates, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    scores = np.asarray(logits, dtype=np.float64)
    if ids.ndim != 2 or mask.shape != ids.shape or scores.shape != ids.shape:
        raise ValueError("candidates, valid and logits must be aligned N x K arrays")
    count = len(ids)
    dense = np.full((count, count), -np.inf, dtype=np.float64)
    for source in range(count):
        dense[source, ids[source, mask[source]]] = scores[source, mask[source]]
    target = np.argmax(dense, axis=1).astype(np.int32)
    row_top_two = np.partition(dense, kth=count - 2, axis=1)[:, -2:]
    row_margin = row_top_two[:, 1] - row_top_two[:, 0]
    incoming = np.argmax(dense, axis=0).astype(np.int32)
    column_top_two = np.partition(dense, kth=count - 2, axis=0)[-2:]
    column_margin = column_top_two[1] - column_top_two[0]
    source = np.arange(count, dtype=np.int32)
    reciprocal = incoming[target] == source
    confidence = np.minimum(row_margin, column_margin[target])
    return {
        "target": target,
        "reciprocal": np.ascontiguousarray(reciprocal),
        "confidence": np.ascontiguousarray(confidence, dtype=np.float32),
    }


def verifier_contract(model: TriEmitterEdgeVerifier) -> dict[str, Any]:
    """Return the fixed portable architecture and compliance contract."""

    return {
        "architecture": "vectorized-tri-emitter-relation-local-verifier-v1",
        "emitters": list(EMITTERS),
        "top_k_per_emitter": TOP_K,
        "raw_top_k_always_preserved_in_union": True,
        "relation_local_content": {
            "raw": "ordered 20-pixel RGB boundary plus inward gradient difference/product",
            "dino": "ordered 14-token two-band fixed-projection difference/product",
            "scalar_auxiliary_only": False,
        },
        "dino_projection_dim": model.dino_dim,
        "auxiliary_dim": model.auxiliary_dim,
        "width": model.width,
        "hidden": model.hidden,
        "absolute_position_or_source_identity": False,
        "pixels_modified": False,
        "output_material": "original upright tile identities only",
    }


__all__ = [
    "AUXILIARY_DIM",
    "CandidatePool",
    "DINO_PROJECTION_DIM",
    "DINO_PROJECTION_SEED",
    "EMITTERS",
    "SIDE_NAMES",
    "TOP_K",
    "TriEmitterEdgeVerifier",
    "build_candidate_pool",
    "candidate_pool_digest",
    "compress_dino_boundary_tokens",
    "fixed_dino_projection",
    "ordered_raw_side_sequences",
    "sparse_reciprocal_evidence",
    "verifier_contract",
]
