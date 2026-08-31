"""Raw-preserving restored-view candidate supply and residual border reranking.

The restored pixels in this module are an auxiliary matcher view only.  The
candidate roster always contains the frozen raw SocketMatcher top-k entries,
and the learned model predicts a residual over the raw SocketMatcher ordering.
No function here assembles or returns model-rendered tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CandidateUnion:
    """One direction's target-blind raw/restored candidate union."""

    rows: tuple[np.ndarray, ...]
    raw_topk: np.ndarray
    restored_topk: np.ndarray
    scalar_features: tuple[np.ndarray, ...]
    baseline_scores: tuple[np.ndarray, ...]


def _validate_square(scores: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or len(value) < 3:
        raise ValueError(f"{label} must be one square matrix of size >= 3")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value


def _rank_percentiles(scores: np.ndarray) -> np.ndarray:
    value = _validate_square(scores, label="scores")
    count = len(value)
    output = np.full_like(value, -1.0)
    indices = np.arange(count)
    percentiles = np.linspace(1.0, 0.0, count - 1, dtype=np.float32)
    for row in range(count):
        candidates = indices[indices != row]
        order = np.argsort(-value[row, candidates], kind="stable")
        output[row, candidates[order]] = percentiles
    return output


def _standardise_rows(scores: np.ndarray) -> np.ndarray:
    value = _validate_square(scores, label="scores")
    count = len(value)
    diagonal = np.eye(count, dtype=bool)
    finite = np.where(diagonal, np.nan, value)
    centre = np.nanmean(finite, axis=1, keepdims=True)
    scale = np.nanstd(finite, axis=1, keepdims=True)
    output = np.clip((value - centre) / np.maximum(scale, 1e-6), -4.0, 4.0)
    output[diagonal] = -4.0
    return np.ascontiguousarray(output, dtype=np.float32)


def _topk(scores: np.ndarray, *, k: int) -> np.ndarray:
    value = _validate_square(scores, label="scores").copy()
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k < len(value):
        raise ValueError("k must be in [1, count - 1]")
    np.fill_diagonal(value, -np.inf)
    order = np.argsort(-value, axis=1, kind="stable")[:, :k]
    return np.ascontiguousarray(order, dtype=np.int32)


def restored_descriptor_scores(restored_tiles: np.ndarray, *, direction: int) -> np.ndarray:
    """Return high-is-good normalised border descriptor similarities."""

    tiles = np.asarray(restored_tiles)
    if tiles.ndim != 4 or tiles.shape[1:] != (20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("restored_tiles must be uint8 N x 20 x 20 x 3")
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (right) or 1 (down)")
    gray = (
        0.299 * tiles[..., 0].astype(np.float32)
        + 0.587 * tiles[..., 1].astype(np.float32)
        + 0.114 * tiles[..., 2].astype(np.float32)
    )
    width = 6
    if direction == 0:
        outgoing = gray[..., -width:].reshape(len(tiles), -1)
        incoming = gray[..., :width].reshape(len(tiles), -1)
    else:
        outgoing = gray[:, -width:, :].reshape(len(tiles), -1)
        incoming = gray[:, :width, :].reshape(len(tiles), -1)

    def normalise(value: np.ndarray) -> np.ndarray:
        value = value - value.mean(axis=1, keepdims=True)
        return value / np.maximum(value.std(axis=1, keepdims=True), 1e-4)

    outgoing = normalise(outgoing)
    incoming = normalise(incoming)
    scores = outgoing @ incoming.T / float(outgoing.shape[1])
    np.fill_diagonal(scores, -4.0)
    return np.ascontiguousarray(scores, dtype=np.float32)


def build_candidate_union(
    raw_scores: np.ndarray,
    restored_scores: np.ndarray,
    *,
    topk: int = 32,
) -> CandidateUnion:
    """Union raw and restored top-k, retaining raw baseline ordering exactly."""

    raw = _validate_square(raw_scores, label="raw_scores")
    restored = _validate_square(restored_scores, label="restored_scores")
    if raw.shape != restored.shape:
        raise ValueError("raw and restored score shapes differ")
    raw_topk = _topk(raw, k=topk)
    restored_topk = _topk(restored, k=topk)
    raw_row_rank = _rank_percentiles(raw)
    restored_row_rank = _rank_percentiles(restored)
    raw_column_rank = _rank_percentiles(raw.T).T
    restored_column_rank = _rank_percentiles(restored.T).T
    raw_z = _standardise_rows(raw)
    restored_z = _standardise_rows(restored)
    rows: list[np.ndarray] = []
    features: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    for anchor in range(len(raw)):
        candidates = np.unique(
            np.concatenate((raw_topk[anchor], restored_topk[anchor]))
        ).astype(np.int32, copy=False)
        candidates = candidates[candidates != anchor]
        # Stable raw-score ordering makes a zero residual exactly equivalent to
        # the frozen SocketMatcher within the union.
        order = np.argsort(-raw[anchor, candidates], kind="stable")
        candidates = np.ascontiguousarray(candidates[order], dtype=np.int32)
        raw_rank = raw_row_rank[anchor, candidates]
        restored_rank = restored_row_rank[anchor, candidates]
        feature = np.stack(
            (
                raw_rank,
                raw_column_rank[anchor, candidates],
                restored_rank,
                restored_column_rank[anchor, candidates],
                raw_z[anchor, candidates],
                restored_z[anchor, candidates],
                raw_rank * restored_rank,
                np.abs(raw_rank - restored_rank),
            ),
            axis=1,
        )
        rows.append(candidates)
        features.append(np.ascontiguousarray(feature, dtype=np.float32))
        baselines.append(np.ascontiguousarray(raw_z[anchor, candidates], dtype=np.float32))
    return CandidateUnion(
        rows=tuple(rows),
        raw_topk=raw_topk,
        restored_topk=restored_topk,
        scalar_features=tuple(features),
        baseline_scores=tuple(baselines),
    )


def restored_seam_features(
    restored_tiles: torch.Tensor,
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    directions: torch.Tensor,
    *,
    border: int = 6,
) -> torch.Tensor:
    """Vectorise the historical E20 seven-channel restored seam input."""

    if restored_tiles.ndim != 4 or restored_tiles.shape[1:] != (3, 20, 20):
        raise ValueError("restored_tiles must be N x 3 x 20 x 20")
    if anchors.shape != candidates.shape or anchors.shape != directions.shape:
        raise ValueError("anchors, candidates and directions must have equal shape")
    if not 1 <= border <= 10:
        raise ValueError("border must be in [1, 10]")
    anchor = restored_tiles[anchors]
    candidate = restored_tiles[candidates]
    down = directions.to(torch.bool)[:, None, None, None]
    anchor = torch.where(down, anchor.transpose(-2, -1), anchor)
    candidate = torch.where(down, candidate.transpose(-2, -1), candidate)
    seam = torch.cat((anchor[..., -border:], candidate[..., :border]), dim=3)
    gray = 0.299 * seam[:, :1] + 0.587 * seam[:, 1:2] + 0.114 * seam[:, 2:3]
    gradient_x = F.pad(gray[..., 2:] - gray[..., :-2], (1, 1, 0, 0))
    gradient_y = F.pad(gray[..., 2:, :] - gray[..., :-2, :], (0, 0, 1, 1))
    direction_channel = directions[:, None, None, None].to(seam).expand_as(gray)
    return torch.cat((seam, gray, gradient_x, gradient_y, direction_channel), dim=1)


class RestoredBorderRanker(nn.Module):
    """Learned residual over raw d64 ordering for one target-blind union."""

    def __init__(self, *, base: int = 32, scalar_features: int = 8) -> None:
        super().__init__()
        if base <= 0 or scalar_features <= 0:
            raise ValueError("base and scalar_features must be positive")
        self.base = int(base)
        self.scalar_feature_count = int(scalar_features)
        self.seam = nn.Sequential(
            nn.Conv2d(7, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 2 * base, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * base),
            nn.SiLU(),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1),
            nn.GroupNorm(8, 2 * base),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * base + scalar_features, 2 * base),
            nn.SiLU(),
            nn.Linear(2 * base, base),
            nn.SiLU(),
            nn.Linear(base, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        restored_tiles: torch.Tensor,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        directions: torch.Tensor,
        scalar_features: torch.Tensor,
        baseline_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scalar_features.ndim != 2 or scalar_features.shape[1] != self.scalar_feature_count:
            raise ValueError("scalar feature shape differs from model contract")
        seam = restored_seam_features(
            restored_tiles,
            anchors,
            candidates,
            directions,
        )
        encoded = self.seam(seam)
        residual = self.head(torch.cat((encoded, scalar_features), dim=1)).squeeze(1)
        return baseline_scores + residual, residual


def pad_candidate_rows(
    rows: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, int]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pack labelled variable-size candidate rows for listwise CE."""

    if not rows:
        raise ValueError("cannot pack an empty row list")
    maximum = max(len(row[2]) for row in rows)
    valid = torch.zeros((len(rows), maximum), dtype=torch.bool, device=device)
    indices = torch.full((len(rows), maximum), -1, dtype=torch.long, device=device)
    anchors: list[int] = []
    candidates: list[int] = []
    directions: list[int] = []
    features: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    offset = 0
    for row_index, (anchor, direction, candidate, feature, baseline, _) in enumerate(rows):
        length = len(candidate)
        valid[row_index, :length] = True
        indices[row_index, :length] = torch.arange(offset, offset + length, device=device)
        anchors.extend([anchor] * length)
        candidates.extend(candidate.tolist())
        directions.extend([direction] * length)
        features.append(feature)
        baselines.append(baseline)
        offset += length
    return {
        "anchors": torch.as_tensor(anchors, dtype=torch.long, device=device),
        "candidates": torch.as_tensor(candidates, dtype=torch.long, device=device),
        "directions": torch.as_tensor(directions, dtype=torch.long, device=device),
        "features": torch.from_numpy(np.concatenate(features)).to(device),
        "baseline": torch.from_numpy(np.concatenate(baselines)).to(device),
        "valid": valid,
        "indices": indices,
        "targets": torch.as_tensor([row[5] for row in rows], dtype=torch.long, device=device),
    }


def unpack_candidate_logits(flat: torch.Tensor, packed: dict[str, torch.Tensor]) -> torch.Tensor:
    """Restore flat pair scores to padded row logits."""

    index = packed["indices"]
    output = flat[index.clamp_min(0)]
    return output.masked_fill(~packed["valid"], -1e4)


__all__ = [
    "CandidateUnion",
    "RestoredBorderRanker",
    "build_candidate_union",
    "pad_candidate_rows",
    "restored_descriptor_scores",
    "restored_seam_features",
    "unpack_candidate_logits",
]
