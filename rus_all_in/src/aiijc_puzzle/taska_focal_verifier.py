"""Audited inference port of the historical TASKA focal seam verifier.

The recovered verifier is a joint classifier over the two dirty sides of one
candidate seam.  It is deliberately narrower than the TASKA matcher: candidate
membership stays frozen and the verifier supplies only a larger-is-better
ordering priority.  Component placement and the Hungarian fill continue to use
the untouched matcher cost matrices.

Two historical scalar-feature contracts existed and must not be conflated:

``train_exact_top5``
    The exact contract in ``train_verify.py``.  Row mean/spread are computed
    from the five highest compatibilities used to train the checkpoint.

``historical_tip_top8``
    The repository-tip ``verify_edges.py`` inference contract, which rebuilt
    those two statistics from the eight highest row compatibilities.

Both modes use only raw dirty tiles, matcher costs, and the harvested edge
identity.  They do not consume targets, restored labels, filenames, canonical
tile ids, source-grid coordinates, or clean pixels.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.taska_edge_calibrator import (
    PrioritizedRawTailResult,
    solve_prioritized_raw_tail_global,
)

FocalFeatureMode = Literal["train_exact_top5", "historical_tip_top8"]

TASKA_FOCAL_VERIFIER_SHA256 = (
    "3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7"
)
TASKA_FOCAL_VERIFIER_SIZE_BYTES = 817_615
TASKA_FOCAL_VERIFIER_ARGS = {"ch": 64, "blocks": 4, "strip": 4}
TASKA_FOCAL_VERIFIER_PARAMETER_COUNT = 200_322
TASKA_FOCAL_VERIFIER_STATE_ELEMENT_COUNT = 200_838
TASKA_FOCAL_FEATURE_COUNT = 6
TASKA_FOCAL_FEATURE_TOP_K: dict[FocalFeatureMode, int] = {
    "train_exact_top5": 5,
    "historical_tip_top8": 8,
}


class TaskaFocalCheckpointError(RuntimeError):
    """The recovered focal checkpoint failed its audited load contract."""


class SeamVerifier(nn.Module):
    """Historical joint seam CNN, ported without architectural changes."""

    def __init__(
        self,
        ch: int = 64,
        blocks: int = 4,
        feats: int = TASKA_FOCAL_FEATURE_COUNT,
        strip: int = 4,
    ) -> None:
        super().__init__()
        self.strip = strip
        layers: list[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1), nn.GELU()]
        for index in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(ch, ch, 3, padding=1, dilation=1),
                    nn.BatchNorm2d(ch),
                    nn.GELU(),
                ]
            )
            if index % 2 == 1:
                layers.extend(
                    [nn.Conv2d(ch, ch, (3, 1), padding=(1, 0)), nn.GELU()]
                )
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(ch * 2 + feats, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.out = nn.Linear(64, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.prior = nn.Parameter(torch.ones(1))

    def forward(self, patch: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 4 or patch.shape[1:] != (3, 20, 2 * self.strip):
            raise ValueError(
                f"patch must have shape (batch, 3, 20, {2 * self.strip})"
            )
        if features.ndim != 2 or features.shape != (
            patch.shape[0],
            TASKA_FOCAL_FEATURE_COUNT,
        ):
            raise ValueError(
                "features must have shape (batch, TASKA_FOCAL_FEATURE_COUNT)"
            )
        hidden = self.trunk(patch)
        joint = hidden[:, :, :, self.strip - 1 : self.strip + 1].mean((2, 3))
        whole = hidden.mean((2, 3))
        encoded = self.head(torch.cat([joint, whole, features], dim=1))
        return self.out(encoded).squeeze(1) + self.prior * features[:, 0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_taska_focal_verifier(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> SeamVerifier:
    """Load the one audited checkpoint after size and SHA-256 verification.

    The digest gate executes before ``torch.load``.  Deserialisation is
    weights-only, the metadata contract is exact, and state loading is strict.
    """

    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise TaskaFocalCheckpointError(f"checkpoint does not exist: {path}")
    size = path.stat().st_size
    if size != TASKA_FOCAL_VERIFIER_SIZE_BYTES:
        raise TaskaFocalCheckpointError(
            "checkpoint size mismatch: "
            f"expected {TASKA_FOCAL_VERIFIER_SIZE_BYTES}, got {size}"
        )
    digest = _file_sha256(path)
    if digest != TASKA_FOCAL_VERIFIER_SHA256:
        raise TaskaFocalCheckpointError(
            "checkpoint SHA-256 mismatch: "
            f"expected {TASKA_FOCAL_VERIFIER_SHA256}, got {digest}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - exact torch exception is version-specific.
        raise TaskaFocalCheckpointError("weights-only checkpoint load failed") from error
    if not isinstance(payload, Mapping) or set(payload) != {"model", "args"}:
        raise TaskaFocalCheckpointError("checkpoint top-level key contract differs")
    if payload["args"] != TASKA_FOCAL_VERIFIER_ARGS:
        raise TaskaFocalCheckpointError("checkpoint architecture metadata does not match")
    state = payload["model"]
    if not isinstance(state, Mapping):
        raise TaskaFocalCheckpointError("checkpoint model state is not a mapping")
    model = SeamVerifier(
        ch=TASKA_FOCAL_VERIFIER_ARGS["ch"],
        blocks=TASKA_FOCAL_VERIFIER_ARGS["blocks"],
        feats=TASKA_FOCAL_FEATURE_COUNT,
        strip=TASKA_FOCAL_VERIFIER_ARGS["strip"],
    )
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TaskaFocalCheckpointError("checkpoint tensor contract does not match") from error
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != TASKA_FOCAL_VERIFIER_PARAMETER_COUNT:
        raise TaskaFocalCheckpointError("verifier architecture parameter count changed")
    state_element_count = sum(tensor.numel() for tensor in model.state_dict().values())
    if state_element_count != TASKA_FOCAL_VERIFIER_STATE_ELEMENT_COUNT:
        raise TaskaFocalCheckpointError("verifier state element count changed")
    try:
        model = model.to(torch.device(device))
    except (RuntimeError, TypeError, ValueError) as error:
        raise TaskaFocalCheckpointError(f"invalid or unavailable device: {device}") from error
    model.eval().requires_grad_(False)
    model.checkpoint_path = path
    model.checkpoint_sha256 = digest
    model.checkpoint_args = dict(TASKA_FOCAL_VERIFIER_ARGS)
    return model


def _validate_grid(grid: int) -> int:
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
        raise ValueError("grid must be an integer at least two")
    return grid * grid


def _as_finite_cost_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    matrix = np.asarray(current, dtype=np.float64)
    if matrix.shape != (count, count):
        raise ValueError(f"{name} must have shape {(count, count)}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _validated_edges(
    candidate_edges: Sequence[RawTailEdge],
    *,
    count: int,
) -> tuple[RawTailEdge, ...]:
    edges = tuple(candidate_edges)
    seen: set[tuple[int, int, str]] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, RawTailEdge):
            raise TypeError(f"candidate_edges[{index}] must be a RawTailEdge")
        if edge.axis not in {"right", "down"}:
            raise ValueError(f"candidate_edges[{index}] has an invalid axis")
        if isinstance(edge.source, bool) or not isinstance(edge.source, int):
            raise ValueError(f"candidate_edges[{index}].source must be an integer")
        if isinstance(edge.target, bool) or not isinstance(edge.target, int):
            raise ValueError(f"candidate_edges[{index}].target must be an integer")
        if not 0 <= edge.source < count or not 0 <= edge.target < count:
            raise ValueError(f"candidate_edges[{index}] is outside the input bag")
        if edge.source == edge.target:
            raise ValueError(f"candidate_edges[{index}] is a self-edge")
        identity = (edge.source, edge.target, edge.axis)
        if identity in seen:
            raise ValueError(f"candidate_edges[{index}] duplicates an earlier edge")
        seen.add(identity)
    return edges


def _validated_dirty_tiles(value: Any, *, count: int) -> np.ndarray:
    tiles = np.asarray(value)
    if tiles.shape != (count, 20, 20, 3):
        raise ValueError(
            f"dirty_tiles must have shape {(count, 20, 20, 3)}, got {tiles.shape}"
        )
    if tiles.dtype != np.uint8:
        raise ValueError("dirty_tiles must be raw uint8 pixels")
    return np.ascontiguousarray(tiles)


def build_focal_seam_patches(
    dirty_tiles: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    grid: int = 24,
    strip: int = TASKA_FOCAL_VERIFIER_ARGS["strip"],
) -> np.ndarray:
    """Build the exact historical joint patches, aligned to candidate edges."""

    count = _validate_grid(grid)
    if strip != TASKA_FOCAL_VERIFIER_ARGS["strip"]:
        raise ValueError("strip differs from the audited focal checkpoint")
    tiles = _validated_dirty_tiles(dirty_tiles, count=count)
    edges = _validated_edges(candidate_edges, count=count)
    patches = np.empty((len(edges), 3, 20, 2 * strip), dtype=np.float32)
    right_indices = np.asarray(
        [index for index, edge in enumerate(edges) if edge.axis == "right"],
        dtype=np.int64,
    )
    down_indices = np.asarray(
        [index for index, edge in enumerate(edges) if edge.axis == "down"],
        dtype=np.int64,
    )
    for indices, axis in ((right_indices, "right"), (down_indices, "down")):
        if not len(indices):
            continue
        source = np.asarray([edges[index].source for index in indices], dtype=np.int64)
        target = np.asarray([edges[index].target for index in indices], dtype=np.int64)
        if axis == "right":
            joined = np.concatenate(
                [tiles[source, :, -strip:], tiles[target, :, :strip]],
                axis=2,
            )
        else:
            joined = np.concatenate(
                [tiles[source, -strip:, :], tiles[target, :strip, :]],
                axis=1,
            ).transpose(0, 2, 1, 3)
        patches[indices] = joined.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(patches)


def extract_focal_edge_features(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    mode: FocalFeatureMode,
    grid: int = 24,
) -> np.ndarray:
    """Rebuild the exact six historical features for every harvested edge."""

    if mode not in TASKA_FOCAL_FEATURE_TOP_K:
        raise ValueError(f"unsupported focal feature mode: {mode!r}")
    count = _validate_grid(grid)
    top_k = TASKA_FOCAL_FEATURE_TOP_K[mode]
    if count < top_k:
        raise ValueError(f"grid has fewer than the required top-{top_k} candidates")
    right = -_as_finite_cost_matrix(cost_right, count=count, name="cost_right")
    down = -_as_finite_cost_matrix(cost_down, count=count, name="cost_down")
    np.fill_diagonal(right, -1e9)
    np.fill_diagonal(down, -1e9)
    edges = _validated_edges(candidate_edges, count=count)
    features = np.empty((len(edges), TASKA_FOCAL_FEATURE_COUNT), dtype=np.float64)
    by_axis = (("right", right), ("down", down))
    for axis, matrix in by_axis:
        indices = [index for index, edge in enumerate(edges) if edge.axis == axis]
        if not indices:
            continue
        row_sorted = np.sort(matrix, axis=1)[:, ::-1]
        best = row_sorted[:, 0]
        top_mean = row_sorted[:, :top_k].mean(axis=1)
        spread = row_sorted[:, 0] - row_sorted[:, top_k - 1]
        for index in indices:
            edge = edges[index]
            score = float(matrix[edge.source, edge.target])
            row_best = float(best[edge.source])
            features[index] = (
                score / 10.0,
                score - row_best,
                float(np.count_nonzero(matrix[edge.source] > score)),
                float(score == row_best),
                float(top_mean[edge.source]) / 10.0,
                float(spread[edge.source]),
            )
    return np.ascontiguousarray(features, dtype=np.float32)


@dataclass(frozen=True)
class TaskaFocalScoreBatch:
    """Read-only verifier logits and exact inputs aligned to one edge roster."""

    logits: np.ndarray
    features: np.ndarray
    edges: tuple[RawTailEdge, ...]
    mode: FocalFeatureMode
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float32).copy()
        features = np.asarray(self.features, dtype=np.float32).copy()
        if logits.shape != (len(self.edges),):
            raise ValueError("logits and edges must be aligned")
        if features.shape != (len(self.edges), TASKA_FOCAL_FEATURE_COUNT):
            raise ValueError("features and edges must be aligned")
        if not np.isfinite(logits).all() or not np.isfinite(features).all():
            raise ValueError("focal score batch must contain only finite values")
        if self.mode not in TASKA_FOCAL_FEATURE_TOP_K:
            raise ValueError("focal score batch has an invalid feature mode")
        if self.checkpoint_sha256 != TASKA_FOCAL_VERIFIER_SHA256:
            raise ValueError("focal score batch checkpoint provenance differs")
        logits.setflags(write=False)
        features.setflags(write=False)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "features", features)


def score_focal_edges(
    model: SeamVerifier,
    dirty_tiles: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    mode: FocalFeatureMode,
    grid: int = 24,
    device: str | torch.device | None = None,
    chunk_size: int = 8192,
) -> TaskaFocalScoreBatch:
    """Return larger-is-better verifier logits without changing edge membership."""

    if not isinstance(model, SeamVerifier):
        raise TypeError("model must be a SeamVerifier")
    if getattr(model, "checkpoint_sha256", None) != TASKA_FOCAL_VERIFIER_SHA256:
        raise ValueError("model is not the SHA-gated audited focal checkpoint")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    count = _validate_grid(grid)
    edges = _validated_edges(candidate_edges, count=count)
    patches = build_focal_seam_patches(dirty_tiles, edges, grid=grid, strip=model.strip)
    features = extract_focal_edge_features(
        cost_right,
        cost_down,
        edges,
        mode=mode,
        grid=grid,
    )
    if device is None:
        try:
            inference_device = next(model.parameters()).device
        except StopIteration as error:  # pragma: no cover - the audited model has parameters.
            raise ValueError("model has no parameters") from error
    else:
        inference_device = torch.device(device)
        if next(model.parameters()).device != inference_device:
            raise ValueError("requested device differs from the loaded model device")
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(edges), chunk_size):
            stop = min(start + chunk_size, len(edges))
            patch_tensor = torch.from_numpy(patches[start:stop]).to(inference_device)
            feature_tensor = torch.from_numpy(features[start:stop]).to(inference_device)
            outputs.append(model(patch_tensor, feature_tensor).detach().cpu().numpy())
    logits = (
        np.concatenate(outputs).astype(np.float32, copy=False)
        if outputs
        else np.empty(0, dtype=np.float32)
    )
    return TaskaFocalScoreBatch(
        logits=logits,
        features=features,
        edges=edges,
        mode=mode,
        checkpoint_sha256=TASKA_FOCAL_VERIFIER_SHA256,
    )


def solve_focal_raw_tail_global(
    cost_right: Any,
    cost_down: Any,
    score_batch: TaskaFocalScoreBatch,
    *,
    border_unary: Any | None = None,
    grid: int = 24,
    config: RawTailGlobalConfig | None = None,
) -> PrioritizedRawTailResult:
    """Use focal logits only for component order; retain original costs elsewhere."""

    if not isinstance(score_batch, TaskaFocalScoreBatch):
        raise TypeError("score_batch must be a TaskaFocalScoreBatch")
    return solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        score_batch.edges,
        score_batch.logits,
        border_unary=border_unary,
        grid=grid,
        config=config,
    )


__all__ = [
    "FocalFeatureMode",
    "SeamVerifier",
    "TASKA_FOCAL_FEATURE_COUNT",
    "TASKA_FOCAL_FEATURE_TOP_K",
    "TASKA_FOCAL_VERIFIER_ARGS",
    "TASKA_FOCAL_VERIFIER_PARAMETER_COUNT",
    "TASKA_FOCAL_VERIFIER_SHA256",
    "TASKA_FOCAL_VERIFIER_SIZE_BYTES",
    "TASKA_FOCAL_VERIFIER_STATE_ELEMENT_COUNT",
    "TaskaFocalCheckpointError",
    "TaskaFocalScoreBatch",
    "build_focal_seam_patches",
    "extract_focal_edge_features",
    "load_taska_focal_verifier",
    "score_focal_edges",
    "solve_focal_raw_tail_global",
]
