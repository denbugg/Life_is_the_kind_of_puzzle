"""Fixed context-aware TASKA edge ranker over one board's candidate graph.

Unlike the earlier independent 22-feature stackers, this model lets candidate
edges compete through permutation-equivariant outgoing-source and
incoming-target incidence summaries.  It never changes candidate membership or
pixels: it only adds a bounded residual to the frozen recovered-focal logit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.taska_focal_feature_stacker import (
    FOCAL_STACKER_FEATURE_COUNT,
    FOCAL_STACKER_FEATURE_NAMES,
)

INCIDENCE_GNN_SCHEMA = "aiijc-taska-incidence-gnn-v1"
INCIDENCE_GNN_FEATURE_COUNT = FOCAL_STACKER_FEATURE_COUNT
INCIDENCE_GNN_WIDTH = 64
INCIDENCE_GNN_BLOCK_COUNT = 2
INCIDENCE_GNN_RESIDUAL_BOUND = 2.0
INCIDENCE_GNN_NODE_COUNT = 24 * 24
INCIDENCE_GNN_TRAINING = {
    "steps": 400,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "seed": 2_026_083_184,
    "residual_square_weight": 1e-3,
    "board_sampling": "numpy-default-rng-uniform-with-replacement",
    "loss": "per-board-balanced-bce-plus-residual-square",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_features(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != INCIDENCE_GNN_FEATURE_COUNT:
        raise ValueError(
            "features must have shape rows x "
            f"{INCIDENCE_GNN_FEATURE_COUNT}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("features must contain only finite values")
    return np.ascontiguousarray(result)


def _finite_vector(value: Any, *, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite vector of length {rows}")
    return np.ascontiguousarray(result)


def _index_vector(value: Any, *, rows: int, name: str, maximum: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (rows,) or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be one integer vector of length {rows}")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= maximum):
        raise ValueError(f"{name} values must be in [0, {maximum})")
    return result


def fit_global_standardizer(features: Any) -> tuple[np.ndarray, np.ndarray]:
    """Fit the fixed global population-mean/population-scale transform."""

    matrix = _finite_features(features)
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    scale = np.asarray(matrix.std(axis=0), dtype=np.float64)
    scale[scale == 0.0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError("standardizer produced non-finite values")
    return np.ascontiguousarray(mean), np.ascontiguousarray(scale)


def _group_mean_max(
    states: torch.Tensor,
    groups: torch.Tensor,
    *,
    group_count: int,
) -> torch.Tensor:
    """Return mean+max for every group, with exact zeros for empty groups."""

    if states.ndim != 2 or groups.shape != (len(states),):
        raise ValueError("states and groups are misaligned")
    width = states.shape[1]
    sums = states.new_zeros((group_count, width))
    expanded = groups[:, None].expand(-1, width)
    sums.scatter_add_(0, expanded, states)
    counts = states.new_zeros((group_count, 1))
    counts.scatter_add_(0, groups[:, None], states.new_ones((len(states), 1)))
    means = sums / counts.clamp_min(1.0)
    maxima = states.new_full((group_count, width), -torch.inf)
    maxima.scatter_reduce_(0, expanded, states, reduce="amax", include_self=True)
    maxima = torch.where(counts > 0, maxima, torch.zeros_like(maxima))
    return torch.cat((means, maxima), dim=1)


class _IncidenceResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(width * 5, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        states: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        axis: torch.Tensor,
        *,
        node_count: int,
    ) -> torch.Tensor:
        group_count = 2 * node_count
        source_groups = axis * node_count + source
        target_groups = axis * node_count + target
        source_context = _group_mean_max(
            states, source_groups, group_count=group_count
        )[source_groups]
        target_context = _group_mean_max(
            states, target_groups, group_count=group_count
        )[target_groups]
        return states + self.update(
            torch.cat((states, source_context, target_context), dim=1)
        )


class TaskaIncidenceGNN(nn.Module):
    """Two-block board/axis incidence ranker with a bounded focal residual."""

    def __init__(
        self,
        *,
        feature_count: int = INCIDENCE_GNN_FEATURE_COUNT,
        width: int = INCIDENCE_GNN_WIDTH,
        block_count: int = INCIDENCE_GNN_BLOCK_COUNT,
        node_count: int = INCIDENCE_GNN_NODE_COUNT,
    ) -> None:
        super().__init__()
        if feature_count != INCIDENCE_GNN_FEATURE_COUNT:
            raise ValueError("feature-count contract changed")
        if width != INCIDENCE_GNN_WIDTH or block_count != INCIDENCE_GNN_BLOCK_COUNT:
            raise ValueError("fixed incidence-GNN architecture changed")
        if node_count <= 0:
            raise ValueError("node_count must be positive")
        self.node_count = int(node_count)
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_count, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [_IncidenceResidualBlock(width) for _ in range(block_count)]
        )
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(
        self,
        standardized_features: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        axis: torch.Tensor,
    ) -> torch.Tensor:
        rows = len(standardized_features)
        if standardized_features.shape != (rows, INCIDENCE_GNN_FEATURE_COUNT):
            raise ValueError("standardized feature contract changed")
        if source.shape != (rows,) or target.shape != (rows,) or axis.shape != (rows,):
            raise ValueError("incidence vectors are misaligned")
        if rows == 0:
            return standardized_features.new_empty((0,))
        if (
            torch.any(source < 0)
            or torch.any(source >= self.node_count)
            or torch.any(target < 0)
            or torch.any(target >= self.node_count)
            or torch.any(axis < 0)
            or torch.any(axis > 1)
        ):
            raise ValueError("incidence indices are outside the fixed board contract")
        states = self.input_mlp(standardized_features)
        for block in self.blocks:
            states = block(
                states,
                source,
                target,
                axis,
                node_count=self.node_count,
            )
        return INCIDENCE_GNN_RESIDUAL_BOUND * torch.tanh(self.head(states)[:, 0])

    def forward(
        self,
        standardized_features: torch.Tensor,
        focal_logits: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        axis: torch.Tensor,
    ) -> torch.Tensor:
        if focal_logits.shape != (len(standardized_features),):
            raise ValueError("focal logits are misaligned")
        return focal_logits + self.residual(
            standardized_features, source, target, axis
        )


@dataclass(frozen=True)
class TaskaIncidenceGNNBundle:
    model: TaskaIncidenceGNN
    mean: np.ndarray
    scale: np.ndarray
    contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        expected = (INCIDENCE_GNN_FEATURE_COUNT,)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError("standardizer vector shape changed")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("standardizer must be finite")
        if np.any(scale <= 0):
            raise ValueError("standardizer scale must be positive")
        object.__setattr__(self, "mean", np.ascontiguousarray(mean.copy()))
        object.__setattr__(self, "scale", np.ascontiguousarray(scale.copy()))

    def predict_logits(
        self,
        features: Any,
        focal_logits: Any,
        source: Any,
        target: Any,
        axis: Any,
    ) -> np.ndarray:
        matrix = _finite_features(features)
        rows = len(matrix)
        focal = _finite_vector(focal_logits, rows=rows, name="focal_logits")
        source_array = _index_vector(
            source, rows=rows, name="source", maximum=self.model.node_count
        )
        target_array = _index_vector(
            target, rows=rows, name="target", maximum=self.model.node_count
        )
        axis_array = _index_vector(axis, rows=rows, name="axis", maximum=2)
        standardized = np.asarray((matrix - self.mean) / self.scale, dtype=np.float32)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(standardized),
                torch.from_numpy(focal.astype(np.float32)),
                torch.from_numpy(source_array),
                torch.from_numpy(target_array),
                torch.from_numpy(axis_array),
            )
        result = np.ascontiguousarray(logits.numpy(), dtype=np.float64)
        if result.shape != (rows,) or not np.isfinite(result).all():
            raise RuntimeError("incidence GNN emitted malformed logits")
        result.setflags(write=False)
        return result

    def predict_priorities(self, *args: Any) -> np.ndarray:
        logits = self.predict_logits(*args)
        result = np.empty_like(logits)
        positive = logits >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        result.setflags(write=False)
        return result


def train_taska_incidence_gnn(
    *,
    features: Any,
    focal_logits: Any,
    labels: Any,
    offsets: Any,
    source: Any,
    target: Any,
    axis: Any,
) -> tuple[TaskaIncidenceGNNBundle, dict[str, Any]]:
    """Train exactly the fixed 400-step extension128 experiment."""

    matrix = _finite_features(features)
    rows = len(matrix)
    focal = _finite_vector(focal_logits, rows=rows, name="focal_logits")
    raw_labels = np.asarray(labels)
    if raw_labels.shape != (rows,) or not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("labels must be one aligned binary vector")
    binary = np.ascontiguousarray(raw_labels, dtype=np.float32)
    board_offsets = np.asarray(offsets, dtype=np.int64)
    if (
        board_offsets.ndim != 1
        or len(board_offsets) < 2
        or board_offsets[0] != 0
        or board_offsets[-1] != rows
        or np.any(np.diff(board_offsets) <= 0)
    ):
        raise ValueError("offsets must partition all training edges into boards")
    source_array = _index_vector(
        source, rows=rows, name="source", maximum=INCIDENCE_GNN_NODE_COUNT
    )
    target_array = _index_vector(
        target, rows=rows, name="target", maximum=INCIDENCE_GNN_NODE_COUNT
    )
    axis_array = _index_vector(axis, rows=rows, name="axis", maximum=2)
    mean, scale = fit_global_standardizer(matrix)
    standardized = np.asarray((matrix - mean) / scale, dtype=np.float32)

    seed = int(INCIDENCE_GNN_TRAINING["seed"])
    torch.manual_seed(seed)
    model = TaskaIncidenceGNN()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(INCIDENCE_GNN_TRAINING["learning_rate"]),
        weight_decay=float(INCIDENCE_GNN_TRAINING["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    board_count = len(board_offsets) - 1
    schedule = generator.integers(
        0,
        board_count,
        size=int(INCIDENCE_GNN_TRAINING["steps"]),
        dtype=np.int64,
    )
    losses: list[float] = []
    bce_losses: list[float] = []
    penalties: list[float] = []
    model.train()
    for board_index in schedule:
        start = int(board_offsets[board_index])
        stop = int(board_offsets[board_index + 1])
        feature_tensor = torch.from_numpy(standardized[start:stop])
        focal_tensor = torch.from_numpy(focal[start:stop].astype(np.float32))
        label_tensor = torch.from_numpy(binary[start:stop])
        source_tensor = torch.from_numpy(source_array[start:stop])
        target_tensor = torch.from_numpy(target_array[start:stop])
        axis_tensor = torch.from_numpy(axis_array[start:stop])
        logits = model(
            feature_tensor,
            focal_tensor,
            source_tensor,
            target_tensor,
            axis_tensor,
        )
        positive = label_tensor > 0.5
        negative = ~positive
        if not torch.any(positive) or not torch.any(negative):
            raise ValueError("every sampled board must contain both label classes")
        balanced_bce = 0.5 * F.softplus(-logits[positive]).mean() + 0.5 * F.softplus(
            logits[negative]
        ).mean()
        residual = logits - focal_tensor
        penalty = residual.square().mean()
        loss = balanced_bce + float(
            INCIDENCE_GNN_TRAINING["residual_square_weight"]
        ) * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        bce_losses.append(float(balanced_bce.detach()))
        penalties.append(float(penalty.detach()))

    model.eval()
    contract = {
        "schema": INCIDENCE_GNN_SCHEMA,
        "feature_names": list(FOCAL_STACKER_FEATURE_NAMES),
        "architecture": {
            "feature_count": INCIDENCE_GNN_FEATURE_COUNT,
            "width": INCIDENCE_GNN_WIDTH,
            "incidence_block_count": INCIDENCE_GNN_BLOCK_COUNT,
            "node_count": INCIDENCE_GNN_NODE_COUNT,
            "axis_conditioning": "board-axis-grouped",
            "source_aggregate": ["mean", "max"],
            "target_aggregate": ["mean", "max"],
            "activation": "SiLU",
            "residual_bound": INCIDENCE_GNN_RESIDUAL_BOUND,
            "head_zero_initialized": True,
            "base_logit": "frozen-recovered-focal",
        },
        "training": dict(INCIDENCE_GNN_TRAINING),
    }
    bundle = TaskaIncidenceGNNBundle(model=model, mean=mean, scale=scale, contract=contract)
    history = {
        "board_count": board_count,
        "edge_count": rows,
        "positive_count": int(binary.sum()),
        "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_50_loss": float(np.mean(losses[-50:])),
        "final_balanced_bce": bce_losses[-1],
        "final_residual_square": penalties[-1],
        "completed_steps": len(losses),
    }
    return bundle, history


def save_taska_incidence_gnn_bundle(
    bundle: TaskaIncidenceGNNBundle,
    directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Persist weights-only state, standardizer, and their SHA-gated contract."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "incidence-gnn-weights.pt"
    standardizer_path = output / "incidence-gnn-standardizer.npz"
    contract_path = output / "incidence-gnn-contract.json"
    with weights_path.open("xb") as stream:
        torch.save(bundle.model.state_dict(), stream)
    with standardizer_path.open("xb") as stream:
        np.savez_compressed(stream, mean=bundle.mean, scale=bundle.scale)
    contract = {
        **dict(bundle.contract),
        "artifacts": {
            "weights": {
                "filename": weights_path.name,
                "sha256": _sha256(weights_path),
                "state_only": True,
            },
            "standardizer": {
                "filename": standardizer_path.name,
                "sha256": _sha256(standardizer_path),
                "mean_scale_only": True,
            },
        },
    }
    with contract_path.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return weights_path, standardizer_path, contract_path


def load_taska_incidence_gnn_bundle(
    contract_path: str | Path,
    *,
    expected_contract_sha256: str,
) -> TaskaIncidenceGNNBundle:
    """Load only after the caller supplies the frozen contract SHA-256."""

    path = Path(contract_path)
    if _sha256(path) != expected_contract_sha256:
        raise ValueError("incidence-GNN contract SHA-256 mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != INCIDENCE_GNN_SCHEMA:
        raise ValueError("incidence-GNN schema changed")
    if tuple(contract.get("feature_names", ())) != FOCAL_STACKER_FEATURE_NAMES:
        raise ValueError("incidence-GNN feature contract changed")
    if contract.get("training") != INCIDENCE_GNN_TRAINING:
        raise ValueError("incidence-GNN training contract changed")
    architecture = contract.get("architecture", {})
    expected_architecture = {
        "feature_count": INCIDENCE_GNN_FEATURE_COUNT,
        "width": INCIDENCE_GNN_WIDTH,
        "incidence_block_count": INCIDENCE_GNN_BLOCK_COUNT,
        "node_count": INCIDENCE_GNN_NODE_COUNT,
        "axis_conditioning": "board-axis-grouped",
        "source_aggregate": ["mean", "max"],
        "target_aggregate": ["mean", "max"],
        "activation": "SiLU",
        "residual_bound": INCIDENCE_GNN_RESIDUAL_BOUND,
        "head_zero_initialized": True,
        "base_logit": "frozen-recovered-focal",
    }
    if architecture != expected_architecture:
        raise ValueError("incidence-GNN architecture contract changed")
    artifacts = contract.get("artifacts", {})
    weights_path = path.parent / artifacts["weights"]["filename"]
    standardizer_path = path.parent / artifacts["standardizer"]["filename"]
    if _sha256(weights_path) != artifacts["weights"]["sha256"]:
        raise ValueError("incidence-GNN weights SHA-256 mismatch")
    if _sha256(standardizer_path) != artifacts["standardizer"]["sha256"]:
        raise ValueError("incidence-GNN standardizer SHA-256 mismatch")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("weights artifact is not a state mapping")
    model = TaskaIncidenceGNN()
    model.load_state_dict(state, strict=True)
    model.eval()
    with np.load(standardizer_path, allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "scale"}:
            raise ValueError("standardizer key contract changed")
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
    return TaskaIncidenceGNNBundle(
        model=model,
        mean=mean,
        scale=scale,
        contract=contract,
    )


__all__ = [
    "INCIDENCE_GNN_BLOCK_COUNT",
    "INCIDENCE_GNN_FEATURE_COUNT",
    "INCIDENCE_GNN_NODE_COUNT",
    "INCIDENCE_GNN_RESIDUAL_BOUND",
    "INCIDENCE_GNN_SCHEMA",
    "INCIDENCE_GNN_TRAINING",
    "INCIDENCE_GNN_WIDTH",
    "TaskaIncidenceGNN",
    "TaskaIncidenceGNNBundle",
    "fit_global_standardizer",
    "load_taska_incidence_gnn_bundle",
    "save_taska_incidence_gnn_bundle",
    "train_taska_incidence_gnn",
]
