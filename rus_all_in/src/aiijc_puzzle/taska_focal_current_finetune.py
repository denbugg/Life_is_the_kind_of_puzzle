"""One fixed current-distribution fine-tune for the audited TASKA focal verifier.

The model is trained only on binary labels for already-harvested organizer-train
edges.  At inference it consumes the same target-free dirty patches and top-5
matcher features as the recovered verifier, changes only edge ordering, and
keeps the recovered raw-score prior frozen exactly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.taska_edge_calibrator import (
    PrioritizedRawTailResult,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_ARGS,
    TASKA_FOCAL_VERIFIER_SHA256,
    SeamVerifier,
    build_focal_seam_patches,
    extract_focal_edge_features,
    load_taska_focal_verifier,
)

FINETUNE_SCHEMA = "aiijc-taska-focal-current-finetune-checkpoint-v1"
FINETUNE_FEATURE_MODE = "train_exact_top5"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_harvest_edge_labels(
    candidate_edges: Sequence[RawTailEdge],
    reference_tile_at_position: Any,
    *,
    grid: int = 24,
) -> np.ndarray:
    """Return one iff a harvested directed edge is an exact true neighbour."""

    reference = np.asarray(reference_tile_at_position, dtype=np.int64)
    count = grid * grid
    if reference.shape != (count,) or not np.array_equal(
        np.sort(reference), np.arange(count)
    ):
        raise ValueError("reference_tile_at_position must be a strict grid permutation")
    position = np.empty(count, dtype=np.int64)
    position[reference] = np.arange(count)
    labels = np.empty(len(candidate_edges), dtype=np.uint8)
    seen: set[tuple[int, int, str]] = set()
    for index, edge in enumerate(candidate_edges):
        if not isinstance(edge, RawTailEdge):
            raise TypeError("candidate_edges must contain RawTailEdge values")
        identity = (edge.source, edge.target, edge.axis)
        if identity in seen:
            raise ValueError("candidate_edges contains a duplicate")
        seen.add(identity)
        if edge.axis == "right":
            source_position = int(position[edge.source])
            labels[index] = int(
                source_position % grid != grid - 1
                and int(position[edge.target]) == source_position + 1
            )
        elif edge.axis == "down":
            source_position = int(position[edge.source])
            labels[index] = int(
                source_position < count - grid
                and int(position[edge.target]) == source_position + grid
            )
        else:
            raise ValueError("edge axis must be right or down")
    return labels


def board_pair_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """All-positive/all-negative logistic ranking loss for one harvested board."""

    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("logits and labels must be aligned vectors")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if not len(positive) or not len(negative):
        raise ValueError("each training board must contain both edge classes")
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


@dataclass(frozen=True)
class FocalTrainingBoard:
    patches: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    source_filename: str

    def __post_init__(self) -> None:
        patches = np.ascontiguousarray(self.patches, dtype=np.float32)
        features = np.ascontiguousarray(self.features, dtype=np.float32)
        labels = np.ascontiguousarray(self.labels, dtype=np.uint8)
        if patches.ndim != 4 or patches.shape[1:] != (3, 20, 8):
            raise ValueError("patches must have shape (edges, 3, 20, 8)")
        if features.shape != (len(patches), 6):
            raise ValueError("features must have shape (edges, 6)")
        if labels.shape != (len(patches),) or not np.isin(labels, (0, 1)).all():
            raise ValueError("labels must be an aligned binary vector")
        if not labels.any() or labels.all():
            raise ValueError("training board needs positive and negative edges")
        if not np.isfinite(patches).all() or not np.isfinite(features).all():
            raise ValueError("training arrays must be finite")
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)


def make_focal_training_board(
    dirty_tiles: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    reference_tile_at_position: Any,
    *,
    source_filename: str,
    grid: int = 24,
) -> FocalTrainingBoard:
    edges = tuple(candidate_edges)
    return FocalTrainingBoard(
        patches=build_focal_seam_patches(dirty_tiles, edges, grid=grid),
        features=extract_focal_edge_features(
            cost_right,
            cost_down,
            edges,
            mode=FINETUNE_FEATURE_MODE,
            grid=grid,
        ),
        labels=exact_harvest_edge_labels(edges, reference_tile_at_position, grid=grid),
        source_filename=source_filename,
    )


def prepare_finetune_model(
    recovered_checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> tuple[SeamVerifier, float]:
    """Load the audited parent, unfreeze its residual, and freeze ``prior``."""

    model = load_taska_focal_verifier(recovered_checkpoint, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.prior.requires_grad_(False)
    prior = float(model.prior.detach().cpu().item())
    model.train()
    return model, prior


def train_fixed_focal_model(
    model: SeamVerifier,
    boards: Sequence[FocalTrainingBoard],
    *,
    device: str | torch.device,
    epochs: int = 2,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    gradient_clip_norm: float = 1.0,
    seed: int = 2_026_083_103,
) -> list[dict[str, float]]:
    """Run the single preregistered boardwise pair-ranking fine-tune."""

    if epochs != 2 or learning_rate != 3e-5 or weight_decay != 0.01:
        raise ValueError("training hyperparameters differ from the fixed v1 arm")
    if gradient_clip_norm != 1.0 or seed != 2_026_083_103:
        raise ValueError("training controls differ from the fixed v1 arm")
    if not boards:
        raise ValueError("boards must not be empty")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if model.prior.requires_grad or not selected:
        raise ValueError("raw-score prior must be frozen while residual parameters train")
    initial_prior = model.prior.detach().clone()
    optimizer = torch.optim.AdamW(
        selected,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    target_device = torch.device(device)
    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(boards), generator=generator).tolist()
        losses: list[float] = []
        for index in order:
            board = boards[index]
            patches = torch.from_numpy(board.patches).to(target_device)
            features = torch.from_numpy(board.features).to(target_device)
            labels = torch.from_numpy(board.labels.astype(np.float32)).to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(patches, features)
            loss = board_pair_ranking_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selected, gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not torch.equal(model.prior.detach().cpu(), initial_prior.cpu()):
            raise RuntimeError("frozen raw-score prior changed")
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_pair_ranking_loss": float(np.mean(losses)),
            }
        )
    model.eval().requires_grad_(False)
    return history


def score_current_focal_edges(
    model: SeamVerifier,
    dirty_tiles: Any,
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    *,
    device: str | torch.device,
    grid: int = 24,
) -> np.ndarray:
    """Score frozen candidate membership with a recovered or fine-tuned model."""

    edges = tuple(candidate_edges)
    patches = build_focal_seam_patches(dirty_tiles, edges, grid=grid)
    features = extract_focal_edge_features(
        cost_right,
        cost_down,
        edges,
        mode=FINETUNE_FEATURE_MODE,
        grid=grid,
    )
    target_device = torch.device(device)
    model_device = next(model.parameters()).device
    if model_device.type != target_device.type or (
        target_device.index is not None and model_device.index != target_device.index
    ):
        raise ValueError("model and requested inference device differ")
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(patches).to(target_device),
            torch.from_numpy(features).to(target_device),
        )
    result = np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32)
    if result.shape != (len(edges),) or not np.isfinite(result).all():
        raise RuntimeError("verifier returned malformed logits")
    result.setflags(write=False)
    return result


def solve_current_focal_edges(
    cost_right: Any,
    cost_down: Any,
    candidate_edges: Sequence[RawTailEdge],
    logits: Any,
    *,
    grid: int = 24,
    config: RawTailGlobalConfig | None = None,
) -> PrioritizedRawTailResult:
    return solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        candidate_edges,
        logits,
        border_unary=None,
        grid=grid,
        config=config,
    )


def save_finetuned_checkpoint(
    path: str | Path,
    model: SeamVerifier,
    *,
    config_sha256: str,
    train_source_digest: str,
    history: Sequence[Mapping[str, float]],
    frozen_prior: float,
) -> str:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": FINETUNE_SCHEMA,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "args": dict(TASKA_FOCAL_VERIFIER_ARGS),
        "metadata": {
            "parent_checkpoint_sha256": TASKA_FOCAL_VERIFIER_SHA256,
            "config_sha256": config_sha256,
            "train_source_digest": train_source_digest,
            "feature_mode": FINETUNE_FEATURE_MODE,
            "epochs": 2,
            "loss": "all-pairs-logistic-ranking",
            "frozen_prior": float(frozen_prior),
            "history": [dict(row) for row in history],
        },
    }
    torch.save(payload, destination)
    return sha256_file(destination)


def load_finetuned_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_config_sha256: str,
    device: str | torch.device,
) -> SeamVerifier:
    checkpoint = Path(path)
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("fine-tuned checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "model",
        "args",
        "metadata",
    }:
        raise ValueError("fine-tuned checkpoint contract differs")
    if payload["schema"] != FINETUNE_SCHEMA or payload["args"] != TASKA_FOCAL_VERIFIER_ARGS:
        raise ValueError("fine-tuned checkpoint schema or architecture differs")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("fine-tuned checkpoint metadata is malformed")
    if metadata.get("parent_checkpoint_sha256") != TASKA_FOCAL_VERIFIER_SHA256:
        raise ValueError("fine-tuned checkpoint parent differs")
    if metadata.get("config_sha256") != expected_config_sha256:
        raise ValueError("fine-tuned checkpoint config differs")
    if metadata.get("feature_mode") != FINETUNE_FEATURE_MODE or metadata.get("epochs") != 2:
        raise ValueError("fine-tuned checkpoint training contract differs")
    model = SeamVerifier(**TASKA_FOCAL_VERIFIER_ARGS)
    model.load_state_dict(payload["model"], strict=True)
    if float(model.prior.detach().item()) != float(metadata["frozen_prior"]):
        raise ValueError("fine-tuned checkpoint raw-score prior changed")
    model.to(torch.device(device)).eval().requires_grad_(False)
    model.checkpoint_sha256 = expected_sha256
    model.parent_checkpoint_sha256 = TASKA_FOCAL_VERIFIER_SHA256
    model.config_sha256 = expected_config_sha256
    return model


__all__ = [
    "FINETUNE_FEATURE_MODE",
    "FINETUNE_SCHEMA",
    "FocalTrainingBoard",
    "board_pair_ranking_loss",
    "exact_harvest_edge_labels",
    "load_finetuned_checkpoint",
    "make_focal_training_board",
    "prepare_finetune_model",
    "save_finetuned_checkpoint",
    "score_current_focal_edges",
    "solve_current_focal_edges",
    "train_fixed_focal_model",
]
