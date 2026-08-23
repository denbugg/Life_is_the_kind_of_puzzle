"""Supervised training boundary for the E26 relation verifier.

Only this module accepts relation labels or a tile-to-cell permutation.  The
feature extractor in :mod:`e26_relation_verifier` remains label-free.  The
APIs are deliberately pure enough for an external resume-safe orchestrator:
training/evaluation receive explicit objects, optional progress hooks never
discover files, and artifact helpers are create-once and content-addressed.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import e23_i21_residual_candidate_oracle as e23
import e26_relation_verifier as verifier


MODEL_SCHEMA = "pazzle-e26-relation-verifier-mlp-v1"
TRAINING_SCHEMA = "pazzle-e26-relation-verifier-training-v1"
FEATURE_WIDTH = len(verifier.FEATURE_NAMES)
HIDDEN_WIDTH = 128
EMBEDDING_WIDTH = 64
DROPOUT = 0.10
EDGE_LOSS_WEIGHT = 1.0
MAX_HARD_NEGATIVES = 15
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
ARTIFACT_ROOT = Path("E:/pazzle_work/e26_contextual_edge")


class RelationTrainingError(ValueError):
    """A supervised E26 training/calibration invariant failed closed."""


@dataclass(frozen=True, slots=True)
class ModelOutput:
    row_logits: torch.Tensor
    edge_logits: torch.Tensor
    embeddings: torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainingScene:
    table: verifier.RelationQueryTable
    relevance: np.ndarray
    source_group: str


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: torch.Tensor
    listwise: torch.Tensor
    edge: torch.Tensor
    queries: int
    scenes: int
    sources: int


@dataclass(frozen=True, slots=True)
class CalibrationScene:
    table: verifier.RelationQueryTable
    relevance: np.ndarray
    row_logits: np.ndarray
    edge_logits: np.ndarray
    source_group: str = "default"


@dataclass(frozen=True, slots=True)
class CalibrationTemperatures:
    row_temperature: float
    edge_temperature: float
    row_nll: float
    edge_nll: float


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    path: Path
    sha256: str
    size: int
    created: bool


ProgressHook = Callable[[Mapping[str, Any]], None]


class RelationVerifierMLP(nn.Module):
    """Joint listwise-offset and component-edge verifier.

    The edge head sees permutation-invariant summaries of offset embeddings
    plus the explicit NONE embedding.  It therefore cannot infer a result
    from row ordering or numeric component IDs.
    """

    def __init__(self, *, feature_width: int = FEATURE_WIDTH) -> None:
        super().__init__()
        if feature_width != FEATURE_WIDTH:
            raise RelationTrainingError("feature width must equal the frozen 64")
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_WIDTH, EMBEDDING_WIDTH),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
        )
        self.row_head = nn.Linear(EMBEDDING_WIDTH, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(4 * EMBEDDING_WIDTH, EMBEDDING_WIDTH),
            nn.SiLU(),
            nn.Linear(EMBEDDING_WIDTH, 1),
        )

    def forward(
        self, features: torch.Tensor, query_offsets: Sequence[int] | np.ndarray
    ) -> ModelOutput:
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
            raise RelationTrainingError("model features must have shape [rows,64]")
        if not torch.isfinite(features).all():
            raise RelationTrainingError("model features must be finite")
        offsets = tuple(int(value) for value in query_offsets)
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != int(features.shape[0])
            or any(right - left < 2 for left, right in zip(offsets[:-1], offsets[1:]))
        ):
            raise RelationTrainingError("model query offsets are invalid")
        embeddings = self.trunk(features)
        row_logits = self.row_head(embeddings).squeeze(1)
        lengths = torch.tensor(
            [stop - start - 1 for start, stop in zip(offsets[:-1], offsets[1:])],
            dtype=torch.long,
            device=features.device,
        )
        query_ids = torch.repeat_interleave(
            torch.arange(len(offsets) - 1, device=features.device), lengths
        )
        offset_indices = torch.tensor(
            [
                row
                for start, stop in zip(offsets[:-1], offsets[1:])
                for row in range(start, stop - 1)
            ],
            dtype=torch.long,
            device=features.device,
        )
        offset_embeddings = embeddings[offset_indices]
        sums = embeddings.new_zeros((len(offsets) - 1, EMBEDDING_WIDTH))
        sums.index_add_(0, query_ids, offset_embeddings)
        mean = sums / lengths[:, None]
        maximum = embeddings.new_full(
            (len(offsets) - 1, EMBEDDING_WIDTH), -torch.inf
        )
        maximum.scatter_reduce_(
            0,
            query_ids[:, None].expand_as(offset_embeddings),
            offset_embeddings,
            reduce="amax",
            include_self=True,
        )
        exponential_sums = embeddings.new_zeros(
            (len(offsets) - 1, EMBEDDING_WIDTH)
        )
        exponential_sums.index_add_(
            0, query_ids, torch.exp(offset_embeddings - maximum[query_ids])
        )
        logmeanexp = (
            torch.log(exponential_sums) + maximum - torch.log(lengths[:, None])
        )
        none_indices = torch.tensor(
            [stop - 1 for stop in offsets[1:]],
            dtype=torch.long,
            device=features.device,
        )
        edge_input = torch.cat(
            (embeddings[none_indices], mean, maximum, logmeanexp), dim=1
        )
        edge_logits = self.edge_head(edge_input).squeeze(1)
        return ModelOutput(row_logits, edge_logits, embeddings)


def _validate_relevance(
    table: verifier.RelationQueryTable, relevance: np.ndarray
) -> np.ndarray:
    value = verifier._validate_table(table)
    if (
        not isinstance(relevance, np.ndarray)
        or relevance.shape != (value.rows,)
        or relevance.dtype != np.int8
        or not relevance.flags.c_contiguous
        or bool(((relevance != 0) & (relevance != 1)).any())
    ):
        raise RelationTrainingError("relevance must be contiguous binary int8")
    for start, stop in zip(value.query_offsets[:-1], value.query_offsets[1:]):
        if int(relevance[int(start) : int(stop)].sum()) != 1:
            raise RelationTrainingError("relevance must be exactly one-hot per query")
    result = np.array(relevance, dtype=np.int8, copy=True, order="C")
    result.setflags(write=False)
    return result


def build_relation_labels(
    result: e23.CandidatePoolResult,
    table: verifier.RelationQueryTable,
    permutation: np.ndarray,
) -> np.ndarray:
    """Create exact offset-or-NONE one-hot labels from training truth only."""

    if type(result) is not e23.CandidatePoolResult:
        raise RelationTrainingError("label builder requires the exact E23 result")
    value = verifier._validate_table(table)
    if (
        not isinstance(permutation, np.ndarray)
        or permutation.shape != (verifier.NUM_TILES,)
        or permutation.dtype != np.int64
        or not permutation.flags.c_contiguous
        or not np.array_equal(
            np.sort(permutation), np.arange(verifier.NUM_TILES, dtype=np.int64)
        )
    ):
        raise RelationTrainingError("permutation must be a contiguous tile-to-cell bijection")
    offset_rows = np.flatnonzero(value.row_kind == verifier.ROW_OFFSET)
    if not np.array_equal(
        value.hypothesis_ids[offset_rows],
        np.arange(len(result.hypotheses), dtype=np.int64),
    ):
        raise RelationTrainingError("feature table does not bind every E23 hypothesis")
    for row, hypothesis in zip(offset_rows.tolist(), result.hypotheses):
        if (
            int(value.relation_ids[row]) != int(hypothesis.relation_id)
            or tuple(map(int, value.relations[row])) != hypothesis.relation
        ):
            raise RelationTrainingError("feature row relation does not bind to E23")

    component_shifts: dict[int, tuple[int, int] | None] = {}
    for component in result.components:
        shifts = {
            (
                int(permutation[int(tile)] // verifier.GRID) - int(row),
                int(permutation[int(tile)] % verifier.GRID) - int(col),
            )
            for tile, row, col in component.entries
        }
        component_shifts[component.component_id] = (
            next(iter(shifts)) if len(shifts) == 1 else None
        )

    labels = np.zeros(value.rows, dtype=np.int8)
    for start, stop in zip(value.query_offsets[:-1], value.query_offsets[1:]):
        first, last = int(start), int(stop)
        u, v = map(int, value.relations[first, :2])
        shift_u, shift_v = component_shifts[u], component_shifts[v]
        positive: int | None = None
        if shift_u is not None and shift_v is not None:
            expected = (
                u,
                v,
                shift_v[0] - shift_u[0],
                shift_v[1] - shift_u[1],
            )
            matches = [
                row
                for row in range(first, last - 1)
                if tuple(map(int, value.relations[row])) == expected
            ]
            if len(matches) > 1:
                raise RelationTrainingError("query duplicates its exact true offset")
            if matches:
                positive = matches[0]
        labels[positive if positive is not None else last - 1] = 1
    return _validate_relevance(value, labels)


def hard_negative_query_subset(
    table: verifier.RelationQueryTable,
    relevance: np.ndarray,
    row_scores: np.ndarray,
    *,
    max_hard_negatives: int = MAX_HARD_NEGATIVES,
) -> tuple[np.ndarray, ...]:
    """Select positive + explicit NONE + top train-only negative offsets."""

    value = verifier._validate_table(table)
    labels = _validate_relevance(value, relevance)
    scores = np.asarray(row_scores, dtype=np.float64)
    if scores.shape != (value.rows,) or not np.isfinite(scores).all():
        raise RelationTrainingError("hard-negative scores must be finite and aligned")
    if int(max_hard_negatives) != max_hard_negatives or max_hard_negatives < 0:
        raise RelationTrainingError("max_hard_negatives must be non-negative")
    selected: list[np.ndarray] = []
    for start, stop in zip(value.query_offsets[:-1], value.query_offsets[1:]):
        first, last = int(start), int(stop)
        positive = int(np.flatnonzero(labels[first:last])[0] + first)
        none = last - 1
        negative_offsets = [
            row for row in range(first, none) if row != positive
        ]
        negative_offsets.sort(
            key=lambda row: (
                -float(scores[row]),
                int(value.relations[row, 2]),
                int(value.relations[row, 3]),
            )
        )
        rows = sorted(
            {positive, none, *negative_offsets[: int(max_hard_negatives)]}
        )
        selected.append(np.asarray(rows, dtype=np.int64))
    return tuple(selected)


def _scene_loss(
    model: RelationVerifierMLP,
    scene: TrainingScene,
    *,
    device: torch.device,
    training: bool,
    max_hard_negatives: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    table = verifier._validate_table(scene.table)
    labels = _validate_relevance(table, scene.relevance)
    # Feature tables are deliberately read-only; make an owned tensor so no
    # backend can ever write through a non-writable NumPy view.
    features = torch.tensor(table.features, dtype=torch.float32, device=device)
    output = model(features, table.query_offsets)
    if training:
        subsets = hard_negative_query_subset(
            table,
            labels,
            output.row_logits.detach().cpu().numpy(),
            max_hard_negatives=max_hard_negatives,
        )
    else:
        subsets = tuple(
            np.arange(int(start), int(stop), dtype=np.int64)
            for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:])
        )
    if training:
        width = max(map(len, subsets))
        packed = np.zeros((table.queries, width), dtype=np.int64)
        mask = np.zeros((table.queries, width), dtype=np.bool_)
        targets = np.empty(table.queries, dtype=np.int64)
        for query, rows in enumerate(subsets):
            local_labels = labels[rows]
            if int(local_labels.sum()) != 1:
                raise RelationTrainingError("hard-negative subset lost its positive row")
            packed[query, : len(rows)] = rows
            mask[query, : len(rows)] = True
            targets[query] = int(np.flatnonzero(local_labels)[0])
        packed_tensor = torch.as_tensor(packed, dtype=torch.long, device=device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        packed_logits = output.row_logits[packed_tensor].masked_fill(
            ~mask_tensor, -torch.inf
        )
        listwise = F.cross_entropy(
            packed_logits,
            torch.as_tensor(targets, dtype=torch.long, device=device),
        )
    else:
        # Validation retains every row.  It runs under no_grad in the public
        # evaluator, so a streaming expression avoids any padded max-width
        # allocation without retaining autograd graphs.
        losses: list[torch.Tensor] = []
        for rows in subsets:
            local_labels = labels[rows]
            target = int(np.flatnonzero(local_labels)[0])
            one = output.row_logits[
                torch.as_tensor(rows, dtype=torch.long, device=device)
            ]
            losses.append(torch.logsumexp(one, dim=0) - one[target])
        listwise = torch.stack(losses).mean()
    none_rows = table.query_offsets[1:] - 1
    edge_target = torch.as_tensor(
        (labels[none_rows] == 0).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    edge = F.binary_cross_entropy_with_logits(output.edge_logits, edge_target)
    return listwise, edge, table.queries


def balanced_relation_loss(
    model: RelationVerifierMLP,
    scenes: Sequence[TrainingScene],
    *,
    training: bool,
    max_hard_negatives: int = MAX_HARD_NEGATIVES,
    device: torch.device | str = "cpu",
) -> LossBreakdown:
    """Average queries within scene, scenes within source, then sources."""

    values = tuple(scenes)
    if not values:
        raise RelationTrainingError("at least one training scene is required")
    target_device = torch.device(device)
    by_source: dict[str, list[tuple[torch.Tensor, torch.Tensor, int]]] = {}
    for scene in values:
        if type(scene) is not TrainingScene or not scene.source_group:
            raise RelationTrainingError("each scene needs an explicit non-empty source group")
        by_source.setdefault(scene.source_group, []).append(
            _scene_loss(
                model,
                scene,
                device=target_device,
                training=training,
                max_hard_negatives=max_hard_negatives,
            )
        )
    source_listwise = [
        torch.stack([item[0] for item in source_scenes]).mean()
        for source_scenes in by_source.values()
    ]
    source_edge = [
        torch.stack([item[1] for item in source_scenes]).mean()
        for source_scenes in by_source.values()
    ]
    listwise = torch.stack(source_listwise).mean()
    edge = torch.stack(source_edge).mean()
    total = listwise + EDGE_LOSS_WEIGHT * edge
    return LossBreakdown(
        total=total,
        listwise=listwise,
        edge=edge,
        queries=sum(item[2] for source in by_source.values() for item in source),
        scenes=len(values),
        sources=len(by_source),
    )


def train_relation_epoch(
    model: RelationVerifierMLP,
    scenes: Sequence[TrainingScene],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_hard_negatives: int,
) -> LossBreakdown:
    """Backpropagate one source/scene graph at a time with exact balance."""

    values = tuple(scenes)
    if not values:
        raise RelationTrainingError("at least one training scene is required")
    by_source: dict[str, list[TrainingScene]] = {}
    for scene in values:
        if type(scene) is not TrainingScene or not scene.source_group:
            raise RelationTrainingError("each scene needs an explicit non-empty source group")
        by_source.setdefault(scene.source_group, []).append(scene)
    optimizer.zero_grad(set_to_none=True)
    number_sources = len(by_source)
    total_listwise = 0.0
    total_edge = 0.0
    queries = 0
    for source_scenes in by_source.values():
        source_scale = 1.0 / number_sources
        scene_scale = source_scale / len(source_scenes)
        for scene in source_scenes:
            listwise, edge, one_queries = _scene_loss(
                model,
                scene,
                device=device,
                training=True,
                max_hard_negatives=max_hard_negatives,
            )
            (scene_scale * (listwise + EDGE_LOSS_WEIGHT * edge)).backward()
            total_listwise += scene_scale * float(listwise.detach().cpu())
            total_edge += scene_scale * float(edge.detach().cpu())
            queries += one_queries
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    listwise_tensor = torch.tensor(total_listwise, dtype=torch.float32)
    edge_tensor = torch.tensor(total_edge, dtype=torch.float32)
    return LossBreakdown(
        total=listwise_tensor + EDGE_LOSS_WEIGHT * edge_tensor,
        listwise=listwise_tensor,
        edge=edge_tensor,
        queries=queries,
        scenes=len(values),
        sources=number_sources,
    )


def fit_relation_verifier(
    model: RelationVerifierMLP,
    scenes: Sequence[TrainingScene],
    *,
    epochs: int,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    max_hard_negatives: int = MAX_HARD_NEGATIVES,
    seed: int = 2601,
    device: torch.device | str = "cpu",
    progress_hook: ProgressHook | None = None,
) -> tuple[RelationVerifierMLP, tuple[Mapping[str, float | int], ...]]:
    """Fit explicitly supplied scenes; no filesystem or held-out discovery."""

    if int(epochs) != epochs or epochs < 1:
        raise RelationTrainingError("epochs must be positive")
    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    model.to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history: list[Mapping[str, float | int]] = []
    for epoch in range(int(epochs)):
        model.train()
        loss = train_relation_epoch(
            model,
            scenes,
            optimizer,
            max_hard_negatives=max_hard_negatives,
            device=target_device,
        )
        event: Mapping[str, float | int] = MappingProxyType(
            {
                "epoch": epoch + 1,
                "epochs": int(epochs),
                "training_objective": float(loss.total.detach().cpu()),
                "listwise_objective": float(loss.listwise.detach().cpu()),
                "edge_objective": float(loss.edge.detach().cpu()),
                "queries": loss.queries,
            }
        )
        history.append(event)
        if progress_hook is not None:
            progress_hook(event)
    return model, tuple(history)


def evaluate_relation_nll(
    model: RelationVerifierMLP,
    scenes: Sequence[TrainingScene],
    *,
    device: torch.device | str = "cpu",
) -> LossBreakdown:
    """Score every validation row; hard-negative sampling is forbidden here."""

    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    with torch.no_grad():
        return balanced_relation_loss(
            model, scenes, training=False, device=target_device
        )


def _row_nll(scenes: Sequence[CalibrationScene], temperature: float) -> float:
    by_source: dict[str, list[float]] = {}
    for scene in scenes:
        if not scene.source_group:
            raise RelationTrainingError("calibration source group must be non-empty")
        table = verifier._validate_table(scene.table)
        labels = _validate_relevance(table, scene.relevance)
        logits = np.asarray(scene.row_logits, dtype=np.float64)
        if logits.shape != (table.rows,) or not np.isfinite(logits).all():
            raise RelationTrainingError("calibration row logits are invalid")
        scene_losses: list[float] = []
        for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:]):
            first, last = int(start), int(stop)
            values = logits[first:last] / temperature
            maximum = float(values.max())
            target = int(np.flatnonzero(labels[first:last])[0])
            scene_losses.append(
                -(float(values[target]) - maximum - math.log(float(np.exp(values - maximum).sum())))
            )
        by_source.setdefault(scene.source_group, []).append(float(np.mean(scene_losses)))
    return float(
        np.mean([np.mean(source_losses) for source_losses in by_source.values()])
    )


def _edge_nll(scenes: Sequence[CalibrationScene], temperature: float) -> float:
    by_source: dict[str, list[float]] = {}
    for scene in scenes:
        if not scene.source_group:
            raise RelationTrainingError("calibration source group must be non-empty")
        table = verifier._validate_table(scene.table)
        labels = _validate_relevance(table, scene.relevance)
        logits = np.asarray(scene.edge_logits, dtype=np.float64)
        if logits.shape != (table.queries,) or not np.isfinite(logits).all():
            raise RelationTrainingError("calibration edge logits are invalid")
        scene_losses: list[float] = []
        for query, logit in enumerate(logits.tolist()):
            none = int(table.query_offsets[query + 1]) - 1
            target = float(labels[none] == 0)
            scaled = float(logit) / temperature
            scene_losses.append(
                max(scaled, 0.0)
                - scaled * target
                + math.log1p(math.exp(-abs(scaled)))
            )
        by_source.setdefault(scene.source_group, []).append(float(np.mean(scene_losses)))
    return float(
        np.mean([np.mean(source_losses) for source_losses in by_source.values()])
    )


def _golden_temperature(objective: Callable[[float], float]) -> tuple[float, float]:
    left, right = math.log(TEMPERATURE_MIN), math.log(TEMPERATURE_MAX)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    y1, y2 = objective(math.exp(x1)), objective(math.exp(x2))
    for _ in range(96):
        if y1 <= y2:
            right, x2, y2 = x2, x1, y1
            x1 = right - ratio * (right - left)
            y1 = objective(math.exp(x1))
        else:
            left, x1, y1 = x1, x2, y2
            x2 = left + ratio * (right - left)
            y2 = objective(math.exp(x2))
    temperature = min(TEMPERATURE_MAX, max(TEMPERATURE_MIN, math.exp((left + right) / 2)))
    return temperature, objective(temperature)


def fit_calibration_temperatures(
    scenes: Sequence[CalibrationScene],
) -> CalibrationTemperatures:
    """Fit independent scalar temperatures on complete validation queries."""

    values = tuple(scenes)
    if not values:
        raise RelationTrainingError("at least one calibration scene is required")
    row_temperature, row_nll = _golden_temperature(lambda t: _row_nll(values, t))
    edge_temperature, edge_nll = _golden_temperature(lambda t: _edge_nll(values, t))
    return CalibrationTemperatures(row_temperature, edge_temperature, row_nll, edge_nll)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_artifact_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path).resolve()
    root = ARTIFACT_ROOT.resolve()
    try:
        contained = os.path.commonpath((str(root), str(target))) == str(root)
    except ValueError:
        contained = False
    if target.drive.upper() != "E:" or not contained or target == root:
        raise RelationTrainingError(
            "E26 artifacts must live below E:/pazzle_work/e26_contextual_edge"
        )
    return target


def write_create_once_artifact(path: str | os.PathLike[str], payload: bytes) -> ArtifactReceipt:
    """Atomically create an artifact, or verify an identical committed copy."""

    target = _require_artifact_path(path)
    if not isinstance(payload, bytes) or not payload:
        raise RelationTrainingError("artifact payload must be non-empty bytes")
    if not target.parent.is_dir():
        raise RelationTrainingError("artifact parent directory must already exist")
    expected = _sha256_bytes(payload)
    if target.exists():
        observed = _sha256_file(target)
        if observed != expected or target.stat().st_size != len(payload):
            raise RelationTrainingError("create-once artifact already exists with different bytes")
        return ArtifactReceipt(target, observed, len(payload), False)

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            observed = _sha256_file(target)
            if observed != expected or target.stat().st_size != len(payload):
                raise RelationTrainingError("artifact commit raced with different bytes")
            return ArtifactReceipt(target, observed, len(payload), False)
        return ArtifactReceipt(target, expected, len(payload), True)
    finally:
        temporary.unlink(missing_ok=True)


def model_checkpoint_bytes(
    model: RelationVerifierMLP,
    *,
    metadata: Mapping[str, Any],
) -> bytes:
    """Serialize an explicit checkpoint payload for an external stage runner."""

    stream = io.BytesIO()
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "feature_schema": verifier.FEATURE_SCHEMA,
            "feature_names": verifier.FEATURE_NAMES,
            "state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        stream,
    )
    return stream.getvalue()


def commit_stage_manifest(
    path: str | os.PathLike[str],
    *,
    stage: str,
    artifacts: Mapping[str, str | os.PathLike[str]],
    upstream_commits: Mapping[str, str],
    progress_hook: ProgressHook | None = None,
) -> ArtifactReceipt:
    """Commit a canonical create-once manifest over explicit artifact paths."""

    if not stage or not artifacts or not upstream_commits:
        raise RelationTrainingError("stage, artifacts and upstream commits are required")
    records: list[dict[str, Any]] = []
    for name, value in sorted(artifacts.items()):
        artifact = Path(value).resolve()
        if not artifact.is_file():
            raise RelationTrainingError(f"explicit artifact {name!r} does not exist")
        records.append(
            {
                "name": name,
                "path": str(artifact),
                "sha256": _sha256_file(artifact),
                "size": artifact.stat().st_size,
            }
        )
    manifest = {
        "schema": "pazzle-e26-create-once-stage-manifest-v1",
        "stage": stage,
        "feature_schema": verifier.FEATURE_SCHEMA,
        "artifacts": records,
        "upstream_commits": dict(sorted(upstream_commits.items())),
    }
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    receipt = write_create_once_artifact(path, payload)
    if progress_hook is not None:
        progress_hook(
            MappingProxyType(
                {
                    "stage": stage,
                    "status": "committed",
                    "manifest_sha256": receipt.sha256,
                    "manifest_size": receipt.size,
                    "created": receipt.created,
                }
            )
        )
    return receipt


__all__ = (
    "ArtifactReceipt",
    "ARTIFACT_ROOT",
    "CalibrationScene",
    "CalibrationTemperatures",
    "DROPOUT",
    "EDGE_LOSS_WEIGHT",
    "EMBEDDING_WIDTH",
    "FEATURE_WIDTH",
    "HIDDEN_WIDTH",
    "LossBreakdown",
    "MAX_HARD_NEGATIVES",
    "MODEL_SCHEMA",
    "ModelOutput",
    "RelationTrainingError",
    "RelationVerifierMLP",
    "TrainingScene",
    "balanced_relation_loss",
    "build_relation_labels",
    "commit_stage_manifest",
    "evaluate_relation_nll",
    "fit_calibration_temperatures",
    "fit_relation_verifier",
    "hard_negative_query_subset",
    "model_checkpoint_bytes",
    "train_relation_epoch",
    "write_create_once_artifact",
)
