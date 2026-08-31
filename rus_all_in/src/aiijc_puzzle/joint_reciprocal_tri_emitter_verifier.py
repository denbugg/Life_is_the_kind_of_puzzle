"""Joint outgoing/incoming assignment for the tri-emitter edge verifier.

This module intentionally leaves :mod:`tri_emitter_edge_verifier` unchanged.
It reuses that model as a relation-content backbone, then trains each sparse
candidate edge against both its outgoing row and incoming column.  ``NONE`` is
an explicit learned class on both sides, so board borders and true neighbours
missing from the frozen union are not forced onto a false candidate.

The deployment head is target-free and fixed: for each axis and board it keeps
the top five percent of reciprocal row/column winners by a calibrated,
differentiable two-sided confidence.  It never changes candidate identities or
pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    DINO_PROJECTION_DIM,
    TriEmitterEdgeVerifier,
)

MASKED_LOGIT = -1.0e4
SOFTMIN_TAU = 0.25
CONFIDENCE_BCE_WEIGHT = 0.25
DELTA_REGULARIZATION_WEIGHT = 1.0e-3
RECIPROCAL_HEAD_FRACTION = 0.05
MINIMUM_TEMPERATURE = 1.0e-3


@dataclass(frozen=True)
class JointTargets:
    """Exact row/column labels under one immutable sparse candidate roster."""

    row_slots: torch.Tensor
    column_sources: torch.Tensor
    edge_truth: torch.Tensor


@dataclass(frozen=True)
class JointAxisOutput:
    """All differentiable scores for one board and one direction axis."""

    edge_logits: torch.Tensor
    delta: torch.Tensor
    dense_logits: torch.Tensor
    dense_valid: torch.Tensor
    row_none_logits: torch.Tensor
    column_none_logits: torch.Tensor
    row_margins: torch.Tensor
    column_margins: torch.Tensor
    joint_confidence: torch.Tensor
    calibrated_confidence_logits: torch.Tensor


@dataclass(frozen=True)
class JointLoss:
    """Fixed joint objective and its auditable components."""

    total: torch.Tensor
    row_cross_entropy: torch.Tensor
    column_cross_entropy: torch.Tensor
    confidence_bce: torch.Tensor
    delta_regularization: torch.Tensor


@dataclass(frozen=True)
class ReciprocalHead:
    """Fixed-coverage reciprocal edge selection for one axis."""

    selected: np.ndarray
    reciprocal: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    confidences: np.ndarray
    requested_count: int


def _as_scalar_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not torch.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_sparse_inputs(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    edge_logits: torch.Tensor,
) -> tuple[int, int]:
    if candidates.ndim != 2 or valid.shape != candidates.shape:
        raise ValueError("candidates and valid must be aligned N x K tensors")
    if edge_logits.shape != candidates.shape:
        raise ValueError("edge_logits must match candidates")
    if candidates.dtype not in (torch.int32, torch.int64):
        raise ValueError("candidates must contain integer tile identities")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    count, width = candidates.shape
    if count < 2 or width < 1:
        raise ValueError("a sparse assignment needs at least two tiles and one slot")
    if not torch.isfinite(edge_logits[valid]).all():
        raise ValueError("valid edge logits must be finite")
    selected = candidates[valid]
    if len(selected) and (selected.min() < 0 or selected.max() >= count):
        raise ValueError("valid candidate identities must be in [0, tile_count)")
    for source in range(count):
        row = candidates[source, valid[source]]
        if len(row) != len(torch.unique(row)):
            raise ValueError("candidate identities must be unique within each row")
    return count, width


def sparse_to_dense_logits(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    edge_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter one sparse candidate roster into an aligned dense assignment."""

    count, _ = _validate_sparse_inputs(candidates, valid, edge_logits)
    dense = edge_logits.new_full((count, count), MASKED_LOGIT)
    dense_valid = torch.zeros((count, count), dtype=torch.bool, device=valid.device)
    rows = torch.arange(count, device=candidates.device)[:, None].expand_as(candidates)
    dense = dense.index_put(
        (rows[valid], candidates[valid].long()), edge_logits[valid], accumulate=False
    )
    dense_valid[rows[valid], candidates[valid].long()] = True
    return dense, dense_valid


def _leave_one_out_logsumexp(
    edge: torch.Tensor,
    total_logsumexp: torch.Tensor,
) -> torch.Tensor:
    """Stable ``log(exp(total)-exp(edge))`` with a learned NONE competitor."""

    difference = edge - total_logsumexp
    maximum = 1.0 - torch.finfo(edge.dtype).eps
    remaining = torch.log1p(-torch.exp(difference).clamp(max=maximum))
    return total_logsumexp + remaining


def _stable_class_leave_one_out(classes: torch.Tensor) -> torch.Tensor:
    """Leave-one-out logsumexp for all classes without dominant-edge loss.

    Direct ``log(exp(total)-exp(edge))`` is stable for every non-winning class:
    the winner remains in its denominator, so the removed share is at most one
    half.  The selected argmax is the sole problematic class; compute that one
    exactly by masking it before a second logsumexp.  This stays vectorized,
    float32/MPS compatible and transpose-equivariant.
    """

    if classes.ndim != 2 or classes.shape[1] < 2:
        raise ValueError("classes must be a B x C tensor with at least two classes")
    total = torch.logsumexp(classes, dim=1)
    top_index = classes.argmax(dim=1, keepdim=True)
    without_top = classes.scatter(1, top_index, -torch.inf)
    top_other = torch.logsumexp(without_top, dim=1, keepdim=True)
    general = _leave_one_out_logsumexp(classes, total[:, None])
    indices = torch.arange(classes.shape[1], device=classes.device)[None]
    return torch.where(indices == top_index, top_other, general)


def dense_two_sided_confidence(
    dense_logits: torch.Tensor,
    dense_valid: torch.Tensor,
    row_none_logits: torch.Tensor,
    column_none_logits: torch.Tensor,
    *,
    tau: float = SOFTMIN_TAU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row margin, column margin and differentiable two-sided minimum.

    The operation is transpose-equivariant when row/column ``NONE`` logits are
    swapped.  Invalid dense entries are returned as ``-inf`` and never
    contribute to a normalizer.
    """

    if dense_logits.ndim != 2 or dense_logits.shape[0] != dense_logits.shape[1]:
        raise ValueError("dense_logits must be a square matrix")
    if dense_valid.shape != dense_logits.shape or dense_valid.dtype != torch.bool:
        raise ValueError("dense_valid must be a matching boolean matrix")
    count = len(dense_logits)
    if row_none_logits.shape != (count,) or column_none_logits.shape != (count,):
        raise ValueError("row/column NONE logits must have shape N")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not torch.isfinite(dense_logits[dense_valid]).all():
        raise ValueError("valid dense logits must be finite")
    if not torch.isfinite(row_none_logits).all() or not torch.isfinite(
        column_none_logits
    ).all():
        raise ValueError("NONE logits must be finite")

    masked = dense_logits.masked_fill(~dense_valid, -torch.inf)
    row_classes = torch.cat((masked, row_none_logits[:, None]), dim=1)
    column_classes = torch.cat(
        (masked.transpose(0, 1), column_none_logits[:, None]), dim=1
    )
    row_other = _stable_class_leave_one_out(row_classes)[:, :count]
    column_other = _stable_class_leave_one_out(column_classes)[:, :count].transpose(
        0, 1
    )
    # Keep invalid entries finite inside the differentiable graph.  Applying
    # ``where`` only after arithmetic on ``-inf`` can still yield NaN gradients
    # even when those entries are absent from the loss.
    safe_edge = torch.where(dense_valid, masked, torch.zeros_like(masked))
    row_margin = safe_edge - row_other
    column_margin = safe_edge - column_other
    confidence = -tau * torch.logsumexp(
        torch.stack((-row_margin / tau, -column_margin / tau), dim=0), dim=0
    )
    negative_infinity = dense_logits.new_full((), -torch.inf)
    return tuple(
        torch.where(dense_valid, value, negative_infinity)
        for value in (row_margin, column_margin, confidence)
    )


def exact_joint_targets(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    truth_by_source: torch.Tensor,
) -> JointTargets:
    """Build row/column labels; ``-1`` truth maps to learned ``NONE``.

    A true neighbour absent from the frozen candidate union also maps to
    ``NONE`` on both sides.  This preserves the candidate identity freeze.
    """

    dummy_logits = torch.zeros_like(candidates, dtype=torch.float32)
    count, width = _validate_sparse_inputs(candidates, valid, dummy_logits)
    if truth_by_source.shape != (count,) or truth_by_source.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("truth_by_source must be an integer N-vector")
    if ((truth_by_source < -1) | (truth_by_source >= count)).any():
        raise ValueError("truth identities must be -1 or in [0, tile_count)")

    row_slots = torch.full((count,), -1, dtype=torch.long, device=candidates.device)
    edge_truth = torch.zeros_like(valid)
    for source in range(count):
        truth = int(truth_by_source[source])
        if truth < 0:
            continue
        matches = valid[source] & (candidates[source] == truth)
        slots = torch.nonzero(matches, as_tuple=False).flatten()
        if len(slots) == 1:
            row_slots[source] = slots[0]
            edge_truth[source, slots[0]] = True

    column_sources = torch.full(
        (count,), -1, dtype=torch.long, device=candidates.device
    )
    for source in range(count):
        slot = int(row_slots[source])
        if slot < 0:
            continue
        target = int(candidates[source, slot])
        if column_sources[target] >= 0:
            raise ValueError("truth_by_source is not injective on present neighbours")
        column_sources[target] = source
    if row_slots.max().item() >= width:
        raise RuntimeError("row target escaped the candidate width")
    return JointTargets(
        row_slots=row_slots,
        column_sources=column_sources,
        edge_truth=edge_truth,
    )


def build_joint_axis_output(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    edge_logits: torch.Tensor,
    delta: torch.Tensor,
    row_none_logits: torch.Tensor,
    column_none_logits: torch.Tensor,
    confidence_bias: torch.Tensor,
    confidence_temperature: torch.Tensor,
    *,
    tau: float = SOFTMIN_TAU,
) -> JointAxisOutput:
    """Attach assignment, confidence and calibration heads to sparse logits."""

    count, _ = _validate_sparse_inputs(candidates, valid, edge_logits)
    if delta.shape != edge_logits.shape:
        raise ValueError("delta must match edge_logits")
    if row_none_logits.shape != (count,) or column_none_logits.shape != (count,):
        raise ValueError("row/column NONE logits must have shape N")
    bias = _as_scalar_tensor(confidence_bias, name="confidence_bias")
    temperature = _as_scalar_tensor(
        confidence_temperature, name="confidence_temperature"
    )
    if temperature <= 0:
        raise ValueError("confidence_temperature must be positive")
    dense_logits, dense_valid = sparse_to_dense_logits(candidates, valid, edge_logits)
    dense_row, dense_column, dense_confidence = dense_two_sided_confidence(
        dense_logits,
        dense_valid,
        row_none_logits,
        column_none_logits,
        tau=tau,
    )
    safe_candidates = candidates.clamp_min(0).long()
    row_margin = dense_row.gather(1, safe_candidates)
    column_margin = dense_column.gather(1, safe_candidates)
    joint_confidence = dense_confidence.gather(1, safe_candidates)
    # Padded slots gather an invalid dense entry, whose public value is ``-inf``.
    # Mask it to a finite, calibration-neutral value *before* division.  A final
    # ``where`` alone is insufficient: autograd can otherwise form ``0 * inf``
    # in the scalar temperature gradient even though padded slots are absent
    # from the loss.
    safe_joint_confidence = torch.where(
        valid, joint_confidence, bias.expand_as(joint_confidence)
    )
    calibrated = (safe_joint_confidence - bias) / temperature
    negative_infinity = edge_logits.new_full((), -torch.inf)
    return JointAxisOutput(
        edge_logits=edge_logits.masked_fill(~valid, MASKED_LOGIT),
        delta=delta.masked_fill(~valid, 0.0),
        dense_logits=dense_logits,
        dense_valid=dense_valid,
        row_none_logits=row_none_logits,
        column_none_logits=column_none_logits,
        row_margins=torch.where(valid, row_margin, negative_infinity),
        column_margins=torch.where(valid, column_margin, negative_infinity),
        joint_confidence=torch.where(valid, joint_confidence, negative_infinity),
        calibrated_confidence_logits=torch.where(valid, calibrated, negative_infinity),
    )


class JointReciprocalTriEmitterVerifier(nn.Module):
    """Tri-emitter content scorer with joint row/column ``NONE`` supervision."""

    def __init__(
        self,
        *,
        dino_dim: int = DINO_PROJECTION_DIM,
        auxiliary_dim: int = AUXILIARY_DIM,
        width: int = 32,
        hidden: int = 96,
        initial_none_logit: float = 0.0,
        initial_confidence_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_none_logit):
            raise ValueError("initial_none_logit must be finite")
        if initial_confidence_temperature <= MINIMUM_TEMPERATURE:
            raise ValueError("initial confidence temperature is too small")
        self.edge_verifier = TriEmitterEdgeVerifier(
            dino_dim=dino_dim,
            auxiliary_dim=auxiliary_dim,
            width=width,
            hidden=hidden,
        )
        self.row_none_logits = nn.Parameter(
            torch.full((2,), float(initial_none_logit))
        )
        self.column_none_logits = nn.Parameter(
            torch.full((2,), float(initial_none_logit))
        )
        self.confidence_bias = nn.Parameter(torch.zeros(()))
        inverse = math.log(
            math.expm1(initial_confidence_temperature - MINIMUM_TEMPERATURE)
        )
        self.raw_confidence_temperature = nn.Parameter(torch.tensor(inverse))

    @property
    def confidence_temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_confidence_temperature) + MINIMUM_TEMPERATURE

    def forward(
        self,
        raw_sides: torch.Tensor,
        dino_sides: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        auxiliary: torch.Tensor,
        raw_baseline: torch.Tensor,
        *,
        direction: int,
    ) -> JointAxisOutput:
        """Score every source row for one axis in one vectorized model call."""

        if direction not in (0, 1):
            raise ValueError("direction must be 0 (right) or 1 (down)")
        count, _ = candidates.shape
        anchors = torch.arange(count, device=candidates.device)
        directions = torch.full(
            (count,), direction, dtype=torch.long, device=candidates.device
        )
        edge_logits, delta = self.edge_verifier(
            raw_sides,
            dino_sides,
            anchors,
            candidates,
            valid,
            directions,
            auxiliary,
            raw_baseline,
        )
        row_none = self.row_none_logits[direction].expand(count)
        column_none = self.column_none_logits[direction].expand(count)
        return build_joint_axis_output(
            candidates,
            valid,
            edge_logits,
            delta,
            row_none,
            column_none,
            self.confidence_bias,
            self.confidence_temperature,
        )


def joint_assignment_loss(
    output: JointAxisOutput,
    targets: JointTargets,
    valid: torch.Tensor,
    *,
    confidence_weight: float = CONFIDENCE_BCE_WEIGHT,
    delta_regularization_weight: float = DELTA_REGULARIZATION_WEIGHT,
) -> JointLoss:
    """Compute the frozen row CE + column CE + confidence BCE objective."""

    if not math.isfinite(confidence_weight) or confidence_weight < 0:
        raise ValueError("confidence_weight must be finite and nonnegative")
    if not math.isfinite(delta_regularization_weight) or delta_regularization_weight < 0:
        raise ValueError("delta_regularization_weight must be finite and nonnegative")
    count, width = output.edge_logits.shape
    if valid.shape != (count, width) or valid.dtype != torch.bool:
        raise ValueError("valid must match the sparse edge logits")
    if targets.row_slots.shape != (count,) or targets.column_sources.shape != (count,):
        raise ValueError("row/column target shapes changed")
    if targets.edge_truth.shape != valid.shape:
        raise ValueError("edge truth must match the sparse roster")

    row_classes = torch.cat(
        (
            output.edge_logits.masked_fill(~valid, MASKED_LOGIT),
            output.row_none_logits[:, None],
        ),
        dim=1,
    )
    row_targets = torch.where(
        targets.row_slots >= 0,
        targets.row_slots,
        torch.full_like(targets.row_slots, width),
    )
    row_ce = F.cross_entropy(row_classes, row_targets)

    dense = output.dense_logits.masked_fill(~output.dense_valid, MASKED_LOGIT)
    column_classes = torch.cat(
        (dense.transpose(0, 1), output.column_none_logits[:, None]), dim=1
    )
    column_targets = torch.where(
        targets.column_sources >= 0,
        targets.column_sources,
        torch.full_like(targets.column_sources, count),
    )
    column_ce = F.cross_entropy(column_classes, column_targets)

    confidence_bce = F.binary_cross_entropy_with_logits(
        output.calibrated_confidence_logits[valid],
        targets.edge_truth[valid].to(output.edge_logits.dtype),
    )
    delta_regularization = output.delta[valid].square().mean()
    total = (
        row_ce
        + column_ce
        + confidence_weight * confidence_bce
        + delta_regularization_weight * delta_regularization
    )
    return JointLoss(
        total=total,
        row_cross_entropy=row_ce,
        column_cross_entropy=column_ce,
        confidence_bce=confidence_bce,
        delta_regularization=delta_regularization,
    )


def fixed_fraction_reciprocal_head(
    output: JointAxisOutput,
    candidates: Any,
    valid: Any,
    *,
    fraction: float = RECIPROCAL_HEAD_FRACTION,
) -> ReciprocalHead:
    """Select the deterministic top fraction of reciprocal winners.

    The requested count is ``ceil(fraction * N)`` for every axis independently.
    If fewer reciprocal winners exist, all available reciprocal winners are
    returned and the caller can fail its fixed-coverage gate explicitly.
    """

    ids = np.asarray(candidates, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    if ids.ndim != 2 or mask.shape != ids.shape:
        raise ValueError("candidates and valid must be aligned N x K arrays")
    count, width = ids.shape
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    logits = output.edge_logits.detach().cpu().numpy().astype(np.float64)
    confidence = output.joint_confidence.detach().cpu().numpy().astype(np.float64)
    row_none = output.row_none_logits.detach().cpu().numpy().astype(np.float64)
    column_none = output.column_none_logits.detach().cpu().numpy().astype(np.float64)
    if logits.shape != (count, width) or confidence.shape != logits.shape:
        raise ValueError("output and candidate roster shapes changed")

    row_sources: list[int] = []
    row_targets: list[int] = []
    row_slots: list[int] = []
    for source in range(count):
        slots = np.flatnonzero(mask[source])
        if len(slots) == 0:
            continue
        best_slot = int(slots[np.argmax(logits[source, slots])])
        if logits[source, best_slot] <= row_none[source]:
            continue
        row_sources.append(source)
        row_targets.append(int(ids[source, best_slot]))
        row_slots.append(best_slot)

    dense = np.full((count, count), -np.inf, dtype=np.float64)
    for source in range(count):
        dense[source, ids[source, mask[source]]] = logits[source, mask[source]]
    column_sources = np.argmax(dense, axis=0)
    column_scores = dense[column_sources, np.arange(count)]
    column_sources = np.where(column_scores > column_none, column_sources, -1)

    reciprocal = np.zeros_like(mask)
    entries: list[tuple[float, int, int, int]] = []
    for source, target, slot in zip(
        row_sources, row_targets, row_slots, strict=True
    ):
        if column_sources[target] == source:
            reciprocal[source, slot] = True
            entries.append((float(confidence[source, slot]), source, target, slot))
    entries.sort(key=lambda value: (-value[0], value[1], value[2]))
    requested = max(1, math.ceil(fraction * count))
    chosen = entries[:requested]
    selected = np.zeros_like(mask)
    for _, source, _, slot in chosen:
        selected[source, slot] = True
    return ReciprocalHead(
        selected=selected,
        reciprocal=reciprocal,
        sources=np.asarray([value[1] for value in chosen], dtype=np.int32),
        targets=np.asarray([value[2] for value in chosen], dtype=np.int32),
        confidences=np.asarray([value[0] for value in chosen], dtype=np.float32),
        requested_count=requested,
    )


def joint_verifier_contract(model: JointReciprocalTriEmitterVerifier) -> dict[str, Any]:
    """Return the fixed architecture, objective and legal inference contract."""

    base = model.edge_verifier
    return {
        "architecture": "joint-reciprocal-tri-emitter-verifier-v1",
        "content_backbone": "vectorized-tri-emitter-relation-local-verifier-v1",
        "dino_projection_dim": base.dino_dim,
        "auxiliary_dim": base.auxiliary_dim,
        "width": base.width,
        "hidden": base.hidden,
        "learned_none": {"row": True, "column": True, "per_axis": True},
        "objective": {
            "row_cross_entropy_weight": 1.0,
            "column_cross_entropy_weight": 1.0,
            "confidence_bce_weight": CONFIDENCE_BCE_WEIGHT,
            "delta_l2_weight": DELTA_REGULARIZATION_WEIGHT,
        },
        "confidence": {
            "operator": "differentiable-soft-min-of-row-and-column-margins",
            "tau": SOFTMIN_TAU,
            "learned_bias": True,
            "learned_positive_temperature": True,
        },
        "deployment_head": {
            "reciprocal_row_and_column_top1": True,
            "fixed_fraction_per_axis_per_board": RECIPROCAL_HEAD_FRACTION,
            "threshold_sweep": False,
        },
        "candidate_identities_mutated": False,
        "absolute_position_or_source_identity": False,
        "pixels_modified": False,
        "output_material": "original upright tile identities only",
    }


__all__ = [
    "CONFIDENCE_BCE_WEIGHT",
    "DELTA_REGULARIZATION_WEIGHT",
    "JointAxisOutput",
    "JointLoss",
    "JointReciprocalTriEmitterVerifier",
    "JointTargets",
    "MASKED_LOGIT",
    "MINIMUM_TEMPERATURE",
    "RECIPROCAL_HEAD_FRACTION",
    "ReciprocalHead",
    "SOFTMIN_TAU",
    "build_joint_axis_output",
    "dense_two_sided_confidence",
    "exact_joint_targets",
    "fixed_fraction_reciprocal_head",
    "joint_assignment_loss",
    "joint_verifier_contract",
    "sparse_to_dense_logits",
]
