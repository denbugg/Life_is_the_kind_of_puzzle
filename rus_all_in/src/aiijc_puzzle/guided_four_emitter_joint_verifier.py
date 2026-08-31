"""Joint row/column verifier for an append-only guided fourth emitter.

The signed tri-emitter implementation is imported but never modified.  Its raw
baseline and learned relation residual remain the base path for legacy slots
``0..95``.  Slots ``96..127`` use the fixed guided row-z baseline, the same
relation-content backbone, and a new zero-initialised residual over seven
guided scalar features.  A learned per-axis row/column ``NONE`` objective is
applied jointly to the complete 128-slot roster.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.guided_fourth_emitter import (
    GUIDED_AUXILIARY_DIM,
    guided_fourth_pool_digest,
)
from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    MINIMUM_TEMPERATURE,
    RECIPROCAL_HEAD_FRACTION,
    JointAxisOutput,
    build_joint_axis_output,
)
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    DINO_PROJECTION_DIM,
    EMITTERS,
    TOP_K,
    TriEmitterEdgeVerifier,
    candidate_pool_digest,
)

LEGACY_SLOT_WIDTH = len(EMITTERS) * TOP_K
EXTENDED_SLOT_WIDTH = (len(EMITTERS) + 1) * TOP_K
LEGACY_TARGET_FREE_KEYS = frozenset(
    {
        "raw_sides",
        "dino_sides",
        "candidates",
        "valid",
        "auxiliary",
        "raw_baseline",
        "emitter_topk",
    }
)
GUIDED_SIDECAR_KEYS = frozenset(
    {
        "candidates",
        "valid",
        "legacy_slot",
        "guided_auxiliary",
        "guided_baseline",
        "emitter_topk",
        "legacy_identity_digest_ascii",
        "identity_digest_ascii",
    }
)


@dataclass(frozen=True)
class FourEmitterTargetFreeCase:
    """One label-free 128-slot cache consumer payload."""

    raw_sides: np.ndarray
    dino_sides: np.ndarray
    candidates: np.ndarray
    valid: np.ndarray
    legacy_slot: np.ndarray
    legacy_auxiliary: np.ndarray
    legacy_raw_baseline: np.ndarray
    guided_auxiliary: np.ndarray
    guided_baseline: np.ndarray
    emitter_topk: np.ndarray
    legacy_identity_digest: str
    identity_digest: str


def _ascii_digest(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.shape != (64,) or array.dtype != np.uint8:
        raise ValueError(f"{field} must be one 64-byte uint8 ASCII digest")
    try:
        digest = bytes(array).decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} is not ASCII") from error
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return digest


def build_target_free_four_emitter_case(
    legacy: Mapping[str, np.ndarray],
    sidecar: Mapping[str, np.ndarray],
) -> FourEmitterTargetFreeCase:
    """Join immutable legacy content with a target-free guided sidecar.

    Any label-bearing or unexpected key fails closed.  Legacy identities,
    validity, top-k membership, scalar features and baseline are copied into
    their original slot indices without recomputation.
    """

    if set(legacy) != LEGACY_TARGET_FREE_KEYS:
        raise ValueError("legacy consumer input must contain exactly seven target-free keys")
    if set(sidecar) != GUIDED_SIDECAR_KEYS:
        raise ValueError("guided consumer input must contain exactly eight target-free keys")
    old_candidates = np.asarray(legacy["candidates"])
    old_valid = np.asarray(legacy["valid"])
    candidates = np.asarray(sidecar["candidates"])
    valid = np.asarray(sidecar["valid"])
    if old_candidates.ndim != 3 or old_candidates.shape[0] != 2:
        raise ValueError("legacy candidates must have shape 2 x N x 96")
    count = old_candidates.shape[1]
    if old_candidates.shape != (2, count, LEGACY_SLOT_WIDTH):
        raise ValueError("legacy candidate width changed")
    if candidates.shape != (2, count, EXTENDED_SLOT_WIDTH):
        raise ValueError("guided candidate width changed")
    if old_valid.shape != old_candidates.shape or old_valid.dtype != np.bool_:
        raise ValueError("legacy valid mask changed")
    if valid.shape != candidates.shape or valid.dtype != np.bool_:
        raise ValueError("guided valid mask changed")
    if old_candidates.dtype not in (np.int32, np.int64) or candidates.dtype not in (
        np.int32,
        np.int64,
    ):
        raise ValueError("candidate identities must be integer")
    if not np.array_equal(candidates[..., :LEGACY_SLOT_WIDTH], old_candidates):
        raise ValueError("guided sidecar changed legacy candidate slots")
    if not np.array_equal(valid[..., :LEGACY_SLOT_WIDTH], old_valid):
        raise ValueError("guided sidecar changed legacy validity slots")

    legacy_topk = np.asarray(legacy["emitter_topk"])
    emitter_topk = np.asarray(sidecar["emitter_topk"])
    if legacy_topk.shape != (len(EMITTERS), 2, count, TOP_K):
        raise ValueError("legacy emitter top-k shape changed")
    if emitter_topk.shape != (len(EMITTERS) + 1, 2, count, TOP_K):
        raise ValueError("guided emitter top-k shape changed")
    if not np.array_equal(emitter_topk[: len(EMITTERS)], legacy_topk):
        raise ValueError("guided sidecar changed legacy emitter top-k identities")

    old_auxiliary = np.asarray(legacy["auxiliary"])
    old_baseline = np.asarray(legacy["raw_baseline"])
    if old_auxiliary.shape != (2, count, LEGACY_SLOT_WIDTH, AUXILIARY_DIM):
        raise ValueError("legacy auxiliary shape changed")
    if old_baseline.shape != (2, count, LEGACY_SLOT_WIDTH):
        raise ValueError("legacy raw baseline shape changed")
    guided_auxiliary = np.asarray(sidecar["guided_auxiliary"])
    guided_baseline = np.asarray(sidecar["guided_baseline"])
    if guided_auxiliary.shape != (
        2,
        count,
        EXTENDED_SLOT_WIDTH,
        GUIDED_AUXILIARY_DIM,
    ):
        raise ValueError("guided auxiliary shape changed")
    if guided_baseline.shape != (2, count, EXTENDED_SLOT_WIDTH):
        raise ValueError("guided baseline shape changed")
    for name, value in (
        ("legacy auxiliary", old_auxiliary),
        ("legacy raw baseline", old_baseline),
        ("guided auxiliary", guided_auxiliary),
        ("guided baseline", guided_baseline),
    ):
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")

    raw_sides = np.asarray(legacy["raw_sides"])
    dino_sides = np.asarray(legacy["dino_sides"])
    if raw_sides.ndim != 4 or raw_sides.shape[:2] != (4, count) or raw_sides.shape[-1] != 6:
        raise ValueError("legacy raw side sequence shape changed")
    if (
        dino_sides.ndim != 4
        or dino_sides.shape[:2] != (4, count)
        or dino_sides.shape[-1] != DINO_PROJECTION_DIM
    ):
        raise ValueError("legacy DINO side sequence shape changed")
    if not np.isfinite(raw_sides).all() or not np.isfinite(dino_sides).all():
        raise ValueError("legacy content sequences must be finite")

    legacy_slot = np.asarray(sidecar["legacy_slot"])
    if legacy_slot.shape != candidates.shape or legacy_slot.dtype not in (
        np.int16,
        np.int32,
        np.int64,
    ):
        raise ValueError("legacy slot mapping changed")
    expected_slot = np.where(
        old_valid,
        np.arange(LEGACY_SLOT_WIDTH, dtype=np.int16),
        -1,
    )
    if not np.array_equal(legacy_slot[..., :LEGACY_SLOT_WIDTH], expected_slot):
        raise ValueError("legacy slot mapping does not preserve old indices")
    if np.any(legacy_slot[..., LEGACY_SLOT_WIDTH:] != -1):
        raise ValueError("guided-only identity unexpectedly maps to a legacy slot")
    if np.any(valid & ((candidates < 0) | (candidates >= count))):
        raise ValueError("four-emitter candidate identity is out of range")
    expected_membership = (legacy_slot >= 0).astype(guided_auxiliary.dtype)
    if not np.array_equal(
        guided_auxiliary[..., -1][valid], expected_membership[valid]
    ):
        raise ValueError("guided legacy-membership feature changed")
    for axis in range(2):
        for source in range(count):
            row = candidates[axis, source, valid[axis, source]]
            if len(row) != len(np.unique(row)):
                raise ValueError("four-emitter candidate row contains duplicates")
            raw = set(emitter_topk[0, axis, source].tolist())
            if not raw.issubset(set(row.tolist())):
                raise ValueError("four-emitter candidate row dropped raw top32")

    expected_legacy_digest = candidate_pool_digest(old_candidates, old_valid, legacy_topk)
    legacy_digest = _ascii_digest(
        sidecar["legacy_identity_digest_ascii"], field="legacy identity digest"
    )
    if legacy_digest != expected_legacy_digest:
        raise ValueError("legacy identity digest mismatch")
    identity_digest = _ascii_digest(
        sidecar["identity_digest_ascii"], field="four-emitter identity digest"
    )
    expected_identity_digest = guided_fourth_pool_digest(
        candidates, valid, legacy_slot, emitter_topk
    )
    if identity_digest != expected_identity_digest:
        raise ValueError("four-emitter identity digest mismatch")

    legacy_auxiliary = np.zeros(
        (2, count, EXTENDED_SLOT_WIDTH, AUXILIARY_DIM), dtype=np.float32
    )
    legacy_raw_baseline = np.full(
        (2, count, EXTENDED_SLOT_WIDTH), -1e4, dtype=np.float32
    )
    legacy_auxiliary[..., :LEGACY_SLOT_WIDTH, :] = old_auxiliary
    legacy_raw_baseline[..., :LEGACY_SLOT_WIDTH] = old_baseline
    return FourEmitterTargetFreeCase(
        raw_sides=np.ascontiguousarray(raw_sides, dtype=np.float32),
        dino_sides=np.ascontiguousarray(dino_sides, dtype=np.float32),
        candidates=np.ascontiguousarray(candidates, dtype=np.int32),
        valid=np.ascontiguousarray(valid),
        legacy_slot=np.ascontiguousarray(legacy_slot, dtype=np.int16),
        legacy_auxiliary=np.ascontiguousarray(legacy_auxiliary),
        legacy_raw_baseline=np.ascontiguousarray(legacy_raw_baseline),
        guided_auxiliary=np.ascontiguousarray(guided_auxiliary, dtype=np.float32),
        guided_baseline=np.ascontiguousarray(guided_baseline, dtype=np.float32),
        emitter_topk=np.ascontiguousarray(emitter_topk, dtype=np.int32),
        legacy_identity_digest=legacy_digest,
        identity_digest=identity_digest,
    )


class _GuidedAuxiliaryResidual(nn.Module):
    def __init__(self, *, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("guided residual width must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(GUIDED_AUXILIARY_DIM),
            nn.Linear(GUIDED_AUXILIARY_DIM, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class GuidedFourEmitterJointVerifier(nn.Module):
    """Legacy relation scorer plus guided residual and joint learned NONE."""

    def __init__(
        self,
        *,
        dino_dim: int = DINO_PROJECTION_DIM,
        width: int = 32,
        hidden: int = 96,
        guided_width: int = 16,
        initial_none_logit: float = 0.0,
        initial_confidence_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_none_logit):
            raise ValueError("initial NONE logit must be finite")
        if initial_confidence_temperature <= MINIMUM_TEMPERATURE:
            raise ValueError("initial confidence temperature is too small")
        self.edge_verifier = TriEmitterEdgeVerifier(
            dino_dim=dino_dim,
            auxiliary_dim=AUXILIARY_DIM,
            width=width,
            hidden=hidden,
        )
        self.guided_residual = _GuidedAuxiliaryResidual(width=guided_width)
        self.row_none_logits = nn.Parameter(torch.full((2,), float(initial_none_logit)))
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
        legacy_slot: torch.Tensor,
        legacy_auxiliary: torch.Tensor,
        legacy_raw_baseline: torch.Tensor,
        guided_auxiliary: torch.Tensor,
        guided_baseline: torch.Tensor,
        *,
        direction: int,
    ) -> JointAxisOutput:
        """Score one right/down axis over the fixed 128-slot candidate roster."""

        if direction not in (0, 1):
            raise ValueError("direction must be 0 (right) or 1 (down)")
        if candidates.ndim != 2 or candidates.shape[1] != EXTENDED_SLOT_WIDTH:
            raise ValueError("candidates must have shape N x 128")
        count = candidates.shape[0]
        if valid.shape != candidates.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be an aligned boolean mask")
        if legacy_slot.shape != candidates.shape or legacy_slot.dtype not in (
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("legacy_slot must be an aligned integer tensor")
        if legacy_auxiliary.shape != (
            count,
            EXTENDED_SLOT_WIDTH,
            AUXILIARY_DIM,
        ):
            raise ValueError("legacy auxiliary tensor shape changed")
        if legacy_raw_baseline.shape != candidates.shape:
            raise ValueError("legacy raw baseline tensor shape changed")
        if guided_auxiliary.shape != (
            count,
            EXTENDED_SLOT_WIDTH,
            GUIDED_AUXILIARY_DIM,
        ):
            raise ValueError("guided auxiliary tensor shape changed")
        if guided_baseline.shape != candidates.shape:
            raise ValueError("guided baseline tensor shape changed")
        for name, value in (
            ("legacy auxiliary", legacy_auxiliary),
            ("legacy raw baseline", legacy_raw_baseline),
            ("guided auxiliary", guided_auxiliary),
            ("guided baseline", guided_baseline),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        legacy_present = legacy_slot >= 0
        if not torch.equal(legacy_present[:, :LEGACY_SLOT_WIDTH], valid[:, :LEGACY_SLOT_WIDTH]):
            raise ValueError("legacy slot presence differs from old validity")
        if torch.any(legacy_slot[:, LEGACY_SLOT_WIDTH:] != -1):
            raise ValueError("guided-only slots must not map to legacy slots")
        expected_membership = legacy_present.to(guided_auxiliary.dtype)
        if not torch.equal(
            guided_auxiliary[..., -1][valid], expected_membership[valid]
        ):
            raise ValueError("guided legacy-membership feature changed")

        hybrid_baseline = torch.where(
            legacy_present, legacy_raw_baseline, guided_baseline
        )
        anchors = torch.arange(count, device=candidates.device)
        directions = torch.full(
            (count,), direction, dtype=torch.long, device=candidates.device
        )
        base_logits, base_delta = self.edge_verifier(
            raw_sides,
            dino_sides,
            anchors,
            candidates,
            valid,
            directions,
            legacy_auxiliary,
            hybrid_baseline,
        )
        guided_delta = self.guided_residual(guided_auxiliary).masked_fill(~valid, 0.0)
        edge_logits = base_logits + guided_delta
        total_delta = base_delta + guided_delta
        row_none = self.row_none_logits[direction].expand(count)
        column_none = self.column_none_logits[direction].expand(count)
        return build_joint_axis_output(
            candidates,
            valid,
            edge_logits,
            total_delta,
            row_none,
            column_none,
            self.confidence_bias,
            self.confidence_temperature,
        )


def transplant_legacy_joint_state(
    model: GuidedFourEmitterJointVerifier,
    legacy_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Load one exact tri-joint endpoint while leaving the new head at zero."""

    model_keys = set(model.state_dict())
    guided_keys = {key for key in model_keys if key.startswith("guided_residual.")}
    expected_legacy = model_keys - guided_keys
    observed = set(legacy_state_dict)
    if observed != expected_legacy:
        missing = sorted(expected_legacy - observed)
        extra = sorted(observed - expected_legacy)
        raise ValueError(f"legacy state keys changed; missing={missing}, extra={extra}")
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key in guided_keys
    }
    incompatible = model.load_state_dict(dict(legacy_state_dict), strict=False)
    if set(incompatible.missing_keys) != guided_keys or incompatible.unexpected_keys:
        raise RuntimeError("legacy state transplant contract changed")
    after = model.state_dict()
    for key, value in before.items():
        if not torch.equal(after[key], value):
            raise RuntimeError("legacy transplant changed the guided zero-init head")
    return {
        "legacy_key_count": len(expected_legacy),
        "new_guided_key_count": len(guided_keys),
    }


def four_emitter_joint_contract(model: GuidedFourEmitterJointVerifier) -> dict[str, Any]:
    """Return the implementation contract without signing a real protocol."""

    base = model.edge_verifier
    return {
        "architecture": "joint-reciprocal-guided-four-emitter-verifier-v1",
        "candidate_slots": {
            "total": EXTENDED_SLOT_WIDTH,
            "legacy": [0, LEGACY_SLOT_WIDTH - 1],
            "guided_append_only": [LEGACY_SLOT_WIDTH, EXTENDED_SLOT_WIDTH - 1],
            "raw_top32_retained": True,
        },
        "legacy_path": {
            "content_backbone": "vectorized-tri-emitter-relation-local-verifier-v1",
            "raw_baseline_retained": True,
            "learned_relation_residual_retained": True,
            "exact_joint_state_transplant_supported": True,
            "dino_projection_dim": base.dino_dim,
            "auxiliary_dim": base.auxiliary_dim,
            "width": base.width,
            "hidden": base.hidden,
        },
        "guided_path": {
            "auxiliary_dim": GUIDED_AUXILIARY_DIM,
            "append_only_candidate_identities": True,
            "new_residual_zero_initialised": True,
            "guided_row_z_baseline_for_new_slots": True,
        },
        "joint_assignment": {
            "learned_row_none_per_axis": True,
            "learned_column_none_per_axis": True,
            "row_cross_entropy": True,
            "column_cross_entropy": True,
            "confidence_bce": True,
            "delta_l2": True,
        },
        "deployment_head": {
            "fixed_reciprocal_fraction_per_axis_per_board": RECIPROCAL_HEAD_FRACTION,
            "threshold_sweep": False,
        },
        "absolute_position_or_source_identity": False,
        "candidate_identities_mutated": False,
        "pixels_modified": False,
        "output_material": "original upright tile identities only",
        "real_protocol_signed": False,
    }


__all__ = [
    "EXTENDED_SLOT_WIDTH",
    "GUIDED_SIDECAR_KEYS",
    "LEGACY_SLOT_WIDTH",
    "LEGACY_TARGET_FREE_KEYS",
    "FourEmitterTargetFreeCase",
    "GuidedFourEmitterJointVerifier",
    "build_target_free_four_emitter_case",
    "four_emitter_joint_contract",
    "transplant_legacy_joint_state",
]
