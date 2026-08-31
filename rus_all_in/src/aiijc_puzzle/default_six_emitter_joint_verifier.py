"""Target-free rank-only consumer for the fixed default six-emitter roster.

The legacy tri-v2 scorer is used only for identities that already have an
exact frozen tri slot.  Guided-only identities use only their frozen guided
baseline/features.  Wiener/Haar-only identities have a neutral fixed baseline
and are scored by a learned rank-membership residual.  Missing tri auxiliary
statistics are never fabricated for novel identities.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aiijc_puzzle.guided_four_emitter_joint_verifier import (
    GUIDED_SIDECAR_KEYS,
    LEGACY_SLOT_WIDTH,
    LEGACY_TARGET_FREE_KEYS,
    build_target_free_four_emitter_case,
)
from aiijc_puzzle.guided_fourth_emitter import GUIDED_AUXILIARY_DIM
from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    MINIMUM_TEMPERATURE,
    RECIPROCAL_HEAD_FRACTION,
    JointAxisOutput,
    build_joint_axis_output,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    DINO_PROJECTION_DIM,
    TOP_K,
    TriEmitterEdgeVerifier,
)

FROZEN_WAVELET_EMITTER_ORDER = (
    "raw",
    "adapter1600",
    "dinov2",
    "guided",
    "wiener",
    "local_rank",
    "haar_bayesshrink",
)
DEFAULT_EMITTER_ORDER = (
    "raw",
    "adapter1600",
    "dinov2",
    "guided",
    "wiener",
    "haar_bayesshrink",
)
DEFAULT_WAVELET_SOURCE_INDICES = (0, 1, 2, 3, 4, 6)
LOCAL_RANK_SOURCE_INDEX = 5
SUPPLY_FEATURE_DIM = 2 * len(DEFAULT_EMITTER_ORDER)
THEORETICAL_SLOT_CAP = len(DEFAULT_EMITTER_ORDER) * TOP_K
GUIDED_SLOT_CAP = 4 * TOP_K
WAVELET_SIDECAR_KEYS = frozenset({"emitter_topk"})

TARGET_FREE_CASE_KEYS = frozenset(
    {
        "raw_sides",
        "dino_sides",
        "candidates",
        "valid",
        "legacy_slot",
        "guided_slot",
        "legacy_auxiliary",
        "legacy_raw_baseline",
        "guided_auxiliary",
        "guided_baseline",
        "supply_features",
        "emitter_topk",
        "legacy_identity_digest_ascii",
        "guided_identity_digest_ascii",
        "identity_digest_ascii",
    }
)


@dataclass(frozen=True)
class DefaultSixTargetFreeCase:
    """One compact label-free default-six consumer payload."""

    raw_sides: np.ndarray
    dino_sides: np.ndarray
    candidates: np.ndarray
    valid: np.ndarray
    legacy_slot: np.ndarray
    guided_slot: np.ndarray
    legacy_auxiliary: np.ndarray
    legacy_raw_baseline: np.ndarray
    guided_auxiliary: np.ndarray
    guided_baseline: np.ndarray
    supply_features: np.ndarray
    emitter_topk: np.ndarray
    legacy_identity_digest: str
    guided_identity_digest: str
    identity_digest: str


def _update_array_digest(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def default_six_identity_digest(
    *,
    candidates: np.ndarray,
    valid: np.ndarray,
    legacy_slot: np.ndarray,
    guided_slot: np.ndarray,
    supply_features: np.ndarray,
    emitter_topk: np.ndarray,
    legacy_identity_digest: str,
    guided_identity_digest: str,
) -> str:
    """Hash identities, mappings and rank features, never labels or pixels."""

    digest = hashlib.sha256()
    for name, value in (
        ("candidates", candidates),
        ("valid", valid),
        ("legacy_slot", legacy_slot),
        ("guided_slot", guided_slot),
        ("supply_features", supply_features),
        ("emitter_topk", emitter_topk),
    ):
        _update_array_digest(digest, name, value)
    digest.update(legacy_identity_digest.encode())
    digest.update(guided_identity_digest.encode())
    return digest.hexdigest()


def _ascii_digest_array(value: str) -> np.ndarray:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("identity digest must be lowercase SHA-256")
    return np.frombuffer(value.encode(), dtype=np.uint8).copy()


def _decode_ascii_digest(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != (64,) or array.dtype != np.uint8:
        raise ValueError(f"{name} must be one 64-byte uint8 array")
    try:
        decoded = bytes(array).decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not ASCII") from error
    _ascii_digest_array(decoded)
    return decoded


def _validate_topk(topk: np.ndarray, *, count: int) -> None:
    if topk.shape != (len(DEFAULT_EMITTER_ORDER), 2, count, TOP_K):
        raise ValueError("default-six emitter top-k shape changed")
    if topk.dtype != np.int32:
        raise ValueError("default-six emitter top-k must be int32")
    if np.any((topk < 0) | (topk >= count)):
        raise ValueError("default-six emitter top-k identity is out of range")
    for emitter in range(len(DEFAULT_EMITTER_ORDER)):
        for axis in range(2):
            for source in range(count):
                row = topk[emitter, axis, source]
                if source in row:
                    raise ValueError("emitter top-k contains its source identity")
                if len(np.unique(row)) != TOP_K:
                    raise ValueError("emitter top-k contains duplicates")


def _validate_frozen_wavelet_topk(topk: np.ndarray, *, count: int) -> None:
    if topk.shape != (len(FROZEN_WAVELET_EMITTER_ORDER), 2, count, TOP_K):
        raise ValueError("wavelet sidecar must be int32[7,2,N,32]")
    if topk.dtype != np.int32:
        raise ValueError("wavelet emitter top-k must be int32")
    if np.any((topk < 0) | (topk >= count)):
        raise ValueError("wavelet emitter top-k identity is out of range")
    for emitter in range(len(FROZEN_WAVELET_EMITTER_ORDER)):
        for axis in range(2):
            for source in range(count):
                row = topk[emitter, axis, source]
                if source in row or len(np.unique(row)) != TOP_K:
                    raise ValueError("wavelet emitter top-k row is not unique/self-free")


def _supply_features(
    candidates: np.ndarray,
    valid: np.ndarray,
    emitter_topk: np.ndarray,
) -> np.ndarray:
    count = candidates.shape[1]
    features = np.zeros((*candidates.shape, SUPPLY_FEATURE_DIM), dtype=np.float16)
    for emitter in range(len(DEFAULT_EMITTER_ORDER)):
        for axis in range(2):
            for source in range(count):
                ranks = {
                    int(target): rank
                    for rank, target in enumerate(emitter_topk[emitter, axis, source])
                }
                for slot in np.flatnonzero(valid[axis, source]):
                    target = int(candidates[axis, source, slot])
                    rank = ranks.get(target)
                    if rank is None:
                        continue
                    features[axis, source, slot, 2 * emitter] = 1.0
                    features[axis, source, slot, 2 * emitter + 1] = (
                        TOP_K - rank
                    ) / TOP_K
    return np.ascontiguousarray(features)


def _validate_case(case: DefaultSixTargetFreeCase) -> None:
    candidates = np.asarray(case.candidates)
    valid = np.asarray(case.valid)
    if candidates.ndim != 3 or candidates.shape[:2] != (2, candidates.shape[1]):
        raise ValueError("candidates must have shape 2 x N x K")
    _, count, width = candidates.shape
    if not 1 <= width <= THEORETICAL_SLOT_CAP:
        raise ValueError("compact candidate width escaped the theoretical cap")
    if candidates.dtype != np.int32 or valid.dtype != np.bool_ or valid.shape != candidates.shape:
        raise ValueError("candidates/valid dtype or shape changed")

    raw_sides = np.asarray(case.raw_sides)
    dino_sides = np.asarray(case.dino_sides)
    if raw_sides.shape != (4, count, 20, 6) or raw_sides.dtype != np.float16:
        raise ValueError("raw sides must be float16[4,N,20,6]")
    if dino_sides.shape != (4, count, 14, DINO_PROJECTION_DIM) or dino_sides.dtype != np.float16:
        raise ValueError("DINO sides must be float16[4,N,14,16]")

    aligned = {
        "legacy_slot": (case.legacy_slot, np.int16, candidates.shape),
        "guided_slot": (case.guided_slot, np.int16, candidates.shape),
        "legacy_auxiliary": (
            case.legacy_auxiliary,
            np.float16,
            (*candidates.shape, AUXILIARY_DIM),
        ),
        "legacy_raw_baseline": (
            case.legacy_raw_baseline,
            np.float16,
            candidates.shape,
        ),
        "guided_auxiliary": (
            case.guided_auxiliary,
            np.float16,
            (*candidates.shape, GUIDED_AUXILIARY_DIM),
        ),
        "guided_baseline": (case.guided_baseline, np.float16, candidates.shape),
        "supply_features": (
            case.supply_features,
            np.float16,
            (*candidates.shape, SUPPLY_FEATURE_DIM),
        ),
    }
    values: dict[str, np.ndarray] = {}
    for name, (value, dtype, shape) in aligned.items():
        array = np.asarray(value)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{name} dtype or shape changed")
        values[name] = array
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")

    topk = np.asarray(case.emitter_topk)
    _validate_topk(topk, count=count)
    legacy_slot = values["legacy_slot"]
    guided_slot = values["guided_slot"]
    legacy_present = legacy_slot >= 0
    guided_present = guided_slot >= 0
    if np.any(legacy_present & ~guided_present):
        raise ValueError("legacy identity lacks a guided back-pointer")
    if np.any(legacy_slot[legacy_present] >= LEGACY_SLOT_WIDTH):
        raise ValueError("legacy slot escaped the frozen 96-slot pool")
    if np.any(guided_slot[guided_present] >= GUIDED_SLOT_CAP):
        raise ValueError("guided slot escaped the frozen 128-slot pool")
    if np.any(~valid & (candidates != -1)):
        raise ValueError("invalid candidate padding must be -1")
    if np.any(~valid & ((legacy_slot != -1) | (guided_slot != -1))):
        raise ValueError("invalid candidate padding has a source mapping")
    if np.any(valid & ((candidates < 0) | (candidates >= count))):
        raise ValueError("candidate identity is out of range")
    source_ids = np.arange(count, dtype=np.int32)[None, :, None]
    if np.any(valid & (candidates == source_ids)):
        raise ValueError("candidate union contains a self edge")

    for name in ("legacy_auxiliary", "legacy_raw_baseline"):
        if np.any(values[name][~legacy_present] != 0):
            raise ValueError(f"{name} fabricates evidence outside the tri union")
    for name in ("guided_auxiliary", "guided_baseline"):
        if np.any(values[name][~guided_present] != 0):
            raise ValueError(f"{name} fabricates evidence outside the guided union")
    if np.any(values["supply_features"][~valid] != 0):
        raise ValueError("invalid candidate padding has supply features")

    expected_features = _supply_features(candidates, valid, topk)
    if not np.array_equal(values["supply_features"], expected_features):
        raise ValueError("supply membership/rank features changed")
    for axis in range(2):
        for source in range(count):
            row = candidates[axis, source, valid[axis, source]]
            if len(row) != len(np.unique(row)):
                raise ValueError("candidate row contains duplicates")
            union = set(row.tolist())
            for emitter in range(len(DEFAULT_EMITTER_ORDER)):
                if not set(topk[emitter, axis, source].tolist()).issubset(union):
                    raise ValueError("candidate union dropped an emitter top-k identity")

    expected_digest = default_six_identity_digest(
        candidates=candidates,
        valid=valid,
        legacy_slot=legacy_slot,
        guided_slot=guided_slot,
        supply_features=values["supply_features"],
        emitter_topk=topk,
        legacy_identity_digest=case.legacy_identity_digest,
        guided_identity_digest=case.guided_identity_digest,
    )
    if case.identity_digest != expected_digest:
        raise ValueError("default-six identity digest mismatch")


def build_target_free_default_six_case(
    legacy: Mapping[str, np.ndarray],
    guided: Mapping[str, np.ndarray],
    wavelet: Mapping[str, np.ndarray],
) -> DefaultSixTargetFreeCase:
    """Compose the exact frozen default-six top-k roster without labels."""

    if set(legacy) != LEGACY_TARGET_FREE_KEYS:
        raise ValueError("legacy input must contain exactly seven target-free keys")
    if set(guided) != GUIDED_SIDECAR_KEYS:
        raise ValueError("guided input must contain exactly eight target-free keys")
    if set(wavelet) != WAVELET_SIDECAR_KEYS:
        raise ValueError("wavelet input must contain only emitter_topk")

    four = build_target_free_four_emitter_case(legacy, guided)
    count = four.candidates.shape[1]
    all_topk = np.asarray(wavelet["emitter_topk"])
    _validate_frozen_wavelet_topk(all_topk, count=count)
    if not np.array_equal(all_topk[:4], four.emitter_topk):
        raise ValueError("wavelet sidecar changed the frozen guided prefix")
    emitter_topk = np.ascontiguousarray(
        all_topk[np.asarray(DEFAULT_WAVELET_SOURCE_INDICES)], dtype=np.int32
    )
    _validate_topk(emitter_topk, count=count)

    rows: list[list[list[tuple[int, int]]]] = [[], []]
    max_width = 0
    for axis in range(2):
        for source in range(count):
            row: list[tuple[int, int]] = []
            seen: set[int] = set()
            for guided_slot in np.flatnonzero(four.valid[axis, source]):
                target = int(four.candidates[axis, source, guided_slot])
                if target in seen:
                    raise ValueError("guided sidecar contains duplicate valid identities")
                seen.add(target)
                row.append((target, int(guided_slot)))
            for emitter_index in (4, 5):
                for target_value in emitter_topk[emitter_index, axis, source]:
                    target = int(target_value)
                    if target not in seen:
                        seen.add(target)
                        row.append((target, -1))
            rows[axis].append(row)
            max_width = max(max_width, len(row))
    if max_width > THEORETICAL_SLOT_CAP:
        raise RuntimeError("default-six union exceeded its theoretical width")

    shape = (2, count, max_width)
    candidates = np.full(shape, -1, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    legacy_slot = np.full(shape, -1, dtype=np.int16)
    guided_slot = np.full(shape, -1, dtype=np.int16)
    legacy_auxiliary = np.zeros((*shape, AUXILIARY_DIM), dtype=np.float16)
    legacy_raw_baseline = np.zeros(shape, dtype=np.float16)
    guided_auxiliary = np.zeros((*shape, GUIDED_AUXILIARY_DIM), dtype=np.float16)
    guided_baseline = np.zeros(shape, dtype=np.float16)

    legacy_aux_source = np.asarray(legacy["auxiliary"], dtype=np.float16)
    legacy_base_source = np.asarray(legacy["raw_baseline"], dtype=np.float16)
    guided_aux_source = np.asarray(guided["guided_auxiliary"], dtype=np.float16)
    guided_base_source = np.asarray(guided["guided_baseline"], dtype=np.float16)
    for axis in range(2):
        for source, row in enumerate(rows[axis]):
            for slot, (target, source_guided_slot) in enumerate(row):
                candidates[axis, source, slot] = target
                valid[axis, source, slot] = True
                if source_guided_slot < 0:
                    continue
                guided_slot[axis, source, slot] = source_guided_slot
                guided_auxiliary[axis, source, slot] = guided_aux_source[
                    axis, source, source_guided_slot
                ]
                guided_baseline[axis, source, slot] = guided_base_source[
                    axis, source, source_guided_slot
                ]
                source_legacy_slot = int(
                    four.legacy_slot[axis, source, source_guided_slot]
                )
                if source_legacy_slot < 0:
                    continue
                legacy_slot[axis, source, slot] = source_legacy_slot
                legacy_auxiliary[axis, source, slot] = legacy_aux_source[
                    axis, source, source_legacy_slot
                ]
                legacy_raw_baseline[axis, source, slot] = legacy_base_source[
                    axis, source, source_legacy_slot
                ]

    supply_features = _supply_features(candidates, valid, emitter_topk)
    identity_digest = default_six_identity_digest(
        candidates=candidates,
        valid=valid,
        legacy_slot=legacy_slot,
        guided_slot=guided_slot,
        supply_features=supply_features,
        emitter_topk=emitter_topk,
        legacy_identity_digest=four.legacy_identity_digest,
        guided_identity_digest=four.identity_digest,
    )
    case = DefaultSixTargetFreeCase(
        raw_sides=np.ascontiguousarray(legacy["raw_sides"], dtype=np.float16),
        dino_sides=np.ascontiguousarray(legacy["dino_sides"], dtype=np.float16),
        candidates=candidates,
        valid=valid,
        legacy_slot=legacy_slot,
        guided_slot=guided_slot,
        legacy_auxiliary=legacy_auxiliary,
        legacy_raw_baseline=legacy_raw_baseline,
        guided_auxiliary=guided_auxiliary,
        guided_baseline=guided_baseline,
        supply_features=supply_features,
        emitter_topk=emitter_topk,
        legacy_identity_digest=four.legacy_identity_digest,
        guided_identity_digest=four.identity_digest,
        identity_digest=identity_digest,
    )
    _validate_case(case)
    return case


def target_free_case_arrays(case: DefaultSixTargetFreeCase) -> dict[str, np.ndarray]:
    """Return the exact label-free archive payload after full validation."""

    _validate_case(case)
    return {
        "raw_sides": np.ascontiguousarray(case.raw_sides),
        "dino_sides": np.ascontiguousarray(case.dino_sides),
        "candidates": np.ascontiguousarray(case.candidates),
        "valid": np.ascontiguousarray(case.valid),
        "legacy_slot": np.ascontiguousarray(case.legacy_slot),
        "guided_slot": np.ascontiguousarray(case.guided_slot),
        "legacy_auxiliary": np.ascontiguousarray(case.legacy_auxiliary),
        "legacy_raw_baseline": np.ascontiguousarray(case.legacy_raw_baseline),
        "guided_auxiliary": np.ascontiguousarray(case.guided_auxiliary),
        "guided_baseline": np.ascontiguousarray(case.guided_baseline),
        "supply_features": np.ascontiguousarray(case.supply_features),
        "emitter_topk": np.ascontiguousarray(case.emitter_topk),
        "legacy_identity_digest_ascii": _ascii_digest_array(
            case.legacy_identity_digest
        ),
        "guided_identity_digest_ascii": _ascii_digest_array(
            case.guided_identity_digest
        ),
        "identity_digest_ascii": _ascii_digest_array(case.identity_digest),
    }


def freeze_target_free_case_exclusive(
    path: Path,
    case: DefaultSixTargetFreeCase,
) -> dict[str, str]:
    """Write one target-free NPZ exclusively and return its immutable record."""

    arrays = target_free_case_arrays(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "identity_digest": case.identity_digest,
    }


def load_frozen_target_free_case(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity_digest: str,
) -> DefaultSixTargetFreeCase:
    """Load one exact-key frozen archive and reject labels or extra arrays."""

    if sha256_file(path) != expected_sha256:
        raise ValueError("frozen default-six archive SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != TARGET_FREE_CASE_KEYS:
            raise ValueError("frozen default-six archive keys changed")
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    case = DefaultSixTargetFreeCase(
        raw_sides=arrays["raw_sides"],
        dino_sides=arrays["dino_sides"],
        candidates=arrays["candidates"],
        valid=arrays["valid"],
        legacy_slot=arrays["legacy_slot"],
        guided_slot=arrays["guided_slot"],
        legacy_auxiliary=arrays["legacy_auxiliary"],
        legacy_raw_baseline=arrays["legacy_raw_baseline"],
        guided_auxiliary=arrays["guided_auxiliary"],
        guided_baseline=arrays["guided_baseline"],
        supply_features=arrays["supply_features"],
        emitter_topk=arrays["emitter_topk"],
        legacy_identity_digest=_decode_ascii_digest(
            arrays["legacy_identity_digest_ascii"], name="legacy identity digest"
        ),
        guided_identity_digest=_decode_ascii_digest(
            arrays["guided_identity_digest_ascii"], name="guided identity digest"
        ),
        identity_digest=_decode_ascii_digest(
            arrays["identity_digest_ascii"], name="default-six identity digest"
        ),
    )
    _validate_case(case)
    if case.identity_digest != expected_identity_digest:
        raise ValueError("frozen default-six identity digest changed")
    return case


class _SmallResidual(nn.Module):
    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        if input_dim <= 0 or width <= 0:
            raise ValueError("residual dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class DefaultSixEmitterJointVerifier(nn.Module):
    """Frozen tri-v2 legacy path plus guided and rank-only supply residuals."""

    def __init__(
        self,
        *,
        dino_dim: int = DINO_PROJECTION_DIM,
        width: int = 32,
        hidden: int = 96,
        residual_width: int = 16,
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
        self.guided_residual = _SmallResidual(
            GUIDED_AUXILIARY_DIM, residual_width
        )
        self.supply_residual = _SmallResidual(SUPPLY_FEATURE_DIM, residual_width)
        self.row_none_logits = nn.Parameter(torch.full((2,), initial_none_logit))
        self.column_none_logits = nn.Parameter(torch.full((2,), initial_none_logit))
        self.confidence_bias = nn.Parameter(torch.zeros(()))
        inverse = math.log(
            math.expm1(initial_confidence_temperature - MINIMUM_TEMPERATURE)
        )
        self.raw_confidence_temperature = nn.Parameter(torch.tensor(inverse))
        self.freeze_legacy_backbone()

    @property
    def confidence_temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_confidence_temperature) + MINIMUM_TEMPERATURE

    def freeze_legacy_backbone(self) -> None:
        for parameter in self.edge_verifier.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        raw_sides: torch.Tensor,
        dino_sides: torch.Tensor,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        legacy_slot: torch.Tensor,
        guided_slot: torch.Tensor,
        legacy_auxiliary: torch.Tensor,
        legacy_raw_baseline: torch.Tensor,
        guided_auxiliary: torch.Tensor,
        guided_baseline: torch.Tensor,
        supply_features: torch.Tensor,
        *,
        direction: int,
    ) -> JointAxisOutput:
        if direction not in (0, 1):
            raise ValueError("direction must be 0 (right) or 1 (down)")
        if candidates.ndim != 2 or candidates.shape[1] > THEORETICAL_SLOT_CAP:
            raise ValueError("candidates must have shape N x K with K <= 192")
        count, candidate_count = candidates.shape
        if valid.shape != candidates.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be an aligned boolean mask")
        for name, value in (("legacy_slot", legacy_slot), ("guided_slot", guided_slot)):
            if value.shape != candidates.shape or value.dtype not in (
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError(f"{name} must be an aligned integer tensor")
        shapes = {
            "legacy_auxiliary": (legacy_auxiliary, (count, candidate_count, AUXILIARY_DIM)),
            "legacy_raw_baseline": (legacy_raw_baseline, candidates.shape),
            "guided_auxiliary": (
                guided_auxiliary,
                (count, candidate_count, GUIDED_AUXILIARY_DIM),
            ),
            "guided_baseline": (guided_baseline, candidates.shape),
            "supply_features": (
                supply_features,
                (count, candidate_count, SUPPLY_FEATURE_DIM),
            ),
        }
        for name, (value, shape) in shapes.items():
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} shape or finiteness changed")

        legacy_present = (legacy_slot >= 0) & valid
        guided_present = (guided_slot >= 0) & valid
        if torch.any(legacy_present & ~guided_present):
            raise ValueError("legacy identity lacks guided provenance")
        if torch.any(legacy_auxiliary[~legacy_present] != 0):
            raise ValueError("legacy auxiliary fabricates novel-edge evidence")
        if torch.any(legacy_raw_baseline[~legacy_present] != 0):
            raise ValueError("legacy baseline fabricates novel-edge evidence")
        if torch.any(guided_auxiliary[~guided_present] != 0):
            raise ValueError("guided auxiliary fabricates novel-edge evidence")
        if torch.any(guided_baseline[~guided_present] != 0):
            raise ValueError("guided baseline fabricates novel-edge evidence")

        anchors = torch.arange(count, device=candidates.device)
        directions = torch.full(
            (count,), direction, dtype=torch.long, device=candidates.device
        )
        # The frozen legacy head must never index a novel target.  Invalid
        # positions are replaced by the source identity before the content
        # gather, and remain masked from its public output.
        legacy_candidates = torch.where(
            legacy_present, candidates, anchors[:, None]
        )
        legacy_logits, legacy_delta = self.edge_verifier(
            raw_sides,
            dino_sides,
            anchors,
            legacy_candidates,
            legacy_present,
            directions,
            legacy_auxiliary,
            legacy_raw_baseline,
        )
        zero = legacy_logits.new_zeros(())
        base_logits = torch.where(legacy_present, legacy_logits, zero)
        guided_only = guided_present & ~legacy_present
        base_logits = base_logits + torch.where(guided_only, guided_baseline, zero)

        guided_delta = self.guided_residual(
            guided_auxiliary.to(base_logits.dtype)
        ).masked_fill(~guided_present, 0.0)
        supply_delta = self.supply_residual(
            supply_features.to(base_logits.dtype)
        ).masked_fill(~valid, 0.0)
        edge_logits = (base_logits + guided_delta + supply_delta).masked_fill(
            ~valid, -1e4
        )
        total_delta = (
            torch.where(legacy_present, legacy_delta, zero)
            + guided_delta
            + supply_delta
        )
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


def transplant_tri_v2_state(
    model: DefaultSixEmitterJointVerifier,
    legacy_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Transplant the exact tri-v2 state and retain both zero-init residuals."""

    model_keys = set(model.state_dict())
    new_keys = {
        key
        for key in model_keys
        if key.startswith(("guided_residual.", "supply_residual."))
    }
    expected_legacy = model_keys - new_keys
    observed = set(legacy_state_dict)
    if observed != expected_legacy:
        missing = sorted(expected_legacy - observed)
        extra = sorted(observed - expected_legacy)
        raise ValueError(f"tri-v2 state keys changed; missing={missing}, extra={extra}")
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key in new_keys
    }
    incompatible = model.load_state_dict(dict(legacy_state_dict), strict=False)
    if set(incompatible.missing_keys) != new_keys or incompatible.unexpected_keys:
        raise RuntimeError("tri-v2 state transplant contract changed")
    for key, value in before.items():
        if not torch.equal(model.state_dict()[key], value):
            raise RuntimeError("tri-v2 transplant changed a new zero-init head")
    model.freeze_legacy_backbone()
    return {
        "legacy_key_count": len(expected_legacy),
        "new_residual_key_count": len(new_keys),
    }


def parameter_counts(model: DefaultSixEmitterJointVerifier) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def default_six_contract(model: DefaultSixEmitterJointVerifier) -> dict[str, Any]:
    """Return the unsigned architecture/cache contract."""

    return {
        "architecture": "joint-reciprocal-default-six-rank-only-v1",
        "emitter_order": list(DEFAULT_EMITTER_ORDER),
        "frozen_wavelet_source_indices": list(DEFAULT_WAVELET_SOURCE_INDICES),
        "local_rank": {
            "source_index": LOCAL_RANK_SOURCE_INDEX,
            "enabled": False,
        },
        "candidate_width": {
            "dynamic_compact": True,
            "theoretical_cap": THEORETICAL_SLOT_CAP,
        },
        "supply_features": {
            "dim": SUPPLY_FEATURE_DIM,
            "per_emitter": ["membership", "zero_based_rank_quality"],
            "direct_score_fusion": False,
        },
        "legacy_path": {
            "exact_tri_v2_state_transplant": True,
            "used_only_when_legacy_slot_present": True,
            "novel_identity_ever_indexed": False,
            "backbone_frozen": all(
                not parameter.requires_grad
                for parameter in model.edge_verifier.parameters()
            ),
        },
        "guided_only_path": {
            "frozen_guided_baseline": True,
            "frozen_guided_auxiliary": True,
            "tri_auxiliary_fabricated": False,
        },
        "wiener_haar_only_path": {
            "fixed_neutral_baseline": 0.0,
            "rank_membership_residual_only": True,
            "tri_or_guided_auxiliary_fabricated": False,
        },
        "rank_only_limitation": (
            "continuous Wiener/Haar scores and full tri score statistics are not "
            "frozen; using content for novel identities requires a separately "
            "rematerialised target-free cache and corrected architecture"
        ),
        "parameter_counts": parameter_counts(model),
        "joint_reject": {
            "learned_row_none_per_axis": True,
            "learned_column_none_per_axis": True,
            "confidence_bias_and_temperature": True,
        },
        "fixed_reciprocal_fraction": RECIPROCAL_HEAD_FRACTION,
        "real_protocol_signed": False,
    }


__all__ = [
    "DEFAULT_EMITTER_ORDER",
    "DEFAULT_WAVELET_SOURCE_INDICES",
    "FROZEN_WAVELET_EMITTER_ORDER",
    "GUIDED_SLOT_CAP",
    "LOCAL_RANK_SOURCE_INDEX",
    "SUPPLY_FEATURE_DIM",
    "TARGET_FREE_CASE_KEYS",
    "THEORETICAL_SLOT_CAP",
    "WAVELET_SIDECAR_KEYS",
    "DefaultSixEmitterJointVerifier",
    "DefaultSixTargetFreeCase",
    "build_target_free_default_six_case",
    "default_six_contract",
    "default_six_identity_digest",
    "freeze_target_free_case_exclusive",
    "load_frozen_target_free_case",
    "parameter_counts",
    "target_free_case_arrays",
    "transplant_tri_v2_state",
]
