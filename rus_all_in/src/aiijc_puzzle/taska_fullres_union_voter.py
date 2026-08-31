"""Fixed full-resolution-denoised candidate-supply arm for TASKA.

This research module leaves the production TASKA dense cost matrices and its
12-scorer candidate set untouched.  A SHA-gated, stride-one 20x20 denoiser is
used only to create an auxiliary matcher view.  Four restored-view scorers
(v3/local x the first two audited orientations) may propose an edge absent
from the original harvest.  The proposal contract is intentionally fixed:

* support from at least three of the four restored scorers; and
* recovered ``train_exact_top5`` focal logit at least zero.

The resulting optional arm still emits only a permutation of the original
upright tiles.  Restored pixels are never rendered or returned.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    restore_matcher_view,
)
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_edge_calibrator import solve_prioritized_raw_tail_global
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    MATCHER_CONFIG,
    SOLVER_CONFIG,
    TAIL_MAX_SWAPS,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import ORIENTATIONS, calibrated_log_assignments

FULLRES_DENOISER_SHA256 = (
    "a6dfc3e264e97d93ad678f3ee97e070067357c2a6f6875e7b7432f880aa1492c"
)
FULLRES_ARCHITECTURE = "fullres-20x20-naf-boundary-denoiser-v1"
RESTORED_SCORER_COUNT = 4
RESTORED_SUPPORT_MINIMUM = 3
NEW_EDGE_FOCAL_LOGIT_MINIMUM = 0.0
FULLRES_ARM_NAME = "fullres_union_focal"


class TaskaFullresArtifactError(RuntimeError):
    """The fixed denoiser artifact differs from its declared contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fullres_denoiser(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> FullResolutionBoundaryDenoiser:
    """SHA-gate and strictly load the fixed matcher-only denoiser."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise TaskaFullresArtifactError(f"denoiser checkpoint is absent: {path}")
    digest = _sha256_file(path)
    if digest != FULLRES_DENOISER_SHA256:
        raise TaskaFullresArtifactError(
            f"denoiser SHA-256 mismatch: expected {FULLRES_DENOISER_SHA256}, got {digest}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - torch exception is version-specific.
        raise TaskaFullresArtifactError("weights-only denoiser load failed") from error
    required_keys = {"state_dict", "contract", "selection", "training_history"}
    if not isinstance(payload, Mapping) or set(payload) != required_keys:
        raise TaskaFullresArtifactError("denoiser top-level contract differs")
    contract = payload["contract"]
    if not isinstance(contract, Mapping):
        raise TaskaFullresArtifactError("denoiser contract is not a mapping")
    if contract.get("architecture") != FULLRES_ARCHITECTURE:
        raise TaskaFullresArtifactError("denoiser architecture differs")
    config = contract.get("model_config")
    if not isinstance(config, Mapping):
        raise TaskaFullresArtifactError("denoiser model_config is malformed")
    model = FullResolutionBoundaryDenoiser(FullResolutionDenoiserConfig(**dict(config)))
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(torch.device(device)).eval().requires_grad_(False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TaskaFullresArtifactError("denoiser tensor/device contract differs") from error
    model.checkpoint_path = path
    model.checkpoint_sha256 = digest
    return model


def _mutual_edges(matrix: Any, axis: str) -> frozenset[RawTailEdge]:
    """Return depth-one mutual argmax edges, matching the TASKA harvest."""

    if axis not in {"right", "down"}:
        raise ValueError("axis must be right or down")
    scores = np.asarray(matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("score matrix must be square")
    if not np.isfinite(scores).all():
        raise ValueError("score matrix must be finite")
    scores = scores.copy()
    np.fill_diagonal(scores, -np.inf)
    forward = scores.argmax(axis=1)
    backward = scores.argmax(axis=0)
    return frozenset(
        RawTailEdge(int(source), int(forward[source]), axis)
        for source in range(len(scores))
        if int(backward[int(forward[source])]) == source
    )


@torch.inference_mode()
def restored_mutual_scorer_sets(
    restored_tiles: Any,
    matchers: Sequence[torch.nn.Module],
    *,
    device: str | torch.device,
) -> tuple[frozenset[RawTailEdge], ...]:
    """Run the fixed v3/local x first-two-orientation restored scorer roster."""

    if len(matchers) != 2:
        raise ValueError("restored supply requires exactly v3 and local matchers")
    tiles = np.asarray(restored_tiles)
    if tiles.shape != (576, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("restored_tiles must be uint8 576x20x20x3")
    scorers: list[frozenset[RawTailEdge]] = []
    for model in matchers:
        for orientation in ORIENTATIONS[:2]:
            right, down = calibrated_log_assignments(
                model,
                tiles,
                device=device,
                orientation=orientation,
                rounds=MATCHER_CONFIG.rounds,
                cycle_weight=MATCHER_CONFIG.cycle_weight,
                sinkhorn_iterations=MATCHER_CONFIG.sinkhorn_iterations,
                acyclic_weight=MATCHER_CONFIG.acyclic_weight,
            )
            scorers.append(_mutual_edges(right, "right") | _mutual_edges(down, "down"))
    if len(scorers) != RESTORED_SCORER_COUNT:
        raise RuntimeError("restored scorer roster changed")
    return tuple(scorers)


def supported_absent_edges(
    current_edges: Sequence[RawTailEdge],
    scorer_sets: Sequence[Sequence[RawTailEdge]],
) -> tuple[tuple[RawTailEdge, ...], tuple[int, ...]]:
    """Freeze absent restored proposals with support at least three of four."""

    if len(scorer_sets) != RESTORED_SCORER_COUNT:
        raise ValueError("exactly four restored scorer sets are required")
    current = set(current_edges)
    if len(current) != len(tuple(current_edges)):
        raise ValueError("current_edges contains duplicates")
    counts = Counter(edge for scorer in scorer_sets for edge in set(scorer))
    axis_order = {"right": 0, "down": 1}
    proposals = sorted(
        (
            edge
            for edge, support in counts.items()
            if edge not in current and support >= RESTORED_SUPPORT_MINIMUM
        ),
        key=lambda edge: (axis_order[edge.axis], edge.source, edge.target),
    )
    return tuple(proposals), tuple(int(counts[edge]) for edge in proposals)


def accept_focal_proposals(
    proposed_edges: Sequence[RawTailEdge],
    proposed_logits: Any,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    """Apply the one fixed focal-logit >= 0 new-edge gate."""

    proposals = tuple(proposed_edges)
    logits = np.asarray(proposed_logits, dtype=np.float32)
    if logits.shape != (len(proposals),) or not np.isfinite(logits).all():
        raise ValueError("proposed_logits must be one finite value per proposal")
    mask = logits >= NEW_EDGE_FOCAL_LOGIT_MINIMUM
    return (
        tuple(edge for edge, keep in zip(proposals, mask, strict=True) if bool(keep)),
        np.ascontiguousarray(logits[mask]),
    )


def strict_layout(value: Any, *, grid: int = 24) -> np.ndarray:
    """Return one strict original-tile permutation."""

    count = grid * grid
    result = np.ascontiguousarray(value, dtype=np.int32)
    if result.shape != (count,) or not np.array_equal(np.sort(result), np.arange(count)):
        raise ValueError("layout is not a strict original-tile permutation")
    return result


@dataclass(frozen=True)
class FullresUnionComposition:
    """Target-free fifth-arm layout and selector/tail diagnostics."""

    layout: np.ndarray
    fullres_layout: np.ndarray
    choice: str
    total_costs: tuple[tuple[str, float], ...]
    union_edges: tuple[RawTailEdge, ...]
    accepted_new_edges: tuple[RawTailEdge, ...]
    diagnostics: Mapping[str, Any]
    grid_size: int = 24

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layout", strict_layout(self.layout, grid=self.grid_size)
        )
        object.__setattr__(
            self,
            "fullres_layout",
            strict_layout(self.fullres_layout, grid=self.grid_size),
        )


def compose_fullres_union_focal_arm(
    *,
    cost_right: Any,
    cost_down: Any,
    current_edges: Sequence[RawTailEdge],
    current_focal_logits: Any,
    accepted_new_edges: Sequence[RawTailEdge],
    accepted_new_logits: Any,
    four_layouts: Mapping[str, Any],
    grid: int = 24,
) -> FullresUnionComposition:
    """Compose one fifth arm while retaining original dense costs exactly."""

    current = tuple(current_edges)
    new = tuple(accepted_new_edges)
    if len(set(current)) != len(current) or len(set(new)) != len(new):
        raise ValueError("candidate edge lists must be duplicate-free")
    if set(current) & set(new):
        raise ValueError("new edges must be absent from the current harvest")
    old_logits = np.asarray(current_focal_logits, dtype=np.float32)
    new_logits = np.asarray(accepted_new_logits, dtype=np.float32)
    if old_logits.shape != (len(current),) or new_logits.shape != (len(new),):
        raise ValueError("focal priority arrays are not edge-aligned")
    if not np.isfinite(old_logits).all() or not np.isfinite(new_logits).all():
        raise ValueError("focal priorities must be finite")
    if tuple(four_layouts) != ARM_NAMES:
        raise ValueError("four_layouts must follow the production arm order")
    strict_four = {name: strict_layout(layout, grid=grid) for name, layout in four_layouts.items()}
    union = current + new
    priorities = np.concatenate((old_logits, new_logits)).astype(np.float32, copy=False)
    solver = solve_prioritized_raw_tail_global(
        cost_right,
        cost_down,
        union,
        priorities,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    fullres_layout = strict_layout(solver.layout, grid=grid)
    layouts = {**strict_four, FULLRES_ARM_NAME: fullres_layout}
    selection = select_lowest_taska_seam_cost_layout(
        layouts,
        cost_right,
        cost_down,
        grid=grid,
    )
    protected_edges = union if selection.choice == FULLRES_ARM_NAME else current
    tail = polish_unprotected_taska_tail(
        selection.layout,
        cost_right,
        cost_down,
        protected_edges,
        grid=grid,
        max_swaps=TAIL_MAX_SWAPS,
    )
    return FullresUnionComposition(
        layout=tail.layout,
        fullres_layout=fullres_layout,
        choice=selection.choice,
        total_costs=selection.total_costs,
        union_edges=union,
        accepted_new_edges=new,
        diagnostics={
            "fullres_solver": solver.diagnostics.as_dict(),
            "five_arm_tail": asdict(tail.diagnostics),
            "tail_protected_candidate_set": (
                "union" if selection.choice == FULLRES_ARM_NAME else "current"
            ),
            "restored_pixels_matcher_only": True,
            "raw_dense_cost_matrices_unchanged": True,
            "strict_original_upright_tile_permutation": True,
        },
        grid_size=grid,
    )


def restore_fixed_matcher_view(
    model: FullResolutionBoundaryDenoiser,
    dirty_tiles: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 576,
) -> np.ndarray:
    """Named wrapper documenting that restoration is matcher-only."""

    return restore_matcher_view(model, dirty_tiles, device=device, batch_size=batch_size)


__all__ = [
    "FULLRES_ARM_NAME",
    "FULLRES_ARCHITECTURE",
    "FULLRES_DENOISER_SHA256",
    "NEW_EDGE_FOCAL_LOGIT_MINIMUM",
    "RESTORED_SCORER_COUNT",
    "RESTORED_SUPPORT_MINIMUM",
    "FullresUnionComposition",
    "TaskaFullresArtifactError",
    "accept_focal_proposals",
    "compose_fullres_union_focal_arm",
    "load_fullres_denoiser",
    "restore_fixed_matcher_view",
    "restored_mutual_scorer_sets",
    "strict_layout",
    "supported_absent_edges",
]
