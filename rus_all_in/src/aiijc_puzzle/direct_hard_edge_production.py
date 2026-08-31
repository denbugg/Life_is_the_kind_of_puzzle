"""Non-default production adapter for the frozen direct hard-edge priority head.

The adapter is deliberately separate from :mod:`socket_sorter_production` and
does not change its default.  With no direct checkpoint it delegates to that
baseline exactly.  With an explicitly loaded direct checkpoint it runs the
frozen d64 Socket encoder once, builds the target-free board/list features,
reprioritises only the existing hard-projected edges, and exposes both:

* the unchanged decoder144 + cyclic-border5 baseline; and
* decoder144 + cyclic-border5 using the learned component-edge order.

Both variants are assembled from every original upright input tile exactly
once.  No reference image, manifest, filename lookup, restored-only candidate,
absolute-coordinate prior, or competition-test-specific input is accepted by
this API.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch

from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.direct_hard_edge_priority import (
    DirectHardEdgePriority,
    learned_priority_matrices,
    prepare_direct_hard_edge_board,
)
from aiijc_puzzle.protocol import (
    GRID_SIZE,
    IMAGE_SIZE,
    RGB_CHANNELS,
    TILE_COUNT,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_confidence_calibration import extract_hard_edge_features
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_matcher import SocketOutput
from aiijc_puzzle.socket_sorter_production import (
    DECODER_EDGE_BUDGET,
    DECODER_SWAP_STEPS,
    IDENTITY_PIXEL_TAIL,
    LoadedSocketCheckpoint,
    SocketSorterPrediction,
    assemble_audited_original_tiles,
    predict_socket_sorter,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

DIRECT_CHECKPOINT_SCHEMA = "aiijc-direct-hard-edge-board-priority-checkpoint-v1"
DIRECT_ADAPTER_SCHEMA = "aiijc-direct-hard-edge-production-adapter-v1"
FROZEN_DIRECT_HARD_EDGE_SHA256 = (
    "473f8ca09438fc4657919b7fad9777ad4928837aafd997301763198861c6f216"
)
FROZEN_DIRECT_CONFIG_SHA256 = (
    "11ba187b5a739a54193e6f869f443a9bcd04d1559c641c8d3b3ffd0151f514fb"
)
FROZEN_SOCKET_SHA256 = (
    "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670"
)
FROZEN_DIRECT_CONTRACT: dict[str, Any] = {
    "architecture": "direct-hard-edge-deepsets-residual-v1",
    "hidden_dimension": 64,
    "input_dimension": 296,
    "per_axis_pool": ["mean", "max"],
    "provisional_raw_edge_budget_per_axis": 48,
    "raw_priority_zero_init_residual": True,
    "whole_board_pool": ["mean", "max"],
}
FROZEN_DIRECT_PARAMETER_COUNT = 47_057
CYCLIC_BORDER_WEIGHT = 5.0


def _array_sha256(value: np.ndarray, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        integer = int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if value != value.lower() or integer < 0:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_regular_file(path: Path) -> Path:
    """Reject symlinks in every existing path component and require a file."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.parts[0])
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ValueError(f"checkpoint path does not exist: {absolute}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink checkpoint paths are forbidden: {current}")
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"expected a regular checkpoint file: {absolute}")
    return absolute


@dataclass(frozen=True)
class DirectHardEdgeLineage:
    """Content-addressed organizer-train fit and exposed D1 rosters."""

    fit_count: int
    fit_digest: str
    d1_count: int
    d1_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_count": self.fit_count,
            "fit_digest": self.fit_digest,
            "d1_count": self.d1_count,
            "d1_digest": self.d1_digest,
        }


@dataclass(frozen=True)
class LoadedDirectHardEdgeCheckpoint:
    """Frozen learned head plus its strict Socket/config lineage contract."""

    path: Path
    sha256: str
    model: DirectHardEdgePriority
    contract: dict[str, Any]
    config_sha256: str
    socket_checkpoint_sha256: str
    lineage: DirectHardEdgeLineage


def _lineage_from_selection(selection: Any) -> DirectHardEdgeLineage:
    if not isinstance(selection, Mapping):
        raise ValueError("direct checkpoint has no selection lineage mapping")

    def roster(prefix: str) -> tuple[tuple[str, ...], str]:
        raw = selection.get(f"{prefix}_source_filenames")
        count = selection.get(f"{prefix}_source_count")
        digest = selection.get(f"{prefix}_source_order_digest")
        if not isinstance(raw, list) or not all(
            isinstance(name, str) and Path(name).name == name and name.endswith(".png")
            for name in raw
        ):
            raise ValueError(f"direct checkpoint {prefix} roster is malformed")
        names = tuple(raw)
        if len(names) != len(set(names)):
            raise ValueError(f"direct checkpoint {prefix} roster count is invalid")
        if count is not None and count != len(names):
            raise ValueError(f"direct checkpoint {prefix} roster count is invalid")
        observed = _names_digest(names)
        if digest != observed:
            raise ValueError(f"direct checkpoint {prefix} roster digest is invalid")
        return names, observed

    fit, fit_digest = roster("fit")
    d1, d1_digest = roster("d1")
    if set(fit) & set(d1):
        raise ValueError("direct checkpoint fit and D1 rosters overlap")
    return DirectHardEdgeLineage(len(fit), fit_digest, len(d1), d1_digest)


def load_direct_hard_edge_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_sha256: str = FROZEN_DIRECT_HARD_EDGE_SHA256,
    expected_config_sha256: str = FROZEN_DIRECT_CONFIG_SHA256,
) -> LoadedDirectHardEdgeCheckpoint:
    """Load the frozen v1 head only after SHA, contract and lineage checks.

    Alternate checkpoints are intentionally not auto-discovered.  Tests or a
    later explicitly frozen model may pass its own expected file/config hashes;
    omitting them pins the independently confirmed v1 artifact.
    """

    expected_sha256 = _validate_sha256(expected_sha256, name="expected_sha256")
    expected_config_sha256 = _validate_sha256(
        expected_config_sha256,
        name="expected_config_sha256",
    )
    path = _require_regular_file(checkpoint_path)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError("direct hard-edge checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("direct hard-edge checkpoint payload must be a mapping")
    if payload.get("schema") != DIRECT_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported direct hard-edge checkpoint schema")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or dict(contract) != FROZEN_DIRECT_CONTRACT:
        raise ValueError("direct hard-edge checkpoint architecture contract changed")
    config_sha256 = payload.get("config_sha256")
    if config_sha256 != expected_config_sha256:
        raise ValueError("direct hard-edge checkpoint/config SHA lineage mismatch")
    if payload.get("competition_test_opened") is not False:
        raise ValueError("direct hard-edge checkpoint lacks closed-test provenance")
    socket_record = payload.get("socket_checkpoint")
    if not isinstance(socket_record, Mapping):
        raise ValueError("direct hard-edge checkpoint has no Socket lineage")
    socket_path = socket_record.get("path")
    socket_sha256 = _validate_sha256(
        socket_record.get("sha256"),
        name="socket_checkpoint.sha256",
    )
    if not isinstance(socket_path, str) or not socket_path:
        raise ValueError("direct hard-edge Socket checkpoint path is malformed")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("direct hard-edge checkpoint has no state_dict mapping")
    lineage = _lineage_from_selection(payload.get("selection"))
    model = DirectHardEdgePriority(
        int(contract["input_dimension"]),
        hidden_dimension=int(contract["hidden_dimension"]),
    ).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != FROZEN_DIRECT_PARAMETER_COUNT:
        raise ValueError("direct hard-edge parameter-count contract changed")
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise ValueError("direct hard-edge checkpoint contains non-finite parameters")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedDirectHardEdgeCheckpoint(
        path=path,
        sha256=observed_sha256,
        model=model,
        contract=dict(contract),
        config_sha256=config_sha256,
        socket_checkpoint_sha256=socket_sha256,
        lineage=lineage,
    )


@dataclass(frozen=True)
class DirectHardEdgePriorityInference:
    """Target-free matrices and Socket evidence reusable by downstream placers."""

    socket_output: SocketOutput
    component_edge_priority: dict[str, np.ndarray]
    raw_scores: np.ndarray
    learned_scores: np.ndarray
    source: np.ndarray
    target: np.ndarray
    axis: np.ndarray
    matcher_seconds: float
    priority_seconds: float

    def report(self) -> dict[str, Any]:
        return {
            "hard_edges": len(self.learned_scores),
            "hard_edges_per_axis": {
                "right": int(np.count_nonzero(self.axis == 0)),
                "down": int(np.count_nonzero(self.axis == 1)),
            },
            "raw_scores_sha256": _array_sha256(self.raw_scores, dtype="<f4"),
            "learned_scores_sha256": _array_sha256(self.learned_scores, dtype="<f4"),
            "right_priority_sha256": _array_sha256(
                self.component_edge_priority["right"], dtype="<f8"
            ),
            "down_priority_sha256": _array_sha256(
                self.component_edge_priority["down"], dtype="<f8"
            ),
            "matcher_seconds": self.matcher_seconds,
            "priority_seconds": self.priority_seconds,
        }


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("production model has no parameters") from error


def _validate_adapter_lineage(
    socket: LoadedSocketCheckpoint,
    direct: LoadedDirectHardEdgeCheckpoint,
    *,
    device: torch.device,
) -> None:
    if socket.sha256 != direct.socket_checkpoint_sha256:
        raise ValueError("direct head was not trained against this Socket checkpoint")
    if _module_device(socket.model) != device or _module_device(direct.model) != device:
        raise ValueError("Socket, direct head and requested inference device differ")
    if socket.model.training or direct.model.training:
        raise ValueError("production inference requires both models in eval mode")
    socket_dimension = socket.contract.get("dimension")
    expected_input = 20 + 4 * int(socket_dimension) + 2 + 18
    if expected_input != direct.model.input_dimension:
        raise ValueError("Socket token dimension and direct-head input contract differ")


@torch.inference_mode()
def infer_direct_hard_edge_priorities(
    dirty_tiles: np.ndarray,
    socket: LoadedSocketCheckpoint,
    direct: LoadedDirectHardEdgeCheckpoint,
    *,
    device: torch.device,
) -> DirectHardEdgePriorityInference:
    """Convert one board's original dirty tiles to learned hard-edge priorities."""

    _validate_adapter_lineage(socket, direct, device=device)
    tiles = np.asarray(dirty_tiles)
    expected = (TILE_COUNT, 20, 20, RGB_CHANNELS)
    if tiles.shape != expected or tiles.dtype != np.uint8:
        raise ValueError(f"dirty_tiles must be uint8 RGB with shape {expected}")
    tensor = (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )
    matcher_started = perf_counter()
    tokens, output = extract_frozen_socket_context(socket.model, tensor, grid=GRID_SIZE)
    matcher_seconds = perf_counter() - matcher_started
    priority_started = perf_counter()
    features = extract_hard_edge_features(
        right_log_assignment=output.right_log_assignment[0],
        down_log_assignment=output.down_log_assignment[0],
        right_raw=output.right_raw[0],
        down_raw=output.down_raw[0],
        grid=GRID_SIZE,
    )
    board = prepare_direct_hard_edge_board(
        tokens[0],
        features,
        output,
        grid=GRID_SIZE,
        provisional_edge_budget_per_axis=int(
            direct.contract["provisional_raw_edge_budget_per_axis"]
        ),
    )
    if board.values.shape[1] != direct.model.input_dimension:
        raise RuntimeError("runtime direct hard-edge feature contract changed")
    score_tensor = direct.model(board.values, board.raw_priority, board.axis)
    learned = np.ascontiguousarray(score_tensor.float().cpu().numpy(), dtype=np.float32)
    raw = np.ascontiguousarray(board.raw_priority.float().cpu().numpy(), dtype=np.float32)
    axis = np.ascontiguousarray(board.axis.cpu().numpy(), dtype=np.int8)
    priorities = learned_priority_matrices(board, learned, grid=GRID_SIZE)
    expected_per_axis = GRID_SIZE * (GRID_SIZE - 1)
    for axis_name in ("right", "down"):
        matrix = priorities[axis_name]
        if matrix.shape != (TILE_COUNT, TILE_COUNT) or not np.isfinite(matrix).all():
            raise RuntimeError("learned component-edge priority matrix is invalid")
        if np.count_nonzero(matrix) != expected_per_axis:
            raise RuntimeError("learned priorities escaped the frozen hard-edge supply")
    if (
        learned.shape != (2 * expected_per_axis,)
        or raw.shape != learned.shape
        or set(np.unique(axis)) != {0, 1}
    ):
        raise RuntimeError("direct hard-edge score identity contract changed")
    return DirectHardEdgePriorityInference(
        socket_output=output,
        component_edge_priority=priorities,
        raw_scores=raw,
        learned_scores=learned,
        source=np.ascontiguousarray(board.source, dtype=np.int32),
        target=np.ascontiguousarray(board.target, dtype=np.int32),
        axis=axis,
        matcher_seconds=matcher_seconds,
        priority_seconds=perf_counter() - priority_started,
    )


@dataclass(frozen=True)
class DirectHardEdgeProductionPrediction:
    """Baseline and optional learned variants from the same dirty board."""

    selected_variant: Literal["baseline", "direct-hard-edge"]
    fallback_reason: str | None
    selected: SocketSorterPrediction
    baseline: SocketSorterPrediction
    learned: SocketSorterPrediction | None
    direct_checkpoint_sha256: str | None
    priority_report: dict[str, Any] | None

    def report(self) -> dict[str, Any]:
        arms: dict[str, Any] = {
            "baseline": {
                "layout_sha256": _array_sha256(self.baseline.layout, dtype="<i4"),
                "decoder": self.baseline.decoder_report,
                "cyclic": self.baseline.cyclic_report,
                "permutation_audit": self.baseline.audit.as_dict(),
            }
        }
        if self.learned is not None:
            arms["direct-hard-edge"] = {
                "layout_sha256": _array_sha256(self.learned.layout, dtype="<i4"),
                "decoder": self.learned.decoder_report,
                "cyclic": self.learned.cyclic_report,
                "permutation_audit": self.learned.audit.as_dict(),
            }
        return {
            "schema": DIRECT_ADAPTER_SCHEMA,
            "selected_variant": self.selected_variant,
            "fallback_reason": self.fallback_reason,
            "direct_checkpoint_sha256": self.direct_checkpoint_sha256,
            "priority": self.priority_report,
            "arms": arms,
            "policy": {
                "default_without_direct_checkpoint": "baseline",
                "targets_or_manifest_labels_used": False,
                "restored_only_candidates_used": False,
                "cyclic_border_weight": CYCLIC_BORDER_WEIGHT,
                "all_original_upright_tiles_used_exactly_once": True,
            },
        }


def _decode_prediction(
    image: np.ndarray,
    output: SocketOutput,
    *,
    component_edge_priority: Mapping[str, Any] | None,
    matcher_seconds: float,
    started: float,
) -> SocketSorterPrediction:
    decoder = decode_socket_assignments(
        output.right_log_assignment,
        output.down_log_assignment,
        grid=GRID_SIZE,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
        component_edge_priority=component_edge_priority,
    )
    cyclic = select_global_cyclic_translation(
        decoder.layout,
        output.right_log_assignment,
        output.down_log_assignment,
        grid=GRID_SIZE,
        config=CyclicTranslationConfig(border_weight=CYCLIC_BORDER_WEIGHT),
    )
    raw, audit = assemble_audited_original_tiles(image, cyclic.layout)
    final = IDENTITY_PIXEL_TAIL.apply(raw)
    if not np.array_equal(final, raw):
        raise RuntimeError("identity production adapter altered original-tile assembly")
    return SocketSorterPrediction(
        layout=np.ascontiguousarray(cyclic.layout, dtype=np.int32),
        raw=raw,
        output=final,
        audit=audit,
        decoder_report=decoder.report(),
        cyclic_report=cyclic.report(),
        matcher_seconds=matcher_seconds,
        total_seconds=perf_counter() - started,
    )


@torch.inference_mode()
def predict_direct_hard_edge_variants(
    input_image: np.ndarray,
    socket: LoadedSocketCheckpoint,
    *,
    device: torch.device,
    direct: LoadedDirectHardEdgeCheckpoint | None = None,
) -> DirectHardEdgeProductionPrediction:
    """Expose the learned arm explicitly while keeping baseline as fallback.

    Passing ``direct=None`` is the only fallback: it calls the existing
    production baseline bit-for-bit.  A supplied but corrupt, incompatible, or
    failing learned checkpoint is never hidden behind an automatic fallback;
    it fails closed instead.
    """

    image = np.asarray(input_image)
    expected = (IMAGE_SIZE, IMAGE_SIZE, RGB_CHANNELS)
    if image.shape != expected or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB input {expected}, got {image.dtype} {image.shape}")
    if direct is None:
        baseline = predict_socket_sorter(
            image,
            socket,
            device=device,
            cyclic_border5=True,
            pixel_tail=IDENTITY_PIXEL_TAIL,
        )
        return DirectHardEdgeProductionPrediction(
            selected_variant="baseline",
            fallback_reason="direct-checkpoint-not-configured",
            selected=baseline,
            baseline=baseline,
            learned=None,
            direct_checkpoint_sha256=None,
            priority_report=None,
        )

    started = perf_counter()
    inference = infer_direct_hard_edge_priorities(
        split_tiles(image),
        socket,
        direct,
        device=device,
    )
    baseline = _decode_prediction(
        image,
        inference.socket_output,
        component_edge_priority=None,
        matcher_seconds=inference.matcher_seconds,
        started=started,
    )
    learned = _decode_prediction(
        image,
        inference.socket_output,
        component_edge_priority=inference.component_edge_priority,
        matcher_seconds=inference.matcher_seconds,
        started=started,
    )
    return DirectHardEdgeProductionPrediction(
        selected_variant="direct-hard-edge",
        fallback_reason=None,
        selected=learned,
        baseline=baseline,
        learned=learned,
        direct_checkpoint_sha256=direct.sha256,
        priority_report=inference.report(),
    )


__all__ = [
    "CYCLIC_BORDER_WEIGHT",
    "DIRECT_ADAPTER_SCHEMA",
    "DIRECT_CHECKPOINT_SCHEMA",
    "DirectHardEdgeLineage",
    "DirectHardEdgePriorityInference",
    "DirectHardEdgeProductionPrediction",
    "FROZEN_DIRECT_CONFIG_SHA256",
    "FROZEN_DIRECT_CONTRACT",
    "FROZEN_DIRECT_HARD_EDGE_SHA256",
    "FROZEN_SOCKET_SHA256",
    "LoadedDirectHardEdgeCheckpoint",
    "infer_direct_hard_edge_priorities",
    "load_direct_hard_edge_checkpoint",
    "predict_direct_hard_edge_variants",
]
