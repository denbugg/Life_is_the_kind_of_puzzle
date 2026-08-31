"""Execution primitives for the signed BasinCycle Stage-B pilot.

This module deliberately separates three phases:

* fitting on the frozen FIT64 roster;
* target-free freezing on the reserved EVAL32 roster; and
* reference attachment/scoring only after a valid freeze receipt exists.

The scientific preregistration is immutable.  A separate execution binding
hashes this implementation and fills in mechanical details that the scientific
document intentionally left to the runner (pixel kernels, procedural state
construction, and the six-channel clean-boundary target discretisation).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from aiijc_puzzle.basincycle_stage_b import (
    BasinCycleStageB,
    OracleDiagnostic,
    ProposalBank,
    StageBLabels,
    StageBOutput,
    aggregate_oracle_diagnostics,
    frozen_positive_action_mask,
    metric_deltas_for_bank,
    model_static_ledger,
    proposal_oracle_diagnostic,
    radius_two_count,
    select_hard_action,
    stage_b_loss,
)
from aiijc_puzzle.basincycle_stage_b_protocol import (
    BATCH_SIZE,
    EVAL_DRAWS,
    EVAL_SOURCE_COUNT,
    FIT_SOURCE_COUNT,
    FIT_UPDATES,
    GRID_SIZE,
    StageBPlanRow,
    eval_plan,
    fit_plan,
    require_target_free_freeze_receipt,
    validate_frozen_inputs,
    validate_frozen_roster_and_plans,
)
from aiijc_puzzle.basincycle_synthetic import (
    apply_cycle,
    canonical_cycle,
    exact_count,
    is_strict_permutation,
    true_pair_count,
)
from aiijc_puzzle.protocol import IMAGE_SIZE, TILE_SIZE, split_tiles
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_matcher import BORDER_HEAD_EMBEDDING_V2, SocketMatcher

SCIENTIFIC_CONFIG_SCHEMA = "aiijc-basincycle-stage-b-6x6-preregistered-v1"
EXECUTION_BINDING_SCHEMA = "aiijc-basincycle-stage-b-execution-binding-v1"
EXECUTION_BINDING_STATUS = "signed-unexecuted-review-required"
FIT_CHECKPOINT_SCHEMA = "aiijc-basincycle-stage-b-final-checkpoint-v1"
FREEZE_BUNDLE_SCHEMA = "aiijc-basincycle-stage-b-target-free-bundle-v1"
FREEZE_RECEIPT_SCHEMA = "aiijc-basincycle-stage-b-target-free-freeze-v1"
SCORE_REPORT_SCHEMA = "aiijc-basincycle-stage-b-score-report-v1"
PAIR_COUNT = 2 * GRID_SIZE * (GRID_SIZE - 1)
PROPOSAL_CAP = 256
MAX_CYCLE_LENGTH = 3
TARGET_FREE_REFERENCE_SEMANTICS = (
    "reference_opened=false means that no evaluation metric/oracle was attached "
    "before freeze. The deterministic synthetic shuffle inverse was nevertheless "
    "constructed inside case generation, and each procedural control was initialized "
    "from that planted truth. The model received that derived control by design, but "
    "neither the planted truth nor clean pixels were supplied directly to the predictor, "
    "proposal builder, or selector or persisted in the bundle."
)


@dataclass(frozen=True)
class VisibleCase:
    """One model-visible state plus fit-only references kept out of proposal code."""

    filename: str
    state_family: str
    tiles: torch.Tensor
    control: np.ndarray
    truth: np.ndarray
    clean_tiles_by_identity: np.ndarray


@dataclass(frozen=True)
class PreparedCase:
    """CPU-prepared pixels/state before an optional batched Socket decode."""

    filename: str
    state_family: str
    tiles: torch.Tensor
    control: np.ndarray | None
    truth: np.ndarray
    clean_tiles_by_identity: np.ndarray


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value in the binding's canonical representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    """Hash named arrays with shape/dtype framing, independent of NPZ metadata."""

    digest = hashlib.sha256()
    for name, array in named_arrays:
        value = np.ascontiguousarray(array)
        header = {
            "name": name,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
        }
        encoded = canonical_json_bytes(header)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write canonical, human-readable JSON and return its digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically save a compressed NumPy bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256_file(path)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_sidecar(path: Path) -> str:
    """Validate a conventional ``<name>.sha256`` sidecar."""

    sidecar = Path(f"{path}.sha256")
    fields = sidecar.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"malformed SHA-256 sidecar: {sidecar}")
    actual = sha256_file(path)
    if actual != fields[0]:
        raise ValueError(f"SHA-256 sidecar mismatch: {path}")
    return actual


def validate_execution_binding(
    binding: Mapping[str, Any],
    *,
    project_root: Path,
    binding_path: Path,
) -> dict[str, str]:
    """Fail closed on any scientific or execution implementation drift."""

    if binding.get("schema") != EXECUTION_BINDING_SCHEMA:
        raise ValueError("wrong BasinCycle execution-binding schema")
    if binding.get("status") != EXECUTION_BINDING_STATUS:
        raise ValueError("execution binding is not the signed unexecuted review boundary")
    binding_sha256 = validate_sidecar(binding_path)
    scientific = binding.get("scientific_config")
    implementation = binding.get("implementation")
    if not isinstance(scientific, Mapping) or not isinstance(implementation, Mapping):
        raise ValueError("execution binding lacks scientific/implementation inventories")
    observed: dict[str, str] = {"execution_binding": binding_sha256}
    for name, artifact in (("scientific_config", scientific), *implementation.items()):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"binding artifact is not a mapping: {name}")
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"binding artifact lacks path/hash: {name}")
        path = (project_root / relative).resolve()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"binding artifact hash mismatch: {name}")
        observed[str(name)] = actual
    config_path = (project_root / str(scientific["path"])).resolve()
    if validate_sidecar(config_path) != observed["scientific_config"]:
        raise ValueError("scientific config sidecar differs from execution binding")
    config = load_json(config_path)
    if config.get("schema") != SCIENTIFIC_CONFIG_SCHEMA:
        raise ValueError("execution binding points to the wrong scientific config")
    if binding.get("real_execution_authorized") is not False:
        raise ValueError("prepared binding must not self-authorize real execution")
    return observed


def validate_scientific_protocol(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Validate every frozen byte and reconstruct the source/plan metadata."""

    frozen = validate_frozen_inputs(config, project_root=project_root)
    manifest = load_json(project_root / config["frozen_inputs"]["manifest"]["path"])
    socket_report = load_json(
        project_root / config["frozen_inputs"]["socket_parent_report"]["path"]
    )
    active_scale = load_json(
        project_root / config["frozen_inputs"]["active_scale_config"]["path"]
    )
    protected = load_json(
        project_root / config["frozen_inputs"]["protected_roster_audit"]["path"]
    )
    roster = validate_frozen_roster_and_plans(
        config,
        manifest,
        socket_report,
        active_scale,
        protected,
    )
    return {"frozen_inputs": frozen, "roster_and_plans": roster}


def _load_rgb480(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB {IMAGE_SIZE}x{IMAGE_SIZE}: {path}")
        return np.asarray(image, dtype=np.uint8)


def load_clean_crop(path: Path, row: StageBPlanRow) -> np.ndarray:
    """Load the exact aligned 6x6 clean crop as canonical HWC tiles."""

    tiles = split_tiles(_load_rgb480(path)).reshape(24, 24, TILE_SIZE, TILE_SIZE, 3)
    crop = tiles[
        row.crop_tile_row : row.crop_tile_row + GRID_SIZE,
        row.crop_tile_col : row.crop_tile_col + GRID_SIZE,
    ]
    return np.ascontiguousarray(crop.reshape(GRID_SIZE * GRID_SIZE, TILE_SIZE, TILE_SIZE, 3))


def preload_clean_tile_canvases(
    filenames: Sequence[str],
    *,
    targets_root: Path,
    workers: int,
) -> dict[str, np.ndarray]:
    """Read each roster source once with deterministic ordered CPU prefetch."""

    names = tuple(filenames)
    if len(names) != len(set(names)) or workers < 1:
        raise ValueError("preload roster must be unique and workers must be positive")

    def load(name: str) -> np.ndarray:
        image = _load_rgb480(targets_root / name)
        return np.ascontiguousarray(
            split_tiles(image).reshape(24, 24, TILE_SIZE, TILE_SIZE, 3)
        )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="basincycle-preload") as pool:
        canvases = tuple(pool.map(load, names))
    return dict(zip(names, canvases, strict=True))


def crop_from_tile_canvas(canvas: np.ndarray, row: StageBPlanRow) -> np.ndarray:
    """Crop a cached 24x24 clean tile canvas without re-opening the source."""

    value = np.asarray(canvas)
    if value.shape != (24, 24, TILE_SIZE, TILE_SIZE, 3):
        raise ValueError("cached clean tile canvas has the wrong shape")
    crop = value[
        row.crop_tile_row : row.crop_tile_row + GRID_SIZE,
        row.crop_tile_col : row.crop_tile_col + GRID_SIZE,
    ]
    return np.ascontiguousarray(crop.reshape(GRID_SIZE * GRID_SIZE, TILE_SIZE, TILE_SIZE, 3))


def _gaussian_blur(tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        sigma = float(rng.uniform(0.2, 1.6))
        output[index] = ndimage.gaussian_filter(tile, sigma=(sigma, sigma, 0.0), mode="reflect")
    return output


def _motion_blur(tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        length = int(rng.choice((3, 5, 7)))
        kernel = np.zeros((length, length), dtype=np.float32)
        if bool(rng.integers(0, 2)):
            kernel[length // 2, :] = 1.0 / length
        else:
            kernel[:, length // 2] = 1.0 / length
        for channel in range(3):
            output[index, :, :, channel] = ndimage.convolve(
                tile[:, :, channel], kernel, mode="reflect"
            )
    return output


def _jpeg_ringing(tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        buffer = io.BytesIO()
        quality = int(rng.integers(20, 96))
        Image.fromarray(np.rint(tile * 255.0).clip(0, 255).astype(np.uint8)).save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
            optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            decoded.load()
            output[index] = np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0
    return output


def _scale_bias_chroma(tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        scale = rng.uniform(0.65, 1.35, size=(1, 1, 3)).astype(np.float32)
        bias = rng.uniform(-0.2, 0.2, size=(1, 1, 3)).astype(np.float32)
        adjusted = tile * scale + bias
        luma = (
            0.2126 * adjusted[..., 0:1]
            + 0.7152 * adjusted[..., 1:2]
            + 0.0722 * adjusted[..., 2:3]
        )
        chroma_scale = float(rng.uniform(0.7, 1.3))
        output[index] = luma + chroma_scale * (adjusted - luma)
    return output


def _edge_erosion(tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = tiles.copy()
    for index, tile in enumerate(tiles):
        width = int(rng.integers(0, 3))
        if width == 0:
            continue
        interior = tile[width:-width, width:-width]
        fill = interior.mean(axis=(0, 1), keepdims=True)
        alpha = float(rng.uniform(0.35, 0.85))
        mask = np.zeros((TILE_SIZE, TILE_SIZE, 1), dtype=np.float32)
        mask[:width] = 1.0
        mask[-width:] = 1.0
        mask[:, :width] = 1.0
        mask[:, -width:] = 1.0
        output[index] = tile * (1.0 - alpha * mask) + fill * (alpha * mask)
    return output


def corrupt_clean_tiles(
    clean_tiles: np.ndarray,
    *,
    recipe: str,
    seed: int,
) -> np.ndarray:
    """Apply one deterministic binding-defined matcher-only corruption."""

    clean = np.asarray(clean_tiles, dtype=np.float32)
    if clean.shape != (GRID_SIZE * GRID_SIZE, TILE_SIZE, TILE_SIZE, 3):
        raise ValueError("clean crop must contain 36 HWC 20x20 RGB tiles")
    if clean.max(initial=0.0) > 1.0:
        clean = clean / 255.0
    rng = np.random.default_rng(seed)
    if recipe == "gaussian_poisson":
        peak = rng.uniform(8.0, 64.0, size=(len(clean), 1, 1, 1))
        poisson = rng.poisson(np.clip(clean, 0.0, 1.0) * peak) / peak
        sigma = rng.uniform(0.0, 0.18, size=(len(clean), 1, 1, 1))
        output = poisson + sigma * rng.standard_normal(clean.shape)
    elif recipe == "gaussian_blur":
        output = _gaussian_blur(clean, rng)
    elif recipe == "motion_blur":
        output = _motion_blur(clean, rng)
    elif recipe == "jpeg_ringing":
        output = _jpeg_ringing(clean, rng)
    elif recipe == "scale_bias_chroma":
        output = _scale_bias_chroma(clean, rng)
    elif recipe == "edge_erosion":
        output = _edge_erosion(clean, rng)
    elif recipe == "mixed_two_stage":
        roster = (
            "gaussian_poisson",
            "gaussian_blur",
            "motion_blur",
            "jpeg_ringing",
            "scale_bias_chroma",
            "edge_erosion",
        )
        selected = rng.choice(roster, size=2, replace=False)
        first_seed = int(rng.integers(0, 2**31))
        second_seed = int(rng.integers(0, 2**31))
        output = corrupt_clean_tiles(clean, recipe=str(selected[0]), seed=first_seed)
        output = corrupt_clean_tiles(output, recipe=str(selected[1]), seed=second_seed)
    else:
        raise ValueError(f"unknown pixel recipe: {recipe}")
    return np.ascontiguousarray(np.clip(output, 0.0, 1.0), dtype=np.float32)


def _cycle_positions(layout: np.ndarray, positions: Sequence[int]) -> None:
    cycle = tuple(int(value) for value in positions)
    if len(cycle) < 2 or len(set(cycle)) != len(cycle):
        raise ValueError("procedural edit is not a closed cycle")
    layout[:] = apply_cycle(layout, cycle)


def procedural_control(
    truth: np.ndarray,
    *,
    recipe: str,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a deterministic strict incumbent under the bound corruption family."""

    if severity not in (1, 2, 4, 8):
        raise ValueError("procedural severity must be one of 1/2/4/8")
    layout = np.asarray(truth, dtype=np.int64).copy()
    count = len(layout)
    board = layout.reshape(GRID_SIZE, GRID_SIZE)
    for _ in range(severity):
        if recipe == "short_tile_cycle":
            length = int(rng.integers(2, 4))
            _cycle_positions(layout, rng.choice(count, size=length, replace=False))
        elif recipe == "congruent_patch_cycle":
            height = int(rng.integers(1, 3))
            width = int(rng.integers(1, 3))
            first_row = int(rng.integers(0, GRID_SIZE - height + 1))
            first_col = int(rng.integers(0, GRID_SIZE - width + 1))
            candidates = [
                (row, col)
                for row in range(GRID_SIZE - height + 1)
                for col in range(GRID_SIZE - width + 1)
                if row + height <= first_row
                or first_row + height <= row
                or col + width <= first_col
                or first_col + width <= col
            ]
            second_row, second_col = candidates[int(rng.integers(len(candidates)))]
            first = board[first_row : first_row + height, first_col : first_col + width].copy()
            second = board[second_row : second_row + height, second_col : second_col + width].copy()
            board[first_row : first_row + height, first_col : first_col + width] = second
            board[second_row : second_row + height, second_col : second_col + width] = first
        elif recipe == "wrong_edge_weld_cycle":
            anchor = int(rng.integers(count))
            anchor_row, anchor_col = divmod(anchor, GRID_SIZE)
            neighbours = []
            for delta_row, delta_col in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                row = anchor_row + delta_row
                col = anchor_col + delta_col
                if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                    neighbours.append(row * GRID_SIZE + col)
            neighbour = neighbours[int(rng.integers(len(neighbours)))]
            third_choices = np.setdiff1d(np.arange(count), np.array([anchor, neighbour]))
            third = int(rng.choice(third_choices))
            _cycle_positions(layout, (anchor, third, neighbour))
        elif recipe == "band_cyclic_roll":
            if bool(rng.integers(0, 2)):
                row = int(rng.integers(GRID_SIZE))
                board[row] = np.roll(board[row], int(rng.integers(1, GRID_SIZE)))
            else:
                column = int(rng.integers(GRID_SIZE))
                board[:, column] = np.roll(
                    board[:, column], int(rng.integers(1, GRID_SIZE))
                )
        elif recipe == "whole_board_roll":
            row_shift = int(rng.integers(0, GRID_SIZE))
            col_shift = int(rng.integers(0, GRID_SIZE))
            if row_shift == 0 and col_shift == 0:
                col_shift = 1
            board[:] = np.roll(board, shift=(row_shift, col_shift), axis=(0, 1))
        else:
            raise ValueError(f"unknown procedural state recipe: {recipe}")
    if not is_strict_permutation(layout):
        raise AssertionError("procedural corruption broke the strict permutation")
    return np.ascontiguousarray(layout)


def truth_and_tile_order(state_seed: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Return shuffled input identities and their exact raster truth mapping."""

    rng = np.random.default_rng(state_seed)
    tile_order = rng.permutation(GRID_SIZE * GRID_SIZE)
    truth = np.argsort(tile_order).astype(np.int64, copy=False)
    return truth, tile_order, rng


def load_frozen_socket(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> SocketMatcher:
    """Load only the exact frozen v2 SocketMatcher contract."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Socket checkpoint lacks a contract")
    expected = {
        "architecture": "board-conditioned-partial-socket-matcher-v2",
        "dimension": 64,
        "heads": 4,
        "board_layers": 1,
        "socket_layers": 1,
        "sinkhorn_iterations": 12,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Socket checkpoint differs at {key}")
    model = SocketMatcher(
        dimension=64,
        heads=4,
        board_layers=1,
        socket_layers=1,
        sinkhorn_iterations=12,
        border_head_version=BORDER_HEAD_EMBEDDING_V2,
    )
    model.load_state_dict(payload["state_dict"])
    model.requires_grad_(False).eval().to(device)
    return model


@torch.no_grad()
def socket_controls(
    socket: SocketMatcher,
    tiles: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    """Decode dirty-visible incumbents with one batched Socket forward."""

    if not tiles:
        return ()
    output = socket(torch.stack(tuple(tiles)).to(device), grid=GRID_SIZE)
    config = SocketDecoderConfig(
        border_weight=0.20,
        component_shift_unary_weight=0.0,
        component_edge_budget_per_axis=30,
        max_swap_steps=24,
        swap_edge_budget_per_axis=30,
    )
    controls: list[np.ndarray] = []
    for index in range(len(tiles)):
        result = decode_socket_assignments(
            output.right_log_assignment[index],
            output.down_log_assignment[index],
            grid=GRID_SIZE,
            config=config,
        )
        if not result.diagnostics.strict_permutation or not is_strict_permutation(
            result.layout
        ):
            raise AssertionError("frozen Socket control is not a strict permutation")
        controls.append(np.asarray(result.layout, dtype=np.int64))
    return tuple(controls)


def socket_control(
    socket: SocketMatcher,
    tiles: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Compatibility wrapper around the batched frozen Socket path."""

    return socket_controls(socket, (tiles,), device=device)[0]


def prepare_case_pixels(
    row: StageBPlanRow,
    *,
    clean_tile_canvas: np.ndarray,
) -> PreparedCase:
    """Prepare one row entirely on CPU with row-local deterministic RNGs."""

    clean = crop_from_tile_canvas(clean_tile_canvas, row).astype(np.float32) / 255.0
    truth, tile_order, state_rng = truth_and_tile_order(row.state_seed)
    clean_by_identity = np.ascontiguousarray(clean[tile_order])
    dirty = corrupt_clean_tiles(clean_by_identity, recipe=row.pixel_recipe, seed=row.pixel_seed)
    tiles = torch.from_numpy(dirty).permute(0, 3, 1, 2).contiguous()
    if row.state_family == "solver_replay":
        control = None
    elif row.state_family == "procedural":
        control = procedural_control(
            truth,
            recipe=row.state_recipe,
            severity=row.severity,
            rng=state_rng,
        )
    else:
        raise ValueError(f"unknown Stage-B state family: {row.state_family}")
    return PreparedCase(
        filename=row.source_filename,
        state_family=row.state_family,
        tiles=tiles,
        control=control,
        truth=truth,
        clean_tiles_by_identity=clean_by_identity,
    )


def materialize_cases(
    rows: Sequence[StageBPlanRow],
    *,
    clean_tile_canvases: Mapping[str, np.ndarray],
    socket: SocketMatcher,
    socket_device: torch.device,
    workers: int,
) -> tuple[VisibleCase, ...]:
    """Parallelise CPU corruption, then batch the optional Socket forward."""

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="basincycle-corrupt") as pool:
        futures = submit_case_preparation(pool, rows, clean_tile_canvases=clean_tile_canvases)
        prepared = resolve_case_preparation(futures)
    return materialize_prepared_cases(prepared, socket=socket, socket_device=socket_device)


def submit_case_preparation(
    pool: ThreadPoolExecutor,
    rows: Sequence[StageBPlanRow],
    *,
    clean_tile_canvases: Mapping[str, np.ndarray],
) -> tuple[Future[PreparedCase], ...]:
    """Submit one ordered CPU batch without consuming any shared RNG state."""

    return tuple(
        pool.submit(
            prepare_case_pixels,
            row,
            clean_tile_canvas=clean_tile_canvases[row.source_filename],
        )
        for row in rows
    )


def resolve_case_preparation(
    futures: Sequence[Future[PreparedCase]],
) -> tuple[PreparedCase, ...]:
    """Resolve futures in frozen row order regardless of completion order."""

    return tuple(future.result() for future in futures)


def materialize_prepared_cases(
    prepared: Sequence[PreparedCase],
    *,
    socket: SocketMatcher,
    socket_device: torch.device,
) -> tuple[VisibleCase, ...]:
    """Batch Socket rows and combine them with already prepared procedural rows."""

    prepared = tuple(prepared)
    solver_indices = [
        index for index, case in enumerate(prepared) if case.state_family == "solver_replay"
    ]
    decoded = socket_controls(
        socket,
        tuple(prepared[index].tiles for index in solver_indices),
        device=socket_device,
    )
    controls_by_index = dict(zip(solver_indices, decoded, strict=True))
    visible: list[VisibleCase] = []
    for index, case in enumerate(prepared):
        control = case.control if case.control is not None else controls_by_index[index]
        visible.append(
            VisibleCase(
                filename=case.filename,
                state_family=case.state_family,
                tiles=case.tiles,
                control=control,
                truth=case.truth,
                clean_tiles_by_identity=case.clean_tiles_by_identity,
            )
        )
    return tuple(visible)


def iter_prefetched_case_batches(
    rows: Sequence[StageBPlanRow],
    *,
    batch_size: int,
    clean_tile_canvases: Mapping[str, np.ndarray],
    socket: SocketMatcher,
    socket_device: torch.device,
    workers: int,
    thread_name_prefix: str,
) -> Iterator[tuple[int, tuple[VisibleCase, ...]]]:
    """Overlap preparation of batch t+1 with Socket/materialisation of batch t."""

    ordered_rows = tuple(rows)
    if not ordered_rows:
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix) as pool:
        pending = submit_case_preparation(
            pool,
            ordered_rows[:batch_size],
            clean_tile_canvases=clean_tile_canvases,
        )
        for batch_start in range(0, len(ordered_rows), batch_size):
            prepared = resolve_case_preparation(pending)
            next_start = batch_start + batch_size
            if next_start < len(ordered_rows):
                pending = submit_case_preparation(
                    pool,
                    ordered_rows[next_start : next_start + batch_size],
                    clean_tile_canvases=clean_tile_canvases,
                )
            yield batch_start, materialize_prepared_cases(
                prepared,
                socket=socket,
                socket_device=socket_device,
            )


def materialize_case(
    row: StageBPlanRow,
    *,
    targets_root: Path,
    socket: SocketMatcher,
    socket_device: torch.device,
) -> VisibleCase:
    """Materialise one fixed plan row; truth never enters proposal construction."""

    canvas = split_tiles(_load_rgb480(targets_root / row.source_filename)).reshape(
        24, 24, TILE_SIZE, TILE_SIZE, 3
    )
    return materialize_cases(
        (row,),
        clean_tile_canvases={row.source_filename: canvas},
        socket=socket,
        socket_device=socket_device,
        workers=1,
    )[0]


def clean_boundary_targets(clean_tiles: np.ndarray) -> np.ndarray:
    """Return bound RGB + forward tangent-RGB-difference side targets."""

    tiles = np.asarray(clean_tiles, dtype=np.float32)
    if tiles.shape != (GRID_SIZE * GRID_SIZE, TILE_SIZE, TILE_SIZE, 3):
        raise ValueError("clean tiles have the wrong shape")
    right = tiles[:, :, -4:, :].mean(axis=2)
    left = tiles[:, :, :4, :].mean(axis=2)
    bottom = tiles[:, -4:, :, :].mean(axis=1)
    top = tiles[:, :4, :, :].mean(axis=1)
    sides = np.stack((right, left, bottom, top), axis=1)
    differences = np.diff(sides, axis=2, prepend=sides[:, :, :1])
    return np.ascontiguousarray(np.concatenate((sides, differences), axis=-1))


def directional_edge_targets(truth: np.ndarray) -> np.ndarray:
    """Return outgoing right/down neighbour IDs with ``-1`` at the frame."""

    layout = np.asarray(truth, dtype=np.int64)
    if not is_strict_permutation(layout) or layout.size != GRID_SIZE * GRID_SIZE:
        raise ValueError("truth must be one strict 6x6 permutation")
    board = layout.reshape(GRID_SIZE, GRID_SIZE)
    targets = np.full((2, len(layout)), -1, dtype=np.int64)
    targets[0, board[:, :-1]] = board[:, 1:]
    targets[1, board[:-1, :]] = board[1:, :]
    return targets


def _true_directed_pairs(layout: np.ndarray, truth: np.ndarray) -> set[tuple[int, int, int]]:
    value = np.asarray(layout).reshape(GRID_SIZE, GRID_SIZE)
    reference = np.asarray(truth).reshape(GRID_SIZE, GRID_SIZE)
    truth_pairs = {
        (0, int(a), int(b))
        for a, b in zip(reference[:, :-1].flat, reference[:, 1:].flat, strict=True)
    } | {
        (1, int(a), int(b))
        for a, b in zip(reference[:-1, :].flat, reference[1:, :].flat, strict=True)
    }
    realised = {
        (0, int(a), int(b))
        for a, b in zip(value[:, :-1].flat, value[:, 1:].flat, strict=True)
    } | {
        (1, int(a), int(b))
        for a, b in zip(value[:-1, :].flat, value[1:, :].flat, strict=True)
    }
    return truth_pairs & realised


def pair_loss_labels(
    control: np.ndarray,
    truth: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Mark actions that destroy any true directed pair present in the control."""

    baseline = _true_directed_pairs(control, truth)
    result = np.zeros(len(valid), dtype=bool)
    for index in np.flatnonzero(valid):
        candidate_pairs = _true_directed_pairs(candidates[index], truth)
        result[index] = not baseline.issubset(candidate_pairs)
    return result


def labels_for_output(output: StageBOutput, cases: Sequence[VisibleCase]) -> StageBLabels:
    """Attach fit references after the model has frozen proposal identities."""

    batch_size, proposal_cap = output.proposal_bank.valid.shape
    if len(cases) != batch_size:
        raise ValueError("case batch differs from Stage-B output")
    positions = output.proposal_bank.positions.detach().cpu().numpy().copy()
    lengths = output.proposal_bank.lengths.detach().cpu().numpy().copy()
    valid = output.proposal_bank.valid.detach().cpu().numpy().copy()
    candidates = output.candidate_layouts.detach().cpu().numpy().copy()
    metric = np.zeros((batch_size, proposal_cap, 3), dtype=np.int16)
    positive = np.zeros((batch_size, proposal_cap), dtype=bool)
    risk = np.zeros((batch_size, proposal_cap), dtype=bool)
    edges = np.empty((batch_size, 2, GRID_SIZE * GRID_SIZE), dtype=np.int64)
    boundaries = np.empty((batch_size, GRID_SIZE * GRID_SIZE, 4, TILE_SIZE, 6), np.float32)
    for index, case in enumerate(cases):
        metric[index] = metric_deltas_for_bank(
            case.control,
            case.truth,
            positions[index],
            lengths[index],
            valid[index],
            grid_size=GRID_SIZE,
        )
        positive[index] = frozen_positive_action_mask(metric[index], valid[index])
        risk[index] = pair_loss_labels(
            case.control,
            case.truth,
            candidates[index],
            valid[index],
        )
        edges[index] = directional_edge_targets(case.truth)
        boundaries[index] = clean_boundary_targets(case.clean_tiles_by_identity)
    if not np.array_equal(positions, output.proposal_bank.positions.detach().cpu().numpy()):
        raise AssertionError("label attachment changed proposal identities")
    device = output.action_logits.device
    return StageBLabels(
        positive_actions=torch.from_numpy(positive).to(device),
        metric_deltas=torch.from_numpy(metric).to(device=device, dtype=output.quantiles.dtype),
        loses_true_pair=torch.from_numpy(risk).to(device),
        edge_targets=torch.from_numpy(edges).to(device),
        clean_boundary_targets=torch.from_numpy(boundaries).to(device),
    )


def fit_starvation_diagnostic(
    output: StageBOutput,
    labels: StageBLabels,
    cases: Sequence[VisibleCase],
    *,
    include_exhaustive_first_case: bool,
) -> dict[str, Any]:
    """Report proposal supply on FIT labels only; never changes training or selection."""

    valid = output.proposal_bank.valid.detach().cpu().numpy()
    metric = labels.metric_deltas.detach().cpu().numpy()
    proposal_best = [
        int(metric[index, valid[index], 0].max(initial=0)) for index in range(len(cases))
    ]
    result: dict[str, Any] = {
        "state_count": len(cases),
        "mean_proposal_count": float(valid.sum(axis=1).mean()),
        "positive_pair_proposal_fraction": float(np.mean(np.asarray(proposal_best) > 0)),
        "mean_best_proposal_pair_delta": float(np.mean(proposal_best)),
        "keep_is_positive_fraction": float(
            np.mean(labels.positive_actions.detach().cpu().numpy()[:, 0])
        ),
        "exhaustive_first_case": None,
    }
    if include_exhaustive_first_case:
        diagnostic = proposal_oracle_diagnostic(
            cases[0].control,
            cases[0].truth,
            output.proposal_bank.positions[0].detach().cpu().numpy(),
            output.proposal_bank.lengths[0].detach().cpu().numpy(),
            valid[0],
            grid_size=GRID_SIZE,
        )
        result["exhaustive_first_case"] = asdict(diagnostic)
    return result


def build_stage_b_model(config: Mapping[str, Any], *, device: torch.device) -> BasinCycleStageB:
    """Construct exactly the preregistered architecture and verify its ledger."""

    architecture = config["architecture"]
    model = BasinCycleStageB(
        grid_size=int(architecture["grid_size"]),
        feature_channels=int(architecture["feature_channels"]),
        retrieval_dim=int(architecture["retrieval_dim"]),
        state_dim=int(architecture["state_dim"]),
        encoder_blocks=int(architecture["image_encoder_blocks"]),
        state_blocks=int(architecture["state_blocks"]),
        proposal_top_k=int(architecture["proposal_top_k"]),
        proposal_candidate_cap=int(architecture["proposal_candidate_cap"]),
        proposal_seed_count=int(architecture["proposal_seed_count"]),
        max_cycle_length=int(architecture["max_cycle_length"]),
        proposal_cap=int(architecture["proposal_cap_including_keep"]),
    ).to(device)
    ledger = model_static_ledger(model, batch_size=BATCH_SIZE)
    expected = config["compute_ledger"]
    if ledger["trainable_parameters"] != architecture["trainable_parameters_exact"]:
        raise ValueError("Stage-B parameter count differs from preregistration")
    if ledger["forward_learned_macs_per_board"] != expected["forward_learned_macs_per_board"]:
        raise ValueError("Stage-B MAC ledger differs from preregistration")
    return model


def choose_device(name: str) -> torch.device:
    """Resolve the only supported fit devices."""

    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name not in {"cpu", "mps"}:
        raise ValueError("device must be cpu, mps, or auto")
    return torch.device(name)


def fit_model(
    *,
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
    config_sha256: str,
    binding_sha256: str,
    targets_root: Path,
    socket_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run the one fixed 2,000-update fit; no resume or checkpoint selection."""

    if output_dir.exists():
        raise FileExistsError("fit output directory already exists; resume/reuse is forbidden")
    output_dir.mkdir(parents=True)
    seed = int(config["training"]["model_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_stage_b_model(config, device=device)
    socket = load_frozen_socket(socket_checkpoint, device=device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FIT_UPDATES,
        eta_min=0.0,
    )
    rows = fit_plan(config["source_protocol"]["fit_filenames"])
    workers = int(binding["execution"]["cpu_prefetch_workers"])
    clean_tile_canvases = preload_clean_tile_canvases(
        config["source_protocol"]["fit_filenames"],
        targets_root=targets_root,
        workers=workers,
    )
    diagnostic_steps = set(int(value) for value in binding["fit_diagnostics"]["steps_zero_based"])
    log_every = int(binding["execution"]["log_every_updates"])
    diagnostics: list[dict[str, Any]] = []
    recent: list[float] = []
    model.train()
    first_rows = rows[:BATCH_SIZE]
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="basincycle-fit-prefetch",
    ) as pool:
        pending = submit_case_preparation(
            pool,
            first_rows,
            clean_tile_canvases=clean_tile_canvases,
        )
        for step in range(FIT_UPDATES):
            prepared = resolve_case_preparation(pending)
            if step + 1 < FIT_UPDATES:
                next_rows = rows[(step + 1) * BATCH_SIZE : (step + 2) * BATCH_SIZE]
                pending = submit_case_preparation(
                    pool,
                    next_rows,
                    clean_tile_canvases=clean_tile_canvases,
                )
            cases = materialize_prepared_cases(
                prepared,
                socket=socket,
                socket_device=device,
            )
            tiles = torch.stack([case.tiles for case in cases]).to(device)
            layouts = torch.from_numpy(np.stack([case.control for case in cases])).to(device)
            output = model(tiles, layouts)
            labels = labels_for_output(output, cases)
            if step in diagnostic_steps:
                record = fit_starvation_diagnostic(
                    output,
                    labels,
                    cases,
                    include_exhaustive_first_case=True,
                )
                record["step_zero_based"] = step
                diagnostics.append(record)
            loss, parts = stage_b_loss(
                output,
                labels,
                edge_weight=float(training["loss"]["directional_edge_cross_entropy"]),
                restore_weight=float(training["loss"]["clean_boundary_charbonnier"]),
                quantile_weight=float(
                    training["loss"]["pair_exact_radius2_quantile_pinball"]
                ),
                risk_weight=float(training["loss"]["any_true_pair_loss_bce"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training["gradient_clip_norm"]),
                )
            )
            optimizer.step()
            scheduler.step()
            recent.append(float(loss.detach().cpu()))
            if step == 0 or (step + 1) % log_every == 0 or step + 1 == FIT_UPDATES:
                payload = {
                    "event": "basincycle_stage_b_fit",
                    "update": step + 1,
                    "loss_mean_recent": float(np.mean(recent[-log_every:])),
                    "loss_parts": {
                        name: float(value.detach().cpu()) for name, value in parts.items()
                    },
                    "gradient_norm": gradient_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                print(json.dumps(payload, sort_keys=True), flush=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    checkpoint_path = output_dir / "basincycle_stage_b_final.pt"
    torch.save(
        {
            "schema": FIT_CHECKPOINT_SCHEMA,
            "scientific_config_sha256": config_sha256,
            "execution_binding_sha256": binding_sha256,
            "final_update": FIT_UPDATES,
            "selected_checkpoint": "final-update-2000-only",
            "model_seed": seed,
            "state_dict": state,
            "fit_diagnostics": diagnostics,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    report = {
        "schema": "aiijc-basincycle-stage-b-fit-report-v1",
        "status": "fit-complete-unscored-final-endpoint-only",
        "scientific_config_sha256": config_sha256,
        "execution_binding_sha256": binding_sha256,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "final_update": FIT_UPDATES,
        },
        "fit_source_count": FIT_SOURCE_COUNT,
        "update_count": FIT_UPDATES,
        "batch_size": BATCH_SIZE,
        "selection_or_resume_performed": False,
        "fit_only_proposal_starvation_diagnostics": diagnostics,
        "evaluation_sources_or_labels_opened": False,
    }
    atomic_write_json(output_dir / "fit_report.json", report)
    return report


def load_final_model(
    checkpoint_path: Path,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    binding_sha256: str,
    device: torch.device,
) -> tuple[BasinCycleStageB, dict[str, Any]]:
    """Load only the final, non-selected endpoint of the one signed fit."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "schema": FIT_CHECKPOINT_SCHEMA,
        "scientific_config_sha256": config_sha256,
        "execution_binding_sha256": binding_sha256,
        "final_update": FIT_UPDATES,
        "selected_checkpoint": "final-update-2000-only",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"final Stage-B checkpoint differs at {key}")
    model = build_stage_b_model(config, device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _freeze_array_inventory() -> tuple[str, ...]:
    return (
        "source_index",
        "draw_index",
        "state_family_code",
        "control_layout",
        "proposal_positions",
        "proposal_lengths",
        "proposal_valid",
        "candidate_layouts",
        "pair_logits",
        "action_logits",
        "quantiles",
        "risk_logits",
        "selected_index",
        "selected_layout",
    )


def validate_target_free_generation_firewall_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Disambiguate synthetic case construction from later oracle attachment."""

    expected = {
        "reference_semantics": TARGET_FREE_REFERENCE_SEMANTICS,
        "synthetic_shuffle_truth_constructed_for_case_generation": True,
        "procedural_control_initialized_from_planted_truth": True,
        "derived_procedural_control_supplied_to_model": True,
        "synthetic_shuffle_truth_supplied_directly_to_model_proposals_or_selector": False,
        "synthetic_shuffle_truth_persisted_in_bundle": False,
        "clean_pixels_supplied_directly_to_model_proposals_or_selector": False,
        "clean_pixels_persisted_in_bundle": False,
        "evaluation_metric_or_oracle_attached_before_freeze": False,
        "all_selected_outputs_strict": True,
        "selection_or_threshold_sweep_performed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"target-free generation-firewall receipt failed: {key}")


def validate_frozen_array_hashes(
    arrays: Mapping[str, np.ndarray],
    receipt: Mapping[str, Any],
) -> None:
    """Bind in-memory arrays to the same hashes recorded before scoring."""

    prediction_sha256 = sha256_arrays(
        [(name, arrays[name]) for name in _freeze_array_inventory()]
    )
    if prediction_sha256 != receipt.get("prediction_roster_sha256"):
        raise ValueError("target-free prediction roster digest mismatch")
    proposal_sha256 = sha256_arrays(
        [
            ("proposal_positions", arrays["proposal_positions"]),
            ("proposal_lengths", arrays["proposal_lengths"]),
            ("proposal_valid", arrays["proposal_valid"]),
        ]
    )
    if proposal_sha256 != receipt.get("proposal_identity_sha256"):
        raise ValueError("proposal identity digest mismatch")
    control_sha256 = sha256_arrays([("control_layout", arrays["control_layout"])])
    if control_sha256 != receipt.get("control_layout_sha256"):
        raise ValueError("control layout digest mismatch")


def validate_frozen_array_semantics(arrays: Mapping[str, np.ndarray]) -> None:
    """Fail closed on the complete non-reference freeze tensor contract."""

    if set(arrays) != set(_freeze_array_inventory()):
        raise ValueError("target-free array inventory changed")
    case_count = EVAL_SOURCE_COUNT * len(EVAL_DRAWS)
    expected_shapes = {
        "source_index": (case_count,),
        "draw_index": (case_count,),
        "state_family_code": (case_count,),
        "control_layout": (case_count, GRID_SIZE * GRID_SIZE),
        "proposal_positions": (case_count, PROPOSAL_CAP, MAX_CYCLE_LENGTH),
        "proposal_lengths": (case_count, PROPOSAL_CAP),
        "proposal_valid": (case_count, PROPOSAL_CAP),
        "candidate_layouts": (
            case_count,
            PROPOSAL_CAP,
            GRID_SIZE * GRID_SIZE,
        ),
        "pair_logits": (
            case_count,
            2,
            GRID_SIZE * GRID_SIZE,
            GRID_SIZE * GRID_SIZE,
        ),
        "action_logits": (case_count, PROPOSAL_CAP),
        "quantiles": (case_count, PROPOSAL_CAP, 3, 3),
        "risk_logits": (case_count, PROPOSAL_CAP),
        "selected_index": (case_count,),
        "selected_layout": (case_count, GRID_SIZE * GRID_SIZE),
    }
    expected_dtypes = {
        "source_index": np.dtype(np.int16),
        "draw_index": np.dtype(np.int8),
        "state_family_code": np.dtype(np.int8),
        "control_layout": np.dtype(np.int16),
        "proposal_positions": np.dtype(np.int16),
        "proposal_lengths": np.dtype(np.int8),
        "proposal_valid": np.dtype(np.bool_),
        "candidate_layouts": np.dtype(np.int16),
        "pair_logits": np.dtype(np.float32),
        "action_logits": np.dtype(np.float32),
        "quantiles": np.dtype(np.float32),
        "risk_logits": np.dtype(np.float32),
        "selected_index": np.dtype(np.int16),
        "selected_layout": np.dtype(np.int16),
    }
    for name in _freeze_array_inventory():
        value = np.asarray(arrays[name])
        if value.shape != expected_shapes[name] or value.dtype != expected_dtypes[name]:
            raise ValueError(f"target-free array shape/dtype changed: {name}")

    expected_sources = np.repeat(np.arange(EVAL_SOURCE_COUNT, dtype=np.int16), 2)
    expected_draws = np.tile(np.asarray(EVAL_DRAWS, dtype=np.int8), EVAL_SOURCE_COUNT)
    expected_families = np.tile(np.asarray((0, 1), dtype=np.int8), EVAL_SOURCE_COUNT)
    if not np.array_equal(arrays["source_index"], expected_sources):
        raise ValueError("target-free source index roster changed")
    if not np.array_equal(arrays["draw_index"], expected_draws):
        raise ValueError("target-free draw roster changed")
    if not np.array_equal(arrays["state_family_code"], expected_families):
        raise ValueError("target-free state-family roster changed")

    pair_logits = np.asarray(arrays["pair_logits"])
    action_logits = np.asarray(arrays["action_logits"])
    quantiles = np.asarray(arrays["quantiles"])
    risk_logits = np.asarray(arrays["risk_logits"])
    valid = np.asarray(arrays["proposal_valid"])
    if not np.isfinite(pair_logits).all() or not np.isfinite(quantiles).all():
        raise ValueError("target-free learned predictions must be finite")
    if not np.isfinite(action_logits[valid]).all() or not np.isfinite(
        risk_logits[valid]
    ).all():
        raise ValueError("valid target-free action predictions must be finite")
    if not np.isneginf(action_logits[~valid]).all() or not np.isposinf(
        risk_logits[~valid]
    ).all():
        raise ValueError("target-free padding logits changed")
    if np.any(quantiles[..., 0] > quantiles[..., 1]) or np.any(
        quantiles[..., 1] > quantiles[..., 2]
    ):
        raise ValueError("target-free quantile order changed")

    controls = np.asarray(arrays["control_layout"])
    positions = np.asarray(arrays["proposal_positions"])
    lengths = np.asarray(arrays["proposal_lengths"])
    candidates = np.asarray(arrays["candidate_layouts"])
    selected_indices = np.asarray(arrays["selected_index"], dtype=np.int64)
    selected_layouts = np.asarray(arrays["selected_layout"])
    for case_index in range(case_count):
        control = controls[case_index].astype(np.int64, copy=False)
        if not is_strict_permutation(control):
            raise ValueError("target-free control is not a strict permutation")
        case_valid = valid[case_index]
        valid_count = int(case_valid.sum())
        if valid_count < 1 or not np.array_equal(
            case_valid,
            np.arange(PROPOSAL_CAP) < valid_count,
        ):
            raise ValueError("target-free valid proposal prefix changed")
        if lengths[case_index, 0] != 0 or np.any(positions[case_index, 0] != -1):
            raise ValueError("target-free KEEP identity changed")
        if not np.array_equal(candidates[case_index, 0], control):
            raise ValueError("target-free KEEP candidate differs from control")
        seen_cycles: set[tuple[int, ...]] = set()
        for proposal_index in range(1, PROPOSAL_CAP):
            if not case_valid[proposal_index]:
                if lengths[case_index, proposal_index] != 0 or np.any(
                    positions[case_index, proposal_index] != -1
                ):
                    raise ValueError("target-free padding proposal identity changed")
                if not np.array_equal(candidates[case_index, proposal_index], control):
                    raise ValueError("target-free padding candidate changed")
                continue
            length = int(lengths[case_index, proposal_index])
            if length not in (2, 3):
                raise ValueError("target-free action is not a closed 2/3-cycle")
            cycle = tuple(
                int(value)
                for value in positions[case_index, proposal_index, :length]
            )
            canonical = canonical_cycle(cycle)
            if (
                len(set(cycle)) != length
                or min(cycle) < 0
                or max(cycle) >= GRID_SIZE * GRID_SIZE
                or np.any(positions[case_index, proposal_index, length:] != -1)
                or cycle != canonical
                or canonical in seen_cycles
            ):
                raise ValueError("target-free cycle identity is malformed or duplicated")
            seen_cycles.add(canonical)
            expected_candidate = apply_cycle(control, cycle)
            if not np.array_equal(candidates[case_index, proposal_index], expected_candidate):
                raise ValueError("target-free candidate differs from its frozen cycle")
        selected_index = int(selected_indices[case_index])
        if not 0 <= selected_index < PROPOSAL_CAP or not case_valid[selected_index]:
            raise ValueError("target-free selected index is invalid")
        if not np.array_equal(
            selected_layouts[case_index],
            candidates[case_index, selected_index],
        ):
            raise ValueError("target-free selected layout differs from selected index")
        if not is_strict_permutation(selected_layouts[case_index]):
            raise ValueError("target-free selected layout is not a strict permutation")


def validate_frozen_selection(
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> None:
    """Recompute the fixed target-free selector from frozen value predictions."""

    output = StageBOutput(
        pair_logits=torch.from_numpy(np.asarray(arrays["pair_logits"])),
        boundary_prediction=torch.empty(0),
        proposal_bank=ProposalBank(
            positions=torch.from_numpy(np.asarray(arrays["proposal_positions"])).long(),
            lengths=torch.from_numpy(np.asarray(arrays["proposal_lengths"])).long(),
            valid=torch.from_numpy(np.asarray(arrays["proposal_valid"])),
        ),
        candidate_layouts=torch.from_numpy(
            np.asarray(arrays["candidate_layouts"])
        ).long(),
        action_logits=torch.from_numpy(np.asarray(arrays["action_logits"])),
        quantiles=torch.from_numpy(np.asarray(arrays["quantiles"])),
        risk_logits=torch.from_numpy(np.asarray(arrays["risk_logits"])),
    )
    selected_index, selected_layout = select_hard_action(
        output,
        minimum_pair_q10=float(config["inference"]["minimum_pair_q10"]),
        maximum_risk=float(config["inference"]["maximum_pair_loss_risk"]),
        minimum_pair_q50_margin_over_keep=float(
            config["inference"]["minimum_pair_q50_margin_over_keep"]
        ),
    )
    if not np.array_equal(
        selected_index.numpy(),
        np.asarray(arrays["selected_index"], dtype=np.int64),
    ) or not np.array_equal(
        selected_layout.numpy(),
        np.asarray(arrays["selected_layout"], dtype=np.int64),
    ):
        raise ValueError("frozen selection differs from the fixed Stage-B selector")


@torch.no_grad()
def freeze_eval_predictions(
    *,
    model: BasinCycleStageB,
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
    config_sha256: str,
    binding_sha256: str,
    checkpoint_path: Path,
    targets_root: Path,
    socket_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Freeze EVAL32 predictions and proposal identities before metric attachment."""

    if output_dir.exists():
        raise FileExistsError("freeze output directory already exists; overwrite is forbidden")
    output_dir.mkdir(parents=True)
    socket = load_frozen_socket(socket_checkpoint, device=device)
    rows = eval_plan(config["source_protocol"]["eval_filenames"])
    workers = int(binding["execution"]["cpu_prefetch_workers"])
    clean_tile_canvases = preload_clean_tile_canvases(
        config["source_protocol"]["eval_filenames"],
        targets_root=targets_root,
        workers=workers,
    )
    storage: dict[str, list[np.ndarray]] = {name: [] for name in _freeze_array_inventory()}
    all_controls_strict = True
    all_candidates_strict = True
    all_outputs_strict = True
    all_keep = True
    family_codes = {"solver_replay": 0, "procedural": 1}
    for batch_start, cases in iter_prefetched_case_batches(
        rows,
        batch_size=BATCH_SIZE,
        clean_tile_canvases=clean_tile_canvases,
        socket=socket,
        socket_device=device,
        workers=workers,
        thread_name_prefix="basincycle-freeze-prefetch",
    ):
        batch_rows = rows[batch_start : batch_start + BATCH_SIZE]
        # From here onward, truth and clean pixels are deliberately not passed to
        # the predictor, proposal builder, hard selector, or persisted bundle.
        tiles = torch.stack([case.tiles for case in cases]).to(device)
        control = torch.from_numpy(np.stack([case.control for case in cases])).to(device)
        output = model(tiles, control)
        selected_index, selected_layout = select_hard_action(
            output,
            minimum_pair_q10=float(config["inference"]["minimum_pair_q10"]),
            maximum_risk=float(config["inference"]["maximum_pair_loss_risk"]),
            minimum_pair_q50_margin_over_keep=float(
                config["inference"]["minimum_pair_q50_margin_over_keep"]
            ),
        )
        for local_index, (row, case) in enumerate(zip(batch_rows, cases, strict=True)):
            case_index = batch_start + local_index
            candidates = output.candidate_layouts[local_index].detach().cpu().numpy()
            valid = output.proposal_bank.valid[local_index].detach().cpu().numpy()
            all_controls_strict &= is_strict_permutation(case.control)
            all_candidates_strict &= all(
                is_strict_permutation(candidate) for candidate in candidates[valid]
            )
            all_outputs_strict &= is_strict_permutation(
                selected_layout[local_index].cpu().numpy()
            )
            all_keep &= np.array_equal(candidates[0], case.control)
            values = {
                "source_index": np.asarray(case_index // len(EVAL_DRAWS), dtype=np.int16),
                "draw_index": np.asarray(row.batch_slot_or_draw, dtype=np.int8),
                "state_family_code": np.asarray(
                    family_codes[row.state_family], dtype=np.int8
                ),
                "control_layout": case.control.astype(np.int16),
                "proposal_positions": output.proposal_bank.positions[local_index]
                .cpu()
                .numpy()
                .astype(np.int16),
                "proposal_lengths": output.proposal_bank.lengths[local_index]
                .cpu()
                .numpy()
                .astype(np.int8),
                "proposal_valid": valid,
                "candidate_layouts": candidates.astype(np.int16),
                "pair_logits": output.pair_logits[local_index]
                .cpu()
                .numpy()
                .astype(np.float32),
                "action_logits": output.action_logits[local_index]
                .cpu()
                .numpy()
                .astype(np.float32),
                "quantiles": output.quantiles[local_index]
                .cpu()
                .numpy()
                .astype(np.float32),
                "risk_logits": output.risk_logits[local_index]
                .cpu()
                .numpy()
                .astype(np.float32),
                "selected_index": selected_index[local_index]
                .cpu()
                .numpy()
                .astype(np.int16),
                "selected_layout": selected_layout[local_index]
                .cpu()
                .numpy()
                .astype(np.int16),
            }
            for name in _freeze_array_inventory():
                storage[name].append(values[name])
        completed = batch_start + len(batch_rows)
        if completed % 8 == 0 or completed == len(rows):
            print(
                json.dumps(
                    {"event": "basincycle_stage_b_freeze", "case": completed},
                    sort_keys=True,
                ),
                flush=True,
            )
    arrays = {name: np.stack(values) for name, values in storage.items()}
    validate_frozen_array_semantics(arrays)
    validate_frozen_selection(arrays, config)
    bundle_path = output_dir / "target_free_predictions.npz"
    bundle_sha256 = atomic_save_npz(bundle_path, arrays)
    proposal_sha256 = sha256_arrays(
        [
            ("proposal_positions", arrays["proposal_positions"]),
            ("proposal_lengths", arrays["proposal_lengths"]),
            ("proposal_valid", arrays["proposal_valid"]),
        ]
    )
    control_sha256 = sha256_arrays([("control_layout", arrays["control_layout"])])
    prediction_sha256 = sha256_arrays(
        [(name, arrays[name]) for name in _freeze_array_inventory()]
    )
    receipt = {
        "schema": FREEZE_RECEIPT_SCHEMA,
        "config_sha256": config_sha256,
        "execution_binding_sha256": binding_sha256,
        "reference_opened": False,
        "reference_semantics": TARGET_FREE_REFERENCE_SEMANTICS,
        "synthetic_shuffle_truth_constructed_for_case_generation": True,
        "procedural_control_initialized_from_planted_truth": True,
        "derived_procedural_control_supplied_to_model": True,
        "synthetic_shuffle_truth_supplied_directly_to_model_proposals_or_selector": False,
        "synthetic_shuffle_truth_persisted_in_bundle": False,
        "clean_pixels_supplied_directly_to_model_proposals_or_selector": False,
        "clean_pixels_persisted_in_bundle": False,
        "evaluation_metric_or_oracle_attached_before_freeze": False,
        "all_controls_strict": bool(all_controls_strict),
        "all_banks_keep_index0": bool(all_keep),
        "all_candidate_layouts_strict": bool(all_candidates_strict),
        "all_selected_outputs_strict": bool(all_outputs_strict),
        "eval_case_count": len(rows),
        "model_sha256": sha256_file(checkpoint_path),
        "prediction_roster_sha256": prediction_sha256,
        "proposal_identity_sha256": proposal_sha256,
        "control_layout_sha256": control_sha256,
        "bundle": {"path": bundle_path.name, "sha256": bundle_sha256},
        "eval_source_digest": config["source_protocol"]["eval_digest"],
        "eval_plan_digest": config["corruption_plan"]["eval_plan_digest"],
        "selection_or_threshold_sweep_performed": False,
    }
    require_target_free_freeze_receipt(receipt, config_sha256=config_sha256)
    validate_target_free_generation_firewall_receipt(receipt)
    atomic_write_json(output_dir / "freeze_receipt.json", receipt)
    return receipt


def validate_freeze_bundle(
    *,
    bundle_path: Path,
    receipt: Mapping[str, Any],
    config_sha256: str,
    binding_sha256: str,
) -> dict[str, np.ndarray]:
    """Read back and independently hash every frozen prediction tensor."""

    require_target_free_freeze_receipt(receipt, config_sha256=config_sha256)
    validate_target_free_generation_firewall_receipt(receipt)
    if receipt.get("execution_binding_sha256") != binding_sha256:
        raise ValueError("freeze receipt was produced under another execution binding")
    if sha256_file(bundle_path) != receipt.get("bundle", {}).get("sha256"):
        raise ValueError("target-free prediction bundle hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as archive:
        if set(archive.files) != set(_freeze_array_inventory()):
            raise ValueError("target-free bundle array inventory changed")
        arrays = {name: archive[name].copy() for name in archive.files}
    if len(arrays["control_layout"]) != EVAL_SOURCE_COUNT * len(EVAL_DRAWS):
        raise ValueError("target-free bundle case count changed")
    validate_frozen_array_hashes(arrays, receipt)
    validate_frozen_array_semantics(arrays)
    return arrays


def source_clustered_mean_ci(
    values: np.ndarray,
    source_indices: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    """Bootstrap sources, retaining both fixed draws inside each sampled cluster."""

    observed = np.asarray(values, dtype=np.float64)
    sources = np.asarray(source_indices, dtype=np.int64)
    unique = np.unique(sources)
    if observed.shape != sources.shape or len(unique) != EVAL_SOURCE_COUNT:
        raise ValueError("source-clustered values do not match the frozen EVAL32 roster")
    clusters = [observed[sources == source] for source in unique]
    if any(len(cluster) != len(EVAL_DRAWS) for cluster in clusters):
        raise ValueError("each reserved source must retain exactly two fixed draws")
    cluster_means = np.asarray([cluster.mean() for cluster in clusters])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(resamples, len(unique)))
    bootstrap = cluster_means[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "mean": float(observed.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "source_count": float(len(unique)),
        "case_count": float(len(observed)),
    }


def _stratum_summary(
    metric_deltas: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = metric_deltas[mask]
    return {
        "case_count": int(mask.sum()),
        "pair_delta_mean": float(selected[:, 0].mean()),
        "exact_delta_mean": float(selected[:, 1].mean()),
        "radius2_delta_mean": float(selected[:, 2].mean()),
        "pair_nonworsening_fraction": float(np.mean(selected[:, 0] >= 0)),
        "pair_delta_minimum": int(selected[:, 0].min()),
    }


def score_frozen_predictions(
    *,
    arrays: Mapping[str, np.ndarray],
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    binding: Mapping[str, Any],
    binding_sha256: str,
) -> dict[str, Any]:
    """Attach exact references once, score fixed outputs, and evaluate every gate."""

    require_target_free_freeze_receipt(receipt, config_sha256=config_sha256)
    validate_target_free_generation_firewall_receipt(receipt)
    if receipt.get("execution_binding_sha256") != binding_sha256:
        raise ValueError("score receipt was produced under another execution binding")
    if receipt.get("eval_source_digest") != config["source_protocol"]["eval_digest"]:
        raise ValueError("score receipt EVAL source digest changed")
    if receipt.get("eval_plan_digest") != config["corruption_plan"]["eval_plan_digest"]:
        raise ValueError("score receipt EVAL plan digest changed")
    validate_frozen_array_semantics(arrays)
    validate_frozen_array_hashes(arrays, receipt)
    validate_frozen_selection(arrays, config)
    rows = eval_plan(config["source_protocol"]["eval_filenames"])
    selected_deltas = np.zeros((len(rows), 3), dtype=np.int16)
    diagnostics: list[OracleDiagnostic] = []
    strict_controls = True
    strict_candidates = True
    strict_outputs = True
    keep_replays = True
    for index, row in enumerate(rows):
        truth, _, _ = truth_and_tile_order(row.state_seed)
        control = arrays["control_layout"][index].astype(np.int64)
        selected = arrays["selected_layout"][index].astype(np.int64)
        valid = arrays["proposal_valid"][index]
        candidates = arrays["candidate_layouts"][index].astype(np.int64)
        strict_controls &= is_strict_permutation(control)
        strict_candidates &= all(is_strict_permutation(item) for item in candidates[valid])
        strict_outputs &= is_strict_permutation(selected)
        keep_replays &= np.array_equal(candidates[0], control)
        baseline = np.asarray(
            (
                true_pair_count(control, truth, grid_size=GRID_SIZE),
                exact_count(control, truth),
                radius_two_count(control, truth, grid_size=GRID_SIZE),
            )
        )
        achieved = np.asarray(
            (
                true_pair_count(selected, truth, grid_size=GRID_SIZE),
                exact_count(selected, truth),
                radius_two_count(selected, truth, grid_size=GRID_SIZE),
            )
        )
        selected_deltas[index] = achieved - baseline
        diagnostics.append(
            proposal_oracle_diagnostic(
                control,
                truth,
                arrays["proposal_positions"][index],
                arrays["proposal_lengths"][index],
                valid,
                grid_size=GRID_SIZE,
            )
        )
    source_indices = arrays["source_index"]
    bootstrap = binding["score"]["source_clustered_bootstrap"]
    pair_ci = source_clustered_mean_ci(
        selected_deltas[:, 0],
        source_indices,
        seed=int(bootstrap["seed"]),
        resamples=int(bootstrap["resamples"]),
    )
    oracle = aggregate_oracle_diagnostics(diagnostics)
    family = arrays["state_family_code"]
    strata = {
        "solver_replay": _stratum_summary(selected_deltas, family == 0),
        "procedural": _stratum_summary(selected_deltas, family == 1),
        "all": _stratum_summary(selected_deltas, np.ones(len(rows), dtype=bool)),
    }
    gates = config["fail_stop_gates"]
    coverage = oracle["proposal_oracle_coverage"]
    gate_results = {
        "strict_controls_candidates_outputs": bool(
            strict_controls and strict_candidates and strict_outputs
        ),
        "keep_index_zero_exact_replay": bool(keep_replays),
        "proposal_oracle_coverage": bool(
            coverage is not None
            and coverage >= gates["minimum_proposal_oracle_coverage_on_opportunity_states"]
        ),
        "selected_mean_pair_delta": bool(
            strata["all"]["pair_delta_mean"]
            >= gates["minimum_selected_mean_pair_delta_of_60"]
        ),
        "source_clustered_pair_ci_lower_positive": bool(pair_ci["ci95_lower"] > 0.0),
        "selected_mean_exact_delta_nonnegative": bool(
            strata["all"]["exact_delta_mean"] >= gates["minimum_selected_mean_exact_delta"]
        ),
        "selected_mean_radius2_delta_nonnegative": bool(
            strata["all"]["radius2_delta_mean"]
            >= gates["minimum_selected_mean_radius2_delta"]
        ),
        "pair_nonworsening_fraction": bool(
            strata["all"]["pair_nonworsening_fraction"]
            >= gates["minimum_pair_nonworsening_case_fraction"]
        ),
        "pair_catastrophe_bound": bool(
            strata["all"]["pair_delta_minimum"]
            >= gates["maximum_pair_catastrophe_delta"]
        ),
    }
    return {
        "schema": SCORE_REPORT_SCHEMA,
        "status": "pass" if all(gate_results.values()) else "fail-stop",
        "scientific_config_sha256": config_sha256,
        "execution_binding_sha256": binding_sha256,
        "freeze_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipt)).hexdigest(),
        "evaluation_source_count": EVAL_SOURCE_COUNT,
        "evaluation_case_count": len(rows),
        "pair_count_per_case": PAIR_COUNT,
        "selected_action_nonkeep_count": int(np.count_nonzero(arrays["selected_index"])),
        "source_clustered_pair_delta": pair_ci,
        "proposal_oracle": oracle,
        "proposal_diagnostics": [asdict(item) for item in diagnostics],
        "strata": strata,
        "gate_results": gate_results,
        "all_gates_pass": bool(all(gate_results.values())),
        "selection_threshold_or_checkpoint_sweep_performed": False,
        "dev_holdout_terminal_test_opened": False,
        "promotion_authorized": False,
    }


def audit_protocol(
    *,
    project_root: Path,
    config_path: Path,
    binding_path: Path,
) -> dict[str, Any]:
    """Run the complete metadata/hash audit without opening any image pixels."""

    config_sha256 = validate_sidecar(config_path)
    config = load_json(config_path)
    binding = load_json(binding_path)
    binding_hashes = validate_execution_binding(
        binding,
        project_root=project_root,
        binding_path=binding_path,
    )
    if binding_hashes["scientific_config"] != config_sha256:
        raise ValueError("binding and requested scientific config differ")
    protocol = validate_scientific_protocol(config, project_root=project_root)
    return {
        "scientific_config_sha256": config_sha256,
        "execution_binding_sha256": binding_hashes["execution_binding"],
        "binding_hashes": binding_hashes,
        "protocol": protocol,
        "organizer_pixels_opened": False,
        "organizer_labels_opened": False,
    }


def validate_execution_acknowledgement(binding: Mapping[str, Any], acknowledgement: str) -> None:
    """Require a deliberate parent-review acknowledgement before real execution."""

    expected = binding.get("execution", {}).get("required_review_acknowledgement")
    if not isinstance(expected, str) or acknowledgement != expected:
        raise PermissionError(
            "real Stage-B execution requires the exact reviewed-run acknowledgement"
        )
    if binding.get("real_execution_authorized") is not False:
        raise ValueError("binding authorization sentinel changed unexpectedly")


__all__ = [
    "EXECUTION_BINDING_SCHEMA",
    "EXECUTION_BINDING_STATUS",
    "FREEZE_BUNDLE_SCHEMA",
    "FREEZE_RECEIPT_SCHEMA",
    "FIT_CHECKPOINT_SCHEMA",
    "PAIR_COUNT",
    "SCORE_REPORT_SCHEMA",
    "VisibleCase",
    "atomic_save_npz",
    "atomic_write_json",
    "audit_protocol",
    "build_stage_b_model",
    "canonical_json_bytes",
    "choose_device",
    "clean_boundary_targets",
    "corrupt_clean_tiles",
    "directional_edge_targets",
    "fit_model",
    "fit_starvation_diagnostic",
    "freeze_eval_predictions",
    "labels_for_output",
    "load_final_model",
    "load_json",
    "pair_loss_labels",
    "procedural_control",
    "score_frozen_predictions",
    "sha256_arrays",
    "sha256_file",
    "source_clustered_mean_ci",
    "truth_and_tile_order",
    "validate_execution_acknowledgement",
    "validate_execution_binding",
    "validate_frozen_array_semantics",
    "validate_frozen_array_hashes",
    "validate_freeze_bundle",
    "validate_frozen_selection",
    "validate_scientific_protocol",
    "validate_sidecar",
    "validate_target_free_generation_firewall_receipt",
]
