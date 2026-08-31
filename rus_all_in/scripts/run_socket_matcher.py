#!/usr/bin/env python3
"""Train and evaluate the board-conditioned SocketGlue layout candidate.

The run uses only the manifest's ``train`` split.  Synthetic target crops carry
exact permutation labels; real challenge inputs use only the high-confidence
half of target-assisted recovered labels.  Evaluation sources are disjoint from
training sources, and every dirty-only score/layout is frozen before its clean
target is opened for diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from aiijc_puzzle.candidate_supply import RecoveredLayout, recover_layout
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.legacy_upgrade import (
    border_position_scores,
    directional_scores,
    solve_buddies,
    solve_relaxation,
    validate_layout,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import (
    SocketDecoderConfig,
    decode_socket_assignments,
    texture_centrality_unary,
)
from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    BORDER_HEAD_VERSIONS,
    SIDE_NAMES,
    SocketMatcher,
    SocketOutput,
    socket_matching_loss,
    socket_retrieval_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
SELECTION_NAMESPACE = "aiijc-socket-matcher-v1"
GRID = 24
TILE_COUNT = GRID * GRID


@dataclass(frozen=True)
class TrainingBoard:
    filename: str
    clean: np.ndarray
    dirty: np.ndarray
    recovered: RecoveredLayout
    trusted_position: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--eval-offset", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--synthetic-steps", type=int, default=80)
    parser.add_argument("--real-steps", type=int, default=80)
    parser.add_argument("--synthetic-grid", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--board-layers", type=int, default=1)
    parser.add_argument("--socket-layers", type=int, default=1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--border-weight", type=float, default=0.25)
    parser.add_argument(
        "--border-head-version",
        choices=BORDER_HEAD_VERSIONS,
        default=BORDER_HEAD_EMBEDDING_V2,
        help=(
            "embedding_v2 preserves existing checkpoints; score_stats_v3 also uses "
            "permutation-equivariant partner-score distribution statistics"
        ),
    )
    parser.add_argument(
        "--raw-rank-weight",
        type=float,
        default=0.0,
        help="optional bidirectional listwise CE weight on raw socket logits",
    )
    parser.add_argument("--real-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--component-prior-weight",
        type=float,
        default=0.0,
        help=(
            "optional weak component-level texture-to-centre unary weight; "
            "zero keeps the heuristic strictly disabled"
        ),
    )
    parser.add_argument("--schedule", choices=("sequential", "interleave"), default="interleave")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--warmstart-in", type=Path)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        # On current Apple Silicon, this model's many small attention kernels
        # are faster on CPU than MPS.  MPS remains an explicit option for scale.
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def names_digest(records: tuple[Any, ...]) -> str:
    value = "\n".join(str(record["filename"]) for record in records).encode()
    return hashlib.sha256(value).hexdigest()


def _select_records(
    manifest: dict[str, Any], args: argparse.Namespace
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    end = max(args.train_limit, args.eval_offset + args.eval_limit)
    panel = select_manifest_records(
        manifest,
        "train",
        limit=end,
        namespace=SELECTION_NAMESPACE,
    )
    train = tuple(panel[: args.train_limit])
    evaluation = tuple(panel[args.eval_offset : args.eval_offset + args.eval_limit])
    train_names = {record["filename"] for record in train}
    evaluation_names = {record["filename"] for record in evaluation}
    if train_names & evaluation_names or len(evaluation) != args.eval_limit:
        raise ValueError("training and evaluation source panels must be complete and disjoint")
    return train, evaluation


def prepare_training_boards(
    records: tuple[Any, ...], args: argparse.Namespace
) -> tuple[list[TrainingBoard], float]:
    started = perf_counter()
    boards: list[TrainingBoard] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = split_tiles(load_rgb(args.train_inputs / filename))
        clean = split_tiles(load_rgb(args.targets / filename))
        recovered = recover_layout(dirty, clean)
        cut = float(np.median(recovered.margin_at_position))
        trusted = recovered.margin_at_position >= cut
        boards.append(TrainingBoard(filename, clean, dirty, recovered, trusted))
        if index == 1 or index % 16 == 0 or index == len(records):
            print(f"prepared {index}/{len(records)} {filename}", flush=True)
    return boards, perf_counter() - started


def _torch_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high)


def challenge_augment(clean: torch.Tensor) -> torch.Tensor:
    """Fast approximation of the official independent per-tile corruption."""

    count = len(clean)
    gray = 0.299 * clean[:, :1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:3]
    pivot = gray.mean(dim=(1, 2, 3), keepdim=True)
    scale = _torch_uniform((count, 1, 1, 1), 0.70, 1.30, device=clean.device)
    offset = _torch_uniform((count, 1, 1, 1), -30 / 255, 30 / 255, device=clean.device)
    value = scale * (clean - pivot) + pivot + offset
    sigma = _torch_uniform((count, 1, 1, 1), 40 / 255, 55 / 255, device=clean.device)
    value = value + sigma * torch.randn_like(value)
    kernel = value.new_tensor([0.25, 0.5, 0.25])
    horizontal = kernel.reshape(1, 1, 1, 3).expand(3, 1, 1, 3)
    vertical = kernel.reshape(1, 1, 3, 1).expand(3, 1, 3, 1)
    value = F.conv2d(F.pad(value, (1, 1, 0, 0), mode="reflect"), horizontal, groups=3)
    value = F.conv2d(F.pad(value, (0, 0, 1, 1), mode="reflect"), vertical, groups=3)
    # JPEG quality 35..50 is approximated by per-tile quantisation and a weak
    # chroma/block perturbation; the real-input phase closes the remaining gap.
    levels = _torch_uniform((count, 1, 1, 1), 40.0, 72.0, device=clean.device)
    value = torch.round(value.clamp(0, 1) * levels) / levels
    return value.clamp(0.0, 1.0)


def mild_real_augment(tiles: torch.Tensor) -> torch.Tensor:
    count = len(tiles)
    gain = _torch_uniform((count, 3, 1, 1), 0.94, 1.06, device=tiles.device)
    bias = _torch_uniform((count, 3, 1, 1), -0.02, 0.02, device=tiles.device)
    noise = _torch_uniform((count, 1, 1, 1), 0.0, 0.012, device=tiles.device)
    return (tiles * gain + bias + noise * torch.randn_like(tiles)).clamp(0.0, 1.0)


def synthetic_example(
    board: TrainingBoard,
    *,
    grid: int,
    generator: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    clean_grid = board.clean.reshape(GRID, GRID, 20, 20, 3)
    row = int(generator.integers(0, GRID - grid + 1))
    column = int(generator.integers(0, GRID - grid + 1))
    crop = clean_grid[row : row + grid, column : column + grid].reshape(-1, 20, 20, 3)
    clean = torch.from_numpy(crop.astype(np.float32)).permute(0, 3, 1, 2).to(device) / 255.0
    corrupted = challenge_augment(clean)
    permutation = generator.permutation(grid * grid)
    shuffled = corrupted[torch.from_numpy(permutation).to(device)]
    tile_at_position = torch.from_numpy(np.argsort(permutation).copy()).to(device)
    return shuffled.unsqueeze(0), tile_at_position.unsqueeze(0), None


def real_example(
    board: TrainingBoard,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tiles = torch.from_numpy(board.dirty.astype(np.float32)).permute(0, 3, 1, 2).to(device)
    tiles = mild_real_augment(tiles / 255.0)
    layout = torch.from_numpy(board.recovered.dirty_at_position.copy()).to(device)
    trusted = torch.from_numpy(board.trusted_position.copy()).to(device)
    return tiles.unsqueeze(0), layout.unsqueeze(0), trusted.unsqueeze(0)


def train_model(
    model: SocketMatcher,
    boards: list[TrainingBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.synthetic_steps + args.real_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(total_steps, 1), eta_min=args.learning_rate * 0.08
    )
    generator = np.random.default_rng(args.seed + 1)
    history: list[dict[str, float]] = []
    started = perf_counter()
    phases = np.concatenate(
        (
            np.zeros(args.synthetic_steps, dtype=np.int8),
            np.ones(args.real_steps, dtype=np.int8),
        )
    )
    if args.schedule == "interleave":
        generator.shuffle(phases)
    for step, phase_code in enumerate(phases):
        synthetic = not bool(phase_code)
        board = boards[int(generator.integers(len(boards)))]
        if synthetic:
            tiles, layout, trusted = synthetic_example(
                board,
                grid=args.synthetic_grid,
                generator=generator,
                device=device,
            )
            grid = args.synthetic_grid
            phase = "synthetic_exact"
        else:
            tiles, layout, trusted = real_example(board, device=device)
            grid = GRID
            phase = "real_trusted"
        model.train()
        output = model(tiles, grid=grid)
        loss, diagnostics = socket_matching_loss(
            output,
            layout,
            grid=grid,
            trusted_position=trusted,
            border_weight=args.border_weight,
            raw_rank_weight=args.raw_rank_weight,
        )
        optimised_loss = loss if synthetic else args.real_loss_weight * loss
        optimizer.zero_grad(set_to_none=True)
        optimised_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        record = {
            "step": float(step + 1),
            "phase": 0.0 if synthetic else 1.0,
            "loss": diagnostics["loss"],
            "optimised_loss": float(optimised_loss.detach()),
            "right_nll": diagnostics["right_nll"],
            "down_nll": diagnostics["down_nll"],
            "border_nll": diagnostics["border_nll"],
            "raw_rank_nll": diagnostics["raw_rank_nll"],
            "right_raw_rank_nll": diagnostics["right_raw_rank_nll"],
            "down_raw_rank_nll": diagnostics["down_raw_rank_nll"],
            "right_raw_rank_supervised": diagnostics["right_raw_rank_supervised"],
            "down_raw_rank_supervised": diagnostics["down_raw_rank_supervised"],
            "grad_norm": grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == total_steps:
            recent = history[-min(args.log_every, len(history)) :]
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step + 1,
                        "phase": phase,
                        "grid": grid,
                        "loss": float(np.mean([item["loss"] for item in recent])),
                        "raw_rank_nll": float(
                            np.mean([item["raw_rank_nll"] for item in recent])
                        ),
                        "grad_norm": grad_norm,
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


def _socket_contract(args: argparse.Namespace) -> dict[str, Any]:
    architecture = {
        BORDER_HEAD_EMBEDDING_V2: "board-conditioned-partial-socket-matcher-v2",
        BORDER_HEAD_SCORE_STATS_V3: "board-conditioned-partial-socket-matcher-v3",
    }[args.border_head_version]
    border_description = (
        "four per-socket learned logits from contextual socket embeddings"
        if args.border_head_version == BORDER_HEAD_EMBEDDING_V2
        else (
            "four per-socket learned logits from contextual socket embeddings plus "
            "board-relative top1/top2/logmeanexp/entropy/mean/spread score statistics"
        )
    )
    return {
        "architecture": architecture,
        "dimension": args.dimension,
        "heads": args.heads,
        "board_layers": args.board_layers,
        "socket_layers": args.socket_layers,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "synthetic_grid": args.synthetic_grid,
        "synthetic_corruption": (
            "brightness/contrast/noise/gaussian/quantisation approximation of official ranges"
        ),
        "real_label_policy": "top 50% per-board recovered-layout margin; pairwise mask",
        "transport": (
            "576-to-576 fractional partial OT, one dustbin with mass capacity 24 per axis"
        ),
        "border_head_version": args.border_head_version,
        "border_heads": border_description,
        "raw_rank_auxiliary": (
            "bidirectional row/column CE on raw right/down logits; interior trusted pairs only, "
            "with untrusted candidates excluded"
        ),
        "raw_rank_weight": args.raw_rank_weight,
        "input_index_position_embedding": False,
    }


def load_or_create_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[SocketMatcher, dict[str, Any] | None]:
    model = SocketMatcher(
        dimension=args.dimension,
        heads=args.heads,
        board_layers=args.board_layers,
        socket_layers=args.socket_layers,
        sinkhorn_iterations=args.sinkhorn_iterations,
        border_head_version=args.border_head_version,
    ).to(device)
    if args.checkpoint_in is not None and args.warmstart_in is not None:
        raise ValueError("checkpoint-in and warmstart-in are mutually exclusive")
    if args.checkpoint_in is None and args.warmstart_in is None:
        return model, None
    path = args.checkpoint_in if args.checkpoint_in is not None else args.warmstart_in
    assert path is not None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if args.checkpoint_in is not None:
        old_contract = payload.get("contract", {})
        current_contract = _socket_contract(args)
        structural_keys = (
            "architecture",
            "dimension",
            "heads",
            "board_layers",
            "socket_layers",
            "sinkhorn_iterations",
        )
        if any(old_contract.get(key) != current_contract[key] for key in structural_keys):
            raise ValueError("checkpoint architecture differs from current arguments")
        model.load_state_dict(payload["state_dict"])
        return model, payload

    old_contract = payload.get("contract", {})
    old_architecture = old_contract.get("architecture")
    new_architecture = _socket_contract(args)["architecture"]
    allowed_transitions = {
        (
            "board-conditioned-partial-socket-matcher-v1",
            "board-conditioned-partial-socket-matcher-v2",
        ),
        (
            "board-conditioned-partial-socket-matcher-v1",
            "board-conditioned-partial-socket-matcher-v3",
        ),
        (
            "board-conditioned-partial-socket-matcher-v2",
            "board-conditioned-partial-socket-matcher-v2",
        ),
        (
            "board-conditioned-partial-socket-matcher-v2",
            "board-conditioned-partial-socket-matcher-v3",
        ),
        (
            "board-conditioned-partial-socket-matcher-v3",
            "board-conditioned-partial-socket-matcher-v3",
        ),
    }
    if (old_architecture, new_architecture) not in allowed_transitions:
        raise ValueError(
            f"unsupported SocketMatcher warm-start {old_architecture!r} -> {new_architecture!r}"
        )
    for key in ("dimension", "heads", "board_layers", "socket_layers", "sinkhorn_iterations"):
        if old_contract.get(key) != _socket_contract(args)[key]:
            raise ValueError(f"warm-start architecture differs for {key}")
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    expected_missing: set[str] = set()
    if old_architecture == "board-conditioned-partial-socket-matcher-v1":
        expected_missing.update(
            f"border_heads.{side}.{field}"
            for side in SIDE_NAMES
            for field in ("weight", "bias")
        )
    if new_architecture == "board-conditioned-partial-socket-matcher-v3" and (
        old_architecture != "board-conditioned-partial-socket-matcher-v3"
    ):
        expected_missing.update(f"border_distribution_heads.{side}.weight" for side in SIDE_NAMES)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "warm-start state differs beyond the declared border-head upgrade: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model, payload


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, 1e-6)


def learned_border_position_scores(
    *,
    right_out: np.ndarray,
    left_in: np.ndarray,
    bottom_out: np.ndarray,
    top_in: np.ndarray,
) -> np.ndarray:
    """Convert four learned socket-unmatched logits into a tile-to-slot unary."""

    cells = np.arange(TILE_COUNT)
    rows, columns = divmod(cells, GRID)
    unary = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float64)
    unary[:, columns == 0] += _robust_zscore(left_in)[:, None]
    unary[:, columns == GRID - 1] += _robust_zscore(right_out)[:, None]
    unary[:, rows == 0] += _robust_zscore(top_in)[:, None]
    unary[:, rows == GRID - 1] += _robust_zscore(bottom_out)[:, None]
    return unary.astype(np.float32)


@torch.no_grad()
def freeze_predictions(
    model: SocketMatcher,
    records: tuple[Any, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    model.eval()
    frozen: list[dict[str, Any]] = []
    started = perf_counter()
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty_image = load_rgb(args.train_inputs / filename)
        dirty = split_tiles(dirty_image)
        tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2)
        output = model((tensor / 255.0).unsqueeze(0).to(device), grid=GRID)
        raw_right = output.right_raw[0].float().cpu().numpy()
        raw_down = output.down_raw[0].float().cpu().numpy()
        right_assignment = output.right_log_assignment[0].float().cpu().numpy()
        down_assignment = output.down_log_assignment[0].float().cpu().numpy()
        right_out_border = output.right_out_border_logits[0].float().cpu().numpy()
        left_in_border = output.left_in_border_logits[0].float().cpu().numpy()
        bottom_out_border = output.bottom_out_border_logits[0].float().cpu().numpy()
        top_in_border = output.top_in_border_logits[0].float().cpu().numpy()

        # Preserve the real-vs-dustbin mass learned by partial OT.  A second
        # row-wise softmax over the real-real block would erase the outgoing
        # border probability and make every socket look internally matched.
        transport_normaliser = np.log(float(TILE_COUNT + GRID))
        socket_right = (
            right_assignment[:TILE_COUNT, :TILE_COUNT] + transport_normaliser
        )
        socket_down = down_assignment[:TILE_COUNT, :TILE_COUNT] + transport_normaliser
        baseline_right, baseline_down = directional_scores(dirty, views=("bilateral",))["bilateral"]
        fused_right = 0.8 * socket_right + 0.2 * baseline_right
        fused_down = 0.8 * socket_down + 0.2 * baseline_down
        learned_border = learned_border_position_scores(
            right_out=right_out_border,
            left_in=left_in_border,
            bottom_out=bottom_out_border,
            top_in=top_in_border,
        )
        decoder_config = SocketDecoderConfig(
            component_edge_budget_per_axis=144,
            swap_edge_budget_per_axis=144,
            max_swap_steps=24,
        )
        decoder = decode_socket_assignments(
            right_assignment,
            down_assignment,
            grid=GRID,
            config=decoder_config,
        )
        variants = {
            "bilateral_buddies96": solve_buddies(
                baseline_right, baseline_down, max_edges=96
            ).layout,
            "socket_ot_buddies96": solve_buddies(socket_right, socket_down, max_edges=96).layout,
            "fused_ot_buddies96": solve_buddies(fused_right, fused_down, max_edges=96).layout,
            "fused_ot_relax_border": solve_relaxation(
                fused_right,
                fused_down,
                position=border_position_scores(fused_right, fused_down),
                seed=args.seed + index,
            ).layout,
            "fused_ot_relax_learned_border": solve_relaxation(
                fused_right,
                fused_down,
                position=learned_border,
                seed=args.seed + index,
            ).layout,
            "socket_ot_decoder144": decoder.layout,
        }
        decoder_reports = {"socket_ot_decoder144": decoder.report()}
        if args.component_prior_weight > 0:
            component_prior = texture_centrality_unary(dirty, grid=GRID)
            prior_decoder = decode_socket_assignments(
                right_assignment,
                down_assignment,
                grid=GRID,
                config=SocketDecoderConfig(
                    component_edge_budget_per_axis=144,
                    swap_edge_budget_per_axis=144,
                    max_swap_steps=24,
                    component_shift_unary_weight=args.component_prior_weight,
                ),
                component_shift_unary=component_prior,
            )
            variants["socket_ot_decoder144_texture_centre"] = prior_decoder.layout
            decoder_reports["socket_ot_decoder144_texture_centre"] = prior_decoder.report()
        frozen.append(
            {
                "filename": filename,
                "dirty": dirty,
                "raw_right": raw_right,
                "raw_down": raw_down,
                "right_assignment": right_assignment,
                "down_assignment": down_assignment,
                "right_out_border": right_out_border,
                "left_in_border": left_in_border,
                "bottom_out_border": bottom_out_border,
                "top_in_border": top_in_border,
                "baseline_right": baseline_right,
                "baseline_down": baseline_down,
                "decoder_reports": decoder_reports,
                "variants": {
                    name: validate_layout(np.asarray(layout)) for name, layout in variants.items()
                },
            }
        )
        print(f"froze {index}/{len(records)} {filename}", flush=True)
    return frozen, perf_counter() - started


def _mean_numeric(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def directional_retrieval_metrics(
    right: np.ndarray,
    down: np.ndarray,
    recovered: RecoveredLayout,
    *,
    prefix: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, scores, delta in (("right", right, 1), ("down", down, GRID)):
        positions = np.arange(TILE_COUNT)
        valid = positions % GRID != GRID - 1 if name == "right" else positions < TILE_COUNT - GRID
        positions = positions[valid]
        anchors = recovered.dirty_at_position[positions]
        truth = recovered.dirty_at_position[positions + delta]
        order = np.argsort(-scores[anchors], axis=1)
        for k in (1, 5, 16, 32):
            output[f"{prefix}_{name}_r{k}"] = float(
                np.mean(np.any(order[:, :k] == truth[:, None], axis=1))
            )
    for k in (1, 5, 16, 32):
        output[f"{prefix}_pooled_r{k}"] = 0.5 * (
            output[f"{prefix}_right_r{k}"] + output[f"{prefix}_down_r{k}"]
        )
    return output


def evaluate_frozen(frozen: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    boards: list[dict[str, Any]] = []
    for item in frozen:
        clean_image = load_rgb(args.targets / item["filename"])
        clean = split_tiles(clean_image)
        recovered = recover_layout(item["dirty"], clean)
        socket_output = SocketOutput(
            right_raw=torch.from_numpy(item["raw_right"]).unsqueeze(0),
            down_raw=torch.from_numpy(item["raw_down"]).unsqueeze(0),
            right_log_assignment=torch.from_numpy(item["right_assignment"]).unsqueeze(0),
            down_log_assignment=torch.from_numpy(item["down_assignment"]).unsqueeze(0),
            right_out_border_logits=torch.from_numpy(item["right_out_border"]).unsqueeze(0),
            left_in_border_logits=torch.from_numpy(item["left_in_border"]).unsqueeze(0),
            bottom_out_border_logits=torch.from_numpy(item["bottom_out_border"]).unsqueeze(0),
            top_in_border_logits=torch.from_numpy(item["top_in_border"]).unsqueeze(0),
        )
        local = socket_retrieval_metrics(
            socket_output,
            torch.from_numpy(recovered.dirty_at_position).unsqueeze(0),
            grid=GRID,
        )
        local.update(
            directional_retrieval_metrics(
                item["baseline_right"],
                item["baseline_down"],
                recovered,
                prefix="bilateral",
            )
        )
        variants: dict[str, Any] = {}
        for name, layout in item["variants"].items():
            geometry = evaluate_layout(
                layout,
                recovered.dirty_at_position,
                reference_is_exact=False,
            ).as_dict()
            prediction = assemble_tiles(item["dirty"][layout])
            variants[name] = geometry | {"raw_ssim": contest_ssim(clean_image, prediction)}
        boards.append(
            {
                "filename": item["filename"],
                "local": local,
                "variants": variants,
                "decoder_reports": item["decoder_reports"],
            }
        )

    local_mean = _mean_numeric([board["local"] for board in boards])
    variant_names = list(boards[0]["variants"])
    variants_mean = {
        name: _mean_numeric([board["variants"][name] for board in boards]) for name in variant_names
    }
    baseline = variants_mean["bilateral_buddies96"]
    deltas = {
        name: {key: value - baseline[key] for key, value in metrics.items() if key in baseline}
        for name, metrics in variants_mean.items()
        if name != "bilateral_buddies96"
    }
    return {
        "reference": "target-assisted recovered permutation; not organizer ground truth",
        "boards": boards,
        "local_mean": local_mean,
        "variants_mean": variants_mean,
        "deltas_vs_bilateral_buddies96": deltas,
    }


def main() -> None:
    args = parse_args()
    integer_positive = (
        args.train_limit,
        args.eval_limit,
        args.synthetic_grid,
        args.dimension,
        args.heads,
        args.board_layers,
        args.socket_layers,
        args.sinkhorn_iterations,
        args.log_every,
    )
    if min(integer_positive) <= 0 or min(args.synthetic_steps, args.real_steps) < 0:
        raise ValueError("limits, dimensions and step counts must be non-negative/positive")
    if args.eval_offset < args.train_limit:
        raise ValueError("eval-offset must be >= train-limit for disjoint panels")
    if not 2 <= args.synthetic_grid <= GRID:
        raise ValueError("synthetic-grid must be in [2, 24]")
    if args.dimension % args.heads:
        raise ValueError("dimension must be divisible by heads")
    if not np.isfinite(args.border_weight) or args.border_weight < 0:
        raise ValueError("border-weight must be finite and non-negative")
    if not np.isfinite(args.raw_rank_weight) or args.raw_rank_weight < 0:
        raise ValueError("raw-rank-weight must be finite and non-negative")
    if not np.isfinite(args.real_loss_weight) or not 0 < args.real_loss_weight <= 1:
        raise ValueError("real-loss-weight must be in (0, 1]")
    if not np.isfinite(args.component_prior_weight) or args.component_prior_weight < 0:
        raise ValueError("component-prior-weight must be finite and non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records, eval_records = _select_records(manifest, args)
    model, checkpoint_payload = load_or_create_model(args, device)
    total_steps = args.synthetic_steps + args.real_steps
    current_train_names = (
        {str(record["filename"]) for record in train_records} if total_steps else set()
    )
    training_lineage_names = set(current_train_names)
    exposed_lineage_names = set(current_train_names)
    if checkpoint_payload is not None:
        old_selection = checkpoint_payload.get("selection", {})
        old_train_names = old_selection.get(
            "lineage_train_filenames", old_selection.get("train_filenames", [])
        )
        if not isinstance(old_train_names, list) or not all(
            isinstance(name, str) for name in old_train_names
        ):
            raise ValueError("checkpoint training lineage is malformed")
        training_lineage_names.update(old_train_names)
        old_exposed_names = old_selection.get(
            "lineage_exposed_filenames", old_train_names
        )
        if not isinstance(old_exposed_names, list) or not all(
            isinstance(name, str) for name in old_exposed_names
        ):
            raise ValueError("checkpoint source-exposure lineage is malformed")
        exposed_lineage_names.update(old_exposed_names)
        # Older checkpoints predate explicit exposure lineage.  Recover their
        # evaluation panel from the sibling report when available so a reused
        # target-opened panel cannot silently become confirmatory evidence.
        if "lineage_exposed_filenames" not in old_selection:
            source_path = args.checkpoint_in or args.warmstart_in
            assert source_path is not None
            source_report = source_path.parent / "report.json"
            if source_report.exists():
                previous = json.loads(source_report.read_text(encoding="utf-8"))
                previous_eval = previous.get("selection", {}).get("eval_filenames", [])
                if not isinstance(previous_eval, list) or not all(
                    isinstance(name, str) for name in previous_eval
                ):
                    raise ValueError("checkpoint sibling report evaluation lineage is malformed")
                exposed_lineage_names.update(previous_eval)
    eval_names = {str(record["filename"]) for record in eval_records}
    if exposed_lineage_names & eval_names:
        raise ValueError("evaluation sources overlap current or ancestral source exposure")
    if total_steps:
        boards, preparation_seconds = prepare_training_boards(train_records, args)
        history, training_seconds = train_model(model, boards, args, device)
    else:
        history, training_seconds, preparation_seconds = [], 0.0, 0.0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "socket_matcher.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "contract": _socket_contract(args),
        "training_history": history,
        "continued_from": (
            str(args.checkpoint_in or args.warmstart_in)
            if args.checkpoint_in or args.warmstart_in
            else None
        ),
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": names_digest(train_records),
            "lineage_train_filenames": sorted(training_lineage_names),
            "lineage_train_digest": hashlib.sha256(
                "\n".join(sorted(training_lineage_names)).encode()
            ).hexdigest(),
            "lineage_exposed_filenames": sorted(exposed_lineage_names | eval_names),
            "lineage_exposed_digest": hashlib.sha256(
                "\n".join(sorted(exposed_lineage_names | eval_names)).encode()
            ).hexdigest(),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    frozen, inference_seconds = freeze_predictions(model, eval_records, args, device)
    target_access_started = perf_counter()
    evaluation = evaluate_frozen(frozen, args)
    evaluation_seconds = perf_counter() - target_access_started
    report = {
        "experiment": _socket_contract(args)["architecture"],
        "status": "research-candidate-not-production",
        "hypothesis": (
            "whole-board contextual socket matching plus exact partial-OT cardinality "
            "improves exact neighbours and global tile coordinates"
        ),
        "contract": _socket_contract(args),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"device_resolved": str(device)},
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "selection_namespace": SELECTION_NAMESPACE,
            "train_split_only": True,
            "train_eval_source_disjoint": True,
            "checkpoint_exposure_lineage_eval_source_disjoint": True,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "all_dirty_only_predictions_frozen_before_eval_target_access": True,
            "real_labels": "target-assisted, top-half margin mask",
            "synthetic_labels": "exact known permutation",
        },
        "selection": {
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": names_digest(train_records),
            "eval_filenames": [record["filename"] for record in eval_records],
            "eval_digest": names_digest(eval_records),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "continued_checkpoint_present": checkpoint_payload is not None,
            "warmstarted": args.warmstart_in is not None,
            "warmstarted_from_architecture": (
                checkpoint_payload.get("contract", {}).get("architecture")
                if args.warmstart_in is not None and checkpoint_payload is not None
                else None
            ),
        },
        "decoder": {
            "name": "socket-translation-components-qap-v1",
            "component_edge_budget_per_axis": 144,
            "swap_edge_budget_per_axis": 144,
            "max_swap_steps": 24,
            "border_weight": SocketDecoderConfig().border_weight,
            "component_prior": "texture-centrality-v1",
            "component_prior_weight": args.component_prior_weight,
            "component_prior_enabled": args.component_prior_weight > 0,
        },
        "runtime_seconds": {
            "training_board_preparation": preparation_seconds,
            "training": training_seconds,
            "inference_freeze": inference_seconds,
            "target_assisted_evaluation": evaluation_seconds,
        },
        "training_history": history,
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "checkpoint": str(checkpoint_path),
                "local": evaluation["local_mean"],
                "variants": evaluation["variants_mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
