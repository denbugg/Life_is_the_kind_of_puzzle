#!/usr/bin/env python3
"""Preregister, train and locally gate direct d64 hard-edge priorities.

The candidate set is the frozen 552-edge-per-axis Socket hard projection.  The
small head sees dirty-visible evidence only and is trained to order exact true
edges ahead of false edges.  D1 is a source-disjoint organizer-train panel; the
competition test and every holdout/calibration split are outside this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.direct_hard_edge_priority import (
    DirectHardEdgePriority,
    fixed_budget_metrics,
    hard_edge_listwise_loss,
    learned_priority_matrices,
    prepare_direct_hard_edge_board,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_confidence_calibration import (
    exact_edge_labels,
    extract_hard_edge_features,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case, names_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
SELECTION_NAMESPACE = "aiijc-direct-hard-edge-board-priority-v1"
GRID = 24
TILE_COUNT = GRID * GRID
HARD_EDGES_PER_BOARD = 2 * GRID * (GRID - 1)
EDGE_BUDGET_PER_AXIS = 144
DEFAULT_FIT_SOURCES = 256
DEFAULT_D1_SOURCES = 32
DEFAULT_STEPS = 600
DEFAULT_HIDDEN_DIMENSION = 64
EXPECTED_INPUT_DIMENSION = 296
EXPECTED_PARAMETERS = 47_057
MAX_FIT_SOURCES = 512
MAX_STEPS = 800
GATE_CORRECT_GAIN_PER_BOARD = 1.0
GATE_PRECISION_GAIN = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-preregistered-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--exclude-artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--panel-artifact",
        type=Path,
        action="append",
        default=[],
        help=(
            "panel report whose actual filename rosters are excluded while broad "
            "forbidden/available registries are treated as metadata"
        ),
    )
    parser.add_argument("--fit-sources", type=int, default=DEFAULT_FIT_SOURCES)
    parser.add_argument("--d1-sources", type=int, default=DEFAULT_D1_SOURCES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--hidden-dimension", type=int, default=DEFAULT_HIDDEN_DIMENSION)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--capacity-smoke", action="store_true")
    parser.add_argument("--capacity-steps", type=int, default=120)
    parser.add_argument("--benchmark-one-update", action="store_true")
    return parser.parse_args()


def _validate_hyperparameters(args: argparse.Namespace) -> None:
    if not 1 <= args.fit_sources <= MAX_FIT_SOURCES:
        raise ValueError(f"fit-sources must be in [1, {MAX_FIT_SOURCES}]")
    if not 1 <= args.d1_sources <= 64:
        raise ValueError("d1-sources must be in [1, 64]")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if args.hidden_dimension < 2 or args.log_every <= 0:
        raise ValueError("hidden-dimension >=2 and positive log-every are required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and non-negative")
    if args.allow_nondeterministic_mps and args.device != "mps":
        raise ValueError("allow-nondeterministic-mps requires --device mps")


def _collect_filename_lists(value: Any, *, parent_key: str = "") -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.endswith("_filenames"):
                if not isinstance(child, (list, tuple)) or not all(
                    isinstance(item, str) and item for item in child
                ):
                    raise ValueError(f"{key} must be a list of non-empty filenames")
                if len(set(child)) != len(child):
                    raise ValueError(f"{key} contains duplicate filenames")
                names.update(Path(item).name for item in child)
            names.update(_collect_filename_lists(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and not parent_key.endswith("_filenames"):
        for child in value:
            names.update(_collect_filename_lists(child, parent_key=parent_key))
    return names


def _collect_actual_roster_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect actual singular/plural PNG rosters, not broad exclusion metadata."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            metadata = any(
                marker in lowered
                for marker in ("excluded", "forbidden", "remaining", "available")
            )
            if "filename" in lowered and not metadata:
                if isinstance(child, str) and child.lower().endswith(".png"):
                    names.add(Path(child).name)
                elif isinstance(child, (list, tuple)):
                    values = [
                        Path(item).name
                        for item in child
                        if isinstance(item, str) and item.lower().endswith(".png")
                    ]
                    if len(values) != len(set(values)):
                        raise ValueError(f"{key} contains duplicate PNG filenames")
                    names.update(values)
            names.update(_collect_actual_roster_filenames(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and "filename" not in parent_key.lower():
        for child in value:
            names.update(_collect_actual_roster_filenames(child, parent_key=parent_key))
    return names


def _load_json_or_checkpoint(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=False)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"exclusion artifact is not a mapping: {path}")
    return value


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _selection_records(
    manifest: Mapping[str, Any],
    socket_payload: Mapping[str, Any],
    artifacts: Sequence[Path],
    panel_artifacts: Sequence[Path],
    *,
    socket_sha256: str,
    fit_sources: int,
    d1_sources: int,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], dict[str, Any]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("manifest protocol digest is invalid")
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, Mapping) else None
    if not isinstance(train, list):
        raise ValueError("manifest has no train split")
    forbidden = _collect_filename_lists(socket_payload)
    registry: list[dict[str, Any]] = [
        {
            "path": _relative(DEFAULT_SOCKET_CHECKPOINT),
            "sha256": socket_sha256,
            "filename_count": len(forbidden),
            "role": "frozen-socket-complete-declared-lineage",
        }
    ]
    for path in sorted({item.resolve() for item in artifacts}):
        payload = _load_json_or_checkpoint(path)
        names = _collect_filename_lists(payload)
        if not names:
            raise ValueError(f"exclusion artifact contains no *_filenames: {path}")
        forbidden.update(names)
        registry.append(
            {
                "path": _relative(path),
                "sha256": sha256_file(path),
                "filename_count": len(names),
                "filename_digest": names_digest(sorted(names)),
                "role": "declared-panel-or-model-lineage-exclusion",
            }
        )
    for path in sorted({item.resolve() for item in panel_artifacts}):
        payload = _load_json_or_checkpoint(path)
        names = _collect_actual_roster_filenames(payload)
        if not names:
            raise ValueError(f"panel artifact contains no actual filename roster: {path}")
        forbidden.update(names)
        registry.append(
            {
                "path": _relative(path),
                "sha256": sha256_file(path),
                "filename_count": len(names),
                "filename_digest": names_digest(sorted(names)),
                "role": "actual-panel-roster-exclusion; broad forbidden lists ignored",
            }
        )
    ranked = select_manifest_records(
        dict(manifest),
        "train",
        limit=len(train),
        namespace=f"{SELECTION_NAMESPACE}\0{socket_sha256}",
    )
    selected = tuple(
        record
        for record in ranked
        if Path(str(record["filename"])).name not in forbidden
    )[: fit_sources + d1_sources]
    if len(selected) != fit_sources + d1_sources:
        raise ValueError("not enough source-disjoint train records remain")
    fit = selected[:fit_sources]
    d1 = selected[fit_sources:]
    fit_names = {str(record["filename"]) for record in fit}
    d1_names = {str(record["filename"]) for record in d1}
    if fit_names & d1_names or (fit_names | d1_names) & forbidden:
        raise RuntimeError("selection disjointness invariant failed")
    return fit, d1, {
        "excluded_filename_count": len(forbidden),
        "excluded_filename_digest": names_digest(sorted(forbidden)),
        "registry": registry,
    }


def write_preregistered_config(args: argparse.Namespace) -> None:
    if args.config is not None or args.output_dir is not None:
        raise ValueError("config writing does not accept --config/--output-dir")
    assert args.write_preregistered_config is not None
    path = args.write_preregistered_config.resolve()
    if path.exists() or path.with_name(f"{path.name}.sha256").exists():
        raise FileExistsError("refusing to overwrite a preregistered config")
    socket_path = args.socket_checkpoint.resolve()
    socket_sha = sha256_file(socket_path)
    socket_payload = _load_json_or_checkpoint(socket_path)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fit, d1, exclusion = _selection_records(
        manifest,
        socket_payload,
        args.exclude_artifact,
        args.panel_artifact,
        socket_sha256=socket_sha,
        fit_sources=args.fit_sources,
        d1_sources=args.d1_sources,
    )
    fit_names = [str(record["filename"]) for record in fit]
    d1_names = [str(record["filename"]) for record in d1]
    payload = {
        "schema": "aiijc-direct-hard-edge-board-priority-preregistered-v1",
        "registered_before_d1_target_access": True,
        "registered_before_selected_dirty_generation": True,
        "competition_test_opened": False,
        "frozen_inputs": {
            "socket_checkpoint": _relative(socket_path),
            "socket_checkpoint_sha256": socket_sha,
            "socket_dimension": 64,
            "hard_edges_per_axis": GRID * (GRID - 1),
            "component_edge_budget_per_axis": EDGE_BUDGET_PER_AXIS,
            "candidate_supply": "raw hard projection only; no restored-only edges",
        },
        "model": {
            "architecture": "direct-hard-edge-deepsets-residual-v1",
            "input_dimension": EXPECTED_INPUT_DIMENSION,
            "hidden_dimension": args.hidden_dimension,
            "whole_board_pool": ["mean", "max"],
            "per_axis_pool": ["mean", "max"],
            "raw_priority_zero_init_residual": True,
            "provisional_raw_edge_budget_per_axis": 48,
        },
        "training": {
            "fit_sources": args.fit_sources,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "pairwise_loss_weight": 0.75,
            "synthetic_seed": args.seed,
        },
        "selection": {
            "namespace": f"{SELECTION_NAMESPACE}\0{socket_sha}",
            "split": "train",
            "fit_source_filenames": fit_names,
            "fit_source_order_digest": names_digest(fit_names),
            "d1_source_filenames": d1_names,
            "d1_source_order_digest": names_digest(d1_names),
            "d1_draw_indices": [0],
            "d1_source_count": len(d1_names),
            "exclusion": exclusion,
        },
        "d1_gate": {
            "primary": "fixed top144/axis existing hard-edge truth",
            "pass_if": "correct gain >=1/board OR precision gain >=0.01",
            "minimum_correct_gain_per_board": GATE_CORRECT_GAIN_PER_BOARD,
            "minimum_precision_gain": GATE_PRECISION_GAIN,
            "decoder_if_pass": "same opened d1; decoder144 plus cyclic-border5 descriptive",
            "promotion_authorized": False,
        },
        "legality": {
            "output": "strict np permutation of original upright input tiles",
            "competition_test_forbidden": True,
            "restored_candidate_substitution": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "preregistered",
                "path": str(path),
                "sha256": digest,
                "fit_sources": len(fit_names),
                "fit_digest": names_digest(fit_names),
                "d1_sources": len(d1_names),
                "d1_digest": names_digest(d1_names),
                "excluded": exclusion["excluded_filename_count"],
            }
        ),
        flush=True,
    )


def load_frozen_config(path: Path) -> tuple[dict[str, Any], str]:
    digest_path = path.with_name(f"{path.name}.sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("preregistered config SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aiijc-direct-hard-edge-board-priority-preregistered-v1":
        raise ValueError("unsupported direct hard-edge config schema")
    if not payload.get("registered_before_d1_target_access"):
        raise ValueError("config was not frozen before D1 target access")
    return payload, observed


class CleanTileCache:
    def __init__(self, targets: Path, *, maximum_boards: int = 32) -> None:
        self.targets = targets
        self.maximum_boards = maximum_boards
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, record: Mapping[str, Any]) -> np.ndarray:
        filename = str(record["filename"])
        if filename in self.values:
            value = self.values.pop(filename)
            self.values[filename] = value
            return value
        path = self.targets / filename
        expected = record.get("target_sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"manifest target hash mismatch: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"expected RGB 480x480 target: {path}")
            value = split_tiles(np.asarray(image, dtype=np.uint8)).copy()
        self.values[filename] = value
        while len(self.values) > self.maximum_boards:
            self.values.popitem(last=False)
        return value


def _record_lookup(
    manifest: Mapping[str, Any],
    names: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, Mapping) else None
    if not isinstance(train, list):
        raise ValueError("manifest has no train split")
    by_name = {str(record["filename"]): record for record in train}
    if len(by_name) != len(train):
        raise ValueError("manifest train filenames are not unique")
    try:
        records = tuple(by_name[name] for name in names)
    except KeyError as error:
        raise ValueError(f"selected filename is absent from train split: {error}") from error
    return records


def _make_case(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    *,
    draw_index: int,
    seed: int,
) -> tuple[Any, Any]:
    return make_exact_synthetic_case(
        cache.load(record),
        source_filename=str(record["filename"]),
        draw_index=draw_index,
        seed=seed,
    )


def _tile_tensor(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    if tiles.shape != (TILE_COUNT, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("synthetic dirty tiles violate the strict input contract")
    return (
        torch.from_numpy(np.ascontiguousarray(tiles))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def _forward_board(
    socket: LoadedSocketCheckpoint,
    head: DirectHardEdgePriority,
    dirty_tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[Any, Any, torch.Tensor, Any]:
    from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context

    tiles = _tile_tensor(dirty_tiles, device=device)
    with torch.no_grad():
        tokens, output = extract_frozen_socket_context(socket.model, tiles, grid=GRID)
    features = extract_hard_edge_features(
        right_log_assignment=output.right_log_assignment[0],
        down_log_assignment=output.down_log_assignment[0],
        right_raw=output.right_raw[0],
        down_raw=output.down_raw[0],
        grid=GRID,
    )
    board = prepare_direct_hard_edge_board(tokens[0].detach(), features, output, grid=GRID)
    scores = head(board.values, board.raw_priority, board.axis)
    return board, features, scores, output


def run_capacity_smoke(args: argparse.Namespace) -> None:
    if args.capacity_steps <= 0:
        raise ValueError("capacity-steps must be positive")
    torch.manual_seed(args.seed)
    device = choose_deterministic_device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    edges_per_axis = 128
    count = 2 * edges_per_axis
    values = torch.randn(count, EXPECTED_INPUT_DIMENSION, generator=generator).to(device)
    axis = torch.cat(
        (
            torch.zeros(edges_per_axis, dtype=torch.long),
            torch.ones(edges_per_axis, dtype=torch.long),
        )
    ).to(device)
    # A learnable nonlinear truth rule with the raw priority deliberately noisy.
    labels = ((values[:, 0] + 0.8 * values[:, 1] * values[:, 2]) > 0.8).bool()
    for axis_index in (0, 1):
        if int(labels[axis == axis_index].sum()) < 8:
            raise RuntimeError("capacity generator produced too few positives")
    raw = (0.05 * torch.randn(count, generator=generator)).to(device)
    model = DirectHardEdgePriority(
        EXPECTED_INPUT_DIMENSION,
        hidden_dimension=args.hidden_dimension,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    initial = fixed_budget_metrics(
        raw.cpu().numpy(),
        labels.cpu().numpy(),
        axis.cpu().numpy(),
        edge_budget_per_axis=32,
    )
    started = perf_counter()
    first_loss = None
    last_loss = None
    for _ in range(args.capacity_steps):
        score = model(values, raw, axis)
        loss, _ = hard_edge_listwise_loss(score, labels, axis)
        if first_loss is None:
            first_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    final_score = model(values, raw, axis).detach().cpu().numpy()
    final = fixed_budget_metrics(
        final_score,
        labels.cpu().numpy(),
        axis.cpu().numpy(),
        edge_budget_per_axis=32,
    )
    passed = (
        last_loss is not None
        and first_loss is not None
        and last_loss < 0.35 * first_loss
        and final["correct_selected_edges"] > initial["correct_selected_edges"] + 20
    )
    print(
        json.dumps(
            {
                "event": "capacity-smoke",
                "pass": passed,
                "device": str(device),
                "parameters": parameters,
                "input_dimension": EXPECTED_INPUT_DIMENSION,
                "steps": args.capacity_steps,
                "first_loss": first_loss,
                "last_loss": last_loss,
                "initial": initial,
                "final": final,
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )
    if not passed:
        raise RuntimeError("capacity smoke failed")


def _dirty_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("correct_selected_edges", "selected_edge_precision")
    }


def evaluate_d1_gate(
    raw: Mapping[str, float],
    learned: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the frozen low D1 OR gate without authorizing promotion."""

    correct_gain = float(learned["correct_selected_edges"]) - float(
        raw["correct_selected_edges"]
    )
    precision_gain = float(learned["selected_edge_precision"]) - float(
        raw["selected_edge_precision"]
    )
    passed = (
        correct_gain >= GATE_CORRECT_GAIN_PER_BOARD
        or precision_gain >= GATE_PRECISION_GAIN
    )
    return {
        "pass": passed,
        "status": "pass-same-panel-decoder-authorized" if passed else "stop",
        "correct_gain_per_board": correct_gain,
        "required_correct_gain_per_board": GATE_CORRECT_GAIN_PER_BOARD,
        "precision_gain": precision_gain,
        "required_precision_gain": GATE_PRECISION_GAIN,
        "logical_rule": "correct gain OR precision gain",
        "promotion_authorized": False,
    }


def _decoder_panel(
    records: Sequence[Mapping[str, Any]],
    cache: CleanTileCache,
    socket: LoadedSocketCheckpoint,
    head: DirectHardEdgePriority,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
        max_swap_steps=24,
    )
    cyclic_config = CyclicTranslationConfig(border_weight=5.0)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        dirty, reference = _make_case(cache, record, draw_index=0, seed=seed + 10_000)
        board, _, scores, output = _forward_board(
            socket,
            head,
            dirty.tiles,
            device=device,
        )
        priorities = learned_priority_matrices(board, scores, grid=GRID)
        raw = decode_socket_assignments(
            output.right_log_assignment,
            output.down_log_assignment,
            grid=GRID,
            config=decoder_config,
        )
        learned = decode_socket_assignments(
            output.right_log_assignment,
            output.down_log_assignment,
            grid=GRID,
            config=decoder_config,
            component_edge_priority=priorities,
        )
        raw_cyclic = select_global_cyclic_translation(
            raw.layout,
            output.right_log_assignment,
            output.down_log_assignment,
            grid=GRID,
            config=cyclic_config,
        )
        learned_cyclic = select_global_cyclic_translation(
            learned.layout,
            output.right_log_assignment,
            output.down_log_assignment,
            grid=GRID,
            config=cyclic_config,
        )
        for layout in (raw_cyclic.layout, learned_cyclic.layout):
            if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT)):
                raise RuntimeError("decoder output is not a strict original-tile permutation")
        raw_metrics = evaluate_layout(
            raw_cyclic.layout,
            reference.tile_at_position,
            reference_is_exact=True,
        )
        learned_metrics = evaluate_layout(
            learned_cyclic.layout,
            reference.tile_at_position,
            reference_is_exact=True,
        )
        rows.append(
            {
                "source_filename": str(record["filename"]),
                "raw_exact_tiles": raw_metrics.correct_tile_count,
                "learned_exact_tiles": learned_metrics.correct_tile_count,
                "raw_adjacency": raw_metrics.adjacency,
                "learned_adjacency": learned_metrics.adjacency,
                "raw_translation_aligned_tiles": raw_metrics.translation_aligned_count,
                "learned_translation_aligned_tiles": learned_metrics.translation_aligned_count,
            }
        )
        print(
            json.dumps(
                {"event": "same-panel-decoder", "done": index + 1, "total": len(records)}
            ),
            flush=True,
        )
    fields = (
        "exact_tiles",
        "adjacency",
        "translation_aligned_tiles",
    )
    aggregate: dict[str, Any] = {}
    for field in fields:
        raw_values = np.asarray([row[f"raw_{field}"] for row in rows], dtype=np.float64)
        learned_values = np.asarray(
            [row[f"learned_{field}"] for row in rows], dtype=np.float64
        )
        aggregate[field] = {
            "raw_mean": float(raw_values.mean()),
            "learned_mean": float(learned_values.mean()),
            "mean_delta": float((learned_values - raw_values).mean()),
        }
    return {
        "status": "development-after-open-d1-only",
        "cyclic_border_weight": 5.0,
        "rows": rows,
        "aggregate": aggregate,
    }


def run_experiment(args: argparse.Namespace) -> None:
    if args.config is None:
        raise ValueError("experiment/benchmark requires --config")
    config, config_sha = load_frozen_config(args.config)
    frozen = config["frozen_inputs"]
    socket_path = _resolve(str(frozen["socket_checkpoint"]))
    if sha256_file(socket_path) != frozen["socket_checkpoint_sha256"]:
        raise ValueError("Socket checkpoint hash changed after preregistration")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selection = config["selection"]
    fit_names = list(selection["fit_source_filenames"])
    d1_names = list(selection["d1_source_filenames"])
    if names_digest(fit_names) != selection["fit_source_order_digest"]:
        raise ValueError("fit roster digest mismatch")
    if names_digest(d1_names) != selection["d1_source_order_digest"]:
        raise ValueError("D1 roster digest mismatch")
    fit_records = _record_lookup(manifest, fit_names)
    d1_records = _record_lookup(manifest, d1_names)
    training = config["training"]
    if len(fit_records) != int(training["fit_sources"]):
        raise ValueError("fit roster count differs from frozen config")
    seed = int(training["synthetic_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "mps" and args.allow_nondeterministic_mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        device = torch.device("mps")
    else:
        device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": str(device),
                "config_sha256": config_sha,
                "benchmark_only": args.benchmark_one_update,
            }
        ),
        flush=True,
    )
    socket = load_socket_checkpoint(socket_path, device=device)
    if int(socket.contract["dimension"]) != 64:
        raise ValueError("this branch requires the frozen d64 Socket model")
    head = DirectHardEdgePriority(
        int(config["model"]["input_dimension"]),
        hidden_dimension=int(config["model"]["hidden_dimension"]),
    ).to(device)
    parameters = sum(parameter.numel() for parameter in head.parameters())
    if (
        int(config["model"]["hidden_dimension"]) == DEFAULT_HIDDEN_DIMENSION
        and parameters != EXPECTED_PARAMETERS
    ):
        raise RuntimeError("default model parameter count changed")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    cache = CleanTileCache(args.targets)
    generator = np.random.default_rng(seed + 1)
    requested_steps = 1 if args.benchmark_one_update else int(training["steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        requested_steps,
        eta_min=float(training["learning_rate"]) * 0.08,
    )
    history: list[dict[str, Any]] = []
    started = perf_counter()
    head.train()
    for step in range(requested_steps):
        record = fit_records[int(generator.integers(len(fit_records)))]
        dirty, reference = _make_case(cache, record, draw_index=step, seed=seed)
        board, features, scores, _ = _forward_board(
            socket,
            head,
            dirty.tiles,
            device=device,
        )
        labels_np = exact_edge_labels(features, reference.tile_at_position, grid=GRID)
        labels = torch.as_tensor(labels_np, device=device, dtype=torch.bool)
        loss, diagnostics = hard_edge_listwise_loss(scores, labels, board.axis)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == requested_steps:
            learned = fixed_budget_metrics(
                scores.detach().cpu().numpy(),
                labels_np,
                features.axis,
                edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
            )
            raw = fixed_budget_metrics(
                features.values[:, 0],
                labels_np,
                features.axis,
                edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
            )
            row = {
                "step": step + 1,
                **diagnostics,
                "gradient_norm": gradient_norm,
                "learned_correct": learned["correct_selected_edges"],
                "raw_correct": raw["correct_selected_edges"],
                "elapsed_seconds": perf_counter() - started,
            }
            history.append(row)
            print(json.dumps({"event": "train", **row}), flush=True)
    training_seconds = perf_counter() - started
    if args.benchmark_one_update:
        print(
            json.dumps(
                {
                    "event": "one-real-update-benchmark",
                    "device": str(device),
                    "seconds": training_seconds,
                    "parameters": parameters,
                    "input_dimension": EXPECTED_INPUT_DIMENSION,
                }
            ),
            flush=True,
        )
        return
    if args.output_dir is None:
        raise ValueError("full experiment requires --output-dir")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    checkpoint_path = output_dir / "direct_hard_edge_priority.pt"
    frozen_path = output_dir / "d1_dirty_predictions.npz"
    if any(path.exists() for path in (report_path, checkpoint_path, frozen_path)):
        raise FileExistsError("refusing to overwrite a direct hard-edge artifact")

    # Freeze target-free D1 score lists before exact edge labels are inspected.
    head.eval()
    frozen_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, record in enumerate(d1_records):
            dirty, _ = _make_case(cache, record, draw_index=0, seed=seed + 10_000)
            board, features, scores, _ = _forward_board(
                socket,
                head,
                dirty.tiles,
                device=device,
            )
            frozen_rows.append(
                {
                    "source_filename": str(record["filename"]),
                    "case_id": dirty.case_id,
                    "dirty_sha256": _dirty_sha256(dirty.tiles),
                    "raw": features.values[:, 0].copy(),
                    "learned": scores.detach().cpu().numpy().copy(),
                    "source": features.source.copy(),
                    "target": features.target.copy(),
                    "axis": features.axis.copy(),
                }
            )
            print(
                json.dumps(
                    {"event": "freeze-d1", "done": index + 1, "total": len(d1_records)}
                ),
                flush=True,
            )
    np.savez_compressed(
        frozen_path,
        source_filenames=np.asarray(
            [row["source_filename"] for row in frozen_rows], dtype="U64"
        ),
        case_ids=np.asarray([row["case_id"] for row in frozen_rows], dtype="U160"),
        dirty_sha256=np.asarray([row["dirty_sha256"] for row in frozen_rows], dtype="U64"),
        raw_scores=np.stack([row["raw"] for row in frozen_rows]),
        learned_scores=np.stack([row["learned"] for row in frozen_rows]),
        source=np.stack([row["source"] for row in frozen_rows]),
        target=np.stack([row["target"] for row in frozen_rows]),
        axis=np.stack([row["axis"] for row in frozen_rows]),
    )
    frozen_sha = sha256_file(frozen_path)
    print(
        json.dumps(
            {
                "event": "d1-predictions-frozen",
                "path": str(frozen_path),
                "sha256": frozen_sha,
            }
        ),
        flush=True,
    )

    raw_metrics: list[dict[str, Any]] = []
    learned_metrics: list[dict[str, Any]] = []
    for index, (record, frozen_row) in enumerate(zip(d1_records, frozen_rows, strict=True)):
        dirty, reference = _make_case(cache, record, draw_index=0, seed=seed + 10_000)
        if dirty.case_id != frozen_row["case_id"] or _dirty_sha256(dirty.tiles) != frozen_row[
            "dirty_sha256"
        ]:
            raise RuntimeError("D1 regeneration differs from frozen prediction input")
        from aiijc_puzzle.socket_confidence_calibration import HardEdgeFeatures

        proxy = HardEdgeFeatures(
            values=np.zeros((HARD_EDGES_PER_BOARD, 20), dtype=np.float32),
            source=frozen_row["source"],
            target=frozen_row["target"],
            axis=frozen_row["axis"],
        )
        labels = exact_edge_labels(proxy, reference.tile_at_position, grid=GRID)
        raw_metrics.append(
            fixed_budget_metrics(
                frozen_row["raw"],
                labels,
                frozen_row["axis"],
                edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
            )
        )
        learned_metrics.append(
            fixed_budget_metrics(
                frozen_row["learned"],
                labels,
                frozen_row["axis"],
                edge_budget_per_axis=EDGE_BUDGET_PER_AXIS,
            )
        )
        print(
            json.dumps({"event": "score-d1", "done": index + 1, "total": len(d1_records)}),
            flush=True,
        )
    raw_aggregate = _mean_metrics(raw_metrics)
    learned_aggregate = _mean_metrics(learned_metrics)
    gate = evaluate_d1_gate(raw_aggregate, learned_aggregate)
    gate_pass = bool(gate["pass"])
    decoder = (
        _decoder_panel(
            d1_records,
            cache,
            socket,
            head,
            seed=seed,
            device=device,
        )
        if gate_pass
        else {"status": "closed-by-d1-gate"}
    )
    torch.save(
        {
            "schema": "aiijc-direct-hard-edge-board-priority-checkpoint-v1",
            "state_dict": head.state_dict(),
            "contract": config["model"],
            "config_path": str(args.config.resolve()),
            "config_sha256": config_sha,
            "socket_checkpoint": {
                "path": _relative(socket_path),
                "sha256": socket.sha256,
            },
            "selection": config["selection"],
            "competition_test_opened": False,
        },
        checkpoint_path,
    )
    report = {
        "schema": "aiijc-direct-hard-edge-board-priority-report-v1",
        "experiment": "direct frozen-d64 hard-edge truth/listwise priority",
        "config": str(args.config.resolve()),
        "config_sha256": config_sha,
        "architecture": {
            **config["model"],
            "trainable_parameters": parameters,
        },
        "selection": config["selection"],
        "training": {
            **config["training"],
            "device": str(device),
            "seconds": training_seconds,
            "seconds_per_step": training_seconds / requested_steps,
            "history": history,
        },
        "d1": {
            "predictions_frozen_before_reference_scoring": True,
            "frozen_predictions": str(frozen_path),
            "frozen_predictions_sha256": frozen_sha,
            "raw": raw_aggregate,
            "learned": learned_aggregate,
            "raw_rows": raw_metrics,
            "learned_rows": learned_metrics,
            "gate": gate,
            "same_panel_decoder144_cyclic5": decoder,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "legality": config["legality"],
        "competition_test_opened": False,
        "promotion_authorized": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "gate": gate,
                "training_seconds_per_step": training_seconds / requested_steps,
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _validate_hyperparameters(args)
    modes = sum(
        (
            bool(args.capacity_smoke),
            args.write_preregistered_config is not None,
            args.config is not None,
        )
    )
    if modes != 1:
        raise ValueError("choose exactly one of capacity smoke, config writing, or config run")
    if args.capacity_smoke:
        run_capacity_smoke(args)
    elif args.write_preregistered_config is not None:
        write_preregistered_config(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
