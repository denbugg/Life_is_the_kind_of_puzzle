#!/usr/bin/env python3
"""Train a source-disjoint edge-conditioned SocketMatcher permutation refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SocketMatcher,
)
from aiijc_puzzle.socket_permutation_flow import (
    FrozenSocketEvidence,
    SocketPermutationFlow,
    SocketTopKGraph,
    extract_frozen_socket_evidence,
    interpolate_permutations,
    iterative_refine_layout,
    permutation_flow_loss,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    load_checkpoint_with_lineage,
    names_digest,
    select_source_disjoint_train_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_MATCHER = (
    PROJECT_ROOT
    / "outputs"
    / "socket-matcher"
    / "v2-d64-train1024-s1600-r400-dev32"
    / "socket_matcher.pt"
)


@dataclass(frozen=True)
class FlowBoard:
    filename: str
    crop_row: int
    crop_column: int
    corruption_seed: int
    evidence: FrozenSocketEvidence
    decoder_layout: torch.Tensor
    target_layout: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("sanity", "pilot"), required=True)
    parser.add_argument("--matcher-checkpoint", type=Path, default=DEFAULT_MATCHER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--prior-output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--train-sources", type=int)
    parser.add_argument("--eval-sources", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--coordinate-bands", type=int, default=4)
    parser.add_argument("--time-bands", type=int, default=4)
    parser.add_argument("--sinkhorn-iterations", type=int, default=8)
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--random-start-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def _resolved_stage_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "sanity": {"grid": 4, "train_sources": 1, "eval_sources": 1, "steps": 400},
        "pilot": {"grid": 24, "train_sources": 16, "eval_sources": 4, "steps": 300},
    }[args.stage]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _filename_lists(value: Any, key: str = "") -> list[list[str]]:
    result: list[list[str]] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            result.extend(_filename_lists(child, str(child_key)))
    elif key.endswith("filenames") and isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        result.append(value)
    return result


def _known_exposed_sources(root: Path) -> tuple[set[str], int]:
    names: set[str] = set()
    reports = 0
    if root.exists():
        for path in sorted(root.glob("**/report.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for filename_list in _filename_lists(payload):
                names.update(filename_list)
            reports += 1
    return names, reports


def _load_matcher(
    checkpoint_path: Path, device: torch.device
) -> tuple[SocketMatcher, dict[str, Any], Any]:
    payload, lineage = load_checkpoint_with_lineage(
        checkpoint_path,
        project_root=PROJECT_ROOT,
    )
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("matcher checkpoint has no contract")
    architecture = contract.get("architecture")
    version = {
        "board-conditioned-partial-socket-matcher-v2": BORDER_HEAD_EMBEDDING_V2,
        "board-conditioned-partial-socket-matcher-v3": BORDER_HEAD_SCORE_STATS_V3,
    }.get(architecture)
    if version is None:
        raise ValueError(f"unsupported matcher architecture: {architecture!r}")
    model = SocketMatcher(
        dimension=int(contract["dimension"]),
        heads=int(contract["heads"]),
        board_layers=int(contract["board_layers"]),
        socket_layers=int(contract["socket_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
        border_head_version=version,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, contract, lineage


def _torch_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(shape).uniform_(low, high, generator=generator)


def _challenge_augment(clean: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    count = len(clean)
    gray = 0.299 * clean[:, :1] + 0.587 * clean[:, 1:2] + 0.114 * clean[:, 2:3]
    pivot = gray.mean(dim=(1, 2, 3), keepdim=True)
    scale = _torch_uniform((count, 1, 1, 1), 0.70, 1.30, generator=generator)
    offset = _torch_uniform((count, 1, 1, 1), -30 / 255, 30 / 255, generator=generator)
    value = scale * (clean - pivot) + pivot + offset
    sigma = _torch_uniform((count, 1, 1, 1), 40 / 255, 55 / 255, generator=generator)
    value = value + sigma * torch.randn(value.shape, generator=generator)
    kernel = value.new_tensor([0.25, 0.5, 0.25])
    horizontal = kernel.reshape(1, 1, 1, 3).expand(3, 1, 1, 3)
    vertical = kernel.reshape(1, 1, 3, 1).expand(3, 1, 3, 1)
    value = torch.nn.functional.conv2d(
        torch.nn.functional.pad(value, (1, 1, 0, 0), mode="reflect"),
        horizontal,
        groups=3,
    )
    value = torch.nn.functional.conv2d(
        torch.nn.functional.pad(value, (0, 0, 1, 1), mode="reflect"),
        vertical,
        groups=3,
    )
    levels = _torch_uniform((count, 1, 1, 1), 40.0, 72.0, generator=generator)
    return (torch.round(value.clamp(0, 1) * levels) / levels).clamp(0.0, 1.0)


def _source_seed(seed: int, filename: str) -> int:
    digest = hashlib.sha256(f"{seed}:{filename}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _freeze_board(
    record: dict[str, Any],
    matcher: SocketMatcher,
    *,
    targets: Path,
    grid: int,
    top_k: int,
    seed: int,
    device: torch.device,
) -> FlowBoard:
    filename = str(record["filename"])
    target_path = targets / filename
    if sha256_file(target_path) != record.get("target_sha256"):
        raise ValueError(f"target hash mismatch: {filename}")
    clean_tiles = split_tiles(_load_rgb(target_path)).reshape(24, 24, 20, 20, 3)
    local_seed = _source_seed(seed, filename)
    numpy_generator = np.random.default_rng(local_seed)
    crop_row = int(numpy_generator.integers(0, 24 - grid + 1))
    crop_column = int(numpy_generator.integers(0, 24 - grid + 1))
    crop = clean_tiles[
        crop_row : crop_row + grid,
        crop_column : crop_column + grid,
    ].reshape(-1, 20, 20, 3)
    clean = torch.from_numpy(crop.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    torch_generator = torch.Generator().manual_seed(local_seed + 1)
    corrupted = _challenge_augment(clean, generator=torch_generator)
    permutation = numpy_generator.permutation(grid * grid)
    shuffled = corrupted[torch.from_numpy(permutation)]
    target_layout = torch.from_numpy(np.argsort(permutation).astype(np.int64)).unsqueeze(0)
    evidence = extract_frozen_socket_evidence(
        matcher,
        shuffled.unsqueeze(0).to(device),
        grid=grid,
        top_k=top_k,
    )
    edge_budget = min(144, grid * (grid - 1))
    decoder = decode_socket_assignments(
        evidence.right_log_assignment,
        evidence.down_log_assignment,
        grid=grid,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=edge_budget,
            swap_edge_budget_per_axis=edge_budget,
            max_swap_steps=min(24, grid * grid),
        ),
    )
    cpu_evidence = FrozenSocketEvidence(
        tile_features=evidence.tile_features.cpu(),
        graph=SocketTopKGraph(
            indices=evidence.graph.indices.cpu(),
            log_scores=evidence.graph.log_scores.cpu(),
        ),
        right_log_assignment=evidence.right_log_assignment.cpu(),
        down_log_assignment=evidence.down_log_assignment.cpu(),
    )
    return FlowBoard(
        filename=filename,
        crop_row=crop_row,
        crop_column=crop_column,
        corruption_seed=local_seed,
        evidence=cpu_evidence,
        decoder_layout=torch.from_numpy(decoder.layout.astype(np.int64)).unsqueeze(0),
        target_layout=target_layout,
    )


def _graph_to_device(graph: SocketTopKGraph, device: torch.device) -> SocketTopKGraph:
    return SocketTopKGraph(
        indices=graph.indices.to(device),
        log_scores=graph.log_scores.to(device),
    )


def _train(
    model: SocketPermutationFlow,
    boards: list[FlowBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.steps, 1), eta_min=args.learning_rate * 0.08
    )
    numpy_generator = np.random.default_rng(args.seed + 7)
    interpolation_generator = torch.Generator().manual_seed(args.seed + 8)
    history: list[dict[str, float]] = []
    started = perf_counter()
    for step in range(args.steps):
        board = boards[int(numpy_generator.integers(len(boards)))]
        target = board.target_layout
        if float(numpy_generator.random()) < args.random_start_probability:
            start = torch.from_numpy(
                numpy_generator.permutation(args.grid * args.grid).astype(np.int64)
            ).unsqueeze(0)
            start_kind = 1.0
        else:
            start = board.decoder_layout
            start_kind = 0.0
        progress = float(numpy_generator.uniform(0.0, 0.90))
        current = interpolate_permutations(
            start,
            target,
            progress,
            generator=interpolation_generator,
        ).to(device)
        model.train()
        output = model(
            board.evidence.tile_features.to(device),
            _graph_to_device(board.evidence.graph, device),
            current,
            progress,
            grid=args.grid,
        )
        loss, diagnostics = permutation_flow_loss(
            output,
            target.to(device),
            grid=args.grid,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        record = {
            "step": float(step + 1),
            "progress": progress,
            "random_start": start_kind,
            "gradient_norm": gradient_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        } | diagnostics
        history.append(record)
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            recent = history[-min(args.log_every, len(history)) :]
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step + 1,
                        "loss": float(np.mean([item["loss"] for item in recent])),
                        "assignment_nll": float(
                            np.mean([item["assignment_nll"] for item in recent])
                        ),
                        "elapsed_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


@torch.no_grad()
def _evaluate(
    model: SocketPermutationFlow,
    boards: list[FlowBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for board in boards:
        prediction = iterative_refine_layout(
            model,
            board.evidence.tile_features.to(device),
            _graph_to_device(board.evidence.graph, device),
            board.decoder_layout.to(device),
            grid=args.grid,
            steps=args.refinement_steps,
        )[0].cpu().numpy()
        baseline = evaluate_layout(
            board.decoder_layout[0].numpy(),
            board.target_layout[0].numpy(),
            reference_is_exact=True,
        ).as_dict()
        refined = evaluate_layout(
            prediction,
            board.target_layout[0].numpy(),
            reference_is_exact=True,
        ).as_dict()
        rows.append(
            {
                "filename": board.filename,
                "crop_row": board.crop_row,
                "crop_column": board.crop_column,
                "corruption_seed": board.corruption_seed,
                "baseline": baseline,
                "refined": refined,
                "strict_permutation": bool(
                    np.array_equal(np.sort(prediction), np.arange(args.grid * args.grid))
                ),
            }
        )
    metric_names = (
        "direct_placement",
        "row_accuracy",
        "column_accuracy",
        "translation_aligned_placement",
        "adjacency",
    )
    aggregate = {
        variant: {
            metric: float(np.mean([row[variant][metric] for row in rows]))
            for metric in metric_names
        }
        for variant in ("baseline", "refined")
    }
    aggregate["delta"] = {
        metric: aggregate["refined"][metric] - aggregate["baseline"][metric]
        for metric in metric_names
    }
    return {
        "boards": rows,
        "aggregate": aggregate,
        "all_strict_permutations": all(row["strict_permutation"] for row in rows),
    }, perf_counter() - started


def main() -> None:
    args = _resolved_stage_args(parse_args())
    positive = (
        args.grid,
        args.train_sources,
        args.eval_sources,
        args.steps,
        args.top_k,
        args.dimension,
        args.layers,
        args.coordinate_bands,
        args.time_bands,
        args.sinkhorn_iterations,
        args.refinement_steps,
        args.log_every,
    )
    if min(positive) <= 0 or not 2 <= args.grid <= 24:
        raise ValueError("counts must be positive and grid must be in [2, 24]")
    if not args.top_k < args.grid * args.grid:
        raise ValueError("top-k must be less than tile count")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.random_start_probability) or not (
        0 <= args.random_start_probability <= 1
    ):
        raise ValueError("random-start-probability must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matcher, matcher_contract, matcher_lineage = _load_matcher(
        args.matcher_checkpoint, device
    )
    matcher_lineage_digest = names_digest(matcher_lineage.filenames, sort_names=True)
    known_exposed, scanned_reports = _known_exposed_sources(args.prior_output_root)
    excluded = set(matcher_lineage.filenames) | known_exposed
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=excluded,
        limit=args.train_sources + args.eval_sources,
        seed=args.seed,
    )
    train_records = tuple(records[: args.train_sources])
    eval_records = tuple(records[args.train_sources :])
    if {record["filename"] for record in train_records} & {
        record["filename"] for record in eval_records
    }:
        raise RuntimeError("train/evaluation source overlap")

    preparation_started = perf_counter()
    boards: list[FlowBoard] = []
    for index, record in enumerate(records, start=1):
        boards.append(
            _freeze_board(
                record,
                matcher,
                targets=args.targets,
                grid=args.grid,
                top_k=args.top_k,
                seed=args.seed,
                device=device,
            )
        )
        print(f"froze {index}/{len(records)} {record['filename']}", flush=True)
    preparation_seconds = perf_counter() - preparation_started
    train_boards = boards[: args.train_sources]
    eval_boards = boards[args.train_sources :]
    model = SocketPermutationFlow(
        tile_feature_dimension=5 * int(matcher_contract["dimension"]),
        dimension=args.dimension,
        layers=args.layers,
        coordinate_bands=args.coordinate_bands,
        time_bands=args.time_bands,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    history, training_seconds = _train(model, train_boards, args, device)
    train_evaluation, train_evaluation_seconds = _evaluate(
        model, train_boards, args, device
    )
    eval_evaluation, eval_evaluation_seconds = _evaluate(
        model, eval_boards, args, device
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "socket_permutation_flow.pt"
    contract = {
        "architecture": "edge-conditioned-socket-permutation-flow-v1",
        "tile_feature_source": "frozen SocketMatcher context + four post-GNN socket embeddings",
        "graph": f"partial-OT right/left/down/top top-{args.top_k}",
        "state": "strict current permutation + Fourier coordinates + flow time",
        "relational_layers": args.layers,
        "dimension": args.dimension,
        "coordinate_bands": args.coordinate_bands,
        "time_bands": args.time_bands,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "projection": "Hungarian after every refinement step",
        "shuffled_index_embedding": False,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "contract": contract,
        "matcher_checkpoint": str(args.matcher_checkpoint.resolve()),
        "matcher_checkpoint_sha256": sha256_file(args.matcher_checkpoint),
        "matcher_lineage_digest": matcher_lineage_digest,
        "selection": {
            "train_filenames": [str(record["filename"]) for record in train_records],
            "train_digest": names_digest([str(record["filename"]) for record in train_records]),
            "eval_filenames": [str(record["filename"]) for record in eval_records],
            "eval_digest": names_digest([str(record["filename"]) for record in eval_records]),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "experiment": "edge-conditioned-socket-permutation-flow-v1",
        "stage": args.stage,
        "status": "research-prototype-not-default",
        "hypothesis": (
            "current-layout-conditioned sparse socket relations can refine exact coordinates "
            "where raw absolute tile heads failed"
        ),
        "contract": contract,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {"device_resolved": str(device)},
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "split": "train only",
            "matcher_lineage_source_disjoint": True,
            "flow_train_eval_source_disjoint": True,
            "known_exposed_source_disjoint": True,
            "known_exposure_reports_scanned": scanned_reports,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "labels": "exact synthetic inverse shuffle",
            "start_states": "random or socket decoder144, swap-interpolated toward truth",
        },
        "matcher": {
            "checkpoint": str(args.matcher_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.matcher_checkpoint),
            "contract": matcher_contract,
            "lineage_source_count": len(matcher_lineage.filenames),
            "lineage_digest": matcher_lineage_digest,
        },
        "selection": checkpoint["selection"],
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "runtime_seconds": {
            "evidence_preparation": preparation_seconds,
            "training": training_seconds,
            "train_evaluation": train_evaluation_seconds,
            "source_disjoint_evaluation": eval_evaluation_seconds,
        },
        "training_history": history,
        "capacity_train": train_evaluation,
        "source_disjoint_evaluation": eval_evaluation,
        "default_changed": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "capacity": train_evaluation["aggregate"],
                "source_disjoint": eval_evaluation["aggregate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
