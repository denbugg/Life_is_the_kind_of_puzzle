#!/usr/bin/env python3
"""Fine-tune L1 side embeddings on high-confidence partial real pseudo-pairs.

The pseudo artifact contains only a partial input-slot to clean-position mapping.
It is never treated as a full permutation: a right/down training edge is emitted
only when both endpoint positions have independently survived the configured
confidence threshold.  Exact synthetic validation uses disjoint source images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from puzzle_assembly.geometry import GRID, TILE_COUNT
from puzzle_assembly.learned import (
    DirectionLabels,
    direction_labels,
    embedding_loss,
    embedding_retrieval_metrics,
    load_embedding_checkpoint,
    save_embedding_checkpoint,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_PSEUDO_GOLD = "runs/denoise_v2/real_gold_train_512.npz"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--pseudo-gold", default=DEFAULT_PSEUDO_GOLD)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=16)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=4)
    parser.add_argument(
        "--val-panels",
        default="primary_kornia,independent_libjpeg",
        help="comma-separated exact validation panels",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=0.0,
        help="keep zero: outside labels are not fully observed in partial pseudo maps",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in sorted(records[0])
    }


def _partial_labels(
    input_slots: np.ndarray,
    clean_positions: np.ndarray,
) -> tuple[DirectionLabels, dict[str, int]]:
    slots = np.asarray(input_slots, dtype=np.int64)
    positions = np.asarray(clean_positions, dtype=np.int64)
    if slots.ndim != 1 or positions.ndim != 1 or len(slots) != len(positions):
        raise ValueError("partial pseudo arrays must be equally sized 1D arrays")
    if len(slots) == 0:
        raise ValueError("partial pseudo mapping is empty")
    if np.any((slots < 0) | (slots >= TILE_COUNT)):
        raise ValueError("input slot is outside the 576-tile grid")
    if np.any((positions < 0) | (positions >= TILE_COUNT)):
        raise ValueError("clean position is outside the 576-tile grid")
    if len(np.unique(slots)) != len(slots) or len(np.unique(positions)) != len(positions):
        raise ValueError("partial pseudo mapping must be one-to-one")

    position_to_slot = {int(position): int(slot) for slot, position in zip(slots, positions)}
    right_queries: list[int] = []
    right_targets: list[int] = []
    down_queries: list[int] = []
    down_targets: list[int] = []
    for position, slot in sorted(position_to_slot.items()):
        if position % GRID < GRID - 1 and position + 1 in position_to_slot:
            right_queries.append(slot)
            right_targets.append(position_to_slot[position + 1])
        if position < TILE_COUNT - GRID and position + GRID in position_to_slot:
            down_queries.append(slot)
            down_targets.append(position_to_slot[position + GRID])
    if not right_queries or not down_queries:
        raise ValueError("partial pseudo mapping has no eligible edge in one direction")

    # Only mapped entries are known, so this tensor is diagnostic unless the
    # caller explicitly enables outside_weight.  Training keeps it disabled.
    outside = np.zeros((TILE_COUNT, 4), dtype=np.float32)
    for position, slot in position_to_slot.items():
        row, column = divmod(position, GRID)
        outside[slot] = (column == 0, column == GRID - 1, row == 0, row == GRID - 1)
    labels = DirectionLabels(
        right_queries=np.asarray(right_queries, dtype=np.int32),
        right_targets=np.asarray(right_targets, dtype=np.int32),
        down_queries=np.asarray(down_queries, dtype=np.int32),
        down_targets=np.asarray(down_targets, dtype=np.int32),
        outside=outside,
    )
    return labels, {
        "mapped_tiles": len(position_to_slot),
        "right_edges": len(right_queries),
        "down_edges": len(down_queries),
        "directed_edges": len(right_queries) + len(down_queries),
    }


def _run_source(
    model: torch.nn.Module,
    tiles: torch.Tensor,
    labels: DirectionLabels,
    *,
    optimizer: torch.optim.Optimizer | None,
    outside_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        outputs = model(tiles)
        loss, loss_metrics = embedding_loss(
            outputs,
            labels,
            temperature=core_model.temperature,
            outside_weight=outside_weight,
        )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
    retrieval = embedding_retrieval_metrics(
        outputs, labels, temperature=core_model.temperature
    )
    return {**loss_metrics, **retrieval}


def _exact_validation(
    model: torch.nn.Module,
    names: list[str],
    panels: list[str],
    *,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    records: dict[str, list[dict[str, float]]] = {panel: [] for panel in panels}
    for panel_name in panels:
        for name in names:
            seed = per_source_seed(args.seed, f"l1-real-pseudo-validation-{panel_name}", name, 0)
            target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
            panel = make_exact_panel(target, panel=panel_name, seed=seed)
            denoised = restore_tiles_uint8(
                restorer,
                panel.slot_tiles,
                device,
                batch_size=args.denoise_batch_size,
            )
            tiles = torch.from_numpy(
                np.ascontiguousarray(denoised.transpose(0, 3, 1, 2))
            ).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                records[panel_name].append(
                    _run_source(
                        model,
                        tiles,
                        direction_labels(panel.slot_to_target),
                        optimizer=None,
                        outside_weight=0.0,
                        grad_clip=args.grad_clip,
                    )
                )
    return {panel: _mean(panel_records) for panel, panel_records in records.items()}


def _validation_score(validation: dict[str, dict[str, float]]) -> float:
    return float(np.mean([metrics["recall_at_1"] for metrics in validation.values()]))


def main() -> None:
    args = parse_args()
    if args.train_sources <= 0 or args.val_sources <= 0 or args.epochs <= 0:
        raise SystemExit("source counts and epochs must be positive")
    if args.train_offset < 0 or args.val_offset < 0:
        raise SystemExit("offsets must be non-negative")
    if args.confidence_threshold <= 0:
        raise SystemExit("--confidence-threshold must be positive")
    if args.outside_weight != 0.0:
        raise SystemExit("partial pseudo maps do not authorize nonzero --outside-weight")
    panels = [value.strip() for value in args.val_panels.split(",") if value.strip()]
    allowed_panels = {"primary_kornia", "independent_libjpeg"}
    if not panels or any(panel not in allowed_panels for panel in panels):
        raise SystemExit(f"--val-panels must select from {sorted(allowed_panels)}")

    output = Path(args.output)
    latest_output = output.with_name(f"{output.stem}_latest{output.suffix}")
    report_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, latest_output, report_path)) and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)

    pseudo_path = Path(args.pseudo_gold)
    with np.load(pseudo_path, allow_pickle=False) as pseudo:
        pseudo_meta = json.loads(str(pseudo["meta"]))
        if pseudo_meta.get("kind") != "high_purity_real_tile_pairs":
            raise SystemExit("unsupported pseudo-gold artifact kind")
        if pseudo_meta.get("split") != "train":
            raise SystemExit("real pseudo fine-tune requires a train-split artifact")
        all_names = pseudo["source_names"].astype(str).tolist()
        source_index = pseudo["source_index"].astype(np.int64)
        input_slot = pseudo["input_slot"].astype(np.int64)
        clean_tile_index = pseudo["clean_tile_index"].astype(np.int64)
        confidence = pseudo["joint_confidence"].astype(np.float32)

    train_names = all_names[args.train_offset : args.train_offset + args.train_sources]
    if len(train_names) != args.train_sources:
        raise SystemExit("requested pseudo train slice extends past the artifact")
    val_names = source_names_for_split(
        "assembly_cal", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.val_offset : args.val_offset + args.val_sources]
    if len(val_names) != args.val_sources:
        raise SystemExit("requested validation slice extends past assembly_cal")
    overlap = sorted(set(train_names) & set(val_names))
    if overlap:
        raise SystemExit(f"pseudo train and exact validation sources overlap: {overlap[:5]}")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    for parameter in restorer.parameters():
        parameter.requires_grad_(False)
    model, init_metadata = load_embedding_checkpoint(args.init_checkpoint, device=device)
    forward_model: torch.nn.Module = model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    selected_by_source: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pseudo_stats: dict[str, dict[str, int]] = {}
    for local_index, name in enumerate(train_names, start=args.train_offset):
        keep = (source_index == local_index) & (confidence >= args.confidence_threshold)
        slots = input_slot[keep]
        positions = clean_tile_index[keep]
        labels, stats = _partial_labels(slots, positions)
        selected_by_source[name] = (slots, positions)
        pseudo_stats[name] = stats

    started = time.perf_counter()
    baseline_validation = _exact_validation(
        forward_model,
        val_names,
        panels,
        args=args,
        restorer=restorer,
        device=device,
    )
    best_score = _validation_score(baseline_validation)
    best_epoch = 0
    base_metadata = {
        "research_only": True,
        "supervision": "partial_real_pseudo_pairs",
        "pseudo_gold": str(pseudo_path),
        "pseudo_gold_sha256": _sha256(pseudo_path),
        "confidence_threshold": args.confidence_threshold,
        "train_names": train_names,
        "val_names": val_names,
        "val_panels": panels,
        "init_checkpoint": args.init_checkpoint,
        "init_checkpoint_sha256": _sha256(Path(args.init_checkpoint)),
        "manifest": args.manifest,
        "manifest_sha256": _sha256(Path(args.manifest)),
        "quarantine": args.quarantine,
        "quarantine_sha256": _sha256(Path(args.quarantine)),
        "denoiser": denoiser_metadata,
    }
    save_embedding_checkpoint(
        output,
        model,
        metadata={**base_metadata, "selected_epoch": 0, "validation": baseline_validation},
    )
    history = [
        {
            "epoch": 0,
            "train": None,
            "validation": baseline_validation,
            "validation_score": best_score,
            "seconds": time.perf_counter() - started,
        }
    ]
    print(json.dumps({"event": "l1_real_pseudo_baseline", **history[0]}, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        order = list(train_names)
        random.Random(args.seed + epoch).shuffle(order)
        train_records: list[dict[str, float]] = []
        for index, name in enumerate(order):
            image = _read_rgb(Path(args.data_root) / "train" / "inputs" / name)
            denoised = restore_tiles_uint8(
                restorer,
                split_tiles_numpy(image),
                device,
                batch_size=args.denoise_batch_size,
            )
            tiles = torch.from_numpy(
                np.ascontiguousarray(denoised.transpose(0, 3, 1, 2))
            ).to(device=device, dtype=torch.float32)
            slots, positions = selected_by_source[name]
            labels, stats = _partial_labels(slots, positions)
            metrics = _run_source(
                forward_model,
                tiles,
                labels,
                optimizer=optimizer,
                outside_weight=0.0,
                grad_clip=args.grad_clip,
            )
            train_records.append(metrics)
            print(
                json.dumps(
                    {
                        "event": "l1_real_pseudo_train_source",
                        "epoch": epoch,
                        "index": index + 1,
                        "count": len(order),
                        "source": name,
                        **stats,
                        "loss": metrics["loss"],
                        "recall_at_1": metrics["recall_at_1"],
                        "recall_at_32": metrics["recall_at_32"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        validation = _exact_validation(
            forward_model,
            val_names,
            panels,
            args=args,
            restorer=restorer,
            device=device,
        )
        score = _validation_score(validation)
        epoch_record = {
            "epoch": epoch,
            "train": _mean(train_records),
            "validation": validation,
            "validation_score": score,
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        save_embedding_checkpoint(
            latest_output,
            model,
            metadata={**base_metadata, "selected_epoch": epoch, "validation": validation},
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_embedding_checkpoint(
                output,
                model,
                metadata={**base_metadata, "selected_epoch": epoch, "validation": validation},
            )
        print(json.dumps({"event": "l1_real_pseudo_epoch", **epoch_record}, sort_keys=True), flush=True)

    aggregate_pseudo = {
        key: int(sum(stats[key] for stats in pseudo_stats.values()))
        for key in ("mapped_tiles", "right_edges", "down_edges", "directed_edges")
    }
    report = {
        "schema_version": 1,
        "kind": "puzzle_side_embedding_l1_real_pseudo_training_report",
        "research_only": True,
        "pseudo_ground_truth_is_partial": True,
        "args": vars(args),
        "device": str(device),
        "denoiser_metadata": denoiser_metadata,
        "init_metadata": init_metadata,
        "pseudo_metadata": pseudo_meta,
        "pseudo_stats": aggregate_pseudo,
        "train_names": train_names,
        "val_names": val_names,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "latest_checkpoint": str(latest_output),
        "latest_checkpoint_sha256": _sha256(latest_output),
        "seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "l1_real_pseudo_complete", "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
