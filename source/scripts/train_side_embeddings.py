#!/usr/bin/env python3
"""Train L1 directional side embeddings on frozen-denoised exact panels."""

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

from puzzle_assembly.learned import (
    SideEmbeddingNet,
    SideSequenceEmbeddingNet,
    direction_labels,
    embedding_hard_triplet_loss,
    embedding_loss,
    embedding_retrieval_metrics,
    load_embedding_checkpoint,
    save_embedding_checkpoint,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--warm-start-stem",
        help="load compatible stem/outside weights while changing embedding/input config",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--panel", choices=["primary_kornia", "independent_libjpeg"], default="primary_kornia")
    parser.add_argument("--view", choices=["denoised", "raw"], default="denoised")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=8)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--replica-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--model-type", choices=["pooled", "sequence"], default="pooled")
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--side-band", type=int, default=4)
    parser.add_argument("--tangent-bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--input-mode",
        choices=["rgb_norm", "rgb_sobel", "sobel_only", "binary_edges"],
        default="rgb_norm",
    )
    parser.add_argument("--edge-threshold", type=float, default=0.12)
    parser.add_argument("--loss", choices=["infonce", "hard_triplet"], default="infonce")
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--cross-entropy-weight", type=float, default=0.25)
    parser.add_argument("--embedding-l2-weight", type=float, default=1e-4)
    parser.add_argument("--outside-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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


def _prepare_source(
    name: str,
    *,
    args: argparse.Namespace,
    stage: str,
    replica: int,
    restorer: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, object, int]:
    seed = per_source_seed(args.seed, f"l1-{args.panel}-{stage}", name, replica)
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel=args.panel, seed=seed)
    solver_tiles = (
        restore_tiles_uint8(
            restorer,
            panel.slot_tiles,
            device,
            batch_size=args.denoise_batch_size,
        )
        if args.view == "denoised"
        else panel.slot_tiles
    )
    tensor = torch.from_numpy(np.ascontiguousarray(solver_tiles.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    return tensor, direction_labels(panel.slot_to_target), seed


def _run_source(
    model: torch.nn.Module,
    tiles: torch.Tensor,
    labels: object,
    *,
    optimizer: torch.optim.Optimizer | None,
    outside_weight: float,
    grad_clip: float,
    loss_name: str,
    triplet_margin: float,
    cross_entropy_weight: float,
    embedding_l2_weight: float,
) -> dict[str, float]:
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        outputs = model(tiles)
        if loss_name == "hard_triplet":
            loss, loss_metrics = embedding_hard_triplet_loss(
                outputs,
                labels,
                temperature=core_model.temperature,
                margin=triplet_margin,
                cross_entropy_weight=cross_entropy_weight,
                embedding_l2_weight=embedding_l2_weight,
                outside_weight=outside_weight,
            )
        else:
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


def _warm_start_stem(
    model: SideEmbeddingNet,
    checkpoint: str,
) -> tuple[dict, list[str]]:
    warm_model, metadata = load_embedding_checkpoint(checkpoint, device="cpu")
    if not isinstance(warm_model, SideEmbeddingNet):
        raise SystemExit("--warm-start-stem requires a pooled SideEmbeddingNet checkpoint")
    target = model.stem.state_dict()
    source = warm_model.stem.state_dict()
    copied: list[str] = []
    for key, value in source.items():
        if key not in target:
            continue
        if target[key].shape == value.shape:
            target[key] = value
            copied.append(f"stem.{key}")
        elif (
            key == "0.weight"
            and target[key].ndim == 4
            and value.ndim == 4
            and target[key].shape[0] == value.shape[0]
            and target[key].shape[2:] == value.shape[2:]
            and target[key].shape[1] > value.shape[1]
        ):
            merged = target[key].cpu().clone()
            merged[:, : value.shape[1]] = value
            target[key] = merged
            copied.append(f"stem.{key}[first_{value.shape[1]}_channels]")
    model.stem.load_state_dict(target, strict=True)
    target_outside = model.outside_head.state_dict()
    source_outside = warm_model.outside_head.state_dict()
    if target_outside.keys() == source_outside.keys() and all(
        target_outside[key].shape == source_outside[key].shape for key in target_outside
    ):
        model.outside_head.load_state_dict(source_outside, strict=True)
        copied.extend(f"outside_head.{key}" for key in target_outside)
    return metadata, copied


def main() -> None:
    args = parse_args()
    if args.train_sources <= 0 or args.val_sources <= 0 or args.epochs <= 0:
        raise SystemExit("source counts and epochs must be positive")
    if args.train_offset < 0 or args.val_offset < 0:
        raise SystemExit("offsets must be non-negative")
    output = Path(args.output)
    report_path = output.with_suffix(".json")
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output} or {report_path}")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    train_names = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.train_offset : args.train_offset + args.train_sources]
    val_names = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.val_offset : args.val_offset + args.val_sources]
    if len(train_names) != args.train_sources or len(val_names) != args.val_sources:
        raise SystemExit("requested source slice extends past the split")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    for parameter in restorer.parameters():
        parameter.requires_grad_(False)
    # The frozen TileNAF denoiser is intentionally kept on cuda:0.  PyTorch
    # DataParallel on Kaggle's T4x2 image can trigger a misaligned-address
    # failure inside its depthwise decoder.  The trainable edge model below is
    # still distributed over both GPUs.
    if args.init_checkpoint and args.warm_start_stem:
        raise SystemExit("choose either --init-checkpoint or --warm-start-stem")
    init_metadata = None
    warm_start_metadata = None
    warm_start_copied: list[str] = []
    if args.init_checkpoint:
        model, init_metadata = load_embedding_checkpoint(
            args.init_checkpoint, device=device
        )
    else:
        if args.model_type == "pooled":
            model = SideEmbeddingNet(
                channels=args.channels,
                embedding_dim=args.embedding_dim,
                side_band=args.side_band,
                tangent_bins=args.tangent_bins,
                temperature=args.temperature,
                input_mode=args.input_mode,
                edge_threshold=args.edge_threshold,
            ).to(device)
        else:
            model = SideSequenceEmbeddingNet(
                channels=args.channels,
                embedding_dim=args.embedding_dim,
                side_band=args.side_band,
                temperature=args.temperature,
            ).to(device)
        if args.warm_start_stem:
            if not isinstance(model, SideEmbeddingNet):
                raise SystemExit("--warm-start-stem currently supports only --model-type pooled")
            warm_start_metadata, warm_start_copied = _warm_start_stem(
                model, args.warm_start_stem
            )
    forward_model: torch.nn.Module = model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best_recall = -1.0
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_records = []
        for index, name in enumerate(train_names):
            tiles, labels, panel_seed = _prepare_source(
                name,
                args=args,
                stage="train",
                replica=args.replica_offset + epoch,
                restorer=restorer,
                device=device,
            )
            metrics = _run_source(
                forward_model,
                tiles,
                labels,
                optimizer=optimizer,
                outside_weight=args.outside_weight,
                grad_clip=args.grad_clip,
                loss_name=args.loss,
                triplet_margin=args.triplet_margin,
                cross_entropy_weight=args.cross_entropy_weight,
                embedding_l2_weight=args.embedding_l2_weight,
            )
            train_records.append(metrics)
            print(
                json.dumps(
                    {
                        "event": "l1_train_source",
                        "epoch": epoch + 1,
                        "index": index + 1,
                        "count": len(train_names),
                        "source": name,
                        "panel_seed": panel_seed,
                        "loss": metrics["loss"],
                        "recall_at_1": metrics["recall_at_1"],
                        "recall_at_32": metrics["recall_at_32"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        val_records = []
        for name in val_names:
            tiles, labels, _ = _prepare_source(
                name,
                args=args,
                stage="validation",
                replica=0,
                restorer=restorer,
                device=device,
            )
            with torch.inference_mode():
                val_records.append(
                    _run_source(
                        forward_model,
                        tiles,
                        labels,
                        optimizer=None,
                        outside_weight=args.outside_weight,
                        grad_clip=args.grad_clip,
                        loss_name=args.loss,
                        triplet_margin=args.triplet_margin,
                        cross_entropy_weight=args.cross_entropy_weight,
                        embedding_l2_weight=args.embedding_l2_weight,
                    )
                )
        epoch_record = {
            "epoch": epoch + 1,
            "train": _mean(train_records),
            "validation": _mean(val_records),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        print(json.dumps({"event": "l1_epoch", **epoch_record}, sort_keys=True), flush=True)
        if epoch_record["validation"]["recall_at_1"] > best_recall:
            best_recall = epoch_record["validation"]["recall_at_1"]
            best_epoch = epoch + 1
            metadata = {
                "seed": args.seed,
                "panel": args.panel,
                "train_names": train_names,
                "val_names": val_names,
                "epoch": best_epoch,
                "best_validation_recall_at_1": best_recall,
                "denoiser": denoiser_metadata,
                "init_checkpoint": args.init_checkpoint,
                "init_checkpoint_sha256": (
                    _sha256(Path(args.init_checkpoint)) if args.init_checkpoint else None
                ),
                "warm_start_stem": args.warm_start_stem,
                "warm_start_stem_sha256": (
                    _sha256(Path(args.warm_start_stem)) if args.warm_start_stem else None
                ),
                "warm_start_copied": warm_start_copied,
                "manifest": str(args.manifest),
                "manifest_sha256": _sha256(Path(args.manifest)),
                "quarantine": str(args.quarantine),
                "quarantine_sha256": _sha256(Path(args.quarantine)),
            }
            save_embedding_checkpoint(output, model, metadata=metadata)

    report = {
        "schema_version": 1,
        "kind": "puzzle_side_embedding_l1_training_report",
        "research_only": True,
        "args": vars(args),
        "device": str(device),
        "model_config": model.config(),
        "denoiser_metadata": denoiser_metadata,
        "init_metadata": init_metadata,
        "warm_start_metadata": warm_start_metadata,
        "warm_start_copied": warm_start_copied,
        "train_names": train_names,
        "val_names": val_names,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_recall_at_1": best_recall,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "l1_complete", "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
