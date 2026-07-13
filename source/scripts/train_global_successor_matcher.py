#!/usr/bin/env python3
"""Train a SuperGlue-style global partial-bijection matcher on frozen side embeddings."""

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
    GlobalSuccessorMatcher,
    direction_labels,
    embedding_retrieval_metrics,
    global_matching_loss,
    load_embedding_checkpoint,
    save_global_matcher_checkpoint,
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
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument(
        "--panel", choices=["primary_kornia", "independent_libjpeg"], default="primary_kornia"
    )
    parser.add_argument("--view", choices=["denoised", "raw"], default="denoised")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=8)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--replica-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], object, int]:
    seed = per_source_seed(args.seed, f"g0-{args.panel}-{stage}", name, replica)
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel=args.panel, seed=seed)
    solver_tiles = (
        restore_tiles_uint8(
            restorer, panel.slot_tiles, device, batch_size=args.denoise_batch_size
        )
        if args.view == "denoised"
        else panel.slot_tiles
    )
    tensor = torch.from_numpy(
        np.ascontiguousarray(solver_tiles.transpose(0, 3, 1, 2))
    ).to(device=device, dtype=torch.float32)
    encoder.eval()
    with torch.inference_mode():
        embeddings = {key: value.detach() for key, value in encoder(tensor).items()}
    return embeddings, direction_labels(panel.slot_to_target), seed


def _run_source(
    matcher: GlobalSuccessorMatcher,
    embeddings: dict[str, torch.Tensor],
    labels: object,
    *,
    encoder_temperature: float,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    matcher.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        outputs = matcher(embeddings)
        loss, metrics = global_matching_loss(outputs, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), grad_clip)
            optimizer.step()
    baseline = embedding_retrieval_metrics(
        embeddings, labels, temperature=encoder_temperature
    )
    return {**metrics, **{f"baseline_{key}": value for key, value in baseline.items()}}


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
        raise SystemExit("requested source slice extends past split")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    encoder, encoder_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for model in (restorer, encoder):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    embedding_dim = int(encoder.config()["embedding_dim"])
    matcher = GlobalSuccessorMatcher(
        embedding_dim=embedding_dim,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        sinkhorn_iterations=args.sinkhorn_iterations,
        dropout=args.dropout,
        base_temperature=float(encoder.temperature),
    ).to(device)
    optimizer = torch.optim.AdamW(
        matcher.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best_recall = -1.0
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_records = []
        for index, name in enumerate(train_names):
            embeddings, labels, panel_seed = _prepare_source(
                name,
                args=args,
                stage="train",
                replica=args.replica_offset + epoch,
                restorer=restorer,
                encoder=encoder,
                device=device,
            )
            metrics = _run_source(
                matcher,
                embeddings,
                labels,
                encoder_temperature=float(encoder.temperature),
                optimizer=optimizer,
                grad_clip=args.grad_clip,
            )
            train_records.append(metrics)
            print(
                json.dumps(
                    {
                        "event": "g0_train_source",
                        "epoch": epoch + 1,
                        "index": index + 1,
                        "count": len(train_names),
                        "source": name,
                        "panel_seed": panel_seed,
                        "loss": metrics["loss"],
                        "recall_at_1": metrics["recall_at_1"],
                        "baseline_recall_at_1": metrics["baseline_recall_at_1"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        validation_records = []
        for name in val_names:
            embeddings, labels, _ = _prepare_source(
                name,
                args=args,
                stage="validation",
                replica=0,
                restorer=restorer,
                encoder=encoder,
                device=device,
            )
            with torch.inference_mode():
                validation_records.append(
                    _run_source(
                        matcher,
                        embeddings,
                        labels,
                        encoder_temperature=float(encoder.temperature),
                        optimizer=None,
                        grad_clip=args.grad_clip,
                    )
                )
        epoch_record = {
            "epoch": epoch + 1,
            "train": _mean(train_records),
            "validation": _mean(validation_records),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        print(json.dumps({"event": "g0_epoch", **epoch_record}, sort_keys=True), flush=True)
        if epoch_record["validation"]["recall_at_1"] > best_recall:
            best_recall = epoch_record["validation"]["recall_at_1"]
            best_epoch = epoch + 1
            metadata = {
                "seed": args.seed,
                "panel": args.panel,
                "view": args.view,
                "train_names": train_names,
                "val_names": val_names,
                "epoch": best_epoch,
                "best_validation_recall_at_1": best_recall,
                "denoiser": denoiser_metadata,
                "embedding_checkpoint": args.embedding_checkpoint,
                "embedding_checkpoint_sha256": _sha256(Path(args.embedding_checkpoint)),
                "embedding_metadata": encoder_metadata,
                "manifest_sha256": _sha256(Path(args.manifest)),
                "quarantine_sha256": _sha256(Path(args.quarantine)),
            }
            save_global_matcher_checkpoint(output, matcher, metadata=metadata)

    report = {
        "schema_version": 1,
        "kind": "puzzle_global_successor_matcher_g0_training_report",
        "research_only": True,
        "args": vars(args),
        "device": str(device),
        "model_config": matcher.config(),
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": encoder_metadata,
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
    print(json.dumps({"event": "g0_complete", "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
