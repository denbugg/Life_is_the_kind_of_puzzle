#!/usr/bin/env python3
"""Train L2b row/column priors on frozen denoised L1 embeddings."""

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
from torch.nn import functional as F

from puzzle_assembly.learned import (
    PositionPriorHead,
    embedding_position_features,
    load_embedding_checkpoint,
    save_position_prior_checkpoint,
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
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=16)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--replica-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
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


@torch.no_grad()
def _prepare_source(
    name: str,
    *,
    args: argparse.Namespace,
    stage: str,
    replica: int,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    seed = per_source_seed(args.seed, f"l2b-primary-{stage}", name, replica)
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel="primary_kornia", seed=seed)
    denoised = restore_tiles_uint8(
        restorer, panel.slot_tiles, device, batch_size=args.denoise_batch_size
    )
    tiles = torch.from_numpy(np.ascontiguousarray(denoised.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    embedding_model.eval()
    features = embedding_position_features(embedding_model(tiles)).detach()
    positions = torch.as_tensor(panel.slot_to_target, device=device, dtype=torch.long)
    return features, positions // 24, positions % 24, seed


def _metrics(
    row_logits: torch.Tensor,
    column_logits: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> dict[str, float]:
    row_correct = row_logits.argmax(dim=1) == rows
    column_correct = column_logits.argmax(dim=1) == columns
    row_loss = F.cross_entropy(row_logits, rows)
    column_loss = F.cross_entropy(column_logits, columns)
    return {
        "loss": float((0.5 * (row_loss + column_loss)).detach().cpu()),
        "row_accuracy": float(row_correct.float().mean().cpu()),
        "column_accuracy": float(column_correct.float().mean().cpu()),
        "position_accuracy": float((row_correct & column_correct).float().mean().cpu()),
    }


def main() -> None:
    args = parse_args()
    if min(args.train_sources, args.val_sources, args.epochs) <= 0:
        raise SystemExit("source counts and epochs must be positive")
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

    restorer, device, denoiser_metadata = load_restorer(args.denoiser, device=args.device)
    embedding_model, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for frozen_model in (restorer, embedding_model):
        frozen_model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)
    with torch.inference_mode():
        probe = embedding_model(torch.zeros(1, 3, 20, 20, device=device))
        feature_dim = int(embedding_position_features(probe).shape[1])
    model = PositionPriorHead(feature_dim=feature_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best = -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_records = []
        model.train()
        for index, name in enumerate(train_names):
            features, rows, columns, panel_seed = _prepare_source(
                name,
                args=args,
                stage="train",
                replica=args.replica_offset + epoch,
                restorer=restorer,
                embedding_model=embedding_model,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            row_logits, column_logits = model(features)
            loss = 0.5 * (
                F.cross_entropy(row_logits, rows) + F.cross_entropy(column_logits, columns)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            record = _metrics(row_logits, column_logits, rows, columns)
            train_records.append(record)
            print(
                json.dumps(
                    {
                        "event": "l2b_train_source",
                        "epoch": epoch + 1,
                        "index": index + 1,
                        "count": len(train_names),
                        "source": name,
                        "panel_seed": panel_seed,
                        **record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        val_records = []
        model.eval()
        with torch.inference_mode():
            for name in val_names:
                features, rows, columns, _ = _prepare_source(
                    name,
                    args=args,
                    stage="validation",
                    replica=0,
                    restorer=restorer,
                    embedding_model=embedding_model,
                    device=device,
                )
                row_logits, column_logits = model(features)
                val_records.append(_metrics(row_logits, column_logits, rows, columns))
        train_mean = {
            key: float(np.mean([record[key] for record in train_records]))
            for key in train_records[0]
        }
        val_mean = {
            key: float(np.mean([record[key] for record in val_records]))
            for key in val_records[0]
        }
        record = {"epoch": epoch + 1, "train": train_mean, "validation": val_mean}
        history.append(record)
        print(json.dumps({"event": "l2b_epoch", **record}, sort_keys=True), flush=True)
        selection = 0.5 * (val_mean["row_accuracy"] + val_mean["column_accuracy"])
        if selection > best:
            best = selection
            save_position_prior_checkpoint(
                output,
                model,
                metadata={
                    "epoch": epoch + 1,
                    "seed": args.seed,
                    "train_names": train_names,
                    "val_names": val_names,
                    "best_mean_axis_accuracy": best,
                    "embedding_checkpoint_sha256": _sha256(Path(args.embedding_checkpoint)),
                    "embedding_metadata": embedding_metadata,
                    "denoiser": denoiser_metadata,
                },
            )
    report = {
        "schema_version": 1,
        "kind": "puzzle_position_prior_l2b_training_report",
        "args": vars(args),
        "device": str(device),
        "model_config": model.config(),
        "embedding_metadata": embedding_metadata,
        "denoiser_metadata": denoiser_metadata,
        "history": history,
        "best_mean_axis_accuracy": best,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "l2b_complete", "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
