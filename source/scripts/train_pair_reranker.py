#!/usr/bin/env python3
"""Train L0 pair CNN on hard denoised seam candidates."""

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

from puzzle_assembly.compatibility import prediction_compatibility
from puzzle_assembly.learned import (
    SeamPairNet,
    candidate_union,
    direction_labels,
    pair_rerank_compatibility,
    save_pair_checkpoint,
    seam_pair_patches,
)
from puzzle_assembly.metrics import retrieval_metrics
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
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
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
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--side-band", type=int, default=6)
    parser.add_argument("--queries-per-source", type=int, default=256)
    parser.add_argument("--hard-negatives", type=int, default=15)
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


def _prepare(
    name: str,
    *,
    args: argparse.Namespace,
    stage: str,
    replica: int,
    restorer: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, object, int]:
    seed = per_source_seed(args.seed, f"l0-primary-{stage}", name, replica)
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel="primary_kornia", seed=seed)
    denoised = restore_tiles_uint8(
        restorer, panel.slot_tiles, device, batch_size=args.denoise_batch_size
    )
    return denoised, panel.slot_to_target, direction_labels(panel.slot_to_target), seed


def _hard_groups(
    score: object,
    labels: object,
    *,
    rng: np.random.Generator,
    query_count: int,
    negatives: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_direction = query_count // 2
    first_groups = []
    second_groups = []
    direction_groups = []
    for direction, matrix, queries, targets in (
        (0, score.right, labels.right_queries, labels.right_targets),
        (1, score.down, labels.down_queries, labels.down_targets),
    ):
        take = min(per_direction, len(queries))
        selected = rng.choice(len(queries), size=take, replace=False)
        for selected_index in selected.tolist():
            query = int(queries[selected_index])
            target = int(targets[selected_index])
            ordered = np.argsort(matrix[query], kind="stable")
            hard = [
                int(candidate)
                for candidate in ordered.tolist()
                if candidate != query and candidate != target
            ][:negatives]
            if len(hard) != negatives:
                raise RuntimeError("not enough hard negatives")
            candidates = [target, *hard]
            first_groups.extend([query] * len(candidates))
            second_groups.extend(candidates)
            direction_groups.extend([direction] * len(candidates))
    return (
        np.asarray(first_groups, dtype=np.int64),
        np.asarray(second_groups, dtype=np.int64),
        np.asarray(direction_groups, dtype=np.int64),
    )


def _train_one(
    model: torch.nn.Module,
    denoised: np.ndarray,
    score: object,
    labels: object,
    *,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    first, second, directions = _hard_groups(
        score,
        labels,
        rng=rng,
        query_count=args.queries_per_source,
        negatives=args.hard_negatives,
    )
    group_size = args.hard_negatives + 1
    tensor = torch.from_numpy(np.ascontiguousarray(denoised.transpose(0, 3, 1, 2))).to(
        device=device, dtype=torch.float32
    )
    patches = seam_pair_patches(
        tensor,
        torch.as_tensor(first, device=device),
        torch.as_tensor(second, device=device),
        torch.as_tensor(directions, device=device),
        side_band=core_model.side_band,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(patches).reshape(-1, group_size)
    targets = torch.zeros(len(logits), dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "forced_group_recall_at_1": float((logits.argmax(dim=1) == 0).float().mean().cpu()),
    }


def _validate_one(
    model: SeamPairNet,
    denoised: np.ndarray,
    slot_to_target: np.ndarray,
    score: object,
    *,
    device: torch.device,
) -> dict[str, float]:
    candidates = candidate_union(
        {score.name: score}, names=[score.name], per_score_top_k=32, cap=32
    )
    reranked = pair_rerank_compatibility(
        model, denoised, candidates, device=device, name="denoised_l0_pbc32"
    )
    return retrieval_metrics(reranked, slot_to_target)["combined"]


def main() -> None:
    args = parse_args()
    if min(args.train_sources, args.val_sources, args.epochs) <= 0:
        raise SystemExit("source counts and epochs must be positive")
    if args.queries_per_source < 2 or args.hard_negatives <= 0:
        raise SystemExit("query and hard-negative counts are invalid")
    output = Path(args.output)
    report_path = output.with_suffix(".json")
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    train_names = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.train_offset : args.train_offset + args.train_sources]
    val_names = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.val_offset : args.val_offset + args.val_sources]
    restorer, device, denoiser_metadata = load_restorer(args.denoiser, device=args.device)
    for parameter in restorer.parameters():
        parameter.requires_grad_(False)
    # Keep the frozen denoiser on cuda:0; its depthwise decoder is not stable
    # under DataParallel on Kaggle T4x2.  Pair-model batches still use both GPUs.
    model = SeamPairNet(channels=args.channels, side_band=args.side_band).to(device)
    forward_model: torch.nn.Module = model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    best = -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_records = []
        for index, name in enumerate(train_names):
            denoised, slot_to_target, labels, panel_seed = _prepare(
                name,
                args=args,
                stage="train",
                replica=args.replica_offset + epoch,
                restorer=restorer,
                device=device,
            )
            score = prediction_compatibility(denoised)
            metrics = _train_one(
                forward_model,
                denoised,
                score,
                labels,
                rng=np.random.default_rng(panel_seed),
                optimizer=optimizer,
                args=args,
                device=device,
            )
            train_records.append(metrics)
            print(json.dumps({"event": "l0_train_source", "epoch": epoch + 1, "index": index + 1, "count": len(train_names), "source": name, **metrics}, sort_keys=True), flush=True)
        val_records = []
        for name in val_names:
            denoised, slot_to_target, _, _ = _prepare(
                name,
                args=args,
                stage="validation",
                replica=0,
                restorer=restorer,
                device=device,
            )
            score = prediction_compatibility(denoised)
            val_records.append(
                _validate_one(model, denoised, slot_to_target, score, device=device)
            )
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
        print(json.dumps({"event": "l0_epoch", **record}, sort_keys=True), flush=True)
        if val_mean["recall_at_1"] > best:
            best = val_mean["recall_at_1"]
            save_pair_checkpoint(
                output,
                model,
                metadata={
                    "epoch": epoch + 1,
                    "seed": args.seed,
                    "train_names": train_names,
                    "val_names": val_names,
                    "best_validation_recall_at_1": best,
                    "candidate_policy": "pbc_top32",
                    "denoiser": denoiser_metadata,
                },
            )
    report = {
        "schema_version": 1,
        "kind": "puzzle_seam_pair_l0_training_report",
        "args": vars(args),
        "device": str(device),
        "model_config": model.config(),
        "denoiser_metadata": denoiser_metadata,
        "history": history,
        "best_validation_recall_at_1": best,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "l0_complete", "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
