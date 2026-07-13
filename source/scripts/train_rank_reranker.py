#!/usr/bin/env python3
"""Train X0 sparse-candidate reranker from multi-score rank features."""

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

from puzzle_assembly.compatibility import build_classical_score_bank
from puzzle_assembly.learned import (
    RankFeatureNet,
    candidate_rank_features,
    candidate_union,
    direction_labels,
    load_rank_feature_checkpoint,
    rank_feature_compatibility,
    save_rank_feature_checkpoint,
)
from puzzle_assembly.metrics import retrieval_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
FEATURE_SUFFIXES = (
    "pbc",
    "mgc",
    "tone_l1_w2",
    "lab_l1_w2",
    "rgb_l1_w1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--init-checkpoint")
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
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
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


def _prepare_source(
    name: str,
    *,
    args: argparse.Namespace,
    stage: str,
    replica: int,
    restorer: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, object, dict, tuple[np.ndarray, np.ndarray], np.ndarray, list[str], int]:
    seed = per_source_seed(args.seed, f"x0-primary-{stage}", name, replica)
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel="primary_kornia", seed=seed)
    denoised = restore_tiles_uint8(
        restorer, panel.slot_tiles, device, batch_size=args.denoise_batch_size
    )
    bank = build_classical_score_bank(denoised, prefix="denoised", chunk_size=64)
    feature_names = [f"denoised_{suffix}" for suffix in FEATURE_SUFFIXES]
    candidates = candidate_union(
        bank,
        names=feature_names,
        per_score_top_k=args.candidate_top_k,
        cap=args.candidate_cap,
    )
    features = candidate_rank_features(bank, candidates, names=feature_names)
    return (
        denoised,
        direction_labels(panel.slot_to_target),
        bank,
        candidates,
        features,
        feature_names,
        seed,
    )


def _supervised_queries(
    features: np.ndarray,
    candidates: tuple[np.ndarray, np.ndarray],
    labels: object,
) -> tuple[np.ndarray, np.ndarray, float]:
    feature_parts = []
    target_parts = []
    present_total = 0
    query_total = 0
    for direction, queries, targets in (
        (0, labels.right_queries, labels.right_targets),
        (1, labels.down_queries, labels.down_targets),
    ):
        selected_candidates = candidates[direction][queries]
        matches = selected_candidates == targets[:, None]
        present = matches.any(axis=1)
        feature_parts.append(features[direction, queries[present]])
        target_parts.append(matches[present].argmax(axis=1).astype(np.int64))
        present_total += int(present.sum())
        query_total += len(queries)
    return (
        np.concatenate(feature_parts),
        np.concatenate(target_parts),
        present_total / query_total,
    )


def _run_train_source(
    model: torch.nn.Module,
    features: np.ndarray,
    candidates: tuple[np.ndarray, np.ndarray],
    labels: object,
    *,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    selected, targets, coverage = _supervised_queries(features, candidates, labels)
    values = torch.from_numpy(selected).to(device=device)
    target_tensor = torch.from_numpy(targets).to(device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(values)
    loss = F.cross_entropy(logits, target_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    conditional_top1 = (logits.argmax(dim=1) == target_tensor).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "candidate_recall": coverage,
        "conditional_recall_at_1": float(conditional_top1.detach().cpu()),
        "unconditional_recall_at_1": float(conditional_top1.detach().cpu()) * coverage,
    }


def _validate_source(
    model: RankFeatureNet,
    features: np.ndarray,
    candidates: tuple[np.ndarray, np.ndarray],
    labels: object,
    slot_to_target: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    _, _, coverage = _supervised_queries(features, candidates, labels)
    score = rank_feature_compatibility(
        model, features, candidates, device=device, name="denoised_x0_rank_reranker"
    )
    metrics = retrieval_metrics(score, slot_to_target)["combined"]
    metrics["candidate_recall"] = coverage
    return metrics


def main() -> None:
    args = parse_args()
    if min(args.train_sources, args.val_sources, args.epochs) <= 0:
        raise SystemExit("source counts and epochs must be positive")
    if args.candidate_cap < args.candidate_top_k or args.candidate_cap >= 576:
        raise SystemExit("candidate cap must be >= top-k and < 576")
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
    for parameter in restorer.parameters():
        parameter.requires_grad_(False)
    feature_names = [f"denoised_{suffix}" for suffix in FEATURE_SUFFIXES]
    feature_dim = 4 * len(feature_names) + 4
    init_metadata = None
    if args.init_checkpoint:
        core_model, init_feature_names, init_metadata = load_rank_feature_checkpoint(
            args.init_checkpoint, device=device
        )
        if init_feature_names != feature_names:
            raise SystemExit("init checkpoint feature order does not match")
        if core_model.feature_dim != feature_dim:
            raise SystemExit("init checkpoint feature dimension does not match")
    else:
        core_model = RankFeatureNet(
            feature_dim=feature_dim, hidden_dim=args.hidden_dim
        ).to(device)
    forward_model: torch.nn.Module = core_model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = torch.nn.DataParallel(core_model)
    optimizer = torch.optim.AdamW(
        core_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best = -1.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_records = []
        for index, name in enumerate(train_names):
            denoised, labels, _, candidates, features, names, panel_seed = _prepare_source(
                name,
                args=args,
                stage="train",
                replica=args.replica_offset + epoch,
                restorer=restorer,
                device=device,
            )
            del denoised
            if names != feature_names:
                raise RuntimeError("feature order changed")
            metrics = _run_train_source(
                forward_model,
                features,
                candidates,
                labels,
                optimizer=optimizer,
                grad_clip=args.grad_clip,
                device=device,
            )
            train_records.append(metrics)
            print(
                json.dumps(
                    {
                        "event": "x0_train_source",
                        "epoch": epoch + 1,
                        "index": index + 1,
                        "count": len(train_names),
                        "source": name,
                        "panel_seed": panel_seed,
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        val_records = []
        for name in val_names:
            denoised, labels, _, candidates, features, names, _ = _prepare_source(
                name,
                args=args,
                stage="validation",
                replica=0,
                restorer=restorer,
                device=device,
            )
            del denoised
            target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
            seed = per_source_seed(args.seed, "x0-primary-validation", name, 0)
            panel = make_exact_panel(target, panel="primary_kornia", seed=seed)
            val_records.append(
                _validate_source(
                    core_model,
                    features,
                    candidates,
                    labels,
                    panel.slot_to_target,
                    device=device,
                )
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
        print(json.dumps({"event": "x0_epoch", **record}, sort_keys=True), flush=True)
        if val_mean["recall_at_1"] > best:
            best = val_mean["recall_at_1"]
            save_rank_feature_checkpoint(
                output,
                core_model,
                feature_names=feature_names,
                metadata={
                    "epoch": epoch + 1,
                    "seed": args.seed,
                    "train_names": train_names,
                    "val_names": val_names,
                    "candidate_top_k": args.candidate_top_k,
                    "candidate_cap": args.candidate_cap,
                    "best_validation_recall_at_1": best,
                    "denoiser": denoiser_metadata,
                },
            )
    report = {
        "schema_version": 1,
        "kind": "puzzle_rank_feature_x0_training_report",
        "args": vars(args),
        "device": str(device),
        "model_config": core_model.config(),
        "feature_names": feature_names,
        "init_metadata": init_metadata,
        "denoiser_metadata": denoiser_metadata,
        "history": history,
        "best_validation_recall_at_1": best,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "x0_complete", "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
