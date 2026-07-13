#!/usr/bin/env python3
"""Train a compact contextual corrector on structured QAP-like layout errors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from puzzle_assembly.context_reorg import (
    ContextReorganizationNet,
    QAPSeedResult,
    build_hbt_qap_seed,
    extract_context_reorg_features,
    hard_corrupt_layout,
    hungarian_layout_from_logits,
    iterative_reorganization,
    save_context_reorg_checkpoint,
    topk_similar_slots,
)
from puzzle_assembly.geometry import TILE_COUNT, inverse_permutation
from puzzle_assembly.learned import (
    load_context_position_checkpoint,
    load_embedding_checkpoint,
)
from puzzle_assembly.metrics import layout_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
CORRUPTION_MODES = (
    "qap",
    "component_translation",
    "block_swap",
    "topk_wrong",
    "hybrid",
)


@dataclass(frozen=True)
class PreparedSource:
    name: str
    replica: int
    panel_seed: int
    features: np.ndarray
    true_position_to_slot: np.ndarray
    slot_to_target: np.ndarray
    qap_position_to_slot: np.ndarray
    similar_slots: np.ndarray
    qap_diagnostics: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--context-checkpoint")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--train-split", default="edge_train")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=24)
    parser.add_argument("--val-split", default="edge_development")
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=4)
    parser.add_argument(
        "--panel",
        choices=["primary_kornia", "independent_libjpeg"],
        default="primary_kornia",
    )
    parser.add_argument("--replicas-per-source", type=int, default=1)
    parser.add_argument("--replica-offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--corruptions-per-source", type=int, default=4)
    parser.add_argument("--corruption-min-severity", type=float, default=0.10)
    parser.add_argument("--corruption-max-severity", type=float, default=0.35)
    parser.add_argument("--similar-top-k", type=int, default=8)
    parser.add_argument("--correction-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--qap-iterations", type=int, default=8)
    parser.add_argument("--qap-restarts", type=int, default=1)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.05)
    parser.add_argument("--qap-refine-swaps", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--match-dim", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qap_record(result: QAPSeedResult, truth: np.ndarray) -> dict[str, object]:
    metrics = layout_metrics(result.position_to_slot, truth)
    return {
        "score_name": result.score_name,
        "layout_sha256": hashlib.sha256(
            result.position_to_slot.astype(np.int32).tobytes()
        ).hexdigest(),
        "position_accuracy": metrics["position_accuracy"],
        "wrong_positions": int(
            TILE_COUNT - round(float(metrics["position_accuracy"]) * TILE_COUNT)
        ),
        "combined_adjacency": metrics["combined_adjacency"],
        "soft_cycle_accepted_edges": result.soft_cycle_accepted_edges,
        "soft_cycle_proposed_edges": result.soft_cycle_proposed_edges,
        "soft_cycle_component_sizes": list(result.soft_cycle_component_sizes),
        "qap_objective": result.qap_objective,
        "qap_relaxed_objective": result.qap_relaxed_objective,
        "qap_restart": result.qap_restart,
        "qap_iterations": result.qap_iterations,
        "qap_converged": result.qap_converged,
        "qap_history": list(result.qap_history),
    }


@torch.inference_mode()
def _prepare_source(
    name: str,
    *,
    replica: int,
    stage: str,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    context_model: torch.nn.Module | None,
    device: torch.device,
) -> PreparedSource:
    panel_seed = per_source_seed(args.seed, f"context-reorg-panel-{stage}", name, replica)
    target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel=args.panel, seed=panel_seed)
    denoised = restore_tiles_uint8(
        restorer,
        panel.slot_tiles,
        device,
        batch_size=args.denoise_batch_size,
    )
    feature_bundle = extract_context_reorg_features(
        panel.slot_tiles,
        denoised,
        embedding_model=embedding_model,
        context_model=context_model,
        device=device,
    )
    qap_seed = per_source_seed(
        args.seed, f"context-reorg-qap-{stage}", name, replica
    )
    qap = build_hbt_qap_seed(
        panel.slot_tiles,
        denoised,
        embedding_model=embedding_model,
        device=device,
        seed=qap_seed,
        chunk_size=args.chunk_size,
        qap_iterations=args.qap_iterations,
        qap_restarts=args.qap_restarts,
        qap_boundary_weight=args.qap_boundary_weight,
        qap_refine_swaps=args.qap_refine_swaps,
    )
    truth = inverse_permutation(panel.slot_to_target)
    diagnostics = _qap_record(qap, panel.slot_to_target)
    return PreparedSource(
        name=name,
        replica=replica,
        panel_seed=panel_seed,
        features=feature_bundle.values.astype(np.float16),
        true_position_to_slot=truth,
        slot_to_target=panel.slot_to_target.copy(),
        qap_position_to_slot=qap.position_to_slot.copy(),
        similar_slots=topk_similar_slots(
            feature_bundle.values, top_k=args.similar_top_k
        ).astype(np.int16),
        qap_diagnostics=diagnostics,
    )


def _prepare_many(
    names: list[str],
    *,
    replicas: int,
    replica_offset: int,
    stage: str,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    context_model: torch.nn.Module | None,
    device: torch.device,
) -> list[PreparedSource]:
    prepared = []
    total = len(names) * replicas
    for name_index, name in enumerate(names):
        for local_replica in range(replicas):
            replica = replica_offset + local_replica
            started = time.perf_counter()
            source = _prepare_source(
                name,
                replica=replica,
                stage=stage,
                args=args,
                restorer=restorer,
                embedding_model=embedding_model,
                context_model=context_model,
                device=device,
            )
            prepared.append(source)
            print(
                json.dumps(
                    {
                        "event": "context_reorg_source_prepared",
                        "stage": stage,
                        "index": len(prepared),
                        "count": total,
                        "source": name,
                        "replica": replica,
                        "panel_seed": source.panel_seed,
                        "qap": source.qap_diagnostics,
                        "seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return prepared


def _wrong_positions(layout: np.ndarray, truth: np.ndarray) -> int:
    return int(np.sum(np.asarray(layout) != np.asarray(truth)))


def _validation_metrics(
    model: ContextReorganizationNet,
    sources: list[PreparedSource],
    *,
    device: torch.device,
    rounds: int,
) -> dict[str, float]:
    records = []
    model.eval()
    for source in sources:
        result = iterative_reorganization(
            model,
            source.features.astype(np.float32),
            source.qap_position_to_slot,
            device=device,
            rounds=rounds,
        )
        seed_wrong = _wrong_positions(
            source.qap_position_to_slot, source.true_position_to_slot
        )
        final_wrong = _wrong_positions(
            result.position_to_slot, source.true_position_to_slot
        )
        records.append(
            {
                "seed_wrong": seed_wrong,
                "final_wrong": final_wrong,
                "final_position_accuracy": 1.0 - final_wrong / TILE_COUNT,
            }
        )
    seed_wrong_total = sum(record["seed_wrong"] for record in records)
    final_wrong_total = sum(record["final_wrong"] for record in records)
    return {
        "seed_wrong_positions": float(seed_wrong_total),
        "final_wrong_positions": float(final_wrong_total),
        "wrong_position_reduction": float(
            (seed_wrong_total - final_wrong_total) / max(seed_wrong_total, 1)
        ),
        "final_position_accuracy": float(
            np.mean([record["final_position_accuracy"] for record in records])
        ),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.train_sources,
        args.val_sources,
        args.replicas_per_source,
        args.epochs,
        args.corruptions_per_source,
        args.correction_rounds,
    ) <= 0:
        raise SystemExit("source, epoch, corruption, and round counts must be positive")
    if args.correction_rounds > args.max_rounds:
        raise SystemExit("correction-rounds cannot exceed max-rounds")
    if not (
        0.0 < args.corruption_min_severity <= args.corruption_max_severity <= 1.0
    ):
        raise SystemExit("invalid corruption severity interval")
    if (
        args.qap_iterations < 0
        or args.qap_restarts <= 0
        or args.qap_boundary_weight < 0.0
    ):
        raise SystemExit("invalid QAP settings")
    output = Path(args.output)
    report = Path(args.report) if args.report else output.with_suffix(".json")
    if output.resolve() == report.resolve():
        raise SystemExit("checkpoint and report paths must differ")
    if not args.overwrite and (output.exists() or report.exists()):
        raise SystemExit(f"output exists; pass --overwrite: {output} or {report}")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_all = source_names_for_split(
        args.train_split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    val_all = source_names_for_split(
        args.val_split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    train_names = train_all[args.train_offset : args.train_offset + args.train_sources]
    val_names = val_all[args.val_offset : args.val_offset + args.val_sources]
    if len(train_names) != args.train_sources or len(val_names) != args.val_sources:
        raise SystemExit("requested source slice extends past its split")
    overlap = sorted(set(train_names) & set(val_names))
    if overlap:
        raise SystemExit(f"whole-source split overlap: {overlap[:8]}")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    embedding_model, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    context_model = context_metadata = None
    if args.context_checkpoint:
        context_model, context_metadata = load_context_position_checkpoint(
            args.context_checkpoint, device=device
        )
    frozen_models = [restorer, embedding_model]
    if context_model is not None:
        frozen_models.append(context_model)
    for frozen in frozen_models:
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    started = time.perf_counter()
    train_sources = _prepare_many(
        train_names,
        replicas=args.replicas_per_source,
        replica_offset=args.replica_offset,
        stage="train",
        args=args,
        restorer=restorer,
        embedding_model=embedding_model,
        context_model=context_model,
        device=device,
    )
    val_sources = _prepare_many(
        val_names,
        replicas=1,
        replica_offset=0,
        stage="validation",
        args=args,
        restorer=restorer,
        embedding_model=embedding_model,
        context_model=context_model,
        device=device,
    )
    feature_dim = int(train_sources[0].features.shape[1])
    embedding_dim = int((feature_dim - 15 - (48 if context_model is not None else 0)) // 8 - 1)
    if feature_dim != 2 * (4 * embedding_dim + 4) + (48 if context_model is not None else 0) + 15:
        raise RuntimeError("could not infer HBT embedding dimension from prepared features")
    model = ContextReorganizationNet(
        embedding_dim=embedding_dim,
        has_context_prior=context_model is not None,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        match_dim=args.match_dim,
        max_rounds=args.max_rounds,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_key = (-float("inf"), -float("inf"))
    output.parent.mkdir(parents=True, exist_ok=True)
    corruption_modes = list(CORRUPTION_MODES)
    for epoch in range(args.epochs):
        model.train()
        train_records = []
        source_order = np.random.default_rng(args.seed + epoch).permutation(
            len(train_sources)
        )
        for source_index in source_order.tolist():
            source = train_sources[source_index]
            features = torch.from_numpy(source.features.astype(np.float32)).to(device)
            target = torch.from_numpy(source.slot_to_target.astype(np.int64)).to(device)
            for corruption_index in range(args.corruptions_per_source):
                mode = corruption_modes[
                    (epoch + source_index + corruption_index) % len(corruption_modes)
                ]
                corruption_seed = per_source_seed(
                    args.seed,
                    f"context-reorg-corruption-{epoch}-{mode}",
                    source.name,
                    source.replica * args.corruptions_per_source + corruption_index,
                )
                rng = np.random.default_rng(corruption_seed)
                severity = float(
                    rng.uniform(
                        args.corruption_min_severity,
                        args.corruption_max_severity,
                    )
                )
                base = (
                    source.qap_position_to_slot
                    if mode in {"qap", "hybrid"}
                    else source.true_position_to_slot
                )
                current = hard_corrupt_layout(
                    source.true_position_to_slot,
                    mode=mode,
                    rng=rng,
                    severity=severity,
                    similar_slots=source.similar_slots,
                    base_position_to_slot=base,
                )
                initial_wrong = _wrong_positions(
                    current, source.true_position_to_slot
                )
                optimizer.zero_grad(set_to_none=True)
                losses = []
                round_wrong = []
                for round_index in range(args.correction_rounds):
                    current_tensor = torch.from_numpy(current).to(device=device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        logits = model(
                            features,
                            current_tensor,
                            round_index=round_index,
                        ).squeeze(0)
                        losses.append(F.cross_entropy(logits, target))
                    current = hungarian_layout_from_logits(logits)
                    round_wrong.append(
                        _wrong_positions(current, source.true_position_to_slot)
                    )
                loss = torch.stack(losses).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                record = {
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "initial_wrong": initial_wrong,
                    "final_wrong": round_wrong[-1],
                }
                train_records.append(record)
                print(
                    json.dumps(
                        {
                            "event": "context_reorg_train_example",
                            "epoch": epoch + 1,
                            "source": source.name,
                            "replica": source.replica,
                            "mode": mode,
                            "severity": severity,
                            "corruption_seed": corruption_seed,
                            "round_wrong": round_wrong,
                            **record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        validation = _validation_metrics(
            model,
            val_sources,
            device=device,
            rounds=args.correction_rounds,
        )
        train_mean = {
            key: float(np.mean([record[key] for record in train_records]))
            for key in train_records[0]
        }
        epoch_record = {
            "epoch": epoch + 1,
            "train": train_mean,
            "validation": validation,
        }
        history.append(epoch_record)
        print(
            json.dumps({"event": "context_reorg_epoch", **epoch_record}, sort_keys=True),
            flush=True,
        )
        selection_key = (
            validation["wrong_position_reduction"],
            validation["final_position_accuracy"],
        )
        if selection_key > best_key:
            best_key = selection_key
            save_context_reorg_checkpoint(
                output,
                model,
                metadata={
                    "best_epoch": epoch + 1,
                    "best_validation_wrong_position_reduction": best_key[0],
                    "best_validation_position_accuracy": best_key[1],
                    "seed": args.seed,
                    "train_split": args.train_split,
                    "train_names": train_names,
                    "val_split": args.val_split,
                    "val_names": val_names,
                    "panel": args.panel,
                    "corruption_modes": list(CORRUPTION_MODES),
                    "correction_rounds": args.correction_rounds,
                    "qap": {
                        "iterations": args.qap_iterations,
                        "restarts": args.qap_restarts,
                        "boundary_weight": args.qap_boundary_weight,
                        "refine_swaps": args.qap_refine_swaps,
                    },
                    "denoiser_sha256": _sha256(args.denoiser),
                    "embedding_checkpoint_sha256": _sha256(
                        args.embedding_checkpoint
                    ),
                    "context_checkpoint_sha256": (
                        _sha256(args.context_checkpoint)
                        if args.context_checkpoint
                        else None
                    ),
                },
            )

    payload = {
        "schema_version": 1,
        "kind": "puzzle_context_reorganization_r0_training_report",
        "args": vars(args),
        "device": str(device),
        "model_config": model.config(),
        "feature_schema": {
            "raw_hbt": 4 * embedding_dim + 4,
            "denoised_hbt": 4 * embedding_dim + 4,
            "context_t0_probabilities": 48 if context_model is not None else 0,
            "raw_denoised_appearance": 15,
            "total": feature_dim,
        },
        "corruption_modes": list(CORRUPTION_MODES),
        "deterministic": {
            "python_random_seed": args.seed,
            "numpy_seed": args.seed % (2**32),
            "torch_seed": args.seed,
            "deterministic_algorithms_warn_only": True,
            "cudnn_benchmark": False,
        },
        "whole_source_split": {
            "train_split": args.train_split,
            "train_names": train_names,
            "validation_split": args.val_split,
            "validation_names": val_names,
            "intersection": overlap,
        },
        "manifest_sha256": _sha256(args.manifest),
        "quarantine_sha256": _sha256(args.quarantine),
        "denoiser_metadata": denoiser_metadata,
        "denoiser_checkpoint_sha256": _sha256(args.denoiser),
        "embedding_metadata": embedding_metadata,
        "embedding_checkpoint_sha256": _sha256(args.embedding_checkpoint),
        "context_metadata": context_metadata,
        "context_checkpoint_sha256": (
            _sha256(args.context_checkpoint) if args.context_checkpoint else None
        ),
        "prepared_train": [
            {
                "source": source.name,
                "replica": source.replica,
                "panel_seed": source.panel_seed,
                "qap": source.qap_diagnostics,
            }
            for source in train_sources
        ],
        "prepared_validation": [
            {
                "source": source.name,
                "replica": source.replica,
                "panel_seed": source.panel_seed,
                "qap": source.qap_diagnostics,
            }
            for source in val_sources
        ],
        "history": history,
        "best_validation_wrong_position_reduction": best_key[0],
        "best_validation_position_accuracy": best_key[1],
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "seconds": time.perf_counter() - started,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "context_reorg_training_complete",
                "checkpoint": str(output),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "report": str(report),
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
