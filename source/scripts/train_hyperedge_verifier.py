#!/usr/bin/env python3
"""Train and precision-calibrate the sparse learned 2x2 hyperedge verifier."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from puzzle_assembly.compatibility import build_classical_score_bank, fuse_ranked_scores
from puzzle_assembly.hyperedge import (
    HyperedgeVerifierNet,
    PlaquetteCandidate,
    ScoredPlaquette,
    accepted_hyperedge_metrics,
    generate_candidate_plaquettes,
    is_true_plaquette,
    plaquette_pair_features,
    plaquette_pixels,
    save_hyperedge_checkpoint,
    score_plaquettes,
    select_sparse_hyperedges,
    true_plaquettes,
)
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_EMBEDDING = "runs/assembly_v1/hbt_d320_denoised_rgb_sobel.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint", default=DEFAULT_EMBEDDING)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-sources", type=int, default=64)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--val-sources", type=int, default=8)
    parser.add_argument(
        "--panels",
        default="primary_kornia,independent_libjpeg",
        help="exact degradation panels alternated by source/epoch",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-per-anchor", type=int, default=4)
    parser.add_argument("--positives-per-source", type=int, default=96)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--pair-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--edge-threshold", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--min-calibration-accepted", type=int, default=24)
    parser.add_argument("--max-hyperedges", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in sorted(records[0])
    }


def build_pair_scores(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    embedding_model: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[Any, Any, list[Any]]:
    bank = build_classical_score_bank(raw_tiles, prefix="raw", chunk_size=64)
    bank.update(build_classical_score_bank(denoised_tiles, prefix="denoised", chunk_size=64))
    c1_views = {}
    for view in ("raw", "denoised"):
        names = [
            name
            for name in sorted(bank)
            if name.startswith(f"{view}_") and not name.endswith("_c2")
        ]
        c1_views[view] = fuse_ranked_scores(
            bank, names=names, name=f"{view}_C1_equal_rank_fusion"
        )
        bank[c1_views[view].name] = c1_views[view]
    cross_c1 = fuse_ranked_scores(
        bank,
        names=[c1_views["raw"].name, c1_views["denoised"].name],
        weights={c1_views["denoised"].name: 2.0},
        name="raw_denoised_C1_dn2_rank_fusion",
    )
    bank[cross_c1.name] = cross_c1
    hbt, _ = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_hbt_embedding",
    )
    bank[hbt.name] = hbt
    qap_score = fuse_ranked_scores(
        bank,
        names=[c1_views["denoised"].name, hbt.name],
        weights={hbt.name: 4.0},
        name="denoised_C1_HBTw4_rank_fusion",
    )
    return c1_views["denoised"], hbt, [
        c1_views["raw"],
        c1_views["denoised"],
        cross_c1,
        hbt,
        qap_score,
    ]


def prepare_exact_source(
    name: str,
    panel_name: str,
    panel_seed: int,
    *,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, list[PlaquetteCandidate]]:
    target = read_rgb(Path(args.data_root) / "train" / "targets" / name)
    panel = make_exact_panel(target, panel=panel_name, seed=panel_seed)
    raw_tiles = panel.slot_tiles
    denoised = restore_tiles_uint8(
        restorer, raw_tiles, device, batch_size=args.denoise_batch_size
    )
    c1, hbt, candidate_scores = build_pair_scores(
        raw_tiles, denoised, embedding_model, device=device
    )
    candidates = generate_candidate_plaquettes(
        candidate_scores,
        top_k=args.top_k,
        max_per_anchor_per_score=args.max_per_anchor,
    )
    return raw_tiles, denoised, panel.slot_to_target, c1, hbt, candidates


def training_examples(
    candidates: list[PlaquetteCandidate],
    slot_to_target: np.ndarray,
    *,
    positives_per_source: int,
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[list[PlaquetteCandidate], np.ndarray]:
    positives = true_plaquettes(slot_to_target)
    positive_indices = rng.choice(
        len(positives), size=min(positives_per_source, len(positives)), replace=False
    )
    selected_positives = [positives[int(index)] for index in sorted(positive_indices.tolist())]
    hard_negatives = [
        candidate for candidate in candidates if not is_true_plaquette(candidate, slot_to_target)
    ]
    negative_count = min(len(hard_negatives), len(selected_positives) * negative_ratio)
    # Candidates are already ordered by top-k cycle cost. Sample from a tight
    # hard pool to avoid learning only trivial random-tile negatives.
    hard_pool = hard_negatives[: max(negative_count * 4, negative_count)]
    if negative_count:
        indices = rng.choice(len(hard_pool), size=negative_count, replace=False)
        selected_negatives = [hard_pool[int(index)] for index in sorted(indices.tolist())]
    else:
        selected_negatives = []
    examples = [*selected_positives, *selected_negatives]
    labels = np.concatenate(
        [
            np.ones(len(selected_positives), dtype=np.float32),
            np.zeros(len(selected_negatives), dtype=np.float32),
        ]
    )
    order = rng.permutation(len(examples))
    return [examples[int(index)] for index in order.tolist()], labels[order]


def run_training_source(
    forward_model: torch.nn.Module,
    core_model: HyperedgeVerifierNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    c1_score: Any,
    hbt_score: Any,
    examples: list[PlaquetteCandidate],
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    grad_clip: float,
) -> dict[str, float]:
    forward_model.train()
    features = plaquette_pair_features(examples, c1_score, hbt_score)
    losses = []
    correct = 0
    total = 0
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    positive_weight = torch.tensor(negatives / positives, device=device)
    for start in range(0, len(examples), batch_size):
        batch_candidates = examples[start : start + batch_size]
        target = torch.from_numpy(labels[start : start + batch_size]).to(device=device)
        pixels = plaquette_pixels(raw_tiles, denoised_tiles, batch_candidates).to(
            device=device, dtype=torch.float32
        )
        pair = torch.from_numpy(features[start : start + batch_size]).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            enabled=device.type == "cuda",
        ):
            logits = forward_model(pixels, pair)
            loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(core_model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        predictions = logits.detach() >= 0.0
        correct += int((predictions == (target >= 0.5)).sum().item())
        total += len(target)
        losses.append(float(loss.detach().item()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(correct / total) if total else 0.0,
        "examples": float(total),
        "positive_fraction": float(labels.mean()) if len(labels) else 0.0,
    }


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-probabilities, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def calibrate_threshold(
    records: list[dict[str, Any]],
    *,
    target_precision: float,
    min_accepted: int,
    max_hyperedges: int,
) -> dict[str, Any]:
    all_probabilities = np.asarray(
        [item.probability for record in records for item in record["scored"]],
        dtype=np.float64,
    )
    if not len(all_probabilities):
        return {
            "threshold": 1.0,
            "precision": 1.0,
            "coverage": 0.0,
            "accepted": 0,
            "correct": 0,
            "precision_gate": False,
        }
    quantiles = np.quantile(all_probabilities, np.linspace(0.50, 0.999, 96))
    thresholds = sorted(
        set([0.5, 0.75, 0.9, 0.95, 0.99, *quantiles.tolist()]), reverse=True
    )
    candidates = []
    for threshold in thresholds:
        accepted_count = 0
        correct_count = 0
        coverage = []
        for record in records:
            accepted = select_sparse_hyperedges(
                record["scored"],
                threshold=float(threshold),
                max_hyperedges=max_hyperedges,
            )
            metrics = accepted_hyperedge_metrics(accepted, record["slot_to_target"])
            accepted_count += int(metrics["accepted"])
            correct_count += int(metrics["correct"])
            coverage.append(float(metrics["coverage"]))
        precision = correct_count / accepted_count if accepted_count else 1.0
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "coverage": float(np.mean(coverage)),
                "accepted": accepted_count,
                "correct": correct_count,
                "precision_gate": bool(
                    accepted_count >= min_accepted and precision >= target_precision
                ),
            }
        )
    passing = [candidate for candidate in candidates if candidate["precision_gate"]]
    if passing:
        return max(
            passing,
            key=lambda item: (
                item["coverage"], item["precision"], item["accepted"], item["threshold"]
            ),
        )
    viable = [candidate for candidate in candidates if candidate["accepted"] >= min_accepted]
    if viable:
        return max(
            viable,
            key=lambda item: (
                item["precision"], item["coverage"], item["threshold"]
            ),
        )
    return max(candidates, key=lambda item: (item["accepted"], item["precision"]))


@torch.inference_mode()
def validate_model(
    model: HyperedgeVerifierNet,
    names: list[str],
    panels: list[str],
    *,
    epoch: int,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records = []
    labels_flat = []
    probabilities_flat = []
    candidate_recall = []
    for index, name in enumerate(names):
        panel_name = panels[index % len(panels)]
        seed = per_source_seed(args.seed, f"hyperedge-validation-{panel_name}", name, 0)
        raw, denoised, slot_to_target, c1, hbt, candidates = prepare_exact_source(
            name,
            panel_name,
            seed,
            args=args,
            restorer=restorer,
            embedding_model=embedding_model,
            device=device,
        )
        scored = score_plaquettes(
            model,
            raw,
            denoised,
            candidates,
            c1,
            hbt,
            device=device,
            batch_size=args.batch_size,
        )
        labels = np.asarray(
            [is_true_plaquette(item.candidate, slot_to_target) for item in scored],
            dtype=np.int32,
        )
        probabilities = np.asarray([item.probability for item in scored], dtype=np.float64)
        labels_flat.append(labels)
        probabilities_flat.append(probabilities)
        candidate_recall.append(float(labels.sum() / (23**2)))
        records.append(
            {
                "name": name,
                "panel": panel_name,
                "seed": seed,
                "slot_to_target": slot_to_target,
                "scored": scored,
                "candidate_count": len(candidates),
                "candidate_positives": int(labels.sum()),
            }
        )
        print(
            json.dumps(
                {
                    "event": "hyperedge_validation_source",
                    "epoch": epoch,
                    "index": index + 1,
                    "count": len(names),
                    "source": name,
                    "panel": panel_name,
                    "candidates": len(candidates),
                    "candidate_positives": int(labels.sum()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    labels_array = np.concatenate(labels_flat) if labels_flat else np.empty(0, dtype=np.int32)
    probability_array = (
        np.concatenate(probabilities_flat) if probabilities_flat else np.empty(0)
    )
    calibration = calibrate_threshold(
        records,
        target_precision=args.target_precision,
        min_accepted=args.min_calibration_accepted,
        max_hyperedges=args.max_hyperedges,
    )
    summary = {
        "average_precision": average_precision(labels_array, probability_array),
        "candidate_recall": float(np.mean(candidate_recall)) if candidate_recall else 0.0,
        "candidate_count": int(len(labels_array)),
        "candidate_positives": int(labels_array.sum()),
        "calibration": calibration,
    }
    return summary, records


def main() -> None:
    args = parse_args()
    if (
        args.train_sources <= 0
        or args.val_sources <= 0
        or args.epochs <= 0
        or args.batch_size <= 0
        or args.positives_per_source <= 0
        or args.negative_ratio <= 0
    ):
        raise SystemExit("source counts, epochs, batch sizes, and sampling limits must be positive")
    if args.train_offset < 0 or args.val_offset < 0:
        raise SystemExit("offsets must be non-negative")
    if not 0.5 <= args.target_precision <= 1.0:
        raise SystemExit("target-precision must lie in [0.5, 1]")
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    allowed_panels = {"primary_kornia", "independent_libjpeg"}
    if not panels or any(panel not in allowed_panels for panel in panels):
        raise SystemExit(f"--panels must select from {sorted(allowed_panels)}")
    output = Path(args.output)
    report = Path(args.report) if args.report else output.with_suffix(".json")
    if output.resolve() == report.resolve():
        raise SystemExit("checkpoint and report paths must differ")
    if not args.overwrite and (output.exists() or report.exists()):
        raise SystemExit("output exists; pass --overwrite")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_names = source_names_for_split(
        "edge_train", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.train_offset : args.train_offset + args.train_sources]
    val_names = source_names_for_split(
        "edge_development", manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.val_offset : args.val_offset + args.val_sources]
    if len(train_names) != args.train_sources or len(val_names) != args.val_sources:
        raise SystemExit("requested source slice extends past its whole-source split")
    if set(train_names) & set(val_names):
        raise RuntimeError("whole-source train/validation leakage")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    for parameter in restorer.parameters():
        parameter.requires_grad_(False)
    embedding_model, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for parameter in embedding_model.parameters():
        parameter.requires_grad_(False)
    embedding_model.eval()
    model = HyperedgeVerifierNet(
        channels=args.channels,
        pair_hidden=args.pair_hidden,
        dropout=args.dropout,
        edge_threshold=args.edge_threshold,
    ).to(device)
    forward_model: torch.nn.Module = model
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        forward_model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_key: tuple[Any, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    best_epoch = -1
    started = time.perf_counter()
    for epoch_index in range(args.epochs):
        train_records = []
        for source_index, name in enumerate(train_names):
            panel_name = panels[(epoch_index + source_index) % len(panels)]
            panel_seed = per_source_seed(
                args.seed,
                f"hyperedge-train-{panel_name}",
                name,
                epoch_index,
            )
            raw, denoised, slot_to_target, c1, hbt, candidates = prepare_exact_source(
                name,
                panel_name,
                panel_seed,
                args=args,
                restorer=restorer,
                embedding_model=embedding_model,
                device=device,
            )
            rng = np.random.default_rng(
                per_source_seed(args.seed, "hyperedge-example-sampling", name, epoch_index)
            )
            examples, labels = training_examples(
                candidates,
                slot_to_target,
                positives_per_source=args.positives_per_source,
                negative_ratio=args.negative_ratio,
                rng=rng,
            )
            metrics = run_training_source(
                forward_model,
                model,
                optimizer,
                scaler,
                raw,
                denoised,
                c1,
                hbt,
                examples,
                labels,
                device=device,
                batch_size=args.batch_size,
                grad_clip=args.grad_clip,
            )
            train_records.append(metrics)
            print(
                json.dumps(
                    {
                        "event": "hyperedge_train_source",
                        "epoch": epoch_index + 1,
                        "index": source_index + 1,
                        "count": len(train_names),
                        "source": name,
                        "panel": panel_name,
                        "panel_seed": panel_seed,
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        validation, _ = validate_model(
            model,
            val_names,
            panels,
            epoch=epoch_index + 1,
            args=args,
            restorer=restorer,
            embedding_model=embedding_model,
            device=device,
        )
        epoch_record = {
            "epoch": epoch_index + 1,
            "train": mean_metrics(train_records),
            "validation": validation,
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        calibration = validation["calibration"]
        key = (
            bool(calibration["precision_gate"]),
            float(calibration["coverage"]),
            float(calibration["precision"]),
            float(validation["average_precision"]),
            -float(calibration["threshold"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_validation = deepcopy(validation)
            best_epoch = epoch_index + 1
        print(json.dumps({"event": "hyperedge_epoch", **epoch_record}, sort_keys=True), flush=True)

    if best_state is None or best_validation is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state, strict=True)
    threshold = float(best_validation["calibration"]["threshold"])
    metadata = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "train_names": train_names,
        "train_names_sha256": names_sha256(train_names),
        "val_names": val_names,
        "val_names_sha256": names_sha256(val_names),
        "whole_source_split": True,
        "panels": panels,
        "candidate_generation": {
            "top_k": args.top_k,
            "max_per_anchor_per_score": args.max_per_anchor,
            "scores": "raw C1, denoised C1, cross C1, HBT, denoised-C1+HBTw4",
            "hard_negatives": "false top-k square closures",
        },
        "target_use": "training labels and exact validation only; never inference features",
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": embedding_metadata,
        "denoiser_sha256": sha256(Path(args.denoiser)),
        "embedding_checkpoint_sha256": sha256(Path(args.embedding_checkpoint)),
        "manifest_sha256": sha256(Path(args.manifest)),
        "quarantine_sha256": sha256(Path(args.quarantine)),
    }
    save_hyperedge_checkpoint(
        output, model, threshold=threshold, metadata=metadata
    )
    report_payload = {
        "schema_version": 1,
        "kind": "puzzle_hyperedge_verifier_h0_training_report",
        "research_only": True,
        "args": vars(args),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "data_parallel": isinstance(forward_model, torch.nn.DataParallel),
        "model_config": model.config(),
        "train_names": train_names,
        "val_names": val_names,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "threshold": threshold,
        "checkpoint": str(output),
        "checkpoint_sha256": sha256(output),
        "seconds": time.perf_counter() - started,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "hyperedge_training_complete",
                "checkpoint": str(output),
                "checkpoint_sha256": report_payload["checkpoint_sha256"],
                "report": str(report),
                "threshold": threshold,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
