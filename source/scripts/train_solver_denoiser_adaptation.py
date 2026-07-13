#!/usr/bin/env python3
"""Fine-tune a warm-start three-view HBT scorer on whole-source exact panels."""

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

from puzzle_assembly.denoiser_adaptation import (
    MultiViewSideEmbeddingNet,
    names_sha256,
    save_multiview_checkpoint,
    sha256_file,
    validate_protocol_safety,
)
from puzzle_assembly.learned import (
    SideEmbeddingNet,
    direction_labels,
    embedding_hard_triplet_loss,
    embedding_loss,
    embedding_retrieval_metrics,
    load_embedding_checkpoint,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--old-denoiser")
    parser.add_argument("--new-denoiser")
    parser.add_argument("--production-hbt")
    parser.add_argument("--variant", choices=("balanced", "hard_negative"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-sources", type=int)
    parser.add_argument("--validation-sources", type=int, default=8)
    parser.add_argument("--denoise-batch-size", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values)).permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float32
    )


def _verify_path(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA mismatch: {actual}")


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in sorted(records[0])
    }


def _views_for_panel(
    target: np.ndarray,
    *,
    panel_name: str,
    panel_seed: int,
    old_restorer: torch.nn.Module,
    new_restorer: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], object]:
    panel = make_exact_panel(target, panel=panel_name, seed=panel_seed)
    old_denoised = restore_tiles_uint8(
        old_restorer, panel.slot_tiles, device, batch_size=batch_size
    )
    new_denoised = restore_tiles_uint8(
        new_restorer, panel.slot_tiles, device, batch_size=batch_size
    )
    return (
        {
            "dirty": _tensor(panel.slot_tiles, device),
            "old_denoised": _tensor(old_denoised, device),
            "new_denoised": _tensor(new_denoised, device),
        },
        direction_labels(panel.slot_to_target),
    )


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    validate_protocol_safety(config)
    if config.get("upstream_new_denoiser_result", {}).get("permits_adaptation") is not True:
        raise SystemExit("upstream denoiser selected no checkpoint; scorer training is blocked")
    interlock = config["launch_interlock"]
    if not (
        interlock.get("gpu_training_authorized_now") is True
        and interlock.get("root_launch_signal") == "ROOT_AUTHORIZED"
        and interlock.get("candidate_graph_oracle_verdict")
        in interlock.get("accepted_oracle_verdict_values", [])
    ):
        raise SystemExit("runtime config launch interlock is not cleared")

    output = _resolve(args.output)
    report_path = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or report_path.exists()):
        raise SystemExit("checkpoint or report exists; pass --overwrite")
    assets = config["authoritative_inputs"]
    old_path = _resolve(args.old_denoiser or assets["old_denoiser"]["path"])
    new_path = _resolve(args.new_denoiser or assets["new_denoiser"]["path"])
    hbt_path = _resolve(args.production_hbt or assets["production_hbt"]["path"])
    _verify_path(old_path, assets["old_denoiser"]["sha256"], "old denoiser")
    _verify_path(new_path, assets["new_denoiser"]["sha256"], "new denoiser")
    _verify_path(hbt_path, assets["production_hbt"]["sha256"], "production HBT")

    training = config["training"]
    variant = training["variants"][args.variant]
    epochs = args.epochs or int(training["epochs"])
    train_count = args.train_sources or int(training["sources_per_epoch"])
    batch_size = args.denoise_batch_size or int(training["denoise_batch_size"])
    if min(epochs, train_count, args.validation_sources, batch_size) <= 0:
        raise SystemExit("epochs/source counts/batch size must be positive")
    if train_count > config["source_partitions"]["scorer_training"]["count"]:
        raise SystemExit("train source count exceeds the frozen training slice")
    if args.validation_sources > config["source_partitions"]["exact_selection"]["count"]:
        raise SystemExit("validation source count exceeds the exact selection slice")

    seed = int(variant["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = _resolve(assets["manifest"]["path"])
    quarantine = _resolve(assets["quarantine"]["path"])
    train_spec = config["source_partitions"]["scorer_training"]
    train_names = source_names_for_split(
        train_spec["split"], manifest_path=manifest, quarantine_path=quarantine
    )[train_spec["offset"] : train_spec["offset"] + train_count]
    exact_names = list(config["source_partitions"]["exact_selection"]["names"])
    val_names = exact_names[: args.validation_sources]
    if set(train_names) & set(val_names):
        raise RuntimeError("whole-source training/validation overlap")
    expected_train_prefix = source_names_for_split(
        train_spec["split"], manifest_path=manifest, quarantine_path=quarantine
    )[
        train_spec["offset"] : train_spec["offset"] + train_spec["count"]
    ]
    if names_sha256(expected_train_prefix) != train_spec["names_sha256"]:
        raise RuntimeError("frozen training source slice drifted")
    if names_sha256(exact_names) != config["source_partitions"]["exact_selection"]["names_sha256"]:
        raise RuntimeError("frozen exact selection source slice drifted")

    old_restorer, device, old_metadata = load_restorer(
        old_path, device=args.device, state=assets["old_denoiser"]["state"]
    )
    new_restorer, new_device, new_metadata = load_restorer(
        new_path, device=str(device), state=assets["new_denoiser"]["state"]
    )
    if new_device != device:
        raise RuntimeError("old and new denoisers resolved to different devices")
    base_hbt, hbt_metadata = load_embedding_checkpoint(hbt_path, device=device)
    if not isinstance(base_hbt, SideEmbeddingNet):
        raise TypeError("production HBT checkpoint is not a pooled SideEmbeddingNet")
    model = MultiViewSideEmbeddingNet.from_production_encoder(base_hbt).to(device)
    for restorer in (old_restorer, new_restorer):
        restorer.eval()
        for parameter in restorer.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    freeze_sources = int(variant["freeze_encoder_sources"])
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(freeze_sources <= 0)

    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best_recall = -1.0
    best_epoch = -1
    processed = 0
    started = time.perf_counter()
    data_root = _resolve(args.data_root)
    for epoch in range(epochs):
        train_records: list[dict[str, float]] = []
        for index, name in enumerate(train_names):
            if processed == freeze_sources:
                for parameter in model.encoder.parameters():
                    parameter.requires_grad_(True)
            panel_name = training["panels"][(epoch + index) % len(training["panels"])]
            panel_seed = per_source_seed(
                seed, f"solver-denoiser-adaptation-train-{panel_name}", name, epoch
            )
            target = _read_rgb(data_root / "train" / "targets" / name)
            views, labels = _views_for_panel(
                target,
                panel_name=panel_name,
                panel_seed=panel_seed,
                old_restorer=old_restorer,
                new_restorer=new_restorer,
                device=device,
                batch_size=batch_size,
            )
            view_mask = torch.tensor(
                [
                    float(rng.random() >= float(variant["dirty_view_dropout"])),
                    float(rng.random() >= float(variant["new_view_dropout"])),
                ],
                device=device,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(views, view_mask=view_mask)
            if variant["loss"] == "hard_triplet":
                loss, metrics = embedding_hard_triplet_loss(
                    outputs,
                    labels,
                    temperature=model.temperature,
                    margin=float(variant["triplet_margin"]),
                    cross_entropy_weight=float(variant["cross_entropy_weight"]),
                    outside_weight=float(training["outside_weight"]),
                )
            else:
                loss, metrics = embedding_loss(
                    outputs,
                    labels,
                    temperature=model.temperature,
                    outside_weight=float(training["outside_weight"]),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            optimizer.step()
            processed += 1
            metrics.update(
                {
                    "dirty_gate": float(outputs["residual_gates"][0].detach().cpu()),
                    "new_gate": float(outputs["residual_gates"][1].detach().cpu()),
                }
            )
            train_records.append(metrics)
            if (index + 1) % 16 == 0 or index + 1 == len(train_names):
                print(
                    json.dumps(
                        {
                            "event": "solver_adaptation_train_source",
                            "variant": args.variant,
                            "epoch": epoch + 1,
                            "index": index + 1,
                            "count": len(train_names),
                            "panel": panel_name,
                            "source": name,
                            "loss": metrics["loss"],
                            "recall_at_1": metrics["recall_at_1"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        validation_records: list[dict[str, float]] = []
        model.eval()
        for panel_name in training["panels"]:
            for name in val_names:
                panel_seed = per_source_seed(
                    config["panel_seeds"]["master"],
                    f"solver-denoiser-adaptation-validation-{panel_name}",
                    name,
                    0,
                )
                target = _read_rgb(data_root / "train" / "targets" / name)
                views, labels = _views_for_panel(
                    target,
                    panel_name=panel_name,
                    panel_seed=panel_seed,
                    old_restorer=old_restorer,
                    new_restorer=new_restorer,
                    device=device,
                    batch_size=batch_size,
                )
                with torch.inference_mode():
                    metrics = embedding_retrieval_metrics(
                        model(views), labels, temperature=model.temperature, ks=(1, 5, 10)
                    )
                metrics["panel_primary"] = float(panel_name == "primary_kornia")
                validation_records.append(metrics)
        epoch_record = {
            "epoch": epoch + 1,
            "train": _mean(train_records),
            "validation": _mean(validation_records),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        validation_recall = epoch_record["validation"]["recall_at_1"]
        print(json.dumps({"event": "solver_adaptation_epoch", **epoch_record}, sort_keys=True), flush=True)
        if validation_recall > best_recall:
            best_recall = validation_recall
            best_epoch = epoch + 1
            save_multiview_checkpoint(
                output,
                model,
                metadata={
                    "variant": args.variant,
                    "seed": seed,
                    "epoch": best_epoch,
                    "validation_recall_at_1": best_recall,
                    "train_names_sha256": names_sha256(train_names),
                    "validation_names_sha256": names_sha256(val_names),
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "old_denoiser_sha256": sha256_file(old_path),
                    "new_denoiser_sha256": sha256_file(new_path),
                    "production_hbt_sha256": sha256_file(hbt_path),
                },
            )

    report = {
        "schema_version": 1,
        "kind": "solver_denoiser_adaptation_training_report",
        "status": "research_checkpoint_requires_stage1_gate",
        "variant": args.variant,
        "seed": seed,
        "device": str(device),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "train_names": train_names,
        "train_names_sha256": names_sha256(train_names),
        "validation_names": val_names,
        "validation_names_sha256": names_sha256(val_names),
        "old_denoiser_metadata": old_metadata,
        "new_denoiser_metadata": new_metadata,
        "production_hbt_metadata": hbt_metadata,
        "model_config": model.config(),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_recall_at_1": best_recall,
        "checkpoint": str(output),
        "checkpoint_sha256": sha256_file(output),
        "seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "solver_denoiser_adaptation_training_complete",
                "variant": args.variant,
                "checkpoint": str(output),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
