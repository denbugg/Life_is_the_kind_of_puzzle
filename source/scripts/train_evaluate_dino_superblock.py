#!/usr/bin/env python3
"""Train and gate a frozen-DINOv2 4x4-superblock global position probe."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.dino_superblock import (
    DinoSetPositionHead,
    FrozenDinoFeatureAdapter,
    SUPERBLOCK_COUNT,
    apply_superblock_mapping,
    coarse_assignment_metrics,
    hungarian_mapping,
    layout_pair_cost,
    layout_superblocks,
    oracle_superblock_mapping,
    position_tiles_to_superblocks,
    save_superblock_checkpoint,
    seam_guarded_layout,
    state_dict_sha256,
    synthetic_smoke,
    wrong_position_count,
)
from puzzle_assembly.geometry import TILE_COUNT
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


AUTHORITATIVE_REAL16_SSIM = 0.18281991502795386
AUTHORITATIVE_REPORT_SHA256 = (
    "cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60"
)
AUTHORITATIVE_VARIANT = "qap_softcycle_l1_k8__denoised_render"
DINO_HUB_REPOSITORY = "facebookresearch/dinov2"
DINO_HUB_MODEL = "dinov2_vits14"
DINO_HF_MODEL = "facebook/dinov2-small"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class RuntimeBudgetExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("runtime cap must be positive")
        self.started = time.perf_counter()
        self.seconds = float(seconds)
        self.stage = "initialization"

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float:
        return self.seconds - self.elapsed

    def ensure(self, stage: str, *, reserve: float = 0.0) -> None:
        self.stage = stage
        if self.remaining <= reserve:
            raise RuntimeBudgetExceeded(
                f"runtime cap reached before {stage}: "
                f"remaining={self.remaining:.1f}s reserve={reserve:.1f}s"
            )


@dataclass(frozen=True)
class QAPBundle:
    layout: np.ndarray
    score: CompatibilityMatrices
    diagnostics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument(
        "--manifest", default="configs/denoise_splits_seed20260710.json"
    )
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--authoritative-reference", required=True)
    parser.add_argument("--train-sources", type=int, default=512)
    parser.add_argument("--dev-sources", type=int, default=64)
    parser.add_argument("--exact-sources", type=int, default=8)
    parser.add_argument("--real-sources", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--feature-source-batch", type=int, default=4)
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--dino-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=16)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--column-regularizer", type=float, default=0.10)
    parser.add_argument("--min-corruption", type=float, default=0.10)
    parser.add_argument("--max-corruption", type=float, default=0.35)
    parser.add_argument("--dev-min-cell-accuracy", type=float, default=0.10)
    parser.add_argument("--dev-min-manhattan-reduction", type=float, default=0.25)
    parser.add_argument("--seam-max-ratio", type=float, default=1.02)
    parser.add_argument("--promotion-min-ssim", type=float, default=0.010)
    parser.add_argument("--promotion-min-wins", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--runtime-cap-seconds", type=float, default=2520.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _name_seed(name: str) -> int:
    base = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
    )
    return base + 7001


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _hub_checkpoint_candidates() -> list[Path]:
    roots = [Path(torch.hub.get_dir()) / "checkpoints", Path("/kaggle/input")]
    values: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        values.extend(root.glob("**/dinov2_vits14*pretrain*.pth"))
        values.extend(root.glob("**/dinov2_vits14*.pth"))
    return sorted(set(path.resolve() for path in values if path.is_file()))


def _load_dino(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    errors: dict[str, str] = {}
    backend: str
    identifier: str
    backbone: nn.Module
    try:
        backbone = torch.hub.load(
            DINO_HUB_REPOSITORY,
            DINO_HUB_MODEL,
            pretrained=True,
            trust_repo=True,
            force_reload=False,
            verbose=True,
        )
        backend = "torch_hub"
        identifier = f"{DINO_HUB_REPOSITORY}:{DINO_HUB_MODEL}"
    except Exception as exc:
        errors["torch_hub"] = f"{type(exc).__name__}: {exc}"
        try:
            from transformers import AutoModel, __version__ as transformers_version

            backbone = AutoModel.from_pretrained(
                DINO_HF_MODEL, low_cpu_mem_usage=False
            )
            backend = "transformers"
            identifier = DINO_HF_MODEL
            errors["transformers_version"] = transformers_version
        except Exception as fallback_exc:
            errors["transformers"] = (
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
            raise RuntimeError(
                "unable to load official DINOv2 ViT-S/14 via torch.hub or "
                f"the facebook Hugging Face fallback: {errors}"
            ) from fallback_exc
    _freeze(backbone)
    backbone.to(device)
    state_hash = state_dict_sha256(backbone)
    adapter = FrozenDinoFeatureAdapter(backbone, backend=backend).to(device)
    _freeze(adapter)
    checkpoint_candidates = _hub_checkpoint_candidates()
    candidate_records = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in checkpoint_candidates
    ]
    return adapter, {
        "architecture": "DINOv2 ViT-S/14",
        "backend": backend,
        "identifier": identifier,
        "state_dict_sha256": state_hash,
        "frozen_parameter_count": int(
            sum(parameter.numel() for parameter in backbone.parameters())
        ),
        "fallback_diagnostics": errors,
        "cache_checkpoint_candidates": candidate_records,
    }


@torch.inference_mode()
def _dino_features(
    model: nn.Module,
    blocks: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(blocks)
    if values.ndim != 4 or values.shape[1:] != (80, 80, 3):
        raise ValueError("DINO blocks must be Nx80x80x3")
    if values.dtype != np.uint8:
        raise TypeError("DINO blocks must be uint8")
    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(
            np.ascontiguousarray(values[start : start + batch_size].transpose(0, 3, 1, 2))
        ).to(device=device, dtype=torch.float32).div_(255.0)
        tensor = F.interpolate(
            tensor, size=(224, 224), mode="bicubic", align_corners=False, antialias=True
        )
        tensor = (tensor - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            features = model(tensor)
        outputs.append(features.detach().float().cpu().numpy())
    result = np.concatenate(outputs)
    if result.ndim != 2 or len(result) != len(values):
        raise RuntimeError(f"unexpected DINO feature shape {result.shape}")
    return result.astype(np.float32, copy=False)


def _ordered_restored_tiles(panel: Any, restored_slots: np.ndarray) -> np.ndarray:
    ordered = np.empty_like(restored_slots)
    ordered[panel.slot_to_target] = restored_slots
    return ordered


def _corrupt_ordered_tiles(
    ordered: np.ndarray,
    *,
    seed: int,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, float]:
    if not 0.0 <= minimum <= maximum < 1.0:
        raise ValueError("corruption bounds must satisfy 0 <= min <= max < 1")
    rng = np.random.default_rng(seed)
    fraction = float(rng.uniform(minimum, maximum))
    count = max(2, int(round(TILE_COUNT * fraction)))
    indices = rng.choice(TILE_COUNT, size=count, replace=False)
    permuted = indices[rng.permutation(count)]
    if np.array_equal(indices, permuted):
        permuted = np.roll(indices, 1)
    output = ordered.copy()
    output[indices] = ordered[permuted]
    return output, fraction


def _extract_training_features(
    names: list[str],
    *,
    args: argparse.Namespace,
    data_root: Path,
    restorer: nn.Module,
    dino: nn.Module,
    device: torch.device,
    deadline: Deadline,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    feature_sets: list[np.ndarray] = []
    label_sets: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for start in range(0, len(names), args.feature_source_batch):
        deadline.ensure("training feature extraction", reserve=120.0)
        chunk = names[start : start + args.feature_source_batch]
        panels = []
        for name in chunk:
            panel_seed = per_source_seed(args.seed, "dino-superblock-train-panel", name)
            target = _read_rgb(data_root / "train" / "targets" / name)
            panels.append(
                make_exact_panel(target, panel="primary_kornia", seed=panel_seed)
            )
        combined = np.concatenate([panel.slot_tiles for panel in panels])
        restored = restore_tiles_uint8(
            restorer, combined, device, batch_size=args.denoise_batch_size
        )
        block_sets = []
        for index, (name, panel) in enumerate(zip(chunk, panels, strict=True)):
            source_restored = restored[
                index * TILE_COUNT : (index + 1) * TILE_COUNT
            ]
            ordered = _ordered_restored_tiles(panel, source_restored)
            corruption_seed = per_source_seed(
                args.seed, "dino-superblock-train-corruption", name
            )
            corrupted, fraction = _corrupt_ordered_tiles(
                ordered,
                seed=corruption_seed,
                minimum=args.min_corruption,
                maximum=args.max_corruption,
            )
            block_sets.append(position_tiles_to_superblocks(corrupted))
            records.append(
                {
                    "source": name,
                    "panel_seed": int(panel.seed),
                    "corruption_seed": int(corruption_seed),
                    "corruption_fraction": fraction,
                }
            )
        features = _dino_features(
            dino,
            np.concatenate(block_sets),
            device=device,
            batch_size=args.dino_batch_size,
        ).reshape(len(chunk), SUPERBLOCK_COUNT, -1)
        feature_sets.append(features.astype(np.float16))
        label_sets.append(
            np.tile(
                np.arange(SUPERBLOCK_COUNT, dtype=np.int64),
                (len(chunk), 1),
            )
        )
        print(
            json.dumps(
                {
                    "event": "dino_training_features",
                    "completed_sources": min(start + len(chunk), len(names)),
                    "total_sources": len(names),
                    "feature_dim": int(features.shape[2]),
                    "elapsed": deadline.elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return np.concatenate(feature_sets), np.concatenate(label_sets), records


def _train_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    deadline: Deadline,
) -> tuple[DinoSetPositionHead, list[dict[str, float]]]:
    if features.shape[:2] != labels.shape or features.shape[1] != SUPERBLOCK_COUNT:
        raise ValueError("training feature and label shapes disagree")
    model = DinoSetPositionHead(
        feature_dim=int(features.shape[2]),
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        deadline.ensure("set-head training", reserve=90.0)
        model.train()
        source_order = rng.permutation(len(features))
        losses = []
        cell_losses = []
        column_losses = []
        token_accuracies = []
        for start in range(0, len(source_order), args.head_batch_size):
            indices = source_order[start : start + args.head_batch_size]
            batch_features = torch.from_numpy(features[indices].astype(np.float32)).to(device)
            batch_labels = torch.from_numpy(labels[indices]).to(device)
            token_noise = torch.rand(
                (len(indices), SUPERBLOCK_COUNT), device=device
            )
            token_order = token_noise.argsort(dim=1)
            batch_features = torch.gather(
                batch_features,
                1,
                token_order.unsqueeze(2).expand(-1, -1, batch_features.shape[2]),
            )
            batch_labels = torch.gather(batch_labels, 1, token_order)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            cell_loss = F.cross_entropy(
                logits.reshape(-1, SUPERBLOCK_COUNT), batch_labels.reshape(-1)
            )
            column_mass = logits.softmax(dim=2).sum(dim=1)
            column_loss = (column_mass - 1.0).square().mean()
            loss = cell_loss + args.column_regularizer * column_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            cell_losses.append(float(cell_loss.detach().cpu()))
            column_losses.append(float(column_loss.detach().cpu()))
            token_accuracies.append(
                float((logits.argmax(dim=2) == batch_labels).float().mean().cpu())
            )
        record = {
            "epoch": float(epoch + 1),
            "loss": float(np.mean(losses)),
            "cell_loss": float(np.mean(cell_losses)),
            "column_loss": float(np.mean(column_losses)),
            "token_argmax_accuracy": float(np.mean(token_accuracies)),
        }
        history.append(record)
        print(json.dumps({"event": "dino_head_epoch", **record}, sort_keys=True), flush=True)
    model.eval()
    return model, history


def _promoted_qap(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    embedding_model: nn.Module,
    device: torch.device,
    source_name: str,
) -> QAPBundle:
    bank = build_classical_score_bank(
        denoised_tiles, prefix="denoised", chunk_size=64
    )
    c1_names = [
        name
        for name in sorted(bank)
        if name.startswith("denoised_") and not name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(
        bank, names=c1_names, name="denoised_C1_equal_rank_fusion"
    )
    bank[c1.name] = c1
    l1, _ = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_l1_embedding",
    )
    bank[l1.name] = l1
    score = fuse_ranked_scores(
        bank,
        names=[c1.name, l1.name],
        weights={l1.name: 4.0},
        name="denoised_C1_L1w4_rank_fusion",
    )
    soft = soft_cycle_component_solver(
        l1,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        reciprocal_weight=0.35,
        loop_weight=1.0,
    )
    qap = directional_qap(
        score,
        initial=soft.position_to_slot,
        iterations=25,
        restarts=2,
        seed=_name_seed(source_name),
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
    )
    return QAPBundle(
        layout=qap.position_to_slot,
        score=score,
        diagnostics={
            "score_name": score.name,
            "seed": _name_seed(source_name),
            "soft_cycle_accepted_edges": int(soft.accepted_edges),
            "soft_cycle_proposed_edges": int(soft.proposed_edges),
            "soft_cycle_component_sizes": [int(value) for value in soft.component_sizes],
            "qap_objective": float(qap.objective),
            "qap_relaxed_objective": float(qap.relaxed_objective),
            "qap_restart": int(qap.restart),
            "qap_iterations": int(qap.iterations),
            "qap_converged": bool(qap.converged),
        },
    )


@torch.inference_mode()
def _head_mapping(
    head: DinoSetPositionHead,
    dino: nn.Module,
    blocks: np.ndarray,
    *,
    device: torch.device,
    dino_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = _dino_features(
        dino, blocks, device=device, batch_size=dino_batch_size
    )
    logits = head(torch.from_numpy(features).to(device).unsqueeze(0)).squeeze(0)
    return hungarian_mapping(logits), features


def _evaluate_exact_source(
    name: str,
    *,
    panel_name: str,
    stage: str,
    args: argparse.Namespace,
    data_root: Path,
    restorer: nn.Module,
    embedding_model: nn.Module,
    dino: nn.Module,
    head: DinoSetPositionHead,
    device: torch.device,
    deadline: Deadline,
) -> dict[str, Any]:
    deadline.ensure(f"{stage} source {name}", reserve=90.0)
    target = _read_rgb(data_root / "train" / "targets" / name)
    panel_seed = per_source_seed(args.seed, f"dino-superblock-{stage}", name)
    panel = make_exact_panel(target, panel=panel_name, seed=panel_seed)
    denoised = restore_tiles_uint8(
        restorer, panel.slot_tiles, device, batch_size=args.denoise_batch_size
    )
    qap = _promoted_qap(
        panel.slot_tiles,
        denoised,
        embedding_model=embedding_model,
        device=device,
        source_name=name,
    )
    predicted_mapping, features = _head_mapping(
        head,
        dino,
        layout_superblocks(denoised, qap.layout),
        device=device,
        dino_batch_size=args.dino_batch_size,
    )
    oracle = oracle_superblock_mapping(qap.layout, panel.slot_to_target)
    coarse = coarse_assignment_metrics(
        predicted_mapping, oracle.source_to_destination
    )
    candidate = apply_superblock_mapping(qap.layout, predicted_mapping)
    oracle_layout = apply_superblock_mapping(
        qap.layout, oracle.source_to_destination
    )
    baseline_wrong = wrong_position_count(qap.layout, panel.slot_to_target)
    candidate_wrong = wrong_position_count(candidate, panel.slot_to_target)
    oracle_wrong = wrong_position_count(oracle_layout, panel.slot_to_target)
    baseline_cost = layout_pair_cost(qap.layout, qap.score)
    candidate_cost = layout_pair_cost(candidate, qap.score)
    return {
        "source": name,
        "panel": panel_name,
        "panel_seed": int(panel_seed),
        "coarse": coarse,
        "oracle_attainable_tile_fraction": oracle.attainable_tile_fraction,
        "baseline_wrong_positions": baseline_wrong,
        "candidate_wrong_positions": candidate_wrong,
        "oracle_wrong_positions": oracle_wrong,
        "wrong_position_reduction": (
            (baseline_wrong - candidate_wrong) / baseline_wrong
            if baseline_wrong
            else 0.0
        ),
        "baseline_layout_metrics": layout_metrics(qap.layout, panel.slot_to_target),
        "candidate_layout_metrics": layout_metrics(candidate, panel.slot_to_target),
        "baseline_pair_cost": baseline_cost,
        "candidate_pair_cost": candidate_cost,
        "candidate_to_baseline_pair_cost_ratio": candidate_cost
        / max(baseline_cost, 1e-12),
        "baseline_layout_sha256": _array_sha256(qap.layout),
        "candidate_layout_sha256": _array_sha256(candidate),
        "predicted_mapping_sha256": _array_sha256(predicted_mapping),
        "feature_sha256": _array_sha256(features.astype(np.float16)),
        "qap": qap.diagnostics,
    }


def _aggregate_exact(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty exact record list")
    source_count = len(records)
    predicted_accuracy = float(
        np.mean([record["coarse"]["predicted_cell_accuracy"] for record in records])
    )
    baseline_accuracy = float(
        np.mean([record["coarse"]["baseline_cell_accuracy"] for record in records])
    )
    predicted_manhattan = float(
        np.mean([record["coarse"]["predicted_mean_manhattan"] for record in records])
    )
    baseline_manhattan = float(
        np.mean([record["coarse"]["baseline_mean_manhattan"] for record in records])
    )
    baseline_wrong = int(sum(record["baseline_wrong_positions"] for record in records))
    candidate_wrong = int(sum(record["candidate_wrong_positions"] for record in records))
    oracle_wrong = int(sum(record["oracle_wrong_positions"] for record in records))
    return {
        "source_count": source_count,
        "predicted_cell_accuracy": predicted_accuracy,
        "baseline_cell_accuracy": baseline_accuracy,
        "predicted_mean_manhattan": predicted_manhattan,
        "baseline_mean_manhattan": baseline_manhattan,
        "manhattan_reduction": (
            (baseline_manhattan - predicted_manhattan) / baseline_manhattan
            if baseline_manhattan
            else 0.0
        ),
        "baseline_wrong_positions": baseline_wrong,
        "candidate_wrong_positions": candidate_wrong,
        "oracle_wrong_positions": oracle_wrong,
        "wrong_position_reduction": (
            (baseline_wrong - candidate_wrong) / baseline_wrong
            if baseline_wrong
            else 0.0
        ),
        "mean_oracle_attainable_tile_fraction": float(
            np.mean(
                [record["oracle_attainable_tile_fraction"] for record in records]
            )
        ),
        "mean_candidate_to_baseline_pair_cost_ratio": float(
            np.mean(
                [
                    record["candidate_to_baseline_pair_cost_ratio"]
                    for record in records
                ]
            )
        ),
    }


def _evaluate_exact_set(
    names: list[str],
    panels: list[str],
    *,
    stage: str,
    args: argparse.Namespace,
    data_root: Path,
    restorer: nn.Module,
    embedding_model: nn.Module,
    dino: nn.Module,
    head: DinoSetPositionHead,
    device: torch.device,
    deadline: Deadline,
) -> dict[str, Any]:
    if len(names) != len(panels):
        raise ValueError("exact source and panel lists differ")
    records = []
    for index, (name, panel) in enumerate(zip(names, panels, strict=True)):
        record = _evaluate_exact_source(
            name,
            panel_name=panel,
            stage=stage,
            args=args,
            data_root=data_root,
            restorer=restorer,
            embedding_model=embedding_model,
            dino=dino,
            head=head,
            device=device,
            deadline=deadline,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": f"dino_superblock_{stage}_source",
                    "index": index + 1,
                    "count": len(names),
                    "source": name,
                    "coarse": record["coarse"],
                    "wrong_position_reduction": record[
                        "wrong_position_reduction"
                    ],
                    "elapsed": deadline.elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "source_names": names,
        "panels": panels,
        "macro": _aggregate_exact(records),
        "sources": records,
    }


def _predict_real_input_only(
    name: str,
    *,
    args: argparse.Namespace,
    data_root: Path,
    restorer: nn.Module,
    embedding_model: nn.Module,
    dino: nn.Module,
    head: DinoSetPositionHead,
    device: torch.device,
    deadline: Deadline,
) -> dict[str, Any]:
    """Predict and freeze layouts without accepting or opening a target path."""
    deadline.ensure(f"real input-only prediction {name}", reserve=90.0)
    input_image = _read_rgb(data_root / "train" / "inputs" / name)
    raw_tiles = split_tiles_numpy(input_image)
    denoised = restore_tiles_uint8(
        restorer, raw_tiles, device, batch_size=args.denoise_batch_size
    )
    qap = _promoted_qap(
        raw_tiles,
        denoised,
        embedding_model=embedding_model,
        device=device,
        source_name=name,
    )
    predicted_mapping, features = _head_mapping(
        head,
        dino,
        layout_superblocks(denoised, qap.layout),
        device=device,
        dino_batch_size=args.dino_batch_size,
    )
    raw_candidate = apply_superblock_mapping(qap.layout, predicted_mapping)
    guarded, guard = seam_guarded_layout(
        qap.layout,
        raw_candidate,
        qap.score,
        max_ratio=args.seam_max_ratio,
    )
    return {
        "source": name,
        "baseline_layout": qap.layout.copy(),
        "raw_candidate_layout": raw_candidate.copy(),
        "selected_layout": guarded.copy(),
        "denoised_tiles": denoised,
        "public": {
            "source": name,
            "baseline_layout_sha256": _array_sha256(qap.layout),
            "raw_candidate_layout_sha256": _array_sha256(raw_candidate),
            "selected_layout_sha256": _array_sha256(guarded),
            "predicted_mapping_sha256": _array_sha256(predicted_mapping),
            "feature_sha256": _array_sha256(features.astype(np.float16)),
            "seam_guard": guard,
            "qap": qap.diagnostics,
        },
    }


def _bootstrap_interval(
    deltas: np.ndarray, *, seed: int, resamples: int
) -> tuple[float, float]:
    if deltas.ndim != 1 or len(deltas) == 0 or resamples <= 0:
        raise ValueError("invalid paired bootstrap inputs")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, len(deltas), size=(count, len(deltas)))
        means[start : start + count] = deltas[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _evaluate_real16(
    names: list[str],
    *,
    args: argparse.Namespace,
    data_root: Path,
    restorer: nn.Module,
    embedding_model: nn.Module,
    dino: nn.Module,
    head: DinoSetPositionHead,
    device: torch.device,
    deadline: Deadline,
) -> dict[str, Any]:
    frozen = []
    for index, name in enumerate(names):
        record = _predict_real_input_only(
            name,
            args=args,
            data_root=data_root,
            restorer=restorer,
            embedding_model=embedding_model,
            dino=dino,
            head=head,
            device=device,
            deadline=deadline,
        )
        frozen.append(record)
        print(
            json.dumps(
                {
                    "event": "dino_superblock_real_layout_frozen",
                    "index": index + 1,
                    "count": len(names),
                    "source": name,
                    "seam_guard": record["public"]["seam_guard"],
                    "elapsed": deadline.elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    freeze_digest = hashlib.sha256()
    for record in frozen:
        freeze_digest.update(record["source"].encode("utf-8"))
        freeze_digest.update(record["baseline_layout"].tobytes())
        freeze_digest.update(record["selected_layout"].tobytes())
    freeze_sha256 = freeze_digest.hexdigest()
    freeze_elapsed = deadline.elapsed

    # Target access begins only after every real layout above is frozen.
    scored = []
    for record in frozen:
        target = _read_rgb(data_root / "train" / "targets" / record["source"])
        baseline = predicted_image_metrics(
            record["baseline_layout"], record["denoised_tiles"], target
        )
        candidate = predicted_image_metrics(
            record["selected_layout"], record["denoised_tiles"], target
        )
        scored.append(
            {
                **record["public"],
                "baseline_metrics": baseline,
                "candidate_metrics": candidate,
                "ssim_delta": candidate["predicted_layout_ssim"]
                - baseline["predicted_layout_ssim"],
            }
        )
    baseline_values = np.asarray(
        [record["baseline_metrics"]["predicted_layout_ssim"] for record in scored],
        dtype=np.float64,
    )
    candidate_values = np.asarray(
        [record["candidate_metrics"]["predicted_layout_ssim"] for record in scored],
        dtype=np.float64,
    )
    deltas = candidate_values - baseline_values
    interval = _bootstrap_interval(
        deltas, seed=args.seed + 991, resamples=args.bootstrap_resamples
    )
    return {
        "source_names": names,
        "anti_leakage": {
            "predictor_accepts_target": False,
            "all_layouts_frozen_before_any_target_read": True,
            "freeze_layouts_sha256": freeze_sha256,
            "freeze_elapsed_seconds": freeze_elapsed,
            "target_pixels_used_for_selection": False,
        },
        "macro": {
            "baseline_ssim": float(baseline_values.mean()),
            "candidate_ssim": float(candidate_values.mean()),
            "ssim_delta": float(deltas.mean()),
            "wins": int(np.sum(deltas > 0.0)),
            "ties": int(np.sum(deltas == 0.0)),
            "losses": int(np.sum(deltas < 0.0)),
            "paired_bootstrap_95_ci": list(interval),
            "seam_candidate_accepts": int(
                sum(
                    bool(record["seam_guard"]["candidate_accepted"])
                    for record in scored
                )
            ),
            "max_selected_seam_ratio": float(
                max(
                    1.0
                    if not record["seam_guard"]["candidate_accepted"]
                    else record["seam_guard"][
                        "candidate_to_baseline_ratio"
                    ]
                    for record in scored
                )
            ),
        },
        "sources": scored,
    }


def _load_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") == "puzzle_qap_v2_real16_reference_manifest":
        if payload.get("source_report_sha256") != AUTHORITATIVE_REPORT_SHA256:
            raise RuntimeError("compact reference points to the wrong source report")
        if payload.get("variant") != AUTHORITATIVE_VARIANT:
            raise RuntimeError("compact reference uses the wrong baseline variant")
        source_names = [str(name) for name in payload.get("source_names", [])]
        per_source = {
            str(name): float(value)
            for name, value in payload.get("per_source_ssim", {}).items()
        }
        if len(source_names) != 16 or set(source_names) != set(per_source):
            raise RuntimeError("compact reference must contain the same 16 source metrics")
        baseline = float(payload.get("baseline_ssim"))
        if abs(baseline - float(np.mean([per_source[name] for name in source_names]))) > 1e-12:
            raise RuntimeError("compact reference macro is not its per-source mean")
        payload = {
            "source_names": source_names,
            "macro": {
                AUTHORITATIVE_VARIANT: {"predicted_layout_ssim": baseline}
            },
            "sources": [
                {
                    "source": name,
                    "variants": {
                        AUTHORITATIVE_VARIANT: {
                            "predicted_layout_ssim": per_source[name]
                        }
                    },
                }
                for name in source_names
            ],
            "compact_manifest_sha256": _sha256(path),
            "source_report_sha256": AUTHORITATIVE_REPORT_SHA256,
        }
    elif _sha256(path) != AUTHORITATIVE_REPORT_SHA256:
        raise RuntimeError(
            "authoritative v2 report hash mismatch: "
            f"expected {AUTHORITATIVE_REPORT_SHA256}, got {_sha256(path)}"
        )
    source_names = payload.get("source_names")
    baseline = payload.get("macro", {}).get(AUTHORITATIVE_VARIANT, {}).get(
        "predicted_layout_ssim"
    )
    if len(source_names or []) != 16 or abs(float(baseline) - AUTHORITATIVE_REAL16_SSIM) > 1e-12:
        raise RuntimeError("authoritative v2 report contents do not match constants")
    return payload


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _execute(args: argparse.Namespace, progress: dict[str, Any], deadline: Deadline) -> None:
    data_root = Path(args.data_root)
    manifest = Path(args.manifest)
    quarantine = Path(args.quarantine)
    reference_path = Path(args.authoritative_reference)
    reference = _load_reference(reference_path)
    train_names = source_names_for_split(
        "edge_train", manifest_path=manifest, quarantine_path=quarantine
    )[: args.train_sources]
    development_all = source_names_for_split(
        "edge_development", manifest_path=manifest, quarantine_path=quarantine
    )
    dev_names = development_all[: args.dev_sources]
    exact_names = development_all[
        args.dev_sources : args.dev_sources + args.exact_sources
    ]
    real_expected = source_names_for_split(
        "assembly_cal", manifest_path=manifest, quarantine_path=quarantine
    )[: args.real_sources]
    real_names = [str(name) for name in reference["source_names"]]
    if len(train_names) != args.train_sources or len(dev_names) != args.dev_sources:
        raise RuntimeError("requested source split slice is incomplete")
    if len(exact_names) != args.exact_sources or len(real_names) != args.real_sources:
        raise RuntimeError("exact/real source counts are not fixed 8/16")
    if real_names != real_expected:
        raise RuntimeError("assembly_cal source list differs from authoritative v2 real16")
    split_sets = {
        "train": set(train_names),
        "development": set(dev_names),
        "exact": set(exact_names),
        "real": set(real_names),
    }
    intersections = {
        f"{left}_vs_{right}": sorted(split_sets[left] & split_sets[right])
        for index, left in enumerate(split_sets)
        for right in list(split_sets)[index + 1 :]
    }
    if any(intersections.values()):
        raise RuntimeError(f"whole-source split overlap: {intersections}")
    progress["splits"] = {
        "train": train_names,
        "development": dev_names,
        "exact": exact_names,
        "real": real_names,
        "intersections": intersections,
        "safe": True,
    }
    progress["reference"] = {
        "path": str(reference_path),
        "sha256": _sha256(reference_path),
        "variant": AUTHORITATIVE_VARIANT,
        "baseline_ssim": AUTHORITATIVE_REAL16_SSIM,
    }

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the bounded job requires CUDA")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("the bounded job requires T4x2")

    restorer, resolved_device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device
    )
    if resolved_device != device:
        device = resolved_device
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    _freeze(restorer)
    _freeze(embedding)
    # Keep denoiser/HBT on one device with the authoritative batch size.  The
    # v2 baseline was produced this way; DINO feature extraction below is the
    # only DataParallel stage, so baseline parity does not depend on a changed
    # convolution batch partition.
    deadline.ensure("DINOv2 load", reserve=180.0)
    dino_base, dino_metadata = _load_dino(device)
    if torch.cuda.device_count() >= 2:
        dino: nn.Module = nn.DataParallel(dino_base, device_ids=[0, 1])
    else:
        dino = dino_base
    probe_features = _dino_features(
        dino,
        np.zeros((2, 80, 80, 3), dtype=np.uint8),
        device=device,
        batch_size=2,
    )
    dino_metadata["feature_dim"] = int(probe_features.shape[1])
    progress["frozen_models"] = {
        "denoiser": denoiser_metadata,
        "embedding_checkpoint": str(args.embedding_checkpoint),
        "embedding_checkpoint_sha256": _sha256(args.embedding_checkpoint),
        "embedding_metadata": embedding_metadata,
        "dino": dino_metadata,
        "only_trainable_model": "DinoSetPositionHead",
    }

    features, labels, feature_records = _extract_training_features(
        train_names,
        args=args,
        data_root=data_root,
        restorer=restorer,
        dino=dino,
        device=device,
        deadline=deadline,
    )
    progress["training_features"] = {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "sha256": _array_sha256(features),
        "labels_sha256": _array_sha256(labels),
        "records": feature_records,
    }
    head, history = _train_head(
        features,
        labels,
        args=args,
        device=device,
        deadline=deadline,
    )
    checkpoint = Path(args.output)
    save_superblock_checkpoint(
        checkpoint,
        head,
        metadata={
            "seed": args.seed,
            "train_names": train_names,
            "dino": dino_metadata,
            "training_feature_sha256": _array_sha256(features),
            "history": history,
        },
    )
    progress["training"] = {
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in head.parameters())
        ),
        "frozen_dino_gradients_enabled": bool(
            any(parameter.requires_grad for parameter in dino.parameters())
        ),
    }
    del features, labels

    development = _evaluate_exact_set(
        dev_names,
        ["primary_kornia"] * len(dev_names),
        stage="development",
        args=args,
        data_root=data_root,
        restorer=restorer,
        embedding_model=embedding,
        dino=dino,
        head=head,
        device=device,
        deadline=deadline,
    )
    dev_macro = development["macro"]
    dev_gates = {
        "cell_accuracy_at_least_0_10": dev_macro["predicted_cell_accuracy"]
        >= args.dev_min_cell_accuracy,
        "manhattan_reduction_at_least_0_25": dev_macro["manhattan_reduction"]
        >= args.dev_min_manhattan_reduction,
    }
    development["gates"] = dev_gates
    development["passed"] = bool(all(dev_gates.values()))
    progress["development64"] = development

    exact_panels = [
        "primary_kornia" if index < args.exact_sources // 2 else "independent_libjpeg"
        for index in range(args.exact_sources)
    ]
    progress["exact8"] = _evaluate_exact_set(
        exact_names,
        exact_panels,
        stage="exact_gate",
        args=args,
        data_root=data_root,
        restorer=restorer,
        embedding_model=embedding,
        dino=dino,
        head=head,
        device=device,
        deadline=deadline,
    )

    if not development["passed"]:
        progress["real16"] = {
            "skipped": True,
            "reason": "development kill gate failed before real target access",
            "targets_opened": False,
        }
        progress["promotion"] = {
            "accepted": False,
            "reason": "development kill gate failed",
            "gates": dev_gates,
        }
        progress["status"] = "development_gate_failed"
        return

    deadline.ensure("real16 gate", reserve=360.0)
    real = _evaluate_real16(
        real_names,
        args=args,
        data_root=data_root,
        restorer=restorer,
        embedding_model=embedding,
        dino=dino,
        head=head,
        device=device,
        deadline=deadline,
    )
    macro = real["macro"]
    baseline_delta = macro["baseline_ssim"] - AUTHORITATIVE_REAL16_SSIM
    reference_source_ssim = {
        str(source["source"]): float(
            source["variants"][AUTHORITATIVE_VARIANT]["predicted_layout_ssim"]
        )
        for source in reference["sources"]
    }
    observed_source_ssim = {
        str(source["source"]): float(
            source["baseline_metrics"]["predicted_layout_ssim"]
        )
        for source in real["sources"]
    }
    if set(reference_source_ssim) != set(observed_source_ssim):
        raise RuntimeError("per-source baseline parity lists differ")
    per_source_baseline_delta = {
        name: observed_source_ssim[name] - reference_source_ssim[name]
        for name in real_names
    }
    max_per_source_baseline_delta = max(
        abs(value) for value in per_source_baseline_delta.values()
    )
    real_gates = {
        "development_passed": development["passed"],
        "authoritative_baseline_reproduced_within_1e_6": abs(baseline_delta)
        <= 1e-6
        and max_per_source_baseline_delta <= 1e-6,
        "mean_ssim_gain_at_least_0_010": macro["ssim_delta"]
        >= args.promotion_min_ssim,
        "wins_at_least_10_of_16": macro["wins"] >= args.promotion_min_wins,
        "bootstrap_lower_bound_above_zero": macro[
            "paired_bootstrap_95_ci"
        ][0]
        > 0.0,
        "seam_guard_at_most_1_02": macro["max_selected_seam_ratio"]
        <= args.seam_max_ratio + 1e-12,
        "split_safe": progress["splits"]["safe"],
        "anti_leakage_passed": real["anti_leakage"][
            "all_layouts_frozen_before_any_target_read"
        ],
    }
    real["authoritative_baseline"] = {
        "expected": AUTHORITATIVE_REAL16_SSIM,
        "observed": macro["baseline_ssim"],
        "delta": baseline_delta,
        "per_source_delta": per_source_baseline_delta,
        "max_absolute_per_source_delta": max_per_source_baseline_delta,
        "reference_sha256": AUTHORITATIVE_REPORT_SHA256,
    }
    real["gates"] = real_gates
    real["passed"] = bool(all(real_gates.values()))
    progress["real16"] = real
    progress["promotion"] = {
        "accepted": real["passed"],
        "gates": real_gates,
        "rule": (
            "promote only after dev cell accuracy >=10% and Manhattan reduction "
            ">=25%, authoritative baseline parity, seam ratio <=1.02, real16 "
            "SSIM gain >=0.010, >=10 wins, and paired-bootstrap lower bound >0"
        ),
    }
    progress["status"] = "complete"


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        print(json.dumps({"event": "synthetic_smoke", **synthetic_smoke()}, sort_keys=True))
        return
    if (
        args.train_sources != 512
        or args.dev_sources != 64
        or args.exact_sources != 8
        or args.real_sources != 16
    ):
        raise SystemExit("this bounded gate is fixed to train512/dev64/exact8/real16")
    if not 0.0 < args.dev_min_cell_accuracy <= 1.0:
        raise SystemExit("invalid dev cell accuracy threshold")
    if not 0.0 < args.dev_min_manhattan_reduction <= 1.0:
        raise SystemExit("invalid dev Manhattan threshold")
    output = Path(args.output)
    report_path = Path(args.report)
    if (output.exists() or report_path.exists()) and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    deadline = Deadline(args.runtime_cap_seconds)
    progress: dict[str, Any] = {
        "schema_version": 1,
        "kind": "puzzle_dino_vits14_superblock_probe_gate",
        "args": vars(args),
        "status": "running",
        "synthetic_smoke": synthetic_smoke(),
        "runtime_cap_seconds": args.runtime_cap_seconds,
    }
    try:
        _execute(args, progress, deadline)
    except RuntimeBudgetExceeded as exc:
        progress["status"] = "runtime_cap_exceeded"
        progress["runtime_cap"] = {
            "stage": deadline.stage,
            "message": str(exc),
            "elapsed_seconds": deadline.elapsed,
            "real_targets_opened": bool(
                progress.get("real16", {})
                .get("anti_leakage", {})
                .get("all_layouts_frozen_before_any_target_read", False)
            ),
        }
        progress["promotion"] = {
            "accepted": False,
            "reason": "hard runtime cap reached",
        }
    progress["seconds"] = deadline.elapsed
    progress["accepted"] = bool(
        progress.get("promotion", {}).get("accepted", False)
    )
    _write_report(report_path, progress)
    print(
        json.dumps(
            {
                "event": "dino_superblock_probe_complete",
                "status": progress["status"],
                "accepted": progress["accepted"],
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "seconds": deadline.elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
