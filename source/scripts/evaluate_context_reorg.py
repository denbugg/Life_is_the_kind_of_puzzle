#!/usr/bin/env python3
"""Leakage-safe exact/real gate for contextual iterative reorganization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from puzzle_assembly.context_reorg import (
    IterativeReorganizationResult,
    QAPSeedResult,
    build_hbt_qap_seed,
    extract_context_reorg_features,
    iterative_reorganization,
    load_context_reorg_checkpoint,
)
from puzzle_assembly.geometry import TILE_COUNT, inverse_permutation
from puzzle_assembly.learned import (
    load_context_position_checkpoint,
    load_embedding_checkpoint,
)
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


@dataclass(frozen=True)
class FrozenPrediction:
    name: str
    denoised_tiles: np.ndarray
    qap: QAPSeedResult
    reorganization: IterativeReorganizationResult
    timings: dict[str, float]
    frozen_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--context-checkpoint")
    parser.add_argument("--reorg-checkpoint", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--exact-split", default="edge_development")
    parser.add_argument("--exact-offset", type=int, default=8)
    parser.add_argument("--exact-sources", type=int, default=8)
    parser.add_argument(
        "--exact-panel",
        choices=["primary_kornia", "independent_libjpeg"],
        default="primary_kornia",
    )
    parser.add_argument("--real-split", default="assembly_cal")
    parser.add_argument("--real-offset", type=int, default=0)
    parser.add_argument("--real-sources", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--qap-iterations", type=int, default=25)
    parser.add_argument("--qap-restarts", type=int, default=2)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.05)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--min-real-ssim-delta", type=float, default=0.02)
    parser.add_argument("--min-exact-wrong-reduction", type=float, default=0.10)
    parser.add_argument("--output", required=True)
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


def _layout_sha256(layout: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(layout, dtype=np.int32).tobytes()).hexdigest()


def _qap_diagnostics(result: QAPSeedResult) -> dict[str, object]:
    return {
        "layout_sha256": _layout_sha256(result.position_to_slot),
        "soft_cycle_layout_sha256": _layout_sha256(
            result.soft_cycle_position_to_slot
        ),
        "score_name": result.score_name,
        "soft_cycle_accepted_edges": result.soft_cycle_accepted_edges,
        "soft_cycle_proposed_edges": result.soft_cycle_proposed_edges,
        "soft_cycle_component_sizes": list(result.soft_cycle_component_sizes),
        "objective": result.qap_objective,
        "relaxed_objective": result.qap_relaxed_objective,
        "restart": result.qap_restart,
        "iterations": result.qap_iterations,
        "converged": result.qap_converged,
        "history": list(result.qap_history),
    }


def _reorg_diagnostics(result: IterativeReorganizationResult) -> dict[str, object]:
    return {
        "layout_sha256": _layout_sha256(result.position_to_slot),
        "round_layout_sha256": [
            _layout_sha256(layout) for layout in result.round_layouts
        ],
        "assigned_mean_logits": list(result.assigned_mean_logits),
        "moved_positions": list(result.moved_positions),
        "rounds_completed": len(result.round_layouts),
        "converged": result.converged,
        "cycle_detected": result.cycle_detected,
    }


def _freeze(values: np.ndarray) -> np.ndarray:
    frozen = np.asarray(values).copy()
    frozen.setflags(write=False)
    return frozen


@torch.inference_mode()
def _predict_input_only(
    name: str,
    raw_tiles: np.ndarray,
    *,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    context_model: torch.nn.Module | None,
    reorg_model: torch.nn.Module,
    device: torch.device,
    process_started: float,
) -> FrozenPrediction:
    """Freeze both layouts without accepting a target path or target pixels."""
    started = time.perf_counter()
    denoise_started = time.perf_counter()
    denoised = restore_tiles_uint8(
        restorer,
        raw_tiles,
        device,
        batch_size=args.denoise_batch_size,
    )
    denoise_seconds = time.perf_counter() - denoise_started
    feature_started = time.perf_counter()
    features = extract_context_reorg_features(
        raw_tiles,
        denoised,
        embedding_model=embedding_model,
        context_model=context_model,
        device=device,
    )
    feature_seconds = time.perf_counter() - feature_started
    qap_started = time.perf_counter()
    qap = build_hbt_qap_seed(
        raw_tiles,
        denoised,
        embedding_model=embedding_model,
        device=device,
        seed=int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
        )
        + 7001,
        chunk_size=args.chunk_size,
        qap_iterations=args.qap_iterations,
        qap_restarts=args.qap_restarts,
        qap_boundary_weight=args.qap_boundary_weight,
        qap_refine_swaps=args.qap_refine_swaps,
    )
    qap_seconds = time.perf_counter() - qap_started
    reorg_started = time.perf_counter()
    reorganization = iterative_reorganization(
        reorg_model,
        features,
        qap.position_to_slot,
        device=device,
        rounds=args.rounds,
    )
    reorg_seconds = time.perf_counter() - reorg_started
    frozen_qap = QAPSeedResult(
        position_to_slot=_freeze(qap.position_to_slot),
        soft_cycle_position_to_slot=_freeze(qap.soft_cycle_position_to_slot),
        score_name=qap.score_name,
        soft_cycle_accepted_edges=qap.soft_cycle_accepted_edges,
        soft_cycle_proposed_edges=qap.soft_cycle_proposed_edges,
        soft_cycle_component_sizes=qap.soft_cycle_component_sizes,
        qap_objective=qap.qap_objective,
        qap_relaxed_objective=qap.qap_relaxed_objective,
        qap_restart=qap.qap_restart,
        qap_iterations=qap.qap_iterations,
        qap_converged=qap.qap_converged,
        qap_history=qap.qap_history,
    )
    frozen_reorg = IterativeReorganizationResult(
        position_to_slot=_freeze(reorganization.position_to_slot),
        round_layouts=tuple(_freeze(layout) for layout in reorganization.round_layouts),
        assigned_mean_logits=reorganization.assigned_mean_logits,
        moved_positions=reorganization.moved_positions,
        converged=reorganization.converged,
        cycle_detected=reorganization.cycle_detected,
    )
    return FrozenPrediction(
        name=name,
        denoised_tiles=_freeze(denoised),
        qap=frozen_qap,
        reorganization=frozen_reorg,
        timings={
            "denoise_seconds": denoise_seconds,
            "feature_seconds": feature_seconds,
            "qap_seconds": qap_seconds,
            "reorganization_seconds": reorg_seconds,
            "total_prediction_seconds": time.perf_counter() - started,
        },
        frozen_at=time.perf_counter() - process_started,
    )


def _mean_metrics(records: list[dict], key: str) -> dict[str, float]:
    if not records:
        return {}
    metric_names = records[0][key]
    return {
        metric: float(np.mean([record[key][metric] for record in records]))
        for metric in metric_names
        if isinstance(records[0][key][metric], (int, float))
        and not isinstance(records[0][key][metric], bool)
    }


def main() -> None:
    args = parse_args()
    if args.exact_sources < 0 or args.real_sources < 0:
        raise SystemExit("source counts must be non-negative")
    if args.exact_sources + args.real_sources <= 0:
        raise SystemExit("at least one exact or real source is required")
    if min(args.rounds, args.denoise_batch_size, args.chunk_size) <= 0:
        raise SystemExit("rounds and batch/chunk sizes must be positive")
    if (
        args.qap_iterations < 0
        or args.qap_restarts <= 0
        or args.qap_boundary_weight < 0.0
    ):
        raise SystemExit("invalid QAP settings")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")

    exact_all = source_names_for_split(
        args.exact_split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    real_all = source_names_for_split(
        args.real_split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )
    exact_names = exact_all[
        args.exact_offset : args.exact_offset + args.exact_sources
    ]
    real_names = real_all[args.real_offset : args.real_offset + args.real_sources]
    if len(exact_names) != args.exact_sources or len(real_names) != args.real_sources:
        raise SystemExit("requested source slice extends past its split")
    overlap = sorted(set(exact_names) & set(real_names))
    if overlap:
        raise SystemExit(f"exact/real whole-source overlap: {overlap[:8]}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    reorg_model, reorg_metadata = load_context_reorg_checkpoint(
        args.reorg_checkpoint, device=device
    )
    if bool(context_model is not None) != bool(reorg_model.has_context_prior):
        raise SystemExit(
            "context checkpoint presence does not match reorg model feature schema"
        )
    if args.rounds > reorg_model.max_rounds:
        raise SystemExit("rounds exceed the reorg checkpoint max_rounds")
    frozen_models = [restorer, embedding_model, reorg_model]
    if context_model is not None:
        frozen_models.append(context_model)
    for model in frozen_models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    process_started = time.perf_counter()
    exact_records = []
    for index, name in enumerate(exact_names):
        source_started = time.perf_counter()
        target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
        panel_seed = per_source_seed(
            args.seed, "context-reorg-gate-exact-panel", name, 0
        )
        panel = make_exact_panel(target, panel=args.exact_panel, seed=panel_seed)
        prediction = _predict_input_only(
            name,
            panel.slot_tiles,
            args=args,
            restorer=restorer,
            embedding_model=embedding_model,
            context_model=context_model,
            reorg_model=reorg_model,
            device=device,
            process_started=process_started,
        )
        seed_layout = layout_metrics(
            prediction.qap.position_to_slot, panel.slot_to_target
        )
        final_layout = layout_metrics(
            prediction.reorganization.position_to_slot, panel.slot_to_target
        )
        seed_wrong = int(
            np.sum(
                prediction.qap.position_to_slot
                != inverse_permutation(panel.slot_to_target)
            )
        )
        final_wrong = int(
            np.sum(
                prediction.reorganization.position_to_slot
                != inverse_permutation(panel.slot_to_target)
            )
        )
        record = {
            "source": name,
            "panel_seed": panel_seed,
            "qap": _qap_diagnostics(prediction.qap),
            "reorganization": _reorg_diagnostics(prediction.reorganization),
            "seed_layout": seed_layout,
            "final_layout": final_layout,
            "seed_image": predicted_image_metrics(
                prediction.qap.position_to_slot,
                prediction.denoised_tiles,
                target,
            ),
            "final_image": predicted_image_metrics(
                prediction.reorganization.position_to_slot,
                prediction.denoised_tiles,
                target,
            ),
            "seed_wrong_positions": seed_wrong,
            "final_wrong_positions": final_wrong,
            "wrong_position_reduction": float(
                (seed_wrong - final_wrong) / max(seed_wrong, 1)
            ),
            "timings": prediction.timings,
            "seconds": time.perf_counter() - source_started,
        }
        exact_records.append(record)
        print(
            json.dumps(
                {
                    "event": "context_reorg_exact_complete",
                    "index": index + 1,
                    "count": len(exact_names),
                    "source": name,
                    "seed_wrong": seed_wrong,
                    "final_wrong": final_wrong,
                    "ssim_delta": record["final_image"]["predicted_layout_ssim"]
                    - record["seed_image"]["predicted_layout_ssim"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Strict two-phase real protocol: freeze all input-only layouts first, then
    # and only then open any corresponding clean target.
    real_predictions: list[FrozenPrediction] = []
    for index, name in enumerate(real_names):
        input_image = _read_rgb(Path(args.data_root) / "train" / "inputs" / name)
        prediction = _predict_input_only(
            name,
            split_tiles_numpy(input_image),
            args=args,
            restorer=restorer,
            embedding_model=embedding_model,
            context_model=context_model,
            reorg_model=reorg_model,
            device=device,
            process_started=process_started,
        )
        real_predictions.append(prediction)
        print(
            json.dumps(
                {
                    "event": "context_reorg_real_layout_frozen",
                    "index": index + 1,
                    "count": len(real_names),
                    "source": name,
                    "qap_layout_sha256": _layout_sha256(
                        prediction.qap.position_to_slot
                    ),
                    "final_layout_sha256": _layout_sha256(
                        prediction.reorganization.position_to_slot
                    ),
                    "frozen_at": prediction.frozen_at,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    all_real_layouts_frozen_at = time.perf_counter() - process_started
    real_records = []
    for prediction in real_predictions:
        target_read_at = time.perf_counter() - process_started
        target = _read_rgb(
            Path(args.data_root) / "train" / "targets" / prediction.name
        )
        seed_image = predicted_image_metrics(
            prediction.qap.position_to_slot,
            prediction.denoised_tiles,
            target,
        )
        final_image = predicted_image_metrics(
            prediction.reorganization.position_to_slot,
            prediction.denoised_tiles,
            target,
        )
        record = {
            "source": prediction.name,
            "qap": _qap_diagnostics(prediction.qap),
            "reorganization": _reorg_diagnostics(prediction.reorganization),
            "seed_image": seed_image,
            "final_image": final_image,
            "ssim_delta": final_image["predicted_layout_ssim"]
            - seed_image["predicted_layout_ssim"],
            "timings": prediction.timings,
            "layout_frozen_at": prediction.frozen_at,
            "target_read_at": target_read_at,
            "target_read_after_all_real_layouts_frozen": bool(
                target_read_at >= all_real_layouts_frozen_at
            ),
        }
        real_records.append(record)

    exact_macro: dict[str, object] = {}
    if exact_records:
        seed_wrong_total = sum(
            record["seed_wrong_positions"] for record in exact_records
        )
        final_wrong_total = sum(
            record["final_wrong_positions"] for record in exact_records
        )
        exact_macro = {
            "seed_layout": _mean_metrics(exact_records, "seed_layout"),
            "final_layout": _mean_metrics(exact_records, "final_layout"),
            "seed_image": _mean_metrics(exact_records, "seed_image"),
            "final_image": _mean_metrics(exact_records, "final_image"),
            "seed_wrong_positions": seed_wrong_total,
            "final_wrong_positions": final_wrong_total,
            "wrong_position_reduction": float(
                (seed_wrong_total - final_wrong_total) / max(seed_wrong_total, 1)
            ),
        }
    real_macro: dict[str, object] = {}
    if real_records:
        seed_image = _mean_metrics(real_records, "seed_image")
        final_image = _mean_metrics(real_records, "final_image")
        real_macro = {
            "seed_image": seed_image,
            "final_image": final_image,
            "ssim_delta": final_image["predicted_layout_ssim"]
            - seed_image["predicted_layout_ssim"],
            "all_targets_read_after_all_layouts_frozen": all(
                record["target_read_after_all_real_layouts_frozen"]
                for record in real_records
            ),
        }
    complete_gate = bool(exact_records and real_records)
    promotion = {
        "complete": complete_gate,
        "exact_wrong_position_reduction_threshold": args.min_exact_wrong_reduction,
        "real_ssim_delta_threshold": args.min_real_ssim_delta,
        "exact_pass": bool(
            exact_records
            and exact_macro["wrong_position_reduction"]
            >= args.min_exact_wrong_reduction
        ),
        "real_pass": bool(
            real_records
            and real_macro["ssim_delta"] >= args.min_real_ssim_delta
        ),
    }
    promotion["promote"] = bool(
        promotion["complete"] and promotion["exact_pass"] and promotion["real_pass"]
    )

    payload = {
        "schema_version": 1,
        "kind": "puzzle_context_reorganization_r0_gate_report",
        "args": vars(args),
        "device": str(device),
        "anti_leakage": {
            "real_prediction_function_accepts_target": False,
            "real_layouts_frozen_before_any_target_read": bool(
                not real_records
                or real_macro["all_targets_read_after_all_layouts_frozen"]
            ),
            "all_real_layouts_frozen_at": all_real_layouts_frozen_at,
            "whole_source_exact_real_intersection": overlap,
        },
        "source_names": {
            "exact": exact_names,
            "real": real_names,
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
        "reorg_metadata": reorg_metadata,
        "reorg_model_config": reorg_model.config(),
        "reorg_checkpoint_sha256": _sha256(args.reorg_checkpoint),
        "qap_config": {
            "score": "denoised_C1_L1w4_rank_fusion",
            "soft_cycle_score": "denoised_l1_embedding",
            "soft_cycle_top_k": 8,
            "soft_cycle_keep_fraction": 0.5,
            "iterations": args.qap_iterations,
            "restarts": args.qap_restarts,
            "boundary_weight": args.qap_boundary_weight,
            "refine_swaps": args.qap_refine_swaps,
            "seed_formula": "filename_sha256_first4_le + 7001",
        },
        "exact": {
            "split": args.exact_split,
            "offset": args.exact_offset,
            "panel": args.exact_panel,
            "records": exact_records,
            "macro": exact_macro,
        },
        "real": {
            "split": args.real_split,
            "offset": args.real_offset,
            "records": real_records,
            "macro": real_macro,
        },
        "promotion": promotion,
        "seconds": time.perf_counter() - process_started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "context_reorg_gate_complete",
                "output": str(output),
                "promotion": promotion,
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
