#!/usr/bin/env python3
"""Leakage-safe exact/real gate for sparse learned 2x2 hyperedge anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from puzzle_assembly.compatibility import build_classical_score_bank, fuse_ranked_scores
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.hyperedge import (
    accepted_hyperedge_metrics,
    generate_candidate_plaquettes,
    hyperedge_anchor_assignment_solver,
    load_hyperedge_checkpoint,
    score_plaquettes,
)
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_EMBEDDING = "runs/assembly_v1/hbt_d320_denoised_rgb_sobel.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint", default=DEFAULT_EMBEDDING)
    parser.add_argument("--hyperedge-checkpoint", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument(
        "--exact-panel",
        choices=["primary_kornia", "independent_libjpeg"],
        required=True,
    )
    parser.add_argument("--exact-split", default="edge_development")
    parser.add_argument("--exact-offset", type=int, default=0)
    parser.add_argument("--exact-sources", type=int, default=4)
    parser.add_argument("--real-split", default="assembly_cal")
    parser.add_argument("--real-offset", type=int, default=0)
    parser.add_argument("--real-sources", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--verifier-batch-size", type=int, default=256)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--max-per-anchor", type=int, default=4)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--max-hyperedges", type=int, default=64)
    parser.add_argument("--displacement-weight", type=float, default=0.35)
    parser.add_argument("--qap-iterations", type=int, default=25)
    parser.add_argument("--qap-restarts", type=int, default=2)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.05)
    parser.add_argument("--qap-initial-weight", type=float, default=0.75)
    parser.add_argument("--qap-noisy-components", type=int, default=3)
    parser.add_argument("--qap-noise-scale", type=float, default=1.0)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def build_pair_scores(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    embedding_model: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[Any, Any, Any, list[Any]]:
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
    return c1_views["denoised"], hbt, qap_score, [
        c1_views["raw"],
        c1_views["denoised"],
        cross_c1,
        hbt,
        qap_score,
    ]


def predict_input_only(
    raw_tiles: np.ndarray,
    *,
    source_name: str,
    args: argparse.Namespace,
    restorer: torch.nn.Module,
    embedding_model: torch.nn.Module,
    hyperedge_model: torch.nn.Module,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    """Freeze all layouts without accepting a target path or target pixels."""
    started = time.perf_counter()
    denoised = restore_tiles_uint8(
        restorer, raw_tiles, device, batch_size=args.denoise_batch_size
    )
    score_started = time.perf_counter()
    c1, hbt, qap_score, candidate_scores = build_pair_scores(
        raw_tiles, denoised, embedding_model, device=device
    )
    candidates = generate_candidate_plaquettes(
        candidate_scores,
        top_k=args.candidate_top_k,
        max_per_anchor_per_score=args.max_per_anchor,
    )
    scored = score_plaquettes(
        hyperedge_model,
        raw_tiles,
        denoised,
        candidates,
        c1,
        hbt,
        device=device,
        batch_size=args.verifier_batch_size,
    )
    score_seconds = time.perf_counter() - score_started
    solver_started = time.perf_counter()
    seed_layout = soft_cycle_component_solver(
        hbt,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        reciprocal_weight=0.35,
        loop_weight=1.0,
    ).position_to_slot
    qap_seed = int.from_bytes(
        hashlib.sha256(source_name.encode("utf-8")).digest()[:4], "little"
    ) + 7001
    qap_result = directional_qap(
        qap_score,
        initial=seed_layout,
        iterations=args.qap_iterations,
        restarts=args.qap_restarts,
        seed=qap_seed,
        boundary_weight=args.qap_boundary_weight,
        initial_weight=args.qap_initial_weight,
        noisy_components=args.qap_noisy_components,
        noise_scale=args.qap_noise_scale,
        refine_swaps=args.qap_refine_swaps,
    )
    baseline = qap_result.position_to_slot.copy()
    hyperedge_result = hyperedge_anchor_assignment_solver(
        qap_score,
        baseline,
        scored,
        threshold=threshold,
        max_hyperedges=args.max_hyperedges,
        displacement_weight=args.displacement_weight,
    )
    solver_seconds = time.perf_counter() - solver_started
    return {
        "raw_tiles": raw_tiles,
        "denoised_tiles": denoised,
        "baseline_layout": baseline,
        "candidate_layout": hyperedge_result.position_to_slot.copy(),
        "scored": scored,
        "hyperedge_result": hyperedge_result,
        "diagnostics": {
            "candidate_count": len(candidates),
            "threshold": threshold,
            "accepted": len(hyperedge_result.accepted),
            "anchored_tiles": hyperedge_result.anchored_tiles,
            "coverage": hyperedge_result.coverage,
            "realized_hyperedges": hyperedge_result.realized_hyperedges,
            "skipped_for_placement": hyperedge_result.skipped_for_placement,
            "qap_seed": int(qap_seed),
            "qap_objective": qap_result.objective,
            "qap_best_restart": qap_result.restart,
            "qap_iterations": qap_result.iterations,
            "score_seconds": score_seconds,
            "solver_seconds": solver_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }


def numeric_mean(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    common = set(records[0])
    for record in records[1:]:
        common &= set(record)
    output = {}
    for key in sorted(common):
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float, np.number, bool)) for value in values):
            output[key] = float(np.mean(values))
    return output


def layout_record(layout: np.ndarray) -> dict[str, Any]:
    values = np.asarray(layout, dtype=np.int32)
    return {"sha256": array_sha256(values), "position_to_slot": values.tolist()}


def main() -> None:
    args = parse_args()
    if (
        args.exact_offset < 0
        or args.real_offset < 0
        or args.exact_sources <= 0
        or args.real_sources <= 0
        or args.qap_iterations < 0
        or args.qap_restarts <= 0
    ):
        raise SystemExit("invalid source slices or QAP settings")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    exact_names = source_names_for_split(
        args.exact_split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[args.exact_offset : args.exact_offset + args.exact_sources]
    real_names = source_names_for_split(
        args.real_split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[args.real_offset : args.real_offset + args.real_sources]
    if len(exact_names) != args.exact_sources or len(real_names) != args.real_sources:
        raise SystemExit("requested source slice extends past split")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    embedding_model, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    hyperedge_model, checkpoint_threshold, hyperedge_metadata = load_hyperedge_checkpoint(
        args.hyperedge_checkpoint, device=device
    )
    threshold = checkpoint_threshold if args.threshold is None else float(args.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise SystemExit("threshold must lie in [0, 1]")
    for model in (restorer, embedding_model, hyperedge_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    exact_records = []
    exact_hyperedge_accepted = 0
    exact_hyperedge_correct = 0
    exact_coverages = []
    started = time.perf_counter()
    for index, name in enumerate(exact_names):
        target = read_rgb(Path(args.data_root) / "train" / "targets" / name)
        panel_seed = per_source_seed(
            args.seed, f"hyperedge-exact-{args.exact_panel}", name, 0
        )
        panel = make_exact_panel(target, panel=args.exact_panel, seed=panel_seed)
        prediction = predict_input_only(
            panel.slot_tiles,
            source_name=name,
            args=args,
            restorer=restorer,
            embedding_model=embedding_model,
            hyperedge_model=hyperedge_model,
            threshold=threshold,
            device=device,
        )
        # Exact truth is consulted only after both layouts have been frozen.
        baseline_layout = prediction["baseline_layout"]
        candidate_layout = prediction["candidate_layout"]
        baseline_layout_metrics = layout_metrics(baseline_layout, panel.slot_to_target)
        candidate_layout_metrics = layout_metrics(candidate_layout, panel.slot_to_target)
        baseline_image = predicted_image_metrics(
            baseline_layout, prediction["denoised_tiles"], target
        )
        candidate_image = predicted_image_metrics(
            candidate_layout, prediction["denoised_tiles"], target
        )
        accepted_metrics = accepted_hyperedge_metrics(
            prediction["hyperedge_result"].accepted, panel.slot_to_target
        )
        exact_hyperedge_accepted += int(accepted_metrics["accepted"])
        exact_hyperedge_correct += int(accepted_metrics["correct"])
        exact_coverages.append(float(accepted_metrics["coverage"]))
        record = {
            "source": name,
            "panel": args.exact_panel,
            "panel_seed": panel_seed,
            "baseline": {
                "layout": layout_record(baseline_layout),
                "layout_metrics": baseline_layout_metrics,
                "image_metrics": baseline_image,
            },
            "candidate": {
                "layout": layout_record(candidate_layout),
                "layout_metrics": candidate_layout_metrics,
                "image_metrics": candidate_image,
            },
            "hyperedges": accepted_metrics,
            "diagnostics": prediction["diagnostics"],
            "adjacency_gain": float(
                candidate_layout_metrics["combined_adjacency"]
                - baseline_layout_metrics["combined_adjacency"]
            ),
            "ssim_gain": float(
                candidate_image["predicted_layout_ssim"]
                - baseline_image["predicted_layout_ssim"]
            ),
        }
        exact_records.append(record)
        print(
            json.dumps(
                {
                    "event": "hyperedge_exact_source",
                    "index": index + 1,
                    "count": len(exact_names),
                    "source": name,
                    "adjacency_gain": record["adjacency_gain"],
                    "ssim_gain": record["ssim_gain"],
                    "hyperedges": accepted_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    real_records = []
    for index, name in enumerate(real_names):
        input_image = read_rgb(Path(args.data_root) / "train" / "inputs" / name)
        prediction = predict_input_only(
            split_tiles_numpy(input_image),
            source_name=name,
            args=args,
            restorer=restorer,
            embedding_model=embedding_model,
            hyperedge_model=hyperedge_model,
            threshold=threshold,
            device=device,
        )
        # Critical anti-leakage boundary: target is opened only now, after the
        # input-only function has returned immutable baseline/candidate layouts.
        baseline_layout = prediction["baseline_layout"].copy()
        candidate_layout = prediction["candidate_layout"].copy()
        target = read_rgb(Path(args.data_root) / "train" / "targets" / name)
        baseline_image = predicted_image_metrics(
            baseline_layout, prediction["denoised_tiles"], target
        )
        candidate_image = predicted_image_metrics(
            candidate_layout, prediction["denoised_tiles"], target
        )
        record = {
            "source": name,
            "target_opened_after_layouts_frozen": True,
            "baseline": {
                "layout": layout_record(baseline_layout),
                "image_metrics": baseline_image,
            },
            "candidate": {
                "layout": layout_record(candidate_layout),
                "image_metrics": candidate_image,
            },
            "hyperedges": {
                "accepted": len(prediction["hyperedge_result"].accepted),
                "coverage": prediction["hyperedge_result"].coverage,
                "realized": prediction["hyperedge_result"].realized_hyperedges,
            },
            "diagnostics": prediction["diagnostics"],
            "ssim_gain": float(
                candidate_image["predicted_layout_ssim"]
                - baseline_image["predicted_layout_ssim"]
            ),
        }
        real_records.append(record)
        print(
            json.dumps(
                {
                    "event": "hyperedge_real_source",
                    "index": index + 1,
                    "count": len(real_names),
                    "source": name,
                    "ssim_gain": record["ssim_gain"],
                    "accepted": record["hyperedges"]["accepted"],
                    "coverage": record["hyperedges"]["coverage"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    exact_baseline_layout = numeric_mean(
        [record["baseline"]["layout_metrics"] for record in exact_records]
    )
    exact_candidate_layout = numeric_mean(
        [record["candidate"]["layout_metrics"] for record in exact_records]
    )
    exact_baseline_image = numeric_mean(
        [record["baseline"]["image_metrics"] for record in exact_records]
    )
    exact_candidate_image = numeric_mean(
        [record["candidate"]["image_metrics"] for record in exact_records]
    )
    real_baseline_image = numeric_mean(
        [record["baseline"]["image_metrics"] for record in real_records]
    )
    real_candidate_image = numeric_mean(
        [record["candidate"]["image_metrics"] for record in real_records]
    )
    hyperedge_precision = (
        exact_hyperedge_correct / exact_hyperedge_accepted
        if exact_hyperedge_accepted
        else 1.0
    )
    exact_adjacency_gain = float(
        exact_candidate_layout["combined_adjacency"]
        - exact_baseline_layout["combined_adjacency"]
    )
    real_ssim_gain = float(
        real_candidate_image["predicted_layout_ssim"]
        - real_baseline_image["predicted_layout_ssim"]
    )
    gates = {
        "precision_at_least_0_90": hyperedge_precision >= 0.90,
        "coverage_at_least_0_15": float(np.mean(exact_coverages)) >= 0.15,
        "exact_adjacency_gain_at_least_0_03": exact_adjacency_gain >= 0.03,
        "real_ssim_gain_at_least_0_015": real_ssim_gain >= 0.015,
    }
    payload = {
        "schema_version": 1,
        "kind": "puzzle_hyperedge_solver_gate_shard",
        "args": vars(args),
        "anti_leakage": {
            "prediction_function_accepts_target": False,
            "real_target_opened_after_layouts_frozen": True,
            "target_pixels_used_for_real_layout_selection": False,
        },
        "device": str(device),
        "threshold": threshold,
        "checkpoint_threshold": checkpoint_threshold,
        "source_lists": {
            "exact": exact_names,
            "exact_sha256": names_sha256(exact_names),
            "real": real_names,
            "real_sha256": names_sha256(real_names),
        },
        "asset_hashes": {
            "denoiser": sha256(Path(args.denoiser)),
            "embedding_checkpoint": sha256(Path(args.embedding_checkpoint)),
            "hyperedge_checkpoint": sha256(Path(args.hyperedge_checkpoint)),
            "manifest": sha256(Path(args.manifest)),
            "quarantine": sha256(Path(args.quarantine)),
        },
        "metadata": {
            "denoiser": denoiser_metadata,
            "embedding": embedding_metadata,
            "hyperedge": hyperedge_metadata,
        },
        "exact": {
            "panel": args.exact_panel,
            "source_count": len(exact_records),
            "baseline_layout": exact_baseline_layout,
            "candidate_layout": exact_candidate_layout,
            "baseline_image": exact_baseline_image,
            "candidate_image": exact_candidate_image,
            "adjacency_gain": exact_adjacency_gain,
            "hyperedge_accepted": exact_hyperedge_accepted,
            "hyperedge_correct": exact_hyperedge_correct,
            "hyperedge_precision": float(hyperedge_precision),
            "coverage": float(np.mean(exact_coverages)),
            "records": exact_records,
        },
        "real": {
            "source_count": len(real_records),
            "baseline_image": real_baseline_image,
            "candidate_image": real_candidate_image,
            "ssim_gain": real_ssim_gain,
            "records": real_records,
        },
        "gates": gates,
        "accepted": bool(all(gates.values())),
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "hyperedge_gate_complete",
                "output": str(output),
                "output_sha256": sha256(output),
                "gates": gates,
                "accepted": payload["accepted"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
