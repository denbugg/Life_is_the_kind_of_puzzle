#!/usr/bin/env python3
"""Exact-only gate for protected, incremental QAP annealing.

This script deliberately stops before the real ``assembly_cal`` targets.  It
compares the authoritative C1+HBTw4 boundary-QAP and a pure-HBT boundary-QAP,
then refines both with the broader move set in ``anneal_refine``.  Only a
predeclared improvement on both exact corruption engines permits a later
input-only real-layout freeze.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image

from puzzle_assembly.anneal_refine import anneal_refine
from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_EMBEDDING = (
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt"
)
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
AUTHORITATIVE_BASE = "qap_w4_b0.05_i25"
PURE_HBT_BASE = "qap_l1_b0.05_i25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint", default=DEFAULT_EMBEDDING)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument(
        "--exact-panels", default="primary_kornia,independent_libjpeg"
    )
    parser.add_argument("--exact-sources", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--evaluations", type=int, default=6_000)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--protection-strengths", default="0,0.10,0.25")
    parser.add_argument("--confidence-quantile", type=float, default=0.75)
    parser.add_argument("--max-protected-edges", type=int, default=384)
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


def _layout_sha256(layout: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(layout, dtype=np.int32).tobytes()).hexdigest()


def _filename_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _c1_fusion(
    bank: dict[str, CompatibilityMatrices], prefix: str
) -> CompatibilityMatrices:
    names = [
        name
        for name in sorted(bank)
        if name.startswith(prefix + "_") and not name.endswith("_c2")
    ]
    return fuse_ranked_scores(bank, names=names, name=f"{prefix}_C1_equal_rank_fusion")


def _score_views(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    embedding_model: object,
    device: object,
    chunk_size: int,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices]:
    bank = build_classical_score_bank(raw_tiles, prefix="raw", chunk_size=chunk_size)
    bank.update(
        build_classical_score_bank(
            denoised_tiles, prefix="denoised", chunk_size=chunk_size
        )
    )
    denoised_c1 = _c1_fusion(bank, "denoised")
    bank[denoised_c1.name] = denoised_c1
    hbt, _ = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_hbt_l1",
    )
    bank[hbt.name] = hbt
    w4 = fuse_ranked_scores(
        bank,
        names=[denoised_c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="denoised_C1_HBTw4_rank_fusion",
    )
    return hbt, w4


def _strength_label(value: float) -> str:
    return f"p{value:.3f}".replace(".", "p")


def _predict_layouts(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    source: str,
    panel: str,
    embedding_model: object,
    device: object,
    chunk_size: int,
    protection_strengths: tuple[float, ...],
    evaluations: int,
    restarts: int,
    confidence_quantile: float,
    max_protected_edges: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], float]:
    started = time.perf_counter()
    hbt, w4 = _score_views(
        raw_tiles,
        denoised_tiles,
        embedding_model=embedding_model,
        device=device,
        chunk_size=chunk_size,
    )
    component_seed = soft_cycle_component_solver(
        hbt,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    ).position_to_slot
    qap_seed = _filename_seed(source) + 7001
    layouts: dict[str, np.ndarray] = {"softcycle_l1_k8": component_seed.copy()}
    diagnostics: dict[str, dict[str, Any]] = {}
    bases = ((AUTHORITATIVE_BASE, w4), (PURE_HBT_BASE, hbt))
    for base_label, score in bases:
        qap = directional_qap(
            score,
            initial=component_seed,
            iterations=25,
            restarts=2,
            seed=qap_seed,
            boundary_weight=0.05,
            initial_weight=0.75,
            noisy_components=3,
            noise_scale=1.0,
            refine_swaps=8,
            refine_weak_cells=32,
        )
        layouts[base_label] = qap.position_to_slot.copy()
        diagnostics[base_label] = {
            "kind": "directional_qap",
            "objective": float(qap.objective),
            "relaxed_objective": float(qap.relaxed_objective),
            "restart": int(qap.restart),
            "iterations": int(qap.iterations),
            "converged": bool(qap.converged),
        }
        for strength in protection_strengths:
            label = f"anneal_{base_label}_{_strength_label(strength)}"
            anneal_seed = _filename_seed(f"{source}|{panel}|{base_label}|{strength}")
            refined = anneal_refine(
                qap.position_to_slot,
                score,
                seed=anneal_seed,
                seed_compatibility=hbt,
                protected_layout=component_seed,
                evaluations_per_restart=evaluations,
                restarts=restarts,
                boundary_weight=0.05,
                protection_strength=strength,
                confidence_quantile=confidence_quantile,
                max_protected_edges=max_protected_edges,
            )
            layouts[label] = refined.position_to_slot.copy()
            diagnostics[label] = {
                "kind": "incremental_protected_anneal",
                "base": base_label,
                **asdict(refined),
            }
            diagnostics[label].pop("position_to_slot")
    for label, layout in layouts.items():
        diagnostics.setdefault(label, {})["layout_sha256"] = _layout_sha256(layout)
    return layouts, diagnostics, time.perf_counter() - started


def _mean_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        key
        for key in records[0]
        if all(
            isinstance(record.get(key), (int, float, bool, np.integer, np.floating))
            for record in records
        )
    )
    return {key: float(np.mean([float(record[key]) for record in records])) for key in keys}


def _paired_gate(
    records: list[dict[str, Any]],
    *,
    candidate: str,
    base: str,
) -> dict[str, Any]:
    base_records = {
        (record["panel"], record["source"]): record
        for record in records
        if record["label"] == base
    }
    candidate_records = {
        (record["panel"], record["source"]): record
        for record in records
        if record["label"] == candidate
    }
    if base_records.keys() != candidate_records.keys() or not base_records:
        raise RuntimeError(f"paired records do not match for {candidate} vs {base}")
    keys = sorted(base_records)
    ssim_delta = np.asarray(
        [
            candidate_records[key]["predicted_layout_ssim"]
            - base_records[key]["predicted_layout_ssim"]
            for key in keys
        ],
        dtype=np.float64,
    )
    adjacency_delta = np.asarray(
        [
            candidate_records[key]["combined_adjacency"]
            - base_records[key]["combined_adjacency"]
            for key in keys
        ],
        dtype=np.float64,
    )
    base_manhattan = np.asarray(
        [base_records[key]["mean_manhattan"] for key in keys], dtype=np.float64
    )
    candidate_manhattan = np.asarray(
        [candidate_records[key]["mean_manhattan"] for key in keys],
        dtype=np.float64,
    )
    panel_delta = {
        panel: float(
            np.mean(
                [
                    ssim_delta[index]
                    for index, key in enumerate(keys)
                    if key[0] == panel
                ]
            )
        )
        for panel in sorted({key[0] for key in keys})
    }
    changed = sum(
        candidate_records[key]["layout_sha256"] != base_records[key]["layout_sha256"]
        for key in keys
    )
    required_wins = int(np.ceil(0.75 * len(keys)))
    manhattan_reduction_fraction = float(
        (base_manhattan.mean() - candidate_manhattan.mean())
        / max(base_manhattan.mean(), 1e-12)
    )
    checks = {
        "mean_ssim_delta_at_least_0_010": float(ssim_delta.mean()) >= 0.010,
        "mean_adjacency_delta_at_least_0_020": float(adjacency_delta.mean())
        >= 0.020,
        "mean_manhattan_reduction_at_least_10pct": manhattan_reduction_fraction
        >= 0.10,
        "ssim_wins_at_least_75pct": int(np.sum(ssim_delta > 0.0)) >= required_wins,
        "no_exact_panel_has_negative_mean_ssim_delta": min(panel_delta.values()) >= 0.0,
    }
    return {
        "candidate": candidate,
        "base": base,
        "pairs": len(keys),
        "changed_layouts": int(changed),
        "mean_ssim_delta": float(ssim_delta.mean()),
        "median_ssim_delta": float(np.median(ssim_delta)),
        "ssim_wins": int(np.sum(ssim_delta > 0.0)),
        "ssim_ties": int(np.sum(ssim_delta == 0.0)),
        "ssim_losses": int(np.sum(ssim_delta < 0.0)),
        "required_wins": required_wins,
        "mean_adjacency_delta": float(adjacency_delta.mean()),
        "mean_manhattan_reduction_fraction": manhattan_reduction_fraction,
        "mean_ssim_delta_by_panel": panel_delta,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    if args.exact_sources <= 0 or args.evaluations < 0 or args.restarts <= 0:
        raise SystemExit("source/budget/restart values are invalid")
    panels = tuple(
        value.strip() for value in args.exact_panels.split(",") if value.strip()
    )
    if set(panels) != {"primary_kornia", "independent_libjpeg"}:
        raise SystemExit(
            "exact panels must contain primary_kornia and independent_libjpeg"
        )
    strengths = tuple(
        float(value.strip())
        for value in args.protection_strengths.split(",")
        if value.strip()
    )
    if not strengths or any(value < 0.0 or not np.isfinite(value) for value in strengths):
        raise SystemExit("protection strengths must be finite and non-negative")
    if len(strengths) != len(set(strengths)):
        raise SystemExit("protection strengths must be unique")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))

    source_names = source_names_for_split(
        "edge_development",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[: args.exact_sources]
    if len(source_names) != args.exact_sources:
        raise SystemExit("requested exact source count exceeds split")
    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    data_root = Path(args.data_root)
    for panel in panels:
        for index, source in enumerate(source_names):
            target = _read_rgb(data_root / "train" / "targets" / source)
            panel_seed = per_source_seed(
                args.seed, f"anneal-gate-{panel}", source, 0
            )
            exact = make_exact_panel(target, panel=panel, seed=panel_seed)
            denoised = restore_tiles_uint8(
                restorer, exact.slot_tiles, device, batch_size=args.batch_size
            )
            layouts, diagnostics, seconds = _predict_layouts(
                exact.slot_tiles,
                denoised,
                source=source,
                panel=panel,
                embedding_model=embedding,
                device=device,
                chunk_size=args.chunk_size,
                protection_strengths=strengths,
                evaluations=args.evaluations,
                restarts=args.restarts,
                confidence_quantile=args.confidence_quantile,
                max_protected_edges=args.max_protected_edges,
            )
            for label, layout in layouts.items():
                records.append(
                    {
                        "panel": panel,
                        "source": source,
                        "label": label,
                        **layout_metrics(layout, exact.slot_to_target),
                        **predicted_image_metrics(layout, denoised, target),
                        "layout_sha256": _layout_sha256(layout),
                        "solver": diagnostics[label],
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "anneal_exact_source",
                        "panel": panel,
                        "index": index + 1,
                        "count": len(source_names),
                        "source": source,
                        "candidate_count": len(layouts),
                        "seconds": seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    labels = sorted({record["label"] for record in records})
    aggregate = {
        label: _mean_metrics([record for record in records if record["label"] == label])
        for label in labels
    }
    gates = []
    for base in (AUTHORITATIVE_BASE, PURE_HBT_BASE):
        for strength in strengths:
            candidate = f"anneal_{base}_{_strength_label(strength)}"
            gates.append(_paired_gate(records, candidate=candidate, base=base))
    gates.sort(
        key=lambda gate: (
            -gate["mean_ssim_delta"],
            -gate["mean_adjacency_delta"],
            -gate["mean_manhattan_reduction_fraction"],
            gate["candidate"],
        )
    )
    passed = [gate for gate in gates if gate["passed"]]
    pure_vs_authoritative = _paired_gate(
        records, candidate=PURE_HBT_BASE, base=AUTHORITATIVE_BASE
    )
    # This comparison is descriptive, not a promotion gate for annealing.
    pure_vs_authoritative["passed"] = None
    pure_vs_authoritative["checks"] = {}

    experiment_config = {
        "qap": {
            "iterations": 25,
            "restarts": 2,
            "seed_formula": "filename_sha256_first4_le + 7001",
            "boundary_weight": 0.05,
            "initial_weight": 0.75,
            "noisy_components": 3,
            "noise_scale": 1.0,
            "refine_swaps": 8,
            "refine_weak_cells": 32,
        },
        "anneal": {
            "evaluations_per_restart": args.evaluations,
            "restarts": args.restarts,
            "protection_strengths": strengths,
            "confidence_quantile": args.confidence_quantile,
            "max_protected_edges": args.max_protected_edges,
            "moves": "swap, segment relocation, small-block swap, whole-band swap",
        },
    }
    report = {
        "schema_version": 1,
        "kind": "protected_incremental_anneal_exact_gate",
        "status": (
            "exact_gate_passed_real_input_freeze_allowed"
            if passed
            else "scientific_gate_failed_stop_annealing"
        ),
        "decision": {
            "promoted_candidate": passed[0]["candidate"] if passed else None,
            "hard_pivot_if_failed": (
                "Do not spend real16 target access or a longer annealing budget; "
                "pivot to a better learned compatibility/global assignment model."
            ),
            "next_if_passed": (
                "Repeat exact8 with two anneal seeds; only then freeze every real16 "
                "input layout before attaching targets."
            ),
        },
        "anti_leakage": {
            "split": "edge_development",
            "predictor_accepts_target": False,
            "target_use": "exact synthetic corruption construction and scoring only",
            "assembly_cal_targets_opened": False,
            "selection_before_any_real_target": True,
        },
        "hard_gate": {
            "mean_ssim_delta": 0.010,
            "mean_adjacency_delta": 0.020,
            "mean_manhattan_reduction_fraction": 0.10,
            "minimum_ssim_win_fraction": 0.75,
            "minimum_panel_mean_ssim_delta": 0.0,
        },
        "expected_runtime": {
            "default_exact4_two_panels_t4": "approximately 5-12 minutes",
            "exact8_two_seeds_t4_if_gate_passes": "approximately 20-45 minutes",
            "note": "GPU is used for TileNAF/HBT; QAP and incremental annealing are CPU-heavy.",
        },
        "args": vars(args),
        "source_names": source_names,
        "panels": panels,
        "experiment_config": experiment_config,
        "experiment_config_sha256": _canonical_sha256(experiment_config),
        "assets": {
            "denoiser": args.denoiser,
            "denoiser_sha256": _sha256(Path(args.denoiser)),
            "denoiser_metadata": denoiser_metadata,
            "embedding": args.embedding_checkpoint,
            "embedding_sha256": _sha256(Path(args.embedding_checkpoint)),
            "embedding_metadata": embedding_metadata,
        },
        "aggregate": aggregate,
        "pure_hbt_qap_vs_authoritative_w4_qap": pure_vs_authoritative,
        "anneal_gates": gates,
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "anneal_gate_complete",
                "status": report["status"],
                "output": str(output),
                "output_sha256": _sha256(output),
                "best_gate": gates[0],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
