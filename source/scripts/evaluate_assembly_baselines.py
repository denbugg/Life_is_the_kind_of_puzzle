#!/usr/bin/env python3
"""Evaluate denoised classical assembly baselines on exact synthetic shuffles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    build_edge_filter_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.growing_consensus import discover_order2_consensus
from puzzle_assembly.components import (
    mutual_topk_component_solver,
    propose_mutual_topk_edges,
    propose_reciprocal_edges,
    propose_soft_cycle_edges,
    reciprocal_component_solver,
    rigid_soft_cycle_qap_projection,
    select_confident_edges,
    soft_cycle_component_solver,
    successive_topk_lp_solver,
    translation_consensus_component_solver,
    weighted_l1_component_solver,
)
from puzzle_assembly.cpsat import topk_cpsat_grid_solver
from puzzle_assembly.geometry import inverse_permutation
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics, retrieval_metrics
from puzzle_assembly.learned import (
    candidate_rank_features,
    candidate_union,
    context_position_logits,
    global_matcher_compatibility,
    learned_compatibility,
    load_context_position_checkpoint,
    load_embedding_checkpoint,
    load_global_matcher_checkpoint,
    load_pair_checkpoint,
    load_rank_feature_checkpoint,
    pair_rerank_compatibility,
    rank_feature_compatibility,
)
from puzzle_assembly.line_seam import line_seam_compatibility
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.particle import particle_beam_solver
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_assembly.solvers import (
    beam_row_major,
    faithful_multi_phase_relaxation_solver,
    four_side_hungarian_refine,
    greedy_row_major,
    identity_layout,
    large_neighborhood_reassign,
    multi_phase_relaxation_solver,
    random_layout,
    relaxation_labeling_solver,
    segment_preserving_genetic_solver,
    outside_logits_placement_unary,
    position_logits_placement_unary,
    segment_block_refine,
    simulated_anneal_mixed,
    simulated_anneal_swaps,
    swap_refine,
)
from puzzle_assembly.spatial_prior import spatial_prior_cost
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy


DEFAULT_CHECKPOINT = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--embedding-checkpoint")
    parser.add_argument("--global-matcher-checkpoint")
    parser.add_argument("--pair-checkpoint")
    parser.add_argument("--rank-checkpoint")
    parser.add_argument("--context-checkpoint")
    parser.add_argument("--spatial-prior")
    parser.add_argument("--spatial-prior-weights", default="0.05,0.2,1.0")
    parser.add_argument("--line-seam", action="store_true")
    parser.add_argument("--line-seam-auxiliary-weight", type=float, default=0.35)
    parser.add_argument("--line-seam-fusion-weight", type=float, default=0.5)
    parser.add_argument(
        "--pair-candidate-policy",
        choices=["pbc32", "union64"],
        default="union64",
        help="candidate generator for the optional L0 pair reranker",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument(
        "--split",
        choices=["edge_development", "assembly_cal"],
        default="edge_development",
    )
    parser.add_argument(
        "--panel",
        choices=["clean_shuffle", "primary_kornia", "independent_libjpeg"],
        default="clean_shuffle",
    )
    parser.add_argument("--view", choices=["denoised", "raw"], default="denoised")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20260710)
    parser.add_argument(
        "--panel-seed-stage",
        help="override the deterministic panel seed namespace for replica stress tests",
    )
    parser.add_argument("--panel-replica", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-candidate-pool", type=int, default=4)
    parser.add_argument("--swap-weak-cells", type=int, default=32)
    parser.add_argument("--swap-max", type=int, default=8)
    parser.add_argument("--segment-moves", type=int, default=4)
    parser.add_argument("--anneal-evaluations", type=int, default=1000)
    parser.add_argument(
        "--anneal-refine-seeds",
        default="",
        help="comma-separated completed solver layouts for long post-solve annealing",
    )
    parser.add_argument("--anneal-refine-evaluations", type=int, default=0)
    parser.add_argument("--anneal-refine-mixed", action="store_true")
    parser.add_argument("--relaxation-iterations", type=int, default=0)
    parser.add_argument("--four-side-refine-iterations", type=int, default=0)
    parser.add_argument(
        "--four-side-refine-seeds",
        default="greedy",
        help="comma-separated solver layout names to refine with the global score",
    )
    parser.add_argument("--genetic-generations", type=int, default=0)
    parser.add_argument("--genetic-population", type=int, default=24)
    parser.add_argument("--genetic-elite", type=int, default=6)
    parser.add_argument(
        "--genetic-seeds",
        default="greedy",
        help="comma-separated completed solver layouts used as GA parents",
    )
    parser.add_argument("--lns-iterations", type=int, default=0)
    parser.add_argument("--lns-subset-size", type=int, default=128)
    parser.add_argument(
        "--lns-seeds",
        default="greedy",
        help="comma-separated completed solver layouts for destroy/reassign",
    )
    parser.add_argument("--multi-phase-rl-phases", type=int, default=0)
    parser.add_argument("--multi-phase-rl-topk", type=int, default=8)
    parser.add_argument("--multi-phase-rl-iterations", type=int, default=4)
    parser.add_argument("--multi-phase-rl-anchor-batch", type=int, default=48)
    parser.add_argument(
        "--multi-phase-rl-seeds",
        default="greedy",
        help="comma-separated completed solver layouts for progressive RL",
    )
    parser.add_argument("--faithful-rl-phases", type=int, default=0)
    parser.add_argument("--faithful-rl-topk", type=int, default=17)
    parser.add_argument("--faithful-rl-max-iterations", type=int, default=48)
    parser.add_argument("--faithful-rl-convergence", type=float, default=1e-4)
    parser.add_argument("--faithful-rl-anchor-probability", type=float, default=0.70)
    parser.add_argument("--particle-beam-particles", type=int, default=0)
    parser.add_argument("--particle-beam-topk", type=int, default=3)
    parser.add_argument("--particle-beam-anchor-hypotheses", type=int, default=4)
    parser.add_argument("--particle-beam-frontier-limit", type=int, default=24)
    parser.add_argument(
        "--particle-beam-seeds",
        default="greedy",
        help="comma-separated completed layouts used to initialize particle growth",
    )
    parser.add_argument("--qap-iterations", type=int, default=0)
    parser.add_argument("--qap-restarts", type=int, default=1)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.0)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--qap-initial-weight", type=float, default=0.75)
    parser.add_argument("--qap-noisy-components", type=int, default=3)
    parser.add_argument("--qap-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--qap-seeds",
        default="greedy",
        help="comma-separated completed layouts used as QAP initializations; empty uses barycenter",
    )
    parser.add_argument(
        "--rigid-component-qap-projection",
        action="store_true",
        help="research-only rigid soft-cycle projection around one frozen QAP layout",
    )
    parser.add_argument(
        "--rigid-projection-component-seed",
        default="component_l1_softcycle_k8_p1",
    )
    parser.add_argument(
        "--rigid-projection-reference",
        default="qap_component_l1_softcycle_k8_p1",
    )
    parser.add_argument("--rigid-projection-reference-weight", type=float, default=0.5)
    parser.add_argument("--rigid-projection-beam-width", type=int, default=8)
    parser.add_argument("--rigid-projection-beam-components", type=int, default=16)
    parser.add_argument("--rigid-projection-translations", type=int, default=8)
    parser.add_argument(
        "--order2-consensus-topk",
        type=int,
        default=0,
        help="input-only top-k graph for the bounded Growing Consensus QAP pilot",
    )
    parser.add_argument(
        "--order2-consensus-min-support",
        type=int,
        default=13,
        help="minimum distinct incomplete 2x2 witnesses for a soft edge bonus",
    )
    parser.add_argument(
        "--order2-consensus-bonus",
        type=float,
        default=0.05,
        help="bounded rank-cost bonus per accepted order-2 consensus edge",
    )
    parser.add_argument("--cpsat-time-seconds", type=float, default=0.0)
    parser.add_argument("--cpsat-topk", type=int, default=8)
    parser.add_argument("--cpsat-workers", type=int, default=1)
    parser.add_argument("--cpsat-square-terms", type=int, default=2048)
    parser.add_argument(
        "--cpsat-seeds",
        default="greedy",
        help="comma-separated completed layouts used as CP-SAT hints",
    )
    parser.add_argument(
        "--global-score",
        default="fusion",
        help="single score alias used by greedy, beam and dense relaxation solvers",
    )
    parser.add_argument(
        "--component-scores",
        default="pbc,fusion",
        help="comma-separated score aliases (pbc,fusion) or exact score-bank names",
    )
    parser.add_argument(
        "--skip-component-refine",
        action="store_true",
        help="skip the loop-priority + bounded-swap component variant",
    )
    parser.add_argument("--component-placement-beam-width", type=int, default=1)
    parser.add_argument("--component-placement-beam-components", type=int, default=12)
    parser.add_argument(
        "--mutual-topk",
        default="",
        help="optional comma-separated k values for mutual-top-k geometric growth",
    )
    parser.add_argument(
        "--soft-cycle-topk",
        default="",
        help="optional comma-separated k values for soft 2x2 cycle growth",
    )
    parser.add_argument("--soft-cycle-keep-per-tile", type=int, default=2)
    parser.add_argument("--soft-cycle-loop-weight", type=float, default=1.0)
    parser.add_argument("--soft-cycle-reciprocal-weight", type=float, default=0.35)
    parser.add_argument("--soft-cycle-keep-fraction", type=float, default=1.0)
    parser.add_argument(
        "--translation-consensus-topk",
        default="",
        help="optional comma-separated k values for multi-edge component merging",
    )
    parser.add_argument("--translation-consensus-min-support", type=int, default=2)
    parser.add_argument("--translation-consensus-seed-fraction", type=float, default=0.5)
    parser.add_argument(
        "--successive-lp-topk",
        default="",
        help="optional comma-separated k values for successive four-side L1 LP",
    )
    parser.add_argument("--successive-lp-iterations", type=int, default=8)
    parser.add_argument("--successive-lp-residual-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--lp-scores",
        default="",
        help="optional comma-separated score aliases for weighted-L1 G2 variants",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        result = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if result.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {result.shape}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_dict(records: list[dict]) -> dict:
    keys = sorted(set.intersection(*(set(record) for record in records))) if records else []
    result: dict[str, float] = {}
    for key in keys:
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float, bool)) for value in values):
            result[key] = float(np.mean(values))
    return result


def _macro_report(sources: list[dict]) -> dict:
    solver_names = sorted(sources[0]["solvers"]) if sources else []
    score_names = sorted(sources[0]["retrieval"]) if sources else []
    return {
        "solvers": {
            solver: {
                "layout": _mean_dict([source["solvers"][solver]["layout"] for source in sources]),
                "image": _mean_dict([source["solvers"][solver]["image"] for source in sources]),
                "seconds": float(np.mean([source["solvers"][solver]["seconds"] for source in sources])),
            }
            for solver in solver_names
        },
        "retrieval": {
            score: {
                direction: _mean_dict(
                    [source["retrieval"][score][direction] for source in sources]
                )
                for direction in ("right", "down", "combined")
            }
            for score in score_names
        },
    }


def _resolve_component_scores(
    requested: str,
    *,
    view: str,
    bank: dict,
    fused_name: str,
) -> list[tuple[str, str]]:
    aliases = {
        "pbc": f"{view}_pbc",
        "fusion": fused_name,
        "l1": f"{view}_l1_embedding",
        "l1fusion": f"{view}_C1_L1_equal_rank_fusion",
        "l1w2": f"{view}_C1_L1w2_rank_fusion",
        "l1w4": f"{view}_C1_L1w4_rank_fusion",
        "l1compact": f"{view}_compact_L1w4_rank_fusion",
        "l0": f"{view}_l0_pair_reranker",
        "x0": f"{view}_x0_rank_reranker",
        "x0fusion": f"{view}_C1_X0_equal_rank_fusion",
        "x0w2": f"{view}_C1_X0w2_rank_fusion",
        "l1x0": f"{view}_L1_X0_equal_rank_fusion",
        "l1x0full": f"{view}_C1_L1w4_X0w2_rank_fusion",
        "sobel": f"{view}_sobel_l1_w2",
        "binary": f"{view}_binary_edge_hamming_w2",
        "edgefusion": f"{view}_edge_filter_rank_fusion",
        "fusionedge": f"{view}_C1_edge_rank_fusion",
        "line": f"{view}_raw_line_seam_fused" if view == "denoised" else f"{view}_line_seam",
        "l1w4line": f"{view}_C1_L1w4_line_rank_fusion",
        "g0": f"{view}_g0_global_matcher",
    }
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_name in requested.split(","):
        alias = raw_name.strip()
        if not alias:
            continue
        score_name = aliases.get(alias, alias)
        if score_name not in bank:
            raise SystemExit(
                f"unknown --component-scores entry {alias!r}; available: {sorted(bank)}"
            )
        if score_name in seen:
            continue
        seen.add(score_name)
        label = alias.replace("_C1_equal_rank_fusion", "fusion")
        resolved.append((label, score_name))
    if not resolved:
        raise SystemExit("--component-scores must select at least one score")
    return resolved


def _proposal_quality(proposals: list, slot_to_target: np.ndarray) -> dict[str, float]:
    correct = 0
    loop_count = 0
    correct_loop_count = 0
    for edge in proposals:
        first_position = int(slot_to_target[edge.first])
        second_position = int(slot_to_target[edge.second])
        first_row, first_column = divmod(first_position, 24)
        second_row, second_column = divmod(second_position, 24)
        is_correct = (
            second_column - first_column == edge.dx
            and second_row - first_row == edge.dy
        )
        correct += int(is_correct)
        loop_count += int(edge.in_loop)
        correct_loop_count += int(edge.in_loop and is_correct)
    count = len(proposals)
    return {
        "count": float(count),
        "precision": float(correct / count) if count else 0.0,
        "true_edge_coverage": float(correct / (2 * 24 * 23)),
        "loop_count": float(loop_count),
        "loop_precision": float(correct_loop_count / loop_count) if loop_count else 0.0,
    }


def _order2_consensus_quality(result, slot_to_target: np.ndarray) -> dict[str, object]:
    correct = 0
    supports = []
    for proposal in result.proposals:
        edge = proposal.edge
        first_position = int(slot_to_target[edge.first])
        second_position = int(slot_to_target[edge.second])
        first_row, first_column = divmod(first_position, 24)
        second_row, second_column = divmod(second_position, 24)
        correct += int(
            second_column - first_column == edge.dx
            and second_row - first_row == edge.dy
        )
        supports.append(proposal.support)
    count = len(result.proposals)
    return {
        "count": count,
        "precision": float(correct / count) if count else 0.0,
        "true_edge_coverage": float(correct / (2 * 24 * 23)),
        "complete_loop_count": len(result.complete_loops),
        "input_edge_count": result.input_edge_count,
        "support_min": min(supports) if supports else 0,
        "support_max": max(supports) if supports else 0,
        "support_mean": float(np.mean(supports)) if supports else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit <= 0:
        raise SystemExit("--offset must be non-negative and --limit positive")
    try:
        mutual_topks = [int(value) for value in args.mutual_topk.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit("--mutual-topk must contain integers") from exc
    try:
        soft_cycle_topks = [
            int(value) for value in args.soft_cycle_topk.split(",") if value.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--soft-cycle-topk must contain integers") from exc
    if any(value <= 0 or value >= 576 for value in mutual_topks):
        raise SystemExit("--mutual-topk values must be in [1, 575]")
    if any(value < 2 or value >= 576 for value in soft_cycle_topks):
        raise SystemExit("--soft-cycle-topk values must be in [2, 575]")
    if (
        not np.isfinite(args.rigid_projection_reference_weight)
        or args.rigid_projection_reference_weight < 0
    ):
        raise SystemExit("--rigid-projection-reference-weight must be finite and non-negative")
    if (
        args.rigid_projection_beam_width <= 1
        or args.rigid_projection_beam_components <= 0
        or args.rigid_projection_translations <= 0
    ):
        raise SystemExit("rigid projection beam limits must be positive and width > 1")
    if not 0 <= args.order2_consensus_topk < 576:
        raise SystemExit("--order2-consensus-topk must be in [0, 575]")
    if args.order2_consensus_min_support < 2:
        raise SystemExit("--order2-consensus-min-support must be at least 2")
    if args.order2_consensus_bonus < 0 or not np.isfinite(
        args.order2_consensus_bonus
    ):
        raise SystemExit("--order2-consensus-bonus must be finite and non-negative")
    try:
        translation_consensus_topks = [
            int(value)
            for value in args.translation_consensus_topk.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--translation-consensus-topk must contain integers") from exc
    if any(value < 1 or value >= 576 for value in translation_consensus_topks):
        raise SystemExit("--translation-consensus-topk values must be in [1, 575]")
    try:
        successive_lp_topks = [
            int(value) for value in args.successive_lp_topk.split(",") if value.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--successive-lp-topk must contain integers") from exc
    if any(value < 2 or value >= 576 for value in successive_lp_topks):
        raise SystemExit("--successive-lp-topk values must be in [2, 575]")
    try:
        spatial_prior_weights = [
            float(value)
            for value in args.spatial_prior_weights.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--spatial-prior-weights must contain numbers") from exc
    if any(value < 0 or not np.isfinite(value) for value in spatial_prior_weights):
        raise SystemExit("--spatial-prior-weights must be finite and non-negative")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    names = source_names_for_split(
        args.split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[args.offset : args.offset + args.limit]
    if not names:
        raise SystemExit("selected source range is empty")

    model = device = model_metadata = None
    embedding_model = embedding_metadata = None
    global_matcher_model = global_matcher_metadata = None
    pair_model = pair_metadata = None
    rank_model = rank_feature_names = rank_metadata = None
    context_model = context_metadata = None
    spatial_model = None
    checkpoint = Path(args.checkpoint)
    if args.view == "denoised":
        model, device, model_metadata = load_restorer(checkpoint, device=args.device, state="ema")
        if args.embedding_checkpoint:
            embedding_model, embedding_metadata = load_embedding_checkpoint(
                args.embedding_checkpoint, device=device
            )
        if args.global_matcher_checkpoint:
            if embedding_model is None:
                raise SystemExit("--global-matcher-checkpoint requires --embedding-checkpoint")
            global_matcher_model, global_matcher_metadata = load_global_matcher_checkpoint(
                args.global_matcher_checkpoint, device=device
            )
        if args.pair_checkpoint:
            pair_model, pair_metadata = load_pair_checkpoint(
                args.pair_checkpoint, device=device
            )
        if args.rank_checkpoint:
            rank_model, rank_feature_names, rank_metadata = load_rank_feature_checkpoint(
                args.rank_checkpoint, device=device
            )
        if args.context_checkpoint:
            context_model, context_metadata = load_context_position_checkpoint(
                args.context_checkpoint, device=device
            )
    else:
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        if args.embedding_checkpoint:
            embedding_model, embedding_metadata = load_embedding_checkpoint(
                args.embedding_checkpoint, device=device
            )
        if args.global_matcher_checkpoint:
            if embedding_model is None:
                raise SystemExit("--global-matcher-checkpoint requires --embedding-checkpoint")
            global_matcher_model, global_matcher_metadata = load_global_matcher_checkpoint(
                args.global_matcher_checkpoint, device=device
            )
        if args.pair_checkpoint:
            pair_model, pair_metadata = load_pair_checkpoint(
                args.pair_checkpoint, device=device
            )
        if args.rank_checkpoint:
            rank_model, rank_feature_names, rank_metadata = load_rank_feature_checkpoint(
                args.rank_checkpoint, device=device
            )
        if args.context_checkpoint:
            context_model, context_metadata = load_context_position_checkpoint(
                args.context_checkpoint, device=device
            )
    if args.spatial_prior:
        spatial_model = joblib.load(args.spatial_prior)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    source_reports = []
    run_started = time.perf_counter()
    for source_index, name in enumerate(names):
        source_started = time.perf_counter()
        target_path = Path(args.data_root) / "train" / "targets" / name
        clean_target = _read_rgb(target_path)
        panel_seed_stage = args.panel_seed_stage or f"assembly-{args.panel}"
        seed = per_source_seed(
            args.master_seed, panel_seed_stage, name, args.panel_replica
        )
        panel_started = time.perf_counter()
        panel = make_exact_panel(clean_target, panel=args.panel, seed=seed)
        panel_seconds = time.perf_counter() - panel_started

        view_started = time.perf_counter()
        if args.view == "denoised":
            assert model is not None and device is not None
            solver_tiles = restore_tiles_uint8(
                model, panel.slot_tiles, device, batch_size=args.batch_size
            )
        else:
            solver_tiles = panel.slot_tiles
        view_seconds = time.perf_counter() - view_started
        spatial_placement = (
            spatial_prior_cost(spatial_model, solver_tiles)
            if spatial_model is not None
            else None
        )

        score_started = time.perf_counter()
        bank = build_classical_score_bank(solver_tiles, prefix=args.view, chunk_size=64)
        context_placement = None
        context_inference_seconds = 0.0
        if context_model is not None:
            context_started = time.perf_counter()
            context_row_logits, context_column_logits = context_position_logits(
                context_model, solver_tiles, device=device
            )
            context_placement = position_logits_placement_unary(
                context_row_logits, context_column_logits
            )
            context_inference_seconds = time.perf_counter() - context_started
        c1_names = [name for name in sorted(bank) if not name.endswith("_c2")]
        fused = fuse_ranked_scores(
            bank,
            names=c1_names,
            name=f"{args.view}_C1_equal_rank_fusion",
        )
        bank[fused.name] = fused
        line_score_name = None
        if args.line_seam:
            line_score = line_seam_compatibility(
                solver_tiles,
                prefix=args.view,
                auxiliary_tiles=panel.slot_tiles if args.view == "denoised" else None,
                auxiliary_prefix="raw",
                auxiliary_weight=args.line_seam_auxiliary_weight,
            )
            bank[line_score.name] = line_score
            line_score_name = line_score.name
        edge_bank = build_edge_filter_score_bank(
            solver_tiles, prefix=args.view, chunk_size=64
        )
        bank.update(edge_bank)
        edge_names = sorted(edge_bank)
        edge_fused = fuse_ranked_scores(
            bank,
            names=edge_names,
            name=f"{args.view}_edge_filter_rank_fusion",
        )
        bank[edge_fused.name] = edge_fused
        c1_edge_fused = fuse_ranked_scores(
            bank,
            names=[fused.name, edge_fused.name],
            name=f"{args.view}_C1_edge_rank_fusion",
        )
        bank[c1_edge_fused.name] = c1_edge_fused
        learned_outside_logits = None
        if embedding_model is not None:
            learned_score, learned_outside_logits = learned_compatibility(
                embedding_model,
                solver_tiles,
                device=device,
                name=f"{args.view}_l1_embedding",
            )
            bank[learned_score.name] = learned_score
            learned_fused = fuse_ranked_scores(
                bank,
                names=[*c1_names, learned_score.name],
                name=f"{args.view}_C1_L1_equal_rank_fusion",
            )
            bank[learned_fused.name] = learned_fused
            for learned_weight in (2.0, 4.0):
                weighted_name = f"{args.view}_C1_L1w{int(learned_weight)}_rank_fusion"
                bank[weighted_name] = fuse_ranked_scores(
                    bank,
                    names=[*c1_names, learned_score.name],
                    weights={learned_score.name: learned_weight},
                    name=weighted_name,
                )
            compact_names = [
                f"{args.view}_pbc",
                f"{args.view}_mgc",
                f"{args.view}_tone_l1_w2",
                f"{args.view}_lab_l1_w2",
                learned_score.name,
            ]
            compact_name = f"{args.view}_compact_L1w4_rank_fusion"
            bank[compact_name] = fuse_ranked_scores(
                bank,
                names=compact_names,
                weights={learned_score.name: 4.0},
                name=compact_name,
            )
            if line_score_name is not None:
                line_fusion_name = f"{args.view}_C1_L1w4_line_rank_fusion"
                bank[line_fusion_name] = fuse_ranked_scores(
                    bank,
                    names=[f"{args.view}_C1_L1w4_rank_fusion", line_score_name],
                    weights={line_score_name: args.line_seam_fusion_weight},
                    name=line_fusion_name,
                )
        if global_matcher_model is not None:
            assert embedding_model is not None
            global_score = global_matcher_compatibility(
                global_matcher_model,
                embedding_model,
                solver_tiles,
                device=device,
                name=f"{args.view}_g0_global_matcher",
            )
            bank[global_score.name] = global_score
        if pair_model is not None:
            if args.pair_candidate_policy == "pbc32":
                pair_candidate_names = [f"{args.view}_pbc"]
                pair_candidate_cap = 32
            else:
                pair_candidate_names = [
                    f"{args.view}_pbc",
                    f"{args.view}_mgc",
                    f"{args.view}_tone_l1_w2",
                    f"{args.view}_lab_l1_w2",
                ]
                learned_name = f"{args.view}_l1_embedding"
                if learned_name in bank:
                    pair_candidate_names.append(learned_name)
                pair_candidate_cap = 64
            candidates = candidate_union(
                bank,
                names=pair_candidate_names,
                per_score_top_k=32,
                cap=pair_candidate_cap,
            )
            pair_score = pair_rerank_compatibility(
                pair_model,
                solver_tiles,
                candidates,
                device=device,
                name=f"{args.view}_l0_pair_reranker",
            )
            bank[pair_score.name] = pair_score
        if rank_model is not None:
            assert rank_feature_names is not None and rank_metadata is not None
            candidate_top_k = int(rank_metadata.get("candidate_top_k", 32))
            candidate_cap = int(rank_metadata.get("candidate_cap", 64))
            rank_candidates = candidate_union(
                bank,
                names=rank_feature_names,
                per_score_top_k=candidate_top_k,
                cap=candidate_cap,
            )
            rank_features = candidate_rank_features(
                bank, rank_candidates, names=rank_feature_names
            )
            rank_score = rank_feature_compatibility(
                rank_model,
                rank_features,
                rank_candidates,
                device=device,
                name=f"{args.view}_x0_rank_reranker",
            )
            bank[rank_score.name] = rank_score
            for rank_weight in (1.0, 2.0):
                rank_fusion_name = (
                    f"{args.view}_C1_X0_equal_rank_fusion"
                    if rank_weight == 1.0
                    else f"{args.view}_C1_X0w2_rank_fusion"
                )
                bank[rank_fusion_name] = fuse_ranked_scores(
                    bank,
                    names=[fused.name, rank_score.name],
                    weights={rank_score.name: rank_weight},
                    name=rank_fusion_name,
                )
            learned_name = f"{args.view}_l1_embedding"
            if learned_name in bank:
                combined_name = f"{args.view}_L1_X0_equal_rank_fusion"
                bank[combined_name] = fuse_ranked_scores(
                    bank,
                    names=[learned_name, rank_score.name],
                    name=combined_name,
                )
                full_name = f"{args.view}_C1_L1w4_X0w2_rank_fusion"
                bank[full_name] = fuse_ranked_scores(
                    bank,
                    names=[fused.name, learned_name, rank_score.name],
                    weights={learned_name: 4.0, rank_score.name: 2.0},
                    name=full_name,
                )
        component_scores = _resolve_component_scores(
            args.component_scores,
            view=args.view,
            bank=bank,
            fused_name=fused.name,
        )
        lp_scores = (
            _resolve_component_scores(
                args.lp_scores,
                view=args.view,
                bank=bank,
                fused_name=fused.name,
            )
            if args.lp_scores.strip()
            else []
        )
        global_scores = _resolve_component_scores(
            args.global_score,
            view=args.view,
            bank=bank,
            fused_name=fused.name,
        )
        if len(global_scores) != 1:
            raise SystemExit("--global-score must resolve to exactly one score")
        global_score_name = global_scores[0][1]
        global_score = bank[global_score_name]
        score_seconds = time.perf_counter() - score_started

        retrieval = {
            score_name: retrieval_metrics(score, panel.slot_to_target)
            for score_name, score in bank.items()
        }
        solver_layouts: dict[str, np.ndarray] = {
            "identity": identity_layout(),
            "seeded_random": random_layout(seed),
        }
        context_hungarian_seconds = 0.0
        if context_placement is not None:
            context_started = time.perf_counter()
            positions, slots = linear_sum_assignment(context_placement)
            context_layout = np.empty(576, dtype=np.int32)
            context_layout[positions] = slots
            solver_layouts["context_hungarian"] = context_layout
            context_hungarian_seconds = time.perf_counter() - context_started
        greedy_started = time.perf_counter()
        solver_layouts["greedy"] = greedy_row_major(global_score)
        greedy_seconds = time.perf_counter() - greedy_started
        beam_started = time.perf_counter()
        solver_layouts["beam"] = beam_row_major(
            global_score,
            width=args.beam_width,
            candidate_pool=args.beam_candidate_pool,
        )
        beam_seconds = time.perf_counter() - beam_started
        swap_started = time.perf_counter()
        solver_layouts["greedy_swap"] = swap_refine(
            solver_layouts["greedy"],
            global_score,
            weak_cells=args.swap_weak_cells,
            max_swaps=args.swap_max,
        )
        greedy_swap_seconds = time.perf_counter() - swap_started
        segment_started = time.perf_counter()
        solver_layouts["greedy_segment"] = segment_block_refine(
            solver_layouts["greedy"],
            global_score,
            weak_cells=args.swap_weak_cells,
            max_moves=args.segment_moves,
        )
        segment_seconds = time.perf_counter() - segment_started
        anneal_started = time.perf_counter()
        solver_layouts["greedy_anneal"] = simulated_anneal_swaps(
            solver_layouts["greedy"],
            global_score,
            seed=seed,
            evaluations=args.anneal_evaluations,
        )
        anneal_seconds = time.perf_counter() - anneal_started
        relaxation_seconds: dict[str, float] = {}
        if args.relaxation_iterations > 0:
            for solver_name, inertia, temperature in (
                ("projected_power", 0.0, 0.25),
                ("relaxation_labeling", 0.20, 0.20),
            ):
                relaxation_started = time.perf_counter()
                solver_layouts[solver_name] = relaxation_labeling_solver(
                    global_score,
                    initial=solver_layouts["greedy"],
                    iterations=args.relaxation_iterations,
                    inertia=inertia,
                    temperature=temperature,
                )
                relaxation_seconds[solver_name] = time.perf_counter() - relaxation_started

        solver_seconds = {
            "identity": 0.0,
            "seeded_random": 0.0,
            "greedy": greedy_seconds,
            "beam": beam_seconds,
            "greedy_swap": greedy_seconds + greedy_swap_seconds,
            "greedy_segment": greedy_seconds + segment_seconds,
            "greedy_anneal": greedy_seconds + anneal_seconds,
            **relaxation_seconds,
        }
        if context_placement is not None:
            solver_seconds["context_hungarian"] = (
                context_inference_seconds + context_hungarian_seconds
            )
        component_diagnostics: dict[str, dict] = {}
        for label, score_name in component_scores:
            score = bank[score_name]
            for suffix, include_loops, only_loops, keep_fraction, refine, consensus in (
                ("reciprocal", False, False, 1.0, False, False),
                ("reciprocal_q10", False, False, 0.1, False, False),
                ("reciprocal_q25", False, False, 0.25, False, False),
                ("reciprocal_q50", False, False, 0.5, False, False),
                ("loop_only", True, True, 1.0, False, False),
                ("loops", True, False, 1.0, False, False),
                ("g1_consensus", True, False, 1.0, False, True),
                ("loops_q10", True, False, 0.1, False, False),
                ("loops_q25", True, False, 0.25, False, False),
                ("loops_q50", True, False, 0.5, False, False),
                ("loops_swap", True, False, 1.0, True, False),
            ):
                if refine and args.skip_component_refine:
                    continue
                solver_name = f"component_{label}_{suffix}"
                component_started = time.perf_counter()
                result = reciprocal_component_solver(
                    score,
                    include_verified_loops=include_loops,
                    only_verified_loops=only_loops,
                    proposal_keep_fraction=keep_fraction,
                    consensus=consensus,
                    refine=refine,
                    refine_weak_cells=args.swap_weak_cells,
                    refine_max_swaps=args.swap_max,
                )
                solver_seconds[solver_name] = time.perf_counter() - component_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_reciprocal_edges(
                    score,
                    include_verified_loops=include_loops,
                    only_verified_loops=only_loops,
                )
                proposals = select_confident_edges(
                    proposals, keep_fraction=keep_fraction
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                component_diagnostics[solver_name] = diagnostics
            if spatial_placement is not None:
                for spatial_weight in spatial_prior_weights:
                    solver_name = f"component_{label}_spatial_w{spatial_weight:g}"
                    spatial_started = time.perf_counter()
                    result = reciprocal_component_solver(
                        score,
                        include_verified_loops=True,
                        refine=False,
                        placement_costs=spatial_placement,
                        boundary_weight=spatial_weight,
                    )
                    solver_seconds[solver_name] = time.perf_counter() - spatial_started
                    solver_layouts[solver_name] = result.position_to_slot
                    diagnostics = asdict(result)
                    diagnostics.pop("position_to_slot")
                    component_diagnostics[solver_name] = diagnostics
            for consensus_top_k in translation_consensus_topks:
                solver_name = (
                    f"component_{label}_translation_consensus_k{consensus_top_k}"
                    f"_s{args.translation_consensus_min_support}"
                )
                consensus_started = time.perf_counter()
                result = translation_consensus_component_solver(
                    score,
                    seed_keep_fraction=args.translation_consensus_seed_fraction,
                    top_k=consensus_top_k,
                    min_support=args.translation_consensus_min_support,
                )
                solver_seconds[solver_name] = time.perf_counter() - consensus_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                component_diagnostics[solver_name] = diagnostics
            for lp_top_k in successive_lp_topks:
                for snap_global in (False, True):
                    suffix = "snap" if snap_global else "components"
                    solver_name = f"successive_lp_{label}_k{lp_top_k}_{suffix}"
                    lp_started = time.perf_counter()
                    result = successive_topk_lp_solver(
                        score,
                        top_k=lp_top_k,
                        max_iterations=args.successive_lp_iterations,
                        residual_tolerance=args.successive_lp_residual_tolerance,
                        snap_global=snap_global,
                    )
                    solver_seconds[solver_name] = time.perf_counter() - lp_started
                    solver_layouts[solver_name] = result.position_to_slot
                    diagnostics = asdict(result)
                    diagnostics.pop("position_to_slot")
                    component_diagnostics[solver_name] = diagnostics
            if args.component_placement_beam_width > 1:
                solver_name = f"component_{label}_loops_placebeam"
                component_started = time.perf_counter()
                result = reciprocal_component_solver(
                    score,
                    include_verified_loops=True,
                    refine=False,
                    placement_beam_width=args.component_placement_beam_width,
                    placement_beam_components=args.component_placement_beam_components,
                )
                solver_seconds[solver_name] = time.perf_counter() - component_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_reciprocal_edges(
                    score, include_verified_loops=True
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                component_diagnostics[solver_name] = diagnostics
            for mutual_top_k in mutual_topks:
                solver_name = f"component_{label}_mutual_k{mutual_top_k}"
                component_started = time.perf_counter()
                result = mutual_topk_component_solver(
                    score,
                    top_k=mutual_top_k,
                )
                solver_seconds[solver_name] = time.perf_counter() - component_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_mutual_topk_edges(
                    score, top_k=mutual_top_k
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                component_diagnostics[solver_name] = diagnostics
            for soft_cycle_top_k in soft_cycle_topks:
                solver_name = (
                    f"component_{label}_softcycle_k{soft_cycle_top_k}"
                    f"_p{args.soft_cycle_keep_per_tile}"
                )
                component_started = time.perf_counter()
                result = soft_cycle_component_solver(
                    score,
                    top_k=soft_cycle_top_k,
                    keep_per_tile=args.soft_cycle_keep_per_tile,
                    reciprocal_weight=args.soft_cycle_reciprocal_weight,
                    loop_weight=args.soft_cycle_loop_weight,
                    proposal_keep_fraction=args.soft_cycle_keep_fraction,
                )
                solver_seconds[solver_name] = time.perf_counter() - component_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_soft_cycle_edges(
                    score,
                    top_k=soft_cycle_top_k,
                    keep_per_tile=args.soft_cycle_keep_per_tile,
                    reciprocal_weight=args.soft_cycle_reciprocal_weight,
                    loop_weight=args.soft_cycle_loop_weight,
                )
                proposals = select_confident_edges(
                    proposals, keep_fraction=args.soft_cycle_keep_fraction
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                component_diagnostics[solver_name] = diagnostics
        if learned_outside_logits is not None:
            learned_placement = outside_logits_placement_unary(learned_outside_logits)
            for label, score_name in (
                ("l1", f"{args.view}_l1_embedding"),
                ("l1compact", f"{args.view}_compact_L1w4_rank_fusion"),
            ):
                if score_name not in bank:
                    continue
                for boundary_weight in (0.2, 1.0):
                    solver_name = f"component_{label}_outside_w{boundary_weight:g}"
                    boundary_started = time.perf_counter()
                    result = reciprocal_component_solver(
                        bank[score_name],
                        include_verified_loops=True,
                        refine=False,
                        boundary_weight=boundary_weight,
                        placement_costs=learned_placement,
                    )
                    solver_seconds[solver_name] = time.perf_counter() - boundary_started
                    solver_layouts[solver_name] = result.position_to_slot
                    diagnostics = asdict(result)
                    diagnostics.pop("position_to_slot")
                    proposals = propose_reciprocal_edges(
                        bank[score_name], include_verified_loops=True
                    )
                    diagnostics["proposal_quality"] = _proposal_quality(
                        proposals, panel.slot_to_target
                    )
                    component_diagnostics[solver_name] = diagnostics
        if context_placement is not None and f"{args.view}_l1_embedding" in bank:
            for boundary_weight in (0.05, 0.2, 1.0):
                solver_name = f"component_l1_context_w{boundary_weight:g}"
                context_started = time.perf_counter()
                result = reciprocal_component_solver(
                    bank[f"{args.view}_l1_embedding"],
                    include_verified_loops=True,
                    refine=False,
                    boundary_weight=boundary_weight,
                    placement_costs=context_placement,
                )
                solver_seconds[solver_name] = time.perf_counter() - context_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_reciprocal_edges(
                    bank[f"{args.view}_l1_embedding"], include_verified_loops=True
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                component_diagnostics[solver_name] = diagnostics
            for context_label, context_score_name in (
                ("l1x0", f"{args.view}_L1_X0_equal_rank_fusion"),
                ("l1x0full", f"{args.view}_C1_L1w4_X0w2_rank_fusion"),
            ):
                if context_score_name not in bank:
                    continue
                for boundary_weight in (0.05, 0.2, 1.0):
                    solver_name = (
                        f"component_{context_label}_context_w{boundary_weight:g}"
                    )
                    context_started = time.perf_counter()
                    result = reciprocal_component_solver(
                        bank[context_score_name],
                        include_verified_loops=True,
                        refine=False,
                        boundary_weight=boundary_weight,
                        placement_costs=context_placement,
                    )
                    solver_seconds[solver_name] = time.perf_counter() - context_started
                    solver_layouts[solver_name] = result.position_to_slot
                    diagnostics = asdict(result)
                    diagnostics.pop("position_to_slot")
                    proposals = propose_reciprocal_edges(
                        bank[context_score_name], include_verified_loops=True
                    )
                    diagnostics["proposal_quality"] = _proposal_quality(
                        proposals, panel.slot_to_target
                    )
                    component_diagnostics[solver_name] = diagnostics
        l1_score_name = f"{args.view}_l1_embedding"
        x0_score_name = f"{args.view}_x0_rank_reranker"
        if l1_score_name in bank and x0_score_name in bank:
            for consensus_top_k in (8, 16, 32):
                solver_name = f"component_l1_x0_consensus_k{consensus_top_k}"
                consensus_started = time.perf_counter()
                result = reciprocal_component_solver(
                    bank[l1_score_name],
                    include_verified_loops=True,
                    refine=False,
                    consensus=True,
                    consensus_top_k=consensus_top_k,
                    consensus_max_additions=256,
                    consensus_compatibility=bank[x0_score_name],
                )
                solver_seconds[solver_name] = time.perf_counter() - consensus_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_reciprocal_edges(
                    bank[l1_score_name], include_verified_loops=True
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                diagnostics["consensus_score"] = x0_score_name
                diagnostics["consensus_top_k"] = consensus_top_k
                component_diagnostics[solver_name] = diagnostics
        for label, score_name in lp_scores:
            score = bank[score_name]
            for suffix, only_loops, keep_fraction in (
                ("all", False, 1.0),
                ("q10", False, 0.1),
                ("q25", False, 0.25),
                ("q50", False, 0.5),
                ("loop_only", True, 1.0),
            ):
                solver_name = f"lp_{label}_{suffix}"
                lp_started = time.perf_counter()
                result = weighted_l1_component_solver(
                    score,
                    include_verified_loops=True,
                    only_verified_loops=only_loops,
                    proposal_keep_fraction=keep_fraction,
                )
                solver_seconds[solver_name] = time.perf_counter() - lp_started
                solver_layouts[solver_name] = result.position_to_slot
                diagnostics = asdict(result)
                diagnostics.pop("position_to_slot")
                proposals = propose_reciprocal_edges(
                    score,
                    include_verified_loops=True,
                    only_verified_loops=only_loops,
                )
                proposals = select_confident_edges(
                    proposals, keep_fraction=keep_fraction
                )
                diagnostics["proposal_quality"] = _proposal_quality(
                    proposals, panel.slot_to_target
                )
                component_diagnostics[solver_name] = diagnostics
        if args.four_side_refine_iterations > 0:
            refine_seeds = [
                value.strip()
                for value in args.four_side_refine_seeds.split(",")
                if value.strip()
            ]
            for seed_name in refine_seeds:
                if seed_name not in solver_layouts:
                    raise SystemExit(
                        f"unknown --four-side-refine-seeds entry {seed_name!r}; "
                        f"available: {sorted(solver_layouts)}"
                    )
                solver_name = f"{seed_name}_four_side"
                refine_started = time.perf_counter()
                solver_layouts[solver_name] = four_side_hungarian_refine(
                    solver_layouts[seed_name],
                    global_score,
                    iterations=args.four_side_refine_iterations,
                )
                solver_seconds[solver_name] = (
                    solver_seconds[seed_name] + time.perf_counter() - refine_started
                )
        if args.anneal_refine_evaluations > 0:
            anneal_seeds = [
                value.strip()
                for value in args.anneal_refine_seeds.split(",")
                if value.strip()
            ]
            for seed_index, seed_name in enumerate(anneal_seeds):
                if seed_name not in solver_layouts:
                    raise SystemExit(
                        f"unknown --anneal-refine-seeds entry {seed_name!r}; "
                        f"available: {sorted(solver_layouts)}"
                    )
                solver_name = f"{seed_name}_anneal_long"
                refine_started = time.perf_counter()
                solver_layouts[solver_name] = simulated_anneal_swaps(
                    solver_layouts[seed_name],
                    global_score,
                    seed=seed + 1009 * (seed_index + 1),
                    evaluations=args.anneal_refine_evaluations,
                )
                solver_seconds[solver_name] = (
                    solver_seconds[seed_name] + time.perf_counter() - refine_started
                )
                if args.anneal_refine_mixed:
                    mixed_name = f"{seed_name}_anneal_mixed"
                    mixed_started = time.perf_counter()
                    solver_layouts[mixed_name] = simulated_anneal_mixed(
                        solver_layouts[seed_name],
                        global_score,
                        seed=seed + 2003 * (seed_index + 1),
                        evaluations=args.anneal_refine_evaluations,
                    )
                    solver_seconds[mixed_name] = (
                        solver_seconds[seed_name]
                        + time.perf_counter()
                        - mixed_started
                    )
        if args.genetic_generations > 0:
            genetic_seed_names = [
                value.strip()
                for value in args.genetic_seeds.split(",")
                if value.strip()
            ]
            missing = [name for name in genetic_seed_names if name not in solver_layouts]
            if missing:
                raise SystemExit(
                    f"unknown --genetic-seeds entries {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            genetic_started = time.perf_counter()
            solver_layouts["segment_genetic"] = segment_preserving_genetic_solver(
                [solver_layouts[name] for name in genetic_seed_names],
                global_score,
                seed=seed + 4001,
                population_size=args.genetic_population,
                generations=args.genetic_generations,
                elite_size=args.genetic_elite,
            )
            solver_seconds["segment_genetic"] = (
                max(solver_seconds[name] for name in genetic_seed_names)
                + time.perf_counter()
                - genetic_started
            )
        if args.lns_iterations > 0:
            lns_seed_names = [
                value.strip()
                for value in args.lns_seeds.split(",")
                if value.strip()
            ]
            missing = [name for name in lns_seed_names if name not in solver_layouts]
            if missing:
                raise SystemExit(
                    f"unknown --lns-seeds entries {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            for seed_index, seed_name in enumerate(lns_seed_names):
                solver_name = f"{seed_name}_lns"
                lns_started = time.perf_counter()
                solver_layouts[solver_name] = large_neighborhood_reassign(
                    solver_layouts[seed_name],
                    global_score,
                    seed=seed + 5003 * (seed_index + 1),
                    iterations=args.lns_iterations,
                    subset_size=args.lns_subset_size,
                )
                solver_seconds[solver_name] = (
                    solver_seconds[seed_name] + time.perf_counter() - lns_started
                )
        if args.multi_phase_rl_phases > 0:
            rl_seed_names = [
                value.strip()
                for value in args.multi_phase_rl_seeds.split(",")
                if value.strip()
            ]
            missing = [name for name in rl_seed_names if name not in solver_layouts]
            if missing:
                raise SystemExit(
                    f"unknown --multi-phase-rl-seeds entries {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            for seed_name in rl_seed_names:
                solver_name = f"{seed_name}_multi_phase_rl"
                rl_started = time.perf_counter()
                solver_layouts[solver_name] = multi_phase_relaxation_solver(
                    global_score,
                    initial=solver_layouts[seed_name],
                    top_k=args.multi_phase_rl_topk,
                    phases=args.multi_phase_rl_phases,
                    iterations_per_phase=args.multi_phase_rl_iterations,
                    anchor_batch=args.multi_phase_rl_anchor_batch,
                )
                solver_seconds[solver_name] = (
                    solver_seconds[seed_name] + time.perf_counter() - rl_started
                )
        if args.faithful_rl_phases > 0:
            faithful_started = time.perf_counter()
            solver_layouts["faithful_multi_phase_rl"] = (
                faithful_multi_phase_relaxation_solver(
                    global_score,
                    top_k=args.faithful_rl_topk,
                    phases=args.faithful_rl_phases,
                    convergence_threshold=args.faithful_rl_convergence,
                    max_iterations=args.faithful_rl_max_iterations,
                    anchor_probability=args.faithful_rl_anchor_probability,
                )
            )
            solver_seconds["faithful_multi_phase_rl"] = (
                time.perf_counter() - faithful_started
            )
        if args.particle_beam_particles > 0:
            particle_seed_names = [
                value.strip()
                for value in args.particle_beam_seeds.split(",")
                if value.strip()
            ]
            missing = [
                name for name in particle_seed_names if name not in solver_layouts
            ]
            if missing:
                raise SystemExit(
                    f"unknown --particle-beam-seeds entries {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            particle_started = time.perf_counter()
            particle_result = particle_beam_solver(
                global_score,
                seed_layouts=[solver_layouts[name] for name in particle_seed_names],
                particles=args.particle_beam_particles,
                top_k=args.particle_beam_topk,
                anchor_hypotheses=args.particle_beam_anchor_hypotheses,
                frontier_limit=args.particle_beam_frontier_limit,
            )
            particle_name = (
                f"particle_beam_p{args.particle_beam_particles}"
                f"_k{args.particle_beam_topk}"
            )
            solver_layouts[particle_name] = particle_result.position_to_slot
            solver_seconds[particle_name] = time.perf_counter() - particle_started
            diagnostics = asdict(particle_result)
            diagnostics.pop("position_to_slot")
            diagnostics["seed_names"] = particle_seed_names
            component_diagnostics[particle_name] = diagnostics
        order2_consensus_score = None
        order2_consensus_result = None
        order2_consensus_score_name = None
        if args.order2_consensus_topk > 0:
            consensus_started = time.perf_counter()
            top_k = args.order2_consensus_topk

            def topk_rows(matrix: np.ndarray) -> np.ndarray:
                finite = np.asarray(matrix, dtype=np.float64).copy()
                np.fill_diagonal(finite, np.inf)
                return np.argsort(finite, axis=1, kind="stable")[:, :top_k]

            order2_consensus_result = discover_order2_consensus(
                topk_rows(global_score.right),
                topk_rows(global_score.down),
                tile_count=576,
                min_support=args.order2_consensus_min_support,
            )
            consensus_right = np.asarray(global_score.right, dtype=np.float32).copy()
            consensus_down = np.asarray(global_score.down, dtype=np.float32).copy()
            for proposal in order2_consensus_result.proposals:
                edge = proposal.edge
                strength = min(
                    proposal.support / args.order2_consensus_min_support,
                    2.0,
                )
                reward = args.order2_consensus_bonus * strength
                matrix = consensus_right if edge.dx else consensus_down
                matrix[edge.first, edge.second] -= reward
            order2_consensus_score_name = (
                f"{global_score.name}_order2_consensus"
                f"_k{top_k}_s{args.order2_consensus_min_support}"
            )
            order2_consensus_score = CompatibilityMatrices(
                name=order2_consensus_score_name,
                right=consensus_right,
                down=consensus_down,
            )
            retrieval[order2_consensus_score_name] = retrieval_metrics(
                order2_consensus_score, panel.slot_to_target
            )
            solver_seconds["order2_consensus_inference"] = (
                time.perf_counter() - consensus_started
            )
        if args.qap_iterations > 0:
            qap_seed_names = [
                value.strip() for value in args.qap_seeds.split(",") if value.strip()
            ]
            missing = [name for name in qap_seed_names if name not in solver_layouts]
            if missing:
                raise SystemExit(
                    f"unknown --qap-seeds entries {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            qap_initializations: list[tuple[str, np.ndarray | None]] = (
                [(name, solver_layouts[name]) for name in qap_seed_names]
                if qap_seed_names
                else [("barycenter", None)]
            )
            for qap_index, (seed_name, initial_layout) in enumerate(qap_initializations):
                qap_started = time.perf_counter()
                qap_result = directional_qap(
                    global_score,
                    initial=initial_layout,
                    iterations=args.qap_iterations,
                    restarts=args.qap_restarts,
                    seed=seed + 7001 * (qap_index + 1),
                    boundary_weight=args.qap_boundary_weight,
                    initial_weight=args.qap_initial_weight,
                    noisy_components=args.qap_noisy_components,
                    noise_scale=args.qap_noise_scale,
                    refine_swaps=args.qap_refine_swaps,
                )
                qap_name = f"qap_{seed_name}"
                solver_layouts[qap_name] = qap_result.position_to_slot
                solver_seconds[qap_name] = time.perf_counter() - qap_started
                diagnostics = asdict(qap_result)
                diagnostics.pop("position_to_slot")
                component_diagnostics[qap_name] = diagnostics
                if order2_consensus_score is not None:
                    consensus_qap_started = time.perf_counter()
                    consensus_qap_result = directional_qap(
                        order2_consensus_score,
                        initial=initial_layout,
                        iterations=args.qap_iterations,
                        restarts=args.qap_restarts,
                        seed=seed + 7001 * (qap_index + 1),
                        boundary_weight=args.qap_boundary_weight,
                        initial_weight=args.qap_initial_weight,
                        noisy_components=args.qap_noisy_components,
                        noise_scale=args.qap_noise_scale,
                        refine_swaps=args.qap_refine_swaps,
                    )
                    consensus_qap_name = f"qap_order2_consensus_{seed_name}"
                    solver_layouts[consensus_qap_name] = (
                        consensus_qap_result.position_to_slot
                    )
                    solver_seconds[consensus_qap_name] = (
                        solver_seconds["order2_consensus_inference"]
                        + time.perf_counter()
                        - consensus_qap_started
                    )
                    consensus_diagnostics = asdict(consensus_qap_result)
                    consensus_diagnostics.pop("position_to_slot")
                    consensus_diagnostics["order2_consensus"] = {
                        "top_k": args.order2_consensus_topk,
                        "min_support": args.order2_consensus_min_support,
                        "bonus": args.order2_consensus_bonus,
                        "score_name": order2_consensus_score_name,
                        "proposal_quality": _order2_consensus_quality(
                            order2_consensus_result, panel.slot_to_target
                        ),
                    }
                    component_diagnostics[consensus_qap_name] = (
                        consensus_diagnostics
                    )
        if args.rigid_component_qap_projection:
            component_seed_name = args.rigid_projection_component_seed
            reference_name = args.rigid_projection_reference
            missing = [
                name
                for name in (component_seed_name, reference_name)
                if name not in solver_layouts
            ]
            if missing:
                raise SystemExit(
                    f"rigid projection requires completed layouts {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            component_score_by_label = dict(component_scores)
            if "l1" not in component_score_by_label:
                raise SystemExit(
                    "rigid projection precommit requires --component-scores to include l1"
                )
            rigid_started = time.perf_counter()
            rigid_result = rigid_soft_cycle_qap_projection(
                bank[component_score_by_label["l1"]],
                global_score,
                solver_layouts[reference_name],
                top_k=8,
                keep_per_tile=1,
                reciprocal_weight=0.35,
                loop_weight=1.0,
                proposal_keep_fraction=0.5,
                reference_weight=args.rigid_projection_reference_weight,
                beam_width=args.rigid_projection_beam_width,
                beam_components=args.rigid_projection_beam_components,
                translations_per_state=args.rigid_projection_translations,
            )
            rigid_name = f"rigid_{reference_name}"
            solver_layouts[rigid_name] = rigid_result.position_to_slot
            solver_seconds[rigid_name] = (
                solver_seconds[reference_name]
                + time.perf_counter()
                - rigid_started
            )
            rigid_diagnostics = asdict(rigid_result)
            rigid_diagnostics.pop("position_to_slot")
            rigid_diagnostics.update(
                {
                    "component_seed": component_seed_name,
                    "reference_layout": reference_name,
                    "component_score": component_score_by_label["l1"],
                    "placement_score": global_score_name,
                    "post_projection_qap": False,
                }
            )
            component_diagnostics[rigid_name] = rigid_diagnostics
        if args.cpsat_time_seconds > 0:
            cpsat_seed_names = [
                value.strip()
                for value in args.cpsat_seeds.split(",")
                if value.strip()
            ]
            missing = [name for name in cpsat_seed_names if name not in solver_layouts]
            if missing:
                raise SystemExit(
                    f"unknown --cpsat-seeds entries {missing}; "
                    f"available: {sorted(solver_layouts)}"
                )
            for cpsat_index, seed_name in enumerate(cpsat_seed_names):
                cpsat_started = time.perf_counter()
                cpsat_result = topk_cpsat_grid_solver(
                    global_score,
                    top_k=args.cpsat_topk,
                    max_time_seconds=args.cpsat_time_seconds,
                    workers=args.cpsat_workers,
                    seed=(seed + 9001 * (cpsat_index + 1)) % 2_147_483_648,
                    max_square_terms=args.cpsat_square_terms,
                    initial_position_to_slot=solver_layouts[seed_name],
                )
                cpsat_name = f"cpsat_{seed_name}"
                solver_layouts[cpsat_name] = cpsat_result.position_to_slot
                solver_seconds[cpsat_name] = time.perf_counter() - cpsat_started
                component_diagnostics[cpsat_name] = cpsat_result.diagnostics
        solvers = {}
        for solver_name, position_to_slot in solver_layouts.items():
            layout = layout_metrics(position_to_slot, panel.slot_to_target)
            image = predicted_image_metrics(position_to_slot, solver_tiles, clean_target)
            solvers[solver_name] = {
                "layout": layout,
                "image": image,
                "seconds": solver_seconds[solver_name],
                "position_to_slot": position_to_slot.tolist(),
            }
            if solver_name in component_diagnostics:
                solvers[solver_name]["diagnostics"] = component_diagnostics[solver_name]
            if preview_dir is not None and source_index == 0:
                preview = merge_tiles_numpy(solver_tiles[position_to_slot])
                Image.fromarray(preview, mode="RGB").save(
                    preview_dir / f"{Path(name).stem}_{args.panel}_{args.view}_{solver_name}.png"
                )
        source_reports.append(
            {
                "source": name,
                "seed": seed,
                "slot_to_target": panel.slot_to_target.tolist(),
                "retrieval": retrieval,
                "solvers": solvers,
                "timings": {
                    "panel": panel_seconds,
                    "view": view_seconds,
                    "score_bank": score_seconds,
                    "total": time.perf_counter() - source_started,
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "assembly_source_complete",
                    "index": source_index + 1,
                    "count": len(names),
                    "source": name,
                    "panel": args.panel,
                    "view": args.view,
                    "fused_recall_at_1": retrieval[fused.name]["combined"]["recall_at_1"],
                    "greedy_adjacency": solvers["greedy"]["layout"]["combined_adjacency"],
                    "beam_adjacency": solvers["beam"]["layout"]["combined_adjacency"],
                    "best_graph_adjacency": max(
                        solver["layout"]["combined_adjacency"]
                        for solver_name, solver in solvers.items()
                        if solver_name.startswith(("component_", "lp_"))
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "schema_version": 1,
        "kind": "assembly_baseline_exact_panel_report",
        "research_only": True,
        "split": args.split,
        "panel": args.panel,
        "view": args.view,
        "master_seed": args.master_seed,
        "panel_seed_stage": args.panel_seed_stage or f"assembly-{args.panel}",
        "panel_replica": args.panel_replica,
        "offset": args.offset,
        "limit": args.limit,
        "source_names": names,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "quarantine": str(args.quarantine),
        "quarantine_sha256": _sha256(Path(args.quarantine)),
        "checkpoint": str(checkpoint) if args.view == "denoised" else None,
        "checkpoint_sha256": _sha256(checkpoint) if args.view == "denoised" else None,
        "model_metadata": model_metadata,
        "embedding_checkpoint": args.embedding_checkpoint,
        "embedding_checkpoint_sha256": (
            _sha256(Path(args.embedding_checkpoint)) if args.embedding_checkpoint else None
        ),
        "embedding_metadata": embedding_metadata,
        "global_matcher_checkpoint": args.global_matcher_checkpoint,
        "global_matcher_checkpoint_sha256": (
            _sha256(Path(args.global_matcher_checkpoint))
            if args.global_matcher_checkpoint
            else None
        ),
        "global_matcher_metadata": global_matcher_metadata,
        "pair_checkpoint": args.pair_checkpoint,
        "pair_checkpoint_sha256": (
            _sha256(Path(args.pair_checkpoint)) if args.pair_checkpoint else None
        ),
        "pair_metadata": pair_metadata,
        "rank_checkpoint": args.rank_checkpoint,
        "rank_checkpoint_sha256": (
            _sha256(Path(args.rank_checkpoint)) if args.rank_checkpoint else None
        ),
        "rank_metadata": rank_metadata,
        "rank_feature_names": rank_feature_names,
        "context_checkpoint": args.context_checkpoint,
        "context_checkpoint_sha256": (
            _sha256(Path(args.context_checkpoint)) if args.context_checkpoint else None
        ),
        "context_metadata": context_metadata,
        "spatial_prior": args.spatial_prior,
        "spatial_prior_sha256": (
            _sha256(Path(args.spatial_prior)) if args.spatial_prior else None
        ),
        "line_seam": {
            "enabled": args.line_seam,
            "auxiliary_weight": args.line_seam_auxiliary_weight,
            "fusion_weight": args.line_seam_fusion_weight,
        },
        "score_bank": sorted(source_reports[0]["retrieval"]),
        "solver_configs": {
            "beam_width": args.beam_width,
            "beam_candidate_pool": args.beam_candidate_pool,
            "swap_weak_cells": args.swap_weak_cells,
            "swap_max": args.swap_max,
            "segment_moves": args.segment_moves,
            "anneal_evaluations": args.anneal_evaluations,
            "anneal_refine_seeds": args.anneal_refine_seeds,
            "anneal_refine_evaluations": args.anneal_refine_evaluations,
            "anneal_refine_mixed": args.anneal_refine_mixed,
            "relaxation_iterations": args.relaxation_iterations,
            "four_side_refine_iterations": args.four_side_refine_iterations,
            "four_side_refine_seeds": args.four_side_refine_seeds,
            "genetic_generations": args.genetic_generations,
            "genetic_population": args.genetic_population,
            "genetic_elite": args.genetic_elite,
            "genetic_seeds": args.genetic_seeds,
            "lns_iterations": args.lns_iterations,
            "lns_subset_size": args.lns_subset_size,
            "lns_seeds": args.lns_seeds,
            "multi_phase_rl_phases": args.multi_phase_rl_phases,
            "multi_phase_rl_topk": args.multi_phase_rl_topk,
            "multi_phase_rl_iterations": args.multi_phase_rl_iterations,
            "multi_phase_rl_anchor_batch": args.multi_phase_rl_anchor_batch,
            "multi_phase_rl_seeds": args.multi_phase_rl_seeds,
            "faithful_rl_phases": args.faithful_rl_phases,
            "faithful_rl_topk": args.faithful_rl_topk,
            "faithful_rl_max_iterations": args.faithful_rl_max_iterations,
            "faithful_rl_convergence": args.faithful_rl_convergence,
            "faithful_rl_anchor_probability": args.faithful_rl_anchor_probability,
            "particle_beam_particles": args.particle_beam_particles,
            "particle_beam_topk": args.particle_beam_topk,
            "particle_beam_anchor_hypotheses": args.particle_beam_anchor_hypotheses,
            "particle_beam_frontier_limit": args.particle_beam_frontier_limit,
            "particle_beam_seeds": args.particle_beam_seeds,
            "qap_iterations": args.qap_iterations,
            "qap_restarts": args.qap_restarts,
            "qap_boundary_weight": args.qap_boundary_weight,
            "qap_refine_swaps": args.qap_refine_swaps,
            "qap_initial_weight": args.qap_initial_weight,
            "qap_noisy_components": args.qap_noisy_components,
            "qap_noise_scale": args.qap_noise_scale,
            "order2_consensus_topk": args.order2_consensus_topk,
            "order2_consensus_min_support": args.order2_consensus_min_support,
            "order2_consensus_bonus": args.order2_consensus_bonus,
            "qap_seeds": args.qap_seeds,
            "rigid_component_qap_projection": args.rigid_component_qap_projection,
            "rigid_projection_component_seed": args.rigid_projection_component_seed,
            "rigid_projection_reference": args.rigid_projection_reference,
            "rigid_projection_reference_weight": args.rigid_projection_reference_weight,
            "rigid_projection_beam_width": args.rigid_projection_beam_width,
            "rigid_projection_beam_components": args.rigid_projection_beam_components,
            "rigid_projection_translations": args.rigid_projection_translations,
            "cpsat_time_seconds": args.cpsat_time_seconds,
            "cpsat_topk": args.cpsat_topk,
            "cpsat_workers": args.cpsat_workers,
            "cpsat_square_terms": args.cpsat_square_terms,
            "cpsat_seeds": args.cpsat_seeds,
            "global_score": args.global_score,
            "component_placement_beam_width": args.component_placement_beam_width,
            "component_placement_beam_components": args.component_placement_beam_components,
            "mutual_topk": mutual_topks,
            "translation_consensus_topk": translation_consensus_topks,
            "translation_consensus_min_support": args.translation_consensus_min_support,
            "translation_consensus_seed_fraction": args.translation_consensus_seed_fraction,
            "successive_lp_topk": successive_lp_topks,
            "successive_lp_iterations": args.successive_lp_iterations,
            "successive_lp_residual_tolerance": args.successive_lp_residual_tolerance,
            "spatial_prior_weights": spatial_prior_weights,
            "pair_candidate_policy": args.pair_candidate_policy,
            "component_scores": args.component_scores,
            "component_refine": not args.skip_component_refine,
            "lp_scores": args.lp_scores,
            "c1_fusion_scores": c1_names,
        },
        "sources": source_reports,
        "macro": _macro_report(source_reports),
        "seconds": time.perf_counter() - run_started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "assembly_baseline_complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
