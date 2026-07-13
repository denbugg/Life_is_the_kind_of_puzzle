#!/usr/bin/env python3
"""Input-only real assembly prediction with target-only post-hoc scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from puzzle_assembly.compatibility import (
    build_classical_score_bank,
    build_edge_filter_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import (
    mutual_topk_component_solver,
    reciprocal_component_solver,
    soft_cycle_component_solver,
    weighted_l1_component_solver,
)
from puzzle_assembly.cpsat import topk_cpsat_grid_solver
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
from puzzle_assembly.metrics import predicted_image_metrics
from puzzle_assembly.particle import particle_beam_solver
from puzzle_assembly.protocol import source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_assembly.solvers import (
    faithful_multi_phase_relaxation_solver,
    large_neighborhood_reassign,
    multi_phase_relaxation_solver,
    outside_logits_placement_unary,
    position_logits_placement_unary,
    simulated_anneal_mixed,
    simulated_anneal_swaps,
)
from puzzle_assembly.spatial_prior import spatial_prior_cost
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint")
    parser.add_argument("--global-matcher-checkpoint")
    parser.add_argument(
        "--embedding-view",
        choices=["denoised", "raw"],
        default="denoised",
        help="tile view consumed by the optional learned edge scorer",
    )
    parser.add_argument("--pair-checkpoint")
    parser.add_argument("--rank-checkpoint")
    parser.add_argument("--context-checkpoint")
    parser.add_argument("--spatial-prior")
    parser.add_argument("--spatial-prior-weights", default="0.05,0.2")
    parser.add_argument("--line-seam", action="store_true")
    parser.add_argument("--line-seam-auxiliary-weight", type=float, default=0.35)
    parser.add_argument("--line-seam-fusion-weight", type=float, default=0.5)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--split", choices=["assembly_cal", "edge_development"], default="assembly_cal")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--soft-cycle-topk",
        default="",
        help="optional comma-separated k values for the soft-cycle solver",
    )
    parser.add_argument(
        "--soft-cycle-scores",
        default="denoised_c1,l1",
        help="comma-separated aliases: raw_c1, denoised_c1, cross_c1, cross_c1_dn2, l1",
    )
    parser.add_argument("--soft-cycle-keep-per-tile", type=int, default=1)
    parser.add_argument("--soft-cycle-keep-fraction", type=float, default=0.5)
    parser.add_argument("--soft-cycle-loop-weight", type=float, default=1.0)
    parser.add_argument("--soft-cycle-reciprocal-weight", type=float, default=0.35)
    parser.add_argument("--anneal-refine-evaluations", type=int, default=0)
    parser.add_argument("--anneal-refine-mixed", action="store_true")
    parser.add_argument(
        "--anneal-refine-seeds",
        default="",
        help="comma-separated layout labels produced by this evaluator",
    )
    parser.add_argument(
        "--anneal-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--lns-iterations", type=int, default=0)
    parser.add_argument("--lns-subset-size", type=int, default=128)
    parser.add_argument("--lns-seeds", default="")
    parser.add_argument(
        "--lns-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--multi-phase-rl-phases", type=int, default=0)
    parser.add_argument("--multi-phase-rl-topk", type=int, default=8)
    parser.add_argument("--multi-phase-rl-iterations", type=int, default=3)
    parser.add_argument("--multi-phase-rl-anchor-batch", type=int, default=48)
    parser.add_argument("--multi-phase-rl-seeds", default="")
    parser.add_argument(
        "--multi-phase-rl-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--faithful-rl-phases", type=int, default=0)
    parser.add_argument("--faithful-rl-topk", type=int, default=17)
    parser.add_argument("--faithful-rl-max-iterations", type=int, default=48)
    parser.add_argument("--faithful-rl-convergence", type=float, default=1e-4)
    parser.add_argument("--faithful-rl-anchor-probability", type=float, default=0.70)
    parser.add_argument(
        "--faithful-rl-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--particle-beam-particles", type=int, default=0)
    parser.add_argument("--particle-beam-topk", type=int, default=3)
    parser.add_argument("--particle-beam-anchor-hypotheses", type=int, default=4)
    parser.add_argument("--particle-beam-frontier-limit", type=int, default=24)
    parser.add_argument("--particle-beam-seeds", default="")
    parser.add_argument(
        "--particle-beam-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--qap-iterations", type=int, default=0)
    parser.add_argument("--qap-restarts", type=int, default=1)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.0)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--qap-initial-weight", type=float, default=0.75)
    parser.add_argument("--qap-noisy-components", type=int, default=3)
    parser.add_argument("--qap-noise-scale", type=float, default=1.0)
    parser.add_argument("--qap-seeds", default="")
    parser.add_argument(
        "--qap-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--cpsat-time-seconds", type=float, default=0.0)
    parser.add_argument("--cpsat-topk", type=int, default=8)
    parser.add_argument("--cpsat-workers", type=int, default=1)
    parser.add_argument("--cpsat-square-terms", type=int, default=2048)
    parser.add_argument("--cpsat-seeds", default="")
    parser.add_argument(
        "--cpsat-score",
        choices=["denoised_c1", "cross_c1", "cross_c1_dn2", "l1", "l1w4", "cross_l1w4", "line", "l1w4line"],
        default="l1w4",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-dir")
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


def _layout_pair_cost(layout: np.ndarray, score: object) -> float:
    grid = np.asarray(layout, dtype=np.int32).reshape(24, 24)
    right = score.right[grid[:, :-1], grid[:, 1:]]
    down = score.down[grid[:-1, :], grid[1:, :]]
    values = np.concatenate([right.ravel(), down.ravel()])
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("inf")


def _predict_input_only(
    input_path: Path,
    *,
    restorer: object,
    embedding_model: object | None,
    global_matcher_model: object | None,
    pair_model: object | None,
    rank_model: object | None,
    rank_feature_names: list[str] | None,
    rank_metadata: dict | None,
    context_model: object | None,
    embedding_view: str,
    device: object,
    batch_size: int,
    line_seam_enabled: bool,
    line_seam_auxiliary_weight: float,
    line_seam_fusion_weight: float,
    soft_cycle_topks: list[int],
    soft_cycle_scores: list[str],
    soft_cycle_keep_per_tile: int,
    soft_cycle_keep_fraction: float,
    soft_cycle_loop_weight: float,
    soft_cycle_reciprocal_weight: float,
    anneal_refine_evaluations: int,
    anneal_refine_seeds: list[str],
    anneal_score_alias: str,
    anneal_refine_mixed: bool,
    spatial_model: object | None,
    spatial_prior_weights: list[float],
    lns_iterations: int,
    lns_subset_size: int,
    lns_seeds: list[str],
    lns_score_alias: str,
    multi_phase_rl_phases: int,
    multi_phase_rl_topk: int,
    multi_phase_rl_iterations: int,
    multi_phase_rl_anchor_batch: int,
    multi_phase_rl_seeds: list[str],
    multi_phase_rl_score_alias: str,
    faithful_rl_phases: int,
    faithful_rl_topk: int,
    faithful_rl_max_iterations: int,
    faithful_rl_convergence: float,
    faithful_rl_anchor_probability: float,
    faithful_rl_score_alias: str,
    particle_beam_particles: int,
    particle_beam_topk: int,
    particle_beam_anchor_hypotheses: int,
    particle_beam_frontier_limit: int,
    particle_beam_seeds: list[str],
    particle_beam_score_alias: str,
    qap_iterations: int,
    qap_restarts: int,
    qap_boundary_weight: float,
    qap_refine_swaps: int,
    qap_initial_weight: float,
    qap_noisy_components: int,
    qap_noise_scale: float,
    qap_seeds: list[str],
    qap_score_alias: str,
    cpsat_time_seconds: float,
    cpsat_topk: int,
    cpsat_workers: int,
    cpsat_square_terms: int,
    cpsat_seeds: list[str],
    cpsat_score_alias: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, dict[str, dict[str, float]]]:
    """Never accepts or opens a target path/pixel array."""
    started = time.perf_counter()
    input_image = _read_rgb(input_path)
    raw_tiles = split_tiles_numpy(input_image)
    denoised = restore_tiles_uint8(restorer, raw_tiles, device, batch_size=batch_size)
    spatial_placement = (
        spatial_prior_cost(spatial_model, denoised)
        if spatial_model is not None
        else None
    )
    score_started = time.perf_counter()
    bank = build_classical_score_bank(raw_tiles, prefix="raw", chunk_size=64)
    bank.update(build_classical_score_bank(denoised, prefix="denoised", chunk_size=64))
    c1_scores = {}
    edge_scores = {}
    for score_view in ("raw", "denoised"):
        c1_names = [
            name
            for name in sorted(bank)
            if name.startswith(f"{score_view}_") and not name.endswith("_c2")
        ]
        c1 = fuse_ranked_scores(
            bank, names=c1_names, name=f"{score_view}_C1_equal_rank_fusion"
        )
        bank[c1.name] = c1
        c1_scores[score_view] = c1
        edge_bank = build_edge_filter_score_bank(
            raw_tiles if score_view == "raw" else denoised,
            prefix=score_view,
            chunk_size=64,
        )
        bank.update(edge_bank)
        edge_fused = fuse_ranked_scores(
            bank,
            names=sorted(edge_bank),
            name=f"{score_view}_edge_filter_rank_fusion",
        )
        bank[edge_fused.name] = edge_fused
        c1_edge_fused = fuse_ranked_scores(
            bank,
            names=[c1.name, edge_fused.name],
            name=f"{score_view}_C1_edge_rank_fusion",
        )
        bank[c1_edge_fused.name] = c1_edge_fused
        edge_scores[score_view] = {
            "sobel": f"{score_view}_sobel_l1_w2",
            "binary": f"{score_view}_binary_edge_hamming_w2",
            "edgefusion": edge_fused.name,
            "fusionedge": c1_edge_fused.name,
        }
    line_score_name = None
    if line_seam_enabled:
        line_score = line_seam_compatibility(
            denoised,
            prefix="denoised",
            auxiliary_tiles=raw_tiles,
            auxiliary_prefix="raw",
            auxiliary_weight=line_seam_auxiliary_weight,
        )
        bank[line_score.name] = line_score
        line_score_name = line_score.name
    outside_logits = None
    cross_c1_scores = {}
    for label, denoised_weight in (("cross_c1", 1.0), ("cross_c1_dn2", 2.0)):
        score = fuse_ranked_scores(
            bank,
            names=[c1_scores["raw"].name, c1_scores["denoised"].name],
            weights={c1_scores["denoised"].name: denoised_weight},
            name=f"raw_denoised_C1_dn{denoised_weight:g}_rank_fusion",
        )
        bank[score.name] = score
        cross_c1_scores[label] = score
    l1_name = None
    compact_name = None
    embedding_score_names: dict[str, str] = {}
    if embedding_model is not None:
        embedding_tiles = denoised if embedding_view == "denoised" else raw_tiles
        l1_name = f"{embedding_view}_l1_embedding"
        l1, outside_logits = learned_compatibility(
            embedding_model, embedding_tiles, device=device, name=l1_name
        )
        bank[l1.name] = l1
        embedding_score_names["component_l1"] = l1.name
        for label, learned_weight in (
            ("component_l1fusion", 1.0),
            ("component_l1w2", 2.0),
            ("component_l1w4", 4.0),
        ):
            fusion_name = (
                f"{embedding_view}_C1_L1_equal_rank_fusion"
                if learned_weight == 1.0
                else f"{embedding_view}_C1_L1w{int(learned_weight)}_rank_fusion"
            )
            bank[fusion_name] = fuse_ranked_scores(
                bank,
                names=[c1_scores[embedding_view].name, l1.name],
                weights={l1.name: learned_weight},
                name=fusion_name,
            )
            embedding_score_names[label] = fusion_name
        compact_names = [
            f"{embedding_view}_pbc",
            f"{embedding_view}_mgc",
            f"{embedding_view}_tone_l1_w2",
            f"{embedding_view}_lab_l1_w2",
            l1.name,
        ]
        compact_name = f"{embedding_view}_compact_L1w4_rank_fusion"
        compact = fuse_ranked_scores(
            bank,
            names=compact_names,
            weights={l1.name: 4.0},
            name=compact_name,
        )
        bank[compact.name] = compact
        embedding_score_names["component_l1compact"] = compact.name
        cross_l1w4_name = "raw_denoised_C1_L1w4_rank_fusion"
        bank[cross_l1w4_name] = fuse_ranked_scores(
            bank,
            names=[cross_c1_scores["cross_c1_dn2"].name, l1.name],
            weights={l1.name: 4.0},
            name=cross_l1w4_name,
        )
        embedding_score_names["component_cross_l1w4"] = cross_l1w4_name
        if line_score_name is not None:
            line_fusion_name = "denoised_C1_L1w4_line_rank_fusion"
            bank[line_fusion_name] = fuse_ranked_scores(
                bank,
                names=[embedding_score_names["component_l1w4"], line_score_name],
                weights={line_score_name: line_seam_fusion_weight},
                name=line_fusion_name,
            )
            embedding_score_names["component_l1w4line"] = line_fusion_name
    if global_matcher_model is not None:
        if embedding_model is None:
            raise ValueError("global matcher requires an embedding model")
        embedding_tiles = denoised if embedding_view == "denoised" else raw_tiles
        global_name = f"{embedding_view}_g0_global_matcher"
        bank[global_name] = global_matcher_compatibility(
            global_matcher_model,
            embedding_model,
            embedding_tiles,
            device=device,
            name=global_name,
        )
        embedding_score_names["component_g0"] = global_name
    if pair_model is not None:
        candidates = candidate_union(
            bank,
            names=["denoised_pbc"],
            per_score_top_k=32,
            cap=32,
        )
        l0 = pair_rerank_compatibility(
            pair_model,
            denoised,
            candidates,
            device=device,
            name="denoised_l0_pair_reranker",
        )
        bank[l0.name] = l0
    rank_score_names: dict[str, str] = {}
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
            name="denoised_x0_rank_reranker",
        )
        bank[rank_score.name] = rank_score
        rank_score_names["component_x0"] = rank_score.name
        if l1_name is not None:
            combined_name = "denoised_L1_X0_equal_rank_fusion"
            bank[combined_name] = fuse_ranked_scores(
                bank, names=[l1_name, rank_score.name], name=combined_name
            )
            rank_score_names["component_l1x0"] = combined_name
            full_name = "denoised_C1_L1w4_X0w2_rank_fusion"
            bank[full_name] = fuse_ranked_scores(
                bank,
                names=[c1_scores["denoised"].name, l1_name, rank_score.name],
                weights={l1_name: 4.0, rank_score.name: 2.0},
                name=full_name,
            )
            rank_score_names["component_l1x0full"] = full_name
    score_seconds = time.perf_counter() - score_started

    context_placement = None
    context_seconds = 0.0
    if context_model is not None:
        context_started = time.perf_counter()
        row_logits, column_logits = context_position_logits(
            context_model, denoised, device=device
        )
        context_placement = position_logits_placement_unary(row_logits, column_logits)
        context_seconds = time.perf_counter() - context_started

    layouts = {"identity": np.arange(576, dtype=np.int32)}
    solver_timings = {"identity": 0.0}
    if context_placement is not None:
        context_started = time.perf_counter()
        positions, slots = linear_sum_assignment(context_placement)
        context_layout = np.empty(576, dtype=np.int32)
        context_layout[positions] = slots
        layouts["context_hungarian"] = context_layout
        solver_timings["context_hungarian"] = (
            context_seconds + time.perf_counter() - context_started
        )
    for score_view in ("raw", "denoised"):
        for suffix, score_name in (
            ("component_pbc", f"{score_view}_pbc"),
            ("component_fusion", c1_scores[score_view].name),
        ):
            label = f"{score_view}_{suffix}"
            solver_started = time.perf_counter()
            layouts[label] = reciprocal_component_solver(
                bank[score_name], include_verified_loops=True, refine=False
            ).position_to_slot
            solver_timings[label] = time.perf_counter() - solver_started
        label = f"{score_view}_lp_fusion_q50"
        solver_started = time.perf_counter()
        layouts[label] = weighted_l1_component_solver(
            c1_scores[score_view], proposal_keep_fraction=0.5
        ).position_to_slot
        solver_timings[label] = time.perf_counter() - solver_started
        for edge_label, score_name in edge_scores[score_view].items():
            label = f"{score_view}_component_{edge_label}"
            solver_started = time.perf_counter()
            layouts[label] = reciprocal_component_solver(
                bank[score_name], include_verified_loops=True, refine=False
            ).position_to_slot
            solver_timings[label] = time.perf_counter() - solver_started

    for cross_label, cross_score in cross_c1_scores.items():
        label = f"component_{cross_label}"
        solver_started = time.perf_counter()
        layouts[label] = reciprocal_component_solver(
            cross_score, include_verified_loops=True, refine=False
        ).position_to_slot
        solver_timings[label] = time.perf_counter() - solver_started

    soft_cycle_score_names = {
        "raw_c1": c1_scores["raw"].name,
        "denoised_c1": c1_scores["denoised"].name,
        "cross_c1": cross_c1_scores["cross_c1"].name,
        "cross_c1_dn2": cross_c1_scores["cross_c1_dn2"].name,
    }
    if l1_name is not None:
        soft_cycle_score_names["l1"] = l1_name
    if line_score_name is not None:
        soft_cycle_score_names["line"] = line_score_name
    if "component_l1w4line" in embedding_score_names:
        soft_cycle_score_names["l1w4line"] = embedding_score_names[
            "component_l1w4line"
        ]
    for score_alias in soft_cycle_scores:
        if score_alias not in soft_cycle_score_names:
            continue
        score_name = soft_cycle_score_names[score_alias]
        for top_k in soft_cycle_topks:
            label = f"softcycle_{score_alias}_k{top_k}"
            solver_started = time.perf_counter()
            layouts[label] = soft_cycle_component_solver(
                bank[score_name],
                top_k=top_k,
                keep_per_tile=soft_cycle_keep_per_tile,
                proposal_keep_fraction=soft_cycle_keep_fraction,
                loop_weight=soft_cycle_loop_weight,
                reciprocal_weight=soft_cycle_reciprocal_weight,
            ).position_to_slot
            solver_timings[label] = time.perf_counter() - solver_started

    if embedding_model is not None:
        assert l1_name is not None and compact_name is not None
        for label, score_name in embedding_score_names.items():
            for suffix, keep_fraction in (("", 1.0), ("_q50", 0.5)):
                variant_label = f"{label}{suffix}"
                solver_started = time.perf_counter()
                layouts[variant_label] = reciprocal_component_solver(
                    bank[score_name],
                    include_verified_loops=True,
                    proposal_keep_fraction=keep_fraction,
                    refine=False,
                ).position_to_slot
                solver_timings[variant_label] = time.perf_counter() - solver_started
        solver_started = time.perf_counter()
        layouts["component_l1_placebeam"] = reciprocal_component_solver(
            bank[l1_name],
            include_verified_loops=True,
            refine=False,
            placement_beam_width=4,
            placement_beam_components=8,
        ).position_to_slot
        solver_timings["component_l1_placebeam"] = (
            time.perf_counter() - solver_started
        )
        solver_started = time.perf_counter()
        layouts["component_l1_mutual2"] = mutual_topk_component_solver(
            bank[l1_name], top_k=2
        ).position_to_slot
        solver_timings["component_l1_mutual2"] = time.perf_counter() - solver_started
        assert outside_logits is not None
        placement = outside_logits_placement_unary(outside_logits)
        solver_started = time.perf_counter()
        layouts["component_l1compact_outside"] = reciprocal_component_solver(
            bank[compact_name],
            include_verified_loops=True,
            refine=False,
            placement_costs=placement,
            boundary_weight=0.2,
        ).position_to_slot
        solver_timings["component_l1compact_outside"] = (
            time.perf_counter() - solver_started
        )
        if context_placement is not None:
            for boundary_weight in (0.05, 0.2, 1.0):
                label = f"component_l1_context_w{boundary_weight:g}"
                solver_started = time.perf_counter()
                layouts[label] = reciprocal_component_solver(
                    bank[l1_name],
                    include_verified_loops=True,
                    refine=False,
                    boundary_weight=boundary_weight,
                    placement_costs=context_placement,
                ).position_to_slot
                solver_timings[label] = time.perf_counter() - solver_started
    for label, score_name in rank_score_names.items():
        solver_started = time.perf_counter()
        layouts[label] = reciprocal_component_solver(
            bank[score_name], include_verified_loops=True, refine=False
        ).position_to_slot
        solver_timings[label] = time.perf_counter() - solver_started
        if context_placement is not None and label in {
            "component_l1x0",
            "component_l1x0full",
        }:
            for boundary_weight in (0.05, 0.2, 1.0):
                context_label = f"{label}_context_w{boundary_weight:g}"
                solver_started = time.perf_counter()
                layouts[context_label] = reciprocal_component_solver(
                    bank[score_name],
                    include_verified_loops=True,
                    refine=False,
                    boundary_weight=boundary_weight,
                    placement_costs=context_placement,
                ).position_to_slot
                solver_timings[context_label] = time.perf_counter() - solver_started
    if pair_model is not None:
        for label, solve in (
            (
                "component_l0",
                lambda: reciprocal_component_solver(
                    bank["denoised_l0_pair_reranker"],
                    include_verified_loops=True,
                    refine=False,
                ).position_to_slot,
            ),
            (
                "lp_l0_all",
                lambda: weighted_l1_component_solver(
                    bank["denoised_l0_pair_reranker"],
                    proposal_keep_fraction=1.0,
                ).position_to_slot,
            ),
        ):
            solver_started = time.perf_counter()
            layouts[label] = solve()
            solver_timings[label] = time.perf_counter() - solver_started
    if spatial_placement is not None:
        spatial_scores = {
            "denoised_c1": c1_scores["denoised"].name,
        }
        if l1_name is not None:
            spatial_scores["l1"] = l1_name
        for score_alias, score_name in spatial_scores.items():
            for spatial_weight in spatial_prior_weights:
                label = f"component_{score_alias}_spatial_w{spatial_weight:g}"
                solver_started = time.perf_counter()
                layouts[label] = reciprocal_component_solver(
                    bank[score_name],
                    include_verified_loops=True,
                    refine=False,
                    placement_costs=spatial_placement,
                    boundary_weight=spatial_weight,
                ).position_to_slot
                solver_timings[label] = time.perf_counter() - solver_started
    anneal_score_names = {
        "denoised_c1": c1_scores["denoised"].name,
        "cross_c1": cross_c1_scores["cross_c1"].name,
        "cross_c1_dn2": cross_c1_scores["cross_c1_dn2"].name,
    }
    if l1_name is not None:
        anneal_score_names["l1"] = l1_name
    if "component_l1w4" in embedding_score_names:
        anneal_score_names["l1w4"] = embedding_score_names["component_l1w4"]
    if "component_cross_l1w4" in embedding_score_names:
        anneal_score_names["cross_l1w4"] = embedding_score_names[
            "component_cross_l1w4"
        ]
    if line_score_name is not None:
        anneal_score_names["line"] = line_score_name
    if "component_l1w4line" in embedding_score_names:
        anneal_score_names["l1w4line"] = embedding_score_names[
            "component_l1w4line"
        ]
    if anneal_refine_evaluations > 0:
        if anneal_score_alias not in anneal_score_names:
            raise ValueError(
                f"anneal score {anneal_score_alias!r} requires a compatible checkpoint"
            )
        anneal_score = bank[anneal_score_names[anneal_score_alias]]
        base_seed = int.from_bytes(
            hashlib.sha256(input_path.name.encode("utf-8")).digest()[:4], "little"
        )
        for seed_index, seed_label in enumerate(anneal_refine_seeds):
            if seed_label not in layouts:
                raise ValueError(
                    f"unknown anneal seed {seed_label!r}; available: {sorted(layouts)}"
                )
            label = f"{seed_label}_anneal_long"
            solver_started = time.perf_counter()
            layouts[label] = simulated_anneal_swaps(
                layouts[seed_label],
                anneal_score,
                seed=base_seed + 1009 * (seed_index + 1),
                evaluations=anneal_refine_evaluations,
            )
            solver_timings[label] = (
                solver_timings[seed_label] + time.perf_counter() - solver_started
            )
            if anneal_refine_mixed:
                mixed_label = f"{seed_label}_anneal_mixed"
                solver_started = time.perf_counter()
                layouts[mixed_label] = simulated_anneal_mixed(
                    layouts[seed_label],
                    anneal_score,
                    seed=base_seed + 2003 * (seed_index + 1),
                    evaluations=anneal_refine_evaluations,
                )
                solver_timings[mixed_label] = (
                    solver_timings[seed_label]
                    + time.perf_counter()
                    - solver_started
                )
    if lns_iterations > 0:
        if lns_score_alias not in anneal_score_names:
            raise ValueError(
                f"LNS score {lns_score_alias!r} requires a compatible checkpoint"
            )
        lns_score = bank[anneal_score_names[lns_score_alias]]
        base_seed = int.from_bytes(
            hashlib.sha256(input_path.name.encode("utf-8")).digest()[:4], "little"
        )
        for seed_index, seed_label in enumerate(lns_seeds):
            if seed_label not in layouts:
                raise ValueError(
                    f"unknown LNS seed {seed_label!r}; available: {sorted(layouts)}"
                )
            label = f"{seed_label}_lns"
            solver_started = time.perf_counter()
            layouts[label] = large_neighborhood_reassign(
                layouts[seed_label],
                lns_score,
                seed=base_seed + 5003 * (seed_index + 1),
                iterations=lns_iterations,
                subset_size=lns_subset_size,
            )
            solver_timings[label] = (
                solver_timings[seed_label] + time.perf_counter() - solver_started
            )
    if multi_phase_rl_phases > 0:
        if multi_phase_rl_score_alias not in anneal_score_names:
            raise ValueError(
                f"multi-phase RL score {multi_phase_rl_score_alias!r} "
                "requires a compatible checkpoint"
            )
        rl_score = bank[anneal_score_names[multi_phase_rl_score_alias]]
        for seed_label in multi_phase_rl_seeds:
            if seed_label not in layouts:
                raise ValueError(
                    f"unknown multi-phase RL seed {seed_label!r}; "
                    f"available: {sorted(layouts)}"
                )
            label = f"{seed_label}_multi_phase_rl"
            solver_started = time.perf_counter()
            layouts[label] = multi_phase_relaxation_solver(
                rl_score,
                initial=layouts[seed_label],
                top_k=multi_phase_rl_topk,
                phases=multi_phase_rl_phases,
                iterations_per_phase=multi_phase_rl_iterations,
                anchor_batch=multi_phase_rl_anchor_batch,
            )
            solver_timings[label] = (
                solver_timings[seed_label] + time.perf_counter() - solver_started
            )
    if faithful_rl_phases > 0:
        if faithful_rl_score_alias not in anneal_score_names:
            raise ValueError(
                f"faithful multi-phase RL score {faithful_rl_score_alias!r} "
                "requires a compatible checkpoint"
            )
        solver_started = time.perf_counter()
        layouts["faithful_multi_phase_rl"] = faithful_multi_phase_relaxation_solver(
            bank[anneal_score_names[faithful_rl_score_alias]],
            top_k=faithful_rl_topk,
            phases=faithful_rl_phases,
            convergence_threshold=faithful_rl_convergence,
            max_iterations=faithful_rl_max_iterations,
            anchor_probability=faithful_rl_anchor_probability,
        )
        solver_timings["faithful_multi_phase_rl"] = (
            time.perf_counter() - solver_started
        )
    if particle_beam_particles > 0:
        if particle_beam_score_alias not in anneal_score_names:
            raise ValueError(
                f"particle-beam score {particle_beam_score_alias!r} "
                "requires a compatible checkpoint"
            )
        missing = [name for name in particle_beam_seeds if name not in layouts]
        if missing:
            raise ValueError(
                f"unknown particle-beam seeds {missing}; available: {sorted(layouts)}"
            )
        solver_started = time.perf_counter()
        particle_result = particle_beam_solver(
            bank[anneal_score_names[particle_beam_score_alias]],
            seed_layouts=[layouts[name] for name in particle_beam_seeds],
            particles=particle_beam_particles,
            top_k=particle_beam_topk,
            anchor_hypotheses=particle_beam_anchor_hypotheses,
            frontier_limit=particle_beam_frontier_limit,
        )
        particle_label = f"particle_beam_p{particle_beam_particles}_k{particle_beam_topk}"
        layouts[particle_label] = particle_result.position_to_slot
        solver_timings[particle_label] = time.perf_counter() - solver_started
    if qap_iterations > 0:
        if qap_score_alias not in anneal_score_names:
            raise ValueError(
                f"QAP score {qap_score_alias!r} requires a compatible checkpoint"
            )
        missing = [name for name in qap_seeds if name not in layouts]
        if missing:
            raise ValueError(f"unknown QAP seeds {missing}; available: {sorted(layouts)}")
        initializations: list[tuple[str, np.ndarray | None]] = (
            [(name, layouts[name]) for name in qap_seeds]
            if qap_seeds
            else [("barycenter", None)]
        )
        base_seed = int.from_bytes(
            hashlib.sha256(input_path.name.encode("utf-8")).digest()[:4], "little"
        )
        for qap_index, (seed_name, initial_layout) in enumerate(initializations):
            solver_started = time.perf_counter()
            qap_result = directional_qap(
                bank[anneal_score_names[qap_score_alias]],
                initial=initial_layout,
                iterations=qap_iterations,
                restarts=qap_restarts,
                seed=base_seed + 7001 * (qap_index + 1),
                boundary_weight=qap_boundary_weight,
                initial_weight=qap_initial_weight,
                noisy_components=qap_noisy_components,
                noise_scale=qap_noise_scale,
                refine_swaps=qap_refine_swaps,
            )
            qap_label = f"qap_{seed_name}"
            layouts[qap_label] = qap_result.position_to_slot
            solver_timings[qap_label] = time.perf_counter() - solver_started
    if cpsat_time_seconds > 0:
        if cpsat_score_alias not in anneal_score_names:
            raise ValueError(
                f"CP-SAT score {cpsat_score_alias!r} requires a compatible checkpoint"
            )
        missing = [name for name in cpsat_seeds if name not in layouts]
        if missing:
            raise ValueError(
                f"unknown CP-SAT seeds {missing}; available: {sorted(layouts)}"
            )
        base_seed = int.from_bytes(
            hashlib.sha256(input_path.name.encode("utf-8")).digest()[:4], "little"
        )
        for cpsat_index, seed_name in enumerate(cpsat_seeds):
            solver_started = time.perf_counter()
            cpsat_result = topk_cpsat_grid_solver(
                bank[anneal_score_names[cpsat_score_alias]],
                top_k=cpsat_topk,
                max_time_seconds=cpsat_time_seconds,
                workers=cpsat_workers,
                seed=(base_seed + 9001 * (cpsat_index + 1)) % 2_147_483_648,
                max_square_terms=cpsat_square_terms,
                initial_position_to_slot=layouts[seed_name],
            )
            cpsat_label = f"cpsat_{seed_name}"
            layouts[cpsat_label] = cpsat_result.position_to_slot
            solver_timings[cpsat_label] = time.perf_counter() - solver_started
    selector_score_names = [
        "denoised_pbc",
        "denoised_C1_equal_rank_fusion",
        l1_name,
        compact_name,
        *embedding_score_names.values(),
        *rank_score_names.values(),
    ]
    selector_scores = {
        label: {
            score_name: _layout_pair_cost(layout, bank[score_name])
            for score_name in selector_score_names
            if score_name is not None and score_name in bank
        }
        for label, layout in layouts.items()
    }
    return {"raw": raw_tiles, "denoised": denoised}, layouts, {
        "score_seconds": score_seconds,
        "solver_seconds": solver_timings,
        "total_prediction_seconds": time.perf_counter() - started,
    }, selector_scores


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit <= 0:
        raise SystemExit("offset must be non-negative and limit positive")
    try:
        soft_cycle_topks = [
            int(value) for value in args.soft_cycle_topk.split(",") if value.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--soft-cycle-topk must contain integers") from exc
    if any(value < 2 or value >= 576 for value in soft_cycle_topks):
        raise SystemExit("--soft-cycle-topk values must be in [2, 575]")
    soft_cycle_scores = [
        value.strip() for value in args.soft_cycle_scores.split(",") if value.strip()
    ]
    anneal_refine_seeds = [
        value.strip() for value in args.anneal_refine_seeds.split(",") if value.strip()
    ]
    lns_seeds = [value.strip() for value in args.lns_seeds.split(",") if value.strip()]
    multi_phase_rl_seeds = [
        value.strip()
        for value in args.multi_phase_rl_seeds.split(",")
        if value.strip()
    ]
    particle_beam_seeds = [
        value.strip()
        for value in args.particle_beam_seeds.split(",")
        if value.strip()
    ]
    qap_seeds = [
        value.strip() for value in args.qap_seeds.split(",") if value.strip()
    ]
    cpsat_seeds = [
        value.strip() for value in args.cpsat_seeds.split(",") if value.strip()
    ]
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
    unknown_soft_cycle_scores = set(soft_cycle_scores) - {
        "raw_c1",
        "denoised_c1",
        "cross_c1",
        "cross_c1_dn2",
        "l1",
        "line",
        "l1w4line",
    }
    if unknown_soft_cycle_scores:
        raise SystemExit(
            f"unknown --soft-cycle-scores aliases: {sorted(unknown_soft_cycle_scores)}"
        )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    names = source_names_for_split(
        args.split, manifest_path=args.manifest, quarantine_path=args.quarantine
    )[args.offset : args.offset + args.limit]
    if len(names) != args.limit:
        raise SystemExit("requested source slice extends past split")
    restorer, device, denoiser_metadata = load_restorer(args.denoiser, device=args.device)
    embedding_model = embedding_metadata = None
    global_matcher_model = global_matcher_metadata = None
    pair_model = pair_metadata = None
    rank_model = rank_feature_names = rank_metadata = None
    context_model = context_metadata = None
    spatial_model = joblib.load(args.spatial_prior) if args.spatial_prior else None
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
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    for index, name in enumerate(names):
        input_path = Path(args.data_root) / "train" / "inputs" / name
        render_tiles, layouts, timings, selector_scores = _predict_input_only(
            input_path,
            restorer=restorer,
            embedding_model=embedding_model,
            global_matcher_model=global_matcher_model,
            pair_model=pair_model,
            rank_model=rank_model,
            rank_feature_names=rank_feature_names,
            rank_metadata=rank_metadata,
            context_model=context_model,
            embedding_view=args.embedding_view,
            device=device,
            batch_size=args.batch_size,
            line_seam_enabled=args.line_seam,
            line_seam_auxiliary_weight=args.line_seam_auxiliary_weight,
            line_seam_fusion_weight=args.line_seam_fusion_weight,
            soft_cycle_topks=soft_cycle_topks,
            soft_cycle_scores=soft_cycle_scores,
            soft_cycle_keep_per_tile=args.soft_cycle_keep_per_tile,
            soft_cycle_keep_fraction=args.soft_cycle_keep_fraction,
            soft_cycle_loop_weight=args.soft_cycle_loop_weight,
            soft_cycle_reciprocal_weight=args.soft_cycle_reciprocal_weight,
            anneal_refine_evaluations=args.anneal_refine_evaluations,
            anneal_refine_seeds=anneal_refine_seeds,
            anneal_score_alias=args.anneal_score,
            anneal_refine_mixed=args.anneal_refine_mixed,
            spatial_model=spatial_model,
            spatial_prior_weights=spatial_prior_weights,
            lns_iterations=args.lns_iterations,
            lns_subset_size=args.lns_subset_size,
            lns_seeds=lns_seeds,
            lns_score_alias=args.lns_score,
            multi_phase_rl_phases=args.multi_phase_rl_phases,
            multi_phase_rl_topk=args.multi_phase_rl_topk,
            multi_phase_rl_iterations=args.multi_phase_rl_iterations,
            multi_phase_rl_anchor_batch=args.multi_phase_rl_anchor_batch,
            multi_phase_rl_seeds=multi_phase_rl_seeds,
            multi_phase_rl_score_alias=args.multi_phase_rl_score,
            faithful_rl_phases=args.faithful_rl_phases,
            faithful_rl_topk=args.faithful_rl_topk,
            faithful_rl_max_iterations=args.faithful_rl_max_iterations,
            faithful_rl_convergence=args.faithful_rl_convergence,
            faithful_rl_anchor_probability=args.faithful_rl_anchor_probability,
            faithful_rl_score_alias=args.faithful_rl_score,
            particle_beam_particles=args.particle_beam_particles,
            particle_beam_topk=args.particle_beam_topk,
            particle_beam_anchor_hypotheses=args.particle_beam_anchor_hypotheses,
            particle_beam_frontier_limit=args.particle_beam_frontier_limit,
            particle_beam_seeds=particle_beam_seeds,
            particle_beam_score_alias=args.particle_beam_score,
            qap_iterations=args.qap_iterations,
            qap_restarts=args.qap_restarts,
            qap_boundary_weight=args.qap_boundary_weight,
            qap_refine_swaps=args.qap_refine_swaps,
            qap_initial_weight=args.qap_initial_weight,
            qap_noisy_components=args.qap_noisy_components,
            qap_noise_scale=args.qap_noise_scale,
            qap_seeds=qap_seeds,
            qap_score_alias=args.qap_score,
            cpsat_time_seconds=args.cpsat_time_seconds,
            cpsat_topk=args.cpsat_topk,
            cpsat_workers=args.cpsat_workers,
            cpsat_square_terms=args.cpsat_square_terms,
            cpsat_seeds=cpsat_seeds,
            cpsat_score_alias=args.cpsat_score,
        )
        # Target access begins only after all input-only layouts are frozen.
        target = _read_rgb(Path(args.data_root) / "train" / "targets" / name)
        variants = {}
        for label, layout in layouts.items():
            for render_view, tiles in render_tiles.items():
                variant = f"{label}__{render_view}_render"
                variants[variant] = {
                    **predicted_image_metrics(layout, tiles, target),
                    "position_to_slot": layout.tolist(),
                }
        if preview_dir is not None and index == 0:
            for label, layout in layouts.items():
                for render_view, tiles in render_tiles.items():
                    Image.fromarray(merge_tiles_numpy(tiles[layout]), mode="RGB").save(
                        preview_dir / f"{Path(name).stem}_{label}_{render_view}_render.png"
                    )
        sources.append(
            {
                "source": name,
                "variants": variants,
                "selector_scores": selector_scores,
                "timings": timings,
            }
        )
        print(
            json.dumps(
                {
                    "event": "real_assembly_source",
                    "index": index + 1,
                    "count": len(names),
                    "source": name,
                    "best_ssim": max(value["predicted_layout_ssim"] for value in variants.values()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    variant_names = sorted(sources[0]["variants"])
    macro = {
        variant: {
            metric: float(np.mean([source["variants"][variant][metric] for source in sources]))
            for metric in ("predicted_layout_ssim", "psnr", "mae")
        }
        for variant in variant_names
    }
    report = {
        "schema_version": 1,
        "kind": "real_input_only_assembly_target_only_score",
        "anti_leakage": {
            "predictor_accepts_target": False,
            "target_opened_after_layouts_frozen": True,
            "pseudo_mapping_used": False,
        },
        "factorial": {
            "compatibility_views": ["raw", "denoised"],
            "render_views": ["raw", "denoised"],
            "purpose": "separate permutation quality from restoration quality",
        },
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "source_names": names,
        "soft_cycle": {
            "top_k": soft_cycle_topks,
            "scores": soft_cycle_scores,
            "keep_per_tile": args.soft_cycle_keep_per_tile,
            "keep_fraction": args.soft_cycle_keep_fraction,
            "loop_weight": args.soft_cycle_loop_weight,
            "reciprocal_weight": args.soft_cycle_reciprocal_weight,
        },
        "anneal": {
            "evaluations": args.anneal_refine_evaluations,
            "seeds": anneal_refine_seeds,
            "score": args.anneal_score,
            "mixed": args.anneal_refine_mixed,
        },
        "spatial_prior": {
            "path": args.spatial_prior,
            "sha256": _sha256(Path(args.spatial_prior)) if args.spatial_prior else None,
            "weights": spatial_prior_weights,
        },
        "line_seam": {
            "enabled": args.line_seam,
            "auxiliary_weight": args.line_seam_auxiliary_weight,
            "fusion_weight": args.line_seam_fusion_weight,
        },
        "lns": {
            "iterations": args.lns_iterations,
            "subset_size": args.lns_subset_size,
            "seeds": lns_seeds,
            "score": args.lns_score,
        },
        "multi_phase_rl": {
            "phases": args.multi_phase_rl_phases,
            "top_k": args.multi_phase_rl_topk,
            "iterations_per_phase": args.multi_phase_rl_iterations,
            "anchor_batch": args.multi_phase_rl_anchor_batch,
            "seeds": multi_phase_rl_seeds,
            "score": args.multi_phase_rl_score,
        },
        "faithful_multi_phase_rl": {
            "phases": args.faithful_rl_phases,
            "top_k": args.faithful_rl_topk,
            "max_iterations": args.faithful_rl_max_iterations,
            "convergence_threshold": args.faithful_rl_convergence,
            "anchor_probability": args.faithful_rl_anchor_probability,
            "score": args.faithful_rl_score,
        },
        "particle_beam": {
            "particles": args.particle_beam_particles,
            "top_k": args.particle_beam_topk,
            "anchor_hypotheses": args.particle_beam_anchor_hypotheses,
            "frontier_limit": args.particle_beam_frontier_limit,
            "seeds": particle_beam_seeds,
            "score": args.particle_beam_score,
        },
        "qap": {
            "iterations": args.qap_iterations,
            "restarts": args.qap_restarts,
            "boundary_weight": args.qap_boundary_weight,
            "refine_swaps": args.qap_refine_swaps,
            "initial_weight": args.qap_initial_weight,
            "noisy_components": args.qap_noisy_components,
            "noise_scale": args.qap_noise_scale,
            "seeds": qap_seeds,
            "score": args.qap_score,
        },
        "cpsat": {
            "time_seconds": args.cpsat_time_seconds,
            "top_k": args.cpsat_topk,
            "workers": args.cpsat_workers,
            "square_terms": args.cpsat_square_terms,
            "seeds": cpsat_seeds,
            "score": args.cpsat_score,
        },
        "denoiser": denoiser_metadata,
        "embedding_checkpoint": args.embedding_checkpoint,
        "embedding_view": args.embedding_view,
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
        "rank_feature_names": rank_feature_names,
        "rank_metadata": rank_metadata,
        "context_checkpoint": args.context_checkpoint,
        "context_checkpoint_sha256": (
            _sha256(Path(args.context_checkpoint)) if args.context_checkpoint else None
        ),
        "context_metadata": context_metadata,
        "sources": sources,
        "macro": macro,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "real_assembly_complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
