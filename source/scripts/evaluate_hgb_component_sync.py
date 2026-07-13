#!/usr/bin/env python3
"""Calibrate HGB-seeded translation consensus, then run one frozen v4 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import CompatibilityMatrices, fuse_ranked_scores
from puzzle_assembly.components import (
    ProposedEdge,
    _complete_with_hungarian,
    _place_components_beam,
    grow_component_translation_consensus,
    grow_components_with_edges,
    soft_cycle_component_solver,
)
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import PreparedSource, prepare_source, read_rgb
from train_full_union_tabular_verifier import load_v4
from evaluate_full_union_hgb_qap import hgb_compatibility


PANELS = ("primary_kornia", "independent_libjpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/assembly_v1/full_union_tabular/v1/full_union_tabular.joblib",
    )
    parser.add_argument(
        "--tabular-report",
        default="runs/assembly_v1/full_union_tabular/v1/report.json",
    )
    parser.add_argument("--hgb-weight", type=float, default=0.25)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--denoise-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--calibration-iterations", type=int, default=10)
    parser.add_argument("--final-iterations", type=int, default=25)
    parser.add_argument("--max-calibration-sources", type=int)
    parser.add_argument("--skip-v4", action="store_true")
    parser.add_argument(
        "--fixture-root",
        default="runs/assembly_v1/candidate_graph_oracle_fixtures_v4_6c0fe4e8524ce39d830d9a5bee118d8b",
    )
    parser.add_argument(
        "--graph-root",
        default="runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_readback/candidate_graph_oracle_v4_phase_a/finalized",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_residual(
    model: object, prepared: PreparedSource, *, hgb_weight: float
) -> tuple[np.ndarray, CompatibilityMatrices]:
    probability = model.predict_proba(prepared.features)[:, 1]
    hgb = hgb_compatibility(
        model, prepared.graph, prepared.features
    )
    bank = {
        "c1": prepared.scores["c1"],
        "hbt": prepared.scores["hbt"],
        "hgb": hgb,
    }
    residual = fuse_ranked_scores(
        bank,
        names=["c1", "hbt", "hgb"],
        weights={"hbt": 4.0, "hgb": hgb_weight},
        name="C1_HBTw4_HGB_residual",
    )
    return probability, residual


def seed_components(
    prepared: PreparedSource, probability: np.ndarray, threshold: float
) -> tuple[list[dict[int, tuple[int, int]]], list[ProposedEdge]]:
    selected = np.flatnonzero(probability >= threshold)
    selected = selected[np.argsort(-probability[selected], kind="stable")]
    proposals = [
        ProposedEdge(
            first=int(prepared.graph.source[index]),
            second=int(prepared.graph.destination[index]),
            dx=1 if int(prepared.graph.direction[index]) == 0 else 0,
            dy=0 if int(prepared.graph.direction[index]) == 0 else 1,
            cost=float(1.0 - probability[index]),
            margin=float(probability[index]),
            reciprocal=False,
            in_loop=int(prepared.graph.origin_mask[index]).bit_count() >= 2,
        )
        for index in selected.tolist()
    ]
    return grow_components_with_edges(proposals)


def solve(
    prepared: PreparedSource,
    residual: CompatibilityMatrices,
    *,
    threshold: float,
    probability: np.ndarray,
    top_k: int,
    min_support: int,
    qap_seed: int,
    iterations: int,
    initial_override: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    components, accepted = seed_components(prepared, probability, threshold)
    merges = 0
    if top_k > 0:
        components, merges = grow_component_translation_consensus(
            components,
            residual,
            top_k=top_k,
            min_support=min_support,
            max_merges=576,
            reciprocal_weight=0.35,
        )
    if initial_override is None:
        grid, placed = _place_components_beam(
            components,
            residual,
            boundary_weight=0.05,
            beam_width=8,
            beam_components=8,
            translations_per_state=8,
        )
        initial, unresolved = _complete_with_hungarian(
            grid.copy(), residual, boundary_weight=0.05
        )
    else:
        initial = initial_override
        placed = 0
        unresolved = 0
    qap = directional_qap(
        residual,
        initial=initial,
        iterations=iterations,
        restarts=1 if iterations <= 10 else 2,
        seed=qap_seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    )
    return qap.position_to_slot, {
        "seed_edges": len(accepted),
        "translation_merges": int(merges),
        "component_sizes": sorted(
            (len(component) for component in components), reverse=True
        ),
        "placed_tiles": int(placed),
        "unresolved_before_hungarian": int(unresolved),
    }


def baseline_layout(
    prepared: PreparedSource, *, qap_seed: int, iterations: int = 25
) -> np.ndarray:
    soft = soft_cycle_component_solver(
        prepared.scores["w4"],
        top_k=8,
        keep_per_tile=1,
        reciprocal_weight=0.35,
        loop_weight=1.0,
        proposal_keep_fraction=0.5,
        boundary_weight=0.05,
    )
    return directional_qap(
        prepared.scores["w4"],
        initial=soft.position_to_slot,
        iterations=iterations,
        restarts=2,
        seed=qap_seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    ).position_to_slot


def score_layout(
    layout: np.ndarray,
    prepared: PreparedSource,
    clean: np.ndarray,
) -> dict[str, float]:
    geometry = layout_metrics(layout, prepared.truth)
    image = predicted_image_metrics(layout, prepared.denoised_tiles, clean)
    return {
        "adjacency": float(geometry["combined_adjacency"]),
        "ssim": float(image["predicted_layout_ssim"]),
    }


def summarize(records: list[dict], candidate: str) -> dict:
    panels = {}
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        delta_ssim = np.asarray(
            [
                record["candidates"][candidate]["ssim"] - record["baseline"]["ssim"]
                for record in selected
            ]
        )
        delta_adjacency = np.asarray(
            [
                record["candidates"][candidate]["adjacency"]
                - record["baseline"]["adjacency"]
                for record in selected
            ]
        )
        panels[panel] = {
            "mean_ssim_delta": float(delta_ssim.mean()),
            "mean_adjacency_delta": float(delta_adjacency.mean()),
            "ssim_wins": int(np.count_nonzero(delta_ssim > 0)),
        }
    source_names = sorted({record["name"] for record in records})
    source_delta = []
    for name in source_names:
        values = [
            record["candidates"][candidate]["ssim"] - record["baseline"]["ssim"]
            for record in records
            if record["name"] == name
        ]
        source_delta.append(float(np.mean(values)))
    return {
        "panels": panels,
        "source_macro_mean_ssim_delta": float(np.mean(source_delta)),
        "worst_panel_mean_ssim_delta": min(
            value["mean_ssim_delta"] for value in panels.values()
        ),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_payload = joblib.load(args.model)
    model = model_payload["model"]
    tabular_report = json.loads(Path(args.tabular_report).read_text())
    thresholds = {
        target: float(
            tabular_report["calibration"]["frontiers"][str(target)]["threshold"]
        )
        for target in (0.75, 0.80, 0.85)
    }
    variants = []
    for target, threshold in thresholds.items():
        variants.append((f"p{int(target * 100)}_seed", threshold, 0, 0))
        for top_k in (8, 16):
            for support in (2, 3):
                variants.append(
                    (
                        f"p{int(target * 100)}_k{top_k}_s{support}",
                        threshold,
                        top_k,
                        support,
                    )
                )
    restorer, device, _ = load_restorer(args.denoiser, device=args.device)
    embedding, _ = load_embedding_checkpoint(args.embedding_checkpoint, device=device)
    source_names = source_names_for_split(
        "edge_development",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[24:32]
    if args.max_calibration_sources is not None:
        source_names = source_names[: args.max_calibration_sources]
    calibration_records = []
    started = time.time()
    for source_index, name in enumerate(source_names):
        clean = read_rgb(Path(args.data_root) / "train/targets" / name)
        for panel in PANELS:
            panel_seed = per_source_seed(
                args.seed, f"full-union-tabular-{panel}", name, 0
            )
            prepared = prepare_source(
                name,
                panel,
                panel_seed,
                args=args,
                restorer=restorer,
                embedding_model=embedding,
                device=device,
            )
            probability, residual = make_residual(
                model, prepared, hgb_weight=args.hgb_weight
            )
            qap_seed = per_source_seed(
                args.seed, f"hgb-component-sync-qap-{panel}", name, 0
            )
            base_layout = baseline_layout(prepared, qap_seed=qap_seed)
            record = {
                "name": name,
                "panel": panel,
                "baseline": score_layout(base_layout, prepared, clean),
                "candidates": {},
                "diagnostics": {},
            }
            for variant, threshold, top_k, support in variants:
                layout, diagnostics = solve(
                    prepared,
                    residual,
                    threshold=threshold,
                    probability=probability,
                    top_k=top_k,
                    min_support=support,
                    qap_seed=qap_seed,
                    iterations=args.calibration_iterations,
                )
                record["candidates"][variant] = score_layout(
                    layout, prepared, clean
                )
                record["diagnostics"][variant] = diagnostics
            calibration_records.append(record)
        print(
            json.dumps(
                {
                    "stage": "calibration",
                    "done": source_index + 1,
                    "total": len(source_names),
                }
            ),
            flush=True,
        )
    summaries = {
        variant: summarize(calibration_records, variant)
        for variant, _, _, _ in variants
    }
    eligible = [
        variant
        for variant in summaries
        if summaries[variant]["worst_panel_mean_ssim_delta"] >= 0.0
        and summaries[variant]["source_macro_mean_ssim_delta"] > 0.0
    ]
    selected = (
        max(
            eligible,
            key=lambda variant: (
                summaries[variant]["worst_panel_mean_ssim_delta"],
                summaries[variant]["source_macro_mean_ssim_delta"],
                variant,
            ),
        )
        if eligible
        else None
    )
    v4_records = []
    if selected is not None and not args.skip_v4:
        selected_spec = next(spec for spec in variants if spec[0] == selected)
        _, threshold, top_k, support = selected_spec
        for record_index, (meta, prepared) in enumerate(
            load_v4(Path(args.fixture_root), Path(args.graph_root)), 1
        ):
            probability, residual = make_residual(
                model, prepared, hgb_weight=args.hgb_weight
            )
            # The frozen v4 qap_w4 layout is the exact production baseline.
            graph_path = (
                Path(args.graph_root)
                / "artifacts"
                / f"{meta['opaque_id']}.graph.npz"
            )
            label_path = (
                Path(args.fixture_root)
                / "fixture_label/records"
                / f"{meta['opaque_id']}.npz"
            )
            input_path = (
                Path(args.fixture_root)
                / "fixture_input/records"
                / f"{meta['opaque_id']}.npz"
            )
            with np.load(graph_path, allow_pickle=False) as graph_values, np.load(
                label_path, allow_pickle=False
            ) as label_values, np.load(input_path, allow_pickle=False) as input_values:
                base_layout = np.asarray(graph_values["qap_w4_layout"])
                clean = np.asarray(label_values["clean_target_rgb"])
                qap_seed = int(input_values["qap_seed"])
            layout, diagnostics = solve(
                prepared,
                residual,
                threshold=threshold,
                probability=probability,
                top_k=top_k,
                min_support=support,
                qap_seed=qap_seed,
                iterations=args.final_iterations,
            )
            v4_records.append(
                {
                    "name": prepared.name,
                    "panel": prepared.panel,
                    "baseline": score_layout(base_layout, prepared, clean),
                    "candidate": score_layout(layout, prepared, clean),
                    "diagnostics": diagnostics,
                }
            )
            print(
                json.dumps({"stage": "v4", "done": record_index, "total": 64}),
                flush=True,
            )
    v4_summary = None
    if v4_records:
        delta = np.asarray(
            [r["candidate"]["ssim"] - r["baseline"]["ssim"] for r in v4_records]
        )
        v4_summary = {
            "records": len(v4_records),
            "mean_ssim_delta": float(delta.mean()),
            "ssim_wins": int(np.count_nonzero(delta > 0)),
            "mean_baseline_ssim": float(
                np.mean([r["baseline"]["ssim"] for r in v4_records])
            ),
            "mean_candidate_ssim": float(
                np.mean([r["candidate"]["ssim"] for r in v4_records])
            ),
        }
    report = {
        "schema_version": 1,
        "kind": "hgb_seeded_component_translation_sync",
        "hgb_weight": args.hgb_weight,
        "variants": [spec[0] for spec in variants],
        "calibration_summaries": summaries,
        "selected": selected,
        "selection_rule": "maximize worst-panel mean SSIM delta among variants nonnegative on both panels and positive source-macro",
        "calibration_records": calibration_records,
        "v4_summary": v4_summary,
        "v4_records": v4_records,
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "selected_calibration": summaries.get(selected) if selected else None,
                "v4_summary": v4_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
