#!/usr/bin/env python3
"""Frozen one-axis exact path-cover prerequisite on one corruption panel.

This diagnostic never assembles or renders a two-dimensional candidate.  It
compares an exact cover by 24 directed paths of 24 tiles with the row/column
paths induced by the unchanged production QAP.  Clean target pixels are used
only by the established exact-panel generator; the path-cover API receives
only input-derived compatibility costs and the input-derived QAP reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import GRID, TILE_COUNT, true_neighbour_slots
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.path_cover import (
    extract_union_directed_candidates,
    path_cover_edges,
    solve_path_cover,
    validate_exact_path_cover,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import prepare_source


PANELS = ("primary_kornia", "independent_libjpeg")
FROZEN_SPLIT = "edge_development"
FROZEN_SOURCE_OFFSET = 332
FROZEN_SOURCE_COUNT = 8
FROZEN_SOURCE_NAMES_SHA256 = (
    "93a429dec71ad1abd28df5b981b9142ac89525a0d3d092dc0078a4a0d27f128c"
)
FROZEN_OUTGOING_TOP_K = 16
FROZEN_INCOMING_TOP_K = 16
FROZEN_TIME_LIMIT_SECONDS = 30.0
EXPECTED_ASSET_SHA256 = {
    "denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "embedding": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--denoiser",
        default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt",
    )
    parser.add_argument(
        "--embedding-checkpoint",
        default=(
            "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
            "hbt_d320_denoised_rgb_sobel.pt"
        ),
    )
    parser.add_argument(
        "--manifest", default="configs/denoise_splits_seed20260710.json"
    )
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument(
        "--audit-exclusion", default="configs/assembly_audit_exclusion_v1.json"
    )
    parser.add_argument("--split", default=FROZEN_SPLIT)
    parser.add_argument("--source-offset", type=int, default=FROZEN_SOURCE_OFFSET)
    parser.add_argument("--sources", type=int, default=FROZEN_SOURCE_COUNT)
    parser.add_argument(
        "--outgoing-top-k", type=int, default=FROZEN_OUTGOING_TOP_K
    )
    parser.add_argument(
        "--incoming-top-k", type=int, default=FROZEN_INCOMING_TOP_K
    )
    parser.add_argument(
        "--time-limit-seconds", type=float, default=FROZEN_TIME_LIMIT_SECONDS
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def _assert_finite_payload(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError(f"non-finite report value at {path}")


def _filename_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _reference_paths(layout: np.ndarray, axis: str) -> tuple[tuple[int, ...], ...]:
    grid = np.asarray(layout, dtype=np.int32).reshape(GRID, GRID)
    values = grid if axis == "right" else grid.T
    return validate_exact_path_cover(
        tuple(tuple(int(node) for node in line) for line in values.tolist()),
        node_count=TILE_COUNT,
        path_count=GRID,
        path_length=GRID,
    )


def _axis_accuracy(
    paths: Iterable[Sequence[int]], true_neighbour: np.ndarray
) -> float:
    edges = path_cover_edges(paths)
    correct = sum(int(true_neighbour[source] == destination) for source, destination in edges)
    if len(edges) != TILE_COUNT - GRID:
        raise RuntimeError("axis cover does not contain exactly 552 directed edges")
    return correct / len(edges)


def _path_purity(
    paths: Iterable[Sequence[int]], slot_to_target: np.ndarray, axis: str
) -> float:
    purities = []
    for path in paths:
        target_positions = slot_to_target[np.asarray(path, dtype=np.int64)]
        labels = target_positions // GRID if axis == "right" else target_positions % GRID
        counts = np.bincount(labels, minlength=GRID)
        purities.append(float(counts.max()) / GRID)
    if len(purities) != GRID:
        raise RuntimeError("axis cover does not contain exactly 24 paths")
    return float(np.mean(purities))


def _production_reference(prepared: Any, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    seed_result = soft_cycle_component_solver(
        prepared.scores["hbt"],
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    )
    result = directional_qap(
        prepared.scores["w4"],
        initial=seed_result.position_to_slot,
        iterations=25,
        restarts=2,
        seed=_filename_seed(name) + 7001,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    )
    return result.position_to_slot, {
        "soft_cycle_accepted_edges": int(seed_result.accepted_edges),
        "soft_cycle_component_sizes": list(seed_result.component_sizes),
        "qap_objective": float(result.objective),
        "qap_relaxed_objective": float(result.relaxed_objective),
        "qap_restart": int(result.restart),
        "qap_iterations": int(result.iterations),
        "qap_converged": bool(result.converged),
        "qap_seed": _filename_seed(name) + 7001,
    }


def _panel_summary(records: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    values = [record["axes"][axis] for record in records]
    adjacency = np.asarray([item["adjacency_delta"] for item in values], dtype=np.float64)
    purity = np.asarray([item["path_purity_delta"] for item in values], dtype=np.float64)
    runtimes = np.asarray([item["solver_seconds"] for item in values], dtype=np.float64)
    rescue = np.asarray(
        [item["selected_rescue_only_fraction"] for item in values], dtype=np.float64
    )
    return {
        "records": len(values),
        "mean_adjacency_delta": float(adjacency.mean()),
        "min_adjacency_delta": float(adjacency.min()),
        "adjacency_wins": int(np.count_nonzero(adjacency > 0.0)),
        "adjacency_ties": int(np.count_nonzero(adjacency == 0.0)),
        "adjacency_losses": int(np.count_nonzero(adjacency < 0.0)),
        "mean_path_purity_delta": float(purity.mean()),
        "fallbacks": int(sum(item["used_reference_fallback"] for item in values)),
        "valid_covers": int(sum(item["valid_exact_cover"] for item in values)),
        "mean_solver_seconds": float(runtimes.mean()),
        "max_solver_seconds": float(runtimes.max()),
        "mean_selected_rescue_only_fraction": float(rescue.mean()),
        "max_selected_rescue_only_fraction": float(rescue.max()),
    }


def _panel_gate(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for axis in ("right", "down"):
        cell = summary[axis]
        checks[f"{axis}_mean_adjacency_delta_ge_0.02"] = (
            cell["mean_adjacency_delta"] >= 0.02
        )
        checks[f"{axis}_wins_ge_6_of_8"] = cell["adjacency_wins"] >= 6
        checks[f"{axis}_mean_path_purity_delta_ge_0"] = (
            cell["mean_path_purity_delta"] >= 0.0
        )
        checks[f"{axis}_no_source_regression_below_-0.02"] = (
            cell["min_adjacency_delta"] >= -0.02
        )
        checks[f"{axis}_valid_exact_cover_8_of_8"] = cell["valid_covers"] == 8
        checks[f"{axis}_fallbacks_le_1_of_8"] = cell["fallbacks"] <= 1
        checks[f"{axis}_max_rescue_only_fraction_le_0.10"] = (
            cell["max_selected_rescue_only_fraction"] <= 0.10
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "decision": (
            "panel_axis_prerequisite_pass"
            if all(checks.values())
            else "stop_path_cover_no_axis_signal"
        ),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    if (
        args.split != FROZEN_SPLIT
        or args.source_offset != FROZEN_SOURCE_OFFSET
        or args.sources != FROZEN_SOURCE_COUNT
        or args.outgoing_top_k != FROZEN_OUTGOING_TOP_K
        or args.incoming_top_k != FROZEN_INCOMING_TOP_K
        or args.time_limit_seconds != FROZEN_TIME_LIMIT_SECONDS
    ):
        raise SystemExit("frozen path-cover diagnostic protocol drift")
    output.parent.mkdir(parents=True, exist_ok=True)

    actual_assets = {
        "denoiser": sha256(args.denoiser),
        "embedding": sha256(args.embedding_checkpoint),
    }
    if actual_assets != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"frozen asset hash drift: {actual_assets}")

    names = source_names_for_split(
        args.split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
        audit_exclusion_path=args.audit_exclusion,
    )[args.source_offset : args.source_offset + args.sources]
    if len(names) != FROZEN_SOURCE_COUNT:
        raise RuntimeError("frozen source slice is unavailable")
    actual_names_sha256 = names_sha256(names)
    if actual_names_sha256 != FROZEN_SOURCE_NAMES_SHA256:
        raise RuntimeError("frozen source-name fingerprint drift")
    for protected_split in (
        "assembly_cal",
        "assembly_incremental_gate",
        "assembly_audit_exposed",
        "assembly_final_audit",
    ):
        protected = set(
            source_names_for_split(
                protected_split,
                manifest_path=args.manifest,
                quarantine_path=args.quarantine,
                audit_exclusion_path=args.audit_exclusion,
            )
        )
        if protected.intersection(names):
            raise RuntimeError(f"whole-source overlap with {protected_split}")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device
    )
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    restorer.eval()
    embedding.eval()
    for frozen in (restorer, embedding):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    learned_names = (
        set(embedding_metadata.get("train_names", []))
        | set(embedding_metadata.get("val_names", []))
        | set(embedding_metadata.get("validation_names", []))
    )
    if learned_names.intersection(names):
        raise RuntimeError("whole-source overlap with frozen embedding fit/validation")

    started = time.time()
    records: list[dict[str, Any]] = []
    for source_index, name in enumerate(names):
        panel_seed = per_source_seed(
            args.seed, f"path-cover-axis-{args.panel}", name, 0
        )
        prepared = prepare_source(
            name,
            args.panel,
            panel_seed,
            args=args,
            restorer=restorer,
            embedding_model=embedding,
            device=device,
        )
        reference_layout, reference_diagnostics = _production_reference(prepared, name)
        axis_payload: dict[str, Any] = {}
        # The exact-panel helper has already constructed truth for later metrics,
        # but neither truth nor labels are passed to the two solver calls below.
        frozen_results: dict[str, tuple[Any, tuple[tuple[int, ...], ...], set[tuple[int, int]]]] = {}
        for axis, matrix in (
            ("right", prepared.scores["w4"].right),
            ("down", prepared.scores["w4"].down),
        ):
            reference = _reference_paths(reference_layout, axis)
            rescue_edges = set(path_cover_edges(reference))
            regular = extract_union_directed_candidates(
                matrix,
                outgoing_top_k=args.outgoing_top_k,
                incoming_top_k=args.incoming_top_k,
            )
            regular_edges = {candidate.edge for candidate in regular}
            rescue_only = rescue_edges - regular_edges
            axis_started = time.perf_counter()
            result = solve_path_cover(
                matrix,
                path_count=GRID,
                path_length=GRID,
                outgoing_top_k=args.outgoing_top_k,
                incoming_top_k=args.incoming_top_k,
                rescue_edges=rescue_edges,
                reference_paths=reference,
                time_limit_seconds=args.time_limit_seconds,
                random_seed=(panel_seed + (0 if axis == "right" else 1)) % (2**31),
                require_optimal=True,
                require_strict_reference_improvement=True,
                reference_improvement_feasibility=True,
            )
            elapsed = time.perf_counter() - axis_started
            validate_exact_path_cover(
                result.paths,
                node_count=TILE_COUNT,
                path_count=GRID,
                path_length=GRID,
            )
            frozen_results[axis] = (result, reference, rescue_only)
            axis_payload[axis] = {
                "solver_seconds": elapsed,
                "accepted_candidate": bool(result.accepted_candidate),
                "used_reference_fallback": bool(result.used_reference_fallback),
                "fallback_reason": result.fallback_reason,
                "valid_exact_cover": True,
                "solver": result.diagnostics,
            }

        true_right, true_down = true_neighbour_slots(prepared.truth)
        for axis, true_neighbour in (("right", true_right), ("down", true_down)):
            result, reference, rescue_only = frozen_results[axis]
            selected_edges = set(path_cover_edges(result.paths))
            candidate_accuracy = _axis_accuracy(result.paths, true_neighbour)
            reference_accuracy = _axis_accuracy(reference, true_neighbour)
            candidate_purity = _path_purity(result.paths, prepared.truth, axis)
            reference_purity = _path_purity(reference, prepared.truth, axis)
            axis_payload[axis].update(
                {
                    "candidate_adjacency": candidate_accuracy,
                    "reference_adjacency": reference_accuracy,
                    "adjacency_delta": candidate_accuracy - reference_accuracy,
                    "candidate_path_purity": candidate_purity,
                    "reference_path_purity": reference_purity,
                    "path_purity_delta": candidate_purity - reference_purity,
                    "rescue_only_edge_count": len(rescue_only),
                    "selected_rescue_only_edges": len(selected_edges & rescue_only),
                    "selected_rescue_only_fraction": (
                        len(selected_edges & rescue_only) / (TILE_COUNT - GRID)
                    ),
                }
            )
        records.append(
            {
                "name": name,
                "panel": args.panel,
                "seed": panel_seed,
                "reference": reference_diagnostics,
                "axes": axis_payload,
                "solver_target_or_truth_argument": False,
                "layout_or_ssim_constructed": False,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "path_cover_axis_gate",
                    "panel": args.panel,
                    "done": source_index + 1,
                    "total": len(names),
                }
            ),
            flush=True,
        )

    summary = {
        axis: _panel_summary(records, axis) for axis in ("right", "down")
    }
    total_solver_times = [
        record["axes"]["right"]["solver_seconds"]
        + record["axes"]["down"]["solver_seconds"]
        for record in records
    ]
    summary["max_path_cover_seconds_per_source"] = float(max(total_solver_times))
    gate = _panel_gate({axis: summary[axis] for axis in ("right", "down")})
    gate["checks"]["max_60_path_cover_seconds_per_source"] = (
        summary["max_path_cover_seconds_per_source"] <= 60.5
    )
    gate["passed"] = all(gate["checks"].values())
    if not gate["passed"]:
        gate["decision"] = "stop_path_cover_no_axis_signal"

    payload = {
        "schema_version": 1,
        "kind": "exact_24x24_axis_path_cover_prerequisite_panel",
        "safe_for_submission": False,
        "panel": args.panel,
        "protocol": {
            "split": args.split,
            "source_offset": args.source_offset,
            "source_count": args.sources,
            "source_names": names,
            "source_names_sha256": actual_names_sha256,
            "source_names_hash_contract": "sha256((newline.join(names) + newline).utf8)",
            "outgoing_top_k": args.outgoing_top_k,
            "incoming_top_k": args.incoming_top_k,
            "path_count": GRID,
            "path_length": GRID,
            "solver_deterministic_time_limit_per_axis": args.time_limit_seconds,
            "solver_time_limit_kind": "cp_sat_deterministic_time",
            "solver_workers": 1,
            "parameter_sweeps": 0,
            "reference": "unchanged promoted soft-cycle-L1 seed plus QAP-L1w4 layout",
            "reference_rescue_edges_only": True,
            "assembly_layout_constructed": False,
            "layout_ssim_opened": False,
            "solver_accepts_target_or_truth": False,
            "exact_panel_generator_constructs_truth_before_solver": True,
        },
        "assets": {
            "denoiser": {"path": str(args.denoiser), "sha256": actual_assets["denoiser"]},
            "embedding": {
                "path": str(args.embedding_checkpoint),
                "sha256": actual_assets["embedding"],
            },
        },
        "code_and_configs": {
            "evaluator_sha256": sha256(__file__),
            "path_cover_core_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/path_cover.py"),
            "candidate_producer_sha256": sha256(SCRIPT_ROOT / "train_binary_edge_verifier.py"),
            "components_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/components.py"),
            "qap_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/qap.py"),
            "compatibility_sha256": sha256(
                REPO_ROOT / "src/puzzle_assembly/compatibility.py"
            ),
            "learned_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/learned.py"),
            "panels_sha256": sha256(REPO_ROOT / "src/puzzle_assembly/panels.py"),
            "inference_sha256": sha256(
                REPO_ROOT / "src/puzzle_denoise_v2/inference.py"
            ),
            "manifest_sha256": sha256(args.manifest),
            "quarantine_sha256": sha256(args.quarantine),
            "audit_exclusion_sha256": sha256(args.audit_exclusion),
        },
        "summary": summary,
        "gate": gate,
        "records": records,
        "denoiser_metadata": denoiser_metadata,
        "embedding_metadata": embedding_metadata,
        "seconds": time.time() - started,
    }
    _assert_finite_payload(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256(output), "gate": gate},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
