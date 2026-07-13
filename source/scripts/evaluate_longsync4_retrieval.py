#!/usr/bin/env python3
"""Evaluate a parameter-free LongSync-4 rerank of frozen HGB top-2 edges.

This is deliberately a retrieval-only diagnostic.  It never constructs an
image layout and therefore cannot consume assembly targets.  The frozen HGB
model defines the candidate graph; LongSync may only swap the first two
candidates of an outgoing (direction, source-tile) query when both candidates
survive canonical undirected deduplication and both have simple 4-cycle
support.  All other candidates and all unsupported queries retain the exact
HGB ordering.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.geometry import TILE_COUNT
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.longsync_translation import (
    LongSyncTranslationResult,
    longsync4_translation,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_denoise_v2.inference import load_restorer
from train_binary_edge_verifier import (
    PreparedSource,
    binary_metrics,
    component_metrics,
    feature_names,
    prepare_source,
)


PANELS = ("primary_kornia", "independent_libjpeg")
FROZEN_SPLIT = "edge_development"
FROZEN_SOURCE_OFFSET = 316
FROZEN_SOURCE_COUNT = 8
FROZEN_SOURCE_NAMES_SHA256 = (
    "c0f9548268a4e72a07e987cfdedf98047313e61967758401b297ff60f82ff7c7"
)
FROZEN_TOP_K = 2
FROZEN_ITERATIONS = 10
EXPECTED_ASSET_SHA256 = {
    "model": "c5929a76c843f7541119f622bf1c5b6774006ad79e3811407e36edfe60bd0f10",
    "denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "embedding": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}


@dataclass(frozen=True)
class SparseHypothesisGraph:
    """Canonical one-measurement-per-unordered-pair graph for LongSync."""

    edges: np.ndarray
    displacements: np.ndarray
    owner_candidate_indices: np.ndarray
    query_top_indices: tuple[tuple[int, int], ...]
    selected_candidates: int
    deduplicated_candidates: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/assembly_v1/full_union_tabular/v1/full_union_tabular.joblib",
    )
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
    parser.add_argument("--top-k", type=int, default=FROZEN_TOP_K)
    parser.add_argument("--iterations", type=int, default=FROZEN_ITERATIONS)
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


def names_sha256(names: list[str]) -> str:
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


def _stable_query_order(
    prepared: PreparedSource, probability: np.ndarray, direction: int, source: int
) -> np.ndarray:
    graph = prepared.graph
    indices = np.flatnonzero(
        (graph.direction == direction) & (graph.source == source)
    )
    if len(indices) == 0:
        return indices
    # Candidate rows are already stable, but destination and row index make the
    # tie contract explicit and independent of incidental array ordering.
    return np.asarray(
        sorted(
            indices.tolist(),
            key=lambda index: (
                -float(probability[index]),
                int(graph.destination[index]),
                int(index),
            ),
        ),
        dtype=np.int64,
    )


def _canonical_measurement(
    source: int, destination: int, direction: int
) -> tuple[tuple[int, int], np.ndarray]:
    if source == destination:
        raise ValueError("self candidate cannot define a translation edge")
    if direction == 0:
        displacement = np.asarray([1.0, 0.0], dtype=np.float64)
    elif direction == 1:
        displacement = np.asarray([0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError(f"unsupported direction: {direction}")
    if source < destination:
        return (source, destination), displacement
    return (destination, source), -displacement


def build_sparse_hypothesis_graph(
    prepared: PreparedSource,
    probability: np.ndarray,
    *,
    top_k: int = FROZEN_TOP_K,
    require_complete: bool = False,
) -> SparseHypothesisGraph:
    """Select HGB top-k per query, then deterministically deduplicate pairs.

    LongSync's graph has a single group measurement for each unordered node
    pair.  If conflicting directional hypotheses canonicalize to the same pair,
    the highest-HGB candidate owns that measurement.  Ties go to the smaller
    candidate row index.  A query is later eligible for reranking only when
    both of its top-2 rows are owners, so deduplication can never silently lend
    one candidate another candidate's cycle score.
    """

    values = np.asarray(probability, dtype=np.float64)
    if values.shape != (len(prepared.labels),) or not np.all(np.isfinite(values)):
        raise ValueError("probability must be a finite vector aligned with candidates")
    if top_k != FROZEN_TOP_K:
        raise ValueError(f"this frozen diagnostic requires top_k={FROZEN_TOP_K}")
    graph = prepared.graph
    arrays = (
        np.asarray(graph.direction),
        np.asarray(graph.source),
        np.asarray(graph.destination),
    )
    if any(array.shape != values.shape for array in arrays):
        raise ValueError("candidate graph arrays are not aligned")
    if not np.all(np.isin(arrays[0], [0, 1])):
        raise ValueError("candidate direction is outside {0,1}")
    if (
        np.any(arrays[1] < 0)
        or np.any(arrays[1] >= TILE_COUNT)
        or np.any(arrays[2] < 0)
        or np.any(arrays[2] >= TILE_COUNT)
    ):
        raise ValueError("candidate endpoint is outside [0,576)")
    if np.any(arrays[1] == arrays[2]):
        raise ValueError("candidate graph contains a self edge")

    query_top: list[tuple[int, int]] = []
    selected: list[int] = []
    for direction in (0, 1):
        for source in range(TILE_COUNT):
            order = _stable_query_order(prepared, values, direction, source)
            if len(order) < top_k:
                if require_complete:
                    raise ValueError(
                        f"query {(direction, source)} has fewer than {top_k} candidates"
                    )
                continue
            chosen = tuple(int(index) for index in order[:top_k])
            query_top.append((chosen[0], chosen[1]))
            selected.extend(chosen)

    if require_complete and len(query_top) != 2 * TILE_COUNT:
        raise RuntimeError("expected exactly 1,152 outgoing query groups")

    owners: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for candidate_index in selected:
        pair, displacement = _canonical_measurement(
            int(graph.source[candidate_index]),
            int(graph.destination[candidate_index]),
            int(graph.direction[candidate_index]),
        )
        previous = owners.get(pair)
        if previous is None:
            owners[pair] = (candidate_index, displacement)
            continue
        previous_index = previous[0]
        candidate_key = (
            -float(values[candidate_index]),
            int(graph.direction[candidate_index]),
            int(graph.source[candidate_index]),
            int(graph.destination[candidate_index]),
            int(candidate_index),
        )
        previous_key = (
            -float(values[previous_index]),
            int(graph.direction[previous_index]),
            int(graph.source[previous_index]),
            int(graph.destination[previous_index]),
            int(previous_index),
        )
        if candidate_key < previous_key:
            owners[pair] = (candidate_index, displacement)

    ordered_pairs = sorted(owners)
    edges = np.asarray(ordered_pairs, dtype=np.int64).reshape(-1, 2)
    displacements = np.asarray(
        [owners[pair][1] for pair in ordered_pairs], dtype=np.float64
    ).reshape(-1, 2)
    owner_indices = np.asarray(
        [owners[pair][0] for pair in ordered_pairs], dtype=np.int64
    )
    return SparseHypothesisGraph(
        edges=edges,
        displacements=displacements,
        owner_candidate_indices=owner_indices,
        query_top_indices=tuple(query_top),
        selected_candidates=len(selected),
        deduplicated_candidates=len(selected) - len(ordered_pairs),
    )


def rerank_frozen_top2(
    probability: np.ndarray,
    sparse: SparseHypothesisGraph,
    result: LongSyncTranslationResult,
) -> tuple[np.ndarray, dict[str, int]]:
    """Swap only top-1/top-2 HGB values when LongSync strictly prefers top-2."""

    base = np.asarray(probability, dtype=np.float64)
    if len(sparse.edges) != len(result.corruption):
        raise ValueError("LongSync result is not aligned with sparse graph edges")
    owner_to_edge = {
        int(candidate_index): edge_index
        for edge_index, candidate_index in enumerate(
            sparse.owner_candidate_indices.tolist()
        )
    }
    adjusted = base.copy()
    counts = {
        "query_groups": len(sparse.query_top_indices),
        "eligible_groups": 0,
        "dedup_fallback_groups": 0,
        "unsupported_fallback_groups": 0,
        "swaps": 0,
    }
    for top_first, top_second in sparse.query_top_indices:
        first_edge = owner_to_edge.get(top_first)
        second_edge = owner_to_edge.get(top_second)
        if first_edge is None or second_edge is None:
            counts["dedup_fallback_groups"] += 1
            continue
        if not (result.supported[first_edge] and result.supported[second_edge]):
            counts["unsupported_fallback_groups"] += 1
            continue
        counts["eligible_groups"] += 1
        first_corruption = float(result.corruption[first_edge])
        second_corruption = float(result.corruption[second_edge])
        if second_corruption < first_corruption:
            adjusted[top_first], adjusted[top_second] = (
                adjusted[top_second],
                adjusted[top_first],
            )
            counts["swaps"] += 1
    return adjusted, counts


def retrieval_metrics(prepared: PreparedSource, score: np.ndarray) -> dict[str, float]:
    graph = prepared.graph
    values = np.asarray(score, dtype=np.float64)
    hits = {1: 0, 2: 0, 5: 0, 32: 0}
    reciprocal_rank = 0.0
    groups = 0
    for direction in (0, 1):
        for source in range(TILE_COUNT):
            order = _stable_query_order(prepared, values, direction, source)
            if len(order) == 0:
                continue
            positive = np.flatnonzero(prepared.labels[order] > 0.5)
            rank = int(positive[0]) + 1 if len(positive) else None
            groups += 1
            if rank is not None:
                reciprocal_rank += 1.0 / rank
                for cutoff in hits:
                    hits[cutoff] += int(rank <= cutoff)
    return {
        "groups": groups,
        "r1": hits[1] / groups,
        "r2": hits[2] / groups,
        "r5": hits[5] / groups,
        "r32": hits[32] / groups,
        "mrr": reciprocal_rank / groups,
    }


def _panel_summary(records: list[dict[str, Any]], panel: str) -> dict[str, Any]:
    selected = [record for record in records if record["panel"] == panel]
    deltas = {
        metric: np.asarray(
            [record["delta"][metric] for record in selected], dtype=np.float64
        )
        for metric in ("average_precision", "r1", "r5", "mrr")
    }
    return {
        "records": len(selected),
        "mean_ap_delta": float(deltas["average_precision"].mean()),
        "mean_r1_delta": float(deltas["r1"].mean()),
        "median_r1_delta": float(np.median(deltas["r1"])),
        "mean_mrr_delta": float(deltas["mrr"].mean()),
        "mean_r5_delta": float(deltas["r5"].mean()),
        "ap_wins": int(np.count_nonzero(deltas["average_precision"] > 0.0)),
        "r1_wins": int(np.count_nonzero(deltas["r1"] > 0.0)),
        "r1_ties": int(np.count_nonzero(deltas["r1"] == 0.0)),
        "r1_losses": int(np.count_nonzero(deltas["r1"] < 0.0)),
        "mean_eligible_group_fraction": float(
            np.mean(
                [
                    record["rerank"]["eligible_groups"]
                    / max(1, record["rerank"]["query_groups"])
                    for record in selected
                ]
            )
        ),
        "mean_supported_edge_fraction": float(
            np.mean([record["graph"]["supported_edge_fraction"] for record in selected])
        ),
    }


def _gate(panel_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for panel in PANELS:
        summary = panel_summaries[panel]
        checks[f"{panel}_ap_delta_ge_0.005"] = summary["mean_ap_delta"] >= 0.005
        checks[f"{panel}_r1_delta_ge_0.01"] = summary["mean_r1_delta"] >= 0.01
        checks[f"{panel}_mrr_delta_ge_0"] = summary["mean_mrr_delta"] >= 0.0
        checks[f"{panel}_r5_delta_ge_0"] = summary["mean_r5_delta"] >= 0.0
        checks[f"{panel}_ap_wins_ge_6_of_8"] = summary["ap_wins"] >= 6
        checks[f"{panel}_r1_wins_ge_6_of_8"] = summary["r1_wins"] >= 6
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "decision": "continue_to_disjoint_assembly_gate" if passed else "stop_no_retrieval_signal",
        "scope": "retrieval only; no assembly target was opened",
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
        or args.top_k != FROZEN_TOP_K
        or args.iterations != FROZEN_ITERATIONS
    ):
        raise SystemExit("frozen LongSync-4 diagnostic protocol drift")
    output.parent.mkdir(parents=True, exist_ok=True)

    actual_asset_hashes = {
        "model": sha256(args.model),
        "denoiser": sha256(args.denoiser),
        "embedding": sha256(args.embedding_checkpoint),
    }
    if actual_asset_hashes != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"frozen asset hash drift: {actual_asset_hashes}")
    artifact = joblib.load(args.model)
    if not isinstance(artifact, dict) or set(artifact) < {
        "model",
        "feature_names",
        "threshold",
        "fit_names",
        "calibration_names",
    }:
        raise RuntimeError("unexpected frozen HGB artifact schema")
    if list(artifact["feature_names"]) != feature_names():
        raise RuntimeError("frozen HGB feature schema drift")
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    source_names = source_names_for_split(
        args.split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
        audit_exclusion_path=args.audit_exclusion,
    )[args.source_offset : args.source_offset + args.sources]
    if len(source_names) != FROZEN_SOURCE_COUNT:
        raise RuntimeError("frozen source slice is unavailable")
    actual_names_sha256 = names_sha256(source_names)
    if actual_names_sha256 != FROZEN_SOURCE_NAMES_SHA256:
        raise RuntimeError("frozen source-name fingerprint drift")
    training_names = set(artifact["fit_names"]) | set(artifact["calibration_names"])
    if training_names & set(source_names):
        raise RuntimeError("whole-source overlap with frozen HGB fit/calibration")
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
        if protected & set(source_names):
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

    started = time.time()
    records: list[dict[str, Any]] = []
    for source_index, name in enumerate(source_names):
        for panel in PANELS:
            seed = per_source_seed(args.seed, f"longsync4-retrieval-{panel}", name, 0)
            prepared = prepare_source(
                name,
                panel,
                seed,
                args=args,
                restorer=restorer,
                embedding_model=embedding,
                device=device,
            )
            probability = np.asarray(
                model.predict_proba(prepared.features)[:, 1], dtype=np.float64
            )
            sparse = build_sparse_hypothesis_graph(
                prepared,
                probability,
                top_k=args.top_k,
                require_complete=True,
            )
            result = longsync4_translation(
                TILE_COUNT,
                sparse.edges,
                sparse.displacements,
                iterations=args.iterations,
            )
            if source_index == 0 and panel == PANELS[0]:
                repeated = longsync4_translation(
                    TILE_COUNT,
                    sparse.edges,
                    sparse.displacements,
                    iterations=args.iterations,
                )
                if (
                    not np.array_equal(result.corruption_history, repeated.corruption_history)
                    or not np.array_equal(result.support_counts, repeated.support_counts)
                    or result.alternate_paths != repeated.alternate_paths
                ):
                    raise RuntimeError("LongSync rerun is not bitwise deterministic")
            adjusted, rerank = rerank_frozen_top2(probability, sparse, result)
            base_retrieval = retrieval_metrics(prepared, probability)
            candidate_retrieval = retrieval_metrics(prepared, adjusted)
            base_binary = binary_metrics(prepared.labels, probability)
            candidate_binary = binary_metrics(prepared.labels, adjusted)
            base_components = component_metrics(prepared, probability, threshold)
            candidate_components = component_metrics(prepared, adjusted, threshold)
            records.append(
                {
                    "name": name,
                    "panel": panel,
                    "seed": seed,
                    "base": {
                        "binary": base_binary,
                        "retrieval": base_retrieval,
                        "components": base_components,
                    },
                    "candidate": {
                        "binary": candidate_binary,
                        "retrieval": candidate_retrieval,
                        "components": candidate_components,
                    },
                    "delta": {
                        "average_precision": (
                            candidate_binary["average_precision"]
                            - base_binary["average_precision"]
                        ),
                        "r1": candidate_retrieval["r1"] - base_retrieval["r1"],
                        "mrr": candidate_retrieval["mrr"] - base_retrieval["mrr"],
                        "r5": candidate_retrieval["r5"] - base_retrieval["r5"],
                        "accepted_precision": (
                            candidate_components["accepted_precision"]
                            - base_components["accepted_precision"]
                        ),
                        "largest_component": (
                            candidate_components["largest_component"]
                            - base_components["largest_component"]
                        ),
                    },
                    "graph": {
                        "selected_candidates": sparse.selected_candidates,
                        "canonical_edges": len(sparse.edges),
                        "deduplicated_candidates": sparse.deduplicated_candidates,
                        "supported_edges": int(np.count_nonzero(result.supported)),
                        "supported_edge_fraction": float(np.mean(result.supported)),
                        "mean_support_count": float(result.support_counts.mean()),
                        "max_support_count": int(result.support_counts.max(initial=0)),
                    },
                    "rerank": rerank,
                }
            )
        print(
            json.dumps(
                {
                    "stage": "longsync4_retrieval",
                    "done": source_index + 1,
                    "total": len(source_names),
                }
            ),
            flush=True,
        )

    panel_summaries = {
        panel: _panel_summary(records, panel) for panel in PANELS
    }
    if len(records) != FROZEN_SOURCE_COUNT * len(PANELS):
        raise RuntimeError("expected exactly 16 source-panel records")
    gate = _gate(panel_summaries)
    payload = {
        "schema_version": 1,
        "kind": "longsync4_translation_hgb_top2_retrieval_diagnostic",
        "safe_for_submission": False,
        "protocol": {
            "split": args.split,
            "source_offset": args.source_offset,
            "source_count": args.sources,
            "source_names": source_names,
            "source_names_sha256": actual_names_sha256,
            "source_names_hash_contract": "sha256((newline.join(names) + newline).utf8)",
            "panels": list(PANELS),
            "top_k": args.top_k,
            "iterations": args.iterations,
            "rerank_rule": (
                "swap only the frozen HGB top-1/top-2 probability values when "
                "both candidates own canonical edges, both are 4-cycle-supported, "
                "and LongSync corruption(top-2) < corruption(top-1); otherwise "
                "preserve the exact HGB order"
            ),
            "parameter_sweeps": 0,
            "assembly_targets_opened": False,
            "whole_source_disjoint_from_hgb_fit_calibration": True,
        },
        "assets": {
            "model": {"path": str(args.model), "sha256": actual_asset_hashes["model"]},
            "denoiser": {"path": str(args.denoiser), "sha256": actual_asset_hashes["denoiser"]},
            "embedding": {
                "path": str(args.embedding_checkpoint),
                "sha256": actual_asset_hashes["embedding"],
            },
        },
        "code_and_configs": {
            "evaluator_sha256": sha256(__file__),
            "longsync_core_sha256": sha256(
                REPO_ROOT / "src/puzzle_assembly/longsync_translation.py"
            ),
            "candidate_producer_sha256": sha256(
                SCRIPT_ROOT / "train_binary_edge_verifier.py"
            ),
            "manifest_sha256": sha256(args.manifest),
            "quarantine_sha256": sha256(args.quarantine),
            "audit_exclusion_sha256": sha256(args.audit_exclusion),
        },
        "frozen_hgb_threshold": threshold,
        "panels": panel_summaries,
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
