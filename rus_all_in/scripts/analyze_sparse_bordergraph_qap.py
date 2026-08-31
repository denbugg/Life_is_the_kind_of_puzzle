#!/usr/bin/env python3
"""Post-hoc decomposition on the already-opened Rank2 exact16 panel.

This analysis opens no new source.  It reconstructs the frozen dirty evidence
for the exact roster already stored in the pilot report and separates unary
from genuinely quadratic energy.  It also measures sparse top-8 coverage and
R@1 before/after the learned edge residual.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.sparse_bordergraph_qap import (
    edge_truth_labels,
    layout_to_probability,
    sparse_quadratic_energy,
)

try:
    from scripts.run_component_relation_reranker import CleanTileCache, prepare_case
    from scripts.run_sparse_bordergraph_qap import (
        GRID,
        TILE_COUNT,
        _load_frozen_models,
        _load_json,
        _model_from_config,
        _tensor_example,
        prepare_blind_evidence,
    )
except ModuleNotFoundError:  # Direct ``python scripts/*.py`` execution.
    from run_component_relation_reranker import CleanTileCache, prepare_case
    from run_sparse_bordergraph_qap import (
        GRID,
        TILE_COUNT,
        _load_frozen_models,
        _load_json,
        _model_from_config,
        _tensor_example,
        prepare_blind_evidence,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    return parser.parse_args()


def _layout_components(
    output: Any,
    layout: np.ndarray,
    sources: torch.Tensor,
    targets: torch.Tensor,
    directions: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    probability = layout_to_probability(layout, grid=GRID, device=device).to(
        output.unary_logits.dtype
    )
    unary = float((probability * output.unary_logits).sum().detach().cpu())
    quadratic = float(
        sparse_quadratic_energy(
            probability,
            sources,
            targets,
            directions,
            output.edge_weights,
            grid=GRID,
        )
        .detach()
        .cpu()
    )
    return {
        "unary": unary,
        "quadratic": quadratic,
        "joint": unary + 2.0 * quadratic,
    }


def _edge_metrics(
    output: Any,
    edge_features: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
    directions: torch.Tensor,
    reference: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    tile_to_slot = layout_to_probability(reference, grid=GRID, device=device).argmax(1)
    labels = edge_truth_labels(
        tile_to_slot,
        sources,
        targets,
        directions,
        grid=GRID,
    ).bool()
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (source, direction) in enumerate(
        zip(sources.tolist(), directions.tolist(), strict=True)
    ):
        grouped[(source, direction)].append(index)
    supplied = 0
    priority_correct = 0
    learned_correct = 0
    eligible = 0
    for indices in grouped.values():
        index = torch.tensor(indices, device=device, dtype=torch.long)
        positive = labels[index]
        if not bool(positive.any()):
            continue
        supplied += 1
        eligible += 1
        priority_best = int(torch.argmax(edge_features[index, 0]))
        learned_best = int(torch.argmax(output.edge_logits[index]))
        priority_correct += int(bool(positive[priority_best]))
        learned_correct += int(bool(positive[learned_best]))
    return {
        "query_count": len(grouped),
        "supplied_positive_queries": supplied,
        "top8_coverage": supplied / len(grouped),
        "frozen_priority_r1": priority_correct / eligible,
        "learned_edge_r1": learned_correct / eligible,
        "eligible_queries": eligible,
    }


def main() -> None:
    args = parse_args()
    report = _load_json(args.report)
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    if report.get("experiment") != "sparse-bordergraph-qap-v1":
        raise ValueError("unexpected pilot report")
    if report["config"]["sha256"] != sha256_file(args.config):
        raise ValueError("pilot/config binding changed")
    names = report["selection"]["evaluation_source_filenames"]
    by_name = {record["filename"]: record for record in manifest["splits"]["train"]}
    records = [by_name[name] for name in names]
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    models = _load_frozen_models(config, device=device)
    model = _model_from_config(config, device=device)
    checkpoint_path = Path(report["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != report["checkpoint"]["sha256"]:
        raise ValueError("QAP checkpoint hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    frozen_cases = {
        item["case_id"]: item
        for item in _load_json(
            Path(report["frozen_prediction_artifact"]["metadata_path"])
        )["cases"]
    }
    cache = CleanTileCache(args.targets)
    boards: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            case = prepare_case(
                cache,
                record,
                draw_index=0,
                seed=int(config["selection"]["seed"]) + 100_000,
            )
            evidence = prepare_blind_evidence(
                case,
                models,
                device=device,
                inference_batch=576,
                topk=int(config["sparse_graph"]["topk_per_tile_direction"]),
            )
            if evidence.dirty_tiles_sha256 != frozen_cases[case.case_id]["dirty_tiles_sha256"]:
                raise RuntimeError("reconstructed dirty evidence differs from frozen prediction")
            reference = np.empty(TILE_COUNT, dtype=np.int32)
            reference[case.input_tile_to_position] = np.arange(TILE_COUNT, dtype=np.int32)
            tensors = _tensor_example(evidence, device=device)
            tile_features, edge_features, sources, targets, directions = tensors
            output = model(
                tile_features,
                edge_features,
                sources,
                targets,
                directions,
                evidence.baseline_layout,
                grid=GRID,
            )
            arrays = np.load(report["frozen_prediction_artifact"]["arrays_path"])
            prefix = f"case_{index - 1:04d}"
            candidate = arrays[f"{prefix}__qap"]
            truth = _layout_components(
                output,
                reference,
                sources,
                targets,
                directions,
                device=device,
            )
            baseline = _layout_components(
                output,
                evidence.baseline_layout,
                sources,
                targets,
                directions,
                device=device,
            )
            candidate_energy = _layout_components(
                output,
                candidate,
                sources,
                targets,
                directions,
                device=device,
            )
            boards.append(
                {
                    "case_id": case.case_id,
                    "source_filename": case.source_filename,
                    "energy": {
                        "truth": truth,
                        "baseline": baseline,
                        "candidate": candidate_energy,
                        "truth_minus_baseline": {
                            key: truth[key] - baseline[key] for key in truth
                        },
                    },
                    "edge_metrics": _edge_metrics(
                        output,
                        edge_features,
                        sources,
                        targets,
                        directions,
                        reference,
                        device=device,
                    ),
                }
            )
            print(json.dumps({"event": "analyze", "done": index, "total": len(records)}))
    mean_quadratic = float(
        np.mean([row["energy"]["truth_minus_baseline"]["quadratic"] for row in boards])
    )
    aggregate_edges = {
        key: float(np.mean([row["edge_metrics"][key] for row in boards]))
        for key in ("top8_coverage", "frozen_priority_r1", "learned_edge_r1")
    }
    payload = {
        "schema": "aiijc-sparse-bordergraph-qap-energy-analysis-v1",
        "source_report": {"path": str(args.report.resolve()), "sha256": sha256_file(args.report)},
        "same_already_opened_exact16_only": True,
        "mean_truth_minus_frozen_decoder": {
            key: float(
                np.mean([row["energy"]["truth_minus_baseline"][key] for row in boards])
            )
            for key in ("unary", "quadratic", "joint")
        },
        "edge_metrics_mean": aggregate_edges,
        "quadratic_energy_gate_pass": mean_quadratic > 0,
        "boards": boards,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "sha256": sha256_file(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
