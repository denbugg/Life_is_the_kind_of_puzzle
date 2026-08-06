"""Select a raw/GNN residual and its role in component assembly on validation.

The GNN improves candidate ranking but can replace a few high-precision raw
seeds with weaker edges.  This gate separates component discovery from packing:
raw components may stay frozen while a residual blend scores their contacts.
Configuration selection uses images 18--21 only; 50--55 remain external.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import WORK_ROOT
from eval_candidate_calibrator import load_graph
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import build_buddies_components, solve_components_from_scores
from train_graph_message_refiner import (
    GraphMessageRefiner,
    GraphTensor,
    _dense_scores,
    _paths,
    tensorize,
)


def load_tensors(
    cache_dir: Path,
    images: str,
    top_k: int,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> list[GraphTensor]:
    output = []
    for path in _paths(cache_dir, images):
        graph = tensorize(load_graph(path), top_k)
        graph.features = ((graph.features - mean) / scale).clamp(-8.0, 8.0)
        output.append(graph)
    return output


@torch.inference_mode()
def score_graph(
    model: GraphMessageRefiner, graph: GraphTensor, device: torch.device
) -> np.ndarray:
    value = graph.to(device)
    return (
        model(value.candidates, value.features, value.base, value.node_features)
        .float().cpu().numpy()
    )


def evaluate_config(
    model: GraphMessageRefiner,
    graphs: list[GraphTensor],
    device: torch.device,
    *,
    weight: float,
    component_budget: int,
    component_source: str,
) -> dict[str, float]:
    rows = []
    for graph in graphs:
        refined = score_graph(model, graph, device)
        blend = graph.base.numpy() + weight * (refined - graph.base.numpy())
        raw_right, raw_down = _dense_scores(graph, graph.base.numpy())
        blend_right, blend_down = _dense_scores(graph, blend)
        if component_source == "raw":
            components = build_buddies_components(
                raw_right, raw_down, max_edges=component_budget
            )
        elif component_source == "blend":
            components = build_buddies_components(
                blend_right, blend_down, max_edges=component_budget
            )
        else:
            raise ValueError(component_source)
        placement, objective = solve_components_from_scores(
            blend_right, blend_down, components, repair_passes=0
        )
        truth = np.argsort(graph.permutation)
        rows.append(
            {
                "placement": placement_accuracy(placement, truth)[0],
                "neighbour": neighbour_accuracy(placement, truth)[0],
                "objective": float(objective),
            }
        )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "graph_message_refiner.pt",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--validation-images", default="18,19,20,21")
    parser.add_argument("--external-images", default="50,51,52,53,54,55")
    parser.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,1,1.5")
    parser.add_argument("--budgets", default="128,192,256,384,512")
    parser.add_argument("--component-sources", default="raw,blend")
    parser.add_argument(
        "--report", type=Path,
        default=Path(WORK_ROOT) / "gates" / "graph_message_blend_gate.json",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GraphMessageRefiner(
        checkpoint["feature_dim"], checkpoint["hidden"], checkpoint["rounds"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = checkpoint["mean"].cpu()
    scale = checkpoint["scale"].cpu()
    top_k = int(checkpoint["top_k"])
    validation = load_tensors(
        args.cache_dir, args.validation_images, top_k, mean, scale
    )
    external = load_tensors(
        args.cache_dir, args.external_images, top_k, mean, scale
    )
    weights = [float(value) for value in args.weights.split(",")]
    budgets = [int(value) for value in args.budgets.split(",")]
    sources = [value.strip() for value in args.component_sources.split(",")]
    validation_rows: dict[str, dict[str, float]] = {}
    for source in sources:
        for budget in budgets:
            for weight in weights:
                key = f"{source}:b{budget}:w{weight:g}"
                metrics = evaluate_config(
                    model, validation, device, weight=weight,
                    component_budget=budget, component_source=source,
                )
                validation_rows[key] = metrics
                print(key + " " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    selected = max(
        validation_rows,
        key=lambda key: (
            validation_rows[key]["neighbour"],
            validation_rows[key]["placement"],
        ),
    )
    source_text, budget_text, weight_text = selected.split(":")
    budget = int(budget_text[1:])
    weight = float(weight_text[1:])
    external_selected = evaluate_config(
        model, external, device, weight=weight,
        component_budget=budget, component_source=source_text,
    )
    external_baseline = evaluate_config(
        model, external, device, weight=0.0,
        component_budget=384, component_source="raw",
    )
    delta = {
        key: external_selected[key] - external_baseline[key]
        for key in ("placement", "neighbour")
    }
    report = {
        "experiment": "graph_message_residual_role_selection",
        "selected": selected,
        "validation": validation_rows[selected],
        "external_selected": external_selected,
        "external_baseline_raw384": external_baseline,
        "delta": delta,
        "thresholds": {"neighbour": 0.01, "placement": 0.005},
        "passed": delta["neighbour"] >= 0.01 and delta["placement"] >= 0.005,
        "all_validation": validation_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
