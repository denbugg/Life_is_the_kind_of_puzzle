"""Select a raw-CNN/LambdaRank residual blend on validation-only scenes."""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from config import WORK_ROOT
from eval_candidate_calibrator import (
    evaluate_ranker,
    load_graph,
    model_scores,
    solver_metrics,
)


class ResidualBlend:
    """Row-local standardized blend; feature column zero is direct log-prob."""

    def __init__(self, ranker: object, weight: float) -> None:
        self.ranker = ranker
        self.weight = float(weight)

    @staticmethod
    def _z(value: np.ndarray) -> np.ndarray:
        return (value - value.mean()) / max(float(value.std()), 1.0e-4)

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = features[:, 0]
        learned = model_scores(self.ranker, features)
        return self._z(raw) + self.weight * self._z(learned)


def _graphs(cache_dir: Path, ids: str) -> list:
    return [
        load_graph(cache_dir / f"image_{int(part):04d}_k64.npz")
        for part in ids.split(",")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(WORK_ROOT)
        / "edge_confidence"
        / "candidate_lambdarank_fullrow.pkl",
    )
    parser.add_argument("--validation-images", default="18,19,20,21")
    parser.add_argument("--external-images", default="50,51,52,53,54,55")
    parser.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,1,1.5,2,3")
    parser.add_argument("--budgets", default="128,256,384,512")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "candidate_residual_blend_gate.json",
    )
    args = parser.parse_args()
    with args.checkpoint.open("rb") as handle:
        ranker = pickle.load(handle)["model"]
    validation = _graphs(args.cache_dir, args.validation_images)
    external = _graphs(args.cache_dir, args.external_images)
    weights = tuple(float(value) for value in args.weights.split(","))
    validation_sweep = {}
    # LightGBM emits a feature-name warning for each small row prediction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for weight in weights:
            validation_sweep[str(weight)] = evaluate_ranker(
                ResidualBlend(ranker, weight), validation
            )
        best_weight = max(
            weights,
            key=lambda value: (
                validation_sweep[str(value)]["calibrated_conditional_r1"],
                validation_sweep[str(value)]["calibrated_conditional_r5"],
            ),
        )
        selected = ResidualBlend(ranker, best_weight)
        external_metrics = evaluate_ranker(selected, external)
        budgets = tuple(int(value) for value in args.budgets.split(","))
        solver = solver_metrics(selected, external, budgets)
    best_budget = max(
        solver,
        key=lambda key: (solver[key]["neighbour"], solver[key]["placement"]),
    )
    baseline = validation_sweep["0.0"]
    selected_validation = validation_sweep[str(best_weight)]
    checks = {
        "validation_r1_delta": (
            selected_validation["calibrated_conditional_r1"]
            - baseline["calibrated_conditional_r1"]
            >= 0.015
        ),
        "external_r1_delta": (
            external_metrics["calibrated_conditional_r1"]
            - external_metrics["base_conditional_r1"]
            >= 0.015
        ),
        "external_neighbour": solver[best_budget]["neighbour"] >= 0.18,
    }
    report = {
        "experiment": "validation_selected_raw_lambdarank_residual",
        "status": "pass" if all(checks.values()) else "fail",
        "checkpoint": str(args.checkpoint),
        "validation_sweep": validation_sweep,
        "selected_weight": best_weight,
        "external": external_metrics,
        "solver": solver,
        "best_budget": best_budget,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
