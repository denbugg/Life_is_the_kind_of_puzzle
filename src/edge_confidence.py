"""Scene-conditioned correctness calibration for frozen directed edge predictions."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EdgeConfidenceMLP(nn.Module):
    def __init__(self, features: int, hidden: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.features = int(features)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    scale = features.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1.0e-4)
    return mean, scale


def standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((features - mean) / scale, -8.0, 8.0).astype(np.float32)


def choose_precision_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float,
    minimum_edges: int,
) -> dict[str, float]:
    """Maximum-coverage prefix whose empirical precision meets the target."""
    order = np.argsort(-probabilities)
    sorted_labels = labels[order].astype(np.float64)
    cumulative = np.cumsum(sorted_labels)
    counts = np.arange(1, len(labels) + 1)
    precision = cumulative / counts
    eligible = np.flatnonzero(
        (counts >= minimum_edges) & (precision >= target_precision)
    )
    if not len(eligible):
        return {
            "threshold": float("inf"),
            "precision": 0.0,
            "coverage": 0.0,
            "accepted": 0.0,
        }
    end = int(eligible[-1])
    # Midpoint prevents accidental inclusion of the next lower score.
    if end + 1 < len(order):
        threshold = 0.5 * (
            float(probabilities[order[end]]) + float(probabilities[order[end + 1]])
        )
    else:
        threshold = float(probabilities[order[end]]) - 1.0e-7
    return {
        "threshold": threshold,
        "precision": float(precision[end]),
        "coverage": float((end + 1) / len(labels)),
        "accepted": float(end + 1),
    }


def threshold_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    accepted = probabilities >= threshold
    correct = accepted & labels.astype(bool)
    total = len(labels)
    accepted_count = int(accepted.sum())
    image_precision: list[float] = []
    image_coverage: list[float] = []
    for image in np.unique(image_ids):
        mask = image_ids == image
        count = int((accepted & mask).sum())
        image_precision.append(
            float((correct & mask).sum() / count) if count else 0.0
        )
        image_coverage.append(float((accepted & mask).sum() / mask.sum()))
    return {
        "precision": float(correct.sum() / accepted_count) if accepted_count else 0.0,
        "acceptance_coverage": float(accepted_count / total),
        "exact_edge_coverage": float(correct.sum() / total),
        "accepted": float(accepted_count),
        "rows": float(total),
        "worst_image_precision": float(min(image_precision, default=0.0)),
        "mean_image_precision": float(np.mean(image_precision)),
        "worst_image_coverage": float(min(image_coverage, default=0.0)),
    }


def ranking_diagnostics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.15),
    minimum_edges: int = 20,
) -> dict[str, float]:
    """Describe the attainable precision/coverage tradeoff without a gate."""
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    order = np.argsort(-scores)
    ranked = labels[order].astype(np.float64)
    cumulative = np.cumsum(ranked)
    counts = np.arange(1, len(ranked) + 1)
    precision = cumulative / counts
    eligible = counts >= min(minimum_edges, len(ranked))
    result: dict[str, float] = {
        "positive_rate": float(labels.mean()),
        "max_precision_min_edges": float(precision[eligible].max()) if eligible.any() else 0.0,
    }
    for coverage in coverages:
        count = min(len(ranked), max(1, int(np.ceil(len(ranked) * coverage))))
        result[f"precision_at_{coverage:.2f}_coverage"] = float(precision[count - 1])
    return result


def smoke_test(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    rng = np.random.default_rng(71)
    x = rng.normal(size=(128, 12)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0.5).astype(np.float32)
    mean, scale = fit_standardizer(x)
    standardized = torch.from_numpy(standardize(x, mean, scale)).to(device)
    labels = torch.from_numpy(y).to(device)
    model = EdgeConfidenceMLP(12, hidden=16, dropout=0.0).to(device)
    loss = F.binary_cross_entropy_with_logits(model(standardized), labels)
    loss.backward()
    probabilities = 1.0 / (1.0 + np.exp(-x[:, 0]))
    selected = choose_precision_threshold(
        probabilities, y, target_precision=0.75, minimum_edges=5
    )
    metrics = threshold_metrics(
        probabilities,
        y,
        np.repeat(np.arange(4), 32),
        selected["threshold"],
    )
    if not torch.isfinite(loss):
        raise AssertionError("non-finite confidence loss")
    return {"loss": float(loss.detach()), **selected, **metrics}


if __name__ == "__main__":
    print(smoke_test(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
