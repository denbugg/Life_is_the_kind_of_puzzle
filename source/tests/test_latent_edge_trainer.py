from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/train_evaluate_latent_edge_embedding.py"
)
_SPEC = importlib.util.spec_from_file_location("latent_edge_trainer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
trainer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trainer
_SPEC.loader.exec_module(trainer)


def _metric(r1: float, mrr: float, r5: float, r32: float) -> dict[str, float]:
    return {
        "recall_at_1": r1,
        "recall_at_5": r5,
        "recall_at_10": max(r5, r32 - 0.1),
        "recall_at_20": max(r5, r32 - 0.03),
        "recall_at_32": r32,
        "mrr": mrr,
        "median_rank": 5.0,
        "q90_rank": 80.0,
        "queries": 1104,
    }


def _aggregate(candidate_gain: float) -> dict:
    panels = {}
    for panel in ("primary_kornia", "independent_libjpeg"):
        baseline = _metric(0.20, 0.30, 0.45, 0.70)
        panels[panel] = {
            "source_count": 2,
            "candidate_coverage": 0.75,
            "scores": {
                "hbt": baseline,
                "w4": _metric(0.19, 0.29, 0.46, 0.71),
                "learned": _metric(0.22, 0.32, 0.48, 0.74),
                "alpha_0": _metric(0.19, 0.29, 0.46, 0.71),
                "alpha_0.05": _metric(
                    0.20 + candidate_gain,
                    0.30 + candidate_gain,
                    0.46,
                    0.71,
                ),
            },
            "paired": {
                "alpha_0": {
                    "recall_at_1": trainer._paired_stats(
                        [-0.01, -0.01], seed_label=f"{panel}:alpha0:r1"
                    ),
                    "mrr": trainer._paired_stats(
                        [-0.01, -0.01], seed_label=f"{panel}:alpha0:mrr"
                    ),
                },
                "alpha_0.05": {
                    "recall_at_1": trainer._paired_stats(
                        [candidate_gain, candidate_gain],
                        seed_label=f"{panel}:alpha005:r1",
                    ),
                    "mrr": trainer._paired_stats(
                        [candidate_gain, candidate_gain],
                        seed_label=f"{panel}:alpha005:mrr",
                    ),
                },
            },
        }
    return panels


def test_alpha_gate_requires_both_panels_and_selects_passing_candidate() -> None:
    result = trainer.choose_alpha(_aggregate(0.02), [0.0, 0.05])
    assert result["passed"] is True
    assert result["selected_alpha"] == 0.05
    assert all(panel["passed"] for panel in result["selected"]["panels"].values())


def test_alpha_gate_fails_closed_without_required_delta() -> None:
    result = trainer.choose_alpha(_aggregate(0.004), [0.0, 0.05])
    assert result["passed"] is False


def test_smoke_args_keep_identity_alpha_and_shrink_model() -> None:
    args = trainer.parse_args(["--output-dir", "unused", "--smoke"])
    assert args.train_sources == 2
    assert args.model_dim == 32
    assert trainer._alphas(args)[0] == 0.0


def test_candidate_aligned_loss_uses_only_covered_frozen_proposals() -> None:
    labels = trainer.direction_labels(np.arange(576, dtype=np.int32))
    rng = np.random.default_rng(17)
    candidates = []
    targets = (
        np.where(np.arange(576) % 24 < 23, np.arange(576) + 1, -1),
        np.where(np.arange(576) < 552, np.arange(576) + 24, -1),
    )
    for truth in targets:
        selected = np.empty((576, 8), dtype=np.int32)
        for query in range(576):
            forbidden = {query}
            if truth[query] >= 0:
                forbidden.add(int(truth[query]))
            pool = np.asarray([value for value in range(576) if value not in forbidden])
            row = rng.choice(pool, size=7, replace=False).astype(np.int32)
            first = int(truth[query]) if truth[query] >= 0 else int(pool[0])
            selected[query] = np.concatenate([[first], row])
        candidates.append(selected)
    generator = torch.Generator().manual_seed(19)
    raw = {
        key: torch.randn(576, 12, generator=generator, requires_grad=True)
        for key in ("q_right", "k_left", "q_down", "k_up")
    }
    outputs = {key: torch.nn.functional.normalize(value, dim=1) for key, value in raw.items()}
    loss, metrics = trainer.candidate_aligned_loss(
        outputs, labels, tuple(candidates), temperature=0.07
    )
    assert torch.isfinite(loss)
    assert metrics["candidate_coverage"] == 1.0
    loss.backward()
    assert raw["q_right"].grad is not None
