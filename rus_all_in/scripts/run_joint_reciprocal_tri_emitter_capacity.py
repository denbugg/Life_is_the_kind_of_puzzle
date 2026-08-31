#!/usr/bin/env python3
"""Run the signed synthetic 4x4 collision capacity test only.

This runner has no organizer manifest, target archive, real-panel or Weco path.
It exists solely to verify the plumbing of the joint row/column objective,
learned NONE classes, differentiable confidence and fixed 5% reciprocal head.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    RECIPROCAL_HEAD_FRACTION,
    JointReciprocalTriEmitterVerifier,
    dense_two_sided_confidence,
    exact_joint_targets,
    fixed_fraction_reciprocal_head,
    joint_assignment_loss,
    joint_verifier_contract,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.tri_emitter_edge_verifier import AUXILIARY_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/joint_reciprocal_tri_emitter_capacity_preregistered_v2.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/joint-reciprocal-tri-emitter-verifier/capacity4x4-collision-v2"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed joint reciprocal capacity config is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("joint reciprocal capacity config sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("status") != "signed-fixed-protocol":
        raise RuntimeError("joint reciprocal capacity protocol is not signed/fixed")
    for artifact in config["frozen_implementation"].values():
        target = PROJECT_ROOT / artifact["path"]
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"frozen capacity implementation changed: {target}")
    return config, digest


def _truth_for_grid(*, grid: int, axis: int) -> np.ndarray:
    count = grid * grid
    truth = np.full(count, -1, dtype=np.int64)
    for source in range(count):
        row, column = divmod(source, grid)
        if axis == 0 and column + 1 < grid:
            truth[source] = source + 1
        elif axis == 1 and row + 1 < grid:
            truth[source] = source + grid
    return truth


def _matched_sides(
    generator: np.random.Generator,
    *,
    grid: int,
    length: int,
    channels: int,
) -> np.ndarray:
    count = grid * grid
    sides = generator.normal(size=(4, count, length, channels)).astype(np.float32)
    for axis in range(2):
        truth = _truth_for_grid(grid=grid, axis=axis)
        for source, target in enumerate(truth):
            if target < 0:
                continue
            code = generator.normal(size=(length, channels)).astype(np.float32)
            sides[2 * axis, source] = code
            sides[2 * axis + 1, target] = code
    return sides


def make_collision_capacity_case(
    *,
    seed: int,
    grid: int = 4,
    candidate_width: int = 8,
) -> dict[str, np.ndarray]:
    """Create two sparse axes with raw-favoured many-to-one distractors."""

    if grid != 4:
        raise ValueError("the signed capacity grid is exactly 4x4")
    count = grid * grid
    if not 2 <= candidate_width < count:
        raise ValueError("candidate_width must be in [2, tile_count)")
    generator = np.random.default_rng(seed)
    raw_sides = _matched_sides(
        generator, grid=grid, length=20, channels=6
    )
    dino_sides = _matched_sides(
        generator, grid=grid, length=14, channels=8
    )
    candidates = np.empty((2, count, candidate_width), dtype=np.int64)
    valid = np.ones_like(candidates, dtype=bool)
    auxiliary = generator.normal(
        scale=0.05, size=(2, count, candidate_width, AUXILIARY_DIM)
    ).astype(np.float32)
    raw_baseline = generator.normal(
        scale=0.05, size=(2, count, candidate_width)
    ).astype(np.float32)
    truth = np.stack([_truth_for_grid(grid=grid, axis=axis) for axis in range(2)])
    collision_targets = np.array([5, 10], dtype=np.int64)
    hard_collision = np.zeros_like(valid)

    for axis in range(2):
        source_side = 2 * axis
        target_side = source_side + 1
        for source in range(count):
            required: list[int] = []
            exact = int(truth[axis, source])
            if exact >= 0:
                required.append(exact)
            collision = int(collision_targets[axis])
            if collision != source and collision != exact:
                required.append(collision)
            pool = [
                target
                for target in range(count)
                if target != source and target not in required
            ]
            generator.shuffle(pool)
            row = required + pool[: candidate_width - len(required)]
            generator.shuffle(row)
            candidates[axis, source] = row
            for slot, target in enumerate(row):
                raw_difference = np.mean(
                    np.abs(
                        raw_sides[source_side, source]
                        - raw_sides[target_side, target]
                    )
                )
                dino_difference = np.mean(
                    np.abs(
                        dino_sides[source_side, source]
                        - dino_sides[target_side, target]
                    )
                )
                # Both are target-free seam statistics.  They make this a
                # learnability/plumbing test rather than a memorisation of IDs.
                auxiliary[axis, source, slot, 0] = -raw_difference
                auxiliary[axis, source, slot, 1] = -dino_difference
                auxiliary[axis, source, slot, 2] = float(target == collision)
                if target == exact:
                    raw_baseline[axis, source, slot] += 0.75
                elif target == collision:
                    # The row-only raw control is deliberately wrong: many
                    # rows favour the same target and create a hard collision.
                    raw_baseline[axis, source, slot] += 2.5
                    hard_collision[axis, source, slot] = True

    return {
        "raw_sides": raw_sides,
        "dino_sides": dino_sides,
        "candidates": candidates,
        "valid": valid,
        "auxiliary": auxiliary,
        "raw_baseline": raw_baseline,
        "truth": truth,
        "hard_collision": hard_collision,
        "collision_targets": collision_targets,
    }


def _tensor_case(case: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "raw_sides": torch.from_numpy(case["raw_sides"]),
        "dino_sides": torch.from_numpy(case["dino_sides"]),
        "candidates": torch.from_numpy(case["candidates"]).long(),
        "valid": torch.from_numpy(case["valid"]),
        "auxiliary": torch.from_numpy(case["auxiliary"]),
        "raw_baseline": torch.from_numpy(case["raw_baseline"]),
        "truth": torch.from_numpy(case["truth"]).long(),
        "hard_collision": torch.from_numpy(case["hard_collision"]),
    }


def _classification_metrics(
    output: Any,
    targets: Any,
    valid: torch.Tensor,
) -> dict[str, float]:
    count, width = valid.shape
    row_classes = torch.cat(
        (
            output.edge_logits.masked_fill(~valid, -1.0e4),
            output.row_none_logits[:, None],
        ),
        dim=1,
    )
    row_truth = torch.where(
        targets.row_slots >= 0,
        targets.row_slots,
        torch.full_like(targets.row_slots, width),
    )
    dense = output.dense_logits.masked_fill(~output.dense_valid, -1.0e4)
    column_classes = torch.cat(
        (dense.transpose(0, 1), output.column_none_logits[:, None]), dim=1
    )
    column_truth = torch.where(
        targets.column_sources >= 0,
        targets.column_sources,
        torch.full_like(targets.column_sources, count),
    )
    row_prediction = row_classes.argmax(1)
    column_prediction = column_classes.argmax(1)
    row_edge = row_truth != width
    column_edge = column_truth != count
    return {
        "row_all_r1": float((row_prediction == row_truth).float().mean()),
        "column_all_r1": float((column_prediction == column_truth).float().mean()),
        "row_exact_edge_r1": float(
            (row_prediction[row_edge] == row_truth[row_edge]).float().mean()
        ),
        "column_exact_edge_r1": float(
            (column_prediction[column_edge] == column_truth[column_edge])
            .float()
            .mean()
        ),
        "row_none_accuracy": float(
            (row_prediction[~row_edge] == row_truth[~row_edge]).float().mean()
        ),
        "column_none_accuracy": float(
            (column_prediction[~column_edge] == column_truth[~column_edge])
            .float()
            .mean()
        ),
    }


def _invariance_errors(output: Any, *, seed: int) -> dict[str, float]:
    dense = output.dense_logits.detach()
    mask = output.dense_valid.detach()
    row_none = output.row_none_logits.detach()
    column_none = output.column_none_logits.detach()
    _, _, confidence = dense_two_sided_confidence(
        dense, mask, row_none, column_none
    )
    _, _, transposed = dense_two_sided_confidence(
        dense.transpose(0, 1),
        mask.transpose(0, 1),
        column_none,
        row_none,
    )
    transpose_error = float(
        (confidence[mask] - transposed.transpose(0, 1)[mask]).abs().max()
    )

    generator = np.random.default_rng(seed)
    order = torch.from_numpy(generator.permutation(len(dense))).long()
    permuted_dense = dense[order][:, order]
    permuted_mask = mask[order][:, order]
    _, _, permuted = dense_two_sided_confidence(
        permuted_dense,
        permuted_mask,
        row_none[order],
        column_none[order],
    )
    expected = confidence[order][:, order]
    relabel_error = float((permuted[permuted_mask] - expected[permuted_mask]).abs().max())
    return {
        "transpose_confidence_max_abs_error": transpose_error,
        "relabel_confidence_max_abs_error": relabel_error,
    }


def run_capacity(config: dict[str, Any], config_sha: str, output: Path) -> dict[str, Any]:
    capacity = config["capacity"]
    seed = int(capacity["seed"])
    steps = int(capacity["steps"])
    learning_rate = float(capacity["learning_rate"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(capacity["torch_threads"]))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    case_np = make_collision_capacity_case(
        seed=seed,
        grid=int(capacity["grid"]),
        candidate_width=int(capacity["candidate_width"]),
    )
    case = _tensor_case(case_np)
    model = JointReciprocalTriEmitterVerifier(
        dino_dim=8,
        width=int(capacity["model_width"]),
        hidden=int(capacity["model_hidden"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(capacity["weight_decay"]),
    )
    targets = [
        exact_joint_targets(
            case["candidates"][axis],
            case["valid"][axis],
            case["truth"][axis],
        )
        for axis in range(2)
    ]
    losses: list[float] = []
    components: dict[str, float] = {}
    model.train()
    for _ in range(steps):
        total = torch.zeros(())
        axis_losses = []
        for axis in range(2):
            scored = model(
                case["raw_sides"],
                case["dino_sides"],
                case["candidates"][axis],
                case["valid"][axis],
                case["auxiliary"][axis],
                case["raw_baseline"][axis],
                direction=axis,
            )
            current = joint_assignment_loss(scored, targets[axis], case["valid"][axis])
            total = total + current.total / 2
            axis_losses.append(current)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(total.detach()))
        components = {
            "row_cross_entropy": float(
                torch.stack([value.row_cross_entropy for value in axis_losses])
                .mean()
                .detach()
            ),
            "column_cross_entropy": float(
                torch.stack([value.column_cross_entropy for value in axis_losses])
                .mean()
                .detach()
            ),
            "confidence_bce": float(
                torch.stack([value.confidence_bce for value in axis_losses])
                .mean()
                .detach()
            ),
            "delta_regularization": float(
                torch.stack([value.delta_regularization for value in axis_losses])
                .mean()
                .detach()
            ),
        }

    model.eval()
    axis_reports: list[dict[str, Any]] = []
    positive_confidences: list[float] = []
    collision_confidences: list[float] = []
    with torch.inference_mode():
        for axis in range(2):
            scored = model(
                case["raw_sides"],
                case["dino_sides"],
                case["candidates"][axis],
                case["valid"][axis],
                case["auxiliary"][axis],
                case["raw_baseline"][axis],
                direction=axis,
            )
            classification = _classification_metrics(
                scored, targets[axis], case["valid"][axis]
            )
            positive = scored.joint_confidence[targets[axis].edge_truth]
            collision = scored.joint_confidence[case["hard_collision"][axis]]
            positive_confidences.extend(positive.tolist())
            collision_confidences.extend(collision.tolist())
            head = fixed_fraction_reciprocal_head(
                scored,
                case_np["candidates"][axis],
                case_np["valid"][axis],
            )
            selected_truth = targets[axis].edge_truth.numpy()[head.selected]
            head_precision = float(selected_truth.mean()) if len(selected_truth) else 0.0
            invariants = _invariance_errors(scored, seed=seed + axis + 1)
            axis_reports.append(
                {
                    "axis": ("right", "down")[axis],
                    **classification,
                    "positive_confidence_minimum": float(positive.min()),
                    "hard_collision_confidence_maximum": float(collision.max()),
                    "fixed_head_fraction": RECIPROCAL_HEAD_FRACTION,
                    "fixed_head_requested_count": head.requested_count,
                    "fixed_head_selected_count": int(head.selected.sum()),
                    "fixed_head_precision": head_precision,
                    "reciprocal_winner_count": int(head.reciprocal.sum()),
                    **invariants,
                }
            )

    thresholds = capacity["gate"]
    ratio = losses[-1] / losses[0]
    classification_passed = all(
        axis[metric] >= float(thresholds["r1_minimum"])
        for axis in axis_reports
        for metric in (
            "row_all_r1",
            "column_all_r1",
            "row_exact_edge_r1",
            "column_exact_edge_r1",
            "row_none_accuracy",
            "column_none_accuracy",
        )
    )
    confidence_passed = min(positive_confidences) > max(collision_confidences)
    head_passed = all(
        axis["fixed_head_selected_count"] == axis["fixed_head_requested_count"]
        and axis["fixed_head_precision"] >= float(thresholds["head_precision_minimum"])
        for axis in axis_reports
    )
    invariant_passed = all(
        axis[key] <= float(thresholds["invariance_max_abs_error"])
        for axis in axis_reports
        for key in (
            "transpose_confidence_max_abs_error",
            "relabel_confidence_max_abs_error",
        )
    )
    loss_passed = ratio <= float(thresholds["loss_ratio_maximum"])
    passed = bool(
        classification_passed
        and confidence_passed
        and head_passed
        and invariant_passed
        and loss_passed
    )

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "capacity_model.pt"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "contract": joint_verifier_contract(model),
            "capacity_only_not_reusable_for_real_fit": True,
            "config_sha256": config_sha,
        },
        checkpoint,
    )
    report = {
        "schema": "aiijc-joint-reciprocal-tri-emitter-capacity-v1",
        "status": "pass" if passed else "fail-stop",
        "scope": "synthetic-4x4-collision-capacity-only",
        "config_sha256": config_sha,
        "case": {
            "grid": "4x4",
            "tile_count": 16,
            "candidate_width": int(capacity["candidate_width"]),
            "axes": 2,
            "hard_many_to_one_collision_targets": case_np[
                "collision_targets"
            ].tolist(),
            "organizer_sources": 0,
        },
        "training": {
            "seed": seed,
            "steps": steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "final_to_initial_loss_ratio": ratio,
            "final_components_mean_over_axes": components,
        },
        "axes": axis_reports,
        "confidence_separation": {
            "positive_minimum": min(positive_confidences),
            "hard_collision_maximum": max(collision_confidences),
            "minimum_gap": min(positive_confidences) - max(collision_confidences),
        },
        "gate": {
            "classification_passed": classification_passed,
            "confidence_separation_passed": confidence_passed,
            "fixed_5_percent_head_passed": head_passed,
            "transpose_and_relabel_invariants_passed": invariant_passed,
            "loss_ratio_passed": loss_passed,
            "passed": passed,
            "thresholds": thresholds,
        },
        "contract": joint_verifier_contract(model),
        "artifacts": {
            "config": _record(DEFAULT_CONFIG),
            "module": _record(
                PROJECT_ROOT
                / "src/aiijc_puzzle/joint_reciprocal_tri_emitter_verifier.py"
            ),
            "runner": _record(Path(__file__)),
            "test": _record(
                PROJECT_ROOT / "tests/test_joint_reciprocal_tri_emitter_verifier.py"
            ),
            "checkpoint": _record(checkpoint),
        },
        "real_fit_or_dev_or_terminal_panel_opened": False,
        "competition_test_accessed": False,
        "decoder_run": False,
        "weco_logged": False,
    }
    _write_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha = _load_signed_config(args.config)
    report = run_capacity(config, config_sha, args.output_dir.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
