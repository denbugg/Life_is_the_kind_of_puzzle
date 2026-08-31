#!/usr/bin/env python3
"""Frozen source-disjoint confirmation for an absolute coordinate checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_absolute_coordinate_sorter import (
    DEFAULT_MANIFEST,
    DEFAULT_TARGETS,
    SELECTION_NAMESPACE,
    choose_device,
    evaluate_exact,
    load_socket_backbone,
)

from aiijc_puzzle.absolute_coordinate_sorter import AbsoluteCoordinateSorter
from aiijc_puzzle.protocol import (
    collect_declared_source_filenames,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--eval-draws", type=int, default=2)
    parser.add_argument("--component-unary-weight", type=float, default=0.10)
    parser.add_argument(
        "--historical-per-tile-zscore-comparator",
        action="store_true",
        help="include the old per-tile z-score unary as a labelled diagnostic only",
    )
    parser.add_argument(
        "--cyclic-border5",
        action="store_true",
        help="compose both decoder arms with the independently confirmed cyclic tail",
    )
    parser.add_argument(
        "--axis-unary-diagnostics",
        action="store_true",
        help="include row-only and column-only coordinate unary arms",
    )
    parser.add_argument(
        "--axis-unary-weight",
        dest="axis_unary_weights",
        action="append",
        type=float,
        default=[],
        help="bounded row-only/column-only development weight; may be repeated",
    )
    parser.add_argument(
        "--reuse-panel-report",
        type=Path,
        help="require exact replay of an already-opened checkpoint/seed/source panel",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="additional prior target-opened report whose source/eval panels must be excluded",
    )
    return parser.parse_args()


def _digest_names(records: tuple[Any, ...]) -> str:
    value = "\n".join(str(record["filename"]) for record in records)
    return hashlib.sha256(value.encode()).hexdigest()


def select_confirmation_records(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    exclude_reports: list[Path],
    *,
    limit: int,
) -> tuple[tuple[Any, ...], set[str], set[str], list[dict[str, Any]]]:
    selection = checkpoint.get("selection", {})
    exposed = selection.get(
        "lineage_exposed_filenames",
        selection.get("train_filenames", []),
    )
    if not isinstance(exposed, list) or not all(isinstance(name, str) for name in exposed):
        raise ValueError("coordinate checkpoint exposure lineage is malformed")
    forbidden = set(exposed)
    declared_prior_panels: set[str] = set()
    exclude_audit: list[dict[str, Any]] = []
    for path in exclude_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        declared = collect_declared_source_filenames(report)
        declared_prior_panels.update(declared)
        forbidden.update(declared)
        exclude_audit.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_source_count": len(declared),
            }
        )
    ranked = select_manifest_records(
        manifest,
        "train",
        limit=len(manifest["splits"]["train"]),
        namespace=SELECTION_NAMESPACE,
    )
    records = tuple(
        record for record in ranked if str(record["filename"]) not in forbidden
    )[:limit]
    if len(records) != limit:
        raise ValueError("could not form a complete unexposed confirmation panel")
    if forbidden & {str(record["filename"]) for record in records}:
        raise RuntimeError("confirmation panel overlaps prior source exposure")
    selected_names = {str(record["filename"]) for record in records}
    for audit in exclude_audit:
        audit["selected_panel_overlap_count"] = 0
    if selected_names & forbidden:
        raise RuntimeError("confirmation exclusion audit failed")
    return records, forbidden, declared_prior_panels, exclude_audit


def paired_source_cluster_bootstrap(
    boards: list[dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
    seed: int,
    samples: int = 200_000,
) -> dict[str, Any]:
    """Bootstrap paired metric deltas while keeping draws of one source together."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for board in boards:
        grouped.setdefault(str(board["source_filename"]), []).append(board)
    metrics = (
        "correct_tile_count",
        "correct_row_count",
        "correct_column_count",
        "translation_aligned_count",
        "adjacency_correct",
    )
    generator = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for metric in metrics:
        case_deltas = np.asarray(
            [
                float(board["global"][candidate][metric])
                - float(board["global"][baseline][metric])
                for board in boards
            ],
            dtype=np.float64,
        )
        source_deltas = np.asarray(
            [
                np.mean(
                    [
                        float(board["global"][candidate][metric])
                        - float(board["global"][baseline][metric])
                        for board in source_boards
                    ]
                )
                for source_boards in grouped.values()
            ],
            dtype=np.float64,
        )
        indices = generator.integers(
            0,
            len(source_deltas),
            size=(samples, len(source_deltas)),
        )
        bootstrap = source_deltas[indices].mean(axis=1)
        output[metric] = {
            "mean_delta_per_board": float(case_deltas.mean()),
            "total_delta": float(case_deltas.sum()),
            "source_cluster_bootstrap_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "case_wins": int(np.count_nonzero(case_deltas > 0)),
            "case_ties": int(np.count_nonzero(case_deltas == 0)),
            "case_losses": int(np.count_nonzero(case_deltas < 0)),
        }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "source_count": len(grouped),
        "case_count": len(boards),
        "draws_clustered_by_source": True,
        "bootstrap_samples": samples,
        "seed": seed,
        "metrics": output,
    }


def main() -> None:
    args = parse_args()
    if args.eval_limit <= 0 or args.eval_draws <= 0:
        raise ValueError("eval-limit and eval-draws must be positive")
    if not np.isfinite(args.component_unary_weight) or args.component_unary_weight < 0:
        raise ValueError("component-unary-weight must be finite and non-negative")
    if any(not np.isfinite(value) or value <= 0 for value in args.axis_unary_weights):
        raise ValueError("axis-unary-weight values must be finite and positive")
    if len(set(args.axis_unary_weights)) != len(args.axis_unary_weights):
        raise ValueError("axis-unary-weight values must be unique")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    if contract.get("architecture") != "socket-backed-absolute-coordinate-sorter-v1":
        raise ValueError("unsupported absolute coordinate checkpoint architecture")
    socket_metadata = checkpoint.get("socket_checkpoint", {})
    socket_path = Path(str(socket_metadata.get("path", "")))
    if not socket_path.is_file():
        raise FileNotFoundError(f"socket checkpoint is unavailable: {socket_path}")
    if sha256_file(socket_path) != socket_metadata.get("sha256"):
        raise ValueError("socket checkpoint digest differs from coordinate lineage")
    backbone, _ = load_socket_backbone(socket_path, device)
    model = AbsoluteCoordinateSorter(
        backbone,
        grid=int(contract["grid"]),
        head_dimension=int(contract["head_dimension"]),
        heads=int(contract["heads"]),
        set_layers=int(contract["set_layers"]),
        sinkhorn_iterations=int(contract["sinkhorn_iterations"]),
        freeze_backbone=bool(contract["frozen_socket_backbone"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, prior_forbidden, declared_prior_panels, exclude_audit = (
        select_confirmation_records(
            manifest,
            checkpoint,
            args.exclude_report,
            limit=args.eval_limit,
        )
    )
    replay_audit: dict[str, Any] | None = None
    if args.reuse_panel_report is not None:
        replay = json.loads(args.reuse_panel_report.read_text(encoding="utf-8"))
        replay_names = replay.get("selection", {}).get("eval_filenames")
        if not isinstance(replay_names, list) or not all(
            isinstance(name, str) for name in replay_names
        ):
            raise ValueError("reuse-panel report has a malformed eval source panel")
        selected_names = [str(record["filename"]) for record in records]
        if selected_names != replay_names:
            raise ValueError("selected panel is not the requested already-opened panel")
        if replay.get("selection", {}).get("eval_digest") != _digest_names(records):
            raise ValueError("reuse-panel source digest does not match selected records")
        if replay.get("selection", {}).get("draws_per_source") != args.eval_draws:
            raise ValueError("reuse-panel draw count differs from this replay")
        if replay.get("configuration", {}).get("seed") != args.seed:
            raise ValueError("reuse-panel seed differs from this replay")
        if replay.get("checkpoint", {}).get("sha256") != sha256_file(args.checkpoint):
            raise ValueError("reuse-panel checkpoint differs from this replay")
        replay_audit = {
            "path": str(args.reuse_panel_report.resolve()),
            "sha256": sha256_file(args.reuse_panel_report),
            "source_digest": _digest_names(records),
            "source_count": len(records),
            "draws_per_source": args.eval_draws,
            "seed": args.seed,
            "checkpoint_sha256": sha256_file(args.checkpoint),
        }
    evaluation, runtime_seconds = evaluate_exact(model, records, args, device)
    evaluation["paired_source_cluster_bootstrap"] = paired_source_cluster_bootstrap(
        evaluation["boards"],
        baseline="socket_ot_decoder144",
        candidate="socket_ot_decoder144_coordinate_unary_train_consistent",
        seed=args.seed + 1,
    )
    baseline_name = "socket_ot_decoder144"
    candidate_name = "socket_ot_decoder144_coordinate_unary_train_consistent"
    exact_delta = evaluation["paired_source_cluster_bootstrap"]["metrics"][
        "correct_tile_count"
    ]
    adjacency_loss_percentage_points = 100.0 * (
        evaluation["global_mean"][baseline_name]["adjacency"]
        - evaluation["global_mean"][candidate_name]["adjacency"]
    )
    strict_permutation = all(
        all(board["strict_permutation"].values()) for board in evaluation["boards"]
    )
    evaluation["predeclared_material_gate"] = {
        "primary_candidate": candidate_name,
        "matched_baseline": baseline_name,
        "requirements": {
            "component_unary_weight_exactly": 0.10,
            "mean_exact_tile_delta_per_board_at_least": 0.5,
            "source_clustered_exact_ci95_lower_strictly_above": 0.0,
            "adjacency_loss_percentage_points_at_most": 0.2,
            "strict_tile_permutation": True,
        },
        "observed": {
            "component_unary_weight": args.component_unary_weight,
            "mean_exact_tile_delta_per_board": exact_delta["mean_delta_per_board"],
            "source_clustered_exact_ci95": exact_delta[
                "source_cluster_bootstrap_ci95"
            ],
            "adjacency_loss_percentage_points": adjacency_loss_percentage_points,
            "strict_tile_permutation": strict_permutation,
        },
        "passed": bool(
            np.isclose(args.component_unary_weight, 0.10, rtol=0.0, atol=1e-12)
            and exact_delta["mean_delta_per_board"] >= 0.5
            and exact_delta["source_cluster_bootstrap_ci95"][0] > 0.0
            and adjacency_loss_percentage_points <= 0.2
            and strict_permutation
        ),
    }
    if args.axis_unary_weights:
        diagnostic_bootstraps: dict[str, Any] = {}
        variant_names = evaluation["global_mean"]
        for candidate in variant_names:
            if "_coordinate_row_unary_train_consistent_w" not in candidate and (
                "_coordinate_column_unary_train_consistent_w" not in candidate
            ):
                continue
            baseline = (
                "socket_ot_decoder144_cyclic_border5"
                if candidate.endswith("_cyclic_border5")
                else "socket_ot_decoder144"
            )
            diagnostic_bootstraps[candidate] = paired_source_cluster_bootstrap(
                evaluation["boards"],
                baseline=baseline,
                candidate=candidate,
                seed=args.seed + 10 + len(diagnostic_bootstraps),
            )
        evaluation["axis_development_bootstraps"] = diagnostic_bootstraps
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "socket-backed-absolute-coordinate-sorter-v1-frozen-confirmation",
        "status": (
            "already-opened-panel-axis-development-no-promotion"
            if args.axis_unary_weights
            else
            "fresh-confirmation-material-gate-passed-not-default"
            if evaluation["predeclared_material_gate"]["passed"]
            else "fresh-confirmation-material-gate-failed"
        ),
        "contract": contract,
        "configuration": {
            key: [str(path) for path in value]
            if key == "exclude_report"
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        }
        | {"device_resolved": str(device)},
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_train_split_only": True,
            "checkpoint_frozen": True,
            "source_disjoint_from_training_and_all_declared_prior_exposure": True,
            "prior_forbidden_source_count": len(prior_forbidden),
            "declared_prior_panel_source_count": len(declared_prior_panels),
            "checkpoint_exposure_lineage_source_count": len(
                checkpoint.get("selection", {}).get("lineage_exposed_filenames", [])
            ),
            "declared_exclude_report_audit": exclude_audit,
            "already_opened_panel_replay_audit": replay_audit,
            "train_consistent_component_unary_transform": (
                "subtract each tile-row mean, then divide every board entry by one "
                f"common positive board-global std; decoder weight "
                f"{args.component_unary_weight:.12g}"
            ),
            "historical_per_tile_zscore_is_primary": False,
            "calibration_opened": False,
            "holdout_opened": False,
            "competition_test_opened": False,
            "exact_truth_source": "known synthetic shuffle",
        },
        "selection": {
            "eval_filenames": [record["filename"] for record in records],
            "eval_digest": _digest_names(records),
            "eval_source_count": len(records),
            "draws_per_source": args.eval_draws,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "socket_path": str(socket_path.resolve()),
            "socket_sha256": sha256_file(socket_path),
        },
        "runtime_seconds": runtime_seconds,
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "classifier": evaluation["classifier_mean"],
                "global": evaluation["global_mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
