#!/usr/bin/env python3
"""Fresh exact-synthetic test of whole-board SocketMatcher translation anchoring.

The script freezes a base decoder144 layout and one preregistered candidate
(``border_weight=5``) before opening exact inverse-shuffle labels.  Candidate
sources are manifest-train targets disjoint from checkpoint lineage and from
all earlier exact-synthetic reports found under ``outputs/socket-matcher``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from evaluate_socket_matcher_synthetic_exact import (
    DEFAULT_MANIFEST,
    DEFAULT_TARGETS,
    GRID,
    build_exact_panel,
    choose_device,
    load_model,
)

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    load_checkpoint_with_lineage,
    names_digest,
    select_source_disjoint_train_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "socket-matcher"
    / "v2-d64-train1024-s1600-r400-dev32"
    / "socket_matcher.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "socket-matcher"
    / "global-cyclic-translation-v1-fresh-source24-draw2"
)
SELECTION_NAMESPACE = "aiijc-socket-global-cyclic-translation-v1"
SELECTION_SEED = 20260902
SOURCE_LIMIT = 24
DRAWS_PER_SOURCE = 2
DECODER_EDGE_BUDGET = 144
DECODER_SWAP_STEPS = 24
CANDIDATE_BORDER_WEIGHT = 5.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260902


@dataclass(frozen=True)
class FrozenCase:
    """Target-blind layouts and diagnostics for one synthetic shuffle."""

    case_id: str
    source_filename: str
    draw_index: int
    corrupted_tiles_sha256: str
    layouts: dict[str, np.ndarray]
    base_decoder_report: dict[str, Any]
    candidate_report: dict[str, Any]
    inference_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    return parser.parse_args()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _prior_exact_sources(report_root: Path) -> tuple[str, ...]:
    """Collect filenames exposed by earlier exact-synthetic Socket reports."""

    names: set[str] = set()
    for path in sorted(report_root.glob("**/report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        protocol = report.get("protocol")
        evaluation = report.get("evaluation")
        exact = bool(
            isinstance(protocol, dict)
            and protocol.get("permutation_labels")
            == "exact inverse of an independent deterministic shuffle"
        ) or bool(
            isinstance(evaluation, dict) and evaluation.get("reference_is_exact") is True
        )
        if not exact:
            continue
        selection = report.get("selection")
        if not isinstance(selection, dict):
            continue
        filenames = selection.get("source_filenames")
        if not isinstance(filenames, list) or not all(
            isinstance(name, str) and name for name in filenames
        ):
            continue
        names.update(filenames)
    return tuple(sorted(names))


@torch.no_grad()
def freeze_case(
    model: torch.nn.Module,
    synthetic_input: SyntheticSocketInput,
    *,
    device: torch.device,
) -> FrozenCase:
    """Build both layouts from corrupted tiles only; no label is accepted."""

    started = perf_counter()
    dirty = synthetic_input.tiles
    tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    output = model(tensor.unsqueeze(0).to(device), grid=GRID)
    right = output.right_log_assignment[0].float().cpu().numpy()
    down = output.down_log_assignment[0].float().cpu().numpy()
    base = decode_socket_assignments(
        right,
        down,
        grid=GRID,
        config=SocketDecoderConfig(
            component_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            swap_edge_budget_per_axis=DECODER_EDGE_BUDGET,
            max_swap_steps=DECODER_SWAP_STEPS,
        ),
    )
    candidate = select_global_cyclic_translation(
        base.layout,
        right,
        down,
        grid=GRID,
        config=CyclicTranslationConfig(border_weight=CANDIDATE_BORDER_WEIGHT),
    )
    layouts = {
        "socket_ot_decoder144": np.ascontiguousarray(base.layout),
        "decoder144_global_cyclic_border5": np.ascontiguousarray(candidate.layout),
    }
    return FrozenCase(
        case_id=synthetic_input.case_id,
        source_filename=synthetic_input.source_filename,
        draw_index=synthetic_input.draw_index,
        corrupted_tiles_sha256=_array_sha256(dirty),
        layouts=layouts,
        base_decoder_report=base.report(),
        candidate_report=candidate.report(),
        inference_seconds=perf_counter() - started,
    )


def freeze_artifact(
    predictions: list[FrozenCase],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        for name, layout in prediction.layouts.items():
            arrays[f"{prefix}__layout__{name}"] = layout
        cases.append(
            {
                "array_prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "draw_index": prediction.draw_index,
                "corrupted_tiles_sha256": prediction.corrupted_tiles_sha256,
                "layout_variants": list(prediction.layouts),
                "base_decoder_report": prediction.base_decoder_report,
                "candidate_report": prediction.candidate_report,
                "inference_seconds": prediction.inference_seconds,
            }
        )
    artifact = output_dir / "frozen_predictions.npz"
    np.savez_compressed(artifact, **arrays)
    metadata = output_dir / "frozen_predictions.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "aiijc-socket-global-cyclic-translation-frozen-v1",
                "contains_exact_references": False,
                "contains_clean_pixels": False,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact, metadata


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "correct_tile_count",
        "direct_placement",
        "correct_row_count",
        "row_accuracy",
        "correct_column_count",
        "column_accuracy",
        "translation_aligned_count",
        "translation_aligned_placement",
        "adjacency_correct",
        "adjacency",
    )
    result = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    result["correct_tile_count_total"] = sum(int(row["correct_tile_count"]) for row in rows)
    result["adjacency_correct_total"] = sum(int(row["adjacency_correct"]) for row in rows)
    return result


def _clustered_bootstrap_interval(
    values_by_source: dict[str, list[float]],
) -> tuple[float, float]:
    names = sorted(values_by_source)
    source_means = np.asarray(
        [np.mean(values_by_source[name]) for name in names], dtype=np.float64
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(source_means), size=(BOOTSTRAP_SAMPLES, len(source_means))
    )
    means = source_means[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def evaluate_frozen(
    predictions: list[FrozenCase],
    references: dict[str, ExactSyntheticReference],
) -> dict[str, Any]:
    if {prediction.case_id for prediction in predictions} != set(references):
        raise ValueError("frozen predictions and exact references disagree")
    boards: list[dict[str, Any]] = []
    variants = tuple(predictions[0].layouts)
    for prediction in predictions:
        reference = references[prediction.case_id].tile_at_position
        metrics = {
            name: evaluate_layout(layout, reference, reference_is_exact=True).as_dict()
            for name, layout in prediction.layouts.items()
        }
        boards.append(
            {
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "draw_index": prediction.draw_index,
                "variants": metrics,
            }
        )
    aggregate = {
        name: _aggregate([board["variants"][name] for board in boards])
        for name in variants
    }
    base_name, candidate_name = variants
    delta_keys = (
        "correct_tile_count",
        "correct_row_count",
        "correct_column_count",
        "translation_aligned_count",
        "adjacency",
    )
    deltas: dict[str, Any] = {}
    for key in delta_keys:
        per_source: dict[str, list[float]] = {}
        per_case: list[float] = []
        for board in boards:
            delta = float(board["variants"][candidate_name][key]) - float(
                board["variants"][base_name][key]
            )
            per_case.append(delta)
            per_source.setdefault(board["source_filename"], []).append(delta)
        deltas[key] = {
            "mean": float(np.mean(per_case)),
            "source_clustered_bootstrap_95_ci": list(
                _clustered_bootstrap_interval(per_source)
            ),
        }
    exact_case_delta = np.asarray(
        [
            board["variants"][candidate_name]["correct_tile_count"]
            - board["variants"][base_name]["correct_tile_count"]
            for board in boards
        ]
    )
    return {
        "reference": "exact inverse synthetic permutation",
        "reference_is_exact": True,
        "boards": boards,
        "aggregate": aggregate,
        "candidate_delta_vs_decoder144": deltas,
        "exact_case_wins_ties_losses": {
            "wins": int(np.count_nonzero(exact_case_delta > 0)),
            "ties": int(np.count_nonzero(exact_case_delta == 0)),
            "losses": int(np.count_nonzero(exact_case_delta < 0)),
        },
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = args.checkpoint.resolve()
    payload, lineage = load_checkpoint_with_lineage(checkpoint, project_root=PROJECT_ROOT)
    model, contract = load_model(payload, device=device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prior_exact = _prior_exact_sources(PROJECT_ROOT / "outputs" / "socket-matcher")
    excluded = tuple(sorted(set(lineage.filenames) | set(prior_exact)))
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=excluded,
        limit=SOURCE_LIMIT,
        seed=SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    selected_names = [str(record["filename"]) for record in records]
    if set(selected_names) & set(excluded):
        raise RuntimeError("fresh source selection overlaps prior exposure")
    inputs, references, sources = build_exact_panel(
        records,
        targets_dir=args.targets.resolve(),
        draws_per_source=DRAWS_PER_SOURCE,
        seed=SELECTION_SEED,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions: list[FrozenCase] = []
    for index, synthetic_input in enumerate(inputs, start=1):
        prediction = freeze_case(model, synthetic_input, device=device)
        predictions.append(prediction)
        print(
            f"froze {index}/{len(inputs)} {synthetic_input.case_id} "
            f"in {prediction.inference_seconds:.2f}s",
            flush=True,
        )
    artifact, metadata = freeze_artifact(predictions, output_dir=output_dir)
    artifact_hash = sha256_file(artifact)
    metadata_hash = sha256_file(metadata)
    print(f"frozen artifact committed: {artifact}", flush=True)

    evaluation = evaluate_frozen(predictions, references)
    changed = sum(
        bool(prediction.candidate_report["diagnostics"]["changed"])
        for prediction in predictions
    )
    report = {
        "experiment": "socket-global-cyclic-translation-v1",
        "status": "fresh-source-disjoint-exact-synthetic-confirmation",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "architecture_contract": contract,
            "lineage_filenames": list(lineage.filenames),
            "lineage_digest": names_digest(lineage.filenames, sort_names=True),
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_split": "train",
            "clean_train_targets_only": True,
            "checkpoint_and_prior_exact_source_disjoint": True,
            "prior_exact_source_count_excluded": len(prior_exact),
            "target_hashes_verified_before_use": True,
            "dirty_only_predictions_frozen_before_reference_scoring": True,
            "frozen_artifact_contains_exact_references": False,
            "candidate_accepts_reference_at_inference": False,
            "candidate_replaces_or_warps_tiles": False,
            "candidate_is_strict_one_to_one_permutation": True,
        },
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "seed": SELECTION_SEED,
            "source_limit": SOURCE_LIMIT,
            "draws_per_source": DRAWS_PER_SOURCE,
            "case_count": len(inputs),
            "source_filenames": selected_names,
            "source_digest": names_digest(selected_names),
            "sources": sources,
        },
        "fixed_candidate": {
            "base": "socket_ot_decoder144",
            "candidate": "decoder144_global_cyclic_border5",
            "decoder_edge_budget_per_axis": DECODER_EDGE_BUDGET,
            "decoder_swap_steps": DECODER_SWAP_STEPS,
            "cyclic_candidates": GRID * GRID,
            "border_weight": CANDIDATE_BORDER_WEIGHT,
            "development_evidence": (
                "weight 5 chosen once on prior exact source16-draw2 panel; "
                "fresh roster was not swept"
            ),
        },
        "frozen_predictions": {
            "arrays_path": str(artifact),
            "arrays_sha256": artifact_hash,
            "metadata_path": str(metadata),
            "metadata_sha256": metadata_hash,
        },
        "candidate_diagnostics": {
            "changed_cases": changed,
            "unchanged_cases": len(predictions) - changed,
            "mean_dirty_objective_gain": float(
                np.mean(
                    [
                        prediction.candidate_report["diagnostics"]["objective_gain"]
                        for prediction in predictions
                    ]
                )
            ),
        },
        "runtime_seconds": {
            "dirty_only_inference_total": sum(
                prediction.inference_seconds for prediction in predictions
            )
        },
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "changed_cases": changed,
                "aggregate": evaluation["aggregate"],
                "delta": evaluation["candidate_delta_vs_decoder144"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
