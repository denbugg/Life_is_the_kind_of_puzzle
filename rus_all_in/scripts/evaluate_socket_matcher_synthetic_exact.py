#!/usr/bin/env python3
"""Evaluate a SocketMatcher checkpoint on source-disjoint exact synthetic boards.

Only clean images named by the manifest's ``train`` split are opened.  Each
tile receives an independent reverse-engineered challenge corruption and the
board is shuffled with a known permutation.  Dirty-only local candidates and
global layouts are serialized before the exact permutations are used to score
them.
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
from PIL import Image

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.legacy_upgrade import (
    border_position_scores,
    directional_scores,
    solve_buddies,
    solve_relaxation,
    validate_layout,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_matcher import (
    BORDER_HEAD_EMBEDDING_V2,
    BORDER_HEAD_SCORE_STATS_V3,
    SocketMatcher,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    DEFAULT_SYNTHETIC_NAMESPACE,
    ExactSyntheticReference,
    SyntheticSocketInput,
    exact_local_retrieval_metrics,
    freeze_topk_candidates,
    load_checkpoint_with_lineage,
    make_exact_synthetic_case,
    names_digest,
    select_source_disjoint_train_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
GRID = 24
TILE_COUNT = GRID * GRID
LOCAL_KS = (1, 5, 16, 32)
FUSION_SOCKET_WEIGHT = 0.8
FUSION_BASELINE_WEIGHT = 0.2


@dataclass(frozen=True)
class FrozenPrediction:
    """All dirty-only evidence retained before exact-reference scoring."""

    case_id: str
    source_filename: str
    draw_index: int
    corrupted_tiles_sha256: str
    local_candidates: dict[str, dict[str, np.ndarray]]
    layouts: dict[str, np.ndarray]
    decoder_report: dict[str, Any] | None
    inference_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-limit", type=int, default=8)
    parser.add_argument("--draws-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--source-selection-report",
        type=Path,
        help=(
            "Reuse the exact ordered source_filenames from an earlier exact-synthetic "
            "report for a paired checkpoint comparison. The sources are still required "
            "to belong to manifest train and remain disjoint from this checkpoint lineage."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--decoder-edge-budget", type=int, default=144)
    parser.add_argument("--decoder-swap-steps", type=int, default=24)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_model(
    payload: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[SocketMatcher, dict[str, Any]]:
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no architecture contract")
    architecture = contract.get("architecture")
    supported = {
        "board-conditioned-partial-socket-matcher-v1",
        "board-conditioned-partial-socket-matcher-v2",
        "board-conditioned-partial-socket-matcher-v3",
    }
    if architecture not in supported:
        raise ValueError(f"unsupported SocketMatcher architecture: {architecture!r}")
    fields: dict[str, Any] = {}
    for key in (
        "dimension",
        "heads",
        "board_layers",
        "socket_layers",
        "sinkhorn_iterations",
    ):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint contract {key} must be a positive integer")
        fields[key] = value
    if fields["dimension"] % fields["heads"]:
        raise ValueError("checkpoint dimension is not divisible by heads")

    expected_border_version = (
        BORDER_HEAD_SCORE_STATS_V3
        if architecture == "board-conditioned-partial-socket-matcher-v3"
        else BORDER_HEAD_EMBEDDING_V2
    )
    declared_border_version = contract.get("border_head_version")
    if declared_border_version is None and architecture != (
        "board-conditioned-partial-socket-matcher-v3"
    ):
        # Historical v1/v2 contracts predate the explicit version field.
        declared_border_version = expected_border_version
    if declared_border_version != expected_border_version:
        raise ValueError(
            "checkpoint architecture and border_head_version disagree: "
            f"{architecture!r}, {declared_border_version!r}"
        )
    fields["border_head_version"] = expected_border_version

    model = SocketMatcher(**fields).to(device)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no state_dict")
    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing: set[str] = set()
    if architecture.endswith("-v1"):
        expected_missing = {
            f"border_heads.{side}.{field}"
            for side in ("right", "left", "bottom", "top")
            for field in ("weight", "bias")
        }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint state differs from its declared architecture: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, contract


def build_exact_panel(
    records: tuple[Any, ...],
    *,
    targets_dir: Path,
    draws_per_source: int,
    seed: int,
) -> tuple[list[SyntheticSocketInput], dict[str, ExactSyntheticReference], list[dict[str, Any]]]:
    inputs: list[SyntheticSocketInput] = []
    references: dict[str, ExactSyntheticReference] = {}
    sources: list[dict[str, Any]] = []
    for source_index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        target_path = targets_dir / filename
        observed_hash = sha256_file(target_path)
        expected_hash = record.get("target_sha256")
        if not isinstance(expected_hash, str) or observed_hash != expected_hash:
            raise ValueError(f"manifest target hash mismatch for {filename}")
        clean_tiles = split_tiles(load_rgb(target_path))
        sources.append({"filename": filename, "target_sha256": observed_hash})
        for draw_index in range(draws_per_source):
            synthetic_input, reference = make_exact_synthetic_case(
                clean_tiles,
                source_filename=filename,
                draw_index=draw_index,
                seed=seed,
            )
            if synthetic_input.case_id in references:
                raise RuntimeError("synthetic case identifier collision")
            inputs.append(synthetic_input)
            references[reference.case_id] = reference
        print(f"generated {source_index}/{len(records)} {filename}", flush=True)
    return inputs, references, sources


def records_from_selection_report(
    manifest: dict[str, Any],
    report_path: Path,
    *,
    excluded_filenames: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Recover one previously frozen source roster without weakening lineage checks."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("source selection report has no selection object")
    names = selection.get("source_filenames")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("source selection report has invalid source_filenames")
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    if not isinstance(train, list):
        raise ValueError("manifest has no train split")
    by_name = {
        str(record.get("filename")): record
        for record in train
        if isinstance(record, dict) and isinstance(record.get("filename"), str)
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"selection report names are absent from manifest train: {missing}")
    overlap = sorted(set(names) & set(excluded_filenames))
    if overlap:
        raise ValueError(f"paired source selection overlaps checkpoint lineage: {overlap}")
    return tuple(by_name[name] for name in names)


@torch.no_grad()
def freeze_prediction(
    model: SocketMatcher,
    synthetic_input: SyntheticSocketInput,
    *,
    device: torch.device,
    include_decoder: bool,
    decoder_edge_budget: int,
    decoder_swap_steps: int,
    solver_seed: int,
) -> FrozenPrediction:
    """Run every candidate using corrupted pixels only; no reference is accepted."""

    started = perf_counter()
    dirty = synthetic_input.tiles
    tensor = torch.from_numpy(dirty.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    output = model(tensor.unsqueeze(0).to(device), grid=GRID)
    socket_raw_right = output.right_raw[0].float().cpu().numpy()
    socket_raw_down = output.down_raw[0].float().cpu().numpy()
    right_assignment = output.right_log_assignment[0].float().cpu().numpy()
    down_assignment = output.down_log_assignment[0].float().cpu().numpy()
    transport_normaliser = np.log(float(TILE_COUNT + GRID))
    socket_ot_right = right_assignment[:TILE_COUNT, :TILE_COUNT] + transport_normaliser
    socket_ot_down = down_assignment[:TILE_COUNT, :TILE_COUNT] + transport_normaliser
    baseline_right, baseline_down = directional_scores(dirty, views=("bilateral",))["bilateral"]
    fused_right = (
        FUSION_SOCKET_WEIGHT * socket_ot_right
        + FUSION_BASELINE_WEIGHT * baseline_right
    )
    fused_down = (
        FUSION_SOCKET_WEIGHT * socket_ot_down
        + FUSION_BASELINE_WEIGHT * baseline_down
    )

    score_variants = {
        "bilateral": (baseline_right, baseline_down),
        "socket_raw": (socket_raw_right, socket_raw_down),
        "socket_ot": (socket_ot_right, socket_ot_down),
        "fused_ot": (fused_right, fused_down),
    }
    local_candidates = {
        name: {
            "right": freeze_topk_candidates(right, max_k=max(LOCAL_KS)),
            "down": freeze_topk_candidates(down, max_k=max(LOCAL_KS)),
        }
        for name, (right, down) in score_variants.items()
    }
    layouts = {
        "bilateral_buddies96": solve_buddies(
            baseline_right, baseline_down, max_edges=96
        ).layout,
        "socket_ot_buddies96": solve_buddies(
            socket_ot_right, socket_ot_down, max_edges=96
        ).layout,
        "fused_ot_buddies96": solve_buddies(
            fused_right, fused_down, max_edges=96
        ).layout,
        "fused_ot_relax_border": solve_relaxation(
            fused_right,
            fused_down,
            position=border_position_scores(fused_right, fused_down),
            seed=solver_seed,
        ).layout,
    }
    decoder_report: dict[str, Any] | None = None
    if include_decoder:
        decoder = decode_socket_assignments(
            right_assignment,
            down_assignment,
            grid=GRID,
            config=SocketDecoderConfig(
                component_edge_budget_per_axis=decoder_edge_budget,
                swap_edge_budget_per_axis=decoder_edge_budget,
                max_swap_steps=decoder_swap_steps,
            ),
        )
        layouts["socket_ot_decoder"] = decoder.layout
        decoder_report = decoder.report()
    layouts = {name: validate_layout(layout) for name, layout in layouts.items()}
    return FrozenPrediction(
        case_id=synthetic_input.case_id,
        source_filename=synthetic_input.source_filename,
        draw_index=synthetic_input.draw_index,
        corrupted_tiles_sha256=_array_sha256(dirty),
        local_candidates=local_candidates,
        layouts=layouts,
        decoder_report=decoder_report,
        inference_seconds=perf_counter() - started,
    )


def write_frozen_artifact(
    predictions: list[FrozenPrediction],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist label-free prediction arrays before the scoring phase starts."""

    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        prefix = f"case_{index:04d}"
        local_names: list[str] = []
        layout_names: list[str] = []
        for name, directions in sorted(prediction.local_candidates.items()):
            local_names.append(name)
            for direction, candidates in sorted(directions.items()):
                arrays[f"{prefix}__local__{name}__{direction}"] = candidates
        for name, layout in sorted(prediction.layouts.items()):
            layout_names.append(name)
            arrays[f"{prefix}__layout__{name}"] = layout
        cases.append(
            {
                "array_prefix": prefix,
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "draw_index": prediction.draw_index,
                "corrupted_tiles_sha256": prediction.corrupted_tiles_sha256,
                "local_variants": local_names,
                "layout_variants": layout_names,
                "decoder_report": prediction.decoder_report,
                "inference_seconds": prediction.inference_seconds,
            }
        )
    artifact_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(artifact_path, **arrays)
    metadata_path = output_dir / "frozen_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-socket-exact-synthetic-frozen-predictions-v1",
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
    return artifact_path, metadata_path


def _aggregate_local(boards: list[dict[str, Any]]) -> dict[str, Any]:
    variants = boards[0]["local"].keys()
    result: dict[str, Any] = {}
    for variant in variants:
        rows = [board["local"][variant] for board in boards]
        metrics: dict[str, Any] = {}
        for side in ("right", "down", "pooled"):
            total = sum(int(row[f"{side}_total"]) for row in rows)
            metrics[f"{side}_total"] = total
            for k in LOCAL_KS:
                hits = sum(int(row[f"{side}_hits_at_{k}"]) for row in rows)
                metrics[f"{side}_hits_at_{k}"] = hits
                metrics[f"{side}_r{k}"] = hits / total
        result[variant] = metrics
    return result


def _aggregate_global(boards: list[dict[str, Any]]) -> dict[str, Any]:
    variants = boards[0]["global"].keys()
    result: dict[str, Any] = {}
    for variant in variants:
        rows = [board["global"][variant] for board in boards]
        numeric_keys = (
            "correct_tile_count",
            "direct_placement",
            "correct_row_count",
            "row_accuracy",
            "correct_column_count",
            "column_accuracy",
            "translation_aligned_count",
            "translation_aligned_placement",
            "right_adjacency_correct",
            "right_adjacency",
            "down_adjacency_correct",
            "down_adjacency",
            "adjacency_correct",
            "adjacency",
        )
        result[variant] = {
            key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys
        }
        result[variant]["correct_tile_count_total"] = sum(
            int(row["correct_tile_count"]) for row in rows
        )
        result[variant]["adjacency_correct_total"] = sum(
            int(row["adjacency_correct"]) for row in rows
        )
    return result


def evaluate_frozen_predictions(
    predictions: list[FrozenPrediction],
    references: dict[str, ExactSyntheticReference],
) -> dict[str, Any]:
    """Open the exact labels only after the frozen artifact has been written."""

    prediction_ids = {prediction.case_id for prediction in predictions}
    if prediction_ids != set(references):
        raise ValueError("frozen prediction and exact-reference case sets differ")
    boards: list[dict[str, Any]] = []
    for prediction in predictions:
        reference = references[prediction.case_id]
        local = {
            name: exact_local_retrieval_metrics(
                directions["right"],
                directions["down"],
                reference.tile_at_position,
                ks=LOCAL_KS,
            )
            for name, directions in prediction.local_candidates.items()
        }
        global_metrics = {
            name: evaluate_layout(
                layout,
                reference.tile_at_position,
                reference_is_exact=True,
            ).as_dict()
            for name, layout in prediction.layouts.items()
        }
        boards.append(
            {
                "case_id": prediction.case_id,
                "source_filename": prediction.source_filename,
                "draw_index": prediction.draw_index,
                "local": local,
                "global": global_metrics,
            }
        )
    local = _aggregate_local(boards)
    global_metrics = _aggregate_global(boards)
    baseline = global_metrics["bilateral_buddies96"]
    deltas = {
        name: {
            "direct_placement": metrics["direct_placement"] - baseline["direct_placement"],
            "adjacency": metrics["adjacency"] - baseline["adjacency"],
        }
        for name, metrics in global_metrics.items()
        if name != "bilateral_buddies96"
    }
    return {
        "reference": "exact inverse synthetic permutation",
        "reference_is_exact": True,
        "boards": boards,
        "local_aggregate": local,
        "global_aggregate": global_metrics,
        "global_deltas_vs_bilateral_buddies96": deltas,
    }


def main() -> None:
    args = parse_args()
    if args.source_limit <= 0 or args.draws_per_source <= 0:
        raise ValueError("source-limit and draws-per-source must be positive")
    if not 1 <= args.decoder_edge_budget <= TILE_COUNT - GRID:
        raise ValueError("decoder-edge-budget must be in [1, 552]")
    if args.decoder_swap_steps < 0:
        raise ValueError("decoder-swap-steps must be non-negative")

    device = choose_device(args.device)
    checkpoint_payload, lineage = load_checkpoint_with_lineage(
        args.checkpoint,
        project_root=PROJECT_ROOT,
    )
    model, contract = load_model(checkpoint_payload, device=device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.source_selection_report is None:
        records = select_source_disjoint_train_records(
            manifest,
            excluded_filenames=lineage.filenames,
            limit=args.source_limit,
            seed=args.seed,
            namespace=DEFAULT_SYNTHETIC_NAMESPACE,
        )
        selection_source = "deterministic_source_disjoint_sampler"
    else:
        records = records_from_selection_report(
            manifest,
            args.source_selection_report.resolve(),
            excluded_filenames=lineage.filenames,
        )
        if args.source_limit != len(records):
            raise ValueError(
                "source-limit must equal the source count in source-selection-report: "
                f"{args.source_limit} != {len(records)}"
            )
        selection_source = "paired_source_selection_report"
    selected_names = [str(record["filename"]) for record in records]
    if set(selected_names) & set(lineage.filenames):
        raise RuntimeError("selected source overlaps checkpoint lineage")

    generated_started = perf_counter()
    inputs, references, sources = build_exact_panel(
        records,
        targets_dir=args.targets.resolve(),
        draws_per_source=args.draws_per_source,
        seed=args.seed,
    )
    generation_seconds = perf_counter() - generated_started

    predictions: list[FrozenPrediction] = []
    for index, synthetic_input in enumerate(inputs, start=1):
        prediction = freeze_prediction(
            model,
            synthetic_input,
            device=device,
            include_decoder=not args.skip_decoder,
            decoder_edge_budget=args.decoder_edge_budget,
            decoder_swap_steps=args.decoder_swap_steps,
            solver_seed=args.seed + index,
        )
        predictions.append(prediction)
        print(
            f"froze {index}/{len(inputs)} {synthetic_input.case_id} "
            f"in {prediction.inference_seconds:.2f}s",
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path, metadata_path = write_frozen_artifact(predictions, output_dir=output_dir)
    frozen_artifact_hash = sha256_file(artifact_path)
    frozen_metadata_hash = sha256_file(metadata_path)
    print(f"frozen artifact committed: {artifact_path}", flush=True)

    scoring_started = perf_counter()
    evaluation = evaluate_frozen_predictions(predictions, references)
    scoring_seconds = perf_counter() - scoring_started
    report = {
        "experiment": "socket-matcher-source-disjoint-exact-synthetic-v1",
        "status": "exact-label-train-development-diagnostic",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint.resolve()),
            "architecture_contract": contract,
            "lineage_filenames": list(lineage.filenames),
            "lineage_digest": names_digest(lineage.filenames, sort_names=True),
            "lineage_checkpoint_paths": list(lineage.checkpoint_paths),
        },
        "protocol": {
            "manifest_digest": compute_protocol_digest(manifest),
            "manifest_split": "train",
            "clean_train_targets_only": True,
            "train_inputs_opened": False,
            "calibration_files_opened": False,
            "holdout_files_opened": False,
            "competition_test_files_opened": False,
            "checkpoint_lineage_source_disjoint": True,
            "target_hashes_verified_before_use": True,
            "corruption": {
                "implementation": "aiijc_puzzle.restoration_r6.distort_tiles",
                "scope": "independent parameters, noise, blur, and JPEG per 20x20 tile",
                "brightness_offset_255": [-30.0, 30.0],
                "contrast": [0.70, 1.30],
                "gaussian_noise_sigma_255": [40.0, 55.0],
                "blur": "separable [0.25, 0.5, 0.25] x/y",
                "jpeg_quality_inclusive": [35, 50],
                "claim": "reverse-engineered official-like corruption, not organizer source code",
            },
            "permutation_labels": "exact inverse of an independent deterministic shuffle",
            "dirty_only_predictions_frozen_before_reference_scoring": True,
            "frozen_artifact_contains_exact_references": False,
            "selection_namespace": DEFAULT_SYNTHETIC_NAMESPACE,
        },
        "selection": {
            "seed": args.seed,
            "source_limit": args.source_limit,
            "draws_per_source": args.draws_per_source,
            "case_count": len(inputs),
            "source": selection_source,
            "source_selection_report": (
                None
                if args.source_selection_report is None
                else str(args.source_selection_report.resolve())
            ),
            "source_filenames": selected_names,
            "source_digest": names_digest(selected_names),
            "sources": sources,
        },
        "fixed_candidates": {
            "local_ks": list(LOCAL_KS),
            "baseline": "bilateral directional_scores",
            "socket_local": ["raw", "partial-OT real block"],
            "fusion_weights": {
                "socket_ot": FUSION_SOCKET_WEIGHT,
                "bilateral": FUSION_BASELINE_WEIGHT,
            },
            "global": list(predictions[0].layouts),
            "decoder_edge_budget_per_axis": (
                None if args.skip_decoder else args.decoder_edge_budget
            ),
            "decoder_swap_steps": None if args.skip_decoder else args.decoder_swap_steps,
        },
        "frozen_predictions": {
            "arrays_path": str(artifact_path),
            "arrays_sha256": frozen_artifact_hash,
            "metadata_path": str(metadata_path),
            "metadata_sha256": frozen_metadata_hash,
        },
        "runtime_seconds": {
            "synthetic_generation": generation_seconds,
            "dirty_only_inference_total": sum(
                prediction.inference_seconds for prediction in predictions
            ),
            "exact_reference_scoring": scoring_seconds,
        },
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "local": evaluation["local_aggregate"],
                "global": evaluation["global_aggregate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
