#!/usr/bin/env python3
"""Strict fresh source64xdraw2 gate for v1.1 relation ordering plus cyclic5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_component_relation_confidence import (
    CleanTileCache,
    filename_digest,
    frozen_case_forward,
    prepare_case,
)

from aiijc_puzzle.component_relation_confidence import (
    FEATURE_NAMES,
    LogisticConfidenceCalibrator,
    build_query_confidence_features,
    calibrated_component_edge_priorities,
)
from aiijc_puzzle.component_relation_reranker import ComponentRelationReranker
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/component_relation_cyclic_fresh_gate_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
TILE_COUNT = GRID * GRID
EXPECTED_SOURCES = 64
EXPECTED_DRAWS = 2
EXPECTED_CASES = EXPECTED_SOURCES * EXPECTED_DRAWS
IMAGE_NAME_PATTERN = re.compile(r"img_\d{6}\.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


def load_frozen_config(path: Path) -> tuple[dict[str, Any], str]:
    digest_path = path.with_name(f"{path.name}.sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("fresh-gate config hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("registered_before_selected_target_access")
        and value.get("registered_before_dirty_prediction_generation")
    ):
        raise ValueError("fresh gate was not preregistered before access")
    return value, observed


def _collect_declared_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.endswith("_filenames"):
                if not isinstance(child, (list, tuple)) or not all(
                    isinstance(item, str) and item for item in child
                ):
                    raise ValueError(f"{key} must be a list of non-empty filenames")
                if len(set(child)) != len(child):
                    raise ValueError(f"{key} contains duplicate filenames")
                names.update(Path(item).name for item in child)
            names.update(_collect_declared_filenames(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and not parent_key.endswith("_filenames"):
        for child in value:
            names.update(_collect_declared_filenames(child, parent_key=parent_key))
    return names


def current_lineage_audit(
    contract: Mapping[str, Any],
    socket_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck every frozen path class without reading selected target pixels."""

    audit = contract["lineage_exclusion_audit"]
    markers = tuple(str(value).lower() for value in audit["exact_panel_path_markers"])
    registry: dict[str, tuple[str, int]] = {}
    names: set[str] = set()

    def add(path: Path, found: set[str]) -> None:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        registry[str(path)] = (sha256_file(resolved), len(found))
        names.update(found)

    for path in (PROJECT_ROOT / "outputs").rglob("*.json"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(marker in str(relative).lower() for marker in markers):
            found = set(IMAGE_NAME_PATTERN.findall(path.read_text(errors="ignore")))
            if found:
                add(relative, found)
    for raw_path in audit["explicit_registry_paths"]:
        path = Path(str(raw_path))
        if path.suffix == ".pt":
            add(path, _collect_declared_filenames(socket_payload))
        else:
            text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
            add(path, set(IMAGE_NAME_PATTERN.findall(text)))
    path_digest = hashlib.sha256(
        "\n".join(
            f"{path}\0{digest}\0{count}"
            for path, (digest, count) in sorted(registry.items())
        ).encode()
    ).hexdigest()
    name_digest = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    return {
        "registry_entry_count": len(registry),
        "registry_path_hash_count_digest": path_digest,
        "excluded_filename_union_count": len(names),
        "excluded_filename_union_digest": name_digest,
        "excluded_filenames": names,
    }


def validate_selection(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    current_audit: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("validation manifest protocol digest is invalid")
    selection = config["selection"]
    split = str(selection["manifest_split"])
    splits = manifest.get("splits")
    records = splits.get(split) if isinstance(splits, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("fresh-gate manifest split is missing")
    names = [str(value) for value in selection["source_filenames"]]
    if len(names) != EXPECTED_SOURCES or len(set(names)) != len(names):
        raise ValueError("fresh source roster must contain 64 unique filenames")
    if filename_digest(names) != selection["source_order_digest"] or filename_digest(
        sorted(names)
    ) != selection["source_set_digest"]:
        raise ValueError("fresh source roster digest mismatch")
    namespace = "\0".join(str(value) for value in selection["namespace_parts"])
    ranked = select_manifest_records(
        dict(manifest),
        split,
        limit=len(records),
        namespace=namespace,
    )
    expected = [str(record["filename"]) for record in ranked[:EXPECTED_SOURCES]]
    if names != expected:
        raise ValueError("fresh source roster does not reproduce its namespace selection")
    excluded = set(current_audit["excluded_filenames"])
    overlap = set(names) & excluded
    if overlap:
        raise ValueError(f"fresh roster now overlaps prior lineage: {sorted(overlap)}")
    lookup = {str(record["filename"]): record for record in records}
    return [lookup[name] for name in names], names


def paired_source_cluster_bootstrap(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["source_filename"])].append(float(row["exact_delta_tiles"]))
    bad_draw_count = any(
        len(values) != EXPECTED_DRAWS for values in grouped.values()
    )
    if len(grouped) != EXPECTED_SOURCES or bad_draw_count:
        raise ValueError("bootstrap rows do not form source64xdraw2 clusters")
    source_delta = np.asarray(
        [float(np.mean(grouped[name])) for name in sorted(grouped)],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = samples
    while remaining:
        size = min(remaining, 4096)
        indices = generator.integers(0, len(source_delta), size=(size, len(source_delta)))
        chunks.append(source_delta[indices].mean(axis=1))
        remaining -= size
    distribution = np.concatenate(chunks)
    return {
        "source_count": len(source_delta),
        "case_count": len(case_rows),
        "mean_delta_per_board": float(source_delta.mean()),
        "source_cluster_bootstrap_ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "source_wins_ties_losses": [
            int(np.sum(source_delta > 0)),
            int(np.sum(source_delta == 0)),
            int(np.sum(source_delta < 0)),
        ],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def evaluate_gate(
    bootstrap: Mapping[str, Any],
    *,
    adjacency_delta: float,
    strict_permutations: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    mean_exact = float(bootstrap["mean_delta_per_board"])
    lower = float(bootstrap["source_cluster_bootstrap_ci95"][0])
    checks = {
        "mean_exact_tiles_gain": {
            "observed": mean_exact,
            "required": contract["minimum_mean_exact_tiles_gain_per_board"],
            "pass": mean_exact >= contract["minimum_mean_exact_tiles_gain_per_board"],
        },
        "source_cluster_ci95_lower_strictly_positive": {
            "observed": lower,
            "required_strictly_greater_than": contract[
                "minimum_source_cluster_bootstrap_ci95_lower_exact_gain_strictly_greater_than"
            ],
            "pass": lower
            > contract[
                "minimum_source_cluster_bootstrap_ci95_lower_exact_gain_strictly_greater_than"
            ],
        },
        "adjacency_delta": {
            "observed": adjacency_delta,
            "required": contract["minimum_adjacency_delta_fraction"],
            "pass": adjacency_delta >= contract["minimum_adjacency_delta_fraction"],
        },
        "strict_original_permutations": {
            "observed": strict_permutations,
            "required": contract["strict_original_permutation_count_required"],
            "pass": strict_permutations
            == contract["strict_original_permutation_count_required"],
        },
    }
    passed = all(bool(value["pass"]) for value in checks.values())
    return {
        "status": "pass-await-root-review" if passed else "fail-stop",
        "pass": passed,
        "competition_test_authorized": False,
        "checks": checks,
    }


def _mean_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, float]:
    fields = (
        "correct_tile_count",
        "direct_placement",
        "translation_aligned_count",
        "translation_aligned_placement",
        "adjacency_correct",
        "adjacency",
    )
    return {
        field: float(np.mean([float(row[arm][field]) for row in rows]))
        for field in fields
    }


def main() -> None:
    args = parse_args()
    random.seed(20320909)
    np.random.seed(20320909)
    torch.manual_seed(20320909)
    device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {
                "event": "start",
                "pid": os.getpid(),
                "device": str(device),
                "stage": "fresh-source64xdraw2-freeze-then-score",
            }
        ),
        flush=True,
    )
    config, config_hash = load_frozen_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    socket_path = PROJECT_ROOT / frozen["socket_checkpoint"]
    socket_payload = torch.load(socket_path, map_location="cpu", weights_only=True)
    if not isinstance(socket_payload, Mapping):
        raise ValueError("Socket checkpoint payload must be a mapping")
    if sha256_file(socket_path) != frozen["socket_checkpoint_sha256"]:
        raise ValueError("Socket checkpoint hash mismatch")
    lineage_audit = current_lineage_audit(config, socket_payload)
    records, source_names = validate_selection(config, manifest, lineage_audit)

    relation_path = PROJECT_ROOT / frozen["relation_checkpoint"]
    if sha256_file(relation_path) != frozen["relation_checkpoint_sha256"]:
        raise ValueError("relation checkpoint hash mismatch")
    relation_payload = torch.load(relation_path, map_location="cpu", weights_only=True)
    relation_contract = relation_payload["contract"]
    head = ComponentRelationReranker(
        int(relation_contract["tile_dimension"]),
        grid=int(relation_contract["grid"]),
        hidden_dimension=int(relation_contract["hidden_dimension"]),
    ).to(device)
    head.load_state_dict(relation_payload["state_dict"], strict=True)
    head.eval()
    socket = load_socket_checkpoint(socket_path, device=device)
    if socket.sha256 != relation_payload["socket_checkpoint"]["sha256"]:
        raise ValueError("Socket/relation checkpoint lineage mismatch")
    calibrator_path = PROJECT_ROOT / frozen["confidence_calibrator"]
    if sha256_file(calibrator_path) != frozen["confidence_calibrator_sha256"]:
        raise ValueError("confidence calibrator hash mismatch")
    calibrator = LogisticConfidenceCalibrator.from_dict(
        json.loads(calibrator_path.read_text(encoding="utf-8"))
    )
    if calibrator.parameter_count != 68 or tuple(calibrator.feature_names) != FEATURE_NAMES:
        raise ValueError("confidence calibrator architecture changed")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "frozen_predictions.json"
    report_path = output_dir / "report.json"
    if prediction_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a fresh-gate artifact")
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    cyclic_config = CyclicTranslationConfig(border_weight=5.0)
    cache = CleanTileCache(args.targets)
    predictions: list[dict[str, Any]] = []
    started = perf_counter()
    draws = [int(value) for value in config["selection"]["draw_indices"]]
    for source_index, record in enumerate(records):
        for draw_index in draws:
            case = prepare_case(
                cache,
                record,
                draw_index=draw_index,
                seed=int(config["selection"]["synthetic_seed"]),
            )
            output = frozen_case_forward(
                case,
                socket=socket,
                head=head,
                device=device,
                attach_exact_labels=False,
            )
            if output.labels or output.oracle_relations or output.profiles:
                raise RuntimeError("fresh prediction phase unexpectedly attached labels")
            rows = build_query_confidence_features(
                output.logits,
                output.candidates,
                output.components,
                board_id=case.case_id,
                grid=GRID,
            )
            probabilities = calibrator.predict_probabilities([row.values for row in rows])
            priority, priority_diagnostics = calibrated_component_edge_priorities(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                rows,
                probabilities,
                output.candidates,
                grid=GRID,
                top_cap=32,
                bonus_scale=0.25,
            )
            baseline = decode_socket_assignments(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
            )
            candidate = decode_socket_assignments(
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
                component_edge_priority=priority,
            )
            baseline_cyclic = select_global_cyclic_translation(
                baseline.layout,
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            )
            candidate_cyclic = select_global_cyclic_translation(
                candidate.layout,
                output.socket_output.right_log_assignment,
                output.socket_output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            )
            for layout in (baseline_cyclic.layout, candidate_cyclic.layout):
                if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT)):
                    raise RuntimeError("prediction is not a strict original-tile permutation")
            predictions.append(
                {
                    "source_filename": case.source_filename,
                    "source_index": source_index,
                    "draw_index": draw_index,
                    "case_id": case.case_id,
                    "baseline_tile_at_position": baseline_cyclic.layout.tolist(),
                    "candidate_tile_at_position": candidate_cyclic.layout.tolist(),
                    "baseline_decoder": baseline.report(),
                    "candidate_decoder": candidate.report(),
                    "baseline_cyclic": baseline_cyclic.report(),
                    "candidate_cyclic": candidate_cyclic.report(),
                    "candidate_priority": priority_diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "freeze-prediction",
                        "done": len(predictions),
                        "total": EXPECTED_CASES,
                        "source": source_index + 1,
                        "draw": draw_index,
                    }
                ),
                flush=True,
            )
    if len(predictions) != EXPECTED_CASES:
        raise RuntimeError("fresh prediction count changed")
    prediction_artifact = {
        "schema": "component-relation-cyclic-fresh-frozen-predictions-v1",
        "config": str(args.config.resolve()),
        "config_sha256": config_hash,
        "source_filenames": source_names,
        "source_order_digest": filename_digest(source_names),
        "predictions_frozen_before_reference_scoring": True,
        "competition_test_opened": False,
        "predictions": predictions,
    }
    prediction_path.write_text(
        json.dumps(prediction_artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_hash = sha256_file(prediction_path)
    print(
        json.dumps(
            {
                "event": "predictions-frozen",
                "path": str(prediction_path),
                "sha256": prediction_hash,
                "cases": len(predictions),
            }
        ),
        flush=True,
    )

    # Exact references are first inspected below, after both layouts are persisted.
    scored: list[dict[str, Any]] = []
    prediction_lookup = {
        (str(row["source_filename"]), int(row["draw_index"])): row
        for row in predictions
    }
    for source_index, record in enumerate(records):
        for draw_index in draws:
            case = prepare_case(
                cache,
                record,
                draw_index=draw_index,
                seed=int(config["selection"]["synthetic_seed"]),
            )
            frozen_row = prediction_lookup[(case.source_filename, draw_index)]
            if frozen_row["case_id"] != case.case_id:
                raise RuntimeError("frozen prediction case identity changed before scoring")
            reference = np.argsort(case.input_tile_to_position)
            baseline_metrics = evaluate_layout(
                frozen_row["baseline_tile_at_position"],
                reference,
                reference_is_exact=True,
            )
            candidate_metrics = evaluate_layout(
                frozen_row["candidate_tile_at_position"],
                reference,
                reference_is_exact=True,
            )
            scored.append(
                {
                    "source_filename": case.source_filename,
                    "source_index": source_index,
                    "draw_index": draw_index,
                    "case_id": case.case_id,
                    "baseline": baseline_metrics.as_dict(),
                    "candidate": candidate_metrics.as_dict(),
                    "exact_delta_tiles": (
                        candidate_metrics.correct_tile_count
                        - baseline_metrics.correct_tile_count
                    ),
                    "adjacency_delta": candidate_metrics.adjacency
                    - baseline_metrics.adjacency,
                }
            )
    gate_contract = config["fresh_gate"]
    bootstrap = paired_source_cluster_bootstrap(
        scored,
        samples=int(gate_contract["bootstrap_samples"]),
        seed=int(gate_contract["bootstrap_seed"]),
    )
    adjacency_delta = float(np.mean([float(row["adjacency_delta"]) for row in scored]))
    strict_count = sum(
        np.array_equal(
            np.sort(np.asarray(row["candidate_tile_at_position"])),
            np.arange(TILE_COUNT),
        )
        for row in predictions
    )
    gate = evaluate_gate(
        bootstrap,
        adjacency_delta=adjacency_delta,
        strict_permutations=int(strict_count),
        contract=gate_contract,
    )
    report = {
        "experiment": config["experiment"],
        "status": gate["status"],
        "competition_test_opened": False,
        "promotion_applied": False,
        "config": {"path": str(args.config.resolve()), "sha256": config_hash},
        "selection": {
            "manifest_split": config["selection"]["manifest_split"],
            "source_filenames": source_names,
            "source_order_digest": filename_digest(source_names),
            "source_count": EXPECTED_SOURCES,
            "draws_per_source": EXPECTED_DRAWS,
            "case_count": EXPECTED_CASES,
        },
        "lineage_recheck_before_target_access": {
            key: value
            for key, value in lineage_audit.items()
            if key != "excluded_filenames"
        },
        "freeze": {
            "predictions_frozen_before_reference_scoring": True,
            "prediction_artifact": str(prediction_path),
            "prediction_sha256": prediction_hash,
            "strict_original_permutation_count": int(strict_count),
        },
        "summary": {
            "baseline": _mean_metrics(scored, "baseline"),
            "candidate": _mean_metrics(scored, "candidate"),
            "delta": {
                "exact_tiles_per_board": bootstrap["mean_delta_per_board"],
                "adjacency": adjacency_delta,
                "translation_aligned_tiles_per_board": float(
                    np.mean(
                        [
                            float(row["candidate"]["translation_aligned_count"])
                            - float(row["baseline"]["translation_aligned_count"])
                            for row in scored
                        ]
                    )
                ),
            },
            "exact_source_cluster_bootstrap": bootstrap,
        },
        "gate": gate,
        "runtime_seconds": perf_counter() - started,
        "cases": scored,
        "artifacts": {
            "frozen_predictions": str(prediction_path),
            "frozen_predictions_sha256": prediction_hash,
            "report": str(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "status": gate["status"],
                "gate_pass": gate["pass"],
                "exact_delta": bootstrap["mean_delta_per_board"],
                "ci95": bootstrap["source_cluster_bootstrap_ci95"],
                "adjacency_delta": adjacency_delta,
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
