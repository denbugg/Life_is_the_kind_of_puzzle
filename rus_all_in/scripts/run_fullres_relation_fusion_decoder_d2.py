#!/usr/bin/env python3
"""Run the preregistered source40 exact decoder pilot for fullres fusion.

The first phase is target-blind after synthetic dirty-board construction: both
strict layouts are frozen to disk before the exact permutation is consulted.
The second phase recreates each deterministic case and scores those immutable
layouts.  Restored pixels are only a matcher view; both arms place the original
upright shuffled tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_relation_confidence import (
    relation_forest_score_substitution,
)
from aiijc_puzzle.fullres_relation_decoder import build_fusion_forest_inputs
from aiijc_puzzle.fullres_relation_fusion import (
    FullresRelationFusion,
    fusion_feature_names,
)
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)

try:
    from scripts.run_component_relation_reranker import (
        CleanTileCache,
        _filename_digest,
        prepare_case,
    )
    from scripts.run_fullres_relation_fusion import (
        PROJECT_ROOT,
        _load_config,
        _load_models,
        prepare_fusion_board,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import (
        CleanTileCache,
        _filename_digest,
        prepare_case,
    )
    from run_fullres_relation_fusion import (
        PROJECT_ROOT,
        _load_config,
        _load_models,
        prepare_fusion_board,
    )

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/fullres_relation_fusion_decoder_d2_preregistered_v1.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/fullres-relation-fusion/decoder-d2-source40-draw1"
)
EXPECTED_CONFIG_SHA256 = (
    "46ae0388fee837efbb1bb47f665a442f1badab62bfc6279b6728c1f5a840114a"
)
GRID = 24
TILE_COUNT = GRID * GRID
EXPECTED_SOURCES = 40
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20320910


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--wait-for-access-confirmation",
        action="store_true",
        help="Pause after publishing PID/config/roster and before target access.",
    )
    return parser.parse_args()


def filename_set_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def load_d2_config(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != EXPECTED_CONFIG_SHA256:
        raise ValueError(f"D2 preregistration changed: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "fullres-relation-fusion-decoder-d2-v1":
        raise ValueError("unexpected D2 experiment")
    if not payload.get("registered_before_source40_target_access"):
        raise ValueError("D2 timing contract is absent")
    protocol = payload["protocol"]
    if not (
        protocol["baseline_and_treatment_layouts_frozen_before_exact_scoring"]
        and protocol["strict_original_upright_tile_permutation_required"]
        and not protocol["organizer_holdout_access"]
        and not protocol["competition_test_access"]
    ):
        raise ValueError("D2 protocol is not fail-closed")
    return payload, digest


def validate_frozen_inputs(config: Mapping[str, Any]) -> None:
    for key, raw_path in config["frozen_inputs"].items():
        if not key.endswith("_checkpoint") and not key.endswith("_report") and not key.endswith(
            "_preregistration"
        ):
            continue
        expected = config["frozen_inputs"].get(f"{key}_sha256")
        if not isinstance(expected, str):
            raise ValueError(f"missing frozen hash for {key}")
        path = PROJECT_ROOT / str(raw_path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"frozen input changed for {key}: {observed}")


def selected_records(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("manifest protocol digest is invalid")
    selection = config["selection"]
    names = tuple(str(name) for name in selection["source_filenames"])
    if len(names) != EXPECTED_SOURCES or len(set(names)) != EXPECTED_SOURCES:
        raise ValueError("D2 source roster must contain 40 unique sources")
    if _filename_digest(names) != selection["source_order_digest"]:
        raise ValueError("D2 source order digest mismatch")
    if filename_set_digest(names) != selection["source_set_digest"]:
        raise ValueError("D2 source set digest mismatch")
    split = str(config["protocol"]["manifest_split"])
    rows = manifest.get("splits", {}).get(split)
    if not isinstance(rows, list):
        raise ValueError(f"manifest split is absent: {split}")
    lookup = {str(row["filename"]): row for row in rows}
    if set(names) - set(lookup):
        raise ValueError("D2 roster contains sources absent from the declared split")
    return tuple(lookup[name] for name in names), names


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.asarray(value, dtype=np.int32)
    if layout.shape != (TILE_COUNT,) or not np.array_equal(
        np.sort(layout), np.arange(TILE_COUNT)
    ):
        raise ValueError("decoder output is not a strict original-tile permutation")
    return np.ascontiguousarray(layout)


def _load_fusion(config: Mapping[str, Any], *, device: torch.device) -> FullresRelationFusion:
    frozen = config["frozen_inputs"]
    checkpoint_path = PROJECT_ROOT / str(frozen["fusion_checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    contract = payload["contract"]
    if contract.get("architecture") != "fullres-component-relation-fusion-v1":
        raise ValueError("unexpected fusion checkpoint architecture")
    feature_names = fusion_feature_names()
    if list(feature_names) != contract["feature_names"]:
        raise ValueError("fusion inference feature contract changed")
    model = FullresRelationFusion(
        len(feature_names),
        hidden_dimension=int(contract["hidden_dimension"]),
        residual_limit=float(contract["residual_limit"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        contract["parameters"]
    ):
        raise ValueError("fusion parameter count changed")
    return model.to(device).eval().requires_grad_(False)


def _arm_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, float]:
    fields = (
        "correct_tile_count",
        "direct_placement",
        "translation_aligned_count",
        "translation_aligned_placement",
        "adjacency_correct",
        "adjacency",
    )
    return {
        field: float(np.mean([float(row[arm]["metrics"][field]) for row in rows]))
        for field in fields
    }


def paired_bootstrap(values: Sequence[float]) -> dict[str, Any]:
    difference = np.asarray(values, dtype=np.float64)
    if difference.shape != (EXPECTED_SOURCES,) or not np.isfinite(difference).all():
        raise ValueError("D2 bootstrap requires one finite delta for every source")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[np.ndarray] = []
    remaining = BOOTSTRAP_SAMPLES
    while remaining:
        size = min(remaining, 4096)
        indices = generator.integers(0, len(difference), size=(size, len(difference)))
        samples.append(difference[indices].mean(axis=1))
        remaining -= size
    distribution = np.concatenate(samples)
    return {
        "source_count": len(difference),
        "mean_delta_per_board": float(difference.mean()),
        "source_cluster_bootstrap_ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "wins_ties_losses": [
            int(np.sum(difference > 0)),
            int(np.sum(difference == 0)),
            int(np.sum(difference < 0)),
        ],
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def evaluate_d2_gate(
    *,
    mean_exact_delta: float,
    mean_adjacency_delta: float,
    strict_permutation_count: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    exact = contract["exact_branch"]
    adjacency = contract["adjacency_branch"]
    strict = strict_permutation_count == EXPECTED_SOURCES
    loss_ok = mean_adjacency_delta >= -float(exact["maximum_adjacency_loss_fraction"])
    exact_branch = mean_exact_delta >= float(
        exact["minimum_mean_exact_tiles_gain_per_board"]
    )
    adjacency_branch = (
        mean_exact_delta
        >= float(adjacency["minimum_mean_exact_tiles_gain_per_board"])
        and mean_adjacency_delta
        >= float(adjacency["minimum_mean_adjacency_gain_fraction"])
    )
    passed = strict and loss_ok and (exact_branch or adjacency_branch)
    return {
        "status": "pass-preserve-candidate" if passed else "fail-stop",
        "pass": passed,
        "decoder_authorized": False,
        "promotion_authorized": False,
        "competition_test_authorized": False,
        "checks": {
            "strict_original_permutations": {
                "observed": strict_permutation_count,
                "required": EXPECTED_SOURCES,
                "pass": strict,
            },
            "adjacency_loss_bound": {
                "observed_delta": mean_adjacency_delta,
                "minimum_delta": -float(exact["maximum_adjacency_loss_fraction"]),
                "pass": loss_ok,
            },
            "exact_branch": {
                "observed_exact_tiles_per_board_delta": mean_exact_delta,
                "minimum": float(exact["minimum_mean_exact_tiles_gain_per_board"]),
                "pass": exact_branch,
            },
            "adjacency_branch": {
                "observed_exact_tiles_per_board_delta": mean_exact_delta,
                "minimum_exact": float(
                    adjacency["minimum_mean_exact_tiles_gain_per_board"]
                ),
                "observed_adjacency_delta": mean_adjacency_delta,
                "minimum_adjacency_delta": float(
                    adjacency["minimum_mean_adjacency_gain_fraction"]
                ),
                "pass": adjacency_branch,
            },
        },
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.inference_batch <= 0 or args.log_every <= 0:
        raise ValueError("inference-batch and log-every must be positive")
    config, config_sha256 = load_d2_config(args.config)
    validate_frozen_inputs(config)
    seed = int(config["protocol"]["synthetic_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == "mps":
        if not args.allow_nondeterministic_mps:
            raise ValueError("MPS requires explicit --allow-nondeterministic-mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
        device = torch.device("mps")
    else:
        if args.allow_nondeterministic_mps:
            raise ValueError("allow-nondeterministic-mps requires MPS")
        torch.use_deterministic_algorithms(True)
        device = torch.device("cpu")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, names = selected_records(config, manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_predictions.json"
    report_path = output_dir / "report.json"
    start_path = output_dir / "start.json"
    if frozen_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite D2 predictions/report")
    start = {
        "event": "preregistered-before-target-access",
        "pid": os.getpid(),
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "source_order_digest": config["selection"]["source_order_digest"],
        "source_set_digest": config["selection"]["source_set_digest"],
        "source_count": len(names),
        "device": str(device),
        "target_access_started": False,
    }
    _write_json(start_path, start)
    print(json.dumps(start), flush=True)
    if args.wait_for_access_confirmation:
        input("Access paused; press Enter after PID/config/roster publication.\n")
    start["target_access_started"] = True
    _write_json(start_path, start)
    print(json.dumps({"event": "target-access-open", "pid": os.getpid()}), flush=True)

    d1_config, d1_config_sha256 = _load_config(
        PROJECT_ROOT / str(config["frozen_inputs"]["fusion_preregistration"])
    )
    socket, relation, denoiser, model_metadata = _load_models(d1_config, device=device)
    fusion = _load_fusion(config, device=device)
    candidate_contract = config["candidate_and_decoder"]
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=int(
            candidate_contract["component_edge_budget_per_axis"]
        ),
        max_swap_steps=int(candidate_contract["decoder_max_swap_steps"]),
    )
    cyclic_config = CyclicTranslationConfig(
        border_weight=float(candidate_contract["cyclic_border_weight"])
    )
    cache = CleanTileCache(args.targets)
    frozen_rows: list[dict[str, Any]] = []
    freeze_started = perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(records):
            case_started = perf_counter()
            case = prepare_case(
                cache,
                record,
                draw_index=int(config["protocol"]["draw_index"]),
                seed=seed,
            )
            board = prepare_fusion_board(
                case,
                socket=socket,
                relation=relation,
                denoiser=denoiser,
                device=device,
                inference_batch=args.inference_batch,
                raw_topk=int(
                    candidate_contract["raw_proposal_topk_per_exposed_member"]
                ),
                raw_cap=int(candidate_contract["raw_candidate_cap_per_query"]),
                union_cap=int(candidate_contract["union_candidate_cap_per_query"]),
                attach_exact_labels=False,
            )
            if board.union_labels or board.oracle_relations or board.profiles:
                raise RuntimeError("exact labels leaked into target-blind freeze phase")
            feature_tensor = torch.from_numpy(board.features).to(device)
            relation_tensor = torch.from_numpy(board.frozen_relation_scores).to(device)
            fusion_output = fusion(feature_tensor, relation_tensor)
            raw_keys = frozenset(
                candidate.relation_key for candidate in board.raw_candidates
            )
            forest_inputs = build_fusion_forest_inputs(
                board.union_candidates,
                fusion_output.scores,
                fusion_output.confidence_logits,
                raw_candidate_keys=raw_keys,
                board_id=case.case_id,
            )
            raw_output = board.raw_socket_output
            baseline_decode = decode_socket_assignments(
                raw_output.right_log_assignment,
                raw_output.down_log_assignment,
                grid=GRID,
                config=decoder_config,
            )
            substituted, forest_diagnostics = relation_forest_score_substitution(
                raw_output.right_log_assignment,
                raw_output.down_log_assignment,
                forest_inputs.rows,
                forest_inputs.probabilities,
                board.union_candidates,
                grid=GRID,
                top_cap=int(candidate_contract["forest_top_query_cap"]),
                component_edge_budget_per_axis=int(
                    candidate_contract["component_edge_budget_per_axis"]
                ),
            )
            treatment_decode = decode_socket_assignments(
                substituted["right"],
                substituted["down"],
                grid=GRID,
                config=decoder_config,
            )
            baseline_cyclic = select_global_cyclic_translation(
                baseline_decode.layout,
                raw_output.right_log_assignment,
                raw_output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            )
            treatment_cyclic = select_global_cyclic_translation(
                treatment_decode.layout,
                raw_output.right_log_assignment,
                raw_output.down_log_assignment,
                grid=GRID,
                config=cyclic_config,
            )
            baseline_layout = _strict_layout(baseline_cyclic.layout)
            treatment_layout = _strict_layout(treatment_cyclic.layout)
            frozen_rows.append(
                {
                    "source_filename": case.source_filename,
                    "case_id": case.case_id,
                    "baseline_tile_at_position": baseline_layout.tolist(),
                    "treatment_tile_at_position": treatment_layout.tolist(),
                    "baseline_layout_sha256": hashlib.sha256(
                        baseline_layout.astype("<i4").tobytes()
                    ).hexdigest(),
                    "treatment_layout_sha256": hashlib.sha256(
                        treatment_layout.astype("<i4").tobytes()
                    ).hexdigest(),
                    "raw_candidate_count": len(board.raw_candidates),
                    "union_candidate_count": len(board.union_candidates),
                    "restored_only_candidate_count": len(board.union_candidates)
                    - len(board.raw_candidates),
                    "fusion_forest_inputs": forest_inputs.diagnostics,
                    "forest": forest_diagnostics,
                    "baseline_decoder": baseline_decode.report(),
                    "treatment_decoder": treatment_decode.report(),
                    "baseline_cyclic": baseline_cyclic.report(),
                    "treatment_cyclic": treatment_cyclic.report(),
                    "runtime_seconds": {
                        **board.runtime_seconds,
                        "case_total": perf_counter() - case_started,
                    },
                }
            )
            if (index + 1) % args.log_every == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "event": "freeze-layout",
                            "index": index + 1,
                            "sources": len(records),
                            "case_id": case.case_id,
                            "new_contacts": forest_diagnostics[
                                "new_contacts_absent_from_original_hard_matching"
                            ],
                            "surviving_contacts": forest_diagnostics[
                                "accepted_contacts_surviving_new_hard_matching"
                            ],
                            "elapsed_seconds": perf_counter() - freeze_started,
                        }
                    ),
                    flush=True,
                )

    frozen_payload = {
        "experiment": config["experiment"],
        "phase": "both-layouts-frozen-before-exact-scoring",
        "config_sha256": config_sha256,
        "source_order_digest": config["selection"]["source_order_digest"],
        "source_count": len(frozen_rows),
        "strict_permutation_count": sum(
            int(
                np.array_equal(
                    np.sort(np.asarray(row["baseline_tile_at_position"])),
                    np.arange(TILE_COUNT),
                )
                and np.array_equal(
                    np.sort(np.asarray(row["treatment_tile_at_position"])),
                    np.arange(TILE_COUNT),
                )
            )
            for row in frozen_rows
        ),
        "rows": frozen_rows,
    }
    _write_json(frozen_path, frozen_payload)
    frozen_sha256 = sha256_file(frozen_path)
    print(
        json.dumps(
            {
                "event": "layouts-frozen",
                "path": str(frozen_path),
                "sha256": frozen_sha256,
                "sources": len(frozen_rows),
            }
        ),
        flush=True,
    )

    # Only now attach the exact known synthetic reference and score immutable layouts.
    scoring_cache = CleanTileCache(args.targets)
    scored_rows: list[dict[str, Any]] = []
    for record, frozen_row in zip(records, frozen_rows, strict=True):
        case = prepare_case(
            scoring_cache,
            record,
            draw_index=int(config["protocol"]["draw_index"]),
            seed=seed,
        )
        if case.case_id != frozen_row["case_id"]:
            raise RuntimeError("scoring phase recreated a different synthetic case")
        reference = np.argsort(case.input_tile_to_position).astype(np.int32)
        baseline_layout = _strict_layout(frozen_row["baseline_tile_at_position"])
        treatment_layout = _strict_layout(frozen_row["treatment_tile_at_position"])
        baseline_metrics = evaluate_layout(
            baseline_layout, reference, reference_is_exact=True
        ).as_dict()
        treatment_metrics = evaluate_layout(
            treatment_layout, reference, reference_is_exact=True
        ).as_dict()
        scored_rows.append(
            {
                "source_filename": case.source_filename,
                "case_id": case.case_id,
                "baseline": {"metrics": baseline_metrics},
                "treatment": {"metrics": treatment_metrics},
                "exact_delta_tiles": int(
                    treatment_metrics["correct_tile_count"]
                    - baseline_metrics["correct_tile_count"]
                ),
                "adjacency_delta": float(
                    treatment_metrics["adjacency"] - baseline_metrics["adjacency"]
                ),
                "translation_aligned_delta_tiles": int(
                    treatment_metrics["translation_aligned_count"]
                    - baseline_metrics["translation_aligned_count"]
                ),
            }
        )

    baseline_summary = _arm_summary(scored_rows, "baseline")
    treatment_summary = _arm_summary(scored_rows, "treatment")
    exact_delta = [float(row["exact_delta_tiles"]) for row in scored_rows]
    adjacency_delta = [float(row["adjacency_delta"]) for row in scored_rows]
    mean_exact_delta = float(np.mean(exact_delta))
    mean_adjacency_delta = float(np.mean(adjacency_delta))
    gate = evaluate_d2_gate(
        mean_exact_delta=mean_exact_delta,
        mean_adjacency_delta=mean_adjacency_delta,
        strict_permutation_count=int(frozen_payload["strict_permutation_count"]),
        contract=config["d2_discovery_gate"],
    )
    forest_fields = (
        "selected_queries",
        "accepted_relations",
        "accepted_contacts",
        "new_contacts_absent_from_original_hard_matching",
        "changed_matrix_contacts",
        "accepted_contacts_surviving_new_hard_matching",
    )
    report = {
        "experiment": config["experiment"],
        "preregistration": {
            "path": str(args.config.resolve()),
            "sha256": config_sha256,
            "source_order_digest": config["selection"]["source_order_digest"],
            "source_set_digest": config["selection"]["source_set_digest"],
        },
        "protocol": config["protocol"],
        "device": {
            "value": str(device),
            "nondeterministic_mps_explicitly_allowed": bool(
                args.allow_nondeterministic_mps
            ),
        },
        "frozen_inputs": {
            **config["frozen_inputs"],
            "d1_preregistration_observed_sha256": d1_config_sha256,
            "loaded_models": model_metadata,
        },
        "selection": config["selection"],
        "candidate_and_decoder": config["candidate_and_decoder"],
        "layout_freeze": {
            "path": str(frozen_path),
            "sha256": frozen_sha256,
            "created_before_exact_scoring": True,
            "strict_permutation_count": frozen_payload["strict_permutation_count"],
        },
        "summary": {
            "source_count": len(scored_rows),
            "baseline": baseline_summary,
            "treatment": treatment_summary,
            "delta": {
                key: treatment_summary[key] - baseline_summary[key]
                for key in baseline_summary
            },
            "exact_delta_bootstrap": paired_bootstrap(exact_delta),
            "exact_wins_ties_losses": [
                int(np.sum(np.asarray(exact_delta) > 0)),
                int(np.sum(np.asarray(exact_delta) == 0)),
                int(np.sum(np.asarray(exact_delta) < 0)),
            ],
            "forest_mean_per_board": {
                field: float(np.mean([row["forest"][field] for row in frozen_rows]))
                for field in forest_fields
            },
            "restored_only_query_winners_mean_per_board": float(
                np.mean(
                    [
                        row["fusion_forest_inputs"]["restored_only_query_winners"]
                        for row in frozen_rows
                    ]
                )
            ),
            "runtime_seconds": perf_counter() - freeze_started,
        },
        "gate": gate,
        "promotion": {
            "evaluated": False,
            "authorized": False,
            "required_next_panel": config["future_promotion_gate"]["required_panel"],
        },
        "competition_test_opened": False,
        "rows": scored_rows,
    }
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "mean_exact_delta_tiles_per_board": mean_exact_delta,
                "mean_adjacency_delta": mean_adjacency_delta,
                "gate": gate["status"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
