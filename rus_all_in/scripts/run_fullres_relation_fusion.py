#!/usr/bin/env python3
"""Train/evaluate the preregistered fullres/component-relation fusion head.

This runner has no layout decoder and no competition-test loader.  It uses
clean organizer calibration targets only to synthesize known shuffled boards,
freezes the raw plus full-resolution-restored candidate union before attaching
truth, and evaluates one source-disjoint local relation panel.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.component_relation_reranker import (
    DIRECTION_TO_INDEX,
    ComponentRelationCandidate,
    ComponentRelationReranker,
    ComponentTruthProfile,
    RelationCandidateLabel,
    build_component_relation_candidates,
    component_relation_targets,
    extract_frozen_socket_context,
)
from aiijc_puzzle.component_shift_head import component_descriptors_from_decoder
from aiijc_puzzle.fullres_boundary_denoiser import (
    FullResolutionBoundaryDenoiser,
    FullResolutionDenoiserConfig,
    restore_matcher_view,
)
from aiijc_puzzle.fullres_relation_fusion import (
    FullresRelationFusion,
    build_fusion_features,
    fusion_feature_names,
    fusion_training_loss,
    preserve_raw_union_candidates,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.restored_border_ranker import restored_descriptor_scores
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
)

try:
    from scripts.run_component_relation_reranker import (
        COMPONENT_EDGE_BUDGET,
        GRID,
        CleanTileCache,
        PreparedCase,
        _filename_digest,
        _tile_tensor,
        prepare_case,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_component_relation_reranker import (
        COMPONENT_EDGE_BUDGET,
        GRID,
        CleanTileCache,
        PreparedCase,
        _filename_digest,
        _tile_tensor,
        prepare_case,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/fullres_relation_fusion_preregistered_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/fullres-relation-fusion/v1-fit32-s300-eval16"
HIGH_CONFIDENCE_CAPS = (16, 32, 64, 144)


@dataclass(frozen=True)
class PreparedFusionBoard:
    case_id: str
    source_filename: str
    raw_socket_output: Any
    components: tuple[Any, ...]
    raw_candidates: tuple[ComponentRelationCandidate, ...]
    union_candidates: tuple[ComponentRelationCandidate, ...]
    union_labels: tuple[RelationCandidateLabel, ...]
    oracle_relations: frozenset[tuple[int, str, int, int, int]]
    profiles: tuple[ComponentTruthProfile, ...]
    features: np.ndarray
    frozen_relation_scores: np.ndarray
    runtime_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "fullres-relation-fusion-v1":
        raise ValueError("unexpected fusion preregistration")
    if not payload.get("registered_before_fit_or_eval_target_access"):
        raise ValueError("preregistration timing contract is absent")
    return payload, sha256_file(path)


def _validate_hash(path: Path, expected: str, *, name: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch: {observed} != {expected}")


def _records_from_preregistration(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("manifest protocol digest is invalid")
    split_name = str(selection["manifest_split"])
    records = manifest.get("splits", {}).get(split_name)
    if not isinstance(records, list):
        raise ValueError(f"manifest split {split_name!r} is missing")
    by_name = {str(record["filename"]): record for record in records}
    fit_names = list(selection["fit_filenames"])
    eval_names = list(selection["eval_filenames"])
    if len(fit_names) != len(set(fit_names)) or len(eval_names) != len(set(eval_names)):
        raise ValueError("preregistered rosters contain duplicates")
    if set(fit_names) & set(eval_names):
        raise ValueError("fit/eval source-disjointness failed")
    if _filename_digest(fit_names) != selection["fit_digest"]:
        raise ValueError("fit roster digest mismatch")
    if _filename_digest(eval_names) != selection["eval_digest"]:
        raise ValueError("eval roster digest mismatch")
    try:
        fit = tuple(by_name[name] for name in fit_names)
        local = tuple(by_name[name] for name in eval_names)
    except KeyError as error:
        raise ValueError(f"preregistered source is absent from {split_name}: {error}") from error
    return fit, local


def _validate_component_panel_disjointness(
    prereg: Mapping[str, Any],
    fit_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen = prereg["frozen_inputs"]
    panel_path = PROJECT_ROOT / str(frozen["component_panel_preregistration"])
    _validate_hash(
        panel_path,
        str(frozen["component_panel_preregistration_sha256"]),
        name="component panel preregistration",
    )
    panel = json.loads(panel_path.read_text(encoding="utf-8"))["reserved64_split"]
    current = {
        str(record["filename"]) for record in tuple(fit_records) + tuple(eval_records)
    }
    overlaps = {
        name: sorted(current & set(panel[name]["filenames"]))
        for name in ("confirm24", "decoder40")
    }
    if any(overlaps.values()):
        raise ValueError(f"fusion roster overlaps component-agent panels: {overlaps}")
    return {
        "component_confirm24_overlap": 0,
        "component_decoder40_overlap": 0,
        "component_confirm24_digest": panel["confirm24"]["digest"],
        "component_decoder40_digest": panel["decoder40"]["digest"],
    }


def _load_models(
    prereg: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[
    LoadedSocketCheckpoint,
    ComponentRelationReranker,
    FullResolutionBoundaryDenoiser,
    dict[str, Any],
]:
    frozen = prereg["frozen_inputs"]
    socket_path = PROJECT_ROOT / str(frozen["socket_checkpoint"])
    relation_path = PROJECT_ROOT / str(frozen["component_relation_checkpoint"])
    denoiser_path = PROJECT_ROOT / str(frozen["fullres_denoiser_checkpoint"])
    _validate_hash(socket_path, str(frozen["socket_checkpoint_sha256"]), name="Socket")
    _validate_hash(
        relation_path,
        str(frozen["component_relation_checkpoint_sha256"]),
        name="component relation",
    )
    _validate_hash(
        denoiser_path,
        str(frozen["fullres_denoiser_checkpoint_sha256"]),
        name="fullres denoiser",
    )
    socket = load_socket_checkpoint(socket_path, device=device)

    relation_payload = torch.load(relation_path, map_location="cpu", weights_only=True)
    relation_contract = relation_payload["contract"]
    if relation_contract.get("architecture") != "d64-component-relation-reranker-v1":
        raise ValueError("unexpected component relation checkpoint architecture")
    relation = ComponentRelationReranker(
        int(relation_contract["tile_dimension"]),
        grid=int(relation_contract["grid"]),
        hidden_dimension=int(relation_contract["hidden_dimension"]),
    )
    relation.load_state_dict(relation_payload["state_dict"], strict=True)
    relation.to(device).eval().requires_grad_(False)

    denoiser_payload = torch.load(denoiser_path, map_location="cpu", weights_only=True)
    denoiser_contract = denoiser_payload["contract"]
    if denoiser_contract.get("architecture") != "fullres-20x20-naf-boundary-denoiser-v1":
        raise ValueError("unexpected fullres checkpoint architecture")
    denoiser = FullResolutionBoundaryDenoiser(
        FullResolutionDenoiserConfig(**denoiser_contract["model_config"])
    )
    denoiser.load_state_dict(denoiser_payload["state_dict"], strict=True)
    denoiser.to(device).eval().requires_grad_(False)
    metadata = {
        "socket": {"path": str(socket.path), "sha256": socket.sha256},
        "component_relation": {
            "path": str(relation_path.resolve()),
            "sha256": sha256_file(relation_path),
            "contract": relation_contract,
        },
        "fullres_denoiser": {
            "path": str(denoiser_path.resolve()),
            "sha256": sha256_file(denoiser_path),
            "contract": denoiser_contract,
        },
    }
    return socket, relation, denoiser, metadata


@torch.inference_mode()
def prepare_fusion_board(
    case: PreparedCase,
    *,
    socket: LoadedSocketCheckpoint,
    relation: ComponentRelationReranker,
    denoiser: FullResolutionBoundaryDenoiser,
    device: torch.device,
    inference_batch: int,
    raw_topk: int,
    raw_cap: int,
    union_cap: int,
    attach_exact_labels: bool = True,
) -> PreparedFusionBoard:
    runtime: dict[str, float] = {}
    started = perf_counter()
    raw_tensor = _tile_tensor(case.dirty_tiles, device=device)
    raw_tokens, raw_output = extract_frozen_socket_context(
        socket.model,
        raw_tensor,
        grid=GRID,
    )
    runtime["raw_socket_d64"] = perf_counter() - started

    started = perf_counter()
    restored_tiles = restore_matcher_view(
        denoiser,
        case.dirty_tiles,
        device=device,
        batch_size=inference_batch,
    )
    runtime["fullres_restore"] = perf_counter() - started
    started = perf_counter()
    restored_tensor = _tile_tensor(restored_tiles, device=device)
    restored_tokens, restored_output = extract_frozen_socket_context(
        socket.model,
        restored_tensor,
        grid=GRID,
    )
    runtime["restored_socket_d64"] = perf_counter() - started
    started = perf_counter()
    descriptor = {
        "right": restored_descriptor_scores(restored_tiles, direction=0),
        "down": restored_descriptor_scores(restored_tiles, direction=1),
    }
    runtime["restored_descriptor"] = perf_counter() - started

    started = perf_counter()
    component_build = rebuild_decoder_components(
        raw_output.right_log_assignment,
        raw_output.down_log_assignment,
        grid=GRID,
        edge_budget_per_axis=COMPONENT_EDGE_BUDGET,
    )
    components = component_descriptors_from_decoder(component_build, grid=GRID)
    raw_candidates = build_component_relation_candidates(
        components,
        raw_output,
        grid=GRID,
        proposal_topk=raw_topk,
        max_candidates_per_query=raw_cap,
    )
    expanded = build_component_relation_candidates(
        components,
        raw_output,
        grid=GRID,
        proposal_topk=raw_topk,
        max_candidates_per_query=union_cap,
        additional_proposal_scores={
            "right": restored_output.right_log_assignment[0, : GRID * GRID, : GRID * GRID],
            "down": restored_output.down_log_assignment[0, : GRID * GRID, : GRID * GRID],
        },
    )
    union_candidates = preserve_raw_union_candidates(
        raw_candidates,
        expanded,
        max_candidates_per_query=union_cap,
    )
    runtime["components_and_target_blind_union"] = perf_counter() - started

    started = perf_counter()
    relation_scores = relation(raw_tokens[0], components, union_candidates)
    raw_keys = frozenset(candidate.relation_key for candidate in raw_candidates)
    features = build_fusion_features(
        components,
        union_candidates,
        raw_candidate_keys=raw_keys,
        frozen_relation_scores=relation_scores,
        raw_tile_tokens=raw_tokens[0],
        restored_tile_tokens=restored_tokens[0],
        restored_socket_output=restored_output,
        restored_descriptor_scores=descriptor,
        grid=GRID,
    )
    runtime["frozen_relation_and_target_free_features"] = perf_counter() - started

    union_labels: tuple[RelationCandidateLabel, ...] = ()
    oracle_relations: frozenset[tuple[int, str, int, int, int]] = frozenset()
    profiles: tuple[ComponentTruthProfile, ...] = ()
    if attach_exact_labels:
        # Exact synthetic truth is attached only after candidates/features are frozen.
        started = perf_counter()
        union_labels, oracle_relations, profiles = component_relation_targets(
            union_candidates,
            components,
            case.input_tile_to_position,
            grid=GRID,
        )
        runtime["post_freeze_truth_attachment"] = perf_counter() - started
    return PreparedFusionBoard(
        case_id=case.case_id,
        source_filename=case.source_filename,
        raw_socket_output=raw_output,
        components=components,
        raw_candidates=raw_candidates,
        union_candidates=union_candidates,
        union_labels=union_labels,
        oracle_relations=oracle_relations,
        profiles=profiles,
        features=features,
        frozen_relation_scores=np.ascontiguousarray(
            relation_scores.float().cpu().numpy(),
            dtype=np.float32,
        ),
        runtime_seconds=runtime,
    )


def _method_observations(
    board: PreparedFusionBoard,
    *,
    raw_scores: np.ndarray,
    relation_scores: np.ndarray,
    fusion_scores: np.ndarray,
    fusion_confidence: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(board.union_candidates):
        grouped[candidate.query_key].append(index)
    oracle_queries = {(relation[0], relation[1]) for relation in board.oracle_relations}
    all_queries = sorted(
        set(grouped) | oracle_queries,
        key=lambda item: (item[0], DIRECTION_TO_INDEX[item[1]]),
    )
    methods = {
        "raw_component": raw_scores,
        "frozen_relation": relation_scores,
        "fusion": fusion_scores,
    }
    records: list[dict[str, Any]] = []
    for query in all_queries:
        indices = grouped.get(query, [])
        positives = {index for index in indices if board.union_labels[index].positive}
        row: dict[str, Any] = {
            "board_id": board.case_id,
            "query": query,
            "has_oracle_relation": query in oracle_queries,
            "has_candidates": bool(indices),
            "has_supplied_positive": bool(positives),
            "candidate_count": len(indices),
        }
        for method, scores in methods.items():
            if not indices:
                row[f"{method}_positive_rank"] = None
                row[f"{method}_top1_correct"] = False
                row[f"{method}_confidence"] = None
                continue
            ordered = sorted(indices, key=lambda index: (-float(scores[index]), index))
            rank = next(
                (
                    position
                    for position, index in enumerate(ordered, start=1)
                    if index in positives
                ),
                None,
            )
            if method == "fusion":
                confidence = float(fusion_confidence[ordered[0]])
            else:
                confidence = (
                    0.0
                    if len(ordered) == 1
                    else float(scores[ordered[0]] - scores[ordered[1]])
                )
            row[f"{method}_positive_rank"] = rank
            row[f"{method}_top1_correct"] = ordered[0] in positives
            row[f"{method}_confidence"] = confidence
        records.append(row)
    return records


def _aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    raw_oracle_queries: int,
    raw_supplied_queries: int,
) -> dict[str, Any]:
    oracle = [record for record in records if bool(record["has_oracle_relation"])]
    supplied = [record for record in records if bool(record["has_supplied_positive"])]
    by_board: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record["has_candidates"]):
            by_board[str(record["board_id"])].append(record)
    methods: dict[str, Any] = {}
    for method in ("raw_component", "frozen_relation", "fusion"):
        ranks = [record[f"{method}_positive_rank"] for record in supplied]
        high_confidence: dict[str, Any] = {}
        for cap in HIGH_CONFIDENCE_CAPS:
            correct: list[int] = []
            selected: list[int] = []
            for board_records in by_board.values():
                ordered = sorted(
                    board_records,
                    key=lambda record: (
                        -float(record[f"{method}_confidence"]),
                        int(record["query"][0]),
                        DIRECTION_TO_INDEX[str(record["query"][1])],
                    ),
                )[:cap]
                correct.append(
                    sum(bool(record[f"{method}_top1_correct"]) for record in ordered)
                )
                selected.append(len(ordered))
            total_selected = sum(selected)
            high_confidence[f"top{cap}"] = {
                "correct_per_board": float(np.mean(correct)) if correct else None,
                "precision": sum(correct) / total_selected if total_selected else None,
                "selected_per_board": float(np.mean(selected)) if selected else None,
            }
        methods[method] = {
            "eligible_queries": len(supplied),
            "r1": float(np.mean([rank == 1 for rank in ranks])) if ranks else None,
            "r5": (
                float(np.mean([rank is not None and int(rank) <= 5 for rank in ranks]))
                if ranks
                else None
            ),
            "high_confidence": high_confidence,
        }
    return {
        "board_count": len(by_board),
        "oracle_query_count": len(oracle),
        "union_supplied_positive_query_count": len(supplied),
        "union_candidate_supply_coverage": len(supplied) / len(oracle),
        "raw_oracle_query_count": raw_oracle_queries,
        "raw_supplied_positive_query_count": raw_supplied_queries,
        "raw_candidate_supply_coverage": raw_supplied_queries / raw_oracle_queries,
        "methods": methods,
    }


def _gate(metrics: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    gate = prereg["discovery_gate"]
    methods = metrics["methods"]
    relation = methods["frozen_relation"]
    fusion = methods["fusion"]
    supply_gain = float(metrics["union_candidate_supply_coverage"]) - float(
        metrics["raw_candidate_supply_coverage"]
    )
    r1_gain = float(fusion["r1"]) - float(relation["r1"])
    r5_gain = float(fusion["r5"]) - float(relation["r5"])
    relation_top32 = relation["high_confidence"]["top32"]
    fusion_top32 = fusion["high_confidence"]["top32"]
    correct_gain = float(fusion_top32["correct_per_board"]) - float(
        relation_top32["correct_per_board"]
    )
    precision_gain = float(fusion_top32["precision"]) - float(
        relation_top32["precision"]
    )
    supply_pass = supply_gain >= float(
        gate["minimum_union_oracle_query_coverage_gain_over_raw_roster"]
    )
    ranking_pass = r1_gain >= float(
        gate["ranking_branch"]["minimum_pair_translation_r1_gain_over_frozen_relation"]
    ) and r5_gain >= float(
        gate["ranking_branch"]["minimum_pair_translation_r5_gain_over_frozen_relation"]
    )
    confidence_pass = correct_gain >= float(
        gate["confidence_branch_either"][
            "minimum_top32_correct_attachments_per_board_gain_over_frozen_relation"
        ]
    ) or precision_gain >= float(
        gate["confidence_branch_either"][
            "minimum_top32_precision_gain_over_frozen_relation"
        ]
    )
    passed = supply_pass and (ranking_pass or confidence_pass)
    return {
        "discovery_pass": passed,
        "decoder_authorized": False,
        "promotion_authorized": False,
        "supply": {"gain": supply_gain, "pass": supply_pass},
        "ranking": {"r1_gain": r1_gain, "r5_gain": r5_gain, "pass": ranking_pass},
        "confidence": {
            "top32_correct_per_board_gain": correct_gain,
            "top32_precision_gain": precision_gain,
            "pass": confidence_pass,
        },
    }


def _mean_runtime(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys
    }


def _raw_supply_counts(board: PreparedFusionBoard) -> tuple[int, int]:
    oracle_queries = {(relation[0], relation[1]) for relation in board.oracle_relations}
    raw_keys = {candidate.relation_key for candidate in board.raw_candidates}
    union_positive_keys = {
        candidate.relation_key
        for candidate, label in zip(
            board.union_candidates,
            board.union_labels,
            strict=True,
        )
        if label.positive
    }
    supplied_queries = {
        (key[0], key[1]) for key in raw_keys & union_positive_keys
    }
    return len(oracle_queries), len(supplied_queries)


def main() -> None:
    args = parse_args()
    if args.inference_batch <= 0 or args.log_every <= 0:
        raise ValueError("inference-batch and log-every must be positive")
    prereg, prereg_sha256 = _load_config(args.config)
    training = prereg["training"]
    candidate_contract = prereg["candidate_contract"]
    random.seed(int(training["synthetic_seed"]))
    np.random.seed(int(training["synthetic_seed"]))
    torch.manual_seed(int(training["synthetic_seed"]))
    if args.device == "mps" and args.allow_nondeterministic_mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        torch.use_deterministic_algorithms(False)
        device = torch.device("mps")
    else:
        if args.allow_nondeterministic_mps:
            raise ValueError("allow-nondeterministic-mps requires --device mps")
        device = choose_deterministic_device(args.device)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fit_records, eval_records = _records_from_preregistration(
        manifest,
        prereg["selection"],
    )
    panel_audit = _validate_component_panel_disjointness(
        prereg,
        fit_records,
        eval_records,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fullres_relation_fusion.pt"
    report_path = output_dir / "report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing fusion artifact")

    socket, relation, denoiser, frozen_metadata = _load_models(prereg, device=device)
    feature_names = fusion_feature_names()
    model = FullresRelationFusion(
        len(feature_names),
        hidden_dimension=int(training["hidden_dimension"]),
        residual_limit=float(training["residual_limit"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = int(training["steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        steps,
        eta_min=float(training["learning_rate"]) * 0.08,
    )
    cache = CleanTileCache(args.targets)
    generator = np.random.default_rng(int(training["synthetic_seed"]) + 1)
    raw_topk = int(candidate_contract["raw_proposal_topk_per_exposed_member"])
    raw_cap = int(candidate_contract["raw_cap_per_component_direction_query"])
    union_cap = int(candidate_contract["union_cap_per_component_direction_query"])

    training_history: list[dict[str, Any]] = []
    training_runtimes: list[dict[str, float]] = []
    recent_losses: list[float] = []
    started = perf_counter()
    model.train()
    for step in range(steps):
        record = fit_records[int(generator.integers(len(fit_records)))]
        case = prepare_case(
            cache,
            record,
            draw_index=step,
            seed=int(training["synthetic_seed"]),
        )
        board = prepare_fusion_board(
            case,
            socket=socket,
            relation=relation,
            denoiser=denoiser,
            device=device,
            inference_batch=args.inference_batch,
            raw_topk=raw_topk,
            raw_cap=raw_cap,
            union_cap=union_cap,
        )
        features = torch.from_numpy(board.features).to(device)
        relation_scores = torch.from_numpy(board.frozen_relation_scores).to(device)
        output = model(features, relation_scores)
        loss, diagnostics = fusion_training_loss(
            output,
            board.union_candidates,
            board.union_labels,
            confidence_weight=float(training["confidence_loss_weight"]),
            residual_weight=float(training["residual_loss_weight"]),
            frozen_relation_scores=relation_scores,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training["gradient_clip"]),
            )
        )
        optimizer.step()
        scheduler.step()
        training_runtimes.append(board.runtime_seconds)
        recent_losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == steps:
            row = {
                "step": step + 1,
                "mean_loss": float(np.mean(recent_losses)),
                "listwise": diagnostics["listwise"],
                "confidence_bce": diagnostics["confidence_bce"],
                "gradient_norm": gradient_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "candidate_count": len(board.union_candidates),
                "raw_candidate_count": len(board.raw_candidates),
                "supervised_queries": int(diagnostics["supervised_queries"]),
                "elapsed_seconds": perf_counter() - started,
            }
            training_history.append(row)
            recent_losses.clear()
            print(json.dumps({"event": "train", **row}), flush=True)
    training_seconds = perf_counter() - started

    # This is the only access to the preregistered eval target files.
    print(
        json.dumps(
            {
                "event": "eval-open",
                "preregistration_sha256": prereg_sha256,
                "eval_digest": prereg["selection"]["eval_digest"],
                "eval_sources": len(eval_records),
            }
        ),
        flush=True,
    )
    model.eval()
    observations: list[dict[str, Any]] = []
    eval_runtimes: list[dict[str, float]] = []
    eval_cases: list[dict[str, Any]] = []
    raw_oracle_queries = 0
    raw_supplied_queries = 0
    evaluation = prereg["evaluation"]
    with torch.inference_mode():
        for index, record in enumerate(eval_records):
            case = prepare_case(
                cache,
                record,
                draw_index=int(evaluation["draw_index"]),
                seed=int(evaluation["synthetic_seed"]),
            )
            board = prepare_fusion_board(
                case,
                socket=socket,
                relation=relation,
                denoiser=denoiser,
                device=device,
                inference_batch=args.inference_batch,
                raw_topk=raw_topk,
                raw_cap=raw_cap,
                union_cap=union_cap,
            )
            features = torch.from_numpy(board.features).to(device)
            relation_scores_tensor = torch.from_numpy(
                board.frozen_relation_scores
            ).to(device)
            output = model(features, relation_scores_tensor)
            fusion_scores = output.scores.float().cpu().numpy()
            fusion_confidence = output.confidence_logits.float().cpu().numpy()
            raw_scores = np.asarray(
                [candidate.baseline_score for candidate in board.union_candidates],
                dtype=np.float32,
            )
            observations.extend(
                _method_observations(
                    board,
                    raw_scores=raw_scores,
                    relation_scores=board.frozen_relation_scores,
                    fusion_scores=fusion_scores,
                    fusion_confidence=fusion_confidence,
                )
            )
            raw_oracle, raw_supplied = _raw_supply_counts(board)
            raw_oracle_queries += raw_oracle
            raw_supplied_queries += raw_supplied
            eval_runtimes.append(board.runtime_seconds)
            eval_cases.append(
                {
                    "case_id": board.case_id,
                    "source_filename": board.source_filename,
                    "component_count": len(board.components),
                    "raw_candidate_count": len(board.raw_candidates),
                    "union_candidate_count": len(board.union_candidates),
                    "restored_only_candidate_count": len(board.union_candidates)
                    - len(board.raw_candidates),
                    "runtime_seconds": board.runtime_seconds,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "done": index + 1,
                        "total": len(eval_records),
                        "case_id": board.case_id,
                    }
                ),
                flush=True,
            )
    metrics = _aggregate_metrics(
        observations,
        raw_oracle_queries=raw_oracle_queries,
        raw_supplied_queries=raw_supplied_queries,
    )
    gate = _gate(metrics, prereg)

    selection = {
        **prereg["selection"],
        **panel_audit,
        "fit_eval_overlap": 0,
        "fit_target_files_opened": True,
        "eval_target_files_opened": True,
        "organizer_holdout_target_files_opened": False,
        "competition_test_opened": False,
    }
    contract = {
        "architecture": "fullres-component-relation-fusion-v1",
        "feature_names": list(feature_names),
        "feature_dimension": len(feature_names),
        "parameters": parameter_count,
        "hidden_dimension": int(training["hidden_dimension"]),
        "residual_limit": float(training["residual_limit"]),
        "candidate_contract": candidate_contract,
        "frozen_relation_step_zero_exact": True,
        "confidence_head": (
            "train-only candidate correctness BCE; query confidence is the predicted "
            "top-candidate logit"
        ),
        "fixed_score_fusion": False,
        "restored_pixels_matcher_only": True,
        "original_tiles_only": True,
        "global_decoder_present": False,
    }
    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "contract": contract,
        "selection": selection,
        "frozen_inputs": frozen_metadata,
        "preregistration": {
            "path": str(args.config.resolve()),
            "sha256": prereg_sha256,
        },
        "discovery_gate": gate,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    report = {
        "experiment": contract["architecture"],
        "status": (
            "discovery-pass-preserve-no-decoder"
            if gate["discovery_pass"]
            else "discovery-fail-stop-no-decoder"
        ),
        "quality_panel_opened": False,
        "decoder_authorized": False,
        "competition_test_opened": False,
        "organizer_holdout_opened": False,
        "contract": contract,
        "preregistration": {
            "path": str(args.config.resolve()),
            "sha256": prereg_sha256,
            "gate": prereg["discovery_gate"],
        },
        "frozen_inputs": frozen_metadata,
        "selection": selection,
        "training": {
            "configuration": training,
            "history": training_history,
            "device": str(device),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "local_metrics": metrics,
        "gate": gate,
        "runtime_seconds": {
            "training_total": training_seconds,
            "mean_training_board": _mean_runtime(training_runtimes),
            "mean_eval_board": _mean_runtime(eval_runtimes),
        },
        "eval_cases": eval_cases,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
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
                "status": report["status"],
                "gate": gate,
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
