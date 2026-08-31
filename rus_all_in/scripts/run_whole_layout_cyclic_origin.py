#!/usr/bin/env python3
"""Bounded train-only discovery for the whole-layout cyclic-origin CNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from run_component_relation_confidence import (
    CleanTileCache,
    filename_digest,
    prepare_case,
)

from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.component_relation_reranker import extract_frozen_socket_context
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.socket_decoder import SocketDecoderConfig, decode_socket_assignments
from aiijc_puzzle.socket_sorter_production import (
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.socket_translation_placer import (
    CyclicTranslationConfig,
    select_global_cyclic_translation,
)
from aiijc_puzzle.synthetic_socket_evaluation import (
    names_digest,
    select_source_disjoint_train_records,
)
from aiijc_puzzle.whole_layout_cyclic_origin import (
    COMPONENT_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    SOCKET_FEATURE_NAMES,
    WholeLayoutCyclicOriginCNN,
    WholeLayoutOriginConfig,
    assemble_feature_grid,
    best_roll_nll,
    combine_tile_features,
    cyclic_exact_counts,
    learned_best_roll_nll,
    parameter_count,
    select_learned_cyclic_origin,
    topk_hits_best_rolls,
    uniform_best_roll_nll,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/whole_layout_cyclic_origin_preregistered_v1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
TILE_COUNT = GRID * GRID
MAX_FIT_SOURCES = 256
MAX_STEPS = 400
EXPECTED_EVAL_SOURCES = 16
SELECTION_EXCLUSION_PATHS = (
    "configs/component_relation_reranker_preregistered_v1.json",
    "configs/component_relation_confidence_preregistered_v1_1.json",
    "configs/component_relation_cyclic_fresh_gate_v1.json",
    "configs/fullres_relation_fusion_preregistered_v1.json",
    "configs/fullres_relation_fusion_decoder_d2_preregistered_v1.json",
    "configs/border_pointer_preregistered_v1.json",
    "configs/border_pointer_baseline_repair_preregistered_v1.json",
    "outputs/border-pointer/pilot-d64-train128-s400-exact16-mps/selection_commitment.json",
    "outputs/absolute-coordinate-sorter/confirm-d64-frozen-head32-set2-source32-draw2/report.json",
    "outputs/absolute-coordinate-sorter/component-translation-scale-confirm-source64-draw2/report.json",
    "outputs/absolute-coordinate-sorter/coordinate-cyclic-origin-source64-draw2-development/report.json",
    "outputs/absolute-coordinate-sorter/transpose-continuation-d64-train192-s300-source64-draw2/report.json",
    "outputs/socket-matcher/global-cyclic-translation-v1-fresh-source24-draw2/report.json",
    "outputs/socket-matcher/v3-d64-global-cyclic-fresh-source24-draw2/report.json",
    "outputs/component-relation-confidence/fresh-source64-draw2-v1_1-cyclic5/report.json",
)


@dataclass(frozen=True)
class FrozenFeatureCase:
    source_filename: str
    case_id: str
    tile_features: np.ndarray
    reference_layout: np.ndarray
    decoder_layout: np.ndarray
    decoder_counts: np.ndarray


@dataclass(frozen=True)
class EvaluationCase:
    source_filename: str
    case_id: str
    reference_layout: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("capacity", "benchmark", "selection", "pilot"),
        default="pilot",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def collect_declared_filenames(value: Any, *, parent_key: str = "") -> set[str]:
    """Recursively collect both singular and plural declared PNG filenames."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            skip_roster = any(
                marker in lowered
                for marker in ("excluded", "forbidden", "lineage", "remaining", "available")
            )
            if "filename" in lowered and not skip_roster:
                if isinstance(child, str) and child.lower().endswith(".png"):
                    names.add(Path(child).name)
                elif isinstance(child, (list, tuple)):
                    values = [
                        Path(item).name
                        for item in child
                        if isinstance(item, str) and item.lower().endswith(".png")
                    ]
                    if len(values) != len(set(values)):
                        raise ValueError(f"{key} contains duplicate PNG filenames")
                    names.update(values)
            names.update(collect_declared_filenames(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and "filename" not in parent_key.lower():
        for child in value:
            names.update(collect_declared_filenames(child, parent_key=parent_key))
    return names


def load_frozen_config(path: Path) -> tuple[dict[str, Any], str]:
    digest_path = path.with_name(f"{path.name}.sha256")
    expected = digest_path.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("whole-layout origin preregistration hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("registered_before_evaluation_target_access"):
        raise ValueError("whole-layout origin evaluation was not preregistered")
    return value, observed


def _manifest_lookup(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("validation manifest protocol digest is invalid")
    splits = manifest.get("splits")
    records = splits.get("train") if isinstance(splits, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("manifest train split is missing")
    lookup = {str(record["filename"]): record for record in records}
    if len(lookup) != len(records):
        raise ValueError("manifest train filenames are not unique")
    return lookup


def validate_rosters(
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    lookup = _manifest_lookup(manifest)
    fit_names = [str(value) for value in selection["fit_filenames"]]
    eval_names = [str(value) for value in selection["evaluation_filenames"]]
    if not 1 <= len(fit_names) <= MAX_FIT_SOURCES:
        raise ValueError("fit roster exceeds its bounded source count")
    if len(eval_names) != EXPECTED_EVAL_SOURCES:
        raise ValueError("evaluation roster must contain exactly 16 sources")
    if len(set(fit_names)) != len(fit_names) or len(set(eval_names)) != len(eval_names):
        raise ValueError("selection rosters contain duplicate filenames")
    if set(fit_names) & set(eval_names):
        raise ValueError("fit and evaluation rosters overlap")
    if filename_digest(fit_names) != selection["fit_order_digest"]:
        raise ValueError("fit roster digest mismatch")
    if filename_digest(eval_names) != selection["evaluation_order_digest"]:
        raise ValueError("evaluation roster digest mismatch")
    if any(name not in lookup for name in (*fit_names, *eval_names)):
        raise ValueError("a selected source is outside manifest train")
    return [lookup[name] for name in fit_names], [lookup[name] for name in eval_names]


def freeze_selection(args: argparse.Namespace, device: torch.device) -> None:
    """Commit fit256/eval16 metadata without opening a selected target PNG."""

    if args.output_dir is None:
        raise ValueError("selection mode requires --output-dir")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "selection_commitment.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite a selection commitment")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _manifest_lookup(manifest)
    registry: list[dict[str, Any]] = []
    excluded: set[str] = set()
    for raw_path in SELECTION_EXCLUSION_PATHS:
        source = PROJECT_ROOT / raw_path
        value = json.loads(source.read_text(encoding="utf-8"))
        names = collect_declared_filenames(value)
        excluded.update(names)
        registry.append(
            {
                "path": raw_path,
                "sha256": sha256_file(source),
                "actual_roster_count": len(names),
                "actual_roster_digest": hashlib.sha256(
                    "\n".join(sorted(names)).encode()
                ).hexdigest(),
            }
        )
    socket_path = (
        PROJECT_ROOT
        / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
    )
    socket = load_socket_checkpoint(socket_path, device=device)
    excluded.update(socket.lineage.exposed_filenames)
    excluded_digest = hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
    namespace = f"aiijc-whole-layout-cyclic-origin-v1:{excluded_digest}"
    records = select_source_disjoint_train_records(
        manifest,
        excluded_filenames=sorted(excluded),
        limit=MAX_FIT_SOURCES + EXPECTED_EVAL_SOURCES,
        seed=20260911,
        namespace=namespace,
    )
    names = [str(record["filename"]) for record in records]
    fit_names = names[:MAX_FIT_SOURCES]
    eval_names = names[MAX_FIT_SOURCES:]
    commitment = {
        "schema": "whole-layout-cyclic-origin-selection-v1",
        "created_before_selected_target_access": True,
        "manifest_split": "train",
        "seed": 20260911,
        "namespace": namespace,
        "exclusion": {
            "policy": (
                "Socket exposed lineage plus actual rosters from active fusion/pointer "
                "and prior origin/exact panels; excluded/forbidden registry lists are "
                "not treated as opened sources"
            ),
            "registry": registry,
            "socket_checkpoint": str(socket_path.relative_to(PROJECT_ROOT)),
            "socket_checkpoint_sha256": socket.sha256,
            "socket_exposed_count": socket.lineage.exposed_count,
            "socket_exposed_digest": socket.lineage.exposed_digest,
            "union_count": len(excluded),
            "union_digest": excluded_digest,
        },
        "selection": {
            "fit_filenames": fit_names,
            "fit_order_digest": names_digest(fit_names),
            "fit_set_digest": names_digest(fit_names, sort_names=True),
            "evaluation_filenames": eval_names,
            "evaluation_order_digest": names_digest(eval_names),
            "evaluation_set_digest": names_digest(eval_names, sort_names=True),
            "fit_evaluation_overlap": sorted(set(fit_names) & set(eval_names)),
            "selected_exclusion_overlap": sorted(set(names) & excluded),
        },
        "calibration_holdout_competition_test_opened": False,
    }
    path.write_text(
        json.dumps(commitment, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "selection-frozen",
                "path": str(path),
                "sha256": sha256_file(path),
                "fit_digest": commitment["selection"]["fit_order_digest"],
                "evaluation_digest": commitment["selection"][
                    "evaluation_order_digest"
                ],
                "selected_target_access": False,
            }
        ),
        flush=True,
    )


def load_selection_commitment(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    reference = config["selection_commitment"]
    path = PROJECT_ROOT / reference["path"]
    observed = sha256_file(path)
    if observed != reference["sha256"]:
        raise ValueError("selection commitment hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("created_before_selected_target_access"):
        raise ValueError("selection commitment was not frozen before access")
    selection = value["selection"]
    if selection["fit_evaluation_overlap"] or selection["selected_exclusion_overlap"]:
        raise ValueError("selection commitment records a forbidden overlap")
    return selection, observed


def _tile_tensor(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    value = np.asarray(tiles)
    if value.shape != (TILE_COUNT, 20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("dirty tiles violate the exact original-tile contract")
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


@torch.no_grad()
def extract_case_features(
    dirty_tiles: np.ndarray,
    *,
    socket: Any,
    device: torch.device,
) -> tuple[np.ndarray, Any, Any, tuple[str, ...]]:
    tokens, output = extract_frozen_socket_context(
        socket.model,
        _tile_tensor(dirty_tiles, device=device),
        grid=GRID,
    )
    decoder_config = SocketDecoderConfig(
        component_edge_budget_per_axis=144,
        max_swap_steps=24,
    )
    decoder = decode_socket_assignments(
        output.right_log_assignment,
        output.down_log_assignment,
        grid=GRID,
        config=decoder_config,
    )
    component_build = rebuild_decoder_components(
        output.right_log_assignment,
        output.down_log_assignment,
        grid=GRID,
        edge_budget_per_axis=144,
    )
    features, names = combine_tile_features(
        dirty_tiles,
        tokens,
        output,
        component_build,
        grid=GRID,
    )
    return features, output, decoder, names


def precompute_fit_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    cache: CleanTileCache,
    socket: Any,
    device: torch.device,
    seed: int,
) -> tuple[list[FrozenFeatureCase], tuple[str, ...]]:
    cases: list[FrozenFeatureCase] = []
    feature_names: tuple[str, ...] | None = None
    for index, record in enumerate(records, start=1):
        prepared = prepare_case(cache, record, draw_index=0, seed=seed)
        reference = np.argsort(prepared.input_tile_to_position).astype(np.int32)
        features, _, decoder, observed_names = extract_case_features(
            prepared.dirty_tiles,
            socket=socket,
            device=device,
        )
        if feature_names is None:
            feature_names = observed_names
        elif observed_names != feature_names:
            raise RuntimeError("frozen feature schema changed between cases")
        cases.append(
            FrozenFeatureCase(
                source_filename=prepared.source_filename,
                case_id=prepared.case_id,
                tile_features=features,
                reference_layout=reference,
                decoder_layout=decoder.layout,
                decoder_counts=cyclic_exact_counts(decoder.layout, reference, grid=GRID),
            )
        )
        if index == 1 or index % 16 == 0 or index == len(records):
            print(
                json.dumps(
                    {"event": "precompute-fit", "done": index, "total": len(records)}
                ),
                flush=True,
            )
    if feature_names is None:
        raise RuntimeError("fit roster produced no feature cases")
    return cases, feature_names


def _rolled_training_example(
    case: FrozenFeatureCase,
    *,
    exact_stage: bool,
    row_roll: int,
    column_roll: int,
) -> tuple[np.ndarray, np.ndarray]:
    if exact_stage:
        layout = case.reference_layout
        counts = np.zeros((GRID, GRID), dtype=np.int32)
        counts[0, 0] = TILE_COUNT
    else:
        layout = case.decoder_layout
        counts = case.decoder_counts
    grid = assemble_feature_grid(case.tile_features, layout, grid=GRID)
    grid = np.roll(grid, shift=(row_roll, column_roll), axis=(1, 2))
    shifted_counts = np.roll(
        counts,
        shift=(-row_roll, -column_roll),
        axis=(0, 1),
    )
    return np.ascontiguousarray(grid), np.ascontiguousarray(shifted_counts)


def train_origin_model(
    model: WholeLayoutCyclicOriginCNN,
    cases: Sequence[FrozenFeatureCase],
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    training = config["training"]
    steps = int(training["steps"])
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError("training step count exceeds the bounded contract")
    batch_size = int(training["batch_size"])
    exact_steps = int(training["exact_roll_only_steps"])
    mixed_steps = int(training["mixed_curriculum_steps"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
        eta_min=float(training["learning_rate"]) * 0.1,
    )
    generator = np.random.default_rng(int(training["seed"]) + 17)
    history: list[dict[str, float]] = []
    started = perf_counter()
    model.train()
    for step in range(steps):
        grids: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        exact_examples = 0
        for _ in range(batch_size):
            case = cases[int(generator.integers(len(cases)))]
            if step < exact_steps:
                exact_stage = True
            elif step < exact_steps + mixed_steps:
                exact_stage = bool(generator.integers(2))
            else:
                exact_stage = False
            exact_examples += int(exact_stage)
            grid, target = _rolled_training_example(
                case,
                exact_stage=exact_stage,
                row_roll=int(generator.integers(GRID)),
                column_roll=int(generator.integers(GRID)),
            )
            grids.append(grid)
            counts.append(target)
        feature_tensor = torch.from_numpy(np.stack(grids)).to(device)
        count_tensor = torch.from_numpy(np.stack(counts)).to(device)
        logits = model(feature_tensor)
        loss = best_roll_nll(logits, count_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        top1 = logits.detach().flatten(1).argmax(1)
        flat_counts = count_tensor.flatten(1)
        best = flat_counts == flat_counts.max(1, keepdim=True).values
        batch_index = torch.arange(batch_size, device=device)
        top1_hit = float(best[batch_index, top1].float().mean().cpu())
        row = {
            "step": float(step + 1),
            "loss": float(loss.detach().cpu()),
            "top1_best_roll": top1_hit,
            "exact_stage_fraction": exact_examples / batch_size,
            "grad_norm": grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(row)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            recent = history[-min(25, len(history)) :]
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step + 1,
                        "loss": float(np.mean([value["loss"] for value in recent])),
                        "top1_best_roll": float(
                            np.mean([value["top1_best_roll"] for value in recent])
                        ),
                        "exact_stage_fraction": float(
                            np.mean([value["exact_stage_fraction"] for value in recent])
                        ),
                    }
                ),
                flush=True,
            )
    return history, perf_counter() - started


def _uniform_topk_hit_probability(best_count: int, *, total: int, cap: int) -> float:
    if not 1 <= best_count <= total or not 1 <= cap <= total:
        raise ValueError("uniform top-k probability received invalid counts")
    if total - best_count < cap:
        return 1.0
    miss = 1.0
    for offset in range(cap):
        miss *= (total - best_count - offset) / (total - offset)
    return 1.0 - miss


def evaluate_discovery_gate(
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    delta = summary["delta"]
    diagnostics = summary["roll_diagnostics"]
    exact_delta = float(delta["exact_tiles_per_board"])
    adjacency_delta = float(delta["adjacency"])
    exact_path = exact_delta >= float(contract["minimum_exact_gain_per_board"])
    r1_signal = float(diagnostics["r1_gain_over_uniform"]) >= float(
        contract["minimum_r1_gain_over_uniform"]
    )
    r5_signal = float(diagnostics["r5_gain_over_uniform"]) >= float(
        contract["minimum_r5_gain_over_uniform"]
    )
    nll_signal = float(diagnostics["nll_gain_over_uniform"]) >= float(
        contract["minimum_nll_gain_over_uniform"]
    )
    auxiliary_signal = r1_signal or r5_signal or nll_signal
    exact_nonnegative = exact_delta >= 0.0
    adjacency_ok = adjacency_delta >= float(contract["minimum_adjacency_delta"])
    strict_ok = int(summary["strict_original_permutations"]) == int(
        contract["strict_original_permutations_required"]
    )
    passed = (exact_path or (auxiliary_signal and exact_nonnegative)) and adjacency_ok and strict_ok
    return {
        "status": "discovery-pass-future-fresh-only" if passed else "discovery-fail-stop",
        "pass": passed,
        "promotion_authorized": False,
        "competition_test_authorized": False,
        "checks": {
            "primary_exact_path": exact_path,
            "auxiliary_signal_path": auxiliary_signal,
            "r1_signal": r1_signal,
            "r5_signal": r5_signal,
            "nll_signal": nll_signal,
            "exact_nonnegative_for_auxiliary_path": exact_nonnegative,
            "adjacency_ok": adjacency_ok,
            "strict_permutations_ok": strict_ok,
        },
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], arm: str, field: str) -> float:
    return float(np.mean([float(row[arm][field]) for row in rows]))


def capacity_smoke(device: torch.device) -> None:
    seed = 11
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    channels = len(RAW_FEATURE_NAMES) + 64 + len(SOCKET_FEATURE_NAMES) + len(
        COMPONENT_FEATURE_NAMES
    )
    model = WholeLayoutCyclicOriginCNN(
        WholeLayoutOriginConfig(input_channels=channels)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    canonical = []
    for index in range(8):
        value = torch.from_numpy(
            generator.normal(0.0, 0.15, size=(channels, GRID, GRID)).astype(np.float32)
        )
        value[0, 0] += 3.0
        value[1, :, 0] += 3.0
        value[2, -1] -= 2.0
        value[3, :, -1] -= 2.0
        value[4:8, 4 + index % 5 : 14 + index % 5, 5:17] += 1.0
        canonical.append(value)
    first_loss = math.nan
    final_accuracy = 0.0
    for step in range(80):
        grids = []
        counts = []
        for index, value in enumerate(canonical):
            row_roll = (step * 3 + index * 5) % GRID
            column_roll = (step * 7 + index * 2) % GRID
            grids.append(torch.roll(value, (row_roll, column_roll), (1, 2)))
            target = torch.zeros((GRID, GRID), dtype=torch.long)
            target[-row_roll % GRID, -column_roll % GRID] = TILE_COUNT
            counts.append(target)
        feature_tensor = torch.stack(grids).to(device)
        count_tensor = torch.stack(counts).to(device)
        logits = model(feature_tensor)
        loss = best_roll_nll(logits, count_tensor)
        if step == 0:
            first_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_accuracy = float(
            (
                logits.detach().flatten(1).argmax(1)
                == count_tensor.flatten(1).argmax(1)
            )
            .float()
            .mean()
            .cpu()
        )
    final_loss = float(loss.detach().cpu())
    passed = final_accuracy == 1.0 and final_loss < 0.01 * first_loss
    print(
        json.dumps(
            {
                "event": "capacity-complete",
                "device": str(device),
                "parameters": parameter_count(model),
                "first_loss": first_loss,
                "final_loss": final_loss,
                "final_r1": final_accuracy,
                "pass": passed,
            }
        )
    )
    if not passed:
        raise RuntimeError("whole-layout capacity smoke failed")


def benchmark_update(device: torch.device) -> None:
    # MPS currently lacks a deterministic backward for log-sum-exp indexing.
    # This standalone timing probe may measure it, but the pilot freezes CPU
    # head training and uses MPS only for frozen inference feature extraction.
    if device.type == "mps":
        torch.use_deterministic_algorithms(False)
    torch.manual_seed(23)
    channels = len(RAW_FEATURE_NAMES) + 64 + len(SOCKET_FEATURE_NAMES) + len(
        COMPONENT_FEATURE_NAMES
    )
    model = WholeLayoutCyclicOriginCNN(
        WholeLayoutOriginConfig(input_channels=channels)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    features = torch.randn(8, channels, GRID, GRID, device=device)
    counts = torch.zeros(8, GRID, GRID, dtype=torch.long, device=device)
    counts[:, 3, 7] = TILE_COUNT
    elapsed = []
    for step in range(15):
        started = perf_counter()
        loss = best_roll_nll(model(features), counts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        if step >= 5:
            elapsed.append(perf_counter() - started)
    print(
        json.dumps(
            {
                "event": "benchmark-complete",
                "device": str(device),
                "mean_update_seconds": float(np.mean(elapsed)),
                "minimum_update_seconds": float(np.min(elapsed)),
            }
        )
    )


def run_pilot(args: argparse.Namespace, feature_device: torch.device) -> None:
    if args.output_dir is None:
        raise ValueError("pilot mode requires --output-dir")
    config, config_hash = load_frozen_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selection, selection_hash = load_selection_commitment(config)
    fit_records, eval_records = validate_rosters(selection, manifest)
    architecture = config["architecture"]
    socket_path = PROJECT_ROOT / config["frozen_socket"]["path"]
    if sha256_file(socket_path) != config["frozen_socket"]["sha256"]:
        raise ValueError("frozen Socket checkpoint hash mismatch")
    socket = load_socket_checkpoint(socket_path, device=feature_device)
    selected = set(selection["fit_filenames"]) | set(
        selection["evaluation_filenames"]
    )
    if selected & set(socket.lineage.exposed_filenames):
        raise ValueError("selected origin sources overlap Socket exposure lineage")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / filename
        for name, filename in {
            "checkpoint": "whole_layout_cyclic_origin.pt",
            "predictions": "frozen_predictions.json",
            "report": "report.json",
        }.items()
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite a whole-layout origin artifact")
    cache = CleanTileCache(args.targets, maximum_boards=32)
    started = perf_counter()
    fit_cases, feature_names = precompute_fit_cases(
        fit_records,
        cache=cache,
        socket=socket,
        device=feature_device,
        seed=int(config["synthetic_seed"]),
    )
    training_device = torch.device(str(config["training"]["device"]))
    if training_device.type != "cpu":
        raise ValueError("v1 freezes deterministic CPU head training")
    model_config = WholeLayoutOriginConfig(
        input_channels=len(feature_names),
        width=int(architecture["width"]),
        dilations=tuple(int(value) for value in architecture["dilations"]),
    )
    model = WholeLayoutCyclicOriginCNN(model_config).to(training_device)
    if parameter_count(model) != int(architecture["parameter_count"]):
        raise ValueError("whole-layout origin parameter count changed")
    history, training_seconds = train_origin_model(
        model,
        fit_cases,
        config=config,
        device=training_device,
    )
    checkpoint = {
        "schema": "whole-layout-cyclic-origin-cnn-v1",
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "feature_names": feature_names,
        "config_sha256": config_hash,
        "selection": selection,
        "selection_commitment_sha256": selection_hash,
        "frozen_socket": config["frozen_socket"],
    }
    torch.save(checkpoint, paths["checkpoint"])
    checkpoint_hash = sha256_file(paths["checkpoint"])

    predictions: list[dict[str, Any]] = []
    evaluation_cases: list[EvaluationCase] = []
    model.eval()
    with torch.no_grad():
        for index, record in enumerate(eval_records, start=1):
            prepared = prepare_case(
                cache,
                record,
                draw_index=0,
                seed=int(config["synthetic_seed"]),
            )
            reference = np.argsort(prepared.input_tile_to_position).astype(np.int32)
            features, output, decoder, observed_names = extract_case_features(
                prepared.dirty_tiles,
                socket=socket,
                device=feature_device,
            )
            if observed_names != feature_names:
                raise RuntimeError("evaluation feature schema changed")
            feature_grid = assemble_feature_grid(features, decoder.layout, grid=GRID)
            logits = model(
                torch.from_numpy(feature_grid).unsqueeze(0).to(training_device)
            )[0]
            candidate = select_learned_cyclic_origin(
                decoder.layout,
                logits,
                grid=GRID,
            )
            baseline = select_global_cyclic_translation(
                decoder.layout,
                output.right_log_assignment,
                output.down_log_assignment,
                grid=GRID,
                config=CyclicTranslationConfig(border_weight=5.0),
            )
            for layout in (baseline.layout, candidate.layout):
                if not np.array_equal(np.sort(layout), np.arange(TILE_COUNT)):
                    raise RuntimeError("origin output is not a strict original permutation")
            predictions.append(
                {
                    "source_filename": prepared.source_filename,
                    "case_id": prepared.case_id,
                    "raw_decoder_layout": decoder.layout.tolist(),
                    "baseline_raw_plus_cyclic5_layout": baseline.layout.tolist(),
                    "candidate_learned_roll_layout": candidate.layout.tolist(),
                    "candidate_roll_logits": logits.detach().float().cpu().numpy().tolist(),
                    "baseline_cyclic": baseline.report(),
                    "candidate_learned": candidate.report(),
                    "decoder": decoder.report(),
                }
            )
            evaluation_cases.append(
                EvaluationCase(prepared.source_filename, prepared.case_id, reference)
            )
            print(
                json.dumps(
                    {"event": "freeze-eval", "done": index, "total": len(eval_records)}
                ),
                flush=True,
            )
    prediction_artifact = {
        "schema": "whole-layout-cyclic-origin-frozen-predictions-v1",
        "config_sha256": config_hash,
        "predictions_frozen_before_reference_scoring": True,
        "competition_test_opened": False,
        "predictions": predictions,
    }
    paths["predictions"].write_text(
        json.dumps(prediction_artifact, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    prediction_hash = sha256_file(paths["predictions"])
    print(
        json.dumps(
            {
                "event": "predictions-frozen",
                "sha256": prediction_hash,
                "cases": len(predictions),
            }
        ),
        flush=True,
    )

    scored: list[dict[str, Any]] = []
    for frozen, case in zip(predictions, evaluation_cases, strict=True):
        if frozen["case_id"] != case.case_id:
            raise RuntimeError("evaluation case identity changed before scoring")
        baseline_metrics = evaluate_layout(
            frozen["baseline_raw_plus_cyclic5_layout"],
            case.reference_layout,
            reference_is_exact=True,
        )
        candidate_metrics = evaluate_layout(
            frozen["candidate_learned_roll_layout"],
            case.reference_layout,
            reference_is_exact=True,
        )
        counts = cyclic_exact_counts(
            frozen["raw_decoder_layout"],
            case.reference_layout,
            grid=GRID,
        )
        logits = np.asarray(frozen["candidate_roll_logits"], dtype=np.float64)
        hits = topk_hits_best_rolls(logits, counts, caps=(1, 5))
        best_count = int(np.sum(counts == counts.max()))
        scored.append(
            {
                "source_filename": case.source_filename,
                "case_id": case.case_id,
                "baseline": baseline_metrics.as_dict(),
                "candidate": candidate_metrics.as_dict(),
                "exact_delta_tiles": candidate_metrics.correct_tile_count
                - baseline_metrics.correct_tile_count,
                "adjacency_delta": candidate_metrics.adjacency
                - baseline_metrics.adjacency,
                "dominant_roll_exact_count": int(counts.max()),
                "best_roll_tie_count": best_count,
                "learned_r1_hit": hits[1],
                "learned_r5_hit": hits[5],
                "uniform_r1_probability": best_count / TILE_COUNT,
                "uniform_r5_probability": _uniform_topk_hit_probability(
                    best_count,
                    total=TILE_COUNT,
                    cap=5,
                ),
                "learned_best_roll_nll": learned_best_roll_nll(logits, counts),
                "uniform_best_roll_nll": uniform_best_roll_nll(counts),
            }
        )
    baseline_summary = {
        field: _mean_metric(scored, "baseline", field)
        for field in (
            "correct_tile_count",
            "direct_placement",
            "translation_aligned_count",
            "translation_aligned_placement",
            "adjacency_correct",
            "adjacency",
        )
    }
    candidate_summary = {
        field: _mean_metric(scored, "candidate", field)
        for field in baseline_summary
    }
    roll_diagnostics = {
        "learned_r1": float(np.mean([row["learned_r1_hit"] for row in scored])),
        "uniform_r1": float(
            np.mean([row["uniform_r1_probability"] for row in scored])
        ),
        "learned_r5": float(np.mean([row["learned_r5_hit"] for row in scored])),
        "uniform_r5": float(
            np.mean([row["uniform_r5_probability"] for row in scored])
        ),
        "learned_best_roll_nll": float(
            np.mean([row["learned_best_roll_nll"] for row in scored])
        ),
        "uniform_best_roll_nll": float(
            np.mean([row["uniform_best_roll_nll"] for row in scored])
        ),
        "dominant_roll_exact_tiles_per_board": float(
            np.mean([row["dominant_roll_exact_count"] for row in scored])
        ),
    }
    roll_diagnostics["r1_gain_over_uniform"] = (
        roll_diagnostics["learned_r1"] - roll_diagnostics["uniform_r1"]
    )
    roll_diagnostics["r5_gain_over_uniform"] = (
        roll_diagnostics["learned_r5"] - roll_diagnostics["uniform_r5"]
    )
    roll_diagnostics["nll_gain_over_uniform"] = (
        roll_diagnostics["uniform_best_roll_nll"]
        - roll_diagnostics["learned_best_roll_nll"]
    )
    strict_count = sum(
        np.array_equal(
            np.sort(np.asarray(row["candidate_learned_roll_layout"])),
            np.arange(TILE_COUNT),
        )
        for row in predictions
    )
    summary = {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": {
            "exact_tiles_per_board": candidate_summary["correct_tile_count"]
            - baseline_summary["correct_tile_count"],
            "adjacency": candidate_summary["adjacency"] - baseline_summary["adjacency"],
            "translation_aligned_tiles_per_board": candidate_summary[
                "translation_aligned_count"
            ]
            - baseline_summary["translation_aligned_count"],
        },
        "roll_diagnostics": roll_diagnostics,
        "strict_original_permutations": int(strict_count),
    }
    gate = evaluate_discovery_gate(summary, config["discovery_gate"])
    report = {
        "experiment": config["experiment"],
        "status": gate["status"],
        "competition_test_opened": False,
        "promotion_applied": False,
        "config": {"path": str(args.config.resolve()), "sha256": config_hash},
        "architecture": {
            **architecture,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "no_tile_identity_or_position_embedding": True,
            "circular_shift_equivariant": True,
        },
        "selection": selection,
        "selection_commitment": {
            **config["selection_commitment"],
            "observed_sha256": selection_hash,
        },
        "training": {
            **config["training"],
            "runtime_seconds": training_seconds,
            "history": history,
        },
        "freeze": {
            "predictions_frozen_before_reference_scoring": True,
            "prediction_sha256": prediction_hash,
            "strict_original_permutation_count": int(strict_count),
        },
        "summary": summary,
        "gate": gate,
        "cases": scored,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "checkpoint": str(paths["checkpoint"]),
            "checkpoint_sha256": checkpoint_hash,
            "frozen_predictions": str(paths["predictions"]),
            "frozen_predictions_sha256": prediction_hash,
            "report": str(paths["report"]),
        },
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "status": gate["status"],
                "gate_pass": gate["pass"],
                "exact_delta": summary["delta"]["exact_tiles_per_board"],
                "adjacency_delta": summary["delta"]["adjacency"],
                "r1_gain": roll_diagnostics["r1_gain_over_uniform"],
                "r5_gain": roll_diagnostics["r5_gain_over_uniform"],
                "nll_gain": roll_diagnostics["nll_gain_over_uniform"],
                "report": str(paths["report"]),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    random.seed(20260911)
    np.random.seed(20260911)
    torch.manual_seed(20260911)
    device = choose_deterministic_device(args.device)
    print(
        json.dumps(
            {"event": "start", "pid": os.getpid(), "mode": args.mode, "device": str(device)}
        ),
        flush=True,
    )
    if args.mode == "capacity":
        capacity_smoke(device)
    elif args.mode == "benchmark":
        benchmark_update(device)
    elif args.mode == "selection":
        freeze_selection(args, device)
    else:
        run_pilot(args, device)


if __name__ == "__main__":
    main()
