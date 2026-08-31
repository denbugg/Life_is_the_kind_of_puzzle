#!/usr/bin/env python3
"""Run one preregistered joint dense-contact TASKA component-pose pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_joint_component_pose import (
    CANDIDATE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    JointComponentPoseTransformer,
    JointPoseBoard,
    JointPoseTargets,
    build_joint_pose_board,
    candidate_ranks,
    joint_pose_loss,
    joint_pose_targets,
    pack_multiple_component_anchors,
    select_component_anchors,
)
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR

try:
    from scripts import run_taska_focal_current_finetune as finetune
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-joint-component-pose/pilot-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_joint_component_pose_pilot_v1.json"
GRID = 24
CONTROL_ARM = "confirmed_six_arm_fusion"
CANDIDATE_ARM = "joint_component_pose"
FORMAL_ROOT = PROJECT_ROOT / (
    "outputs/taska-selective-fullres-union-fusion/"
    "fresh32-formal-confirmation-v1"
)


@dataclass(frozen=True)
class PanelSpec:
    name: str
    archive: Path
    metadata: Path
    base_archive: Path
    formal: bool


FIT = PanelSpec(
    name="fit32",
    archive=FORMAL_ROOT / "frozen-target-free-eval.npz",
    metadata=FORMAL_ROOT / "frozen-target-free-eval.json",
    base_archive=FORMAL_ROOT / "frozen-target-free-eval.npz",
    formal=True,
)
LOCAL = PanelSpec(
    name="local32",
    archive=PROJECT_ROOT
    / "outputs/taska-selective-fullres-union-fusion/fixed-v1/held32/frozen-target-free-eval.npz",
    metadata=PROJECT_ROOT
    / "outputs/taska-selective-fullres-union-fusion/fixed-v1/held32/frozen-target-free-eval.json",
    base_archive=PROJECT_ROOT
    / "outputs/taska-seam-replay/held300-diagnostic-mps-v1/frozen-target-free-eval.npz",
    formal=False,
)
PANELS = (FIT, LOCAL)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mode", choices=("cache", "pilot", "all"), default="all")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed joint-pose preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("joint-pose preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "dense_contact_topk": 8,
        "candidate_cap_per_nontrivial_component": 128,
        "width": 64,
        "layers": 2,
        "heads": 4,
        "capacity_steps": 80,
        "pilot_steps": 240,
        "maximum_anchors": 4,
        "coverage_threshold": 0.5,
        "purity_threshold": 0.5,
        "candidate_probability_threshold": 0.05,
        "no_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"joint-pose preregistration mismatch: {key}")
    for relative, expected in config["fixed_source_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"signed source changed: {relative}")
    for relative, expected in config["fixed_input_sha256"].items():
        source = PROJECT_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"signed input changed: {relative}")
    return config, digest


def _rows(spec: PanelSpec) -> list[Mapping[str, Any]]:
    rows = json.loads(spec.metadata.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError(f"{spec.metadata} must contain exactly 32 frozen rows")
    return rows


def _selected_families(choice: str, *, formal: bool) -> tuple[str, ...]:
    if choice == "combined_union_focal":
        return ("combined_union",)
    if choice == "selective_vote500_focal":
        return ("current", "selective_accepted" if formal else "selective_new")
    return ("current",)


def _selected_edges(
    archive: Any,
    prefix: str,
    choice: str,
    *,
    formal: bool,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    edges: list[RawTailEdge] = []
    logits: list[np.ndarray] = []
    for name in _selected_families(choice, formal=formal):
        source = np.asarray(archive[f"{prefix}__{name}__edge_source"], dtype=np.int32)
        target = np.asarray(archive[f"{prefix}__{name}__edge_target"], dtype=np.int32)
        axis = np.asarray(archive[f"{prefix}__{name}__edge_axis"], dtype=np.uint8)
        values = np.asarray(
            archive[f"{prefix}__{name}_focal_logits"], dtype=np.float32
        )
        if not (source.shape == target.shape == axis.shape == values.shape):
            raise ValueError("selected edge arrays are not aligned")
        if not np.isin(axis, (0, 1)).all():
            raise ValueError("selected edge direction is malformed")
        edges.extend(
            RawTailEdge(
                int(first),
                int(second),
                "right" if int(direction) == 0 else "down",
            )
            for first, second, direction in zip(source, target, axis, strict=True)
        )
        logits.append(values)
    return tuple(edges), np.concatenate(logits)


def _layout_key(prefix: str, *, formal: bool) -> str:
    suffix = (
        "selective_unique_fullres_fusion_focal_gated_tail96_layout"
        if formal
        else "combined_union_candidate_layout"
    )
    return f"{prefix}__{suffix}"


def _board_arrays(key: str, board: JointPoseBoard) -> dict[str, np.ndarray]:
    return {
        f"{key}__layout": board.layout,
        f"{key}__component_of_tile": board.component_of_tile,
        f"{key}__component_relative_coordinates": board.component_relative_coordinates,
        f"{key}__component_sizes": board.component_sizes,
        f"{key}__component_origins": board.component_origins,
        f"{key}__node_features": board.node_features,
        f"{key}__pair_index": board.pair_index,
        f"{key}__pair_features": board.pair_features,
        f"{key}__candidate_component": board.candidate_component,
        f"{key}__candidate_shift": board.candidate_shift,
        f"{key}__candidate_features": board.candidate_features,
        f"{key}__candidate_raw_score": board.candidate_raw_score,
    }


def _target_arrays(key: str, targets: JointPoseTargets) -> dict[str, np.ndarray]:
    return {
        f"{key}__dominant_shift": targets.dominant_shift,
        f"{key}__dominant_support": targets.dominant_support,
        f"{key}__purity": targets.purity,
        f"{key}__covered": targets.covered,
        f"{key}__positive_candidate": targets.positive_candidate,
    }


def _load_board(archive: Any, key: str) -> JointPoseBoard:
    return JointPoseBoard(
        layout=np.asarray(archive[f"{key}__layout"], dtype=np.int32),
        component_of_tile=np.asarray(
            archive[f"{key}__component_of_tile"], dtype=np.int32
        ),
        component_relative_coordinates=np.asarray(
            archive[f"{key}__component_relative_coordinates"], dtype=np.int16
        ),
        component_sizes=np.asarray(archive[f"{key}__component_sizes"], dtype=np.int16),
        component_origins=np.asarray(
            archive[f"{key}__component_origins"], dtype=np.int16
        ),
        node_features=np.asarray(archive[f"{key}__node_features"], dtype=np.float32),
        pair_index=np.asarray(archive[f"{key}__pair_index"], dtype=np.int32),
        pair_features=np.asarray(archive[f"{key}__pair_features"], dtype=np.float32),
        candidate_component=np.asarray(
            archive[f"{key}__candidate_component"], dtype=np.int32
        ),
        candidate_shift=np.asarray(
            archive[f"{key}__candidate_shift"], dtype=np.int16
        ),
        candidate_features=np.asarray(
            archive[f"{key}__candidate_features"], dtype=np.float32
        ),
        candidate_raw_score=np.asarray(
            archive[f"{key}__candidate_raw_score"], dtype=np.float32
        ),
    )


def _load_targets(archive: Any, key: str) -> JointPoseTargets:
    return JointPoseTargets(
        dominant_shift=np.asarray(archive[f"{key}__dominant_shift"], dtype=np.int16),
        dominant_support=np.asarray(
            archive[f"{key}__dominant_support"], dtype=np.int16
        ),
        purity=np.asarray(archive[f"{key}__purity"], dtype=np.float32),
        covered=np.asarray(archive[f"{key}__covered"], dtype=bool),
        positive_candidate=np.asarray(
            archive[f"{key}__positive_candidate"], dtype=bool
        ),
    )


def _build_cache(
    *,
    output_dir: Path,
    targets_dir: Path,
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    input_arrays: dict[str, np.ndarray] = {}
    fit_arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    tile_cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    started = perf_counter()
    for spec in PANELS:
        rows = _rows(spec)
        with (
            np.load(spec.archive, allow_pickle=False) as frozen,
            np.load(spec.base_archive, allow_pickle=False) as base,
        ):
            for index, row in enumerate(rows):
                prefix = str(row["prefix"])
                key = f"{spec.name}_{index:03d}"
                source = str(row["source_filename"])
                draw = int(row["draw_index"])
                dirty = finetune._dirty_case(tile_cache, lookup[source], source, draw)
                if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                    raise RuntimeError("cache recreation changed frozen dirty bytes")
                edges, logits = _selected_edges(
                    frozen,
                    prefix,
                    str(row["choice"]),
                    formal=spec.formal,
                )
                board = build_joint_pose_board(
                    layout=frozen[_layout_key(prefix, formal=spec.formal)],
                    dirty_tiles=dirty.dirty_tiles,
                    cost_right=base[f"{prefix}__cost_right"],
                    cost_down=base[f"{prefix}__cost_down"],
                    selected_edges=edges,
                    selected_logits=logits,
                    grid=GRID,
                    dense_topk=int(config["dense_contact_topk"]),
                    candidate_cap=int(config["candidate_cap_per_nontrivial_component"]),
                )
                input_arrays.update(_board_arrays(key, board))
                if spec is FIT:
                    reference = finetune._reference(
                        tile_cache,
                        lookup[source],
                        source,
                        draw,
                        dirty.dirty_tiles,
                    )
                    fit_arrays.update(_target_arrays(key, joint_pose_targets(board, reference)))
                metadata_rows.append(
                    {
                        "key": key,
                        "panel": spec.name,
                        "prefix": prefix,
                        "source_filename": source,
                        "draw_index": draw,
                        "dirty_sha256": str(row["dirty_sha256"]),
                        "fusion_choice": str(row["choice"]),
                        "component_count": int(len(board.component_sizes)),
                        "nontrivial_component_count": int(
                            np.count_nonzero(board.component_sizes >= 2)
                        ),
                        "pair_relation_count": int(len(board.pair_features)),
                        "candidate_count": int(len(board.candidate_features)),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "joint_pose_cache",
                            "panel": spec.name,
                            "case": index + 1,
                            "components": len(board.component_sizes),
                            "candidates": len(board.candidate_features),
                        }
                    ),
                    flush=True,
                )
    inputs = cache_dir / "dirty-visible-inputs.npz"
    labels = cache_dir / "fit-exact-labels.npz"
    metadata = cache_dir / "metadata.json"
    _write_npz(inputs, input_arrays)
    _write_npz(labels, fit_arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-joint-component-pose-cache-v1",
            "input_cache_contains_exact_labels": False,
            "fit_label_cache_contains_only_fit32_exact_labels": True,
            "local32_exact_labels_or_references_cached": False,
            "competition_test_accessed": False,
            "rows": metadata_rows,
        },
    )
    freeze = cache_dir / "cache-freeze.json"
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-joint-component-pose-cache-freeze-v1",
            "artifacts": {
                "dirty_visible_inputs": _record(inputs),
                "fit_exact_labels": _record(labels),
                "metadata": _record(metadata),
                "preregistration": _record(config_path),
                **{
                    f"{spec.name}_{kind}": _record(path)
                    for spec in PANELS
                    for kind, path in (
                        ("archive", spec.archive),
                        ("metadata", spec.metadata),
                        ("base_archive", spec.base_archive),
                    )
                },
            },
        },
    )
    return {
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "inputs": _record(inputs),
            "fit_labels": _record(labels),
            "metadata": _record(metadata),
            "freeze": _record(freeze),
        },
    }


@dataclass(frozen=True)
class FeatureStatistics:
    node_mean: np.ndarray
    node_scale: np.ndarray
    pair_mean: np.ndarray
    pair_scale: np.ndarray
    candidate_mean: np.ndarray
    candidate_scale: np.ndarray


def _feature_statistics(boards: Sequence[JointPoseBoard]) -> FeatureStatistics:
    def stats(values: Sequence[np.ndarray], dimension: int) -> tuple[np.ndarray, np.ndarray]:
        merged = np.concatenate(values, axis=0).astype(np.float64)
        if merged.shape[1] != dimension:
            raise RuntimeError("feature dimension changed")
        mean = merged.mean(axis=0).astype(np.float32)
        scale = merged.std(axis=0).astype(np.float32)
        scale[scale < 1e-4] = 1.0
        return mean, scale

    node = stats([board.node_features for board in boards], NODE_FEATURE_DIM)
    pair = stats([board.pair_features for board in boards], PAIR_FEATURE_DIM)
    candidate = stats(
        [board.candidate_features for board in boards], CANDIDATE_FEATURE_DIM
    )
    return FeatureStatistics(node[0], node[1], pair[0], pair[1], candidate[0], candidate[1])


def _tensor_board(
    board: JointPoseBoard,
    statistics: FeatureStatistics,
    *,
    device: torch.device,
    generator: np.random.Generator | None,
    jitter: float,
    pair_dropout: float,
    candidate_dropout: float,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    node = (board.node_features - statistics.node_mean) / statistics.node_scale
    pair = (board.pair_features - statistics.pair_mean) / statistics.pair_scale
    candidate = (
        board.candidate_features - statistics.candidate_mean
    ) / statistics.candidate_scale
    component = board.candidate_component.copy()
    pair_index = board.pair_index.copy()
    order = np.arange(len(node), dtype=np.int32)
    if generator is not None:
        order = generator.permutation(len(node)).astype(np.int32)
        inverse = np.empty(len(node), dtype=np.int32)
        inverse[order] = np.arange(len(node), dtype=np.int32)
        node = node[order]
        pair_index = inverse[pair_index]
        component = inverse[component]
        node = node + generator.normal(0.0, jitter, size=node.shape).astype(np.float32)
        pair = pair + generator.normal(0.0, jitter, size=pair.shape).astype(np.float32)
        candidate = candidate + generator.normal(
            0.0, jitter, size=candidate.shape
        ).astype(np.float32)
        pair[generator.random(len(pair)) < pair_dropout] = 0.0
        candidate[generator.random(len(candidate)) < candidate_dropout] = 0.0
    tensors = {
        "node_features": torch.as_tensor(node, dtype=torch.float32, device=device),
        "pair_index": torch.as_tensor(pair_index, dtype=torch.long, device=device),
        "pair_features": torch.as_tensor(pair, dtype=torch.float32, device=device),
        "candidate_component": torch.as_tensor(
            component, dtype=torch.long, device=device
        ),
        "candidate_features": torch.as_tensor(
            candidate, dtype=torch.float32, device=device
        ),
        "candidate_raw_score": torch.as_tensor(
            board.candidate_raw_score, dtype=torch.float32, device=device
        ),
    }
    return tensors, order


def _tensor_targets(
    board: JointPoseBoard,
    targets: JointPoseTargets,
    order: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "positive_candidate": torch.as_tensor(
            targets.positive_candidate, dtype=torch.bool, device=device
        ),
        "component_sizes": torch.as_tensor(
            board.component_sizes[order], dtype=torch.long, device=device
        ),
        "dominant_support": torch.as_tensor(
            targets.dominant_support[order], dtype=torch.float32, device=device
        ),
        "purity": torch.as_tensor(targets.purity[order], dtype=torch.float32, device=device),
        "covered": torch.as_tensor(targets.covered[order], dtype=torch.bool, device=device),
    }


def _new_model(config: Mapping[str, Any], device: torch.device) -> JointComponentPoseTransformer:
    return JointComponentPoseTransformer(
        width=int(config["width"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
    ).to(device)


def _train(
    *,
    boards: Sequence[JointPoseBoard],
    targets: Sequence[JointPoseTargets],
    statistics: FeatureStatistics,
    config: Mapping[str, Any],
    device: torch.device,
    steps: int,
    seed: int,
    learning_rate: float,
    augment: bool,
) -> tuple[JointComponentPoseTransformer, dict[str, Any]]:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    started = perf_counter()
    for step in range(steps):
        index = 0 if len(boards) == 1 else int(generator.integers(0, len(boards)))
        tensors, order = _tensor_board(
            boards[index],
            statistics,
            device=device,
            generator=generator if augment else None,
            jitter=float(config["feature_jitter_sigma"]) if augment else 0.0,
            pair_dropout=float(config["pair_feature_dropout"]) if augment else 0.0,
            candidate_dropout=(
                float(config["candidate_feature_dropout"]) if augment else 0.0
            ),
        )
        labels = _tensor_targets(boards[index], targets[index], order, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors)
        loss, _ = joint_pose_loss(
            output,
            candidate_component=tensors["candidate_component"],
            **labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 20 == 0 or step == 0:
            print(
                json.dumps(
                    {
                        "event": "joint_pose_train",
                        "step": step + 1,
                        "steps": steps,
                        "loss": losses[-1],
                    }
                ),
                flush=True,
            )
    return model, {
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail20_loss": float(np.mean(losses[-20:])),
        "runtime_seconds": perf_counter() - started,
    }


def _scores(
    model: JointComponentPoseTransformer,
    board: JointPoseBoard,
    statistics: FeatureStatistics,
    *,
    device: torch.device,
) -> tuple[np.ndarray, Any]:
    tensors, _ = _tensor_board(
        board,
        statistics,
        device=device,
        generator=None,
        jitter=0.0,
        pair_dropout=0.0,
        candidate_dropout=0.0,
    )
    model.eval()
    with torch.inference_mode():
        output = model(**tensors)
    return output.candidate_score.detach().cpu().numpy(), output


def _retrieval_summary(
    boards: Sequence[JointPoseBoard],
    targets: Sequence[JointPoseTargets],
    scores: Sequence[np.ndarray],
) -> dict[str, Any]:
    ranks: list[int] = []
    weights: list[float] = []
    covered_support = 0.0
    total_support = 0.0
    covered_components = 0
    total_components = 0
    for board, labels, values in zip(boards, targets, scores, strict=True):
        result = candidate_ranks(values, board, labels)
        for component, rank in result.items():
            ranks.append(rank)
            weights.append(float(labels.dominant_support[component]))
        mask = board.component_sizes >= 2
        covered_support += float(labels.dominant_support[mask & labels.covered].sum())
        total_support += float(labels.dominant_support[mask].sum())
        covered_components += int(np.count_nonzero(mask & labels.covered))
        total_components += int(np.count_nonzero(mask))
    rank = np.asarray(ranks, dtype=np.int32)
    weight = np.asarray(weights, dtype=np.float64)
    return {
        "conditional_component_count": len(rank),
        "component_coverage": covered_components / total_components,
        "dominant_support_coverage": covered_support / total_support,
        "r_at_1": float(np.mean(rank <= 1)),
        "r_at_5": float(np.mean(rank <= 5)),
        "support_weighted_r_at_1": float(np.sum(weight * (rank <= 1)) / weight.sum()),
        "support_weighted_r_at_5": float(np.sum(weight * (rank <= 5)) / weight.sum()),
    }


def _layout_metrics(layout: Any, reference: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "exact_tiles": int(result.correct_tile_count),
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
    }


def _pilot(
    *,
    output_dir: Path,
    targets_dir: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    config_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache_dir = output_dir / "cache"
    inputs_path = cache_dir / "dirty-visible-inputs.npz"
    labels_path = cache_dir / "fit-exact-labels.npz"
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fit_rows = [row for row in metadata["rows"] if row["panel"] == FIT.name]
    local_rows = [row for row in metadata["rows"] if row["panel"] == LOCAL.name]
    with (
        np.load(inputs_path, allow_pickle=False) as inputs,
        np.load(labels_path, allow_pickle=False) as labels,
    ):
        fit_boards = [_load_board(inputs, str(row["key"])) for row in fit_rows]
        fit_targets = [_load_targets(labels, str(row["key"])) for row in fit_rows]
        local_boards = [_load_board(inputs, str(row["key"])) for row in local_rows]
    statistics = _feature_statistics(fit_boards)
    capacity_model, capacity_train = _train(
        boards=(fit_boards[0],),
        targets=(fit_targets[0],),
        statistics=statistics,
        config=config,
        device=device,
        steps=int(config["capacity_steps"]),
        seed=int(config["capacity_seed"]),
        learning_rate=float(config["capacity_learning_rate"]),
        augment=False,
    )
    capacity_score, _ = _scores(
        capacity_model, fit_boards[0], statistics, device=device
    )
    capacity = {
        "training": capacity_train,
        "retrieval": _retrieval_summary(
            (fit_boards[0],), (fit_targets[0],), (capacity_score,)
        ),
    }
    model, training = _train(
        boards=fit_boards,
        targets=fit_targets,
        statistics=statistics,
        config=config,
        device=device,
        steps=int(config["pilot_steps"]),
        seed=int(config["pilot_seed"]),
        learning_rate=float(config["pilot_learning_rate"]),
        augment=True,
    )
    checkpoint = output_dir / "joint_component_pose.pt"
    torch.save(
        {
            "schema": "aiijc-taska-joint-component-pose-checkpoint-v1",
            "state_dict": model.state_dict(),
            "statistics": asdict(statistics),
            "config": dict(config),
            "config_sha256": config_sha256,
            "fit_source_filenames": sorted(
                {str(row["source_filename"]) for row in fit_rows}
            ),
            "lineage_exposed_filenames": sorted(
                {
                    str(row["source_filename"])
                    for row in (*fit_rows, *local_rows)
                }
            ),
        },
        checkpoint,
    )
    fit_raw = [board.candidate_raw_score for board in fit_boards]
    fit_learned = [
        _scores(model, board, statistics, device=device)[0] for board in fit_boards
    ]
    fit_retrieval = {
        "raw": _retrieval_summary(fit_boards, fit_targets, fit_raw),
        "learned": _retrieval_summary(fit_boards, fit_targets, fit_learned),
    }

    predictions: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    for row, board in zip(local_rows, local_boards, strict=True):
        learned, output = _scores(model, board, statistics, device=device)
        anchors = select_component_anchors(
            output,
            board,
            maximum_anchors=int(config["maximum_anchors"]),
            coverage_threshold=float(config["coverage_threshold"]),
            purity_threshold=float(config["purity_threshold"]),
            candidate_probability_threshold=float(
                config["candidate_probability_threshold"]
            ),
        )
        candidate, diagnostics = pack_multiple_component_anchors(board, anchors, grid=GRID)
        key = str(row["key"])
        predictions[f"{key}__control_layout"] = board.layout
        predictions[f"{key}__candidate_layout"] = candidate
        predictions[f"{key}__learned_score"] = learned
        predictions[f"{key}__coverage_logit"] = (
            output.coverage_logit.detach().cpu().numpy()
        )
        predictions[f"{key}__purity_logit"] = output.purity_logit.detach().cpu().numpy()
        frozen_rows.append(
            {
                **dict(row),
                "anchors": [
                    {
                        "component_index": component,
                        "shift": list(shift),
                        "confidence": confidence,
                    }
                    for component, shift, confidence in anchors
                ],
                "packing": asdict(diagnostics),
            }
        )
    prediction_path = output_dir / "local32-frozen-predictions.npz"
    prediction_metadata = output_dir / "local32-frozen-predictions.json"
    _write_npz(prediction_path, predictions)
    _write_json(
        prediction_metadata,
        {
            "schema": "aiijc-taska-joint-component-pose-local-freeze-v1",
            "created_before_local_exact_reference_reconstruction": True,
            "contains_exact_references_or_labels": False,
            "all_layouts_strict_original_upright_permutations": True,
            "rows": frozen_rows,
        },
    )
    pre_score = output_dir / "pre-score-freeze.json"
    _write_json(
        pre_score,
        {
            "schema": "aiijc-taska-joint-component-pose-pre-score-freeze-v1",
            "created_before_local_exact_reference_reconstruction": True,
            "contains_exact_references_or_labels": False,
            "artifacts": {
                "predictions": _record(prediction_path),
                "prediction_metadata": _record(prediction_metadata),
                "checkpoint": _record(checkpoint),
                "cache_inputs": _record(inputs_path),
                "cache_metadata": _record(metadata_path),
                "preregistration": _record(config_path),
                "runner": _record(Path(__file__).resolve()),
                "model": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/taska_joint_component_pose.py"
                ),
            },
        },
    )

    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    tile_cache = finetune.CleanTileCache(targets_dir, maximum_boards=2)
    local_targets: list[JointPoseTargets] = []
    local_scores: list[np.ndarray] = []
    scored_rows: list[dict[str, Any]] = []
    with np.load(prediction_path, allow_pickle=False) as frozen:
        for row, board in zip(local_rows, local_boards, strict=True):
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(tile_cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("local scoring recreated different dirty bytes")
            reference = finetune._reference(
                tile_cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            labels = joint_pose_targets(board, reference)
            local_targets.append(labels)
            key = str(row["key"])
            local_scores.append(np.asarray(frozen[f"{key}__learned_score"]))
            scored_rows.append(
                {
                    **dict(row),
                    "metrics": {
                        CONTROL_ARM: _layout_metrics(
                            frozen[f"{key}__control_layout"], reference
                        ),
                        CANDIDATE_ARM: _layout_metrics(
                            frozen[f"{key}__candidate_layout"], reference
                        ),
                    },
                }
            )
    raw_retrieval = _retrieval_summary(
        local_boards,
        local_targets,
        [board.candidate_raw_score for board in local_boards],
    )
    learned_retrieval = _retrieval_summary(local_boards, local_targets, local_scores)
    metric_names = ("exact_tiles", "satisfied_adjacent_pairs", "adjacency_recall")
    arms = {
        arm: {
            metric: float(np.mean([row["metrics"][arm][metric] for row in scored_rows]))
            for metric in metric_names
        }
        for arm in (CONTROL_ARM, CANDIDATE_ARM)
    }
    delta = {
        metric: float(arms[CANDIDATE_ARM][metric] - arms[CONTROL_ARM][metric])
        for metric in metric_names
    }
    gate = {
        "capacity_r_at_1_at_least_0_80": capacity["retrieval"]["r_at_1"] >= 0.80,
        "local_support_weighted_r1_gain_at_least_0_02": (
            learned_retrieval["support_weighted_r_at_1"]
            - raw_retrieval["support_weighted_r_at_1"]
            >= 0.02
        ),
        "local_support_weighted_r5_loss_no_worse_than_0_01": (
            learned_retrieval["support_weighted_r_at_5"]
            - raw_retrieval["support_weighted_r_at_5"]
            >= -0.01
        ),
        "local_exact_delta_strictly_positive": delta["exact_tiles"] > 0.0,
        "local_pair_delta_at_least_minus_1": delta["satisfied_adjacent_pairs"] >= -1.0,
    }
    gate["passed"] = all(gate.values())
    return {
        "capacity": capacity,
        "training": training,
        "fit_retrieval": fit_retrieval,
        "local32": {
            "raw_retrieval": raw_retrieval,
            "learned_retrieval": learned_retrieval,
            "arms": arms,
            "candidate_minus_control": delta,
            "changed_layout_count": sum(
                row["packing"]["total_tile_l1_displacement"] > 0
                for row in frozen_rows
            ),
            "rows": scored_rows,
        },
        "gate": gate,
        "artifacts": {
            "checkpoint": _record(checkpoint),
            "predictions": _record(prediction_path),
            "prediction_metadata": _record(prediction_metadata),
            "pre_score_freeze": _record(pre_score),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = _load_config(args.config)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested MPS device is unavailable")
    targets_dir = args.targets.resolve()
    if (
        not targets_dir.is_dir()
        or targets_dir.name != "targets"
        or targets_dir.parent.name != "train"
    ):
        raise ValueError("only organizer-train targets are accepted")
    output_dir = args.output_dir.resolve()
    if args.mode in {"all", "cache"}:
        output_dir.mkdir(parents=True, exist_ok=False)
        cache = _build_cache(
            output_dir=output_dir,
            targets_dir=targets_dir,
            config=config,
            config_path=args.config.resolve(),
        )
        if args.mode == "cache":
            report = {
                "schema": "aiijc-taska-joint-component-pose-cache-report-v1",
                "status": "cache-complete",
                "preregistration_sha256": config_sha256,
                "cache": cache,
                "competition_test_accessed": False,
            }
            _write_json(output_dir / "report.json", report)
            return report
    else:
        cache = {"status": "reused"}
    pilot = _pilot(
        output_dir=output_dir,
        targets_dir=targets_dir,
        config=config,
        config_sha256=config_sha256,
        config_path=args.config.resolve(),
        device=torch.device(args.device),
    )
    report = {
        "schema": "aiijc-taska-joint-component-pose-report-v1",
        "status": "gate-pass-server-scale-authorized"
        if pilot["gate"]["passed"]
        else "bounded-gate-fail-stop",
        "protocol": config,
        "preregistration_sha256": config_sha256,
        "cache": cache,
        **pilot,
        "legality": {
            "organizer_train_targets_only": True,
            "competition_test_accessed": False,
            "local_candidate_frozen_before_local_exact_reconstruction": True,
            "strict_original_upright_576_tile_permutations": True,
            "pixels_modified_or_replaced": False,
            "raw_seam_veto_or_posthoc_objective_guard": False,
            "production_or_submission_modified": False,
        },
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "capacity": report["capacity"],
                "fit_retrieval": report["fit_retrieval"],
                "local32": {
                    key: report["local32"][key]
                    for key in (
                        "raw_retrieval",
                        "learned_retrieval",
                        "arms",
                        "candidate_minus_control",
                        "changed_layout_count",
                    )
                },
                "gate": report["gate"],
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())
