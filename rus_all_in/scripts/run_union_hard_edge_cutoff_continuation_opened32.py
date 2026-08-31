#!/usr/bin/env python3
"""Run one frozen cutoff-aware continuation on the already-opened eval32 panel.

This runner deliberately reuses the parent pilot's target-free fit/eval feature
cache, fit labels, checkpoint, and frozen Union/learned evaluation layouts.  It
does not contain a feature-building path.  The continued checkpoint and all
candidate priorities/layouts are frozen and hash-rostered before clean targets
are opened to recreate evaluation references.

The panel is development evidence only.  A passing result still requires a
separate source-disjoint confirmation before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import LayoutEvaluation, evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.union_hard_edge_cutoff_loss import (
    CUTOFF_EXCHANGE_LOSS_SCHEMA,
    CUTOFF_EXCHANGE_RESIDUAL_WEIGHT,
    union_hard_edge_cutoff_exchange_loss,
)
from aiijc_puzzle.union_hard_edge_priority import (
    FEATURE_NAMES,
    UnionHardEdgePriority,
    union_hard_edge_priority_matrices,
)

try:
    from scripts.run_union_hard_edge_priority_pilot import (
        BOOTSTRAP_SEED,
        DEFAULT_TARGETS,
        PROJECT_ROOT,
        CleanTileCache,
        _cache_paths,
        _cached_board,
        _decode_layout,
        _dirty_sha256,
        _hard_identities,
        _strict_layout,
        prepare_case,
        source_clustered_delta_ci,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_union_hard_edge_priority_pilot import (
        BOOTSTRAP_SEED,
        DEFAULT_TARGETS,
        PROJECT_ROOT,
        CleanTileCache,
        _cache_paths,
        _cached_board,
        _decode_layout,
        _dirty_sha256,
        _hard_identities,
        _strict_layout,
        prepare_case,
        source_clustered_delta_ci,
    )


CONFIG_SCHEMA = "aiijc-union-hard-edge-cutoff-continuation-opened32-v1"
EXPERIMENT = "union-hard-edge-cutoff-continuation-opened32-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/union_hard_edge_cutoff_continuation_opened32_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/union-hard-edge-cutoff-continuation/opened32-v1"

GRID = 24
COUNT = GRID * GRID
HARD_EDGES_PER_AXIS = GRID * (GRID - 1)
HARD_EDGE_COUNT = 2 * HARD_EDGES_PER_AXIS
EDGE_BUDGET_PER_AXIS = 144
FIT_CASE_COUNT = 128
EVAL_CASE_COUNT = 32
EVAL_SOURCE_COUNT = 16
DRAWS = (0, 1)
HIDDEN_DIMENSION = 64
RESIDUAL_LIMIT = 2.0
PARENT_STEPS = 400
ADDITIONAL_STEPS = 200
LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
CASE_ORDER_SEED_OFFSET = 190_001
EVAL_SOURCE_ORDER_DIGEST = "13f7fe84262f9c4d0aee7ce80dfdc1edeec3ce7f1b5082f06ae5c6aceda6fa5f"
EVAL_CASES_DIGEST = "6b872422b32562b845deb514fd478cd21379384cccdcf07a9e4db257e71b6e04"

PARENT_CONFIG_SHA256 = "3cc28b93d88f7e13366740f59a230635a98a528cb11e5e941a0ce3fa9256e7f6"
PARENT_COMMITMENT_SHA256 = "575bb43d850ec3276b61aef616cfa9f2f5fa6f31db35f417d6852b9a38dac540"
PARENT_CHECKPOINT_SHA256 = "472c2770e8960125359c44afdafa6cd31fbb6517d3db33e514b94aa56905efd5"
FIT_LABELS_SHA256 = "c9746421130ac19fab5594ae5a220eb0899b6357d79b0aa46cb7aa41d8b413c9"
FIT_LABELS_METADATA_SHA256 = "9bf766a32baa5c2c65fab2ff4ea8754269251e87c67dd971f5f1ea1b4a626d25"
CACHE_METADATA_SHA256 = "2aa1d4a747cd08dca60a7bd66f5a347df9511c0f29e0f7be3877a2a2fcb0f6b5"
PARENT_FROZEN_EVAL_SHA256 = "86bf9dfa5f0117e3ea35e3c0806f5909a271c176b90cea24c0f1dc7802e11fcc"
PARENT_FROZEN_EVAL_METADATA_SHA256 = (
    "b0e26d3fdf2a05169d6ba18c3ec62561470f205f64c4bf454e8142f8ae39edac"
)
PARENT_REPORT_SHA256 = "c4cf10f37f10a709e5390f2bd05555ecf0304ab958f7ca6ebde713cbb9f17e5e"
PARENT_PILOT_RUNNER_SHA256 = "80704a6017d33f289b22a932aaaa56bd59461a5be91d651e392bcd6f48f8ec86"
UNION_CORE_SHA256 = "71f20876514e9b5cb875b1b5a237d57d4a2f78804273d611ab94bae57248f4ca"

PARENT_KEYS = {
    "config": PARENT_CONFIG_SHA256,
    "selection_commitment": PARENT_COMMITMENT_SHA256,
    "checkpoint": PARENT_CHECKPOINT_SHA256,
    "fit_labels": FIT_LABELS_SHA256,
    "fit_labels_metadata": FIT_LABELS_METADATA_SHA256,
    "target_free_cache_metadata": CACHE_METADATA_SHA256,
    "frozen_target_free_eval": PARENT_FROZEN_EVAL_SHA256,
    "frozen_target_free_eval_metadata": PARENT_FROZEN_EVAL_METADATA_SHA256,
    "report": PARENT_REPORT_SHA256,
}

RUNTIME_SOURCE_PATHS = {
    "continuation_runner": Path(__file__).resolve(),
    "cutoff_loss": PROJECT_ROOT / "src/aiijc_puzzle/union_hard_edge_cutoff_loss.py",
    "union_hard_edge_priority": PROJECT_ROOT / "src/aiijc_puzzle/union_hard_edge_priority.py",
    "parent_pilot_runner": PROJECT_ROOT / "scripts/run_union_hard_edge_priority_pilot.py",
}


@dataclass(frozen=True)
class ParentArtifacts:
    config: Path
    selection_commitment: Path
    checkpoint: Path
    fit_labels: Path
    fit_labels_metadata: Path
    target_free_cache_metadata: Path
    frozen_target_free_eval: Path
    frozen_target_free_eval_metadata: Path
    report: Path

    @property
    def cache_dir(self) -> Path:
        return self.target_free_cache_metadata.parent


@dataclass(frozen=True)
class RunPaths:
    checkpoint: Path
    frozen_eval: Path
    frozen_eval_metadata: Path
    pre_score_freeze: Path
    report: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(value: Any, *, name: str) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is malformed")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _record(path: Path) -> dict[str, str]:
    return {"path": _project_path(path), "sha256": sha256_file(path)}


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _cases_digest(names: Sequence[str], draws: Sequence[int]) -> str:
    values = [f"{name}\0{int(draw)}" for name in names for draw in draws]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _integer_sequence_digest(values: Sequence[int]) -> str:
    return hashlib.sha256("\n".join(str(int(value)) for value in values).encode()).hexdigest()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_torch_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            torch.save(payload, stream)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _require_equal(config: Mapping[str, Any], dotted: str, expected: Any) -> None:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"preregistration field is absent: {dotted}")
        value = value[part]
    if value != expected:
        raise ValueError(f"preregistration field changed: {dotted}")


def _validate_recipe(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate all scientific choices without opening any referenced artifact."""

    fixed = {
        "schema": CONFIG_SCHEMA,
        "experiment": EXPERIMENT,
        "protocol.development_eval_panel_previously_opened": True,
        "protocol.configuration_frozen_before_candidate_training_and_scoring": True,
        "protocol.single_candidate_arm": True,
        "protocol.hyperparameter_or_arm_sweep": False,
        "protocol.eval_predictions_and_layouts_frozen_before_reference_recreation": True,
        "protocol.fresh_source_disjoint_confirmation_required_before_promotion": True,
        "protocol.target_free_feature_cache_reused_without_feature_pass": True,
        "model.architecture": "union-hard-edge-deepsets-bounded-residual-v1",
        "model.feature_dimension": len(FEATURE_NAMES),
        "model.hidden_dimension": HIDDEN_DIMENSION,
        "model.residual_limit": RESIDUAL_LIMIT,
        "model.hard_edge_count": HARD_EDGE_COUNT,
        "model.hard_edges_per_axis": HARD_EDGES_PER_AXIS,
        "model.edge_budget_per_axis": EDGE_BUDGET_PER_AXIS,
        "model.initial_state": "exact parent checkpoint state after 400 updates",
        "continuation.additional_steps": ADDITIONAL_STEPS,
        "continuation.optimizer": "AdamW",
        "continuation.learning_rate": LEARNING_RATE,
        "continuation.weight_decay": WEIGHT_DECAY,
        "continuation.gradient_clip": GRADIENT_CLIP,
        "continuation.cases_per_step": 1,
        "continuation.case_order_seed_offset_from_parent_synthetic_seed": (CASE_ORDER_SEED_OFFSET),
        "continuation.loss.name": "cutoff_exchange",
        "continuation.loss.selection_order": (
            "descending detached learned score, then descending frozen Union base priority, "
            "then ascending source id, then ascending target id"
        ),
        "continuation.loss.all_pairwise_or_bce_mixture": False,
        "continuation.loss.normalised_residual_l2_weight": (CUTOFF_EXCHANGE_RESIDUAL_WEIGHT),
        "continuation.hyperparameter_sweep": False,
        "evaluation.panel": "the already-opened parent eval32 only",
        "evaluation.arms": [
            "union_v2_qap24_cyclic5",
            "parent_learned_priority_qap24_cyclic5",
            "cutoff_continuation_qap24_cyclic5",
        ],
        "evaluation.primary_metric": "satisfied_adjacent_pairs_per_board",
        "evaluation.pair_denominator": 1104,
        "evaluation.secondary_metrics": [
            "adjacency_recall",
            "correct_fixed_top288_hard_edges_per_board",
            "exact_tiles_per_board",
        ],
        "evaluation.development_go_rule.minimum_pair_gain_over_parent_per_board": 0.25,
        "evaluation.development_go_rule.minimum_top288_gain_over_parent_per_board": 0.0,
        "evaluation.development_go_rule.minimum_exact_delta_over_parent_per_board": -0.25,
        "evaluation.development_go_rule.all_layouts_must_be_strict_original_tile_permutations": (
            True
        ),
        "evaluation.bootstrap_intervals_are_reported_but_not_used_for_tuning": True,
        "legality.organizer_train_only": True,
        "legality.restored_or_denoised_pixels_are_matcher_only": True,
        "legality.candidate_cannot_add_or_replace_hard_edge_identities": True,
        "legality.decoder_and_cyclic_translation_are_unchanged": True,
        "legality.output_uses_each_original_upright_20x20_tile_exactly_once": True,
        "legality.competition_test_forbidden": True,
    }
    for name, expected in fixed.items():
        _require_equal(config, name, expected)

    panel = config.get("panel")
    if not isinstance(panel, Mapping):
        raise ValueError("preregistration needs an explicit panel mapping")
    names = panel.get("source_filenames")
    if not isinstance(names, list) or not all(
        isinstance(name, str) and Path(name).name == name and name.endswith(".png")
        for name in names
    ):
        raise ValueError("panel source_filenames are malformed")
    source_names = tuple(names)
    if len(source_names) != EVAL_SOURCE_COUNT or len(set(source_names)) != len(source_names):
        raise ValueError("panel must contain exactly 16 unique sources")
    _require_equal(config, "panel.previously_opened", True)
    _require_equal(config, "panel.source_order_digest", EVAL_SOURCE_ORDER_DIGEST)
    _require_equal(config, "panel.draws", list(DRAWS))
    _require_equal(config, "panel.case_count", EVAL_CASE_COUNT)
    _require_equal(config, "panel.cases_digest", EVAL_CASES_DIGEST)
    if _names_digest(source_names) != EVAL_SOURCE_ORDER_DIGEST:
        raise ValueError("panel source order digest mismatch")
    if _cases_digest(source_names, DRAWS) != EVAL_CASES_DIGEST:
        raise ValueError("panel source/draw digest mismatch")
    return source_names


def _load_preregistration(path: Path) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("preregistration and its .sha256 sidecar must both exist")
    config_sha = sha256_file(path)
    tokens = sidecar.read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != config_sha:
        raise ValueError("preregistration sidecar does not match config bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preregistration root must be a mapping")
    source_names = _validate_recipe(payload)
    return payload, config_sha, source_names


def _validate_record(
    records: Mapping[str, Any],
    name: str,
    *,
    expected_sha256: str | None = None,
    expected_path: Path | None = None,
) -> Path:
    value = records.get(name)
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} needs explicit path and sha256")
    path = _resolve_path(value, name=name)
    observed = sha256_file(path)
    if value.get("sha256") != observed:
        raise ValueError(f"{name} SHA-256 mismatch")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"{name} differs from the frozen parent artifact")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{name} path differs from the frozen runtime source")
    return path


def _validate_parent_artifacts(config: Mapping[str, Any]) -> ParentArtifacts:
    parent = config.get("parent")
    if not isinstance(parent, Mapping) or set(parent) != set(PARENT_KEYS):
        raise ValueError("parent artifact roster changed")
    paths = {
        name: _validate_record(parent, name, expected_sha256=expected)
        for name, expected in PARENT_KEYS.items()
    }
    runtime = config.get("runtime_sources")
    if not isinstance(runtime, Mapping) or set(runtime) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("runtime source roster changed")
    for name, expected_path in RUNTIME_SOURCE_PATHS.items():
        expected_sha = None
        if name == "parent_pilot_runner":
            expected_sha = PARENT_PILOT_RUNNER_SHA256
        elif name == "union_hard_edge_priority":
            expected_sha = UNION_CORE_SHA256
        _validate_record(
            runtime,
            name,
            expected_sha256=expected_sha,
            expected_path=expected_path,
        )
    return ParentArtifacts(**paths)


def _validate_parent_commitment(
    artifacts: ParentArtifacts,
    source_names: Sequence[str],
) -> dict[str, Any]:
    commitment = json.loads(artifacts.selection_commitment.read_text(encoding="utf-8"))
    if commitment.get("schema") != "aiijc-union-hard-priority-pilot-v1":
        raise ValueError("parent selection commitment schema changed")
    if commitment.get("created_before_target_access") is not True:
        raise ValueError("parent selection commitment timing claim changed")
    if commitment.get("config", {}).get("sha256") != PARENT_CONFIG_SHA256:
        raise ValueError("parent commitment points to a different config")
    if commitment.get("synthetic_seed") != 1_267_233_517:
        raise ValueError("parent synthetic seed changed")
    evaluation = commitment.get("eval")
    if not isinstance(evaluation, Mapping):
        raise ValueError("parent commitment eval roster is absent")
    if tuple(evaluation.get("source_filenames", ())) != tuple(source_names):
        raise ValueError("continuation panel differs from parent opened eval roster")
    if evaluation.get("source_order_digest") != EVAL_SOURCE_ORDER_DIGEST:
        raise ValueError("parent eval source order digest changed")
    if evaluation.get("draws") != list(DRAWS) or evaluation.get("case_count") != EVAL_CASE_COUNT:
        raise ValueError("parent eval case expansion changed")
    if evaluation.get("cases_digest") != EVAL_CASES_DIGEST:
        raise ValueError("parent eval case digest changed")
    fit = commitment.get("fit")
    if not isinstance(fit, Mapping) or fit.get("case_count") != FIT_CASE_COUNT:
        raise ValueError("parent fit case roster changed")
    if fit.get("draws") != list(DRAWS):
        raise ValueError("parent fit draws changed")
    return commitment


def _validate_cache(
    artifacts: ParentArtifacts,
    commitment: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = json.loads(artifacts.target_free_cache_metadata.read_text(encoding="utf-8"))
    fixed = {
        "schema": "aiijc-union-hard-edge-target-free-cache-v1",
        "contains_exact_references_or_labels": False,
        "contains_pixels": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_dimension": len(FEATURE_NAMES),
        "hard_edge_count": HARD_EDGE_COUNT,
        "values_storage_dtype": "float32",
    }
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError("target-free cache metadata contract changed")
    files = metadata.get("files")
    if not isinstance(files, Mapping) or set(files) != {"fit", "eval"}:
        raise ValueError("target-free cache file roles changed")
    expected_file_names = set(_cache_paths(artifacts.cache_dir, "fit"))
    for role, case_count in (("fit", FIT_CASE_COUNT), ("eval", EVAL_CASE_COUNT)):
        records = files.get(role)
        if not isinstance(records, Mapping) or set(records) != expected_file_names:
            raise ValueError(f"target-free {role} cache file roster changed")
        expected_paths = _cache_paths(artifacts.cache_dir, role)
        for name, expected_path in expected_paths.items():
            _validate_record(records, name, expected_path=expected_path)
        arrays = {name: np.load(path, mmap_mode="r") for name, path in expected_paths.items()}
        shapes = {
            "values": (case_count, HARD_EDGE_COUNT, len(FEATURE_NAMES)),
            "base": (case_count, HARD_EDGE_COUNT),
            "scale": (case_count, HARD_EDGE_COUNT),
            "axis": (case_count, HARD_EDGE_COUNT),
            "source": (case_count, HARD_EDGE_COUNT),
            "target": (case_count, HARD_EDGE_COUNT),
            "right-assignment": (case_count, COUNT + 1, COUNT + 1),
            "down-assignment": (case_count, COUNT + 1, COUNT + 1),
            "direct-matches": (case_count, 2),
            "fullres-supported": (case_count, 2),
        }
        if any(arrays[name].shape != shape for name, shape in shapes.items()):
            raise ValueError(f"target-free {role} cache array shape changed")

    cases = metadata.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != {"fit", "eval"}:
        raise ValueError("target-free cache case roster changed")
    for role, expected_count in (("fit", FIT_CASE_COUNT), ("eval", EVAL_CASE_COUNT)):
        rows = cases.get(role)
        registered = commitment[role]
        expected = [
            (name, int(draw))
            for name in registered["source_filenames"]
            for draw in registered["draws"]
        ]
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(f"target-free {role} cache rows changed")
        observed = [(row.get("source_filename"), row.get("draw_index")) for row in rows]
        if observed != expected or [row.get("index") for row in rows] != list(
            range(expected_count)
        ):
            raise ValueError(f"target-free {role} cache row order changed")
    return metadata


def _validate_fit_labels(artifacts: ParentArtifacts) -> np.ndarray:
    metadata = json.loads(artifacts.fit_labels_metadata.read_text(encoding="utf-8"))
    fixed = {
        "schema": "aiijc-union-hard-edge-fit-labels-v1",
        "created_after_complete_target_free_fit_and_eval_cache": True,
        "target_free_cache_metadata_sha256": CACHE_METADATA_SHA256,
        "labels_sha256": FIT_LABELS_SHA256,
        "case_count": FIT_CASE_COUNT,
        "exact_references_persisted": False,
    }
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError("fit label metadata contract changed")
    labels = np.load(artifacts.fit_labels, mmap_mode="r")
    if labels.shape != (FIT_CASE_COUNT, HARD_EDGE_COUNT) or labels.dtype != np.bool_:
        raise ValueError("fit labels violate the frozen shape/dtype contract")
    return labels


def _validate_parent_frozen_eval(
    artifacts: ParentArtifacts,
    cache_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = json.loads(artifacts.frozen_target_free_eval_metadata.read_text(encoding="utf-8"))
    fixed = {
        "schema": "aiijc-union-hard-edge-frozen-target-free-eval-v1",
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "target_free_cache_metadata_sha256": CACHE_METADATA_SHA256,
        "contains_exact_references_or_labels": False,
        "contains_strict_original_tile_layouts": True,
        "contains_frozen_hard_priorities": True,
        "strict_layout_count_per_arm": EVAL_CASE_COUNT,
    }
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError("parent frozen eval metadata contract changed")
    rows = metadata.get("rows")
    cache_rows = cache_metadata["cases"]["eval"]
    if not isinstance(rows, list) or len(rows) != EVAL_CASE_COUNT:
        raise ValueError("parent frozen eval row count changed")
    for index, (row, cached) in enumerate(zip(rows, cache_rows, strict=True)):
        expected = {
            "prefix": f"case_{index:04d}",
            "case_id": cached["case_id"],
            "source_filename": cached["source_filename"],
            "draw_index": cached["draw_index"],
            "dirty_sha256": cached["dirty_sha256"],
        }
        if row != expected:
            raise ValueError("parent frozen eval rows differ from target-free cache")
    return metadata


def _validate_parent_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "aiijc-union-hard-edge-priority-checkpoint-v1"
    ):
        raise ValueError("parent checkpoint schema changed")
    contract = payload.get("contract")
    fixed = {
        "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_dimension": len(FEATURE_NAMES),
        "hidden_dimension": HIDDEN_DIMENSION,
        "residual_limit": RESIDUAL_LIMIT,
        "hard_edge_count": HARD_EDGE_COUNT,
        "edge_budget_per_axis": EDGE_BUDGET_PER_AXIS,
        "training_steps": PARENT_STEPS,
        "pixel_prediction": False,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(name) != value for name, value in fixed.items()
    ):
        raise ValueError("parent checkpoint contract changed")
    if payload.get("selection_commitment_sha256") != PARENT_COMMITMENT_SHA256:
        raise ValueError("parent checkpoint commitment lineage changed")
    if payload.get("fit_labels_sha256") != FIT_LABELS_SHA256:
        raise ValueError("parent checkpoint fit-label lineage changed")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("parent checkpoint state_dict is absent")
    return dict(payload)


def _select_device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name == "mps":
        if not allow_nondeterministic_mps:
            raise ValueError("MPS requires --allow-nondeterministic-mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
        return torch.device("mps")
    if name != "cpu":
        raise ValueError("device must be cpu or mps")
    if allow_nondeterministic_mps:
        raise ValueError("--allow-nondeterministic-mps requires MPS")
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def _prepare_run_paths(output_dir: Path) -> RunPaths:
    root = output_dir.resolve()
    paths = RunPaths(
        checkpoint=root / "union-hard-edge-cutoff-continuation.pt",
        frozen_eval=root / "frozen-target-free-eval.npz",
        frozen_eval_metadata=root / "frozen-target-free-eval.json",
        pre_score_freeze=root / "pre-score-freeze.json",
        report=root / "report.json",
    )
    if any(path.exists() for path in paths.__dict__.values()):
        raise FileExistsError("refusing to overwrite a cutoff continuation run")
    root.mkdir(parents=True, exist_ok=True)
    return paths


def _case_order(seed: int) -> list[int]:
    generator = np.random.default_rng(seed)
    order: list[int] = []
    while len(order) < ADDITIONAL_STEPS:
        order.extend(int(value) for value in generator.permutation(FIT_CASE_COUNT))
    return order[:ADDITIONAL_STEPS]


def _train_continuation(
    artifacts: ParentArtifacts,
    paths: RunPaths,
    parent_checkpoint: Mapping[str, Any],
    labels: np.ndarray,
    *,
    config_sha256: str,
    synthetic_seed: int,
    device: torch.device,
    log_every: int,
) -> tuple[UnionHardEdgePriority, list[dict[str, Any]], float, list[int]]:
    continuation_seed = synthetic_seed + CASE_ORDER_SEED_OFFSET
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    torch.manual_seed(continuation_seed)
    model = UnionHardEdgePriority(
        hidden_dimension=HIDDEN_DIMENSION,
        residual_limit=RESIDUAL_LIMIT,
    ).to(device)
    model.load_state_dict(parent_checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    case_order = _case_order(continuation_seed)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    model.train()
    for step, case_index in enumerate(case_order, start=1):
        board = _cached_board(artifacts.cache_dir, "fit", case_index, device=device)
        truth = torch.from_numpy(np.asarray(labels[case_index], dtype=bool).copy()).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(board)
        loss, diagnostics = union_hard_edge_cutoff_exchange_loss(output, board, truth)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite cutoff continuation loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(gradient).item()):
            raise RuntimeError("non-finite cutoff continuation gradient")
        optimizer.step()
        parameters_are_finite = all(
            bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()
        )
        if not parameters_are_finite:
            raise RuntimeError("non-finite cutoff continuation parameter")
        row = {
            "step": step,
            "case_index": case_index,
            **diagnostics,
            "gradient_norm_before_clip": float(gradient.detach().cpu()),
        }
        history.append(row)
        if step == 1 or step % log_every == 0:
            print(json.dumps({"event": "cutoff_continuation_train", **row}), flush=True)
    training_seconds = perf_counter() - started
    model.eval()
    _write_torch_exclusive(
        paths.checkpoint,
        {
            "schema": "aiijc-union-hard-edge-cutoff-continuation-checkpoint-v1",
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "contract": {
                "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
                "feature_names": list(FEATURE_NAMES),
                "feature_dimension": len(FEATURE_NAMES),
                "hidden_dimension": HIDDEN_DIMENSION,
                "residual_limit": RESIDUAL_LIMIT,
                "hard_edge_count": HARD_EDGE_COUNT,
                "edge_budget_per_axis": EDGE_BUDGET_PER_AXIS,
                "parent_training_steps": PARENT_STEPS,
                "additional_steps": ADDITIONAL_STEPS,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "cases_per_step": 1,
                "case_order_seed": continuation_seed,
                "case_order_seed_offset": CASE_ORDER_SEED_OFFSET,
                "case_order_digest": _integer_sequence_digest(case_order),
                "loss_schema": CUTOFF_EXCHANGE_LOSS_SCHEMA,
                "normalised_residual_l2_weight": CUTOFF_EXCHANGE_RESIDUAL_WEIGHT,
                "all_pairwise_or_bce_mixture": False,
                "pixel_prediction": False,
                "feature_pass_performed": False,
            },
            "preregistration_sha256": config_sha256,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "fit_labels_sha256": FIT_LABELS_SHA256,
            "target_free_cache_metadata_sha256": CACHE_METADATA_SHA256,
            "history": history,
        },
    )
    return model, history, training_seconds, case_order


def _identity_arrays(
    archive: Mapping[str, Any],
    prefix: str,
    axis_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(archive[f"{prefix}__axis_{axis_index}_source"], dtype=np.int32)
    target = np.asarray(archive[f"{prefix}__axis_{axis_index}_target"], dtype=np.int32)
    base = np.asarray(archive[f"{prefix}__axis_{axis_index}_baseline_priority"], dtype=np.float64)
    learned = np.asarray(archive[f"{prefix}__axis_{axis_index}_learned_priority"], dtype=np.float64)
    if any(value.shape != (HARD_EDGES_PER_AXIS,) for value in (source, target, base, learned)):
        raise ValueError("parent frozen hard-edge arrays have changed shape")
    if not np.isfinite(base).all() or not np.isfinite(learned).all():
        raise ValueError("parent frozen hard-edge priorities are non-finite")
    identities = list(zip(source.tolist(), target.tolist(), strict=True))
    if len(set(identities)) != HARD_EDGES_PER_AXIS or np.any(source == target):
        raise ValueError("parent frozen hard-edge identities are malformed")
    return source, target, base, learned


def _freeze_eval(
    artifacts: ParentArtifacts,
    paths: RunPaths,
    model: UnionHardEdgePriority,
    cache_metadata: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[str, str]:
    cache_rows = cache_metadata["cases"]["eval"]
    cache_paths = _cache_paths(artifacts.cache_dir, "eval")
    right_cache = np.load(cache_paths["right-assignment"], mmap_mode="r")
    down_cache = np.load(cache_paths["down-assignment"], mmap_mode="r")
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    strict_count = 0
    with np.load(artifacts.frozen_target_free_eval) as parent_archive, torch.inference_mode():
        for index, cached in enumerate(cache_rows):
            prefix = f"case_{index:04d}"
            board = _cached_board(artifacts.cache_dir, "eval", index, device=device)
            output = model(board)
            if not bool(torch.isfinite(output.scores).all().item()):
                raise RuntimeError("candidate produced non-finite eval priorities")
            priorities = union_hard_edge_priority_matrices(board, output.scores)
            right = np.asarray(right_cache[index], dtype=np.float32)
            down = np.asarray(down_cache[index], dtype=np.float32)
            axes = board.axis.detach().cpu().numpy()
            scores = output.scores.detach().float().cpu().numpy()
            for axis_index, axis_name, assignment in (
                (0, "right", right),
                (1, "down", down),
            ):
                selected = axes == axis_index
                source, target, base, _ = _identity_arrays(parent_archive, prefix, axis_index)
                if not np.array_equal(source, board.source[selected]) or not np.array_equal(
                    target, board.target[selected]
                ):
                    raise RuntimeError("candidate hard-edge identities differ from parent")
                if not np.array_equal(
                    base.astype(np.float32),
                    board.base_priority[selected].detach().float().cpu().numpy(),
                ):
                    raise RuntimeError("candidate frozen Union priorities differ from parent")
                board_identities = tuple(sorted(zip(source.tolist(), target.tolist(), strict=True)))
                if board_identities != _hard_identities(assignment, axis=axis_name):
                    raise RuntimeError(f"cached {axis_name} hard identities drifted")
                arrays[f"{prefix}__axis_{axis_index}_source"] = source
                arrays[f"{prefix}__axis_{axis_index}_target"] = target
                arrays[f"{prefix}__axis_{axis_index}_baseline_priority"] = base.astype(np.float32)
                arrays[f"{prefix}__axis_{axis_index}_cutoff_priority"] = scores[selected]

            candidate_layout = _decode_layout(
                right,
                down,
                component_edge_priority=priorities,
            )
            arrays[f"{prefix}__cutoff_continuation_layout"] = candidate_layout
            strict_count += 1
            rows.append(
                {
                    "prefix": prefix,
                    "case_id": cached["case_id"],
                    "source_filename": cached["source_filename"],
                    "draw_index": cached["draw_index"],
                    "dirty_sha256": cached["dirty_sha256"],
                }
            )
    if strict_count != EVAL_CASE_COUNT:
        raise RuntimeError("candidate eval freeze did not produce 32 strict layouts")
    _write_npz_exclusive(paths.frozen_eval, arrays)
    archive_sha = sha256_file(paths.frozen_eval)
    _write_json_exclusive(
        paths.frozen_eval_metadata,
        {
            "schema": "aiijc-union-hard-edge-cutoff-frozen-target-free-eval-v1",
            "checkpoint_sha256": sha256_file(paths.checkpoint),
            "parent_frozen_eval_sha256": PARENT_FROZEN_EVAL_SHA256,
            "target_free_cache_metadata_sha256": CACHE_METADATA_SHA256,
            "contains_exact_references_or_labels": False,
            "contains_pixels": False,
            "contains_strict_original_tile_layouts": True,
            "contains_frozen_hard_priorities": True,
            "hard_edge_identities_unchanged_from_parent": True,
            "feature_pass_performed": False,
            "strict_layout_count": strict_count,
            "archive_sha256": archive_sha,
            "rows": rows,
        },
    )
    return archive_sha, sha256_file(paths.frozen_eval_metadata)


def _freeze_pre_score_roster(
    paths: RunPaths,
    artifacts: ParentArtifacts,
    *,
    config_path: Path,
    config_sha256: str,
) -> dict[str, dict[str, str]]:
    frozen = {
        "preregistration": {"path": _project_path(config_path), "sha256": config_sha256},
        "parent_checkpoint": _record(artifacts.checkpoint),
        "parent_frozen_eval": _record(artifacts.frozen_target_free_eval),
        "parent_frozen_eval_metadata": _record(artifacts.frozen_target_free_eval_metadata),
        "target_free_cache_metadata": _record(artifacts.target_free_cache_metadata),
        "fit_labels": _record(artifacts.fit_labels),
        "continuation_checkpoint": _record(paths.checkpoint),
        "candidate_frozen_eval": _record(paths.frozen_eval),
        "candidate_frozen_eval_metadata": _record(paths.frozen_eval_metadata),
    }
    _write_json_exclusive(
        paths.pre_score_freeze,
        {
            "schema": "aiijc-union-hard-edge-cutoff-pre-score-freeze-v1",
            "experiment": EXPERIMENT,
            "created_before_eval_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "already_opened_development_panel": True,
            "freshness_claimed": False,
            "artifacts": frozen,
        },
    )
    return frozen


def _validate_pre_score_freeze(
    paths: RunPaths,
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    payload = json.loads(paths.pre_score_freeze.read_text(encoding="utf-8"))
    if payload.get("created_before_eval_reference_recreation") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze unexpectedly contains evaluation labels")
    if payload.get("artifacts") != expected:
        raise RuntimeError("pre-score frozen artifact roster changed")
    for name, record in expected.items():
        path = _resolve_path(record, name=f"pre_score_{name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {name}")


def _manifest_lookup(commitment: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    record = commitment.get("manifest")
    if not isinstance(record, Mapping) or set(record) < {"path", "sha256", "split"}:
        raise ValueError("parent commitment manifest record is malformed")
    path = _resolve_path(record, name="parent_manifest")
    if sha256_file(path) != record["sha256"] or record["split"] != "train":
        raise ValueError("parent organizer-train manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("splits", {}).get("train")
    if not isinstance(rows, list) or len(rows) != 5_600:
        raise ValueError("parent organizer-train manifest rows changed")
    lookup = {str(row["filename"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("parent organizer-train manifest has duplicate filenames")
    return lookup


def _fixed_top288_correct(
    source: np.ndarray,
    target: np.ndarray,
    priority: np.ndarray,
    base_priority: np.ndarray,
    *,
    axis_index: int,
    reference: np.ndarray,
) -> int:
    vectors = tuple(np.asarray(value) for value in (source, target, priority, base_priority))
    if any(value.shape != (HARD_EDGES_PER_AXIS,) for value in vectors):
        raise ValueError("fixed-top288 vectors violate axis cardinality")
    source_np = vectors[0].astype(np.int32, copy=False)
    target_np = vectors[1].astype(np.int32, copy=False)
    priority_np = vectors[2].astype(np.float64, copy=False)
    base_np = vectors[3].astype(np.float64, copy=False)
    if not np.isfinite(priority_np).all() or not np.isfinite(base_np).all():
        raise ValueError("fixed-top288 priorities must be finite")
    order = np.lexsort((target_np, source_np, -base_np, -priority_np))
    position = np.empty(COUNT, dtype=np.int32)
    position[reference] = np.arange(COUNT, dtype=np.int32)
    source_position = position[source_np]
    target_position = position[target_np]
    if axis_index == 0:
        truth = (target_position == source_position + 1) & (source_position % GRID != GRID - 1)
    elif axis_index == 1:
        truth = target_position == source_position + GRID
    else:
        raise ValueError("axis_index must be zero or one")
    return int(np.count_nonzero(truth[order[:EDGE_BUDGET_PER_AXIS]]))


def _layout_metrics(evaluation: LayoutEvaluation, fixed_top288_correct: int) -> dict[str, Any]:
    if evaluation.adjacency_total != 1104:
        raise RuntimeError("adjacency denominator changed")
    pair_count = int(evaluation.adjacency_correct)
    recall = float(evaluation.adjacency)
    if pair_count != round(recall * evaluation.adjacency_total):
        raise RuntimeError("integer adjacency count and recall disagree")
    return {
        "satisfied_adjacent_pairs": pair_count,
        "adjacency_recall": recall,
        "fixed_top288_correct": int(fixed_top288_correct),
        "exact_tiles": int(evaluation.correct_tile_count),
    }


def _score_frozen_eval(
    artifacts: ParentArtifacts,
    paths: RunPaths,
    commitment: Mapping[str, Any],
    expected_hashes: Mapping[str, Mapping[str, str]],
    *,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    # This must remain the first operation: no CleanTileCache and no target path
    # inspection can occur until every target-free prediction artifact is frozen.
    _validate_pre_score_freeze(paths, expected_hashes)
    lookup = _manifest_lookup(commitment)
    target_cache = CleanTileCache(targets)
    parent_metadata = json.loads(
        artifacts.frozen_target_free_eval_metadata.read_text(encoding="utf-8")
    )
    candidate_metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with (
        np.load(artifacts.frozen_target_free_eval) as parent_archive,
        np.load(paths.frozen_eval) as candidate_archive,
    ):
        for parent_row, candidate_row in zip(
            parent_metadata["rows"], candidate_metadata["rows"], strict=True
        ):
            if parent_row != candidate_row:
                raise RuntimeError("candidate and parent frozen eval row rosters differ")
            source_name = str(parent_row["source_filename"])
            draw = int(parent_row["draw_index"])
            case = prepare_case(
                target_cache,
                lookup[source_name],
                draw_index=draw,
                seed=int(commitment["synthetic_seed"]),
            )
            if (
                case.case_id != parent_row["case_id"]
                or _dirty_sha256(case.dirty_tiles) != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("eval scoring recreated a different synthetic case")
            reference = _strict_layout(np.argsort(case.input_tile_to_position))
            prefix = str(parent_row["prefix"])
            layouts = {
                "union_v2": _strict_layout(parent_archive[f"{prefix}__union_v2_layout"]),
                "parent_learned_priority": _strict_layout(
                    parent_archive[f"{prefix}__learned_priority_layout"]
                ),
                "cutoff_continuation": _strict_layout(
                    candidate_archive[f"{prefix}__cutoff_continuation_layout"]
                ),
            }
            top_correct = {name: 0 for name in layouts}
            for axis_index in (0, 1):
                source, target, base, parent_learned = _identity_arrays(
                    parent_archive, prefix, axis_index
                )
                if not np.array_equal(
                    source,
                    candidate_archive[f"{prefix}__axis_{axis_index}_source"],
                ) or not np.array_equal(
                    target,
                    candidate_archive[f"{prefix}__axis_{axis_index}_target"],
                ):
                    raise RuntimeError("candidate scoring identities differ from parent")
                cutoff = np.asarray(
                    candidate_archive[f"{prefix}__axis_{axis_index}_cutoff_priority"],
                    dtype=np.float64,
                )
                arm_priorities = {
                    "union_v2": base,
                    "parent_learned_priority": parent_learned,
                    "cutoff_continuation": cutoff,
                }
                for arm, priority in arm_priorities.items():
                    top_correct[arm] += _fixed_top288_correct(
                        source,
                        target,
                        priority,
                        base,
                        axis_index=axis_index,
                        reference=reference,
                    )
            row: dict[str, Any] = {
                "source_filename": source_name,
                "draw_index": draw,
                "case_id": case.case_id,
            }
            for arm, layout in layouts.items():
                evaluation = evaluate_layout(layout, reference, reference_is_exact=True)
                row[arm] = _layout_metrics(evaluation, top_correct[arm])
            rows.append(row)

    metrics = (
        "satisfied_adjacent_pairs",
        "adjacency_recall",
        "fixed_top288_correct",
        "exact_tiles",
    )
    arms = ("union_v2", "parent_learned_priority", "cutoff_continuation")
    sources = [str(row["source_filename"]) for row in rows]

    def delta(metric: str, baseline: str) -> list[float]:
        return [
            float(row["cutoff_continuation"][metric]) - float(row[baseline][metric]) for row in rows
        ]

    summary = {
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in rows]))
                for metric in metrics
            }
            for arm in arms
        },
        "candidate_deltas": {
            baseline: {
                metric: source_clustered_delta_ci(
                    delta(metric, baseline),
                    sources,
                    seed=BOOTSTRAP_SEED + baseline_index * 10 + metric_index,
                )
                for metric_index, metric in enumerate(metrics)
            }
            for baseline_index, baseline in enumerate(("parent_learned_priority", "union_v2"))
        },
    }
    parent_deltas = summary["candidate_deltas"]["parent_learned_priority"]
    gate = {
        "pair_gain_over_parent_at_least_quarter_pair": (
            float(parent_deltas["satisfied_adjacent_pairs"]["mean"]) >= 0.25
        ),
        "fixed_top288_gain_over_parent_nonnegative": (
            float(parent_deltas["fixed_top288_correct"]["mean"]) >= 0.0
        ),
        "exact_delta_over_parent_at_least_minus_quarter_tile": (
            float(parent_deltas["exact_tiles"]["mean"]) >= -0.25
        ),
        "all_layouts_strict": len(rows) == EVAL_CASE_COUNT,
    }
    gate["passed"] = all(gate.values())
    return rows, summary, gate


def run(args: argparse.Namespace) -> None:
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    config, config_sha, source_names = _load_preregistration(args.config)
    artifacts = _validate_parent_artifacts(config)
    commitment = _validate_parent_commitment(artifacts, source_names)
    cache_metadata = _validate_cache(artifacts, commitment)
    labels = _validate_fit_labels(artifacts)
    _validate_parent_frozen_eval(artifacts, cache_metadata)
    parent_checkpoint = _validate_parent_checkpoint(artifacts.checkpoint)
    paths = _prepare_run_paths(args.output_dir)
    device = _select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    started = perf_counter()
    model, history, training_seconds, case_order = _train_continuation(
        artifacts,
        paths,
        parent_checkpoint,
        labels,
        config_sha256=config_sha,
        synthetic_seed=int(commitment["synthetic_seed"]),
        device=device,
        log_every=args.log_every,
    )
    frozen_eval_sha, frozen_metadata_sha = _freeze_eval(
        artifacts,
        paths,
        model,
        cache_metadata,
        device=device,
    )
    frozen = _freeze_pre_score_roster(
        paths,
        artifacts,
        config_path=args.config,
        config_sha256=config_sha,
    )
    print(
        json.dumps(
            {
                "event": "candidate_checkpoint_priorities_and_layouts_frozen_before_scoring",
                "checkpoint_sha256": sha256_file(paths.checkpoint),
                "frozen_eval_sha256": frozen_eval_sha,
                "frozen_eval_metadata_sha256": frozen_metadata_sha,
                "pre_score_freeze_sha256": sha256_file(paths.pre_score_freeze),
                "target_accessed": False,
            }
        ),
        flush=True,
    )
    rows, metrics, gate = _score_frozen_eval(
        artifacts,
        paths,
        commitment,
        frozen,
        targets=args.targets,
    )
    _write_json_exclusive(
        paths.report,
        {
            "schema": "aiijc-union-hard-edge-cutoff-continuation-opened32-report-v1",
            "status": "development-gate-pass" if gate["passed"] else "development-gate-fail",
            "experiment": EXPERIMENT,
            "panel": {
                "previously_opened": True,
                "freshness_claimed": False,
                "source_disjoint_confirmation_required_before_promotion": True,
                "source_count": EVAL_SOURCE_COUNT,
                "case_count": EVAL_CASE_COUNT,
                "source_order_digest": EVAL_SOURCE_ORDER_DIGEST,
                "cases_digest": EVAL_CASES_DIGEST,
            },
            "preregistration": {"path": _project_path(args.config), "sha256": config_sha},
            "device": {
                "value": str(device),
                "nondeterministic_mps_explicitly_allowed": bool(args.allow_nondeterministic_mps),
                "determinism_claimed": device.type != "mps",
            },
            "continuation": {
                "parent_checkpoint": _record(artifacts.checkpoint),
                "checkpoint": _record(paths.checkpoint),
                "additional_steps": ADDITIONAL_STEPS,
                "case_order_seed": int(commitment["synthetic_seed"]) + CASE_ORDER_SEED_OFFSET,
                "case_order_digest": _integer_sequence_digest(case_order),
                "training_seconds": training_seconds,
                "final_20_loss": float(np.mean([row["loss"] for row in history[-20:]])),
                "single_candidate_arm": True,
                "hyperparameter_or_arm_sweep": False,
            },
            "frozen_eval": {
                "archive": _record(paths.frozen_eval),
                "metadata": _record(paths.frozen_eval_metadata),
                "pre_score_freeze": _record(paths.pre_score_freeze),
                "checkpoint_priorities_and_layouts_frozen_before_references": True,
                "contains_exact_references_or_labels": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": rows,
            "runtime_seconds": perf_counter() - started,
            "legality": {
                "organizer_train_only": True,
                "target_free_cache_reused_without_feature_pass": True,
                "new_hard_edges_introduced": False,
                "decoder_or_cyclic_translation_changed": False,
                "restored_pixels_emitted": False,
                "original_upright_tile_permutations_only": True,
                "competition_test_accessed": False,
            },
        },
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
